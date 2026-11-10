# SPDX-License-Identifier: Apache-2.0
"""Is the SketchSSM flush path dispatch-bound on its padded batch?

sketch_flush() launches _sketch_flush_kernel with grid (cdiv(batch, rows_per_program), heads)
over the *whole padded batch*, while only the listed flush rows do work. This
measures the flush path alone for a fixed number of flush rows at different
padded batch sizes.
"""
import json, sys, torch
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_build as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_decode as skd

W, N, G, SLOTS, H, P = 16, 128, 8, 512, 96, 80

def timed(fn, iters=50):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters

def main():
    dev = torch.device("cuda"); torch.set_default_device(dev); torch.manual_seed(0)
    g = torch.Generator(device=dev).manual_seed(1)
    def rnd(*s, dtype=torch.float32, scale=1.0):
        return (torch.randn(s, generator=g, dtype=torch.float32) * scale).to(dtype)
    state_view = rnd(SLOTS, H, P, N, scale=0.1)
    x = rnd(SLOTS, H, P, dtype=torch.bfloat16, scale=0.3)
    dt = rnd(SLOTS, H, dtype=torch.bfloat16, scale=0.5)[..., None].expand(-1, -1, P)
    A = rnd(H, 1, 1, scale=0.1).expand(H, P, N)
    B = rnd(SLOTS, G, N, dtype=torch.bfloat16, scale=0.3)
    C = rnd(SLOTS, G, N, dtype=torch.bfloat16, scale=0.3)
    D = rnd(H, 1, scale=0.1).expand(H, P); dt_bias = rnd(H, 1, scale=0.1).expand(H, P)
    x_cache = rnd(SLOTS, H, W, P, scale=0.3); dt_cache = rnd(SLOTS, H, W, scale=0.5)
    B_cache = rnd(SLOTS, G, W, N, dtype=torch.bfloat16, scale=0.3)
    out = torch.empty(SLOTS, H, P, dtype=torch.bfloat16)
    query = torch.empty(SLOTS, G, N, dtype=torch.float32)
    slots = torch.arange(SLOTS, dtype=torch.int64); idx = torch.arange(SLOTS, dtype=torch.int32)
    frames = torch.linalg.qr(torch.randn(G, N, N, generator=g))[0].float()
    ranks = torch.tensor([0, 1, 2, 3, 4, 5, 9, 20, 38, 64] * H, dtype=torch.int32)[:H]
    ranks = (ranks.float() * (8.0 / ranks.float().clamp(min=1).mean())).round().to(torch.int32)
    tables = sk.SketchTables(ranks, N)
    bufs = [torch.zeros(SLOTS, *s, dtype=d) for s, d in
            zip(sk.sketch_shapes(ranks, P, N), sk.SKETCH_DTYPES)]
    sketch = sk.SketchArgs(bufs[0], bufs[1], bufs[2], tables, frames.transpose(-1, -2).contiguous())
    sk.sketch_build(state_view, idx, idx, torch.ones(SLOTS, dtype=torch.int8), sketch)
    wp = torch.full((SLOTS,), W - 1, dtype=torch.int32)

    rows = []
    print(f"{'n_flush':>8} {'pad_batch':>10} {'flush path ms':>14}")
    for batch in (64, 128, 256, 512):
        for n_flush in (1, 2, 8, 32):
            if n_flush > batch: continue
            r = torch.tensor(list(range(n_flush)) + [-1] * (batch - n_flush), dtype=torch.int32)
            sh = dict(state=state_view, x=x[:batch], dt=dt[:batch], A=A, B=B[:batch],
                      query=query[:batch], D=D, dt_bias=dt_bias, x_cache=x_cache,
                      dt_cache=dt_cache, B_cache=B_cache, write_pos=wp[:batch],
                      state_indices=slots[:batch], sketch_indices=idx[:batch],
                      out=out[:batch], sketch=sketch)
            def flush_path():
                skd.sketch_flush(flush_rows=r, **sh)
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream): flush_path()
            torch.cuda.current_stream().wait_stream(stream)
            gp = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gp): flush_path()
            t = timed(gp.replay)
            rows.append(dict(n_flush=n_flush, pad_batch=batch, flush_path_ms=t))
            print(f"{n_flush:8d} {batch:10d} {t:14.4f}")
    json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "flush_grid_probe.json", "w"), indent=1)

main()
