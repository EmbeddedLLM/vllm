# SPDX-License-Identifier: Apache-2.0
"""Per-kernel breakdown of sketch_flush() on gfx950: where does the fixed cost go?"""
import torch
from torch.profiler import ProfilerActivity, profile
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_build as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_decode as skd

W, N, G, SLOTS, H, P = 16, 128, 8, 512, 96, 80
BATCH = 512

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

    sh = dict(state=state_view, x=x[:BATCH], dt=dt[:BATCH], A=A, B=B[:BATCH],
              query=query[:BATCH], D=D, dt_bias=dt_bias, x_cache=x_cache,
              dt_cache=dt_cache, B_cache=B_cache, write_pos=wp[:BATCH],
              state_indices=slots[:BATCH], sketch_indices=idx[:BATCH], out=out[:BATCH],
              sketch=sketch)
    for n_flush in (1, 2, 8):
        r = torch.tensor(list(range(n_flush)) + [-1] * (BATCH - n_flush), dtype=torch.int32)
        for _ in range(5):
            skd.sketch_flush(flush_rows=r, **sh)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(20):
                skd.sketch_flush(flush_rows=r, **sh)
            torch.cuda.synchronize()
        print(f"--- n_flush={n_flush}")
        for e in prof.key_averages():
            if e.self_device_time_total > 0:
                print(f"   {e.key[:56]:<56} {e.self_device_time_total/20:9.1f} us/call")
main()
