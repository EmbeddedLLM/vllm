# SPDX-License-Identifier: Apache-2.0
"""Where does the SketchSSM flush step spend its time on gfx950 (MI355X)?

The mixed step (1 flush row in W=16) costs much more than an all-non-flush step
(agent benchmark: 0.17 vs 0.038 ms at batch 128), so this measures the flush
path as a function of the number of flush rows, and splits it into its three
launches (query rotation, window keys, flush+rebuild).

Usage: CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/app/local/yh/vllm python3 flush_scaling.py
"""

import json
import sys

import torch

from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_build as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_decode as skd

W, N, G, SLOTS, H, P = 16, 128, 8, 512, 96, 80
BATCH = 512


def timed(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main() -> None:
    dev = torch.device("cuda")
    torch.set_default_device(dev)
    torch.manual_seed(0)
    g = torch.Generator(device=dev).manual_seed(1)

    def rnd(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(shape, generator=g, dtype=torch.float32) * scale).to(dtype)

    state_key = rnd(SLOTS, H, N, P, scale=0.1)
    state_view = state_key.transpose(-1, -2)
    x = rnd(SLOTS, H, P, dtype=torch.bfloat16, scale=0.3)
    dt = rnd(SLOTS, H, dtype=torch.bfloat16, scale=0.5)[..., None].expand(-1, -1, P)
    A = rnd(H, 1, 1, scale=0.1).expand(H, P, N)
    B = rnd(SLOTS, G, N, dtype=torch.bfloat16, scale=0.3)
    C = rnd(SLOTS, G, N, dtype=torch.bfloat16, scale=0.3)
    D = rnd(H, 1, scale=0.1).expand(H, P)
    dt_bias = rnd(H, 1, scale=0.1).expand(H, P)
    x_cache = rnd(SLOTS, H, W, P, scale=0.3)
    dt_cache = rnd(SLOTS, H, W, scale=0.5)
    B_cache = rnd(SLOTS, G, W, N, dtype=torch.bfloat16, scale=0.3)
    out = torch.empty(SLOTS, H, P, dtype=torch.bfloat16)
    bc_pre = torch.empty(SLOTS, G, W)
    query = torch.empty(SLOTS, G, N, dtype=torch.float32)
    slots = torch.arange(SLOTS, dtype=torch.int64)
    idx = torch.arange(SLOTS, dtype=torch.int32)
    frames = torch.linalg.qr(torch.randn(G, N, N, generator=g))[0].float()
    ranks = torch.tensor([0, 1, 2, 3, 4, 5, 9, 20, 38, 64] * H, dtype=torch.int32)[:H]
    ranks = (ranks.float() * (8.0 / ranks.float().clamp(min=1).mean())).round().to(torch.int32)
    tables = sk.SketchTables(ranks, N)
    bufs = [torch.zeros(SLOTS, *s, dtype=d) for s, d in
            zip(sk.sketch_shapes(ranks, P, N), sk.SKETCH_DTYPES)]
    sketch = sk.SketchArgs(bufs[0], bufs[1], bufs[2], tables,
                           frames.transpose(-1, -2).contiguous())
    sk.sketch_build(state_view, idx, idx, torch.ones(SLOTS, dtype=torch.int8), sketch)
    wp = torch.full((SLOTS,), W - 1, dtype=torch.int32)
    noflush = torch.zeros(SLOTS, dtype=torch.int8)
    empty_rows = torch.full((SLOTS,), -1, dtype=torch.int32)

    shared = dict(state=state_view, x=x, dt=dt, A=A, B=B, query=query, D=D,
                  dt_bias=dt_bias, x_cache=x_cache, dt_cache=dt_cache,
                  B_cache=B_cache, write_pos=wp, state_indices=slots,
                  sketch_indices=idx, out=out, sketch=sketch)

    def graph_of(fn):
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        gp = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gp):
            fn()
        return gp

    rows = []

    def rotate_only():
        sk._rotate_groups(B.flatten(1), sketch.frames_t, query.view(SLOTS, -1), block_t=32)

    def decode_only():
        skd.sketch_decode(C=C, bc_pre=bc_pre, is_flush=noflush, **shared)

    t_rot = timed(graph_of(rotate_only).replay)
    t_dec = timed(graph_of(decode_only).replay)
    t_all = timed(graph_of(lambda: (rotate_only(), decode_only())).replay)
    print(f"rotate query {t_rot:.4f} ms | decode non-flush {t_dec:.4f} ms | both {t_all:.4f} ms")
    rows.append(dict(part="rotate_query", ms=t_rot))
    rows.append(dict(part="decode_nonflush", ms=t_dec))
    rows.append(dict(part="rotate+decode", ms=t_all))

    for n_flush in (1, 2, 4, 8, 16, 32, 64, 128, 512):
        mask = torch.zeros(SLOTS, dtype=torch.int8)
        mask[:n_flush] = 1
        r = torch.tensor(list(range(n_flush)) + [-1] * (SLOTS - n_flush), dtype=torch.int32)

        def full():
            rotate_only()
            skd.sketch_decode(C=C, bc_pre=bc_pre, is_flush=mask, **shared)
            skd.sketch_flush(flush_rows=r, **shared)

        t_full = timed(graph_of(full).replay)

        # Flush path alone: window keys + state fold + sketch rebuild, no decode.
        def flush_path():
            skd.sketch_flush(flush_rows=r, **shared)

        t_flush = timed(graph_of(flush_path).replay)
        rows.append(dict(part="step", n_flush=n_flush, full_ms=t_full,
                         flush_path_ms=t_flush, nonflush_part_ms=t_all,
                         flush_increment_ms=t_full - t_all))
        print(f"flush rows {n_flush:4d}: full step {t_full:.4f} ms "
              f"(flush path {t_flush:.4f}, +{t_full - t_all:.4f} over non-flush)")

    with open(sys.argv[1] if len(sys.argv) > 1 else "flush_scaling.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
