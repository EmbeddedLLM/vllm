# SPDX-License-Identifier: Apache-2.0
"""Coordinate sweep of the SketchSSM flush launch config on gfx950 (MI355X).

Shipped flush config (BLOCK_M, num_warps, rows_per_program, BLOCK_V) =
(16, 4, 4, 32) was tuned on an RTX PRO 6000. Measured graph-captured:
  - "mixed" step: 1 flush row in W=16 (the serving steady state),
  - "all-flush" step: every row flushes (worst case, makes the effect visible).

Usage: CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/app/local/yh/vllm python3 sweep_flush_gfx950.py
"""

import json
import sys

import torch

from vllm.model_executor.layers.mamba.ops import sketchssm_config as skc
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_build as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_decode as skd

W, N, G, SLOTS, H, P = 16, 128, 8, 512, 96, 80
SHIPPED = (16, 4, 4, 32)
BATCH = 512


def timed(fn, iters=60):
    for _ in range(10):
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

    stream = torch.cuda.Stream()

    def make_case(n_flush: int):
        flush = torch.zeros(SLOTS, dtype=torch.int8)
        rows = list(range(n_flush))
        flush[rows] = 1
        flush_rows = torch.tensor(rows + [-1] * (SLOTS - len(rows)), dtype=torch.int32)
        return flush, flush_rows

    def step(flush, flush_rows):
        sk._rotate_groups(B.flatten(1), sketch.frames_t, query.view(SLOTS, -1), block_t=32)
        common = dict(state=state_view, x=x, dt=dt, A=A, B=B, query=query, D=D,
                      dt_bias=dt_bias, x_cache=x_cache, dt_cache=dt_cache,
                      B_cache=B_cache, write_pos=wp, state_indices=slots,
                      sketch_indices=idx, out=out, sketch=sketch)
        skd.sketch_decode(C=C, bc_pre=bc_pre, is_flush=flush, **common)
        skd.sketch_flush(flush_rows=flush_rows, **common)

    def bench(flush, flush_rows):
        with torch.cuda.stream(stream):
            step(flush, flush_rows)
        torch.cuda.current_stream().wait_stream(stream)
        gp = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gp):
            step(flush, flush_rows)
        return timed(gp.replay)

    cases = {"mixed (32/512 flush rows)": make_case(SLOTS // W),
             "all-flush (512)": make_case(SLOTS)}
    rows = []
    for name, (flush, flush_rows) in cases.items():
        base = bench(flush, flush_rows)
        print(f"{name}: shipped cfg {SHIPPED} -> {base:.4f} ms", flush=True)
        rows.append(dict(case=name, cfg=SHIPPED, ms=base))
        grid = ([SHIPPED]
                + [(m, *SHIPPED[1:]) for m in (32, 64)]
                + [(SHIPPED[0], w, *SHIPPED[2:]) for w in (1, 2, 8)]
                + [(SHIPPED[0], SHIPPED[1], r, SHIPPED[3]) for r in (1, 2, 8, 16)]
                + [(SHIPPED[0], SHIPPED[1], SHIPPED[2], v) for v in (16, 64, 128)])
        for cfg in dict.fromkeys(grid):
            skc._overrides["mamba2_flush"] = cfg
            try:
                t = bench(flush, flush_rows)
                rows.append(dict(case=name, cfg=cfg, ms=t))
                print(f"  flush cfg {cfg}: {t:.4f} ms "
                      f"({(base - t) / base * 100:+.1f}%)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  flush cfg {cfg}: FAILED {type(e).__name__}: {str(e)[:100]}",
                      flush=True)
                rows.append(dict(case=name, cfg=cfg, error=str(e)[:200]))
        skc._overrides.pop("mamba2_flush", None)
    with open(sys.argv[1] if len(sys.argv) > 1 else "sweep_flush_gfx950.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
