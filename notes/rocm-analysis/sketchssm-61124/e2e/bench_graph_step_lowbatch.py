# SPDX-License-Identifier: Apache-2.0
"""Per-decode-step GPU time of one Mamba-2 layer at batch 1/2/4/8, CUDA-graph
captured: stock vLLM `selective_state_update` vs the SketchSSM step of PR #61124.

Adapted from ../bench_graph_step.py (which covers batch 128/256/512): same tensors,
same kernels, only the batch list and the model shape are narrowed to the
Nano-4B layer (96 heads, head_dim 80, dstate 128, 8 groups) and W=16.

Two SketchSSM variants are timed because the flush has to be amortised over W=16
steps and batch <= 8 makes "rows flushing per step" smaller than one:
  sketch_flush1   -- one flush row every step (the convention bench_graph_step.py
                     uses for large batches: batch/16 rows per step). For batch <= 8
                     this flushes 16x more often than a real sequence does, so it is
                     an upper bound on the SketchSSM step.
  sketch_noflush  -- no row flushes (sketch_decode only), i.e. what 15 of every 16
                     steps cost. Lower bound.
The realised step cost for a rolling batch of b sequences is between the two, at
b/16 flushed rows per step on average.

Usage:
  CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/app/local/yh/vllm \
    python3 bench_graph_step_lowbatch.py bench_graph_step_lowbatch.json
"""

import json
import sys

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_build as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_decode as skd

W, N, G, SLOTS = 16, 128, 8, 512
H, P = 96, 80
BATCHES = (1, 2, 4, 8)


def timed(fn, iters=200):
    for _ in range(20):
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
    print(f"sketch dot precision: {sk.SKETCH_DOT_PRECISION!r}", flush=True)
    rows = []
    for batch in BATCHES:
        g = torch.Generator(device=dev).manual_seed(1)

        def rnd(*shape, dtype=torch.float32, scale=1.0):
            return (torch.randn(shape, generator=g, dtype=torch.float32) * scale).to(dtype)

        state_std = rnd(SLOTS, H, P, N, scale=0.1)
        state_key = state_std.clone().transpose(-1, -2).contiguous()
        state_view = state_key.transpose(-1, -2)
        x = rnd(batch, H, P, dtype=torch.bfloat16, scale=0.3)
        dt = rnd(batch, H, dtype=torch.bfloat16, scale=0.5)[..., None].expand(-1, -1, P)
        A = rnd(H, 1, 1, scale=0.1).expand(H, P, N)
        B = rnd(batch, G, N, dtype=torch.bfloat16, scale=0.3)
        C = rnd(batch, G, N, dtype=torch.bfloat16, scale=0.3)
        D = rnd(H, 1, scale=0.1).expand(H, P)
        dt_bias = rnd(H, 1, scale=0.1).expand(H, P)
        x_cache = rnd(SLOTS, H, W, P, scale=0.3)
        dt_cache = rnd(SLOTS, H, W, scale=0.5)
        B_cache = rnd(SLOTS, G, W, N, dtype=torch.bfloat16, scale=0.3)
        bc_pre = torch.empty(batch, G, W)
        out_std = torch.empty(batch, H, P, dtype=torch.bfloat16)
        out_sk = torch.empty_like(out_std)
        slots = torch.arange(SLOTS, dtype=torch.int64)
        idx = torch.arange(batch, dtype=torch.int32)

        frames = torch.linalg.qr(torch.randn(G, N, N, generator=g))[0].float()
        ranks = torch.tensor([0, 1, 2, 3, 4, 5, 9, 20, 38, 64] * H, dtype=torch.int32)[:H]
        ranks = (ranks.float() * (8.0 / ranks.float().clamp(min=1).mean())).round().to(torch.int32)
        tables = sk.SketchTables(ranks, N)
        bufs = [torch.zeros(SLOTS, *s, dtype=d) for s, d in
                zip(sk.sketch_shapes(ranks, P, N), sk.SKETCH_DTYPES)]
        sketch = sk.SketchArgs(bufs[0], bufs[1], bufs[2], tables,
                               frames.transpose(-1, -2).contiguous())
        sk.sketch_build(state_view, idx, idx, torch.ones(batch, dtype=torch.int8), sketch)

        def std_step():
            selective_state_update(state_std, x, dt, A, B, C, D, dt_bias, z=None,
                                   dt_softplus=True, state_batch_indices=slots[:batch],
                                   out=out_std)

        query = torch.empty(batch, G, N, dtype=torch.float32)
        flush_rows = torch.full((batch,), -1, dtype=torch.int32)
        flush_rows[0] = 0            # one flush row per step: upper bound
        is_flush = torch.zeros(batch, dtype=torch.int8)
        is_flush[0] = 1
        no_flush_rows = torch.full((batch,), -1, dtype=torch.int32)
        no_flush = torch.zeros(batch, dtype=torch.int8)

        def make_step(pos, is_flush_, flush_rows_):
            def step():
                sk._rotate_groups(B.flatten(1), sketch.frames_t, query.view(batch, -1),
                                  block_t=32)
                common = dict(state=state_view, x=x, dt=dt, A=A, B=B, query=query, D=D,
                              dt_bias=dt_bias, x_cache=x_cache, dt_cache=dt_cache,
                              B_cache=B_cache, write_pos=torch.full((batch,), pos,
                                                                    dtype=torch.int32),
                              state_indices=slots[:batch], sketch_indices=idx,
                              out=out_sk, sketch=sketch)
                skd.sketch_decode(C=C, bc_pre=bc_pre, is_flush=is_flush_, **common)
                skd.sketch_flush(flush_rows=flush_rows_, **common)
            return step

        sketch_flush1 = make_step(W - 1, is_flush, flush_rows)
        sketch_noflush = make_step(W - 2, no_flush, no_flush_rows)

        stream = torch.cuda.Stream()
        graphs = {}
        for name, fn in (("std", std_step), ("flush1", sketch_flush1),
                         ("noflush", sketch_noflush)):
            with torch.cuda.stream(stream):
                fn()
            torch.cuda.current_stream().wait_stream(stream)
            gp = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gp):
                fn()
            graphs[name] = gp

        t_std_eager = timed(std_step)
        t_flush1_eager = timed(sketch_flush1)
        t_std = timed(graphs["std"].replay)
        t_flush1 = timed(graphs["flush1"].replay)
        t_noflush = timed(graphs["noflush"].replay)
        row = dict(heads=H, dim=P, dstate=N, groups=G, batch=batch, W=W,
                   mean_rank=float(ranks.float().mean()),
                   std_eager_ms=t_std_eager, flush1_eager_ms=t_flush1_eager,
                   std_graph_ms=t_std, sketch_flush1_graph_ms=t_flush1,
                   sketch_noflush_graph_ms=t_noflush,
                   ratio_flush1_graph=t_flush1 / t_std,
                   ratio_noflush_graph=t_noflush / t_std)
        rows.append(row)
        print(f"batch {batch}: std {t_std:.5f} ms (eager {t_std_eager:.5f})  "
              f"sketch_flush1 {t_flush1:.5f} ms  sketch_noflush {t_noflush:.5f} ms  "
              f"-> flush1/std {t_flush1 / t_std:.3f}x  noflush/std {t_noflush / t_std:.3f}x",
              flush=True)
    out = sys.argv[1] if len(sys.argv) > 1 else "bench_graph_step_lowbatch.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
