# SPDX-License-Identifier: Apache-2.0
"""GPU-time per decode step (CUDA-graph captured, so CPU launch overhead is
excluded) on ROCm: stock vLLM decode vs the SketchSSM kernels of PR #61124.

The engine runs both under CUDA graphs, so this is the number that matters for
serving. Times are per Mamba-2 layer per decode step.

Usage: CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/app/local/yh/vllm python3 bench_graph_step.py
"""

import json
import sys

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_build as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_decode as skd

W, N, G, SLOTS = 16, 128, 8, 512
SHAPES = [
    ("Nano-4B  (H=96,  P=80)", 96, 80),
    ("Nano-9B  (H=128, P=80)", 128, 80),
]


def timed(fn, iters=100):
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
    print(f"sketch dot precision: {sk.SKETCH_DOT_PRECISION!r}")
    import os
    from vllm.model_executor.layers.mamba.ops import sketchssm_config as skc

    if os.environ.get("VLLM_SKETCHSSM_FLUSH_CFG"):
        skc._overrides["mamba2_flush"] = tuple(
            int(x) for x in os.environ["VLLM_SKETCHSSM_FLUSH_CFG"].split(",")
        )
        print(f"flush cfg override: {skc._overrides['mamba2_flush']}")
    if os.environ.get("VLLM_SKETCHSSM_DECODE_CFG"):
        skc._overrides["mamba2_decode"] = tuple(
            int(x) for x in os.environ["VLLM_SKETCHSSM_DECODE_CFG"].split(",")
        )
        print(f"decode cfg override: {skc._overrides['mamba2_decode']}")
    rows = []
    for name, H, P in SHAPES:
        for batch in (128, 256, 512):
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
            write_pos = torch.arange(batch, dtype=torch.int32) % W

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

            query = torch.empty(batch, G, N, dtype=torch.float32)  # reused in the graph

            # Metadata is precomputed on the host by the engine; build it once.
            rows_f = list(range(0, batch, W))
            flush_g = torch.zeros(batch, dtype=torch.int8)
            flush_g[rows_f] = 1
            flush_rows_g = torch.tensor(rows_f + [-1] * (batch - len(rows_f)),
                                        dtype=torch.int32)
            wp_g = torch.full((batch,), W - 1, dtype=torch.int32)

            def sketch_step_indexed(pos):
                """One step at ring position pos for every row: 1 flush row in W."""
                flush, flush_rows, wp = flush_g, flush_rows_g, wp_g
                sk._rotate_groups(B.flatten(1), sketch.frames_t, query.view(batch, -1), block_t=32)
                common = dict(state=state_view, x=x, dt=dt, A=A, B=B, query=query, D=D,
                              dt_bias=dt_bias, x_cache=x_cache, dt_cache=dt_cache,
                              B_cache=B_cache, write_pos=wp, state_indices=slots[:batch],
                              sketch_indices=idx, out=out_sk, sketch=sketch)
                skd.sketch_decode(C=C, bc_pre=bc_pre, is_flush=flush, **common)
                skd.sketch_flush(flush_rows=flush_rows, **common)

            # Sequential (eager) step at the same position, for reference.
            pos_eager = (torch.arange(batch, dtype=torch.int32) % W)
            flush_rows_e = torch.nonzero(pos_eager == W - 1).flatten().to(torch.int32)
            flush_rows_e = torch.cat([flush_rows_e,
                                      torch.full((batch - flush_rows_e.numel(),), -1,
                                                 dtype=torch.int32)])
            is_flush_e = (pos_eager == W - 1).to(torch.int8)

            def sketch_step_eager():
                q = sk.sketch_query(C, sketch.frames_t)
                common = dict(state=state_view, x=x, dt=dt, A=A, B=B, query=q, D=D,
                              dt_bias=dt_bias, x_cache=x_cache, dt_cache=dt_cache,
                              B_cache=B_cache, write_pos=pos_eager, state_indices=slots[:batch],
                              sketch_indices=idx, out=out_sk, sketch=sketch)
                skd.sketch_decode(C=C, bc_pre=bc_pre, is_flush=is_flush_e, **common)
                skd.sketch_flush(flush_rows=flush_rows_e, **common)

            t_std = timed(std_step)
            t_eager = timed(sketch_step_eager)

            # CUDA-graph captured step: excludes CPU launch overhead, like serving.
            gp = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                sketch_step_indexed(W - 1)  # warm the exact graph inputs
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(gp):
                sketch_step_indexed(W - 1)
            t_graph = timed(gp.replay)

            gp_std = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                std_step()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.cuda.graph(gp_std):
                std_step()
            t_std_graph = timed(gp_std.replay)

            row = dict(shape=name, heads=H, dim=P, batch=batch, W=W,
                       mean_rank=float(ranks.float().mean()),
                       std_ms=t_std, std_graph_ms=t_std_graph,
                       sketch_eager_ms=t_eager, sketch_graph_ms=t_graph,
                       speedup_eager=t_std / t_eager,
                       speedup_graph=t_std_graph / t_graph)
            rows.append(row)
            print(f"{name} batch {batch:4d}: std {t_std:.4f}/{t_std_graph:.4f}  "
                  f"sketch {t_eager:.4f}/{t_graph:.4f} ms/step (eager/graph)  "
                  f"graph speedup {t_std_graph / t_graph:.2f}x", flush=True)
    out = sys.argv[1] if len(sys.argv) > 1 else "bench_graph_step.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
