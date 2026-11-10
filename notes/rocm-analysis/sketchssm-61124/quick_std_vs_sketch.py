# SPDX-License-Identifier: Apache-2.0
"""Per-decode-step latency of one Mamba-2 layer on ROCm: stock vLLM decode
(selective_state_update) vs the SketchSSM kernels of PR #61124.

One layer, batch = B rows, FP32 state cache, W=16 window, mean sketch rank 8.
Times the kernels only (no Python-side metadata work).

Usage: CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/app/local/yh/vllm python3 quick_std_vs_sketch.py
"""

import json
import sys

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_build as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_decode as skd
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

W = 16
N = 128  # dstate
G = 8  # groups
SLOTS = 512
SHAPES = {
    "Nano-4B  (H=96,  P=80)": (96, 80),
    "Nano-9B  (H=128, P=80)": (128, 80),
}


def make_sketch(mod_dev, heads, dim, max_reqs):
    g = torch.Generator(device=mod_dev).manual_seed(0)
    frames = torch.linalg.qr(torch.randn(1, G, N, N, generator=g))[0].float()
    ranks = [0, 1, 2, 3, 4, 5, 9, 20, 38, 64] * heads
    ranks = torch.tensor(ranks[:heads], dtype=torch.int32)
    ranks = (ranks.float() * (8.0 / ranks.float().clamp(min=1).mean())).round()
    ranks = ranks.to(torch.int32)
    tables = sk.SketchTables(ranks, N)
    shapes = sk.sketch_shapes(ranks, dim, N)
    bufs = [
        torch.zeros(max_reqs, *s, dtype=d, device=mod_dev)
        for s, d in zip(shapes, sk.SKETCH_DTYPES)
    ]
    frames_t = frames[0].transpose(-1, -2).contiguous()  # (groups, N, N) of this layer
    return sk.SketchArgs(bufs[0], bufs[1], bufs[2], tables, frames_t), ranks


def main() -> None:
    dev = torch.device("cuda")
    torch.set_default_device(dev)  # what the vLLM engine does; SketchTables uses it
    torch.manual_seed(0)
    out_json = []
    print(f"dot precision: {sk.SKETCH_DOT_PRECISION!r} (env VLLM_SKETCHSSM_DOT)")
    for name, (H, P) in SHAPES.items():
        for batch in (128, 256, 512):
            g = torch.Generator(device=dev).manual_seed(1)

            def rnd(*shape, dtype=torch.float32, scale=1.0):
                t = torch.randn(shape, generator=g, dtype=torch.float32) * scale
                return t.to(dtype)

            # Stock path: state is (slots, heads, dim, dstate) and is written each step.
            state_std = rnd(SLOTS, H, P, N, scale=0.1)
            # SketchSSM cache: key-major (slots, heads, dstate, dim).
            state_key = state_std.clone().transpose(-1, -2).contiguous()

            x = rnd(batch, H, P, dtype=torch.bfloat16, scale=0.3)
            dt = rnd(batch, H, dtype=torch.bfloat16, scale=0.5)
            dt3 = dt[..., None].expand(-1, -1, P)
            A = rnd(H, 1, 1, scale=0.1).expand(H, P, N)
            B = rnd(batch, G, N, dtype=torch.bfloat16, scale=0.3)
            C = rnd(batch, G, N, dtype=torch.bfloat16, scale=0.3)
            D = rnd(H, 1, scale=0.1).expand(H, P)
            dt_bias = rnd(H, 1, scale=0.1).expand(H, P)
            x_cache = rnd(SLOTS, H, W, P, scale=0.3)
            dt_cache = rnd(SLOTS, H, W, scale=0.5)
            B_cache = rnd(SLOTS, G, W, N, dtype=torch.bfloat16, scale=0.3)
            bc_pre = torch.empty(batch, G, W, device=dev)
            out_std = torch.empty(batch, H, P, dtype=torch.bfloat16, device=dev)
            out_sk = torch.empty_like(out_std)
            slots = torch.arange(SLOTS, device=dev, dtype=torch.int64)
            idx = torch.arange(batch, device=dev, dtype=torch.int32)
            write_pos = (torch.arange(batch, device=dev, dtype=torch.int32) % W)

            sketch, ranks = make_sketch(dev, H, P, SLOTS)
            state_view = state_key.transpose(-1, -2)
            sk.sketch_build(state_view, idx, idx,
                            torch.ones(batch, dtype=torch.int8, device=dev), sketch)

            def std_step():
                selective_state_update(
                    state_std, x, dt3, A, B, C, D, dt_bias, z=None,
                    dt_softplus=True, state_batch_indices=slots[:batch],
                    out=out_std,
                )

            def sketch_step(is_flush: torch.Tensor, flush_rows: torch.Tensor):
                q = sk.sketch_query(C, sketch.frames_t)
                common = dict(
                    state=state_view, x=x, dt=dt3, A=A, B=B, query=q, D=D,
                    dt_bias=dt_bias, x_cache=x_cache, dt_cache=dt_cache,
                    B_cache=B_cache, write_pos=write_pos, state_indices=slots[:batch],
                    sketch_indices=idx, out=out_sk, sketch=sketch,
                )
                skd.sketch_decode(C=C, bc_pre=bc_pre, is_flush=is_flush, **common)
                skd.sketch_flush(flush_rows=flush_rows, **common)

            no_flush = torch.zeros(batch, dtype=torch.int8, device=dev)
            empty_rows = torch.full((batch,), -1, dtype=torch.int32, device=dev)
            mixed = torch.zeros(batch, dtype=torch.int8, device=dev)
            mixed[::W] = 1
            mixed_rows = torch.nonzero(mixed).flatten().to(torch.int32)
            mixed_rows = torch.cat(
                [mixed_rows, torch.full((batch - mixed_rows.numel(),), -1,
                                        dtype=torch.int32, device=dev)]
            )

            def bench(fn, iters=100):
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

            t_std = bench(std_step)
            t_nf = bench(lambda: sketch_step(no_flush, empty_rows))
            t_mx = bench(lambda: sketch_step(mixed, mixed_rows))
            row = dict(shape=name, heads=H, dim=P, batch=batch, W=W,
                       mean_rank=float(ranks.float().mean()),
                       std_ms=t_std, sketch_nonflush_ms=t_nf,
                       sketch_mixed_ms=t_mx,
                       speedup_nonflush=t_std / t_nf,
                       speedup_mixed=t_std / t_mx)
            out_json.append(row)
            print(
                f"{name} batch {batch:4d}: std {t_std:.4f}  "
                f"sketch(non-flush) {t_nf:.4f} ({t_std / t_nf:.2f}x)  "
                f"sketch(mixed 1/{W}) {t_mx:.4f} ({t_std / t_mx:.2f}x)  ms/step",
                flush=True,
            )
    with open(sys.argv[1] if len(sys.argv) > 1 else "quick_std_vs_sketch.json", "w") as f:
        json.dump(out_json, f, indent=1)


if __name__ == "__main__":
    main()
