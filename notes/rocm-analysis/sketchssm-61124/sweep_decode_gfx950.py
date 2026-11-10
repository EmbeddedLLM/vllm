# SPDX-License-Identifier: Apache-2.0
"""Coordinate sweep of the SketchSSM decode launch config on gfx950 (MI355X).

Graph-captured timing of one Mamba-2 layer decode step, 1 flush row in W=16,
mean sketch rank 8, DSTATE=128, G=8. The shipped config (BLOCK_M, BLOCK_J,
num_warps) = (128, 16, 1) was tuned on an RTX PRO 6000 (sm_120).

Usage: CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/app/local/yh/vllm python3 sweep_decode_gfx950.py
"""

import json
import sys

import torch

from vllm.model_executor.layers.mamba.ops import sketchssm_config as skc
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_build as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_decode as skd
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update

W, N, G, SLOTS = 16, 128, 8, 512
BATCHES = (256, 512)
H, P = 96, 80  # Nemotron-3-Nano-4B Mamba-2 layer
SHIPPED = (128, 16, 1)


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
    rows = []
    g = torch.Generator(device=dev).manual_seed(1)

    def rnd(*shape, dtype=torch.float32, scale=1.0):
        return (torch.randn(shape, generator=g, dtype=torch.float32) * scale).to(dtype)

    state_std = rnd(SLOTS, H, P, N, scale=0.1)
    state_key = state_std.clone().transpose(-1, -2).contiguous()
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
    out_std = torch.empty(SLOTS, H, P, dtype=torch.bfloat16)
    out_sk = torch.empty_like(out_std)
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
    bc_pre = torch.empty(SLOTS, G, W)
    query = torch.empty(SLOTS, G, N, dtype=torch.float32)

    for batch in BATCHES:
        rows_f = list(range(0, batch, W))
        flush = torch.zeros(batch, dtype=torch.int8)
        flush[rows_f] = 1
        flush_rows = torch.tensor(rows_f + [-1] * (batch - len(rows_f)), dtype=torch.int32)
        wp = torch.full((batch,), W - 1, dtype=torch.int32)

        def step():
            sk._rotate_groups(B.flatten(1), sketch.frames_t, query.view(SLOTS, -1), block_t=32)
            common = dict(state=state_view, x=x[:batch], dt=dt[:batch], A=A,
                          B=B[:batch], query=query[:batch], D=D,
                          dt_bias=dt_bias, x_cache=x_cache, dt_cache=dt_cache,
                          B_cache=B_cache, write_pos=wp, state_indices=slots[:batch],
                          sketch_indices=idx[:batch], out=out_sk[:batch], sketch=sketch)
            skd.sketch_decode(C=C[:batch], bc_pre=bc_pre[:batch], is_flush=flush, **common)
            skd.sketch_flush(flush_rows=flush_rows, **common)

        def std_step():
            selective_state_update(state_std, x[:batch], dt[:batch], A, B[:batch],
                                   C[:batch], D, dt_bias, z=None, dt_softplus=True,
                                   state_batch_indices=slots[:batch], out=out_std[:batch])

        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            step()
            std_step()
        torch.cuda.current_stream().wait_stream(stream)
        gp, gp_std = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(gp):
            step()
        with torch.cuda.graph(gp_std):
            std_step()
        t_std = timed(gp_std.replay)
        print(f"batch {batch}: standard {t_std:.4f} ms/step", flush=True)

        grid = [SHIPPED] + [(m, SHIPPED[1], SHIPPED[2]) for m in (32, 64, 256)] \
            + [(SHIPPED[0], j, SHIPPED[2]) for j in (8, 32, 64)] \
            + [(SHIPPED[0], SHIPPED[1], w) for w in (2, 4, 8)]
        for cfg in dict.fromkeys(grid):
            skc._overrides["mamba2_decode"] = cfg
            try:
                with torch.cuda.stream(stream):
                    step()
                torch.cuda.current_stream().wait_stream(stream)
                gp = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gp):
                    step()
                t = timed(gp.replay)
                rows.append(dict(batch=batch, decode_cfg=cfg, sketch_ms=t,
                                 std_ms=t_std, speedup=t_std / t))
                print(f"  decode cfg {cfg}: {t:.4f} ms ({t_std / t:.2f}x)", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  decode cfg {cfg}: FAILED {type(e).__name__}: {str(e)[:120]}",
                      flush=True)
                rows.append(dict(batch=batch, decode_cfg=cfg, error=str(e)[:200]))
        skc._overrides.pop("mamba2_decode", None)

    out = sys.argv[1] if len(sys.argv) > 1 else "sweep_decode_gfx950.json"
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
