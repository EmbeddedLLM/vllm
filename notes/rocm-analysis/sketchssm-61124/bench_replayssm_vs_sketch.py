# SPDX-License-Identifier: Apache-2.0
"""ReplaySSM vs SketchSSM vs stock decode, one Mamba-2 layer, graph-captured, gfx950."""
import json, os, sys
import torch
from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_state_update
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (
    selective_state_update_replayssm_output_only as replayssm_step,
)
from vllm.model_executor.layers.mamba.ops import sketchssm_config as skc
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_build as sk
from vllm.model_executor.layers.mamba.ops import sketchssm_mamba2_decode as skd

W, N, G, SLOTS, H, P = 16, 128, 8, 512, 96, 80
if os.environ.get("VLLM_SKETCHSSM_FLUSH_CFG"):
    skc._overrides["mamba2_flush"] = tuple(int(x) for x in os.environ["VLLM_SKETCHSSM_FLUSH_CFG"].split(","))

def timed(fn, iters=100):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters): fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / iters

def cap(fn):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s): fn()
    torch.cuda.current_stream().wait_stream(s)
    gp = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gp): fn()
    return gp

def main():
    dev = torch.device("cuda"); torch.set_default_device(dev); torch.manual_seed(0)
    g = torch.Generator(device=dev).manual_seed(1)
    def rnd(*s, dtype=torch.float32, scale=1.0):
        return (torch.randn(s, generator=g, dtype=torch.float32) * scale).to(dtype)
    state = rnd(SLOTS, H, P, N, scale=0.1)                 # (slots, heads, dim, dstate)
    state_key = state.clone().transpose(-1, -2).contiguous()
    state_view = state_key.transpose(-1, -2)
    x = rnd(SLOTS, H, P, dtype=torch.bfloat16, scale=0.3)
    dt = rnd(SLOTS, H, dtype=torch.bfloat16, scale=0.5)[..., None].expand(-1, -1, P)
    A = rnd(H, 1, 1, scale=0.1).expand(H, P, N)
    B = rnd(SLOTS, G, N, dtype=torch.bfloat16, scale=0.3)
    C = rnd(SLOTS, G, N, dtype=torch.bfloat16, scale=0.3)
    D = rnd(H, 1, scale=0.1).expand(H, P); dt_bias = rnd(H, 1, scale=0.1).expand(H, P)
    x_cache = rnd(SLOTS, H, W, P, scale=0.3); dt_cache = rnd(SLOTS, H, W, scale=0.5)
    B_cache = rnd(SLOTS, G, W, N, dtype=torch.bfloat16, scale=0.3)
    bc_pre = torch.empty(SLOTS, G, W)
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

    rows = []
    for batch in (128, 256, 512):
        wp = torch.arange(batch, dtype=torch.int32) % W
        r0 = list(range(0, batch, W))
        is_flush = torch.zeros(batch, dtype=torch.int8); is_flush[r0] = 1
        flush_rows = torch.tensor(r0 + [-1] * (batch - len(r0)), dtype=torch.int32)

        def std():
            selective_state_update(state, x[:batch], dt[:batch], A, B[:batch], C[:batch],
                                   D, dt_bias, z=None, dt_softplus=True,
                                   state_batch_indices=slots[:batch], out=out[:batch])
        def rep():
            replayssm_step(state, x[:batch], dt[:batch], A, B[:batch], C[:batch], D, dt_bias,
                           dt_softplus=True, x_cache=x_cache, dt_cache=dt_cache,
                           B_cache=B_cache, bc_pre=bc_pre[:batch], write_pos=wp,
                           is_flush=is_flush, max_cache_len=W,
                           state_batch_indices=slots[:batch], out=out[:batch])
        sh = dict(state=state_view, x=x[:batch], dt=dt[:batch], A=A, B=B[:batch],
                  query=query[:batch], D=D, dt_bias=dt_bias, x_cache=x_cache,
                  dt_cache=dt_cache, B_cache=B_cache, write_pos=wp,
                  state_indices=slots[:batch], sketch_indices=idx[:batch], out=out[:batch],
                  sketch=sketch)
        def sk_mixed():
            sk._rotate_groups(B[:batch].flatten(1), sketch.frames_t, query[:batch].view(batch, -1), block_t=32)
            skd.sketch_decode(C=C[:batch], bc_pre=bc_pre[:batch], is_flush=is_flush, **sh)
            skd.sketch_flush(flush_rows=flush_rows, **sh)

        t_std, t_rep, t_sk = (timed(cap(f).replay) for f in (std, rep, sk_mixed))
        rows.append(dict(batch=batch, std_ms=t_std, replayssm_ms=t_rep, sketchssm_ms=t_sk,
                         speedup_replayssm=t_std/t_rep, speedup_sketchssm=t_std/t_sk,
                         sketch_vs_replayssm=t_rep/t_sk))
        print(f"batch {batch:4d}: standard {t_std:.4f} | ReplaySSM {t_rep:.4f} "
              f"({t_std/t_rep:.2f}x) | SketchSSM {t_sk:.4f} ({t_std/t_sk:.2f}x) | "
              f"sketch/replayssm {t_rep/t_sk:.2f}x", flush=True)
    json.dump(rows, open(sys.argv[1] if len(sys.argv) > 1 else "bench_replayssm_vs_sketch.json", "w"), indent=1)

main()
