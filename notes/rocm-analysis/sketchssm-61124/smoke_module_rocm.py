# SPDX-License-Identifier: Apache-2.0
"""Smoke test: Mamba2SketchSSM.decode() (the module wrapper, incl. the aux
stream that runs the flush concurrently with the sketch reads) on ROCm.

The kernel-level test (tests/kernels/mamba/test_sketchssm_mamba2.py) calls
sketch_decode and sketch_flush sequentially, so it never exercises
maybe_execute_in_parallel / aux_stream. This does, and checks that the
two-stream result matches the sequential one bit for bit.

Usage: CUDA_VISIBLE_DEVICES=0 python3 smoke_module_rocm.py
"""

import torch

from vllm.model_executor.layers.mamba.sketchssm.calibration import (
    SketchSSMCalibration,
)
from vllm.model_executor.layers.mamba.sketchssm.mamba2 import Mamba2SketchSSM
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

H, P, G, N, W = 96, 80, 8, 128, 16  # Nemotron-3-Nano-4B Mamba-2 layer
SLOTS, BATCH = 256, 256
MEAN_RANK = 8


class Meta:
    """The subset of Mamba2AttentionMetadata that decode() reads."""

    def __init__(self, write_pos, is_flush, sketch_idx, flush_rows, bc_pre):
        self.write_pos_d = write_pos
        self.is_flush_d = is_flush
        self.sketch_indices_d = sketch_idx
        self.sketch_flush_rows_d = flush_rows
        self.bc_pre_scratch = bc_pre


def make_calibration(heads: int) -> SketchSSMCalibration:
    g = torch.Generator().manual_seed(0)
    frames = torch.linalg.qr(torch.randn(1, G, N, N, generator=g))[0]
    ranks = [0, 1, 2, 3, 4, 5, 9, 20, 38, 64] * heads  # dense + sketch heads
    ranks = torch.tensor(ranks[:heads], dtype=torch.int32) + 4  # mean 30 -> scaled below
    # Scale to a mean near MEAN_RANK.
    ranks = (ranks.float() * (MEAN_RANK / ranks.float().mean())).round().clamp(min=0)
    return SketchSSMCalibration(
        frames=frames.to(torch.float32),
        ranks=ranks.to(torch.int32).unsqueeze(0),
        layer_ids=(0,),
    )


def build_inputs(dev):
    g = torch.Generator(device="cpu").manual_seed(1)

    def rnd(*shape, dtype=torch.float32, scale=1.0):
        t = torch.randn(shape, generator=g, dtype=torch.float32) * scale
        return t.to(dev, dtype)

    state = rnd(SLOTS, H, N, P, scale=0.1)  # key-major (slots, heads, dstate, dim)
    x = rnd(BATCH, H, P, dtype=torch.bfloat16, scale=0.3)
    dt = rnd(BATCH, H, dtype=torch.bfloat16, scale=0.5).unsqueeze(-1).expand(-1, -1, P)
    A = rnd(H, 1, 1, scale=0.1).expand(H, P, N)  # per-head scalar: stride 0
    B = rnd(BATCH, G, N, dtype=torch.bfloat16, scale=0.3)
    C = rnd(BATCH, G, N, dtype=torch.bfloat16, scale=0.3)
    D = rnd(H, 1, scale=0.1).expand(H, P)
    dt_bias = rnd(H, 1, scale=0.1).expand(H, P)
    x_cache = rnd(SLOTS, H, W, P, scale=0.3)
    dt_cache = rnd(SLOTS, H, W, scale=0.5)
    B_cache = rnd(SLOTS, G, W, N, dtype=torch.bfloat16, scale=0.3)
    is_flush = torch.zeros(BATCH, dtype=torch.int8, device=dev)
    is_flush[::W] = 1  # one flush row per window
    write_pos = torch.arange(BATCH, device=dev, dtype=torch.int32) % W
    slots = torch.arange(SLOTS, device=dev, dtype=torch.int64)
    slots[BATCH - 1] = NULL_BLOCK_ID  # padding row
    sketch_idx = torch.arange(BATCH, device=dev, dtype=torch.int32)
    flush = torch.nonzero(is_flush).flatten().to(torch.int32)
    flush_rows = torch.full((BATCH,), -1, dtype=torch.int32, device=dev)
    flush_rows[: flush.numel()] = flush
    return dict(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        dt_bias=dt_bias,
        x_cache=x_cache,
        dt_cache=dt_cache,
        B_cache=B_cache,
        write_pos=write_pos,
        is_flush=is_flush,
        state_indices=slots,
        sketch_indices=sketch_idx,
        flush_rows=flush_rows,
    )


def main() -> None:
    dev = torch.device("cuda")
    torch.manual_seed(0)
    cal = make_calibration(H)
    mod = Mamba2SketchSSM(
        cal, 0, H, P, G, N, max_num_reqs=SLOTS, tp_size=1, tp_rank=0
    ).to(dev)
    print(f"mean rank {cal.ranks.float().mean():.1f}, "
          f"sketch u {tuple(mod.u.shape)} w {tuple(mod.w.shape)} ag {tuple(mod.ag.shape)}")
    print("aux stream:", mod._flush_stream)

    out_par = torch.empty(BATCH, H, P, dtype=torch.bfloat16, device=dev)
    out_seq = torch.empty_like(out_par)
    bc_pre = torch.empty(BATCH, G, W, device=dev)

    def run(state, out, flush_stream):
        inp = build_inputs(dev)
        meta = Meta(inp["write_pos"], inp["is_flush"], inp["sketch_indices"],
                    inp["flush_rows"], bc_pre)
        saved = mod._flush_stream
        if not flush_stream:
            mod._flush_stream = None  # sequential: both fns on the default stream
        # decode() takes a (slots, heads, dim, dstate) view of the key-major cache.
        mod.decode(
            state=state.transpose(-1, -2), x=inp["x"], dt=inp["dt"], A=inp["A"], B=inp["B"],
            C=inp["C"], D=inp["D"], dt_bias=inp["dt_bias"],
            x_cache=inp["x_cache"], dt_cache=inp["dt_cache"], B_cache=inp["B_cache"],
            state_indices=inp["state_indices"], attn_metadata=meta, out=out,
        )
        mod._flush_stream = saved
        return inp

    # Sequential vs multi-stream, from identical inputs and states.
    torch.manual_seed(0)
    st_a = torch.randn(SLOTS, H, N, P, device=dev) * 0.1
    st_b = st_a.clone()
    # Build every row's sketch from the initial state, as prefill does, so the
    # sketch read is part of the comparison too.
    from vllm.model_executor.layers.mamba.ops.sketchssm_mamba2_build import (
        sketch_build,
    )

    rows = torch.arange(BATCH, device=dev, dtype=torch.int32)
    flag = torch.ones(BATCH, dtype=torch.int8, device=dev)
    # sketch_build takes a (slots, heads, dim, dstate) view of the key-major state.
    sketch_build(st_a.transpose(-1, -2), rows, rows, flag, mod.sketch)
    sketch_build(st_b.transpose(-1, -2), rows, rows, flag, mod.sketch)
    torch.cuda.synchronize()
    print("sketch u nonzero rows:", int((mod.u.abs().sum(-1) > 0).sum()))
    run(st_a, out_seq, flush_stream=False)
    run(st_b, out_par, flush_stream=True)
    torch.cuda.synchronize()
    same_out = torch.equal(out_seq, out_par)
    same_state = torch.equal(st_a, st_b)
    print(f"multi-stream == sequential: out {same_out}, state {same_state}")
    print("out finite:", torch.isfinite(out_par.float()).all().item())

    # Latency of one mixed step (1 flush row per 16) through the wrapper.
    inp = build_inputs(dev)
    meta = Meta(inp["write_pos"], inp["is_flush"], inp["sketch_indices"],
                inp["flush_rows"], bc_pre)
    for _ in range(20):
        mod.decode(state=st_a.transpose(-1, -2), x=inp["x"], dt=inp["dt"], A=inp["A"], B=inp["B"],
                   C=inp["C"], D=inp["D"], dt_bias=inp["dt_bias"],
                   x_cache=inp["x_cache"], dt_cache=inp["dt_cache"],
                   B_cache=inp["B_cache"], state_indices=inp["state_indices"],
                   attn_metadata=meta, out=out_par)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(100):
        mod.decode(state=st_a.transpose(-1, -2), x=inp["x"], dt=inp["dt"], A=inp["A"], B=inp["B"],
                   C=inp["C"], D=inp["D"], dt_bias=inp["dt_bias"],
                   x_cache=inp["x_cache"], dt_cache=inp["dt_cache"],
                   B_cache=inp["B_cache"], state_indices=inp["state_indices"],
                   attn_metadata=meta, out=out_par)
    end.record()
    torch.cuda.synchronize()
    print(f"wrapper decode (mixed step, batch {BATCH}, H={H}, P={P}): "
          f"{start.elapsed_time(end) / 100:.4f} ms/step")


if __name__ == "__main__":
    main()
