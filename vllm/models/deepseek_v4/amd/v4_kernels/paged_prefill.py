# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sparse prefill attention with two KV sources: paged `unified_kv` (history)
and per-fwd flat `kv` (current chunk's input).

Designed for V4 prefill: indexes the two KV sources directly without
materialising a per-fwd `kv_flat_sa` packed tensor. See
`atom/model_ops/v4_kernels/doc/ATOM_V4_PAGED_PREFILL_DESIGN.zh.md` §1, §3
for design rationale.

Caller contract:
  unified_kv:        [total_pages, D] BF16 — prefix source. Same buffer as
    decode kernel: paged SWA slots in `[0, swa_pages)`, compress pages in
    `[swa_pages, total_pages)`. For prefill, prefix indices select
    (a) prior-chunk SWA history, (b) CSA topk, (c) HCA all-committed.
  kv_indices_prefix: [total_prefix_indices] int32 — flat per-token slot
    lists. Per-token entries live in
    `kv_indices_prefix[kv_indptr_prefix[t] : kv_indptr_prefix[t+1]]`.
    `-1` entries are skipped (sentinel).
  kv_indptr_prefix:  [N+1] int32 — true prefix sum (variable per-token len).

  kv:                [total_tokens, D] BF16 — extend source = current
    fwd's just-computed K (NOT yet written to swa_kv ring). Layout matches
    `swa_write` input.
  kv_indices_extend: [total_extend_indices] int32 — flat per-token row idx
    lists into `kv`. Per-token entries live in
    `kv_indices_extend[kv_indptr_extend[t] : kv_indptr_extend[t+1]]`.
    `-1` entries are skipped (rare for extend; usually all valid).
  kv_indptr_extend:  [N+1] int32 — true prefix sum.

  attn_sink:         [H] per-head learnable softmax-denom bias (V4 specific).
  softmax_scale:     float.

Per-token K loop iterates two regions sequentially, sharing the online
softmax accumulator (m_i, l_i, acc) across regions. Order of regions does
not affect correctness (online softmax is order-invariant).

Returns:
  out: [N, H, D] same dtype as q.

Numerics: identical online-softmax + sink finalization to
`sparse_attn_v4_paged_decode` — bit-exact when the extend region is empty
(then equivalent to a decode call with the same prefix indices).
"""

import torch
import triton
import triton.language as tl

from vllm.models.deepseek_v4.amd.v4_kernels.reference import (
    sparse_attn_ragged_torch,
)

_FP8_DTYPE = torch.float8_e4m3fnuz
_FP8_GROUP_SIZE = 64
_PACKED_FP8_DS_MLA = "fp8_ds_mla"

# Best-known Triton launch metadata for DeepSeek-V4 TP=8 prefill on MI355.
# Keep these local to the vLLM kernel wrapper: ATOM's config/loader surface is
# deliberately not part of this integration.
_BLOCK_H = 16
_BLOCK_K = 32
_NUM_WARPS = 4
_NUM_STAGES = 1
_WAVES_PER_EU = 2

try:
    from aiter.ops.pa_sparse_prefill_opus import pa_sparse_prefill_opus

    _HAS_OPUS = True
except ImportError:
    pa_sparse_prefill_opus = None
    _HAS_OPUS = False


def _validate_split_kv_prefill_layout(
    q: torch.Tensor,
    swa_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    kv: torch.Tensor,
    *,
    swa_pages: int,
    compressed_kv_scales: torch.Tensor | None,
    compressed_kv_layout: str,
) -> tuple[
    torch.Tensor,
    int,
    int,
    int,
    int,
    torch.Tensor,
    int,
    torch.Tensor,
    bool,
    bool,
    int,
    int,
    int,
    int,
]:
    if q.dim() != 3:
        raise RuntimeError(
            f"sparse_attn_v4_paged_prefill_split_kv expects 3-D q, got {q.dim()}-D"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"sparse_attn_v4_paged_prefill_split_kv expects fp16/bf16 q, got {q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")

    _, _, D = q.shape
    if swa_kv.dim() == 3:
        if swa_kv.shape[-1] != D or swa_kv.stride(-1) != 1:
            raise RuntimeError(
                f"paged swa_kv must have dense D={D} rows, got "
                f"shape={tuple(swa_kv.shape)}, stride={swa_kv.stride()}"
            )
        swa_block_size = int(swa_kv.shape[1])
        swa_block_stride = int(swa_kv.stride(0))
        swa_pos_stride = int(swa_kv.stride(1))
        swa_stride_d = int(swa_kv.stride(2))
        swa_storage = swa_kv
        swa_storage_pages = int(swa_kv.shape[0] * swa_kv.shape[1])
    elif swa_kv.dim() == 2:
        if swa_kv.shape[1] != D or swa_kv.stride(1) != 1:
            raise RuntimeError(
                f"flat swa_kv must be [pages, {D}] with dense rows, got "
                f"shape={tuple(swa_kv.shape)}, stride={swa_kv.stride()}"
            )
        swa_block_size = 1
        swa_block_stride = int(swa_kv.stride(0))
        swa_pos_stride = int(swa_kv.stride(0))
        swa_stride_d = int(swa_kv.stride(1))
        swa_storage = swa_kv
        swa_storage_pages = int(swa_kv.shape[0])
    else:
        raise RuntimeError(f"swa_kv must be 2-D or 3-D, got {swa_kv.dim()}-D")
    packed_tail = compressed_kv_layout == _PACKED_FP8_DS_MLA
    if compressed_kv_layout not in ("dense", _PACKED_FP8_DS_MLA):
        raise RuntimeError(f"Unsupported compressed_kv_layout={compressed_kv_layout!r}")
    if packed_tail:
        if D != 512:
            raise RuntimeError(f"packed fp8_ds_mla expects D=512, got {D}")
        if compressed_kv_scales is not None:
            raise RuntimeError("packed fp8_ds_mla tail has embedded UE8M0 scales")
        if compressed_kv.dtype != torch.uint8:
            raise RuntimeError(
                f"packed fp8_ds_mla tail expects uint8, got {compressed_kv.dtype}"
            )
        if compressed_kv.dim() != 3 or compressed_kv.shape[-1] != 584:
            raise RuntimeError(
                "packed fp8_ds_mla tail expects [num_blocks, k_per_block, 584], "
                f"got {tuple(compressed_kv.shape)}"
            )
        compressed_storage = compressed_kv
        compressed_pages = compressed_kv.shape[0] * compressed_kv.shape[1]
    else:
        if compressed_kv.dim() == 3:
            if compressed_kv.shape[-1] != D or compressed_kv.stride(-1) != 1:
                raise RuntimeError(
                    f"dense compressed_kv must have dense D={D} rows, got "
                    f"shape={tuple(compressed_kv.shape)}, "
                    f"stride={compressed_kv.stride()}"
                )
            compressed_storage = compressed_kv
            compressed_pages = compressed_kv.shape[0] * compressed_kv.shape[1]
        elif compressed_kv.dim() == 2:
            if compressed_kv.shape[1] != D or compressed_kv.stride(1) != 1:
                raise RuntimeError(
                    f"flat compressed_kv must be [pages, {D}] with dense rows, "
                    f"got shape={tuple(compressed_kv.shape)}, "
                    f"stride={compressed_kv.stride()}"
                )
            compressed_storage = compressed_kv
            compressed_pages = compressed_kv.shape[0]
        else:
            raise RuntimeError(
                f"dense compressed_kv must be 2-D or 3-D, got {compressed_kv.dim()}-D"
            )
    if compressed_storage.dim() == 3:
        tail_block_size = int(compressed_storage.shape[1])
        tail_block_stride = int(compressed_storage.stride(0))
        tail_pos_stride = int(compressed_storage.stride(1))
        tail_stride_d = int(compressed_storage.stride(2))
    else:
        tail_block_size = 1
        tail_block_stride = int(compressed_storage.stride(0))
        tail_pos_stride = int(compressed_storage.stride(0))
        tail_stride_d = int(compressed_storage.stride(1))
    if swa_pages <= 0 or swa_storage_pages != swa_pages:
        raise RuntimeError(
            f"Invalid split KV SWA geometry: swa_pages={swa_pages}, "
            f"swa_storage_pages={swa_storage_pages}"
        )
    if swa_storage.dtype != q.dtype:
        raise RuntimeError(
            f"swa_kv dtype {swa_storage.dtype} does not match q dtype {q.dtype}"
        )
    quant_tail = compressed_kv_scales is not None
    if quant_tail:
        if compressed_storage.dtype != _FP8_DTYPE:
            raise RuntimeError(
                "compressed_kv_scales supplied but compressed_kv is "
                f"{compressed_storage.dtype}, expected {_FP8_DTYPE}"
            )
        if compressed_kv_scales.dtype != torch.float32:
            raise RuntimeError(
                f"compressed_kv_scales must be fp32, got {compressed_kv_scales.dtype}"
            )
        if D % _FP8_GROUP_SIZE != 0:
            raise RuntimeError(f"D={D} must be divisible by {_FP8_GROUP_SIZE}")
        tail_scales = compressed_kv_scales.reshape(-1, D // _FP8_GROUP_SIZE)
        if tail_scales.shape[0] != compressed_pages:
            raise RuntimeError(
                f"compressed_kv_scales pages={tail_scales.shape[0]} does not "
                f"match compressed pages={compressed_pages}"
            )
        if tail_scales.stride(-1) != 1:
            tail_scales = tail_scales.contiguous()
    else:
        if not packed_tail and compressed_storage.dtype != q.dtype:
            raise RuntimeError(
                "compressed_kv dtype must match q when scales are absent: "
                f"compressed={compressed_storage.dtype}, q={q.dtype}"
            )
        tail_scales = q.new_empty(1, dtype=torch.float32)

    return (
        swa_storage,
        swa_block_size,
        swa_block_stride,
        swa_pos_stride,
        swa_stride_d,
        compressed_storage,
        compressed_pages,
        tail_scales,
        quant_tail,
        packed_tail,
        tail_block_size,
        tail_block_stride,
        tail_pos_stride,
        tail_stride_d,
    )


@triton.jit
def _load_prefill_prefix_slot(
    kv_indices_prefix_ptr,
    p_start,
    k_pos,
    in_range,
):
    return tl.load(
        kv_indices_prefix_ptr + p_start + k_pos,
        mask=in_range,
        other=-1,
    )


@triton.jit
def _sparse_attn_v4_paged_prefill_kernel(
    q_ptr,  # [N, H, D]
    unified_kv_ptr,  # [total_pages, D]   — prefix source
    kv_indices_prefix_ptr,  # [total_prefix_indices] int32
    kv_indptr_prefix_ptr,  # [N+1] int32
    kv_ptr,  # [total_tokens, D]    — extend source
    kv_indices_extend_ptr,  # [total_extend_indices] int32
    kv_indptr_extend_ptr,  # [N+1] int32
    attn_sink_ptr,  # [H]
    out_ptr,  # [N, H, D]
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    pkv_stride_n: tl.constexpr,  # unified_kv stride 0 (= D usually)
    pkv_stride_d: tl.constexpr,  # unified_kv stride 1 (= 1 usually)
    ekv_stride_n: tl.constexpr,  # kv stride 0
    ekv_stride_d: tl.constexpr,  # kv stride 1
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PREFIX_BLOCK_K: tl.constexpr,
    CHECK_NEG_ONE_SENTINEL: tl.constexpr,
    HAS_PREFIX: tl.constexpr,
    FULL_D: tl.constexpr,
):
    # int64 avoids overflow in per-token address offsets for long prefills.
    t = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q_ptrs = (
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d
    )
    if FULL_D:
        q = tl.load(q_ptrs, mask=h_mask[:, None], other=0.0)
    else:
        q = tl.load(
            q_ptrs,
            mask=h_mask[:, None] & d_mask[None, :],
            other=0.0,
        )

    neg_large = -3.4028234663852886e38
    qk_scale = softmax_scale * 1.4426950408889634
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    p_k_offs = tl.arange(0, PREFIX_BLOCK_K)

    # ===== Region 1: prefix from unified_kv =====
    if HAS_PREFIX:
        p_start = tl.load(kv_indptr_prefix_ptr + t)
        p_end = tl.load(kv_indptr_prefix_ptr + t + 1)
        p_len = p_end - p_start

        for k_start in tl.range(0, p_len, PREFIX_BLOCK_K):
            k_pos = k_start + p_k_offs
            in_range = k_pos < p_len
            slot = tl.load(
                kv_indices_prefix_ptr + p_start + k_pos,
                mask=in_range,
                other=-1,
            )
            valid = in_range & (slot >= 0) if CHECK_NEG_ONE_SENTINEL else in_range

            kv_ptrs = (
                unified_kv_ptr
                + slot[:, None] * pkv_stride_n
                + d_offs[None, :] * pkv_stride_d
            )
            if FULL_D:
                kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0)
            else:
                kv = tl.load(
                    kv_ptrs,
                    mask=valid[:, None] & d_mask[None, :],
                    other=0.0,
                )

            scores = tl.dot(q, tl.trans(kv)) * qk_scale
            scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

            m_block = tl.max(scores, axis=1)
            m_new = tl.maximum(m_i, m_block)
            alpha = tl.exp2(m_i - m_new)
            p = tl.exp2(scores - m_new[:, None])
            p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
            l_new = l_i * alpha + tl.sum(p, axis=1)

            acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
            m_i = m_new
            l_i = l_new

    # ===== Region 2: extend from kv (per-fwd flat) =====
    e_start = tl.load(kv_indptr_extend_ptr + t)
    e_end = tl.load(kv_indptr_extend_ptr + t + 1)
    e_len = e_end - e_start

    for k_start in tl.range(0, e_len, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < e_len
        slot = tl.load(
            kv_indices_extend_ptr + e_start + k_pos,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (slot >= 0) if CHECK_NEG_ONE_SENTINEL else in_range

        kv_ptrs = kv_ptr + slot[:, None] * ekv_stride_n + d_offs[None, :] * ekv_stride_d
        if FULL_D:
            kv = tl.load(kv_ptrs, mask=valid[:, None], other=0.0)
        else:
            kv = tl.load(
                kv_ptrs,
                mask=valid[:, None] & d_mask[None, :],
                other=0.0,
            )

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    # ===== Sink finalization =====
    # Online softmax + sink integration: sink is a virtual extra K with V=0,
    # contributing only to the denominator. After main loops, (m_i, l_i, acc)
    # are in m_i frame; sink may shift max to m_final = max(m_i, sink), so
    # rescale BOTH l_i (for denom) AND acc (for numerator) by alpha to switch
    # to m_final frame. The sink itself adds exp(sink - m_final) to l_final
    # but contributes 0 to acc since V_sink = 0.
    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    sink = sink * 1.4426950408889634
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp2(m_i - m_final)
    l_final = l_i * alpha + tl.exp2(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(
        l_final[:, None] > 0.0,
        (acc * alpha[:, None]) / denom[:, None],
        0.0,
    )
    out_ptrs = (
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d
    )
    if FULL_D:
        tl.store(out_ptrs, out, mask=h_mask[:, None])
    else:
        tl.store(
            out_ptrs,
            out,
            mask=h_mask[:, None] & d_mask[None, :],
        )


@triton.jit
def _sparse_attn_v4_paged_prefill_csa_kernel(
    q_ptr,
    unified_kv_ptr,
    kv_indices_prefix_ptr,
    kv_indptr_prefix_ptr,
    kv_ptr,
    kv_indices_extend_ptr,
    kv_indptr_extend_ptr,
    attn_sink_ptr,
    out_ptr,
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    pkv_stride_n: tl.constexpr,
    pkv_stride_d: tl.constexpr,
    ekv_stride_n: tl.constexpr,
    ekv_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    H: tl.constexpr,
    softmax_scale: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PREFIX_BLOCK_K: tl.constexpr,
):
    """Specialized long-prefix CSA path for the V4-Pro D=512 shape."""
    t = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    qk_scale = softmax_scale * 1.4426950408889634
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    p_k_offs = tl.arange(0, PREFIX_BLOCK_K)
    p_start = tl.load(kv_indptr_prefix_ptr + t)
    p_end = tl.load(kv_indptr_prefix_ptr + t + 1)
    p_len = p_end - p_start

    for k_start in tl.range(0, p_len, PREFIX_BLOCK_K):
        k_pos = k_start + p_k_offs
        valid = k_pos < p_len
        slot = tl.load(
            kv_indices_prefix_ptr + p_start + k_pos,
            mask=valid,
            other=0,
        )
        kv = tl.load(
            unified_kv_ptr
            + slot[:, None] * pkv_stride_n
            + d_offs[None, :] * pkv_stride_d,
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    k_offs = tl.arange(0, BLOCK_K)
    e_start = tl.load(kv_indptr_extend_ptr + t)
    e_end = tl.load(kv_indptr_extend_ptr + t + 1)
    e_len = e_end - e_start

    for k_start in tl.range(0, e_len, BLOCK_K):
        k_pos = k_start + k_offs
        valid = k_pos < e_len
        slot = tl.load(
            kv_indices_extend_ptr + e_start + k_pos,
            mask=valid,
            other=0,
        )
        kv = tl.load(
            kv_ptr + slot[:, None] * ekv_stride_n + d_offs[None, :] * ekv_stride_d,
            mask=valid[:, None],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * qk_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)
        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new

    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    sink = sink * 1.4426950408889634
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp2(m_i - m_final)
    l_final = l_i * alpha + tl.exp2(sink - m_final)
    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(
        l_final[:, None] > 0.0,
        (acc * alpha[:, None]) / denom[:, None],
        0.0,
    )
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out,
        mask=h_mask[:, None],
    )


@triton.jit
def _sparse_attn_v4_paged_prefill_split_kv_kernel(
    q_ptr,  # [N, H, D]
    swa_kv_ptr,  # [blocks, block_size, D] or flat [swa_pages, D]
    compressed_kv_ptr,  # dense tail or packed [blocks, slots, 584] uint8
    compressed_scales_ptr,  # [tail_pages, NUM_GROUPS] fp32 when QUANT_TAIL
    kv_indices_prefix_ptr,  # [total_prefix_indices] int32, unified slot ids
    kv_indptr_prefix_ptr,  # [N+1] int32
    kv_ptr,  # [total_tokens, D] — extend source
    kv_indices_extend_ptr,  # [total_extend_indices] int32
    kv_indptr_extend_ptr,  # [N+1] int32
    attn_sink_ptr,  # [H]
    out_ptr,  # [N, H, D]
    q_stride_t: tl.constexpr,
    q_stride_h: tl.constexpr,
    q_stride_d: tl.constexpr,
    swa_block_stride,
    swa_pos_stride,
    swa_stride_d,
    tail_pos_stride,
    tail_stride_d,
    tail_block_stride,
    ts_stride_n,
    ekv_stride_n: tl.constexpr,
    ekv_stride_d: tl.constexpr,
    out_stride_t: tl.constexpr,
    out_stride_h: tl.constexpr,
    out_stride_d: tl.constexpr,
    total_pages: tl.constexpr,
    swa_pages: tl.constexpr,
    tail_pages: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    softmax_scale: tl.constexpr,
    SWA_BLOCK_SIZE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QUANT_TAIL: tl.constexpr,
    PACKED_TAIL: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    PACKED_BLOCK_SIZE: tl.constexpr,
):
    t = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offs = tl.arange(0, BLOCK_D)
    h_mask = h_offs < H
    d_mask = d_offs < D

    q = tl.load(
        q_ptr
        + t * q_stride_t
        + h_offs[:, None] * q_stride_h
        + d_offs[None, :] * q_stride_d,
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )

    neg_large = -3.4028234663852886e38
    m_i = tl.full((BLOCK_H,), neg_large, dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offs = tl.arange(0, BLOCK_K)
    if QUANT_TAIL or PACKED_TAIL:
        group_idx = d_offs // GROUP_SIZE

    # ===== Region 1: prefix from split SWA/compressed KV =====
    p_start = tl.load(kv_indptr_prefix_ptr + t)
    p_end = tl.load(kv_indptr_prefix_ptr + t + 1)
    p_len = p_end - p_start

    for k_start in tl.range(0, p_len, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < p_len
        slot = _load_prefill_prefix_slot(
            kv_indices_prefix_ptr,
            p_start,
            k_pos,
            in_range,
        )
        slot_valid = in_range & (slot >= 0) & (slot < total_pages)
        is_swa = slot < swa_pages

        safe_swa_slot = tl.minimum(tl.maximum(slot, 0), swa_pages - 1)
        tail_slot = slot - swa_pages
        safe_tail_slot = tl.minimum(tl.maximum(tail_slot, 0), tail_pages - 1)
        swa_valid = slot_valid & is_swa
        tail_valid = slot_valid & (~is_swa)

        swa_block = (safe_swa_slot // SWA_BLOCK_SIZE).to(tl.int64)
        swa_pos = safe_swa_slot % SWA_BLOCK_SIZE
        swa_kv = tl.load(
            swa_kv_ptr
            + swa_block[:, None] * swa_block_stride
            + swa_pos[:, None] * swa_pos_stride
            + d_offs[None, :] * swa_stride_d,
            mask=swa_valid[:, None] & d_mask[None, :],
            other=0.0,
        )
        if PACKED_TAIL:
            tail_block = (safe_tail_slot // PACKED_BLOCK_SIZE).to(tl.int64)
            tail_pos = safe_tail_slot % PACKED_BLOCK_SIZE
            packed_base = tail_block[:, None] * tail_block_stride
            token_data_base = packed_base + tail_pos[:, None] * 576
            token_scale_base = (
                packed_base + PACKED_BLOCK_SIZE * 576 + tail_pos[:, None] * 8
            )
            is_fp8_dim = d_offs < 448
            fp8_u8 = tl.load(
                compressed_kv_ptr + token_data_base + d_offs[None, :],
                mask=tail_valid[:, None] & d_mask[None, :] & is_fp8_dim[None, :],
                other=0,
            )
            fp8_val = fp8_u8.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scale = tl.full((BLOCK_K, BLOCK_D), 1.0, dtype=tl.float32)
            for g in tl.static_range(0, 7):
                encoded_g = tl.load(
                    compressed_kv_ptr + token_scale_base + g,
                    mask=tail_valid[:, None],
                    other=127,
                )
                scale_g = tl.exp2(encoded_g.to(tl.float32) - 127.0)
                group_mask = (d_offs >= g * GROUP_SIZE) & (
                    d_offs < (g + 1) * GROUP_SIZE
                )
                scale = tl.where(group_mask[None, :], scale_g, scale)
            fp8_dequant = (fp8_val * scale).to(q.dtype)
            bf16_offsets = tl.maximum(d_offs - 448, 0)
            bf16_ptr = (compressed_kv_ptr + token_data_base + 448).to(
                tl.pointer_type(tl.bfloat16)
            )
            bf16_tail = tl.load(
                bf16_ptr + bf16_offsets[None, :],
                mask=tail_valid[:, None] & d_mask[None, :] & (d_offs[None, :] >= 448),
                other=0.0,
            )
            tail_kv = tl.where(is_fp8_dim[None, :], fp8_dequant, bf16_tail)
        else:
            tail_block = (safe_tail_slot // PACKED_BLOCK_SIZE).to(tl.int64)
            tail_pos = safe_tail_slot % PACKED_BLOCK_SIZE
            tail_raw = tl.load(
                compressed_kv_ptr
                + tail_block[:, None] * tail_block_stride
                + tail_pos[:, None] * tail_pos_stride
                + d_offs[None, :] * tail_stride_d,
                mask=tail_valid[:, None] & d_mask[None, :],
                other=0.0,
            )
            if QUANT_TAIL:
                scales = tl.load(
                    compressed_scales_ptr
                    + safe_tail_slot[:, None] * ts_stride_n
                    + group_idx[None, :],
                    mask=tail_valid[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(q.dtype)
                tail_kv = tail_raw.to(q.dtype) * scales
            else:
                tail_kv = tail_raw
        prefix_kv = tl.where(is_swa[:, None], swa_kv, tail_kv)

        scores = tl.dot(q, tl.trans(prefix_kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & slot_valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & slot_valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(q.dtype), prefix_kv)
        m_i = m_new
        l_i = l_new

    # ===== Region 2: extend from kv (per-fwd flat) =====
    e_start = tl.load(kv_indptr_extend_ptr + t)
    e_end = tl.load(kv_indptr_extend_ptr + t + 1)
    e_len = e_end - e_start

    for k_start in tl.range(0, e_len, BLOCK_K):
        k_pos = k_start + k_offs
        in_range = k_pos < e_len
        slot = tl.load(
            kv_indices_extend_ptr + e_start + k_pos,
            mask=in_range,
            other=-1,
        )
        valid = in_range & (slot >= 0)

        kv = tl.load(
            kv_ptr + slot[:, None] * ekv_stride_n + d_offs[None, :] * ekv_stride_d,
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * softmax_scale
        scores = tl.where(h_mask[:, None] & valid[None, :], scores, neg_large)

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    sink = tl.load(attn_sink_ptr + h_offs, mask=h_mask, other=neg_large).to(tl.float32)
    m_final = tl.maximum(m_i, sink)
    alpha = tl.exp(m_i - m_final)
    l_final = l_i * alpha + tl.exp(sink - m_final)

    denom = tl.maximum(l_final, 1.0e-30)
    out = tl.where(l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0)
    tl.store(
        out_ptr
        + t * out_stride_t
        + h_offs[:, None] * out_stride_h
        + d_offs[None, :] * out_stride_d,
        out,
        mask=h_mask[:, None] & d_mask[None, :],
    )


def _sparse_attn_v4_paged_prefill_triton(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if not q.is_cuda:
        raise RuntimeError(
            "Triton sparse_attn_v4_paged_prefill requires CUDA/HIP tensors"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(
            f"sparse_attn_v4_paged_prefill expects fp16/bf16 q, got {q.dtype}"
        )
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: kv={unified_kv.dtype}, q={q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")
    if unified_kv.size(-1) != kv.size(-1):
        raise RuntimeError(
            f"head_dim mismatch: unified_kv={unified_kv.size(-1)}, kv={kv.size(-1)}"
        )

    T, H, D = q.shape
    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise RuntimeError(
            f"out shape/dtype/device mismatch: got shape={tuple(out.shape)} "
            f"dtype={out.dtype} device={out.device}, expected "
            f"shape={tuple(q.shape)} dtype={q.dtype} device={q.device}"
        )
    kv_indices_prefix = kv_indices_prefix.to(torch.int32).contiguous()
    kv_indptr_prefix = kv_indptr_prefix.to(torch.int32).contiguous()
    kv_indices_extend = kv_indices_extend.to(torch.int32).contiguous()
    kv_indptr_extend = kv_indptr_extend.to(torch.int32).contiguous()

    block_h = _BLOCK_H
    block_d = triton.next_power_of_2(D)
    block_k = _BLOCK_K
    full_d = block_d == D
    avg_prefix_len = kv_indices_prefix.numel() / max(T, 1)
    prefix_block_k = min(block_k, 16) if 0 < avg_prefix_len <= 16 else block_k
    check_neg_one_sentinel = not (0 < avg_prefix_len <= 16)
    has_prefix = kv_indices_prefix.numel() > 0
    use_csa_fast_kernel = has_prefix and avg_prefix_len > 16 and D == 512 and full_d
    if use_csa_fast_kernel:
        _sparse_attn_v4_paged_prefill_csa_kernel[(T, triton.cdiv(H, block_h))](
            q,
            unified_kv,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            out,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            unified_kv.stride(0),
            unified_kv.stride(1),
            kv.stride(0),
            kv.stride(1),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            H,
            float(softmax_scale),
            BLOCK_H=block_h,
            BLOCK_D=block_d,
            BLOCK_K=block_k,
            PREFIX_BLOCK_K=prefix_block_k,
            num_warps=_NUM_WARPS,
            num_stages=_NUM_STAGES,
            waves_per_eu=_WAVES_PER_EU,
        )
        return out

    _sparse_attn_v4_paged_prefill_kernel[(T, triton.cdiv(H, block_h))](
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        unified_kv.stride(0),
        unified_kv.stride(1),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        H,
        D,
        float(softmax_scale),
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        PREFIX_BLOCK_K=prefix_block_k,
        CHECK_NEG_ONE_SENTINEL=check_neg_one_sentinel,
        HAS_PREFIX=has_prefix,
        FULL_D=full_d,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
        waves_per_eu=_WAVES_PER_EU,
    )
    return out


def sparse_attn_v4_paged_prefill_split_kv(
    q: torch.Tensor,
    swa_kv: torch.Tensor,
    compressed_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    *,
    swa_pages: int,
    compressed_kv_scales: torch.Tensor | None = None,
    compressed_kv_layout: str = "dense",
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """V4 prefill sparse attention over split prefix KV plus flat extend KV."""
    if q.dim() != 3:
        raise RuntimeError(
            f"sparse_attn_v4_paged_prefill_split_kv expects 3-D q, got {q.dim()}-D"
        )
    T, H, D = q.shape
    (
        swa_storage,
        swa_block_size,
        swa_block_stride,
        swa_pos_stride,
        swa_stride_d,
        compressed_storage,
        compressed_pages,
        tail_scales,
        quant_tail,
        packed_tail,
        tail_block_size,
        tail_block_stride,
        tail_pos_stride,
        tail_stride_d,
    ) = _validate_split_kv_prefill_layout(
        q,
        swa_kv,
        compressed_kv,
        kv,
        swa_pages=swa_pages,
        compressed_kv_scales=compressed_kv_scales,
        compressed_kv_layout=compressed_kv_layout,
    )
    if not q.is_cuda:
        raise RuntimeError(
            "Triton sparse_attn_v4_paged_prefill_split_kv requires CUDA/HIP tensors"
        )

    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise RuntimeError(
            f"out shape/dtype/device mismatch: got shape={tuple(out.shape)} "
            f"dtype={out.dtype} device={out.device}, expected "
            f"shape={tuple(q.shape)} dtype={q.dtype} device={q.device}"
        )
    if T == 0:
        return out
    kv_indices_prefix = kv_indices_prefix.to(torch.int32).contiguous()
    kv_indptr_prefix = kv_indptr_prefix.to(torch.int32).contiguous()
    kv_indices_extend = kv_indices_extend.to(torch.int32).contiguous()
    kv_indptr_extend = kv_indptr_extend.to(torch.int32).contiguous()

    block_h = 16
    block_d = triton.next_power_of_2(D)
    block_k = 16 if (D >= 256 or quant_tail or packed_tail) else 32
    _sparse_attn_v4_paged_prefill_split_kv_kernel[(T, triton.cdiv(H, block_h))](
        q,
        swa_storage,
        compressed_storage,
        tail_scales,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        swa_block_stride,
        swa_pos_stride,
        swa_stride_d,
        tail_pos_stride,
        tail_stride_d,
        tail_block_stride,
        tail_scales.stride(0),
        kv.stride(0),
        kv.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        int(swa_pages + compressed_pages),
        int(swa_pages),
        int(compressed_pages),
        H,
        D,
        float(softmax_scale),
        SWA_BLOCK_SIZE=swa_block_size,
        BLOCK_H=block_h,
        BLOCK_D=block_d,
        BLOCK_K=block_k,
        QUANT_TAIL=bool(quant_tail),
        PACKED_TAIL=bool(packed_tail),
        GROUP_SIZE=_FP8_GROUP_SIZE,
        PACKED_BLOCK_SIZE=tail_block_size,
        num_warps=8,
    )
    return out


def sparse_attn_v4_paged_prefill_reference(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Pure-torch reference via virtual-pool concatenation.

    Builds a virtual KV pool `pool = cat([unified_kv, kv])`, offsets extend
    indices by `len(unified_kv)`, then delegates to the proven
    ``sparse_attn_ragged_torch`` (joint softmax with sink — same impl used by
    the decode kernel's reference). Slow but correct — for unit tests /
    dump-bisect.
    """
    T = q.size(0)
    n_pages = unified_kv.size(0)
    pool = torch.cat([unified_kv, kv], dim=0)  # [n_pages + total_tokens, D]

    p_indptr = kv_indptr_prefix.to(torch.int64)
    e_indptr = kv_indptr_extend.to(torch.int64)
    p_spans = (p_indptr[1:] - p_indptr[:T]).clamp(min=0)
    e_spans = (e_indptr[1:] - e_indptr[:T]).clamp(min=0)
    total_spans = p_spans + e_spans
    k_dim = int(total_spans.max().item()) if T > 0 else 1
    if k_dim == 0:
        k_dim = 1

    topk_idxs = torch.full((T, k_dim), -1, device=q.device, dtype=torch.int32)
    for t in range(T):
        ps = int(p_indptr[t].item())
        pe = int(p_indptr[t + 1].item())
        es = int(e_indptr[t].item())
        ee = int(e_indptr[t + 1].item())
        p_n = pe - ps
        e_n = ee - es
        if p_n > 0:
            # prefix indices point into unified_kv, no offset; -1 stays -1
            topk_idxs[t, :p_n] = kv_indices_prefix[ps:pe].to(torch.int32)
        if e_n > 0:
            # extend indices point into kv → offset by n_pages; preserve -1
            e_idx = kv_indices_extend[es:ee].to(torch.int64)
            shifted = torch.where(
                e_idx >= 0,
                e_idx + n_pages,
                torch.full_like(e_idx, -1),
            )
            topk_idxs[t, p_n : p_n + e_n] = shifted.to(torch.int32)

    return sparse_attn_ragged_torch(q, pool, attn_sink, topk_idxs, softmax_scale)


def sparse_attn_v4_paged_prefill(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """V4 prefill sparse attention over two KV sources (paged unified_kv +
    flat per-fwd kv).

    Args:
      q:                 [T, H, D] BF16/FP16 — query.
      unified_kv:        [total_pages, D] BF16/FP16 — prefix source (paged).
      kv_indices_prefix: [total_prefix] int32 — flat per-token slot lists into
        unified_kv. -1 sentinels skipped.
      kv_indptr_prefix:  [T+1] int32 — true prefix sum.
      kv:                [total_tokens, D] BF16/FP16 — extend source (this
        fwd's input K, NOT yet in swa_kv ring).
      kv_indices_extend: [total_extend] int32 — flat per-token row idx lists
        into kv. -1 sentinels skipped.
      kv_indptr_extend:  [T+1] int32 — true prefix sum.
      attn_sink:         [H] — per-head softmax-denom bias.
      softmax_scale:     float.

    Returns:
      out: [T, H, D] same dtype as q.
    """
    # Prefer OPUS when available; fall back to Triton on import failure or a
    # runtime error from an unsupported shape/device.
    if _HAS_OPUS:
        try:
            return pa_sparse_prefill_opus(
                q,
                unified_kv,
                kv_indices_prefix,
                kv_indptr_prefix,
                kv,
                kv_indices_extend,
                kv_indptr_extend,
                attn_sink,
                softmax_scale,
                out=out,
            )
        except RuntimeError:
            pass
    return _sparse_attn_v4_paged_prefill_triton(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out=out,
    )
