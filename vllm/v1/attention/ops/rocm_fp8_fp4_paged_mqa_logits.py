# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ROCm Triton implementation of DeepGEMM's paged MQA logits API.

The public wrapper mirrors ``deep_gemm.fp8_fp4_paged_mqa_logits`` for the
layouts used by DeepSeek V4's sparse indexer:

* FP8 Q + FP8 K-cache: K values are stored first for the whole page, followed
  by one float32 scale per token.
* MXFP4 Q + MXFP4 K-cache: packed E2M1 values are stored first for the whole
  page, followed by one int32 per token containing four UE8M0 scale bytes.

DeepGEMM's CUDA scheduler metadata is accepted for API compatibility but is not
needed by this Triton implementation.
"""

from __future__ import annotations

import os

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _fp4_e2m1_to_f32(nibble):
    mag = nibble & 0x7
    val = tl.where(
        mag == 0,
        0.0,
        tl.where(
            mag == 1,
            0.5,
            tl.where(
                mag == 2,
                1.0,
                tl.where(
                    mag == 3,
                    1.5,
                    tl.where(
                        mag == 4,
                        2.0,
                        tl.where(mag == 5, 3.0, tl.where(mag == 6, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where((nibble & 0x8) != 0, -val, val)


@triton.jit
def _ue8m0_scale_from_packed_i32(scale_word, block_idx):
    shift = block_idx * 8
    encoded = (scale_word.to(tl.uint32) >> shift) & 0xFF
    return tl.exp2(encoded.to(tl.float32) - 127.0)


@triton.jit
def _ue8m0_scale_byte_from_packed_i32(scale_word, block_idx):
    shift = block_idx * 8
    return ((scale_word.to(tl.uint32) >> shift) & 0xFF).to(tl.uint8)


@triton.jit
def _fp4_logit_key(
    logits,
    global_k,
    valid_k,
    INDEX_PAD: tl.constexpr,
):
    bits = logits.to(tl.uint32, bitcast=True)
    top_mask = tl.full(bits.shape, 0x80000000, dtype=tl.uint32)
    full_mask = tl.full(bits.shape, 0xFFFFFFFF, dtype=tl.uint32)
    ordered = bits ^ tl.where((bits & top_mask) != 0, full_mask, top_mask)
    # The high bits sort float32 values in numeric order. The low bits keep
    # deterministic smaller-index tie breaking.
    safe_global_k = tl.minimum(global_k, INDEX_PAD)
    tie = (INDEX_PAD - safe_global_k).to(tl.uint64)
    key = (ordered.to(tl.uint64) << 32) | tie
    return tl.where(valid_k, key, 0)


@triton.jit
def _fp4_topk_key_to_index(key, INDEX_PAD: tl.constexpr):
    return (INDEX_PAD - (key & 0xFFFFFFFF)).to(tl.int32)


@triton.jit
def _fp8_paged_mqa_logits_kernel(
    q_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    q_stride_b,
    q_stride_n,
    q_stride_h,
    q_stride_d,
    weights_stride_row,
    weights_stride_h,
    context_stride_b,
    context_stride_n,
    block_table_stride_b,
    logits_stride_row,
    max_context_len,
    KV_CACHE_STRIDE_B: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    IS_CONTEXT_LENS_2D: tl.constexpr,
    IS_FNUZ: tl.constexpr,
):
    row = tl.program_id(0)
    logical_block = tl.program_id(1)

    batch = row // NEXT_N
    next_idx = row - batch * NEXT_N
    if IS_CONTEXT_LENS_2D:
        context_len = tl.load(
            context_lens_ptr + batch * context_stride_b + next_idx * context_stride_n
        )
        valid_limit = context_len
    else:
        context_len = tl.load(context_lens_ptr + batch * context_stride_b)
        valid_limit = context_len - NEXT_N + next_idx + 1
    valid_limit = tl.minimum(valid_limit, max_context_len)

    tile_start = logical_block * KV_BLOCK_SIZE
    if tile_start >= valid_limit:
        return

    physical_block = tl.load(
        block_table_ptr + batch * block_table_stride_b + logical_block
    )
    block_base = kv_cache_ptr + physical_block.to(tl.int64) * KV_CACHE_STRIDE_B
    scale_base = (block_base + KV_BLOCK_SIZE * HEAD_DIM).to(tl.pointer_type(tl.float32))

    h = tl.arange(0, NUM_HEADS)
    d = tl.arange(0, HEAD_DIM)
    k_offsets = tl.arange(0, BLOCK_KV)
    token_pos = k_offsets
    global_k = tile_start + k_offsets
    valid_k = (k_offsets < KV_BLOCK_SIZE) & (global_k < valid_limit)

    q = tl.load(
        q_ptr
        + batch * q_stride_b
        + next_idx * q_stride_n
        + h[:, None] * q_stride_h
        + d[None, :] * q_stride_d,
        cache_modifier=".cg",
    )

    k_u8 = tl.load(
        block_base + token_pos[None, :] * HEAD_DIM + d[:, None],
        mask=(k_offsets[None, :] < KV_BLOCK_SIZE),
        other=0,
    )
    if IS_FNUZ:
        k = k_u8.to(tl.float8e4b15, bitcast=True)
    else:
        k = k_u8.to(tl.float8e4nv, bitcast=True)

    scores = tl.dot(q, k, input_precision="ieee")
    k_scales = tl.load(scale_base + token_pos, mask=valid_k, other=0.0)
    scores = scores * k_scales[None, :]
    scores = tl.maximum(scores, 0.0)

    weights = tl.load(
        weights_ptr + row * weights_stride_row + h * weights_stride_h,
        cache_modifier=".cg",
    ).to(tl.float32)
    scores = scores * weights[:, None]
    logits = tl.sum(scores, axis=0)
    tl.store(
        logits_ptr + row * logits_stride_row + global_k,
        logits,
        mask=valid_k,
    )


@triton.jit
def _fp4_paged_mqa_logits_topk_kernel(
    q_ptr,
    q_scale_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    topk_indices_ptr,
    q_stride_b,
    q_stride_n,
    q_stride_h,
    q_stride_d,
    q_scale_stride_b,
    q_scale_stride_n,
    q_scale_stride_h,
    weights_stride_row,
    weights_stride_h,
    context_stride_b,
    context_stride_n,
    block_table_stride_b,
    topk_stride_row,
    max_context_len,
    KV_CACHE_STRIDE_B: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    TOPK: tl.constexpr,
    IS_CONTEXT_LENS_2D: tl.constexpr,
    INDEX_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    tl.assume(q_stride_b > 0)
    tl.assume(q_stride_n > 0)
    tl.assume(q_stride_h > 0)
    tl.assume(q_stride_d > 0)
    tl.assume(q_scale_stride_b > 0)
    tl.assume(q_scale_stride_n > 0)
    tl.assume(q_scale_stride_h > 0)
    tl.assume(weights_stride_row > 0)
    tl.assume(weights_stride_h > 0)
    tl.assume(block_table_stride_b > 0)
    tl.assume(topk_stride_row > 0)
    tl.assume(KV_CACHE_STRIDE_B > 0)
    tl.static_assert(BLOCK_KV >= TOPK)
    tl.static_assert(INDEX_PAD > TOPK)

    batch = row // NEXT_N
    next_idx = row - batch * NEXT_N
    if IS_CONTEXT_LENS_2D:
        context_len = tl.load(
            context_lens_ptr + batch * context_stride_b + next_idx * context_stride_n
        )
        valid_limit = context_len
    else:
        context_len = tl.load(context_lens_ptr + batch * context_stride_b)
        valid_limit = context_len - NEXT_N + next_idx + 1
    valid_limit = tl.minimum(tl.maximum(valid_limit, 0), max_context_len)

    topk_offsets = tl.arange(0, TOPK)
    if valid_limit <= TOPK:
        values = tl.where(topk_offsets < valid_limit, topk_offsets, -1)
        tl.store(topk_indices_ptr + row * topk_stride_row + topk_offsets, values)
        return

    value_dim: tl.constexpr = HEAD_DIM // 2
    scale_dim: tl.constexpr = HEAD_DIM // 32

    h = tl.arange(0, NUM_HEADS)
    d_byte = tl.arange(0, value_dim)
    scale_idx = tl.arange(0, scale_dim)
    k_offsets = tl.arange(0, BLOCK_KV)

    q_packed = tl.load(
        q_ptr
        + batch * q_stride_b
        + next_idx * q_stride_n
        + h[:, None] * q_stride_h
        + d_byte[None, :] * q_stride_d,
        cache_modifier=".cg",
    ).to(tl.uint8)
    q_scale_word = tl.load(
        q_scale_ptr
        + batch * q_scale_stride_b
        + next_idx * q_scale_stride_n
        + h * q_scale_stride_h,
        cache_modifier=".cg",
    )
    q_scale = _ue8m0_scale_byte_from_packed_i32(
        q_scale_word[:, None], scale_idx[None, :]
    )
    weights = tl.load(
        weights_ptr + row * weights_stride_row + h * weights_stride_h,
        cache_modifier=".cg",
    ).to(tl.float32)

    acc = tl.full((TOPK,), 0, dtype=tl.uint64)
    for tile_start in tl.static_range(0, INDEX_PAD, BLOCK_KV):
        global_k = tile_start + k_offsets
        logical_blocks = global_k // KV_BLOCK_SIZE
        token_pos = global_k % KV_BLOCK_SIZE
        valid_k = global_k < valid_limit

        physical_block = tl.load(
            block_table_ptr + batch * block_table_stride_b + logical_blocks,
            mask=valid_k,
            other=0,
        )
        block_base = kv_cache_ptr + physical_block.to(tl.int64) * KV_CACHE_STRIDE_B
        scale_base = (block_base + KV_BLOCK_SIZE * value_dim).to(
            tl.pointer_type(tl.int32)
        )

        k_packed = tl.load(
            block_base + token_pos[None, :] * value_dim + d_byte[:, None],
            mask=valid_k[None, :],
            other=0,
        ).to(tl.uint8)
        k_scale_word = tl.load(scale_base + token_pos, mask=valid_k, other=0)
        k_scale = _ue8m0_scale_byte_from_packed_i32(
            k_scale_word[:, None], scale_idx[None, :]
        )

        scores = tl.dot_scaled(
            q_packed,
            q_scale,
            "e2m1",
            k_packed,
            k_scale,
            "e2m1",
            lhs_k_pack=True,
            rhs_k_pack=True,
            out_dtype=tl.float32,
        )
        scores = tl.maximum(scores, 0.0)
        logits = tl.sum(scores * weights[:, None], axis=0)
        keys = _fp4_logit_key(
            logits,
            global_k,
            valid_k,
            INDEX_PAD,
        )
        tile_topk = tl.topk(keys, TOPK, dim=0)
        acc = tl.topk(tl.cat(acc, tile_topk, can_reorder=True), TOPK, dim=0)

    out = _fp4_topk_key_to_index(acc, INDEX_PAD)
    tl.store(topk_indices_ptr + row * topk_stride_row + topk_offsets, out)


@triton.jit
def _fp4_paged_mqa_logits_tile_topk_stage1_kernel(
    q_ptr,
    q_scale_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    tile_keys_ptr,
    q_stride_b,
    q_stride_n,
    q_stride_h,
    q_stride_d,
    q_scale_stride_b,
    q_scale_stride_n,
    q_scale_stride_h,
    weights_stride_row,
    weights_stride_h,
    context_stride_b,
    context_stride_n,
    block_table_stride_b,
    tile_keys_stride_row,
    tile_keys_stride_tile,
    max_context_len,
    KV_CACHE_STRIDE_B: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_BLOCKS: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    TOPK: tl.constexpr,
    IS_CONTEXT_LENS_2D: tl.constexpr,
    INDEX_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    tile_id = tl.program_id(1)
    tl.assume(q_stride_b > 0)
    tl.assume(q_stride_n > 0)
    tl.assume(q_stride_h > 0)
    tl.assume(q_stride_d > 0)
    tl.assume(q_scale_stride_b > 0)
    tl.assume(q_scale_stride_n > 0)
    tl.assume(q_scale_stride_h > 0)
    tl.assume(weights_stride_row > 0)
    tl.assume(weights_stride_h > 0)
    tl.assume(block_table_stride_b > 0)
    tl.assume(tile_keys_stride_row > 0)
    tl.assume(tile_keys_stride_tile > 0)
    tl.assume(KV_CACHE_STRIDE_B > 0)
    tl.static_assert(BLOCK_KV >= TOPK)
    tl.static_assert(INDEX_PAD > TOPK)

    batch = row // NEXT_N
    next_idx = row - batch * NEXT_N
    if IS_CONTEXT_LENS_2D:
        context_len = tl.load(
            context_lens_ptr + batch * context_stride_b + next_idx * context_stride_n
        )
        valid_limit = context_len
    else:
        context_len = tl.load(context_lens_ptr + batch * context_stride_b)
        valid_limit = context_len - NEXT_N + next_idx + 1
    valid_limit = tl.minimum(tl.maximum(valid_limit, 0), max_context_len)

    tile_start = tile_id * TILE_BLOCKS * KV_BLOCK_SIZE
    topk_offsets = tl.arange(0, TOPK)
    if tile_start >= valid_limit:
        tl.store(
            tile_keys_ptr
            + row * tile_keys_stride_row
            + tile_id * tile_keys_stride_tile
            + topk_offsets,
            tl.full((TOPK,), 0, dtype=tl.uint64),
        )
        return

    value_dim: tl.constexpr = HEAD_DIM // 2
    scale_dim: tl.constexpr = HEAD_DIM // 32

    h = tl.arange(0, NUM_HEADS)
    d_byte = tl.arange(0, value_dim)
    scale_idx = tl.arange(0, scale_dim)
    k_offsets = tl.arange(0, BLOCK_KV)
    global_k = tile_start + k_offsets
    logical_blocks = tile_id * TILE_BLOCKS + k_offsets // KV_BLOCK_SIZE
    token_pos = k_offsets % KV_BLOCK_SIZE
    valid_k = global_k < valid_limit

    physical_block = tl.load(
        block_table_ptr + batch * block_table_stride_b + logical_blocks,
        mask=valid_k,
        other=0,
    )
    block_base = kv_cache_ptr + physical_block.to(tl.int64) * KV_CACHE_STRIDE_B
    scale_base = (block_base + KV_BLOCK_SIZE * value_dim).to(tl.pointer_type(tl.int32))

    q_packed = tl.load(
        q_ptr
        + batch * q_stride_b
        + next_idx * q_stride_n
        + h[:, None] * q_stride_h
        + d_byte[None, :] * q_stride_d,
        cache_modifier=".cg",
    ).to(tl.uint8)
    q_scale_word = tl.load(
        q_scale_ptr
        + batch * q_scale_stride_b
        + next_idx * q_scale_stride_n
        + h * q_scale_stride_h,
        cache_modifier=".cg",
    )
    q_scale = _ue8m0_scale_byte_from_packed_i32(
        q_scale_word[:, None], scale_idx[None, :]
    )

    k_packed = tl.load(
        block_base + token_pos[None, :] * value_dim + d_byte[:, None],
        mask=valid_k[None, :],
        other=0,
    ).to(tl.uint8)
    k_scale_word = tl.load(scale_base + token_pos, mask=valid_k, other=0)
    k_scale = _ue8m0_scale_byte_from_packed_i32(
        k_scale_word[:, None], scale_idx[None, :]
    )

    scores = tl.dot_scaled(
        q_packed,
        q_scale,
        "e2m1",
        k_packed,
        k_scale,
        "e2m1",
        lhs_k_pack=True,
        rhs_k_pack=True,
        out_dtype=tl.float32,
    )
    scores = tl.maximum(scores, 0.0)

    weights = tl.load(
        weights_ptr + row * weights_stride_row + h * weights_stride_h,
        cache_modifier=".cg",
    ).to(tl.float32)
    logits = tl.sum(scores * weights[:, None], axis=0)
    keys = _fp4_logit_key(logits, global_k, valid_k, INDEX_PAD)
    top_keys = tl.topk(keys, TOPK, dim=0)
    tl.store(
        tile_keys_ptr
        + row * tile_keys_stride_row
        + tile_id * tile_keys_stride_tile
        + topk_offsets,
        top_keys,
    )


@triton.jit
def _fp4_paged_mqa_logits_tile_topk_merge_kernel(
    tile_keys_ptr,
    context_lens_ptr,
    topk_indices_ptr,
    tile_keys_stride_row,
    tile_keys_stride_tile,
    context_stride_b,
    context_stride_n,
    topk_stride_row,
    NUM_TILES: tl.constexpr,
    TOPK: tl.constexpr,
    MERGE_TILES: tl.constexpr,
    NEXT_N: tl.constexpr,
    IS_CONTEXT_LENS_2D: tl.constexpr,
    MAX_CONTEXT_LEN: tl.constexpr,
    INDEX_PAD: tl.constexpr,
):
    row = tl.program_id(0)
    tl.assume(tile_keys_stride_row > 0)
    tl.assume(tile_keys_stride_tile > 0)
    tl.assume(topk_stride_row > 0)
    tl.static_assert(MERGE_TILES >= 1)

    merge_width: tl.constexpr = MERGE_TILES * TOPK
    merge_offsets = tl.arange(0, merge_width)
    topk_offsets = tl.arange(0, TOPK)

    batch = row // NEXT_N
    next_idx = row - batch * NEXT_N
    if IS_CONTEXT_LENS_2D:
        context_len = tl.load(
            context_lens_ptr + batch * context_stride_b + next_idx * context_stride_n
        )
        valid_limit = context_len
    else:
        context_len = tl.load(context_lens_ptr + batch * context_stride_b)
        valid_limit = context_len - NEXT_N + next_idx + 1
    valid_limit = tl.minimum(tl.maximum(valid_limit, 0), MAX_CONTEXT_LEN)

    if valid_limit <= TOPK:
        out = tl.where(topk_offsets < valid_limit, topk_offsets, -1)
        tl.store(topk_indices_ptr + row * topk_stride_row + topk_offsets, out)
        return

    acc = tl.full((TOPK,), 0, dtype=tl.uint64)
    for tile_start in tl.static_range(0, NUM_TILES, MERGE_TILES):
        candidate = tile_start * TOPK + merge_offsets
        tile_id = candidate // TOPK
        tile_offset = candidate - tile_id * TOPK
        valid = tile_id < NUM_TILES
        keys = tl.load(
            tile_keys_ptr
            + row * tile_keys_stride_row
            + tile_id * tile_keys_stride_tile
            + tile_offset,
            mask=valid,
            other=0,
        ).to(tl.uint64, bitcast=True)
        local = tl.topk(keys, TOPK, dim=0)
        acc = tl.topk(tl.cat(acc, local, can_reorder=True), TOPK, dim=0)

    out = _fp4_topk_key_to_index(acc, INDEX_PAD)
    tl.store(topk_indices_ptr + row * topk_stride_row + topk_offsets, out)


@triton.jit
def _fp4_paged_mqa_logits_kernel(
    q_ptr,
    q_scale_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    q_stride_b,
    q_stride_n,
    q_stride_h,
    q_stride_d,
    q_scale_stride_b,
    q_scale_stride_n,
    q_scale_stride_h,
    weights_stride_row,
    weights_stride_h,
    context_stride_b,
    context_stride_n,
    block_table_stride_b,
    logits_stride_row,
    max_context_len,
    KV_CACHE_STRIDE_B: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    KV_BLOCKS_PER_PROG: tl.constexpr,
    IS_CONTEXT_LENS_2D: tl.constexpr,
    TRIVIAL_TOPK_LEN: tl.constexpr,
):
    row = tl.program_id(0)
    logical_block_group = tl.program_id(1)
    tl.assume(q_stride_b > 0)
    tl.assume(q_stride_n > 0)
    tl.assume(q_stride_h > 0)
    tl.assume(q_stride_d > 0)
    tl.assume(q_scale_stride_b > 0)
    tl.assume(q_scale_stride_n > 0)
    tl.assume(q_scale_stride_h > 0)
    tl.assume(weights_stride_row > 0)
    tl.assume(weights_stride_h > 0)
    tl.assume(block_table_stride_b > 0)
    tl.assume(logits_stride_row > 0)
    tl.assume(KV_CACHE_STRIDE_B > 0)

    batch = row // NEXT_N
    next_idx = row - batch * NEXT_N
    if IS_CONTEXT_LENS_2D:
        context_len = tl.load(
            context_lens_ptr + batch * context_stride_b + next_idx * context_stride_n
        )
        valid_limit = context_len
    else:
        context_len = tl.load(context_lens_ptr + batch * context_stride_b)
        valid_limit = context_len - NEXT_N + next_idx + 1
    valid_limit = tl.minimum(valid_limit, max_context_len)
    if TRIVIAL_TOPK_LEN > 0 and valid_limit <= TRIVIAL_TOPK_LEN:
        return

    first_logical_block = logical_block_group * KV_BLOCKS_PER_PROG
    tile_start = first_logical_block * KV_BLOCK_SIZE
    if tile_start >= valid_limit:
        return

    value_dim: tl.constexpr = HEAD_DIM // 2
    scale_dim: tl.constexpr = HEAD_DIM // 32

    h = tl.arange(0, NUM_HEADS)
    d_byte = tl.arange(0, value_dim)
    scale_idx = tl.arange(0, scale_dim)
    k_offsets = tl.arange(0, BLOCK_KV)
    global_k = tile_start + k_offsets
    logical_blocks = first_logical_block + k_offsets // KV_BLOCK_SIZE
    token_pos = k_offsets % KV_BLOCK_SIZE
    valid_k = (k_offsets < KV_BLOCK_SIZE) & (global_k < valid_limit)
    if KV_BLOCKS_PER_PROG != 1:
        valid_k = global_k < valid_limit

    physical_block = tl.load(
        block_table_ptr + batch * block_table_stride_b + logical_blocks,
        mask=valid_k,
        other=0,
    )
    block_base = kv_cache_ptr + physical_block.to(tl.int64) * KV_CACHE_STRIDE_B
    scale_base = (block_base + KV_BLOCK_SIZE * value_dim).to(tl.pointer_type(tl.int32))

    q_packed = tl.load(
        q_ptr
        + batch * q_stride_b
        + next_idx * q_stride_n
        + h[:, None] * q_stride_h
        + d_byte[None, :] * q_stride_d,
        cache_modifier=".cg",
    ).to(tl.uint8)
    q_scale_word = tl.load(
        q_scale_ptr
        + batch * q_scale_stride_b
        + next_idx * q_scale_stride_n
        + h * q_scale_stride_h,
        cache_modifier=".cg",
    )
    q_scale = _ue8m0_scale_byte_from_packed_i32(
        q_scale_word[:, None], scale_idx[None, :]
    )

    k_packed = tl.load(
        block_base + token_pos[None, :] * value_dim + d_byte[:, None],
        mask=valid_k[None, :],
        other=0,
    ).to(tl.uint8)
    k_scale_word = tl.load(scale_base + token_pos, mask=valid_k, other=0)
    k_scale = _ue8m0_scale_byte_from_packed_i32(
        k_scale_word[:, None], scale_idx[None, :]
    )

    scores = tl.dot_scaled(
        q_packed,
        q_scale,
        "e2m1",
        k_packed,
        k_scale,
        "e2m1",
        lhs_k_pack=True,
        rhs_k_pack=True,
        out_dtype=tl.float32,
    )
    scores = tl.maximum(scores, 0.0)

    weights = tl.load(
        weights_ptr + row * weights_stride_row + h * weights_stride_h,
        cache_modifier=".cg",
    ).to(tl.float32)
    scores = scores * weights[:, None]
    logits = tl.sum(scores, axis=0)
    tl.store(
        logits_ptr + row * logits_stride_row + global_k,
        logits,
        mask=valid_k,
    )


@triton.jit
def _fp4_paged_mqa_logits_segment_kernel(
    q_ptr,
    q_scale_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    q_stride_b,
    q_stride_n,
    q_stride_h,
    q_stride_d,
    q_scale_stride_b,
    q_scale_stride_n,
    q_scale_stride_h,
    weights_stride_row,
    weights_stride_h,
    context_stride_b,
    context_stride_n,
    block_table_stride_b,
    logits_stride_row,
    max_context_len,
    KV_CACHE_STRIDE_B: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    KV_BLOCKS_PER_PROG: tl.constexpr,
    NUM_LOGICAL_BLOCK_GROUPS: tl.constexpr,
    NUM_GROUP_PROGS: tl.constexpr,
    IS_CONTEXT_LENS_2D: tl.constexpr,
    TRIVIAL_TOPK_LEN: tl.constexpr,
):
    row = tl.program_id(0)
    segment_id = tl.program_id(1)
    tl.assume(q_stride_b > 0)
    tl.assume(q_stride_n > 0)
    tl.assume(q_stride_h > 0)
    tl.assume(q_stride_d > 0)
    tl.assume(q_scale_stride_b > 0)
    tl.assume(q_scale_stride_n > 0)
    tl.assume(q_scale_stride_h > 0)
    tl.assume(weights_stride_row > 0)
    tl.assume(weights_stride_h > 0)
    tl.assume(block_table_stride_b > 0)
    tl.assume(logits_stride_row > 0)
    tl.assume(KV_CACHE_STRIDE_B > 0)

    batch = row // NEXT_N
    next_idx = row - batch * NEXT_N
    if IS_CONTEXT_LENS_2D:
        context_len = tl.load(
            context_lens_ptr + batch * context_stride_b + next_idx * context_stride_n
        )
        valid_limit = context_len
    else:
        context_len = tl.load(context_lens_ptr + batch * context_stride_b)
        valid_limit = context_len - NEXT_N + next_idx + 1
    valid_limit = tl.minimum(tl.maximum(valid_limit, 0), max_context_len)
    if TRIVIAL_TOPK_LEN > 0 and valid_limit <= TRIVIAL_TOPK_LEN:
        return

    valid_logical_blocks = tl.cdiv(valid_limit, KV_BLOCK_SIZE)
    valid_block_groups = tl.cdiv(valid_logical_blocks, KV_BLOCKS_PER_PROG)
    valid_block_groups = tl.minimum(valid_block_groups, NUM_LOGICAL_BLOCK_GROUPS)
    groups_per_segment = tl.cdiv(valid_block_groups, NUM_GROUP_PROGS)
    segment_start = segment_id * groups_per_segment
    segment_end = tl.minimum(segment_start + groups_per_segment, valid_block_groups)
    if segment_start >= segment_end:
        return

    value_dim: tl.constexpr = HEAD_DIM // 2
    scale_dim: tl.constexpr = HEAD_DIM // 32

    h = tl.arange(0, NUM_HEADS)
    d_byte = tl.arange(0, value_dim)
    scale_idx = tl.arange(0, scale_dim)
    k_offsets = tl.arange(0, BLOCK_KV)

    q_packed = tl.load(
        q_ptr
        + batch * q_stride_b
        + next_idx * q_stride_n
        + h[:, None] * q_stride_h
        + d_byte[None, :] * q_stride_d,
        cache_modifier=".cg",
    ).to(tl.uint8)
    q_scale_word = tl.load(
        q_scale_ptr
        + batch * q_scale_stride_b
        + next_idx * q_scale_stride_n
        + h * q_scale_stride_h,
        cache_modifier=".cg",
    )
    q_scale = _ue8m0_scale_byte_from_packed_i32(
        q_scale_word[:, None], scale_idx[None, :]
    )
    weights = tl.load(
        weights_ptr + row * weights_stride_row + h * weights_stride_h,
        cache_modifier=".cg",
    ).to(tl.float32)

    logical_block_group = segment_start
    while logical_block_group < segment_end:
        first_logical_block = logical_block_group * KV_BLOCKS_PER_PROG
        tile_start = first_logical_block * KV_BLOCK_SIZE
        global_k = tile_start + k_offsets
        logical_blocks = first_logical_block + k_offsets // KV_BLOCK_SIZE
        token_pos = k_offsets % KV_BLOCK_SIZE
        valid_k = (k_offsets < KV_BLOCK_SIZE) & (global_k < valid_limit)
        if KV_BLOCKS_PER_PROG != 1:
            valid_k = global_k < valid_limit

        physical_block = tl.load(
            block_table_ptr + batch * block_table_stride_b + logical_blocks,
            mask=valid_k,
            other=0,
        )
        block_base = kv_cache_ptr + physical_block.to(tl.int64) * KV_CACHE_STRIDE_B
        scale_base = (block_base + KV_BLOCK_SIZE * value_dim).to(
            tl.pointer_type(tl.int32)
        )

        k_packed = tl.load(
            block_base + token_pos[None, :] * value_dim + d_byte[:, None],
            mask=valid_k[None, :],
            other=0,
        ).to(tl.uint8)
        k_scale_word = tl.load(scale_base + token_pos, mask=valid_k, other=0)
        k_scale = _ue8m0_scale_byte_from_packed_i32(
            k_scale_word[:, None], scale_idx[None, :]
        )

        scores = tl.dot_scaled(
            q_packed,
            q_scale,
            "e2m1",
            k_packed,
            k_scale,
            "e2m1",
            lhs_k_pack=True,
            rhs_k_pack=True,
            out_dtype=tl.float32,
        )
        scores = tl.maximum(scores, 0.0)
        scores = scores * weights[:, None]
        logits = tl.sum(scores, axis=0)
        tl.store(
            logits_ptr + row * logits_stride_row + global_k,
            logits,
            mask=valid_k,
        )
        logical_block_group += 1


@triton.jit
def _fp4_paged_mqa_logits_tiled_kernel(
    q_ptr,
    q_scale_ptr,
    kv_cache_ptr,
    weights_ptr,
    context_lens_ptr,
    block_table_ptr,
    logits_ptr,
    q_stride_b,
    q_stride_n,
    q_stride_h,
    q_stride_d,
    q_scale_stride_b,
    q_scale_stride_n,
    q_scale_stride_h,
    weights_stride_row,
    weights_stride_h,
    context_stride_b,
    context_stride_n,
    block_table_stride_b,
    logits_stride_row,
    max_context_len,
    KV_CACHE_STRIDE_B: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    NEXT_N: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    KV_BLOCKS_PER_PROG: tl.constexpr,
    KV_TILE_CHUNKS: tl.constexpr,
    IS_CONTEXT_LENS_2D: tl.constexpr,
    TRIVIAL_TOPK_LEN: tl.constexpr,
):
    row = tl.program_id(0)
    logical_block_group = tl.program_id(1)
    tl.assume(q_stride_b > 0)
    tl.assume(q_stride_n > 0)
    tl.assume(q_stride_h > 0)
    tl.assume(q_stride_d > 0)
    tl.assume(q_scale_stride_b > 0)
    tl.assume(q_scale_stride_n > 0)
    tl.assume(q_scale_stride_h > 0)
    tl.assume(weights_stride_row > 0)
    tl.assume(weights_stride_h > 0)
    tl.assume(block_table_stride_b > 0)
    tl.assume(logits_stride_row > 0)
    tl.assume(KV_CACHE_STRIDE_B > 0)
    tl.static_assert(KV_TILE_CHUNKS > 1)

    batch = row // NEXT_N
    next_idx = row - batch * NEXT_N
    if IS_CONTEXT_LENS_2D:
        context_len = tl.load(
            context_lens_ptr + batch * context_stride_b + next_idx * context_stride_n
        )
        valid_limit = context_len
    else:
        context_len = tl.load(context_lens_ptr + batch * context_stride_b)
        valid_limit = context_len - NEXT_N + next_idx + 1
    valid_limit = tl.minimum(valid_limit, max_context_len)
    if TRIVIAL_TOPK_LEN > 0 and valid_limit <= TRIVIAL_TOPK_LEN:
        return

    first_logical_block = logical_block_group * KV_BLOCKS_PER_PROG
    tile_start = first_logical_block * KV_BLOCK_SIZE
    if tile_start >= valid_limit:
        return

    value_dim: tl.constexpr = HEAD_DIM // 2
    scale_dim: tl.constexpr = HEAD_DIM // 32
    sub_block_kv: tl.constexpr = BLOCK_KV // KV_TILE_CHUNKS

    h = tl.arange(0, NUM_HEADS)
    d_byte = tl.arange(0, value_dim)
    scale_idx = tl.arange(0, scale_dim)
    sub_offsets = tl.arange(0, sub_block_kv)

    q_packed = tl.load(
        q_ptr
        + batch * q_stride_b
        + next_idx * q_stride_n
        + h[:, None] * q_stride_h
        + d_byte[None, :] * q_stride_d,
        cache_modifier=".cg",
    ).to(tl.uint8)
    q_scale_word = tl.load(
        q_scale_ptr
        + batch * q_scale_stride_b
        + next_idx * q_scale_stride_n
        + h * q_scale_stride_h,
        cache_modifier=".cg",
    )
    q_scale = _ue8m0_scale_byte_from_packed_i32(
        q_scale_word[:, None], scale_idx[None, :]
    )
    weights = tl.load(
        weights_ptr + row * weights_stride_row + h * weights_stride_h,
        cache_modifier=".cg",
    ).to(tl.float32)

    for chunk_i in tl.static_range(0, KV_TILE_CHUNKS):
        chunk_offset = chunk_i * sub_block_kv
        global_k = tile_start + chunk_offset + sub_offsets
        if tile_start + chunk_offset < valid_limit:
            logical_blocks = (
                first_logical_block + (chunk_offset + sub_offsets) // KV_BLOCK_SIZE
            )
            token_pos = (chunk_offset + sub_offsets) % KV_BLOCK_SIZE
            valid_k = global_k < valid_limit

            physical_block = tl.load(
                block_table_ptr + batch * block_table_stride_b + logical_blocks,
                mask=valid_k,
                other=0,
            )
            block_base = kv_cache_ptr + physical_block.to(tl.int64) * KV_CACHE_STRIDE_B
            scale_base = (block_base + KV_BLOCK_SIZE * value_dim).to(
                tl.pointer_type(tl.int32)
            )

            k_packed = tl.load(
                block_base + token_pos[None, :] * value_dim + d_byte[:, None],
                mask=valid_k[None, :],
                other=0,
            ).to(tl.uint8)
            k_scale_word = tl.load(scale_base + token_pos, mask=valid_k, other=0)
            k_scale = _ue8m0_scale_byte_from_packed_i32(
                k_scale_word[:, None], scale_idx[None, :]
            )

            scores = tl.dot_scaled(
                q_packed,
                q_scale,
                "e2m1",
                k_packed,
                k_scale,
                "e2m1",
                lhs_k_pack=True,
                rhs_k_pack=True,
                out_dtype=tl.float32,
            )
            scores = tl.maximum(scores, 0.0)
            scores = scores * weights[:, None]
            logits = tl.sum(scores, axis=0)
            tl.store(
                logits_ptr + row * logits_stride_row + global_k,
                logits,
                mask=valid_k,
            )


def rocm_get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor,
    block_size: int,
    num_sms: int,
) -> torch.Tensor:
    """Return a DeepGEMM-compatible metadata tensor for ROCm.

    The Triton kernels below schedule directly from ``context_lens`` and
    ``block_table``, so the contents are intentionally unused. Returning the
    same ``[num_sms + 1, 2]`` int32 shape keeps callers compatible with
    DeepGEMM's API and shape checks.
    """
    del block_size
    return torch.empty(
        (int(num_sms) + 1, 2), dtype=torch.int32, device=context_lens.device
    )


@triton.jit
def _fp4_mqa_logits_kernel(
    q_ptr,
    q_scale_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    cu_start_ptr,
    cu_end_ptr,
    logits_ptr,
    seq_len_kv,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    q_scale_stride_s,
    q_scale_stride_h,
    k_stride_s,
    k_stride_d,
    weights_stride_s,
    weights_stride_h,
    logits_stride_s,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    USE_DOT_SCALED: tl.constexpr,
    TRIVIAL_TOPK_LEN: tl.constexpr,
):
    row = tl.num_programs(0) - tl.program_id(0) - 1
    block_idx = tl.program_id(1)
    tl.assume(row >= 0)
    tl.assume(q_stride_s > 0)
    tl.assume(q_stride_h > 0)
    tl.assume(q_stride_d > 0)
    tl.assume(q_scale_stride_s > 0)
    tl.assume(q_scale_stride_h > 0)
    tl.assume(k_stride_s > 0)
    tl.assume(k_stride_d > 0)
    tl.assume(weights_stride_s > 0)
    tl.assume(weights_stride_h > 0)
    tl.assume(logits_stride_s > 0)

    start = tl.load(cu_start_ptr + row)
    end = tl.load(cu_end_ptr + row)
    start = tl.maximum(start, 0)
    end = tl.minimum(end, seq_len_kv)
    if TRIVIAL_TOPK_LEN > 0 and end - start <= TRIVIAL_TOPK_LEN:
        return

    tile_start = start + block_idx * BLOCK_KV
    if tile_start >= end:
        return

    k_offsets = tile_start + tl.arange(0, BLOCK_KV)
    valid_k = k_offsets < end

    h = tl.arange(0, NUM_HEADS)
    if USE_DOT_SCALED:
        value_dim: tl.constexpr = HEAD_DIM // 2
        scale_dim: tl.constexpr = HEAD_DIM // 32
        d_byte = tl.arange(0, value_dim)
        scale_idx = tl.arange(0, scale_dim)

        q_packed = tl.load(
            q_ptr
            + row * q_stride_s
            + h[:, None] * q_stride_h
            + d_byte[None, :] * q_stride_d,
            cache_modifier=".cg",
        ).to(tl.uint8)
        q_scale_word = tl.load(
            q_scale_ptr + row * q_scale_stride_s + h * q_scale_stride_h,
            cache_modifier=".cg",
        )
        q_scale = _ue8m0_scale_byte_from_packed_i32(
            q_scale_word[:, None], scale_idx[None, :]
        )

        k_packed = tl.load(
            k_ptr + k_offsets[None, :] * k_stride_s + d_byte[:, None] * k_stride_d,
            mask=valid_k[None, :],
            other=0,
        ).to(tl.uint8)
        k_scale_word = tl.load(k_scale_ptr + k_offsets, mask=valid_k, other=0)
        k_scale = _ue8m0_scale_byte_from_packed_i32(
            k_scale_word[:, None], scale_idx[None, :]
        )

        scores = tl.dot_scaled(
            q_packed,
            q_scale,
            "e2m1",
            k_packed,
            k_scale,
            "e2m1",
            lhs_k_pack=True,
            rhs_k_pack=True,
            out_dtype=tl.float32,
        )
    else:
        d = tl.arange(0, HEAD_DIM)
        d_byte = d // 2
        d_is_odd = (d & 1) != 0
        d_scale_block = d // 32

        q_packed = tl.load(
            q_ptr
            + row * q_stride_s
            + h[:, None] * q_stride_h
            + d_byte[None, :] * q_stride_d,
            cache_modifier=".cg",
        ).to(tl.uint32)
        q_nibble = tl.where(d_is_odd[None, :], q_packed >> 4, q_packed) & 0xF
        q_values = _fp4_e2m1_to_f32(q_nibble)
        q_scale_word = tl.load(
            q_scale_ptr + row * q_scale_stride_s + h * q_scale_stride_h,
            cache_modifier=".cg",
        )
        q_scale = _ue8m0_scale_from_packed_i32(
            q_scale_word[:, None], d_scale_block[None, :]
        )
        q_values = q_values * q_scale

        k_packed = tl.load(
            k_ptr + k_offsets[None, :] * k_stride_s + d_byte[:, None] * k_stride_d,
            mask=valid_k[None, :],
            other=0,
        ).to(tl.uint32)
        k_nibble = tl.where(d_is_odd[:, None], k_packed >> 4, k_packed) & 0xF
        k_values = _fp4_e2m1_to_f32(k_nibble)
        k_scale_word = tl.load(k_scale_ptr + k_offsets, mask=valid_k, other=0)
        k_scale = _ue8m0_scale_from_packed_i32(
            k_scale_word[None, :], d_scale_block[:, None]
        )
        k_values = k_values * k_scale
        scores = tl.dot(q_values, k_values, input_precision="ieee")
    scores = tl.maximum(scores, 0.0)

    weights = tl.load(
        weights_ptr + row * weights_stride_s + h * weights_stride_h,
        cache_modifier=".cg",
    ).to(tl.float32)
    logits = tl.sum(scores * weights[:, None], axis=0)
    tl.store(
        logits_ptr + row * logits_stride_s + k_offsets,
        logits,
        mask=valid_k,
    )


@triton.jit
def _fp4_mqa_logits_persistent_kernel(
    q_ptr,
    q_scale_ptr,
    k_ptr,
    k_scale_ptr,
    weights_ptr,
    cu_start_ptr,
    cu_end_ptr,
    logits_ptr,
    seq_len_kv,
    q_stride_s,
    q_stride_h,
    q_stride_d,
    q_scale_stride_s,
    q_scale_stride_h,
    k_stride_s,
    k_stride_d,
    weights_stride_s,
    weights_stride_h,
    logits_stride_s,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    NUM_K_BLOCKS: tl.constexpr,
    NUM_K_PROGS: tl.constexpr,
    TRIVIAL_TOPK_LEN: tl.constexpr,
):
    row = tl.num_programs(0) - tl.program_id(0) - 1
    block_idx = tl.program_id(1)
    tl.assume(row >= 0)
    tl.assume(q_stride_s > 0)
    tl.assume(q_stride_h > 0)
    tl.assume(q_stride_d > 0)
    tl.assume(q_scale_stride_s > 0)
    tl.assume(q_scale_stride_h > 0)
    tl.assume(k_stride_s > 0)
    tl.assume(k_stride_d > 0)
    tl.assume(weights_stride_s > 0)
    tl.assume(weights_stride_h > 0)
    tl.assume(logits_stride_s > 0)

    start = tl.load(cu_start_ptr + row)
    end = tl.load(cu_end_ptr + row)
    start = tl.maximum(start, 0)
    end = tl.minimum(end, seq_len_kv)
    if TRIVIAL_TOPK_LEN > 0 and end - start <= TRIVIAL_TOPK_LEN:
        return

    first_tile_start = start + block_idx * BLOCK_KV
    if first_tile_start >= end:
        return

    value_dim: tl.constexpr = HEAD_DIM // 2
    scale_dim: tl.constexpr = HEAD_DIM // 32

    h = tl.arange(0, NUM_HEADS)
    d_byte = tl.arange(0, value_dim)
    scale_idx = tl.arange(0, scale_dim)
    block_offsets = tl.arange(0, BLOCK_KV)

    q_packed = tl.load(
        q_ptr
        + row * q_stride_s
        + h[:, None] * q_stride_h
        + d_byte[None, :] * q_stride_d,
        cache_modifier=".cg",
    ).to(tl.uint8)
    q_scale_word = tl.load(
        q_scale_ptr + row * q_scale_stride_s + h * q_scale_stride_h,
        cache_modifier=".cg",
    )
    q_scale = _ue8m0_scale_byte_from_packed_i32(
        q_scale_word[:, None], scale_idx[None, :]
    )
    weights = tl.load(
        weights_ptr + row * weights_stride_s + h * weights_stride_h,
        cache_modifier=".cg",
    ).to(tl.float32)

    while block_idx < NUM_K_BLOCKS:
        tile_start = start + block_idx * BLOCK_KV
        if tile_start < end:
            k_offsets = tile_start + block_offsets
            valid_k = k_offsets < end

            k_packed = tl.load(
                k_ptr + k_offsets[None, :] * k_stride_s + d_byte[:, None] * k_stride_d,
                mask=valid_k[None, :],
                other=0,
            ).to(tl.uint8)
            k_scale_word = tl.load(k_scale_ptr + k_offsets, mask=valid_k, other=0)
            k_scale = _ue8m0_scale_byte_from_packed_i32(
                k_scale_word[:, None], scale_idx[None, :]
            )

            scores = tl.dot_scaled(
                q_packed,
                q_scale,
                "e2m1",
                k_packed,
                k_scale,
                "e2m1",
                lhs_k_pack=True,
                rhs_k_pack=True,
                out_dtype=tl.float32,
            )
            scores = tl.maximum(scores, 0.0)
            logits = tl.sum(scores * weights[:, None], axis=0)
            tl.store(
                logits_ptr + row * logits_stride_s + k_offsets,
                logits,
                mask=valid_k,
            )
        block_idx += NUM_K_PROGS


def _fp8_is_fnuz() -> bool:
    fnuz_dtype = getattr(torch, "float8_e4m3fnuz", None)
    return fnuz_dtype is not None and current_platform.fp8_dtype() == fnuz_dtype


def _block_kv_for_dot(block_size: int) -> int:
    return max(16, triton.next_power_of_2(block_size))


def _block_kv_for_streaming_topk(block_size: int, topk_tokens: int) -> int:
    # Keep the dot tile large enough for tl.topk but cap it at the largest
    # production tile already exercised by the paged FP4 logits kernel.
    min_tile = max(block_size, topk_tokens)
    return min(2048, triton.next_power_of_2(min_tile))


def _num_logical_blocks_for_launch(
    max_context_len: int,
    block_size: int,
    block_tables: torch.Tensor,
) -> int:
    max_blocks_from_context = triton.cdiv(max_context_len, block_size)
    return min(max_blocks_from_context, int(block_tables.shape[1]))


def _fp4_segment_group_progs() -> int:
    env_value = os.getenv("VLLM_ROCM_FP4_INDEXER_SEGMENT_GROUP_PROGS", "0").strip()
    if not env_value:
        return 0
    try:
        return max(0, int(env_value))
    except ValueError:
        return 0


def _fp4_tile_chunks() -> int:
    env_value = os.getenv("VLLM_ROCM_FP4_INDEXER_TILE_CHUNKS", "0").strip()
    if not env_value:
        return 0
    try:
        value = int(env_value)
    except ValueError:
        return 0
    return value if value in (2, 4) else 0


def _fp4_prefill_persistent_k_progs() -> int:
    env_value = os.getenv(
        "VLLM_ROCM_FP4_INDEXER_PREFILL_PERSISTENT_K_PROGS", "0"
    ).strip()
    if not env_value:
        return 0
    try:
        value = int(env_value)
    except ValueError:
        return 0
    return value if value in (1, 2, 4, 8) else 0


def rocm_fp4_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool = False,
    trivial_topk_len: int = 0,
) -> torch.Tensor:
    """Compute non-paged MXFP4 MQA logits for ROCm sparse-indexer prefill."""
    q_values, q_scales = q
    k_values, k_scales = kv
    seq_len, num_heads, packed_dim = q_values.shape
    seq_len_kv = k_values.shape[0]
    head_dim = packed_dim * 2
    assert head_dim == 128, f"MXFP4 MQA logits expects head_dim=128, got {head_dim}"
    assert q_scales.shape == (seq_len, num_heads)
    assert k_values.shape == (seq_len_kv, packed_dim)
    assert k_scales.shape == (seq_len_kv,)
    assert q_scales.dtype == torch.int32
    assert k_scales.dtype == torch.int32
    assert weights.shape == (seq_len, num_heads)
    assert weights.dtype == torch.float32

    logits_shape = (seq_len, seq_len_kv)
    if clean_logits:
        logits = torch.full(
            logits_shape,
            float("-inf"),
            dtype=torch.float32,
            device=q_values.device,
        )
    else:
        logits = torch.empty(
            logits_shape,
            dtype=torch.float32,
            device=q_values.device,
        )

    valid_lens = (cu_seqlen_ke - cu_seqlen_ks).clamp(min=0)
    max_valid_len = int(valid_lens.max().item()) if seq_len > 0 else 0
    if max_valid_len == 0 or seq_len_kv == 0:
        return logits

    if clean_logits:
        trivial_topk_len = 0
    if trivial_topk_len > 0 and max_valid_len <= trivial_topk_len:
        return logits
    block_kv = 128
    num_k_blocks = triton.cdiv(max_valid_len, block_kv)
    persistent_k_progs = 0 if clean_logits else _fp4_prefill_persistent_k_progs()
    if persistent_k_progs > 0 and num_heads % 32 == 0:
        num_k_progs = min(persistent_k_progs, num_k_blocks)
        grid = (seq_len, num_k_progs)
        _fp4_mqa_logits_persistent_kernel[grid](
            q_values,
            q_scales,
            k_values,
            k_scales,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            logits,
            seq_len_kv,
            q_values.stride(0),
            q_values.stride(1),
            q_values.stride(2),
            q_scales.stride(0),
            q_scales.stride(1),
            k_values.stride(0),
            k_values.stride(1),
            weights.stride(0),
            weights.stride(1),
            logits.stride(0),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            BLOCK_KV=block_kv,
            NUM_K_BLOCKS=num_k_blocks,
            NUM_K_PROGS=num_k_progs,
            TRIVIAL_TOPK_LEN=int(trivial_topk_len),
            num_warps=4,
        )
    else:
        grid = (seq_len, num_k_blocks)
        _fp4_mqa_logits_kernel[grid](
            q_values,
            q_scales,
            k_values,
            k_scales,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            logits,
            seq_len_kv,
            q_values.stride(0),
            q_values.stride(1),
            q_values.stride(2),
            q_scales.stride(0),
            q_scales.stride(1),
            k_values.stride(0),
            k_values.stride(1),
            weights.stride(0),
            weights.stride(1),
            logits.stride(0),
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            BLOCK_KV=block_kv,
            USE_DOT_SCALED=num_heads % 32 == 0,
            TRIVIAL_TOPK_LEN=int(trivial_topk_len),
            num_warps=4,
        )
    return logits


def rocm_fp4_paged_mqa_logits_topk(
    q: tuple[torch.Tensor, torch.Tensor],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    topk_indices: torch.Tensor,
    max_context_len: int,
    topk_tokens: int,
) -> torch.Tensor:
    """Compute exact MXFP4 paged MQA top-k indices without materialized logits.

    This ROCm-only experimental path keeps the sparse indexer API unchanged:
    callers provide the existing output buffer and receive the same tensor.
    """
    q_values, q_scales = q
    batch_size, next_n, num_heads, packed_dim = q_values.shape
    head_dim = packed_dim * 2
    assert head_dim == 128, f"MXFP4 top-k expects head_dim=128, got {head_dim}"
    assert num_heads in (32, 64), f"unsupported num_heads={num_heads}"
    assert q_scales.dtype == torch.int32
    assert q_scales.shape == (batch_size, next_n, num_heads)
    assert kv_cache.dtype == torch.uint8
    assert weights.shape == (batch_size * next_n, num_heads)
    assert weights.dtype == torch.float32
    assert context_lens.dtype == torch.int32
    assert context_lens.dim() in (1, 2)
    assert block_tables.dtype == torch.int32
    assert topk_indices.dtype == torch.int32
    assert topk_indices.shape[0] >= batch_size * next_n
    assert topk_indices.shape[1] >= topk_tokens

    block_size = int(kv_cache.shape[1])
    expected_cache_width = head_dim // 2 + 4
    assert kv_cache.dim() == 4 and kv_cache.shape[2] == 1
    assert kv_cache.shape[3] == expected_cache_width

    if max_context_len == 0 or topk_tokens == 0:
        return topk_indices
    if topk_tokens > 256:
        raise ValueError(
            "streaming FP4 top-k is validation-only for top-k <= 256; "
            "use the chunked fused helper for production top-k sizes"
        )
    block_kv = _block_kv_for_streaming_topk(block_size, topk_tokens)
    if block_kv < topk_tokens:
        raise ValueError(
            f"streaming FP4 top-k tile {block_kv} is smaller than top-k {topk_tokens}"
        )
    if max_context_len >= 2**31:
        raise ValueError(f"max_context_len is too large: {max_context_len}")

    is_context_lens_2d = context_lens.dim() == 2
    grid = (batch_size * next_n,)
    _fp4_paged_mqa_logits_topk_kernel[grid](
        q_values,
        q_scales,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        topk_indices,
        q_values.stride(0),
        q_values.stride(1),
        q_values.stride(2),
        q_values.stride(3),
        q_scales.stride(0),
        q_scales.stride(1),
        q_scales.stride(2),
        weights.stride(0),
        weights.stride(1),
        context_lens.stride(0),
        context_lens.stride(1) if is_context_lens_2d else 0,
        block_tables.stride(0),
        topk_indices.stride(0),
        max_context_len,
        KV_CACHE_STRIDE_B=kv_cache.stride(0),
        KV_BLOCK_SIZE=block_size,
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        BLOCK_KV=block_kv,
        TOPK=int(topk_tokens),
        IS_CONTEXT_LENS_2D=is_context_lens_2d,
        INDEX_PAD=int(max_context_len) + 1,
        num_warps=8,
    )
    return topk_indices


def rocm_fp4_paged_mqa_logits_tile_topk(
    q: tuple[torch.Tensor, torch.Tensor],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    topk_indices: torch.Tensor,
    max_context_len: int,
    topk_tokens: int,
) -> torch.Tensor:
    """Compute exact MXFP4 paged MQA top-k through tile-local fused logits.

    Stage 1 keeps only tile-local top-k keys from the FP4 logits tile. Stage 2
    merges those keys directly into the caller-provided index buffer.
    """
    q_values, q_scales = q
    batch_size, next_n, num_heads, packed_dim = q_values.shape
    head_dim = packed_dim * 2
    assert head_dim == 128, f"MXFP4 tile top-k expects head_dim=128, got {head_dim}"
    assert num_heads in (32, 64), f"unsupported num_heads={num_heads}"
    assert q_scales.dtype == torch.int32
    assert q_scales.shape == (batch_size, next_n, num_heads)
    assert kv_cache.dtype == torch.uint8
    assert weights.shape == (batch_size * next_n, num_heads)
    assert weights.dtype == torch.float32
    assert context_lens.dtype == torch.int32
    assert context_lens.dim() in (1, 2)
    assert block_tables.dtype == torch.int32
    assert topk_indices.dtype == torch.int32
    assert topk_indices.shape[0] >= batch_size * next_n
    assert topk_indices.shape[1] >= topk_tokens

    block_size = int(kv_cache.shape[1])
    expected_cache_width = head_dim // 2 + 4
    assert kv_cache.dim() == 4 and kv_cache.shape[2] == 1
    assert kv_cache.shape[3] == expected_cache_width

    if max_context_len == 0 or topk_tokens == 0:
        return topk_indices
    if topk_tokens > 1024:
        raise ValueError("tile-fused FP4 top-k currently supports top-k <= 1024")
    if max_context_len >= 2**31:
        raise ValueError(f"max_context_len is too large: {max_context_len}")

    num_rows = batch_size * next_n
    tile_blocks = max(1, triton.cdiv(topk_tokens * 2, block_size))
    block_kv = _block_kv_for_dot(block_size * tile_blocks)
    if block_kv < topk_tokens:
        raise ValueError(
            f"tile-fused FP4 top-k tile {block_kv} is smaller than top-k {topk_tokens}"
        )
    num_logical_blocks = _num_logical_blocks_for_launch(
        max_context_len, block_size, block_tables
    )
    num_tiles = triton.cdiv(num_logical_blocks, tile_blocks)
    if num_tiles == 0:
        return topk_indices
    if num_tiles > 8:
        raise ValueError(
            "tile-fused FP4 top-k is validation-only for up to 8 tiles; "
            "use the chunked helper for longer active widths"
        )

    tile_keys = torch.empty(
        (num_rows, num_tiles, topk_tokens),
        dtype=torch.int64,
        device=q_values.device,
    )
    is_context_lens_2d = context_lens.dim() == 2
    _fp4_paged_mqa_logits_tile_topk_stage1_kernel[(num_rows, num_tiles)](
        q_values,
        q_scales,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        tile_keys,
        q_values.stride(0),
        q_values.stride(1),
        q_values.stride(2),
        q_values.stride(3),
        q_scales.stride(0),
        q_scales.stride(1),
        q_scales.stride(2),
        weights.stride(0),
        weights.stride(1),
        context_lens.stride(0),
        context_lens.stride(1) if is_context_lens_2d else 0,
        block_tables.stride(0),
        tile_keys.stride(0),
        tile_keys.stride(1),
        max_context_len,
        KV_CACHE_STRIDE_B=kv_cache.stride(0),
        KV_BLOCK_SIZE=block_size,
        NEXT_N=next_n,
        NUM_HEADS=num_heads,
        HEAD_DIM=head_dim,
        TILE_BLOCKS=tile_blocks,
        BLOCK_KV=block_kv,
        TOPK=int(topk_tokens),
        IS_CONTEXT_LENS_2D=is_context_lens_2d,
        INDEX_PAD=int(max_context_len) + 1,
        num_warps=8,
    )
    merge_tiles = min(2, num_tiles)
    _fp4_paged_mqa_logits_tile_topk_merge_kernel[(num_rows,)](
        tile_keys,
        context_lens,
        topk_indices,
        tile_keys.stride(0),
        tile_keys.stride(1),
        context_lens.stride(0),
        context_lens.stride(1) if is_context_lens_2d else 0,
        topk_indices.stride(0),
        NUM_TILES=num_tiles,
        TOPK=int(topk_tokens),
        MERGE_TILES=merge_tiles,
        NEXT_N=next_n,
        IS_CONTEXT_LENS_2D=is_context_lens_2d,
        MAX_CONTEXT_LEN=int(max_context_len),
        INDEX_PAD=int(max_context_len) + 1,
        num_warps=8,
    )
    return topk_indices


def rocm_fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_context_len: int,
    clean_logits: bool = False,
    logits_dtype: torch.dtype = torch.float32,
    trivial_topk_len: int = 0,
    launch_context_len: int | None = None,
) -> torch.Tensor:
    """Compute DeepGEMM-compatible paged MQA logits on ROCm."""
    del schedule_metadata

    q_values, q_scales = q
    is_fp4 = q_scales is not None
    assert kv_cache.dtype == torch.uint8, (
        f"kv_cache must be uint8, got {kv_cache.dtype}"
    )
    assert context_lens.dtype == torch.int32, (
        f"context_lens must be int32, got {context_lens.dtype}"
    )
    assert block_tables.dtype == torch.int32, (
        f"block_tables must be int32, got {block_tables.dtype}"
    )
    assert weights.dtype == torch.float32, (
        f"weights must be float32, got {weights.dtype}"
    )
    assert context_lens.dim() in (1, 2), (
        f"context_lens must be 1D or 2D, got {context_lens.shape}"
    )

    if is_fp4:
        assert q_scales is not None
        batch_size, next_n, num_heads, packed_dim = q_values.shape
        head_dim = packed_dim * 2
        assert q_scales.dtype == torch.int32, (
            f"MXFP4 q scales must be int32, got {q_scales.dtype}"
        )
        assert q_scales.shape == (batch_size, next_n, num_heads), (
            f"Expected q scales {(batch_size, next_n, num_heads)}, got {q_scales.shape}"
        )
    else:
        batch_size, next_n, num_heads, head_dim = q_values.shape

    assert head_dim in (32, 64, 128), f"unsupported head_dim={head_dim}"
    assert num_heads in (32, 64), f"unsupported num_heads={num_heads}"
    assert weights.shape == (batch_size * next_n, num_heads), (
        f"Expected weights {(batch_size * next_n, num_heads)}, got {weights.shape}"
    )
    if context_lens.dim() == 2:
        assert context_lens.shape == (batch_size, next_n), (
            f"Expected context_lens {(batch_size, next_n)}, got {context_lens.shape}"
        )
    else:
        assert context_lens.shape == (batch_size,), (
            f"Expected context_lens {(batch_size,)}, got {context_lens.shape}"
        )
    assert block_tables.shape[0] == batch_size, (
        f"Expected {batch_size} block-table rows, got {block_tables.shape[0]}"
    )
    assert kv_cache.dim() == 4 and kv_cache.shape[2] == 1, (
        "kv_cache must have shape [num_blocks, block_size, 1, width], got "
        f"{kv_cache.shape}"
    )

    block_size = int(kv_cache.shape[1])
    expected_cache_width = head_dim // 2 + 4 if is_fp4 else head_dim + 4
    assert kv_cache.shape[3] == expected_cache_width, (
        f"Expected kv_cache width {expected_cache_width}, got {kv_cache.shape[3]}"
    )
    assert logits_dtype in (torch.float32, torch.bfloat16), (
        f"logits_dtype must be float32 or bfloat16, got {logits_dtype}"
    )

    logits = torch.empty(
        (batch_size * next_n, max_context_len),
        dtype=logits_dtype,
        device=q_values.device,
    )
    if clean_logits:
        logits.fill_(float("-inf"))
    if max_context_len == 0:
        return logits
    if clean_logits:
        trivial_topk_len = 0
        launch_context_len = None
    if launch_context_len is None:
        launch_context_len = max_context_len
    else:
        launch_context_len = min(max(int(launch_context_len), 0), max_context_len)
        if launch_context_len == 0:
            return logits

    num_logical_blocks = _num_logical_blocks_for_launch(
        launch_context_len, block_size, block_tables
    )
    if num_logical_blocks == 0:
        return logits

    is_context_lens_2d = context_lens.dim() == 2

    if is_fp4:
        assert q_scales is not None
        if num_logical_blocks >= 4096 and batch_size * next_n >= 64:
            kv_blocks_per_prog = 8
            num_warps = 4
        elif num_logical_blocks >= 1024:
            kv_blocks_per_prog = 4
            num_warps = 4
        elif num_logical_blocks >= 256:
            kv_blocks_per_prog = 2
            num_warps = 4
        else:
            kv_blocks_per_prog = 2
            num_warps = 8
        grid = (
            batch_size * next_n,
            triton.cdiv(num_logical_blocks, kv_blocks_per_prog),
        )
        block_kv = _block_kv_for_dot(block_size * kv_blocks_per_prog)
        segment_group_progs = _fp4_segment_group_progs()
        tile_chunks = _fp4_tile_chunks()
        if segment_group_progs > 0:
            num_logical_block_groups = triton.cdiv(
                num_logical_blocks, kv_blocks_per_prog
            )
            group_progs = min(segment_group_progs, num_logical_block_groups)
            _fp4_paged_mqa_logits_segment_kernel[(batch_size * next_n, group_progs)](
                q_values,
                q_scales,
                kv_cache,
                weights,
                context_lens,
                block_tables,
                logits,
                q_values.stride(0),
                q_values.stride(1),
                q_values.stride(2),
                q_values.stride(3),
                q_scales.stride(0),
                q_scales.stride(1),
                q_scales.stride(2),
                weights.stride(0),
                weights.stride(1),
                context_lens.stride(0),
                context_lens.stride(1) if is_context_lens_2d else 0,
                block_tables.stride(0),
                logits.stride(0),
                launch_context_len,
                KV_CACHE_STRIDE_B=kv_cache.stride(0),
                KV_BLOCK_SIZE=block_size,
                NEXT_N=next_n,
                NUM_HEADS=num_heads,
                HEAD_DIM=head_dim,
                BLOCK_KV=block_kv,
                KV_BLOCKS_PER_PROG=kv_blocks_per_prog,
                NUM_LOGICAL_BLOCK_GROUPS=num_logical_block_groups,
                NUM_GROUP_PROGS=group_progs,
                IS_CONTEXT_LENS_2D=is_context_lens_2d,
                TRIVIAL_TOPK_LEN=int(trivial_topk_len),
                num_warps=num_warps,
            )
            return logits
        if tile_chunks > 1:
            _fp4_paged_mqa_logits_tiled_kernel[grid](
                q_values,
                q_scales,
                kv_cache,
                weights,
                context_lens,
                block_tables,
                logits,
                q_values.stride(0),
                q_values.stride(1),
                q_values.stride(2),
                q_values.stride(3),
                q_scales.stride(0),
                q_scales.stride(1),
                q_scales.stride(2),
                weights.stride(0),
                weights.stride(1),
                context_lens.stride(0),
                context_lens.stride(1) if is_context_lens_2d else 0,
                block_tables.stride(0),
                logits.stride(0),
                launch_context_len,
                KV_CACHE_STRIDE_B=kv_cache.stride(0),
                KV_BLOCK_SIZE=block_size,
                NEXT_N=next_n,
                NUM_HEADS=num_heads,
                HEAD_DIM=head_dim,
                BLOCK_KV=block_kv,
                KV_BLOCKS_PER_PROG=kv_blocks_per_prog,
                KV_TILE_CHUNKS=tile_chunks,
                IS_CONTEXT_LENS_2D=is_context_lens_2d,
                TRIVIAL_TOPK_LEN=int(trivial_topk_len),
                num_warps=num_warps,
            )
            return logits
        _fp4_paged_mqa_logits_kernel[grid](
            q_values,
            q_scales,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            logits,
            q_values.stride(0),
            q_values.stride(1),
            q_values.stride(2),
            q_values.stride(3),
            q_scales.stride(0),
            q_scales.stride(1),
            q_scales.stride(2),
            weights.stride(0),
            weights.stride(1),
            context_lens.stride(0),
            context_lens.stride(1) if is_context_lens_2d else 0,
            block_tables.stride(0),
            logits.stride(0),
            launch_context_len,
            KV_CACHE_STRIDE_B=kv_cache.stride(0),
            KV_BLOCK_SIZE=block_size,
            NEXT_N=next_n,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            BLOCK_KV=block_kv,
            KV_BLOCKS_PER_PROG=kv_blocks_per_prog,
            IS_CONTEXT_LENS_2D=is_context_lens_2d,
            TRIVIAL_TOPK_LEN=int(trivial_topk_len),
            num_warps=num_warps,
        )
    else:
        grid = (batch_size * next_n, num_logical_blocks)
        block_kv = _block_kv_for_dot(block_size)
        _fp8_paged_mqa_logits_kernel[grid](
            q_values,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            logits,
            q_values.stride(0),
            q_values.stride(1),
            q_values.stride(2),
            q_values.stride(3),
            weights.stride(0),
            weights.stride(1),
            context_lens.stride(0),
            context_lens.stride(1) if is_context_lens_2d else 0,
            block_tables.stride(0),
            logits.stride(0),
            launch_context_len,
            KV_CACHE_STRIDE_B=kv_cache.stride(0),
            KV_BLOCK_SIZE=block_size,
            NEXT_N=next_n,
            NUM_HEADS=num_heads,
            HEAD_DIM=head_dim,
            BLOCK_KV=block_kv,
            IS_CONTEXT_LENS_2D=is_context_lens_2d,
            IS_FNUZ=_fp8_is_fnuz(),
            num_warps=4,
        )
    return logits
