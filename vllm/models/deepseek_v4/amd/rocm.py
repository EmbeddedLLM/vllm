# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.fusion.quant_activation import QuantizedActivation
from vllm.model_executor.models.utils import extract_layer_index
from vllm.models.deepseek_v4.amd.model_state import (
    DeepseekV4RocmAtomStateMetadata,
    get_deepseek_v4_rocm_atom_state,
)
from vllm.models.deepseek_v4.amd.v4_kernels import (
    csa_translate_pack,
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_decode_split_kv,
    sparse_attn_v4_paged_prefill,
    sparse_attn_v4_paged_prefill_split_kv,
    swa_write,
    write_v4_paged_decode_indices,
    write_v4_paged_prefill_indices,
)
from vllm.models.deepseek_v4.amd.v4_kernels.qk_norm_rope_maybe_quant import (
    qk_norm_rope_maybe_quant,
)
from vllm.models.deepseek_v4.attention import DeepseekV4Attention
from vllm.models.deepseek_v4.common.ops import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)
from vllm.models.deepseek_v4.sparse_mla import (
    DeepseekV4FlashMLABackend,
    DeepseekV4FlashMLAMetadata,
    DeepseekV4FlashMLAMetadataBuilder,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backend import (
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.mla.sparse_swa import (
    DeepseekSparseSWAMetadata,
    DeepseekSparseSWAMetadataBuilder,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    build_ragged_indices_from_dense,
    rocm_inv_rope_einsum,
    rocm_sparse_attn_decode,
    rocm_sparse_attn_prefill,
)
from vllm.v1.worker.workspace import current_workspace_manager

# Module-level reusable buffer for attention output.  Eliminates 61 torch.empty
# calls per decode step under breakable CUDA graph.  Grown (never shrunk) to
# the largest num_tokens × padded_heads × head_dim seen so far.
_O_PADDED_BUF: torch.Tensor | None = None


def _get_o_padded(
    num_tokens: int,
    padded_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    global _O_PADDED_BUF
    needed = num_tokens * padded_heads * head_dim
    if _O_PADDED_BUF is None or _O_PADDED_BUF.numel() < needed:
        _O_PADDED_BUF = torch.empty(
            (num_tokens, padded_heads, head_dim),
            dtype=dtype,
            device=device,
        )
    return _O_PADDED_BUF[:num_tokens]


def _atom_attention_enabled_for_ratio(ratio: int) -> bool:
    return ratio >= 1


def _atom_metadata_cache_hits_allowed() -> bool:
    # Cache hits are a Python-side launch elision. During CUDA/HIP graph
    # capture the dummy block-table tensors can alias across cache groups even
    # though their runtime mappings do not. Eliding a launch then permanently
    # omits it from the captured graph, so every layer must record its metadata
    # update while capture is active.
    return not torch.cuda.is_current_stream_capturing()


def _should_use_atom_split_kv_decode(
    unified_kv: torch.Tensor | None,
    split_swa_kv: torch.Tensor | None,
    split_compressed_kv: torch.Tensor | None,
) -> bool:
    has_split_kv = split_swa_kv is not None and split_compressed_kv is not None
    if not has_split_kv:
        return False
    return unified_kv is None


@dataclass(frozen=True)
class _AtomKVViews:
    unified_kv: torch.Tensor | None
    split_swa_kv: torch.Tensor | None
    split_compressed_kv: torch.Tensor | None
    split_kv_scales: torch.Tensor | None
    split_kv_layout: str
    unified_kv_scales: torch.Tensor | None


# Per-layer KV view cache.  Keyed by (id(atom_state), layer_id) so the cache
# auto-invalidates when the model state is recreated (server restart / KV
# realloc).  Saves ~15 getattr calls × 61 layers = ~915 lookups per step.
_KV_VIEW_CACHE: dict[tuple[int, int | None], _AtomKVViews] = {}


def _resolve_atom_kv_views(
    attn: object,
    atom_state: object,
) -> _AtomKVViews:
    """Resolve ATOM KV views, preferring scheduler/model-state metadata.

    Layer attributes are still the source for the SWA view because packed
    split-only vLLM-owned KV has no homogeneous unified tensor in the metadata
    bundle.  Compressed tail layout and scales come from the per-step metadata
    bundle when it is present so decode/prefill cannot accidentally run with a
    stale default layout string.
    """

    layer_id = getattr(attn, "_atom_layer_id", None)
    cache_key = (id(atom_state), layer_id)
    cached = _KV_VIEW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    unified_kv = getattr(attn, "atom_unified_kv", None)
    swa_cache_layer = getattr(attn, "swa_cache_layer", None)
    paged_swa_kv = getattr(swa_cache_layer, "kv_cache", None)
    paged_swa_bound = (
        isinstance(paged_swa_kv, torch.Tensor)
        and paged_swa_kv.dtype == torch.bfloat16
        and paged_swa_kv.numel() > 0
    )
    if paged_swa_bound:
        split_swa_kv = paged_swa_kv
        if int(getattr(attn, "compress_ratio", 1)) <= 1:
            # ``reshape`` silently materializes a copy for vLLM's packed
            # inter-layer layout.  Decode would then keep reading that stale
            # snapshot while QK+RoPE updated the original paged tensor.  Only
            # expose a flattened unified view when it really aliases storage;
            # otherwise route the SWA-only layer through the stride-aware
            # split-KV kernels with an empty compressed tail.
            if paged_swa_kv.is_contiguous():
                unified_kv = paged_swa_kv.view(-1, int(attn.head_dim))
            else:
                unified_kv = None
                split_compressed_kv = paged_swa_kv.as_strided(
                    (0, int(attn.head_dim)),
                    (int(attn.head_dim), 1),
                )
        else:
            unified_kv = None
    else:
        split_swa_kv = getattr(attn, "atom_split_kv_swa", None)
    split_compressed_kv = getattr(attn, "atom_split_kv_compressed", None)
    split_kv_scales = getattr(attn, "atom_split_kv_scales", None)
    split_kv_layout = getattr(attn, "atom_split_kv_layout", "dense")
    unified_kv_scales = getattr(attn, "atom_unified_kv_scales", None)

    buffers = getattr(atom_state, "unified_kv_buffers", None)
    if buffers is not None and layer_id is not None:
        if layer_id in buffers.compressed_kv_cache:
            split_compressed_kv = buffers.compressed_kv_cache[layer_id]
            split_kv_scales = buffers.compressed_kv_scales.get(layer_id)
            split_kv_layout = buffers.compressed_kv_layout.get(layer_id, "dense")

        if unified_kv is None and not paged_swa_bound:
            unified_kv_by_layer = getattr(buffers, "unified_kv_by_layer", {})
            unified_kv = unified_kv_by_layer.get(layer_id)

        if split_swa_kv is None and unified_kv is not None:
            max_num_reqs = int(getattr(attn, "max_num_reqs", 0) or 0)
            if max_num_reqs <= 0:
                max_num_reqs = max(
                    1,
                    int(getattr(atom_state, "swa_pages", 0))
                    // max(1, int(getattr(atom_state, "win_with_spec", 1))),
                )
            split_swa_kv = unified_kv[: int(atom_state.swa_pages)].view(
                max_num_reqs,
                int(atom_state.win_with_spec),
                int(attn.head_dim),
            )

    views = _AtomKVViews(
        unified_kv=unified_kv,
        split_swa_kv=split_swa_kv,
        split_compressed_kv=split_compressed_kv,
        split_kv_scales=split_kv_scales,
        split_kv_layout=split_kv_layout,
        unified_kv_scales=unified_kv_scales,
    )
    _KV_VIEW_CACHE[cache_key] = views
    return views


def _atom_attention_enabled_for_layer(layer_id: int | None) -> bool:
    return True


def _atom_paged_swa_is_authoritative(
    attn: object,
    swa_metadata: DeepseekSparseSWAMetadata,
    atom_state: DeepseekV4RocmAtomStateMetadata | None,
    atom_swa_kv: torch.Tensor | None,
) -> bool:
    """Whether the vLLM-owned paged BF16 SWA cache is authoritative."""
    if atom_state is None or atom_swa_kv is None:
        return False
    return bool(
        atom_swa_kv.dtype == torch.bfloat16
        and atom_swa_kv.dim() == 3
        and _atom_attention_enabled_for_ratio(int(getattr(attn, "compress_ratio", 0)))
        and _atom_attention_enabled_for_layer(getattr(attn, "_atom_layer_id", None))
        and atom_state.num_actual_tokens > 0
        and swa_metadata.block_table.dim() == 2
    )


def _build_indptr_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    lengths = lengths.to(dtype=torch.int32).contiguous()
    indptr = torch.zeros(lengths.shape[0] + 1, dtype=torch.int32, device=lengths.device)
    torch.cumsum(lengths, dim=0, out=indptr[1:])
    return indptr


# ROCm sparse prefill keeps this dense combine local so AMD-specific SWA changes
# do not touch the shared DeepSeek V4 cache utilities.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    TOPK_WIDTH: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        topk_offset = tl.arange(0, PADDED_TOP_K)
        topk_mask = topk_offset < topk_len
        safe_topk_offset = tl.where(topk_offset < TOPK_WIDTH, topk_offset, 0)
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + safe_topk_offset,
            mask=topk_mask,
            other=-1,
        )
        valid_topk = (topk_indices >= 0) & (topk_indices < N)
        topk_indices = tl.where(valid_topk, topk_indices + M * batch_idx, -1)
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + topk_offset,
            topk_indices,
            mask=topk_mask,
        )

        swa_offset = tl.arange(0, WINDOW_SIZE)
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + swa_offset,
            M * batch_idx + N + swa_offset + pos - swa_len + 1 - gather_start,
            mask=swa_offset < swa_len,
        )

        tl.store(combined_lens_ptr + token_idx, topk_len + swa_len)


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    num_workers = 128
    _combine_topk_swa_indices_kernel[(num_reqs, num_workers)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        TOPK_WIDTH=topk_indices.shape[-1],
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _compute_topk_lens_kernel(
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk
        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        count += tl.sum((local_idx >= 0).to(tl.int32), axis=0)

    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@triton.jit
def _pack_global_topk_ragged_kernel(
    global_topk_ragged_ptr,
    topk_indptr_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    topk,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    offset = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    out_start = tl.load(topk_indptr_ptr + token_idx)
    out_end = tl.load(topk_indptr_ptr + token_idx + 1)
    out_len = out_end - out_start
    if block_idx * BLOCK_SIZE >= out_len:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    mask = (offset < out_len) & (offset < topk)
    local_idx = tl.load(
        topk_indices_ptr + token_idx * topk_indices_stride + offset,
        mask=mask,
        other=-1,
    )
    valid = mask & (local_idx >= 0)
    block_indices = local_idx // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=valid,
        other=0,
    )
    block_offsets = local_idx % block_size
    slot_ids = tl.where(valid, block_numbers * block_size + block_offsets, -1)
    tl.store(global_topk_ragged_ptr + out_start + offset, slot_ids, mask=mask)


def compute_global_topk_ragged_indices_and_indptr(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    topk_indices = topk_indices.reshape(topk_indices.shape[0], -1).contiguous()
    num_tokens = topk_indices.shape[0]
    topk = topk_indices.shape[1]

    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_topk_lens_kernel[(num_tokens,)](
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )

    topk_indptr = _build_indptr_from_lengths(topk_lens)
    global_topk_ragged = torch.empty(
        num_tokens * topk,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    if global_topk_ragged.numel() > 0:
        block = 128
        _pack_global_topk_ragged_kernel[(num_tokens, triton.cdiv(topk, block))](
            global_topk_ragged,
            topk_indptr,
            topk_indices,
            topk_indices.stride(0),
            token_to_req_indices,
            block_table,
            block_table.stride(0),
            block_size,
            topk,
            BLOCK_SIZE=block,
        )
    return global_topk_ragged, topk_indptr, topk_lens


def _copy_ragged_to_graph_buffers(
    ragged_indices: torch.Tensor,
    ragged_indptr: torch.Tensor,
    ragged_indices_buffer: torch.Tensor,
    ragged_indptr_buffer: torch.Tensor,
    num_rows: int,
    max_entries_per_row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Copy dynamic ragged metadata into persistent CUDA graph buffers.

    FULL decode graphs capture kernel argument addresses. Keep the returned
    tensors backed by stable storage, while indptr continues to bound reads.
    """
    indptr_out = ragged_indptr_buffer[: num_rows + 1]
    indptr_out.copy_(ragged_indptr, non_blocking=True)

    max_entries = max(num_rows * max_entries_per_row, 1)
    ragged_out = ragged_indices_buffer[:max_entries]
    nnz = ragged_indices.numel()
    if nnz > 0:
        ragged_out[:nnz].copy_(ragged_indices, non_blocking=True)
    return ragged_out, indptr_out


@triton.jit
def _gather_plain_k_cache_kernel(
    out,
    out_stride_b,
    out_stride_m,
    k_cache,
    k_cache_stride_b,
    k_cache_stride_s,
    seq_lens,
    gather_lens,
    block_table,
    block_table_stride,
    offset: tl.constexpr,
    block_size: tl.constexpr,
    D: tl.constexpr,
    HAS_GATHER_LENS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    token_worker = tl.program_id(1)
    token_workers = tl.num_programs(1)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    seq_len = tl.load(seq_lens + batch_idx)
    gather_len = tl.load(gather_lens + batch_idx) if HAS_GATHER_LENS else seq_len
    start_pos = seq_len - gather_len

    for i in range(token_worker, gather_len, token_workers):
        pos = start_pos + i
        block_in_seq = pos // block_size
        pos_in_block = pos - block_in_seq * block_size
        physical_block = tl.load(
            block_table + batch_idx * block_table_stride + block_in_seq
        )

        vals = tl.load(
            k_cache
            + physical_block.to(tl.int64) * k_cache_stride_b
            + pos_in_block * k_cache_stride_s
            + d_offsets,
            mask=d_mask,
            other=0.0,
        )
        tl.store(
            out + batch_idx * out_stride_b + (offset + i) * out_stride_m + d_offsets,
            vals,
            mask=d_mask,
        )


def _gather_plain_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    *,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    if out.numel() == 0:
        return
    if k_cache.dtype != out.dtype:
        raise RuntimeError(
            "Plain K-cache gather expects cache/output dtype match, got "
            f"{k_cache.dtype} and {out.dtype}."
        )
    block_d = triton.next_power_of_2(out.shape[-1])
    _gather_plain_k_cache_kernel[(seq_lens.shape[0], 128)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        k_cache.stride(0),
        k_cache.stride(1),
        seq_lens,
        gather_lens if gather_lens is not None else seq_lens,
        block_table,
        block_table.stride(0),
        offset=offset,
        block_size=block_size,
        D=out.shape[-1],
        HAS_GATHER_LENS=gather_lens is not None,
        BLOCK_D=block_d,
    )


def _gather_k_cache(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    *,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
) -> None:
    if k_cache.dtype == torch.uint8:
        dequantize_and_gather_k_cache(
            out,
            k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            offset=offset,
            use_fnuz=use_fnuz,
        )
    else:
        _gather_plain_k_cache(
            out,
            k_cache,
            seq_lens=seq_lens,
            gather_lens=gather_lens,
            block_table=block_table,
            block_size=block_size,
            offset=offset,
        )


@triton.jit
def _write_hca_compress_head_kernel(
    batch_id_per_token,
    hca_indptr,
    n_committed_hca_per_seq,
    block_table,
    block_table_stride,
    hca_indices,
    swa_pages,
    hca_block_capacity: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    token_idx = tl.program_id(0)
    bid = tl.load(batch_id_per_token + token_idx)
    if bid < 0:
        return

    n_hca = tl.load(n_committed_hca_per_seq + bid)
    base = tl.load(hca_indptr + token_idx)
    block_base = bid * block_table_stride
    offsets = tl.arange(0, BLOCK_J)
    for j in tl.range(0, n_hca, BLOCK_J):
        hca_offsets = j + offsets
        mask = hca_offsets < n_hca
        block_offsets = hca_offsets // hca_block_capacity
        slot_offsets = hca_offsets - block_offsets * hca_block_capacity
        physical_blocks = tl.load(
            block_table + block_base + block_offsets,
            mask=mask,
            other=0,
        )
        tl.store(
            hca_indices + base + hca_offsets,
            swa_pages + physical_blocks * hca_block_capacity + slot_offsets,
            mask=mask,
        )


def _write_hca_compress_head(
    *,
    block_table: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    n_committed_hca_per_seq: torch.Tensor,
    hca_indices: torch.Tensor,
    hca_indptr: torch.Tensor,
    swa_pages: int,
    hca_block_capacity: int,
    T: int,
) -> None:
    if T == 0:
        return
    if hca_block_capacity <= 0:
        raise RuntimeError(
            f"Invalid HCA block capacity for ATOM decode: {hca_block_capacity}."
        )
    block_j = 128
    _write_hca_compress_head_kernel[(T,)](
        batch_id_per_token,
        hca_indptr,
        n_committed_hca_per_seq,
        block_table,
        block_table.stride(0),
        hca_indices,
        swa_pages,
        hca_block_capacity=hca_block_capacity,
        BLOCK_J=block_j,
    )


@dataclass
class DeepseekV4ROCMAiterMLASparseMetadata(DeepseekV4FlashMLAMetadata):
    """ROCm-specific DeepSeek V4 metadata carrying ragged decode topk."""

    c128a_decode_topk_ragged_indices: torch.Tensor | None = None
    c128a_decode_topk_ragged_indptr: torch.Tensor | None = None


@dataclass
class DeepseekV4ROCMAiterSparseSWAMetadata(DeepseekSparseSWAMetadata):
    decode_swa_ragged_indices: torch.Tensor | None = None
    decode_swa_ragged_indptr: torch.Tensor | None = None


class DeepseekV4ROCMAiterMLASparseMetadataBuilder(DeepseekV4FlashMLAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.c128a_decode_topk_ragged_indices_buffer: torch.Tensor | None = None
        self.c128a_decode_topk_ragged_indptr_buffer: torch.Tensor | None = None
        if self.compress_ratio == 128:
            max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
            self.c128a_decode_topk_ragged_indices_buffer = torch.empty(
                max_tokens * self.c128a_max_compressed,
                dtype=torch.int32,
                device=self.device,
            )
            self.c128a_decode_topk_ragged_indptr_buffer = torch.empty(
                max_tokens + 1,
                dtype=torch.int32,
                device=self.device,
            )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterMLASparseMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        ragged_indices = None
        ragged_indptr = None
        dense_decode = base.c128a_global_decode_topk_indices
        decode_lens = base.c128a_decode_topk_lens
        if dense_decode is not None and decode_lens is not None:
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                dense_decode.reshape(dense_decode.shape[0], -1),
                decode_lens,
            )
            assert self.c128a_decode_topk_ragged_indices_buffer is not None
            assert self.c128a_decode_topk_ragged_indptr_buffer is not None
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.c128a_decode_topk_ragged_indices_buffer,
                self.c128a_decode_topk_ragged_indptr_buffer,
                dense_decode.shape[0],
                self.c128a_max_compressed,
            )
        return DeepseekV4ROCMAiterMLASparseMetadata(
            **vars(base),
            c128a_decode_topk_ragged_indices=ragged_indices,
            c128a_decode_topk_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterSparseSWAMetadataBuilder(DeepseekSparseSWAMetadataBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        max_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        # The non-causal (DSpark draft) path widens each token's SWA index list
        # to ``noncausal_index_width`` (>= window_size), so size the persistent
        # ragged buffer to the wider bound to cover both causal and non-causal.
        swa_index_width = max(self.window_size, self.noncausal_index_width)
        self.decode_swa_ragged_indices_buffer = torch.empty(
            max_tokens * swa_index_width,
            dtype=torch.int32,
            device=self.device,
        )
        self.decode_swa_ragged_indptr_buffer = torch.empty(
            max_tokens + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> DeepseekV4ROCMAiterSparseSWAMetadata:
        base = super().build(
            common_prefix_len=common_prefix_len,
            common_attn_metadata=common_attn_metadata,
            fast_build=fast_build,
        )
        ragged_indices = None
        ragged_indptr = None
        if (
            base.num_decode_tokens > 0
            and base.decode_swa_indices is not None
            and base.decode_swa_lens is not None
        ):
            ragged_indices, ragged_indptr = build_ragged_indices_from_dense(
                base.decode_swa_indices.reshape(base.num_decode_tokens, -1),
                base.decode_swa_lens,
            )
            ragged_indices, ragged_indptr = _copy_ragged_to_graph_buffers(
                ragged_indices,
                ragged_indptr,
                self.decode_swa_ragged_indices_buffer,
                self.decode_swa_ragged_indptr_buffer,
                base.num_decode_tokens,
                # Actual dense width for this build: window_size (causal) or
                # noncausal_index_width (DSpark non-causal draft).
                base.decode_swa_indices.shape[-1],
            )
        return DeepseekV4ROCMAiterSparseSWAMetadata(
            **vars(base),
            decode_swa_ragged_indices=ragged_indices,
            decode_swa_ragged_indptr=ragged_indptr,
        )


class DeepseekV4ROCMAiterMLASparseBackend(DeepseekV4FlashMLABackend):
    @staticmethod
    def get_name() -> str:
        return "ROCM_FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type["DeepseekV4ROCMAiterMLASparseMetadataBuilder"]:
        return DeepseekV4ROCMAiterMLASparseMetadataBuilder


class DeepseekV4ROCMAiterMLAAttention(DeepseekV4Attention):
    """ROCm sparse MLA attention layer for DeepSeek V4."""

    backend_cls = DeepseekV4ROCMAiterMLASparseBackend

    @property
    def _atom_layer_id(self) -> int | None:
        try:
            return extract_layer_index(self.prefix)
        except ValueError:
            return None

    @classmethod
    def get_padded_num_q_heads(cls, num_heads: int) -> int:
        return num_heads

    def _q_norm_maybe_quant(
        self,
        qr: torch.Tensor,
    ) -> torch.Tensor | QuantizedActivation:
        quant_key = getattr(self.wq_b, "input_quant_key", None)
        if quant_key is None:
            return self.q_norm(qr)

        if self.indexer is not None:
            indexer_quant_key = getattr(self.indexer.wq_b, "input_quant_key", None)
            if indexer_quant_key != quant_key:
                return self.q_norm(qr)

        scale_desc = quant_key.scale
        group_shape = scale_desc.group_shape
        if (
            scale_desc.static
            or scale_desc.dtype is not torch.float32
            or not group_shape.is_per_group()
            or group_shape.col <= 0
            or qr.shape[-1] % group_shape.col != 0
        ):
            return self.q_norm(qr)

        qr_quant, qr_scale = rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()(
            qr,
            self.q_norm.weight.data,
            self.eps,
            group_shape.col,
        )
        return QuantizedActivation(
            data=qr_quant,
            scale=qr_scale,
            orig_dtype=qr.dtype,
            orig_shape=qr.shape,
            quant_key=quant_key,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = hidden_states.shape[0]
        o_padded = _get_o_padded(
            num_tokens,
            self.padded_heads,
            self.head_dim,
            hidden_states.dtype,
            hidden_states.device,
        )

        qr_kv, kv_score, indexer_kv_score, indexer_weights = (
            self.attn_gemm_parallel_execute(hidden_states)
        )
        qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
        qr = self._q_norm_maybe_quant(qr)

        self.attention_impl(
            hidden_states,
            qr,
            kv,
            kv_score,
            indexer_kv_score,
            indexer_weights,
            positions,
            o_padded,
        )
        o = o_padded[:, : self.n_local_heads, :]
        return self._o_proj(o, positions)

    def _o_proj(self, o: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # ROCm BF16 reference wo_a path (inverse RoPE + einsum) + wo_b.
        z = rocm_inv_rope_einsum(
            self.rotary_emb,
            o,
            positions,
            self.rope_head_dim,
            self.n_local_groups,
            self.o_lora_rank,
            self.wo_a,
        )
        return self.wo_b(z.flatten(1))

    def _atom_rotary_cos_sin(
        self, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin_cache = self.rotary_emb.cos_sin_cache
        half_rope = self.rope_head_dim // 2
        cache_key = (
            cos_sin_cache.data_ptr(),
            cos_sin_cache.device,
            cos_sin_cache.dtype,
            tuple(cos_sin_cache.shape),
            half_rope,
            dtype,
        )
        cached = getattr(self.rotary_emb, "_atom_split_cos_sin_cache", None)
        if cached is not None and cached[0] == cache_key:
            return cached[1], cached[2]

        cos_cache = cos_sin_cache[..., :half_rope].to(dtype=dtype).contiguous()
        sin_cache = (
            cos_sin_cache[..., half_rope : 2 * half_rope].to(dtype=dtype).contiguous()
        )
        self.rotary_emb._atom_split_cos_sin_cache = (
            cache_key,
            cos_cache,
            sin_cache,
        )
        return cos_cache, sin_cache

    def _fused_qnorm_rope_kv_insert(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: object,
    ) -> torch.Tensor:
        if not isinstance(attn_metadata, dict):
            if self.n_local_heads < self.padded_heads:
                out = q.new_zeros(q.shape[0], self.padded_heads, self.head_dim)
                out[:, : self.n_local_heads, :].copy_(q)
                return out
            return q

        swa_metadata = cast(
            DeepseekSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None
        assert positions.dtype == torch.int64

        cos_cache, sin_cache = self._atom_rotary_cos_sin(kv.dtype)

        q_flat = q.reshape(q.shape[0], self.n_local_heads * self.head_dim)
        atom_state = get_deepseek_v4_rocm_atom_state(swa_metadata)
        atom_swa_kv = self.swa_cache_layer.kv_cache
        paged_swa_authoritative = _atom_paged_swa_is_authoritative(
            self,
            swa_metadata,
            atom_state,
            atom_swa_kv,
        )
        self._atom_paged_swa_authoritative = paged_swa_authoritative
        write_paged_swa = (
            paged_swa_authoritative
            and swa_metadata.num_prefills == 0
            and swa_metadata.slot_mapping.shape[0] >= q_flat.shape[0]
        )
        self._atom_swa_write_fused = write_paged_swa
        q_out, kv_out, _, _ = qk_norm_rope_maybe_quant(
            q_flat,
            kv,
            self.kv_norm.weight.data,
            cos_cache,
            sin_cache,
            positions,
            self.n_local_heads,
            self.head_dim,
            self.rope_head_dim,
            self.eps,
            quant_q=False,
            quant_k=False,
            swa_kv=atom_swa_kv if write_paged_swa else None,
            state_slot_mapping=(
                atom_state.state_slot_mapping if write_paged_swa else None
            ),
            batch_id_per_token=(
                atom_state.batch_id_per_token[: q_flat.shape[0]]
                if write_paged_swa
                and atom_state is not None
                and atom_state.batch_id_per_token.shape[0] >= q_flat.shape[0]
                else None
            ),
            swa_cu_seqlens_q=(atom_state.query_start_loc if write_paged_swa else None),
            swa_cache_size=(swa_metadata.block_size if write_paged_swa else None),
            swa_write_per_batch=(
                swa_metadata.max_query_len if write_paged_swa else None
            ),
            swa_block_tables=(swa_metadata.block_table if write_paged_swa else None),
            swa_block_size=(swa_metadata.block_size if write_paged_swa else None),
            # Keep the block table authoritative. Passing both forms made the
            # local fused fallback prefer ``slot_mapping`` and bypass the
            # content-addressed path that ATOM/AITER paged SWA requires.
            swa_slot_mapping=None,
        )
        self._atom_last_kv = kv_out

        if not write_paged_swa:
            swa_kv_cache = self.swa_cache_layer.kv_cache
            if paged_swa_authoritative:
                # Prefill writes after sparse attention so the current chunk
                # cannot replace prior-window entries before they are read.
                pass
            elif swa_kv_cache.dtype == torch.uint8:
                quantize_and_insert_k_cache(
                    kv_out,
                    swa_kv_cache.view(swa_kv_cache.shape[0], -1),
                    swa_metadata.slot_mapping,
                    swa_metadata.block_size,
                )
            else:
                raise NotImplementedError(
                    "ROCm DeepSeek V4 ATOM q/k path currently expects "
                    "fp8_ds_mla uint8 SWA cache."
                )
        return q_out

    def forward_mqa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        assert output.shape == q.shape, (
            f"output buffer shape {output.shape} must match q shape {q.shape}"
        )
        assert output.dtype == q.dtype, (
            f"output buffer dtype {output.dtype} must match q dtype {q.dtype}"
        )

        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata

        if attn_metadata is None:
            # Warmup dummy run: no real metadata. Reserve the same bf16
            # gather workspace _forward_prefill would; the dequantize / topk
            # / sparse_fwd kernels are skipped this step.
            swa_only = self.compress_ratio <= 1
            N = (
                0
                if swa_only
                else (self.max_model_len + self.compress_ratio - 1)
                // self.compress_ratio
            )
            M = N + self.window_size + self.max_num_batched_tokens
            current_workspace_manager().get_simultaneous(
                ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
            )
            output.zero_()
            return

        assert isinstance(attn_metadata, dict)
        rocm_metadata = cast(
            DeepseekV4ROCMAiterMLASparseMetadata | None,
            attn_metadata.get(self.prefix),
        )
        swa_metadata = cast(
            DeepseekV4ROCMAiterSparseSWAMetadata | None,
            attn_metadata.get(self.swa_cache_layer.prefix),
        )
        assert swa_metadata is not None

        swa_only = self.compress_ratio <= 1
        self_kv_cache = self.kv_cache if not swa_only else None
        swa_kv_cache = self.swa_cache_layer.kv_cache

        num_decodes = swa_metadata.num_decodes
        num_prefills = swa_metadata.num_prefills
        num_decode_tokens = swa_metadata.num_decode_tokens

        if num_prefills == 0:
            self._maybe_atom_swa_write(positions, swa_metadata)

        prefill_used_atom = False
        if num_prefills > 0:
            prefill_used_atom = self._forward_prefill(
                q=q[num_decode_tokens:],
                positions=positions[num_decode_tokens:],
                compressed_k_cache=self_kv_cache,
                swa_k_cache=swa_kv_cache,
                output=output[num_decode_tokens:],
                attn_metadata=rocm_metadata,
                swa_metadata=swa_metadata,
            )
            if not prefill_used_atom:
                self._maybe_atom_swa_write(positions, swa_metadata)
        if num_decodes > 0:
            self._forward_decode(
                q=q[:num_decode_tokens],
                positions=positions[:num_decode_tokens],
                kv_cache=self_kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=rocm_metadata,
                swa_only=swa_only,
                allow_atom=num_prefills == 0 or prefill_used_atom,
                output=output[:num_decode_tokens],
            )

    def _maybe_atom_swa_write(
        self,
        positions: torch.Tensor,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        if getattr(self, "_atom_swa_write_fused", False):
            return
        if not _atom_attention_enabled_for_ratio(
            self.compress_ratio
        ) or not _atom_attention_enabled_for_layer(self._atom_layer_id):
            return
        atom_state = get_deepseek_v4_rocm_atom_state(swa_metadata)
        atom_swa_kv = self.swa_cache_layer.kv_cache
        kv = getattr(self, "_atom_last_kv", None)
        if atom_state is None or atom_swa_kv is None or kv is None:
            return
        if atom_state.num_actual_reqs <= 0 or atom_state.num_actual_tokens <= 0:
            return

        write_per_batch = min(
            int(getattr(swa_metadata, "max_query_len", kv.shape[0]) or kv.shape[0]),
            int(kv.shape[0]),
        )
        if write_per_batch <= 0:
            return

        swa_write(
            kv[: atom_state.num_actual_tokens],
            positions[: atom_state.num_actual_tokens],
            atom_state.query_start_loc[: atom_state.num_actual_reqs + 1],
            atom_state.state_slot_mapping[: atom_state.num_actual_reqs],
            atom_swa_kv,
            swa_metadata.block_size,
            write_per_batch,
            block_tables=swa_metadata.block_table,
            block_size=swa_metadata.block_size,
        )

    def _forward_decode(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        allow_atom: bool,
        output: torch.Tensor,
    ) -> None:
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        if allow_atom and self._maybe_forward_decode_atom(
            q=q,
            positions=positions,
            swa_metadata=swa_metadata,
            attn_metadata=attn_metadata,
            swa_only=swa_only,
            output=output,
        ):
            return

        topk_indices = None
        topk_lens = None
        topk_ragged_indices = None
        topk_ragged_indptr = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                (
                    topk_ragged_indices,
                    topk_ragged_indptr,
                    topk_lens,
                ) = compute_global_topk_ragged_indices_and_indptr(
                    self.topk_indices_buffer[:num_decode_tokens],
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
                topk_ragged_indices = attn_metadata.c128a_decode_topk_ragged_indices
                topk_ragged_indptr = attn_metadata.c128a_decode_topk_ragged_indptr

        rocm_sparse_attn_decode(
            q=q,
            kv_cache=kv_cache,
            swa_k_cache=self.swa_cache_layer.kv_cache,
            swa_only=swa_only,
            topk_indices=topk_indices,
            topk_lens=topk_lens,
            swa_indices=swa_metadata.decode_swa_indices,
            swa_lens=swa_metadata.decode_swa_lens,
            swa_ragged_indices=swa_metadata.decode_swa_ragged_indices,
            swa_ragged_indptr=swa_metadata.decode_swa_ragged_indptr,
            topk_ragged_indices=topk_ragged_indices,
            topk_ragged_indptr=topk_ragged_indptr,
            attn_sink=self.attn_sink,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            output=output,
        )

    def _maybe_forward_decode_atom(
        self,
        *,
        q: torch.Tensor,
        positions: torch.Tensor,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> bool:
        if not _atom_attention_enabled_for_ratio(
            self.compress_ratio
        ) or not _atom_attention_enabled_for_layer(self._atom_layer_id):
            return False
        atom_state = get_deepseek_v4_rocm_atom_state(swa_metadata)
        if atom_state is None or atom_state.decode_buffers is None:
            return False
        kv_views = _resolve_atom_kv_views(self, atom_state)
        unified_kv = kv_views.unified_kv
        split_swa_kv = kv_views.split_swa_kv
        split_compressed_kv = kv_views.split_compressed_kv
        split_kv_scales = kv_views.split_kv_scales
        split_kv_layout = kv_views.split_kv_layout
        use_split_kv_decode = _should_use_atom_split_kv_decode(
            unified_kv,
            split_swa_kv,
            split_compressed_kv,
        )
        if unified_kv is None and not use_split_kv_decode:
            raise RuntimeError(
                "ROCm DeepSeek-V4 ATOM attention requires bound unified or "
                "split KV views."
            )
        if split_swa_kv is not None:
            swa_pages = split_swa_kv.numel() // self.head_dim
        else:
            assert unified_kv is not None
            swa_pages = int(atom_state.swa_pages)
        # Optional bridge for a future all-FP8 ATOM unified-KV pool.  The
        # generic paged-decode wrapper can dequantize in-kernel when this scale
        # tensor is present.  The current production path leaves it unset and
        # uses the validated homogeneous BF16 unified-KV layout.
        unified_kv_scales = kv_views.unified_kv_scales

        def run_paged_decode(
            kv_indices: torch.Tensor,
            kv_indptr: torch.Tensor,
            *,
            csa_topk_local: torch.Tensor | None = None,
            csa_block_tables: torch.Tensor | None = None,
            csa_block_capacity: int = 0,
        ) -> None:
            if use_split_kv_decode and unified_kv_scales is None:
                sparse_attn_v4_paged_decode_split_kv(
                    q,
                    split_swa_kv,
                    split_compressed_kv,
                    kv_indices,
                    kv_indptr,
                    self.attn_sink,
                    self.scale,
                    swa_pages=swa_pages,
                    compressed_kv_scales=split_kv_scales,
                    compressed_kv_layout=split_kv_layout,
                    out=output,
                    csa_topk_local=csa_topk_local,
                    csa_block_tables=csa_block_tables,
                    csa_positions=positions,
                    csa_batch_id_per_token=(
                        atom_state.batch_id_per_token
                        if csa_topk_local is not None
                        else None
                    ),
                    csa_block_capacity=csa_block_capacity,
                    csa_window_size=(self.window_size),
                )
                return
            if unified_kv is None:
                raise RuntimeError("ATOM decode fallback requires atom_unified_kv.")
            sparse_attn_v4_paged_decode(
                q,
                unified_kv,
                kv_indices,
                kv_indptr,
                self.attn_sink,
                self.scale,
                kv_scales=unified_kv_scales,
                out=output,
                kv_splits=1 if swa_only else None,
            )

        if q.shape[0] == 0:
            return True

        T = q.shape[0]
        buffers = atom_state.decode_buffers
        assert buffers is not None
        swa_indptr = buffers.swa_indptr[: T + 1]
        csa_indptr = buffers.csa_indptr[: T + 1]
        hca_indptr = buffers.hca_indptr[: T + 1]
        swa_indices = buffers.swa_indices
        csa_indices = buffers.csa_indices
        hca_indices = buffers.hca_indices
        cache = getattr(atom_state, "decode_cache", None)
        swa_block_table = swa_metadata.block_table
        common_indices_key = (
            int(T),
            int(atom_state.decode_swa_total),
            int(atom_state.decode_csa_total),
            int(atom_state.decode_hca_total),
            int(swa_block_table.data_ptr()),
            int(swa_block_table.storage_offset()),
            int(swa_block_table.stride(0)),
            int(swa_block_table.stride(1)),
            tuple(int(x) for x in swa_block_table.shape),
        )

        def ensure_decode_indices(
            *,
            hca_block_table: torch.Tensor | None,
            hca_block_capacity: int,
        ) -> None:
            write_hca_head = hca_block_table is not None
            allow_cache_hits = _atom_metadata_cache_hits_allowed()
            common_valid = bool(
                allow_cache_hits
                and cache is not None
                and cache.common_indices_key == common_indices_key
            )
            hca_key: tuple[object, ...] | None = None
            if write_hca_head:
                assert hca_block_table is not None
                hca_key = (
                    int(T),
                    int(atom_state.decode_hca_total),
                    int(hca_block_table.data_ptr()),
                    int(hca_block_table.storage_offset()),
                    int(hca_block_table.stride(0)),
                    int(hca_block_table.stride(1)) if hca_block_table.dim() > 1 else 1,
                    tuple(int(x) for x in hca_block_table.shape),
                    int(hca_block_capacity),
                    common_indices_key,
                )
            hca_valid = not write_hca_head or bool(
                allow_cache_hits
                and cache is not None
                and cache.hca_indices_key == hca_key
            )
            if common_valid and hca_valid:
                if cache is not None:
                    cache.common_indices_hits += 1
                    if write_hca_head:
                        cache.hca_indices_hits += 1
                return

            if common_valid and write_hca_head:
                assert hca_block_table is not None
                _write_hca_compress_head(
                    block_table=hca_block_table,
                    batch_id_per_token=atom_state.batch_id_per_token,
                    n_committed_hca_per_seq=atom_state.n_committed_hca_per_seq,
                    hca_indices=hca_indices,
                    hca_indptr=hca_indptr,
                    swa_pages=swa_pages,
                    hca_block_capacity=hca_block_capacity,
                    T=T,
                )
                if cache is not None:
                    cache.hca_indices_key = hca_key
                    cache.hca_indices_writes += 1
                return

            write_v4_paged_decode_indices(
                state_slot_per_seq=atom_state.state_slot_mapping,
                batch_id_per_token=atom_state.batch_id_per_token,
                positions=positions,
                swa_indptr=swa_indptr,
                csa_indptr=csa_indptr,
                hca_indptr=hca_indptr,
                swa_indices=swa_indices,
                csa_indices=csa_indices,
                hca_indices=hca_indices,
                swa_block_tables=swa_metadata.block_table,
                swa_block_size=swa_metadata.block_size,
                T=T,
                win=self.window_size,
                cs=int(atom_state.win_with_spec),
                max_pages=swa_pages,
                hca_block_table=hca_block_table,
                hca_n_committed_per_seq=(
                    atom_state.n_committed_hca_per_seq if write_hca_head else None
                ),
                hca_swa_pages=swa_pages,
                hca_block_capacity=hca_block_capacity,
            )
            if cache is not None:
                cache.common_indices_key = common_indices_key
                cache.common_indices_writes += 1
                if write_hca_head:
                    cache.hca_indices_key = hca_key
                    cache.hca_indices_writes += 1

        fused_hca_index = self.compress_ratio == 128 and attn_metadata is not None
        hca_block_capacity = (
            attn_metadata.block_size // self.compress_ratio if fused_hca_index else 0
        )
        ensure_decode_indices(
            hca_block_table=attn_metadata.block_table if fused_hca_index else None,
            hca_block_capacity=hca_block_capacity,
        )

        if swa_only:
            kv_indices = swa_indices
            kv_indptr = swa_indptr
        elif self.compress_ratio == 4:
            if attn_metadata is None or self.topk_indices_buffer is None:
                return False
            csa_block_capacity = attn_metadata.block_size // self.compress_ratio
            translate_key = (
                int(T),
                int(atom_state.decode_csa_total),
                int(csa_block_capacity),
                int(attn_metadata.block_table.data_ptr()),
                int(attn_metadata.block_table.storage_offset()),
                int(attn_metadata.block_table.stride(0)),
                int(attn_metadata.block_table.stride(1))
                if attn_metadata.block_table.dim() > 1
                else 1,
                tuple(int(x) for x in attn_metadata.block_table.shape),
                int(self.topk_indices_buffer.data_ptr()),
                int(self.topk_indices_buffer.storage_offset()),
                tuple(int(x) for x in self.topk_indices_buffer.shape),
                common_indices_key,
            )
            skip_translate = (
                _atom_metadata_cache_hits_allowed()
                and bool(getattr(self, "skip_topk", False))
                and cache is not None
                and cache.csa_translate_key == translate_key
            )
            if skip_translate:
                assert cache is not None
                cache.csa_translate_hits += 1
            else:
                csa_translate_pack(
                    self.topk_indices_buffer[:T],
                    attn_metadata.block_table,
                    positions,
                    csa_indptr,
                    atom_state.batch_id_per_token,
                    None,
                    csa_indices,
                    swa_pages=swa_pages,
                    csa_block_capacity=csa_block_capacity,
                    window_size=self.window_size,
                )
                if cache is not None:
                    cache.csa_translate_key = translate_key
                    cache.csa_translate_writes += 1
            kv_indices = csa_indices
            kv_indptr = csa_indptr
        elif self.compress_ratio == 128:
            if attn_metadata is None:
                return False
            if not fused_hca_index:
                _write_hca_compress_head(
                    block_table=attn_metadata.block_table,
                    batch_id_per_token=atom_state.batch_id_per_token,
                    n_committed_hca_per_seq=atom_state.n_committed_hca_per_seq,
                    hca_indices=hca_indices,
                    hca_indptr=hca_indptr,
                    swa_pages=swa_pages,
                    hca_block_capacity=attn_metadata.block_size // self.compress_ratio,
                    T=T,
                )
            kv_indices = hca_indices
            kv_indptr = hca_indptr
        else:
            return False
        run_paged_decode(kv_indices, kv_indptr)
        return True

    def _build_atom_prefill_indptrs(
        self,
        *,
        T: int,
        token_offset: int,
        atom_state,
        swa_only: bool,
    ) -> tuple[int, int, int, int]:
        buffers = atom_state.prefill_buffers
        assert buffers is not None
        cache = getattr(atom_state, "prefill_cache", None)
        cache_key = (int(T), int(token_offset), bool(swa_only))
        if (
            _atom_metadata_cache_hits_allowed()
            and cache is not None
            and cache.indptr_key == cache_key
        ):
            cache.indptr_hits += 1
            return cache.totals

        positions_cpu = atom_state.positions_cpu[token_offset : token_offset + T]
        batch_cpu = atom_state.batch_id_per_token_cpu[token_offset : token_offset + T]
        valid = batch_cpu >= 0
        safe_batch = np.where(valid, batch_cpu, 0)
        chunk_start = atom_state.chunk_start_per_seq_cpu[safe_batch]

        token_pos_in_chunk = positions_cpu - chunk_start
        extend_lens = np.where(
            valid,
            np.minimum(token_pos_in_chunk + 1, self.window_size),
            0,
        ).astype(np.int32, copy=False)

        swa_low = np.maximum(positions_cpu - self.window_size + 1, 0)
        prefix_swa_lens = np.where(
            valid,
            np.maximum(chunk_start - swa_low, 0),
            0,
        ).astype(np.int32, copy=False)

        csa_comp_lens = np.zeros(T, dtype=np.int32)
        hca_comp_lens = np.zeros(T, dtype=np.int32)
        if not swa_only:
            csa_committed = atom_state.n_committed_csa_per_seq_cpu[safe_batch]
            topk_tokens = (
                int(getattr(cache, "index_topk", 0) or 0)
                if cache is not None
                else int(
                    getattr(
                        getattr(self, "indexer", None),
                        "topk_tokens",
                        0,
                    )
                    or 0
                )
            )
            csa_comp_lens = np.where(
                valid,
                np.minimum(
                    np.minimum((positions_cpu + 1) // 4, csa_committed),
                    topk_tokens,
                ),
                0,
            ).astype(np.int32, copy=False)

            hca_committed = atom_state.n_committed_hca_per_seq_cpu[safe_batch]
            hca_comp_lens = np.where(
                valid,
                np.minimum((positions_cpu + 1) // 128, hca_committed),
                0,
            ).astype(np.int32, copy=False)

        buffers.extend_indptr_cpu[:1] = 0
        buffers.prefix_swa_indptr_cpu[:1] = 0
        buffers.prefix_csa_indptr_cpu[:1] = 0
        buffers.prefix_hca_indptr_cpu[:1] = 0
        np.cumsum(extend_lens, out=buffers.extend_indptr_cpu[1 : T + 1])
        np.cumsum(prefix_swa_lens, out=buffers.prefix_swa_indptr_cpu[1 : T + 1])
        np.cumsum(
            prefix_swa_lens + csa_comp_lens,
            out=buffers.prefix_csa_indptr_cpu[1 : T + 1],
        )
        np.cumsum(
            prefix_swa_lens + hca_comp_lens,
            out=buffers.prefix_hca_indptr_cpu[1 : T + 1],
        )
        buffers.skip_prefix_len_csa_cpu[:T] = prefix_swa_lens

        buffers.cu_q_per_seq_cpu[: atom_state.num_reqs] = 0
        if valid.any():
            valid_offsets = np.nonzero(valid)[0]
            valid_bids, first_idx = np.unique(batch_cpu[valid], return_index=True)
            buffers.cu_q_per_seq_cpu[valid_bids] = valid_offsets[first_idx].astype(
                np.int32,
            )

        extend_total = int(buffers.extend_indptr_cpu[T])
        prefix_swa_total = int(buffers.prefix_swa_indptr_cpu[T])
        prefix_csa_total = int(buffers.prefix_csa_indptr_cpu[T])
        prefix_hca_total = int(buffers.prefix_hca_indptr_cpu[T])
        if extend_total > buffers.max_extend_indices:
            raise RuntimeError(
                f"ATOM prefill extend index buffer too small: {extend_total} > "
                f"{buffers.max_extend_indices}."
            )
        if prefix_swa_total > buffers.max_prefix_swa_indices:
            raise RuntimeError(
                "ATOM prefill SWA prefix index buffer too small: "
                f"{prefix_swa_total} > {buffers.max_prefix_swa_indices}."
            )
        if prefix_csa_total > buffers.max_prefix_csa_indices:
            raise RuntimeError(
                "ATOM prefill CSA prefix index buffer too small: "
                f"{prefix_csa_total} > {buffers.max_prefix_csa_indices}."
            )
        if prefix_hca_total > buffers.max_prefix_hca_indices:
            raise RuntimeError(
                "ATOM prefill HCA prefix index buffer too small: "
                f"{prefix_hca_total} > {buffers.max_prefix_hca_indices}."
            )

        buffers.extend_indptr[: T + 1].copy_(
            buffers.extend_indptr_cpu_tensor[: T + 1],
            non_blocking=True,
        )
        buffers.prefix_swa_indptr[: T + 1].copy_(
            buffers.prefix_swa_indptr_cpu_tensor[: T + 1],
            non_blocking=True,
        )
        buffers.prefix_csa_indptr[: T + 1].copy_(
            buffers.prefix_csa_indptr_cpu_tensor[: T + 1],
            non_blocking=True,
        )
        buffers.prefix_hca_indptr[: T + 1].copy_(
            buffers.prefix_hca_indptr_cpu_tensor[: T + 1],
            non_blocking=True,
        )
        buffers.skip_prefix_len_csa[:T].copy_(
            buffers.skip_prefix_len_csa_cpu_tensor[:T],
            non_blocking=True,
        )
        buffers.cu_q_per_seq[: atom_state.num_reqs].copy_(
            buffers.cu_q_per_seq_cpu_tensor[: atom_state.num_reqs],
            non_blocking=True,
        )
        totals = (extend_total, prefix_swa_total, prefix_csa_total, prefix_hca_total)
        if cache is not None:
            cache.indptr_key = cache_key
            cache.totals = totals
            cache.common_indices_key = None
            cache.hca_indices_key = None
            cache.indptr_writes += 1
        return totals

    def _maybe_forward_prefill_atom(
        self,
        *,
        q: torch.Tensor,
        positions: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        swa_only: bool,
        token_offset: int,
    ) -> bool:
        if not _atom_attention_enabled_for_ratio(
            self.compress_ratio
        ) or not _atom_attention_enabled_for_layer(self._atom_layer_id):
            return False
        atom_state = get_deepseek_v4_rocm_atom_state(swa_metadata)
        kv_full = getattr(self, "_atom_last_kv", None)
        if atom_state is None or atom_state.prefill_buffers is None:
            return False
        kv_views = _resolve_atom_kv_views(self, atom_state)
        unified_kv = kv_views.unified_kv
        split_swa_kv = kv_views.split_swa_kv
        split_compressed_kv = kv_views.split_compressed_kv
        split_kv_scales = kv_views.split_kv_scales
        split_kv_layout = kv_views.split_kv_layout
        atom_swa_kv = split_swa_kv
        use_split_kv_prefill = (
            split_swa_kv is not None and split_compressed_kv is not None
        )
        if unified_kv is None and not use_split_kv_prefill:
            raise RuntimeError(
                "ROCm DeepSeek-V4 ATOM attention requires bound unified or "
                "split KV views."
            )
        if split_swa_kv is not None:
            swa_pages = split_swa_kv.numel() // self.head_dim
        else:
            swa_pages = int(atom_state.swa_pages)
        if q.shape[0] == 0:
            return True
        if kv_full is None:
            return False
        actual_t = min(
            int(q.shape[0]),
            max(0, int(atom_state.num_actual_tokens) - int(token_offset)),
            max(0, int(atom_state.num_tokens) - int(token_offset)),
        )
        if actual_t <= 0:
            output.zero_()
            return True
        if token_offset + actual_t > kv_full.shape[0]:
            return False

        T = actual_t
        q_actual = q[:T]
        positions_actual = positions[:T]
        if output.shape[0] > T:
            output[T:].zero_()
        buffers = atom_state.prefill_buffers
        assert buffers is not None
        (
            extend_total,
            prefix_swa_total,
            prefix_csa_total,
            prefix_hca_total,
        ) = self._build_atom_prefill_indptrs(
            T=T,
            token_offset=token_offset,
            atom_state=atom_state,
            swa_only=swa_only,
        )
        cache = getattr(atom_state, "prefill_cache", None)
        common_indices_key = (
            int(T),
            int(token_offset),
            int(extend_total),
            int(prefix_swa_total),
            int(prefix_csa_total),
            int(prefix_hca_total),
            int(swa_metadata.block_table.data_ptr()),
            int(swa_metadata.block_table.storage_offset()),
            int(swa_metadata.block_table.stride(0)),
            int(swa_metadata.block_table.stride(1)),
            tuple(int(x) for x in swa_metadata.block_table.shape),
        )

        def ensure_common_indices(
            block_tables: torch.Tensor,
            *,
            write_hca: bool,
        ) -> None:
            allow_cache_hits = _atom_metadata_cache_hits_allowed()
            common_valid = bool(
                allow_cache_hits
                and cache is not None
                and cache.common_indices_key == common_indices_key
            )
            hca_key = (
                int(T),
                int(token_offset),
                int(prefix_hca_total),
                int(block_tables.data_ptr()),
                int(block_tables.storage_offset()),
                int(block_tables.stride(0)),
                int(block_tables.stride(1)) if block_tables.dim() > 1 else 1,
                tuple(int(x) for x in block_tables.shape),
                common_indices_key,
            )
            hca_valid = bool(
                allow_cache_hits
                and cache is not None
                and cache.hca_indices_key == hca_key
            )
            if common_valid and (not write_hca or hca_valid):
                if cache is not None:
                    cache.common_indices_hits += 1
                    if write_hca:
                        cache.hca_indices_hits += 1
                return

            write_v4_paged_prefill_indices(
                positions=positions_actual,
                bid_per_token=atom_state.batch_id_per_token[token_offset:],
                chunk_start_per_seq=atom_state.chunk_start_per_seq,
                cu_seqlens_q_per_seq=buffers.cu_q_per_seq,
                state_slot_per_seq=atom_state.state_slot_mapping,
                n_committed_hca_per_seq=atom_state.n_committed_hca_per_seq,
                swa_block_tables=swa_metadata.block_table,
                swa_block_size=swa_metadata.block_size,
                block_tables=block_tables,
                extend_indptr=buffers.extend_indptr[: T + 1],
                prefix_swa_indptr=buffers.prefix_swa_indptr[: T + 1],
                prefix_csa_indptr=buffers.prefix_csa_indptr[: T + 1],
                prefix_hca_indptr=buffers.prefix_hca_indptr[: T + 1],
                extend_indices=buffers.extend_indices[:extend_total],
                prefix_swa_indices=buffers.prefix_swa_indices[:prefix_swa_total],
                prefix_csa_indices=buffers.prefix_csa_indices[:prefix_csa_total],
                prefix_hca_indices=buffers.prefix_hca_indices[:prefix_hca_total],
                T=T,
                win=self.window_size,
                cs=int(atom_state.win_with_spec),
                swa_pages=swa_pages,
                write_hca=write_hca,
            )
            if cache is not None:
                cache.common_indices_key = common_indices_key
                cache.common_indices_writes += 1
                if write_hca:
                    cache.hca_indices_key = hca_key
                    cache.hca_indices_writes += 1

        if swa_only:
            block_tables = swa_metadata.block_table
            ensure_common_indices(block_tables, write_hca=False)
            kv_indices_prefix = buffers.prefix_swa_indices[:prefix_swa_total]
            kv_indptr_prefix = buffers.prefix_swa_indptr[: T + 1]
        elif self.compress_ratio == 4:
            if attn_metadata is None or self.topk_indices_buffer is None:
                return False
            block_tables = attn_metadata.block_table
            ensure_common_indices(block_tables, write_hca=False)
            csa_block_capacity = attn_metadata.block_size // self.compress_ratio
            translate_key = (
                int(T),
                int(token_offset),
                int(prefix_csa_total),
                int(csa_block_capacity),
                int(block_tables.data_ptr()),
                int(block_tables.storage_offset()),
                int(block_tables.stride(0)) if block_tables.dim() > 0 else 1,
                int(block_tables.stride(1)) if block_tables.dim() > 1 else 1,
                tuple(int(x) for x in block_tables.shape),
                int(self.topk_indices_buffer.data_ptr()),
                int(self.topk_indices_buffer.storage_offset()),
                tuple(int(x) for x in self.topk_indices_buffer.shape),
                common_indices_key,
            )
            skip_translate = (
                _atom_metadata_cache_hits_allowed()
                and bool(getattr(self, "skip_topk", False))
                and cache is not None
                and cache.csa_translate_key == translate_key
            )
            if skip_translate:
                assert cache is not None
                cache.csa_translate_hits += 1
            else:
                csa_translate_pack(
                    self.topk_indices_buffer[token_offset : token_offset + T],
                    block_tables,
                    positions_actual,
                    buffers.prefix_csa_indptr[: T + 1],
                    atom_state.batch_id_per_token[token_offset:],
                    buffers.skip_prefix_len_csa[:T],
                    buffers.prefix_csa_indices,
                    swa_pages=int(atom_state.swa_pages),
                    csa_block_capacity=csa_block_capacity,
                    window_size=0,
                )
                if cache is not None:
                    cache.csa_translate_key = translate_key
                    cache.csa_translate_writes += 1
            kv_indices_prefix = buffers.prefix_csa_indices[:prefix_csa_total]
            kv_indptr_prefix = buffers.prefix_csa_indptr[: T + 1]
        elif self.compress_ratio == 128:
            if attn_metadata is None:
                return False
            block_tables = attn_metadata.block_table
            ensure_common_indices(block_tables, write_hca=True)
            kv_indices_prefix = buffers.prefix_hca_indices[:prefix_hca_total]
            kv_indptr_prefix = buffers.prefix_hca_indptr[: T + 1]
        else:
            return False

        kv_actual = kv_full[token_offset : token_offset + T].contiguous()
        if use_split_kv_prefill and unified_kv is None:
            sparse_attn_v4_paged_prefill_split_kv(
                q_actual,
                split_swa_kv,
                split_compressed_kv,
                kv_indices_prefix,
                kv_indptr_prefix,
                kv_actual,
                buffers.extend_indices[:extend_total],
                buffers.extend_indptr[: T + 1],
                self.attn_sink,
                self.scale,
                swa_pages=swa_pages,
                compressed_kv_scales=split_kv_scales,
                compressed_kv_layout=split_kv_layout,
                out=output[:T],
            )
        else:
            sparse_attn_v4_paged_prefill(
                q_actual,
                unified_kv,
                kv_indices_prefix,
                kv_indptr_prefix,
                kv_actual,
                buffers.extend_indices[:extend_total],
                buffers.extend_indptr[: T + 1],
                self.attn_sink,
                self.scale,
                out=output[:T],
            )

        if atom_swa_kv is not None and not getattr(
            self, "_atom_swa_write_fused", False
        ):
            write_per_batch = min(
                int(getattr(swa_metadata, "max_query_len", kv_full.shape[0]) or T),
                int(atom_state.win_with_spec),
            )
            if write_per_batch > 0 and atom_state.num_actual_reqs > 0:
                swa_write(
                    kv_full[: atom_state.num_actual_tokens],
                    atom_state.positions[: atom_state.num_actual_tokens],
                    atom_state.query_start_loc[: atom_state.num_actual_reqs + 1],
                    atom_state.state_slot_mapping[: atom_state.num_actual_reqs],
                    atom_swa_kv,
                    swa_metadata.block_size,
                    write_per_batch,
                    block_tables=swa_metadata.block_table,
                    block_size=swa_metadata.block_size,
                )
        return True

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> bool:
        swa_only = attn_metadata is None

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None

        query_start_loc_cpu = swa_metadata.query_start_loc_cpu
        query_start_loc = swa_metadata.query_start_loc
        assert query_start_loc_cpu is not None
        assert query_start_loc is not None
        prefill_token_base = query_start_loc_cpu[num_decodes]

        if self._maybe_forward_prefill_atom(
            q=q,
            positions=positions,
            output=output,
            attn_metadata=attn_metadata,
            swa_metadata=swa_metadata,
            swa_only=swa_only,
            token_offset=num_decode_tokens,
        ):
            return True

        if not swa_only:
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                topk_indices = self.topk_indices_buffer[num_decode_tokens:]
                topk_indices = topk_indices[:num_prefill_tokens]
            else:
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
            assert topk_indices is not None
            top_k = topk_indices.shape[-1]
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
            self.PREFILL_CHUNK_SIZE
        )

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            if not swa_only:
                assert attn_metadata is not None
                assert compressed_k_cache is not None
                block_table = attn_metadata.block_table[num_decodes:]
                # compressed_k_cache is OCP on every platform (Triton encoder).
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                    use_fnuz=False,
                )

            swa_block_table = swa_metadata.block_table[num_decodes:]
            _gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
                use_fnuz=current_platform.is_fp8_fnuz(),
            )

            query_start = (
                query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
            )
            query_end = (
                query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
            )

            combined_indices, combined_lens = combine_topk_swa_indices(
                topk_indices[query_start:query_end],
                query_start_loc[
                    num_decodes + chunk_start : num_decodes + chunk_end + 1
                ],
                seq_lens[chunk_start:chunk_end],
                gather_lens[chunk_start:chunk_end],
                self.window_size,
                self.compress_ratio,
                top_k,
                M,
                N,
            )
            rocm_sparse_attn_prefill(
                q=q[query_start:query_end],
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices,
                topk_length=combined_lens,
                scale=self.scale,
                head_dim=self.head_dim,
                nope_head_dim=self.nope_head_dim,
                rope_head_dim=self.rope_head_dim,
                attn_sink=self.attn_sink,
                output=output[query_start:query_end],
            )
        return False
