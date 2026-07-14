# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""V4 paged-decode index scatter — single Triton kernel writes SWA window-
prefix paged offsets into the three ragged-packed destination buffers
(`kv_indices_swa` / `kv_indices_csa` / `kv_indices_hca`).

Each absolute position is translated through the SWA block table inside the
kernel; no `[T, win]` staging buffer or CPU build + H2D copy is required.

Layout: ragged-packed. Each token's slice holds an SWA prefix of length
`n = min(positions[t]+1, win)` plus a per-buffer compress section; the
`swa_indptr` / `csa_indptr` / `hca_indptr` cumsums reflect this ragged
sizing. Within each token's slice the SWA prefix is written at the TAIL
(`[indptr[t+1] - n, indptr[t+1])`) and the compress section (CSA topk /
HCA committed) occupies the head.

Caller contract:
- Grid = T (one program per token).
- `batch_id_per_token[:T]` may carry `-1` sentinels in the CG-padded tail —
  kernel checks and bails (matches `_attach_v4_per_fwd_meta` convention).
- `swa_indptr` / `csa_indptr` / `hca_indptr` must reflect the ragged-packed
  sizing: per-token slot count = `min(positions[t]+1, win) + n_compress[t]`
  where `n_compress[t]` is 0 for SWA, `min(n_committed_csa, index_topk)`
  for CSA, `n_committed_hca` for HCA.
- `swa_indices` / `csa_indices` / `hca_indices` capacity ≥ corresponding
  indptr[T]; this kernel only writes the SWA-prefix segment at the slice
  tail `[indptr[t+1] - n, indptr[t+1])` per token. The compress section is
  filled elsewhere unless optional HCA-head fill arguments are supplied
  (CSA remains `csa_translate_pack` per layer).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _v4_paged_decode_indices_kernel(
    state_slot_per_seq_ptr,  # [bs] int32
    batch_id_per_token_ptr,  # [T+pad] int — sentinel -1 in pad tail
    positions_ptr,  # [T+pad] int — global token position
    swa_indptr_ptr,  # [T+1] int32 — ragged SWA-prefix cumsum
    csa_indptr_ptr,  # [T+1] int32 — ragged (SWA + CSA topk)
    hca_indptr_ptr,  # [T+1] int32 — ragged (SWA + HCA committed)
    swa_indices_ptr,  # [swa_total] int32, output
    csa_indices_ptr,  # [csa_total] int32, output (writes SWA-prefix segment only)
    hca_indices_ptr,  # [hca_total] int32, output (writes SWA-prefix segment only)
    swa_block_tables_ptr,
    swa_block_tables_stride,
    cs,  # kept for the ring-compatible public ABI
    num_reqs,
    max_pages,
    max_swa_indices,
    max_csa_indices,
    max_hca_indices,
    hca_block_table_ptr,
    hca_n_committed_per_seq_ptr,
    hca_block_table_stride,
    hca_swa_pages,
    hca_block_capacity: tl.constexpr,
    win: tl.constexpr,  # window_size — max SWA prefix slots
    SWA_BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,  # next_pow2(win)
    BLOCK_HCA: tl.constexpr,
    WRITE_HCA_HEAD: tl.constexpr,
):
    """One program per token. Writes `n = min(positions[t]+1, win)` paged
    offsets to the SWA prefix segment, placed at the TAIL of each token's
    slice in the SWA/CSA/HCA index buffers (the compress section occupies
    the head).

    For token `t`:
        bid = batch_id_per_token[t]                  # bail if -1 (CG pad)
        slot = state_slot_per_seq[bid]
        pos = positions[t]
        n = min(pos + 1, win)
        # i in [0, n) → abs_pos = pos - n + 1 + i ∈ [0, pos]; written at the
        # slice tail (indptr[t+1] - n) so the compress section fills the head.
        for i in range(n):
            abs_pos = pos - n + 1 + i
            ring = abs_pos % cs
            paged = slot * cs + ring
            swa_indices[swa_indptr[t+1] - n + i] = paged
            csa_indices[csa_indptr[t+1] - n + i] = paged
            hca_indices[hca_indptr[t+1] - n + i] = paged
        # Optional HCA head fill:
        #   n_hca = n_committed_hca_per_seq[bid]
        #   hca_indices[hca_indptr[t] + j] =
        #       hca_swa_pages + block_table[bid, j // hca_block_capacity]
        #                      * hca_block_capacity
        #                    + (j % hca_block_capacity)
    """
    t = tl.program_id(0)
    bid = tl.load(batch_id_per_token_ptr + t)
    if bid < 0:
        return  # CG-padded sentinel — leave outputs untouched
    if bid >= num_reqs:
        return  # CG-padded sentinel — leave outputs untouched

    pos = tl.load(positions_ptr + t)
    if pos < 0:
        return
    # `n` = actual valid SWA prefix count. Cast to match `win` (compile-time
    # int) — pos is i32/i64 from positions buffer.
    n = tl.minimum(pos + 1, win)
    # SWA prefix segment lives at the TAIL of each token's slice (compress
    # section fills the head). Write base = slice END (indptr[t+1]) - n. For
    # the SWA buffer (compress=0) end-n == indptr[t], same as a head write.
    swa_end = tl.load(swa_indptr_ptr + t + 1)
    csa_end = tl.load(csa_indptr_ptr + t + 1)
    hca_end = tl.load(hca_indptr_ptr + t + 1)
    swa_start = swa_end - n
    csa_start = csa_end - n
    hca_start = hca_end - n
    swa_slice_valid = (swa_start >= 0) & (swa_end <= max_swa_indices)
    csa_slice_valid = (csa_start >= 0) & (csa_end <= max_csa_indices)
    hca_slice_valid = (hca_start >= 0) & (hca_end <= max_hca_indices)
    safe_swa_start = tl.maximum(swa_start, 0)
    safe_csa_start = tl.maximum(csa_start, 0)
    safe_hca_start = tl.maximum(hca_start, 0)

    i = tl.arange(0, BLOCK_N)
    mask = i < n
    abs_pos = pos - n + 1 + i  # ∈ [0, pos] for valid i
    logical_blocks = abs_pos // SWA_BLOCK_SIZE
    physical_blocks = tl.load(
        swa_block_tables_ptr + bid * swa_block_tables_stride + logical_blocks,
        mask=mask,
        other=-1,
    )
    paged = physical_blocks * SWA_BLOCK_SIZE + abs_pos % SWA_BLOCK_SIZE
    page_valid = (physical_blocks >= 0) & (paged < max_pages)

    tl.store(
        swa_indices_ptr + safe_swa_start + i,
        paged,
        mask=mask & page_valid & swa_slice_valid,
    )
    tl.store(
        csa_indices_ptr + safe_csa_start + i,
        paged,
        mask=mask & page_valid & csa_slice_valid,
    )
    tl.store(
        hca_indices_ptr + safe_hca_start + i,
        paged,
        mask=mask & page_valid & hca_slice_valid,
    )

    if WRITE_HCA_HEAD:
        hca_base = tl.load(hca_indptr_ptr + t)
        hca_head_end = hca_start
        hca_head_capacity = tl.maximum(hca_head_end - hca_base, 0)
        n_hca = tl.load(hca_n_committed_per_seq_ptr + bid)
        block_base = bid * hca_block_table_stride
        hca_offsets = tl.arange(0, BLOCK_HCA)
        for j in tl.range(0, n_hca, BLOCK_HCA):
            offs = j + hca_offsets
            hca_mask = (offs < n_hca) & (offs < hca_head_capacity)
            block_offsets = offs // hca_block_capacity
            slot_offsets = offs - block_offsets * hca_block_capacity
            physical_blocks = tl.load(
                hca_block_table_ptr + block_base + block_offsets,
                mask=hca_mask,
                other=-1,
            )
            hca_mask = hca_mask & (physical_blocks >= 0)
            tl.store(
                hca_indices_ptr + hca_base + offs,
                hca_swa_pages + physical_blocks * hca_block_capacity + slot_offsets,
                mask=hca_mask & hca_slice_valid,
            )


def write_v4_paged_decode_indices(
    *,
    state_slot_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    swa_indptr: torch.Tensor,
    csa_indptr: torch.Tensor,
    hca_indptr: torch.Tensor,
    swa_indices: torch.Tensor,
    csa_indices: torch.Tensor,
    hca_indices: torch.Tensor,
    swa_block_tables: torch.Tensor,
    swa_block_size: int,
    T: int,
    win: int,
    cs: int,
    max_pages: int | None = None,
    hca_block_table: torch.Tensor | None = None,
    hca_n_committed_per_seq: torch.Tensor | None = None,
    hca_swa_pages: int = 0,
    hca_block_capacity: int = 0,
) -> None:
    """In-place fill SWA / CSA / HCA window-prefix offsets via a single
    Triton kernel. Replaces the prior `_build_window_topk_np` (CPU O(T·win))
    + `index_copy_` chain. All inputs are persistent forward_vars buffers —
    no allocator churn.

    Args (all GPU tensors except T/win/cs):
      state_slot_per_seq:  [bs]   int32 — per-seq state cache slot.
      batch_id_per_token:  [>=T]  int   — token→seq map; -1 sentinel skipped.
      positions:           [>=T]  int   — global token position
                                   (forward_vars["positions"]); used to derive
                                   `n = min(pos+1, win)` per token + the ring
                                   index `(pos - n + 1 + i) % cs`.
      swa_indptr:          [>=T+1] int32 — ragged SWA-prefix cumsum, where
                                   `swa_indptr[t+1] - swa_indptr[t] =
                                    min(positions[t]+1, win)`.
      csa_indptr:          [>=T+1] int32 — ragged CSA buffer indptr (SWA
                                   prefix + CSA topk per token).
      hca_indptr:          [>=T+1] int32 — ragged HCA buffer indptr (SWA
                                   prefix + HCA committed per token).
      swa_indices:         [>=swa_indptr[T]] int32 OUT — fully written by
                                   this kernel (no other source).
      csa_indices:         [>=csa_indptr[T]] int32 OUT — SWA prefix written
                                   here at the slice tail
                                   `[csa_indptr[t+1] - n, csa_indptr[t+1])`;
                                   CSA topk section (slice head) filled
                                   per-layer by `csa_translate_pack`.
      hca_indices:         [>=hca_indptr[T]] int32 OUT — same semantics; HCA
                                   compress section (slice head) is filled by
                                   this kernel when HCA args are present, or
                                   by the caller otherwise.
      T:                   int — number of real tokens (grid size).
      win:                 int — SWA window size (typically 128 for V4-Pro).
      cs:                  int — `win_with_spec = window_size + max_spec_steps`,
                                 stride into unified_kv SWA region per slot
                                 AND modulo for ring-index wrap.
      max_pages:           optional int — flattened row capacity of the paged
                                 SWA tensor.
      hca_block_table / hca_n_committed_per_seq / hca_swa_pages /
      hca_block_capacity:
                                 optional ATOM HCA committed-head fill. When
                                 provided, this kernel also writes the HCA
                                 compressed prefix before the SWA tail, matching
                                 `_write_hca_compress_head` semantics.
    """
    if T == 0:
        return
    assert state_slot_per_seq.dim() == 1
    assert batch_id_per_token.dim() == 1 and batch_id_per_token.shape[0] >= T
    assert positions.dim() == 1 and positions.shape[0] >= T
    assert swa_indptr.dim() == 1 and swa_indptr.shape[0] >= T + 1
    assert csa_indptr.dim() == 1 and csa_indptr.shape[0] >= T + 1
    assert hca_indptr.dim() == 1 and hca_indptr.shape[0] >= T + 1
    assert swa_indices.dim() == 1
    assert csa_indices.dim() == 1
    assert hca_indices.dim() == 1
    assert swa_block_tables.dim() == 2
    assert swa_block_tables.dtype == torch.int32

    BLOCK_N = triton.next_power_of_2(win)
    page_capacity = (
        int(max_pages)
        if max_pages is not None
        else int(state_slot_per_seq.shape[0] * cs)
    )
    write_hca_head = hca_block_table is not None and hca_n_committed_per_seq is not None
    if write_hca_head and hca_block_capacity <= 0:
        raise RuntimeError(
            f"Invalid HCA block capacity for ATOM decode: {hca_block_capacity}."
        )
    hca_block_table_arg = (
        hca_block_table if hca_block_table is not None else hca_indices
    )
    hca_n_committed_arg = (
        hca_n_committed_per_seq
        if hca_n_committed_per_seq is not None
        else state_slot_per_seq
    )
    hca_block_table_stride = (
        hca_block_table_arg.stride(0) if hca_block_table is not None else 0
    )
    _v4_paged_decode_indices_kernel[(T,)](
        state_slot_per_seq,
        batch_id_per_token,
        positions,
        swa_indptr,
        csa_indptr,
        hca_indptr,
        swa_indices,
        csa_indices,
        hca_indices,
        swa_block_tables,
        swa_block_tables.stride(0),
        cs,
        state_slot_per_seq.shape[0],
        page_capacity,
        swa_indices.shape[0],
        csa_indices.shape[0],
        hca_indices.shape[0],
        hca_block_table_arg,
        hca_n_committed_arg,
        hca_block_table_stride,
        int(hca_swa_pages),
        hca_block_capacity=max(1, int(hca_block_capacity)),
        win=win,
        SWA_BLOCK_SIZE=swa_block_size,
        BLOCK_N=BLOCK_N,
        BLOCK_HCA=128,
        WRITE_HCA_HEAD=write_hca_head,
    )


def write_v4_paged_decode_indices_reference(
    *,
    state_slot_per_seq: torch.Tensor,
    batch_id_per_token: torch.Tensor,
    positions: torch.Tensor,
    swa_indptr: torch.Tensor,
    csa_indptr: torch.Tensor,
    hca_indptr: torch.Tensor,
    swa_indices: torch.Tensor,
    csa_indices: torch.Tensor,
    hca_indices: torch.Tensor,
    swa_block_tables: torch.Tensor,
    swa_block_size: int,
    T: int,
    win: int,
    cs: int,
    max_pages: int | None = None,
    hca_block_table: torch.Tensor | None = None,
    hca_n_committed_per_seq: torch.Tensor | None = None,
    hca_swa_pages: int = 0,
    hca_block_capacity: int = 0,
) -> None:
    """Pure-PyTorch reference equivalent of `write_v4_paged_decode_indices`.
    For unit tests and bisect verification. Mirrors the kernel exactly:
    per-token ragged-packed write, no -1 sentinels in output.
    """
    if T == 0:
        return
    page_capacity = (
        int(max_pages)
        if max_pages is not None
        else int(state_slot_per_seq.shape[0] * cs)
    )
    bid = batch_id_per_token[:T].long()
    pos_t = positions[:T].long()
    valid = (bid >= 0) & (bid < state_slot_per_seq.shape[0])
    # n = min(pos+1, win) per token; clamp invalid rows to 0 to skip writes.
    n_per_tok = torch.minimum(pos_t + 1, torch.full_like(pos_t, win))
    n_per_tok = torch.where(valid, n_per_tok, torch.zeros_like(n_per_tok))
    for t in range(T):
        n = int(n_per_tok[t].item())
        if n == 0:
            continue
        p = int(pos_t[t].item())
        i_arr = torch.arange(n, device=positions.device, dtype=torch.long)
        abs_pos = p - n + 1 + i_arr  # [n]
        logical_blocks = torch.div(abs_pos, swa_block_size, rounding_mode="floor")
        physical_blocks = swa_block_tables[int(bid[t].item()), logical_blocks]
        paged = (physical_blocks * swa_block_size + abs_pos % swa_block_size).to(
            torch.int32
        )
        if page_capacity > 0:
            page_valid = paged.to(torch.long) < page_capacity
            if not bool(page_valid.all().item()):
                paged = paged[page_valid]
                n = int(paged.numel())
                if n == 0:
                    continue
        # SWA prefix segment at the slice TAIL (compress section fills the head).
        swa_end = int(swa_indptr[t + 1].item())
        csa_end = int(csa_indptr[t + 1].item())
        hca_end = int(hca_indptr[t + 1].item())
        swa_indices[swa_end - n : swa_end] = paged
        csa_indices[csa_end - n : csa_end] = paged
        hca_indices[hca_end - n : hca_end] = paged
        if hca_block_table is None or hca_n_committed_per_seq is None:
            continue
        if hca_block_capacity <= 0:
            raise RuntimeError(
                f"Invalid HCA block capacity for ATOM decode: {hca_block_capacity}."
            )
        n_hca = int(hca_n_committed_per_seq[int(bid[t].item())].item())
        hca_base = int(hca_indptr[t].item())
        hca_head_capacity = max(0, hca_end - n - hca_base)
        n_hca = min(n_hca, hca_head_capacity)
        if n_hca == 0:
            continue
        hca_offsets = torch.arange(n_hca, device=positions.device, dtype=torch.long)
        block_offsets = hca_offsets // hca_block_capacity
        slot_offsets = hca_offsets - block_offsets * hca_block_capacity
        physical_blocks = hca_block_table[int(bid[t].item()), block_offsets].long()
        hca_indices[hca_base : hca_base + n_hca] = (
            int(hca_swa_pages)
            + physical_blocks * int(hca_block_capacity)
            + slot_offsets
        ).to(torch.int32)
