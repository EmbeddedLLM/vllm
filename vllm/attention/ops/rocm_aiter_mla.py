# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer


def get_aiter_mla_metadata(
    max_batch_size: int, block_size: int, max_block_per_batch: int, device: torch.device
) -> tuple[torch.Tensor, ...]:
    paged_kv_indices = torch.zeros(
        max_batch_size * max_block_per_batch, dtype=torch.int32, device=device
    )
    paged_kv_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int32, device=device)
    paged_kv_last_page_lens = torch.full(
        (max_batch_size,), block_size, dtype=torch.int32
    )
    qo_indptr = torch.zeros(max_batch_size + 1, dtype=torch.int, device=device)
    return paged_kv_indices, paged_kv_indptr, paged_kv_last_page_lens, qo_indptr


@triton.jit
def _fwd_kernel_stage2_reduced(
    Mid_O,
    Mid_lse,
    attn_out,
    lse_out,
    qo_indptr,
    kv_indptr,
    stride_mid_ob,
    stride_mid_oh,
    stride_mid_os,
    stride_obs,
    stride_oh,
    stride_eb,
    stride_eh,
    NUM_KV_SPLITS: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    Lv: tl.constexpr,
    mgc: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_qo_offs = tl.program_id(2)

    cur_qo_start = tl.load(qo_indptr + cur_batch)
    cur_qo_end = tl.load(qo_indptr + cur_batch + 1)
    cur_qo = cur_qo_start + cur_qo_offs
    if cur_qo > cur_qo_end:
        return
    cur_kv_seq_len = tl.load(kv_indptr + cur_batch + 1) - tl.load(kv_indptr + cur_batch)

    offs_d = tl.arange(0, BLOCK_DV)
    mask_d = offs_d < Lv

    e_sum = 0.0
    e_max = -float("inf")
    acc = tl.zeros([BLOCK_DV], dtype=tl.float32)

    offs_v = (cur_qo * stride_mid_ob + cur_head * stride_mid_oh) * Lv + offs_d
    offs_logic = cur_qo * stride_mid_ob + cur_head * stride_mid_oh

    for split_kv_id in range(0, NUM_KV_SPLITS):
        kv_len_per_split = tl.maximum(mgc, tl.cdiv(cur_kv_seq_len, NUM_KV_SPLITS))
        split_kv_start = kv_len_per_split * split_kv_id
        split_kv_end = tl.minimum(split_kv_start + kv_len_per_split, cur_kv_seq_len)

        if split_kv_end > split_kv_start:
            tv = tl.load(
                Mid_O + offs_v + split_kv_id * stride_mid_os * Lv,
                mask=mask_d,
                other=0.0,
            )
            tlogic = tl.load(Mid_lse + offs_logic + split_kv_id * stride_mid_os)
            n_e_max = tl.maximum(tlogic, e_max)

            old_scale = tl.exp(e_max - n_e_max)
            acc *= old_scale
            exp_logic = tl.exp(tlogic - n_e_max)
            acc += exp_logic * tv

            e_sum = e_sum * old_scale + exp_logic
            e_max = n_e_max

    tl.store(
        attn_out + cur_qo * stride_obs + cur_head * stride_oh + offs_d,
        acc / e_sum,
        mask=mask_d,
    )
    lse_val = e_max + tl.log(e_sum)
    tl.store(
        lse_out + cur_qo * stride_eb + cur_head * stride_eh,
        lse_val,
    )


def aiter_mla_decode_fwd(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    sm_scale: float,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    lse_out: torch.Tensor,
    kv_indptr: torch.Tensor | None = None,
    kv_indices: torch.Tensor | None = None,
    kv_last_page_lens: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    logit_cap: float = 0.0,
) -> tuple[torch.Tensor, ...]:
    from aiter.mla import get_meta_param

    device = q.device
    assert logit_cap <= 0, f"{logit_cap=} is not support yet"
    qk_head_dim = kv_buffer.shape[3]
    if sm_scale is None:
        sm_scale = 1.0 / (qk_head_dim**0.5)

    total_s, nhead, v_head_dim = o.shape
    bs = qo_indptr.shape[0] - 1
    total_kv = kv_indices.shape[0]

    num_kv_splits, mgc = get_meta_param(
        num_kv_splits, bs, total_kv, nhead, max_seqlen_qo
    )

    if nhead == 16 and max_seqlen_qo == 1:
        # special case for 16 heads and max_seqlen_q == 1
        logits = torch.empty(
            (total_s, num_kv_splits, nhead, v_head_dim),
            dtype=torch.float32,
            device=device,
        )
    elif nhead in [16, 128]:
        logits = (
            o.view((total_s, num_kv_splits, nhead, v_head_dim))
            if num_kv_splits == 1
            else torch.empty(
                (total_s, num_kv_splits, nhead, v_head_dim),
                dtype=torch.float32,
                device=device,
            )
        )
    else:
        raise ValueError(f"{nhead=} not supported")

    attn_lse = torch.empty(
        (total_s, num_kv_splits, nhead, 1), dtype=torch.float32, device=device
    )

    torch.ops.vllm.rocm_aiter_mla_decode_fwd_stage1(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        logits,
        attn_lse,
        max_seqlen_qo,
        sm_scale,
    )

    if num_kv_splits == 1:
        (
            o.copy_(logits.view(total_s, nhead, v_head_dim)),
            lse_out.copy_(attn_lse.view(total_s, nhead)),
        )

    Lv = v_head_dim
    BLOCK_DV = triton.next_power_of_2(Lv)
    grid = (bs, nhead, max_seqlen_qo)
    extra_kargs = {"waves_per_eu": 4, "matrix_instr_nonkdim": 16, "kpack": 2}

    _fwd_kernel_stage2_reduced[grid](
        logits,
        attn_lse,
        o,
        lse_out,
        qo_indptr,
        kv_indptr,
        attn_lse.stride(0),
        attn_lse.stride(2),
        attn_lse.stride(1),
        o.stride(0),
        o.stride(1),
        lse_out.stride(0),
        lse_out.stride(1),
        NUM_KV_SPLITS=num_kv_splits,
        BLOCK_DV=BLOCK_DV,
        Lv=Lv,
        mgc=mgc,
        num_warps=4,
        num_stages=2,
        **extra_kargs,
    )


def aiter_mla_decode_fwd_stage1_asm_impl(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    logits: torch.Tensor,
    attn_lse: torch.Tensor,
    max_seqlen_qo: int,
    sm_scale: float = 1.0,
) -> None:
    import aiter

    aiter.mla_decode_stage1_asm_fwd(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_qo,
        sm_scale,
        logits,
        attn_lse,
    )


def aiter_mla_decode_fwd_stage1_asm_fake(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_last_page_lens: torch.Tensor,
    logits: torch.Tensor,
    attn_lse: torch.Tensor,
    max_seqlen_qo: int,
    sm_scale: float = 1.0,
) -> None:
    pass


if current_platform.is_rocm():
    if is_torch_equal_or_newer("2.7.0"):
        tags = ()
    else:
        tags = ((torch.Tag.needs_fixed_stride_order,),)
    direct_register_custom_op(
        op_name="rocm_aiter_mla_decode_fwd_stage1",
        op_func=aiter_mla_decode_fwd_stage1_asm_impl,
        mutates_args=["logits", "attn_lse"],
        fake_impl=aiter_mla_decode_fwd_stage1_asm_fake,
        tags=tags,
    )
