# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op


def mhc_pre_aiter(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    hidden_size = residual.shape[-1]
    assert hidden_size % 256 == 0
    from vllm._aiter_ops import rocm_aiter_ops

    if norm_weight is not None:
        # ATOM constructs DeepSeek-V4 under a bf16 default dtype, so its RMSNorm
        # weights reach aiter as bf16. The installed aiter fused-norm MHC path
        # produces NaNs if this argument is fp32.
        if norm_weight.dtype != residual.dtype:
            norm_weight = norm_weight.to(residual.dtype)
        if not norm_weight.is_contiguous():
            norm_weight = norm_weight.contiguous()

    return rocm_aiter_ops.mhc_pre(
        residual,
        fn,
        hc_scale,
        hc_base,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        norm_weight,
        norm_eps,
    )


def _mhc_pre_aiter_fake(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    # Create empty tensors with correct shapes for meta device / shape inference
    post_mix = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return post_mix, comb_mix, layer_input


def mhc_post_aiter(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    hidden_size = residual.shape[-1]

    assert hidden_size % 256 == 0
    from vllm._aiter_ops import rocm_aiter_ops

    return rocm_aiter_ops.mhc_post(
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )


def _mhc_post_aiter_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(residual)


def mhc_fused_post_pre_aiter(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused MHC post + pre using aiter's Triton fusion.

    vLLM stores ``fn`` as (mix_hc, hc_mult * hidden_size), while the aiter
    Triton fusion consumes the transposed K-major view (hc_mult * hidden_size,
    mix_hc). The non-contiguous transpose has the layout expected by the
    kernel and avoids a per-call materialization.
    """

    from aiter.ops.triton.fusions.mhc import mhc_post_pre

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    x_flat = x.view(num_tokens, hidden_size)
    post_mix_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_mix_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens, hc_mult, 1, dtype=torch.float32, device=residual.device
    )
    comb_mix_cur = torch.empty(
        num_tokens, hc_mult, hc_mult, dtype=torch.float32, device=residual.device
    )
    layer_input_cur = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )

    use_hip_domain = (
        os.environ.get("ATOM_USE_HIP_DOMAIN_AITER_TRITON_MHC_FUSED_POST_PRE", "0")
        == "1"
    )
    post_mix_cur, comb_mix_cur, layer_input_cur, residual_cur = mhc_post_pre(
        x_flat,
        residual_flat,
        post_mix_flat,
        comb_mix_flat,
        fn.t(),
        hc_scale,
        hc_base,
        hc_mult,
        eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_iters=sinkhorn_repeat,
        asymmetric_exp_domain=use_hip_domain,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        residual_out=residual_cur,
        h_post=post_mix_cur,
        h_res=comb_mix_cur,
        layer_input_out=layer_input_cur,
    )
    if norm_weight is not None:
        layer_input_cur.copy_(
            F.rms_norm(
                layer_input_cur.float(),
                (hidden_size,),
                norm_weight.float(),
                norm_eps,
            ).to(torch.bfloat16)
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def _mhc_fused_post_pre_aiter_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    residual_cur = torch.empty_like(residual)
    post_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        *outer_shape,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty(
        *outer_shape,
        hidden_size,
        dtype=torch.bfloat16,
        device=residual.device,
    )

    return residual_cur, post_mix_cur, comb_mix_cur, layer_input_cur


direct_register_custom_op(
    op_name="mhc_pre_aiter",
    op_func=mhc_pre_aiter,
    mutates_args=[],
    fake_impl=_mhc_pre_aiter_fake,
)
direct_register_custom_op(
    op_name="mhc_post_aiter",
    op_func=mhc_post_aiter,
    mutates_args=[],
    fake_impl=_mhc_post_aiter_fake,
)
direct_register_custom_op(
    op_name="mhc_fused_post_pre_aiter",
    op_func=mhc_fused_post_pre_aiter,
    mutates_args=[],
    fake_impl=_mhc_fused_post_pre_aiter_fake,
)
