# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch
import torch.nn.functional as F

from vllm.utils.torch_utils import direct_register_custom_op

_USE_AITER_MHC_BIG_FUSE = (
    os.environ.get("VLLM_ROCM_DSV4_USE_AITER_MHC_BIG_FUSE", "0") == "1"
)
_USE_AITER_MHC_LEGACY_OPS = (
    os.environ.get("VLLM_ROCM_DSV4_USE_AITER_MHC_LEGACY_OPS", "0") == "1"
)
_USE_AITER_MHC_LEGACY_PRE = (
    os.environ.get(
        "VLLM_ROCM_DSV4_USE_AITER_MHC_LEGACY_PRE",
        str(int(_USE_AITER_MHC_LEGACY_OPS)),
    )
    == "1"
)
_USE_AITER_MHC_LEGACY_POST = (
    os.environ.get(
        "VLLM_ROCM_DSV4_USE_AITER_MHC_LEGACY_POST",
        str(int(_USE_AITER_MHC_LEGACY_OPS)),
    )
    == "1"
)
_AITER_MHC_EVEN_ROW_WORKAROUND = (
    os.environ.get("VLLM_ROCM_DSV4_AITER_MHC_EVEN_ROW_WORKAROUND", "1") == "1"
)
_AITER_MHC_BIG_FUSE_MIN_TOKENS = int(
    os.environ.get("VLLM_ROCM_DSV4_AITER_MHC_BIG_FUSE_MIN_TOKENS", "0")
)
_AITER_MHC_PRE_MAX_SPLITK = int(
    os.environ.get("VLLM_ROCM_DSV4_AITER_MHC_PRE_MAX_SPLITK", "32")
)


def _cap_mhc_pre_splitk(
    selected_splitk: int,
    tile_k: int,
    hc_hidden_size: int,
) -> int:
    if _AITER_MHC_PRE_MAX_SPLITK <= 0:
        return selected_splitk
    max_splitk = min(selected_splitk, _AITER_MHC_PRE_MAX_SPLITK)
    for splitk in range(max_splitk, 0, -1):
        if hc_hidden_size % (splitk * tile_k) == 0 and (hc_hidden_size // splitk) >= (
            tile_k * 2
        ):
            return splitk
    return selected_splitk


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
    norm_eps: float = 0.0,
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
    if norm_weight is not None and norm_weight.dtype != torch.bfloat16:
        norm_weight = norm_weight.to(torch.bfloat16)

    if _USE_AITER_MHC_LEGACY_PRE and norm_weight is None:
        from vllm._aiter_ops import rocm_aiter_ops

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
        )

    if _USE_AITER_MHC_BIG_FUSE and (
        residual.shape[0] >= _AITER_MHC_BIG_FUSE_MIN_TOKENS
    ):
        import aiter

        if _AITER_MHC_EVEN_ROW_WORKAROUND:
            dup_shape = (residual.shape[0] * 2, *residual.shape[1:])
            residual_dup = torch.empty(
                dup_shape,
                dtype=residual.dtype,
                device=residual.device,
            )
            residual_dup[0::2].copy_(residual)
            residual_dup[1::2].copy_(residual)
            post_mix, comb_mix, layer_input = aiter.mhc_pre(
                residual_dup,
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
            return (
                post_mix[0::2].contiguous(),
                comb_mix[0::2].contiguous(),
                layer_input[0::2].contiguous(),
            )

        return aiter.mhc_pre(
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

    if _USE_AITER_MHC_BIG_FUSE:
        from vllm.model_executor.kernels.mhc.tilelang import mhc_pre_tilelang

        return mhc_pre_tilelang(
            residual,
            fn,
            hc_scale,
            hc_base,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            norm_weight,
            norm_eps,
        )

    from aiter.ops.mhc import get_mhc_pre_splitk, mhc_pre_gemm_sqrsum

    from vllm.model_executor.kernels.mhc.tilelang_kernels import (
        mhc_pre_big_fuse_tilelang,
    )

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]
    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    residual_2d = residual_flat.view(num_tokens, hc_hidden_size)
    selected_splitk, selected_tile_k = get_mhc_pre_splitk(num_tokens, hc_hidden_size)
    selected_splitk = _cap_mhc_pre_splitk(
        selected_splitk,
        selected_tile_k,
        hc_hidden_size,
    )

    # AITER's GEMM kernel writes a 32-column padded output tile even when
    # hc_mult3 is 24. Keep the padded allocation for its stores, then pass the
    # valid columns to the vLLM TileLang fuse stage.
    gemm_out_stride = (hc_mult3 + 31) // 32 * 32
    gemm_out_pad = torch.empty(
        selected_splitk,
        num_tokens,
        gemm_out_stride,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out = gemm_out_pad[:, :, :hc_mult3]
    gemm_out_sqrsum = torch.empty(
        selected_splitk,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    mhc_pre_gemm_sqrsum(
        gemm_out,
        gemm_out_sqrsum,
        residual_2d,
        fn,
        selected_tile_k,
    )

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    layer_input = torch.empty(
        num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
    )
    mhc_pre_big_fuse_tilelang(
        gemm_out.contiguous(),
        gemm_out_sqrsum,
        hc_scale,
        hc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
        hidden_size,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        selected_splitk,
        hc_mult,
    )
    if norm_weight is not None:
        layer_input.copy_(
            F.rms_norm(
                layer_input.float(),
                (hidden_size,),
                norm_weight.float(),
                norm_eps,
            ).to(torch.bfloat16)
        )
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
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
    norm_eps: float = 0.0,
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
    if _USE_AITER_MHC_LEGACY_POST:
        from vllm._aiter_ops import rocm_aiter_ops

        return rocm_aiter_ops.mhc_post(
            x,
            residual,
            post_layer_mix,
            comb_res_mix,
        )

    import aiter

    out = torch.empty_like(residual)
    aiter.mhc_post(
        out,
        x,
        residual,
        post_layer_mix,
        comb_res_mix,
    )
    return out


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
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ROCm AITER Triton fused MHC post+pre.

    The installed AITER package does not expose ATOM's top-level
    ``mhc_fused_post_pre`` symbol, but it does ship the underlying Triton
    fusion. It fuses the MHC post residual update and next MHC pre projection;
    if a caller asks for the sublayer RMSNorm, apply it after the fusion to
    match the TileLang op contract.
    """
    assert x.dtype == torch.bfloat16
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    from aiter.ops.triton.fusions.mhc import mhc_post_pre

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]
    num_tokens = residual.numel() // (hc_mult * hidden_size)

    x_flat = x.view(num_tokens, hidden_size)
    residual_flat = residual.view(num_tokens, hc_mult, hidden_size)
    post_flat = post_layer_mix.view(num_tokens, hc_mult)
    comb_flat = comb_res_mix.view(num_tokens, hc_mult, hc_mult)

    residual_cur = torch.empty_like(residual_flat)
    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        1,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    layer_input_cur = torch.empty_like(x_flat)

    post_mix_cur, comb_mix_cur, layer_input_cur, residual_cur = mhc_post_pre(
        x_flat,
        residual_flat,
        post_flat,
        comb_flat,
        fn.t(),
        hc_scale,
        hc_base,
        hc_mult,
        eps=rms_eps,
        hc_pre_eps=hc_pre_eps,
        hc_post_mult_value=hc_post_mult_value,
        sinkhorn_iters=sinkhorn_repeat,
        asymmetric_exp_domain=True,
        hc_sinkhorn_eps=hc_sinkhorn_eps,
        residual_out=residual_cur,
        h_post=post_mix_cur,
        h_res=comb_mix_cur,
        layer_input_out=layer_input_cur,
    )

    if norm_weight is not None:
        if norm_weight.dtype != torch.bfloat16:
            norm_weight = norm_weight.to(torch.bfloat16)
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
    tile_n: int = 1,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    hc_mult = residual.shape[-2]
    outer_shape = residual.shape[:-2]
    return (
        torch.empty_like(residual),
        torch.empty(
            *outer_shape,
            hc_mult,
            1,
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty(
            *outer_shape,
            hc_mult,
            hc_mult,
            dtype=torch.float32,
            device=residual.device,
        ),
        torch.empty_like(x),
    )


def hc_head_aiter(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    """ATOM-style HC head reduction through AITER ``mhc_pre``.

    For the final HC head, DeepSeek-V4 uses only the pre gate with
    ``sinkhorn_repeat=0``. AITER's public ``mhc_pre`` wrapper supports the
    compact ``fn.shape[0] == hc_mult`` form used by the ATOM model.
    """
    assert hs_flat.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32
    hidden_size = hs_flat.shape[-1]
    assert hidden_size % 256 == 0

    from aiter.ops.mhc import mhc_pre

    with torch.device(hs_flat.device):
        _, _, layer_input = mhc_pre(
            hs_flat,
            fn,
            hc_scale,
            hc_base,
            rms_norm_eps,
            hc_eps,
            sinkhorn_repeat=0,
        )
    return layer_input


def _hc_head_aiter_fake(
    hs_flat: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_norm_eps: float,
    hc_eps: float,
) -> torch.Tensor:
    return torch.empty(
        hs_flat.shape[0],
        hs_flat.shape[-1],
        dtype=torch.bfloat16,
        device=hs_flat.device,
    )


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
direct_register_custom_op(
    op_name="hc_head_aiter",
    op_func=hc_head_aiter,
    mutates_args=[],
    fake_impl=_hc_head_aiter_fake,
)
