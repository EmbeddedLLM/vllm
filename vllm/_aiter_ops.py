# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import wraps
from typing import Callable, Optional

import torch

import vllm.envs as envs
from vllm.platforms import current_platform
from vllm.utils import direct_register_custom_op, is_torch_equal_or_newer


def is_aiter_supported(func: Callable) -> Callable:
    """Decorator that only executes the function if 
    ROCm AITER package is supported on gfx9 archs.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # checks the platform, device arch and aiter library existance.
        from importlib.util import find_spec

        from vllm.platforms.rocm import on_gfx9

        if (current_platform.is_rocm() and on_gfx9()
                and find_spec("aiter") is not None):
            return func(*args, **kwargs)
        else:
            # Return None or do nothing if not supported
            return None

    return wrapper


def _rocm_aiter_fused_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation_method: int = 0,
    quant_method: int = 0,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe

    activation = ActivationType(activation_method)
    quant_type = QuantType(quant_method)

    return fused_moe(hidden_states, w1, w2, topk_weight, topk_ids, expert_mask,
                     activation, quant_type, doweight_stage1, w1_scale,
                     w2_scale, a1_scale, a2_scale)


def _rocm_aiter_fused_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation_method: int = 0,
    quant_method: int = 0,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _rocm_aiter_asm_moe_tkw1_impl(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        fc1_scale: Optional[torch.Tensor] = None,
        fc2_scale: Optional[torch.Tensor] = None,
        fc1_smooth_scale: Optional[torch.Tensor] = None,
        fc2_smooth_scale: Optional[torch.Tensor] = None,
        a16: bool = False,
        per_tensor_quant_scale: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        activation_method: int = 0) -> torch.Tensor:

    from aiter import ActivationType
    from aiter.fused_moe_bf16_asm import asm_moe_tkw1

    activation = ActivationType(activation_method)

    return asm_moe_tkw1(hidden_states,
                        w1,
                        w2,
                        topk_weights,
                        topk_ids,
                        fc1_scale=fc1_scale,
                        fc2_scale=fc2_scale,
                        fc1_smooth_scale=fc1_smooth_scale,
                        fc2_smooth_scale=fc2_smooth_scale,
                        a16=a16,
                        per_tensor_quant_scale=per_tensor_quant_scale,
                        expert_mask=expert_mask,
                        activation=activation)


def _rocm_aiter_asm_moe_tkw1_fake(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        fc1_scale: Optional[torch.Tensor] = None,
        fc2_scale: Optional[torch.Tensor] = None,
        fc1_smooth_scale: Optional[torch.Tensor] = None,
        fc2_smooth_scale: Optional[torch.Tensor] = None,
        a16: bool = False,
        per_tensor_quant_scale: Optional[torch.Tensor] = None,
        expert_mask: Optional[torch.Tensor] = None,
        activation_method: int = 0) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _rocm_aiter_topk_softmax_impl(topk_weights: torch.Tensor,
                                  topk_indices: torch.Tensor,
                                  token_expert_indices: torch.Tensor,
                                  gating_output: torch.Tensor,
                                  renormalize: bool) -> None:
    from aiter import topk_softmax
    topk_softmax(topk_weights, topk_indices, token_expert_indices,
                 gating_output, renormalize)


def _rocm_aiter_topk_softmax_fake(topk_weights: torch.Tensor,
                                  topk_indices: torch.Tensor,
                                  token_expert_indices: torch.Tensor,
                                  gating_output: torch.Tensor,
                                  renormalize: bool) -> None:
    pass


def _rocm_aiter_biased_grouped_topk_impl(
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        routed_scaling_factor: float = 1.0  # mul to topk_weights
) -> None:

    from aiter import biased_grouped_topk

    biased_grouped_topk(gating_output, correction_bias, topk_weights, topk_ids,
                        num_expert_group, topk_group, need_renorm,
                        routed_scaling_factor)


def _rocm_aiter_biased_grouped_topk_fake(
        gating_output: torch.Tensor,
        correction_bias: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        routed_scaling_factor: float = 1.0  # mul to topk_weights
) -> None:
    pass


def _rocm_aiter_grouped_topk_impl(
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0  # mul to topk_weights
) -> None:

    from aiter import grouped_topk

    grouped_topk(gating_output, topk_weights, topk_ids, num_expert_group,
                 topk_group, need_renorm, scoring_func, routed_scaling_factor)


def _rocm_aiter_grouped_topk_fake(
        gating_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_expert_group: int,
        topk_group: int,
        need_renorm: bool,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0  # mul to topk_weights
) -> None:
    pass


def _rocm_aiter_mla_decode_fwd_impl(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    logit_cap: float = 0.0,
) -> None:
    from aiter.mla import mla_decode_fwd

    mla_decode_fwd(q,
                   kv_buffer.view(-1, 1, 1, q.shape[-1]),
                   o,
                   qo_indptr,
                   kv_indptr,
                   kv_indices,
                   kv_last_page_lens,
                   max_seqlen_qo,
                   sm_scale=sm_scale,
                   logit_cap=logit_cap)


def _rocm_aiter_mla_decode_fwd_fake(
    q: torch.Tensor,
    kv_buffer: torch.Tensor,
    o: torch.Tensor,
    qo_indptr: torch.Tensor,
    max_seqlen_qo: int,
    kv_indptr: Optional[torch.Tensor] = None,
    kv_indices: Optional[torch.Tensor] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    sm_scale: float = 1.0,
    logit_cap: float = 0.0,
) -> None:
    pass


def _rocm_aiter_gemm_w8a8_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:

    from aiter import gemm_a8w8_CK

    # gemm_a8w8_CK(a, b, scale_a, scale_b, bias) expects
    # a to be [M, K]
    # b to be [N, K]
    # CutlassScaledMMLinearKernel prepare weight `w_q` in [K, N] format
    return gemm_a8w8_CK(A, B, As, Bs, bias, output_dtype)


def _rocm_aiter_gemm_w8a8_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:

    m = A.shape[0]
    n = B.shape[0]
    Y = torch.empty(m, n, dtype=output_dtype, device=A.device)
    return Y


def _rocm_aiter_gemm_w8a8_blockscale_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    from aiter import gemm_a8w8_blockscale

    return gemm_a8w8_blockscale(A, B, As, Bs, dtype=output_dtype)


def _rocm_aiter_gemm_w8a8_blockscale_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:

    m = A.shape[0]
    n = B.shape[0]
    Y = torch.empty(m, n, dtype=output_dtype, device=A.device)
    return Y


def _rocm_aiter_rms_norm_impl(x: torch.Tensor, weight: torch.Tensor,
                              variance_epsilon: float) -> torch.Tensor:
    from aiter import rms_norm
    if x.dim() > 2:
        x_original_shape = x.shape
        x = x.reshape(-1, x_original_shape[-1])
        x = rms_norm(x, weight, variance_epsilon)
        return x.reshape(x_original_shape)

    return rms_norm(x, weight, variance_epsilon)


def _rocm_aiter_rms_norm_fake(x: torch.Tensor, weight: torch.Tensor,
                              variance_epsilon: float) -> torch.Tensor:
    return torch.empty_like(x)


def _rocm_aiter_rmsnorm2d_fwd_with_add_impl(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:

    from aiter import rmsnorm2d_fwd_with_add

    residual_out = torch.empty_like(residual)
    output = torch.empty_like(x)
    rmsnorm2d_fwd_with_add(
        output,  # output
        x,  # input
        residual,  # residual input
        residual_out,  # residual output
        weight,
        variance_epsilon,
    )
    return output, residual_out


def _rocm_aiter_rmsnorm2d_fwd_with_add_fake(
        x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
        variance_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(residual)


def _rocm_aiter_fp4_gemm_with_dynamic_quant_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    x_scales: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from aiter import gemm_a4w4, per_1x32_f4_quant_hip

    M = x.shape[0]

    if x_scales is None:
        # use hip quant kernel for performance
        x_q, x_s = per_1x32_f4_quant_hip(x, shuffle=True)
    else:
        x_q = x
        x_s = x_scales

        # 32 alignment is enough for dim0 padding of output for
        # gemm_a4w4 kernel
        y = torch.empty((M + 31) // 32 * 32,
                        weight.shape[0],
                        device=x_q.device,
                        dtype=out_dtype)

        gemm_a4w4(x_q,
                  weight,
                  x_s,
                  weight_scale.view(x_s.dtype),
                  y,
                  bpreshuffle=True)
        return y[:M]


def _rocm_aiter_fp4_gemm_with_dynamic_quant_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    x_scales: torch.Tensor = None,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], weight.shape[0]),
                       dtype=out_dtype,
                       device=x.device)


# Global flag to ensure ops are registered only once
_OPS_REGISTERED = False


class rocm_aiter_ops:
    _AITER_ENABLED = envs.VLLM_ROCM_USE_AITER
    _LINEAR_ENABLED = envs.VLLM_ROCM_USE_AITER_LINEAR
    _RMSNORM_ENABLED = envs.VLLM_ROCM_USE_AITER_RMSNORM
    _FMOE_ENABLED = envs.VLLM_ROCM_USE_AITER_MOE
    _MLA_ENABLED = envs.VLLM_ROCM_USE_AITER_MLA
    _PG_ATTN_ENABLED = envs.VLLM_ROCM_USE_AITER_PAGED_ATTN
    _MHA_ENABLED = envs.VLLM_ROCM_USE_AITER_MHA
    _TRITON_UNIFIED_ATTN_ENABLED = envs.VLLM_USE_AITER_UNIFIED_ATTENTION
    _FP8BMM_ENABLED = envs.VLLM_ROCM_USE_AITER_FP8BMM
    _FP4_GEMM_DYNAMIC_QUANT_ASM = envs.VLLM_ROCM_USE_AITER_FP4_ASM_GEMM

    @classmethod
    @is_aiter_supported
    def is_enabled(cls) -> bool:
        """Verifies device specs and availability of aiter main env variable."""
        return cls._AITER_ENABLED

    @classmethod
    @is_aiter_supported
    def is_linear_enabled(cls) -> bool:
        """"Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._LINEAR_ENABLED

    @classmethod
    @is_aiter_supported
    def is_linear_fp8_enaled(cls) -> bool:
        """"Verifies device specs and availability of env variable."""
        return cls.is_linear_enabled() and current_platform.is_fp8_fnuz()

    @classmethod
    @is_aiter_supported
    def is_rmsnorm_enabled(cls) -> bool:
        """"Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._RMSNORM_ENABLED

    @classmethod
    @is_aiter_supported
    def is_fused_moe_enabled(cls) -> bool:
        """"Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._FMOE_ENABLED

    @classmethod
    @is_aiter_supported
    def is_mla_enabled(cls) -> bool:
        """"Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._MLA_ENABLED

    @classmethod
    @is_aiter_supported
    def is_mha_enabled(cls) -> bool:
        """"Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._MHA_ENABLED

    @classmethod
    @is_aiter_supported
    def is_pa_attn_enabled(cls) -> bool:
        """"Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._PG_ATTN_ENABLED

    @classmethod
    @is_aiter_supported
    def is_triton_unified_attn_enabled(cls) -> bool:
        """"Verifies device specs and availability of env variable."""
        return cls._AITER_ENABLED and cls._TRITON_UNIFIED_ATTN_ENABLED

    @classmethod
    @is_aiter_supported
    def is_fp8bmm_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._FP8BMM_ENABLED

    @classmethod
    @is_aiter_supported
    def is_asm_fp4_gemm_dynamic_quant_enabled(cls) -> bool:
        return cls._AITER_ENABLED and cls._FP4_GEMM_DYNAMIC_QUANT_ASM

    @staticmethod
    @is_aiter_supported
    def register_ops_once() -> None:
        global _OPS_REGISTERED
        if not _OPS_REGISTERED:
            tags = tuple() if is_torch_equal_or_newer("2.7.0") else (
                torch.Tag.needs_fixed_stride_order, )

            # register all the custom ops here
            direct_register_custom_op(
                op_name="rocm_aiter_asm_moe_tkw1",
                op_func=_rocm_aiter_asm_moe_tkw1_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_asm_moe_tkw1_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_fused_moe",
                op_func=_rocm_aiter_fused_moe_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_fused_moe_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_topk_softmax",
                op_func=_rocm_aiter_topk_softmax_impl,
                mutates_args=[
                    "topk_weights", "topk_indices", "token_expert_indices"
                ],
                fake_impl=_rocm_aiter_topk_softmax_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_biased_grouped_topk",
                op_func=_rocm_aiter_biased_grouped_topk_impl,
                mutates_args=["topk_weights", "topk_ids"],
                fake_impl=_rocm_aiter_biased_grouped_topk_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_grouped_topk",
                op_func=_rocm_aiter_grouped_topk_impl,
                mutates_args=["topk_weights", "topk_ids"],
                fake_impl=_rocm_aiter_grouped_topk_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_mla_decode_fwd",
                op_func=_rocm_aiter_mla_decode_fwd_impl,
                mutates_args=["o"],
                fake_impl=_rocm_aiter_mla_decode_fwd_fake,
                tags=tags)

            direct_register_custom_op(
                op_name="rocm_aiter_gemm_w8a8",
                op_func=_rocm_aiter_gemm_w8a8_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_gemm_w8a8_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_gemm_w8a8_blockscale",
                op_func=_rocm_aiter_gemm_w8a8_blockscale_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_gemm_w8a8_blockscale_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rms_norm",
                op_func=_rocm_aiter_rms_norm_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_rms_norm_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_rmsnorm2d_fwd_with_add",
                op_func=_rocm_aiter_rmsnorm2d_fwd_with_add_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_rmsnorm2d_fwd_with_add_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="rocm_aiter_fp4_gemm_with_dynamic_quant",
                op_func=_rocm_aiter_fp4_gemm_with_dynamic_quant_impl,
                mutates_args=[],
                fake_impl=_rocm_aiter_fp4_gemm_with_dynamic_quant_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            _OPS_REGISTERED = True

    @staticmethod
    def rms_norm2d_with_add(
            x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
            variance_epsilon: float) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(
            x, residual, weight, variance_epsilon)

    @staticmethod
    def rms_norm(x: torch.Tensor, weight: torch.Tensor,
                 variance_epsilon: float) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_rms_norm(x, weight, variance_epsilon)

    @staticmethod
    def gemm_w8a8(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_gemm_w8a8(A, B, As, Bs, bias,
                                                   output_dtype)

    @staticmethod
    def gemm_w8a8_blockscale(
        A: torch.Tensor,
        B: torch.Tensor,
        As: torch.Tensor,
        Bs: torch.Tensor,
        block_size: list[int],
        output_dtype: torch.dtype = torch.float16,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_gemm_w8a8_blockscale(
            A, B, As, Bs, output_dtype)

    @staticmethod
    def fused_moe(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weight: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_mask: Optional[torch.Tensor] = None,
        activation_method: int = 0,
        quant_method: int = 0,
        doweight_stage1: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_fused_moe(
            hidden_states, w1, w2, topk_weight, topk_ids, expert_mask,
            activation_method, quant_method, doweight_stage1, w1_scale,
            w2_scale, a1_scale, a2_scale)

    @staticmethod
    def asm_moe_tkw1(hidden_states: torch.Tensor,
                     w1: torch.Tensor,
                     w2: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     fc1_scale: Optional[torch.Tensor] = None,
                     fc2_scale: Optional[torch.Tensor] = None,
                     fc1_smooth_scale: Optional[torch.Tensor] = None,
                     fc2_smooth_scale: Optional[torch.Tensor] = None,
                     a16: bool = False,
                     per_tensor_quant_scale: Optional[torch.Tensor] = None,
                     expert_mask: Optional[torch.Tensor] = None,
                     activation_method: int = 0) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_asm_moe_tkw1(
            hidden_states, w1, w2, topk_weights, topk_ids, fc1_scale,
            fc2_scale, fc1_smooth_scale, fc2_smooth_scale, a16,
            per_tensor_quant_scale, expert_mask, activation_method)

    @staticmethod
    def topk_softmax(topk_weights: torch.Tensor, topk_indices: torch.Tensor,
                     token_expert_indices: torch.Tensor,
                     gating_output: torch.Tensor,
                     renormalize: bool) -> tuple[torch.Tensor, ...]:
        torch.ops.vllm.rocm_aiter_topk_softmax(topk_weights, topk_indices,
                                               token_expert_indices,
                                               gating_output, renormalize)
        return topk_weights, topk_indices

    @staticmethod
    def biased_grouped_topk(gating_output: torch.Tensor,
                            correction_bias: torch.Tensor,
                            topk_weights: torch.Tensor,
                            topk_ids: torch.Tensor,
                            num_expert_group: int,
                            topk_group: int,
                            need_renorm: bool,
                            routed_scaling_factor: float = 1.0) -> None:
        torch.ops.vllm.rocm_aiter_biased_grouped_topk(
            gating_output, correction_bias, topk_weights, topk_ids,
            num_expert_group, topk_group, need_renorm, routed_scaling_factor)

    @staticmethod
    def grouped_topk(gating_output: torch.Tensor,
                     topk_weights: torch.Tensor,
                     topk_ids: torch.Tensor,
                     num_expert_group: int,
                     topk_group: int,
                     need_renorm: bool,
                     scoring_func: str = "softmax",
                     routed_scaling_factor: float = 1.0) -> None:
        torch.ops.vllm.rocm_aiter_grouped_topk(gating_output, topk_weights,
                                               topk_ids, num_expert_group,
                                               topk_group, need_renorm,
                                               scoring_func,
                                               routed_scaling_factor)

    @staticmethod
    def mla_decode_fwd(
        q: torch.Tensor,
        kv_buffer: torch.Tensor,
        o: torch.Tensor,
        sm_scale: float,
        qo_indptr: torch.Tensor,
        max_seqlen_qo: int,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_last_page_lens: Optional[torch.Tensor] = None,
        logit_cap: float = 0.0,
    ):
        torch.ops.vllm.rocm_aiter_mla_decode_fwd(q,
                                                 kv_buffer.view(
                                                     -1, 1, 1, q.shape[-1]),
                                                 o,
                                                 qo_indptr,
                                                 max_seqlen_qo,
                                                 kv_indptr,
                                                 kv_indices,
                                                 kv_last_page_lens,
                                                 sm_scale=sm_scale,
                                                 logit_cap=logit_cap)

    @staticmethod
    def asm_fp4_gemm_dynamic_quant(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        x_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.rocm_aiter_fp4_gemm_with_dynamic_quant(
            x,
            weight,
            weight_scale,
            out_dtype,
            x_scales,
        )

    @staticmethod
    def triton_fp4_gemm_dynamic_qaunt(
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        x_scales: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        from aiter.ops.triton.gemm_afp4wfp4 import gemm_afp4wfp4
        from aiter.ops.triton.quant import dynamic_mxfp4_quant

        if x_scales is None:
            x_q, x_s = dynamic_mxfp4_quant(x)
        else:
            x_q = x
            x_s = x_scales

        y = torch.empty(x_q.shape[0],
                        weight.shape[0],
                        device=x_q.device,
                        dtype=out_dtype)

        gemm_afp4wfp4(x_q, weight, x_s, weight_scale.T, out_dtype, y)
        return y

    @staticmethod
    def triton_fp8_bmm(
        X: torch.Tensor,
        WQ: torch.Tensor,
        w_scale: torch.Tensor,
        group_size: int = 128,
        bias: Optional[torch.Tensor] = None,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        splitK: Optional[int] = None,
        YQ: Optional[torch.Tensor] = None,
        transpose_bm: Optional[bool] = False,
        config: Optional[dict] = None,
    ) -> torch.Tensor:
        from aiter.ops.triton.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (  # noqa: E501 # isort: skip
            batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant
            as aiter_triton_fp8_bmm)

        return aiter_triton_fp8_bmm(X,
                                    WQ,
                                    w_scale,
                                    group_size=group_size,
                                    bias=bias,
                                    dtype=dtype,
                                    splitK=splitK,
                                    YQ=YQ,
                                    transpose_bm=transpose_bm,
                                    config=config)

    @staticmethod
    def per_1x128_fp8_quant(
        input_2d: torch.Tensor, ) -> tuple[torch.Tensor, ...]:
        """ Only applies quantization method for fp8 data type."""
        from aiter import QuantType, dtypes, get_hip_quant

        aiter_per1x128_quant = get_hip_quant(QuantType.per_1x128)
        return aiter_per1x128_quant(input_2d, quant_dtype=dtypes.fp8)


rocm_aiter_ops.register_ops_once()
