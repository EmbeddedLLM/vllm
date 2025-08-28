# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from enum import IntEnum
from typing import Optional

import torch

from vllm._aiter_ops import rocm_aiter_ops


class QuantMethod(IntEnum):
    # This allows interfacing with AITER QuantType Enum
    # without importing the QuantType from AITER globally.

    # Note that these quantization methods are
    # supported in AITER package. However,
    # not all are used in this module.

    NO = 0  # a16w16
    PER_TENSOR = 1  # w8a8 (pre_Tensor)
    PER_TOKEN = 2  # w8a8/w8a4 (per_Token)
    BLOCK_1X32 = 3  # fp4x2
    BLOCK_1X128 = 4  # block quantized w8a8 (per_1x128)
    BLOCK_128x128 = 5  # block quantized w8a8 (per_128x128)


class ActivationMethod(IntEnum):
    # This allows interfacing with AITER ActivationType enum
    # without importing the ActivationType enum from AITER globally.
    SILU = 0
    GELU = 1


def rocm_aiter_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    token = hidden_states.shape[0]
    device = hidden_states.device
    topk_ids = torch.empty((token, topk), dtype=torch.int32, device=device)
    topk_weights = torch.empty((token, topk),
                               dtype=torch.float32,
                               device=device)

    if e_score_correction_bias is not None:
        rocm_aiter_ops.biased_grouped_topk(
            gating_output,
            e_score_correction_bias,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
        )
    else:
        assert (scoring_func == "softmax" or scoring_func == "sigmoid")
        rocm_aiter_ops.grouped_topk(
            gating_output,
            topk_weights,
            topk_ids,
            num_expert_group,
            topk_group,
            renormalize,
            scoring_func,
        )

    return topk_weights, topk_ids


def rocm_aiter_fused_experts(
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        use_fp8_w8a8: bool = False,
        per_channel_quant: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        block_shape: Optional[list[int]] = None,
        expert_map: Optional[torch.Tensor] = None) -> torch.Tensor:

    activation_method = (ActivationMethod.SILU
                         if activation == "silu" else ActivationMethod.GELU)
    # All AITER Fused MoE kernels are expecting the following datatypes
    topk_weights = topk_weights.to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    if expert_map is not None:
        expert_mask = (expert_map > -1).to(torch.int32)
    else:
        expert_mask = None

    # w8a8 per-channel quantization
    if per_channel_quant and apply_router_weight_on_input and use_fp8_w8a8:
        # AITER tkw1 kernel for FP8 models with `apply_router_weight_on_input`
        # This applies topk_weights on the GEMM output of the first FC layer
        #  rather than the second FC.
        assert (topk_weights.dim() == 2
                ), "`topk_weights` should be in shape (num_tokens, topk)"
        assert topk_weights.shape[-1] == 1, (
            "Only support topk=1 when"
            " `apply_router_weight_on_input` is True")

        return rocm_aiter_ops.asm_moe_tkw1(hidden_states,
                                           w1,
                                           w2,
                                           topk_weights,
                                           topk_ids,
                                           fc1_scale=w1_scale,
                                           fc2_scale=w2_scale,
                                           fc1_smooth_scale=None,
                                           fc2_smooth_scale=None,
                                           a16=False,
                                           per_tensor_quant_scale=None,
                                           expert_mask=expert_mask,
                                           activation_method=activation_method)

    else:
        quant_method = QuantMethod.NO.value

        # w8a8 block-scaled
        if block_shape is not None and use_fp8_w8a8:
            assert not apply_router_weight_on_input, (
                "apply_router_weight_on_input is\
                not supported for block scaled moe")
            assert w1_scale is not None
            assert w2_scale is not None
            quant_method = QuantMethod.BLOCK_128x128.value
        elif use_fp8_w8a8:
            # Currently only per tensor quantization method is enabled.
            quant_method = QuantMethod.PER_TENSOR.value

        if apply_router_weight_on_input:
            assert (topk_weights.dim() == 2
                    ), "`topk_weights` should be in shape (num_tokens, topk)"
            _, topk = topk_weights.shape
            assert (
                topk == 1
            ), "Only support topk=1 when `apply_router_weight_on_input` is True"

        return rocm_aiter_ops.fused_moe(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            expert_mask=expert_mask,
            quant_method=quant_method,
            activation_method=activation_method,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            doweight_stage1=apply_router_weight_on_input)


def shuffle_weights(
    *tensors: torch.Tensor, layout: tuple[int, int] = (16, 16)
) -> tuple[torch.Tensor, ...]:
    """
    Applies shuffle_weight function from AITER to each 
    input tensor and returns them.
    
    Rearranges (shuffles) the input tensor/s
    into a specified block layout for optimized computation.

    Args:
        *tensors: Variable number of torch.Tensor objects.
        layout: A pair of integers specifying the 
        block sizes used to divide the tensors during shuffling.
        Default is (16, 16).

    Returns:
    A Tuple of shuffled tensors.
    """
    from aiter.ops.shuffle import shuffle_weight

    return tuple(shuffle_weight(tensor, layout=layout) for tensor in tensors)
