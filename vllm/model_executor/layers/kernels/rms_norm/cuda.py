# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.batch_invariant import (
    rms_norm_batch_invariant,
    vllm_is_batch_invariant,
)
from vllm.platforms import Platform

from .pytorch import PytorchRMSNormKernel
from .RMSNormKernel import RMSNormKernel, RMSNormLayerConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.layernorm import RMSNorm


def rms_norm(
    x: torch.Tensor, weight: torch.Tensor, variance_epsilon: float
) -> torch.Tensor:
    from vllm import _custom_ops as ops

    if vllm_is_batch_invariant():
        return rms_norm_batch_invariant(x, weight, variance_epsilon)
    out = torch.empty_like(x)
    ops.rms_norm(
        out,
        x,
        weight,
        variance_epsilon,
    )
    return out


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm import _custom_ops as ops

    if vllm_is_batch_invariant():
        return rms_norm_batch_invariant(
            x + residual, weight, variance_epsilon
        ), x + residual
    ops.fused_add_rms_norm(
        x,
        residual,
        weight,
        variance_epsilon,
    )
    return x, residual


class CudaRMSNormKernel(RMSNormKernel):
    @classmethod
    def is_supported(
        cls, current_platform: Platform, c: RMSNormLayerConfig
    ) -> tuple[bool, str]:
        if not current_platform.is_cuda_alike():
            return False, "Platform is not CUDA-alike"
        return c.weight_dtype in [
            torch.bfloat16,
            torch.float16,
        ], f"weight dtype {c.weight_dtype} is not supported"

    @classmethod
    def supported_fallback_kernels(cls):
        return [PytorchRMSNormKernel]

    def apply(
        self, layer: "RMSNorm", x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        add_residual = residual is not None
        variance_epsilon = layer.variance_epsilon
        weight = layer.weight.data if layer.has_weight else None

        if add_residual:
            return fused_add_rms_norm(x, residual, weight, variance_epsilon)
        else:
            return rms_norm(x, weight.data, variance_epsilon)
