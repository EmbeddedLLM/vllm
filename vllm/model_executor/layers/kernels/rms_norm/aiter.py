# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm._aiter_ops import rocm_aiter_ops
from vllm.platforms import Platform

from .cuda import CudaRMSNormKernel
from .pytorch import PytorchRMSNormKernel
from .RMSNormKernel import RMSNormKernel, RMSNormLayerConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.layernorm import RMSNorm


class AiterRMSNormKernel(RMSNormKernel):
    @classmethod
    def is_supported(
        cls, current_platform: Platform, c: RMSNormLayerConfig
    ) -> tuple[bool, str]:
        if not rocm_aiter_ops.is_rmsnorm_enabled():
            return False, "ROCm Aiter RMSNorm is not enabled"
        return c.weight_dtype in [
            torch.bfloat16,
            torch.float16,
        ], f"weight dtype {c.weight_dtype} is not supported"

    @classmethod
    def supported_fallback_kernels(cls):
        return [CudaRMSNormKernel, PytorchRMSNormKernel]

    def apply(
        self, layer: "RMSNorm", x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        add_residual = residual is not None
        variance_epsilon = layer.variance_epsilon
        weight = layer.weight.data if layer.has_weight else None

        if add_residual:
            return rocm_aiter_ops.rms_norm2d_with_add(
                x, residual, weight, variance_epsilon
            )
        else:
            return rocm_aiter_ops.rms_norm(x, weight.data, variance_epsilon)
