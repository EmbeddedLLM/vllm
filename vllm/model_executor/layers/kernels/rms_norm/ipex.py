# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.platforms import Platform

from .pytorch import PytorchRMSNormKernel
from .RMSNormKernel import RMSNormKernel, RMSNormLayerConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.layernorm import RMSNorm


class XPURMSNormKernel(RMSNormKernel):
    @classmethod
    def is_supported(
        cls, current_platform: Platform, c: RMSNormLayerConfig
    ) -> tuple[bool, str]:
        return current_platform.is_xpu(), f"{current_platform._enum} is not suitable"

    @classmethod
    def supported_fallback_kernels(cls):
        return [PytorchRMSNormKernel]

    def apply(
        self, layer: "RMSNorm", x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        from vllm._ipex_ops import ipex_ops as ops

        weight = layer.weight.data if layer.has_weight else None
        variance_epsilon = layer.variance_epsilon

        if residual is not None:
            ops.fused_add_rms_norm(
                x,
                residual,
                weight,
                variance_epsilon,
            )
            return x, residual

        return ops.rms_norm(x, weight, variance_epsilon)
