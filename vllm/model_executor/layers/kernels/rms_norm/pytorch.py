# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.platforms import Platform

from .RMSNormKernel import RMSNormKernel, RMSNormLayerConfig

if TYPE_CHECKING:
    from vllm.model_executor.layers.layernorm import RMSNorm


class PytorchRMSNormKernel(RMSNormKernel):
    @classmethod
    def is_supported(
        cls, current_platform: Platform, c: RMSNormLayerConfig
    ) -> tuple[bool, str]:
        return c.weight_dtype in [
            torch.float32
        ], f"layer norm weight dtype of {c.weight_dtype} not supported"

    @classmethod
    def supported_fallback_kernels(cls):
        return [PytorchRMSNormKernel]

    def apply(
        self, layer: "RMSNorm", x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        orig_dtype = x.dtype
        variance_epsilon = layer.variance_epsilon
        weight = layer.weight.data if layer.has_weight else None
        variance_size_override = layer.variance_size_override
        hidden_size = layer.hidden_size

        x = x.to(torch.float32)
        if residual is not None:
            # residual promoted f16->f32 automatically,
            # otherwise Inductor eliminates the casts to and from f16,
            # increasing memory usage (and complicating pattern matching)
            x = x + residual
            residual = x.to(orig_dtype)

        if x.shape[-1] != hidden_size:
            raise ValueError(
                f"Expected hidden_size to be {hidden_size}, but found: {x.shape[-1]}"
            )

        if variance_size_override is None:
            x_var = x
        else:
            if hidden_size < variance_size_override:
                raise ValueError(
                    "Expected hidden_size to be at least "
                    f"{variance_size_override}, but found: {hidden_size}"
                )

            x_var = x[:, :, :variance_size_override]

        variance = x_var.pow(2).mean(dim=-1, keepdim=True)

        x = x * torch.rsqrt(variance + variance_epsilon)
        x = x.to(orig_dtype)
        if weight is not None:
            x = x * weight
        if residual is None:
            return x
        else:
            return x, residual
