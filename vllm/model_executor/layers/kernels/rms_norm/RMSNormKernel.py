# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ..base import BaseKernelConfig, CustomKernel

if TYPE_CHECKING:
    from vllm.model_executor.layers.layernorm import RMSNorm


@dataclass
class RMSNormLayerConfig(BaseKernelConfig):
    weight_dtype: torch.dtype


class RMSNormKernel(CustomKernel[RMSNormLayerConfig, "RMSNorm"]):
    def process_weights_after_loading(self, layer: "RMSNorm"): ...

    def apply(
        self,
        layer: "RMSNorm",
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError
