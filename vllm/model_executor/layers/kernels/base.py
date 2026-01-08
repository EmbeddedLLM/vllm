# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from typing_extensions import Self

from vllm.platforms import Platform


@dataclass
class BaseKernelConfig:
    """Base class for all kernel configurations."""

    pass


CustomKernelConfig = TypeVar("CustomKernelConfig", bound=BaseKernelConfig)
CustomLayer = TypeVar("CustomLayer", bound=torch.nn.Module)


class CustomKernel(ABC, Generic[CustomKernelConfig, CustomLayer]):
    @classmethod
    @abstractmethod
    def is_supported(
        cls, current_platform: Platform, c: CustomKernelConfig
    ) -> tuple[bool, str]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def supported_fallback_kernels(cls) -> list[type[Self]]:
        """
        The list must be ordered based on priority
        """
        raise NotImplementedError

    @abstractmethod
    def process_weights_after_loading(self, layer: CustomLayer) -> None:
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: CustomLayer,
        *args,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError
