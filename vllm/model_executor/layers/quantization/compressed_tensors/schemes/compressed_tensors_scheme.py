# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import torch

from vllm.model_executor.linear_params import LinearParamsBase

__all__ = ["CompressedTensorsScheme"]

_ParamsT = TypeVar("_ParamsT", bound=LinearParamsBase)


class CompressedTensorsScheme(ABC, Generic[_ParamsT]):
    """
    Abstract class used to describe the weight creation and forward pass
    of different quantization schemes supported by CompressedTensors.
    """

    @classmethod
    @abstractmethod
    def get_min_capability(cls) -> int:
        """
        Get minimum device capability.
        """
        raise NotImplementedError()

    @abstractmethod
    def create_weights(self, *args, **kwargs):
        """
        Weight creation for the particular scheme. Inputs to this function

        """
        raise NotImplementedError()

    @abstractmethod
    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ):
        """
        Run the forward pass for the particular scheme. This is where
        scheme-specific dequant/quant steps/kernels should be applied.

        :param layer: torch.nn.Module with the registered weights and
            other parameters relevant to the particular scheme.
        :param x: input to the layer
        :param bias: bias parameter

        """
        raise NotImplementedError()

    @abstractmethod
    def process_weights_after_loading(self, layer: torch.nn.Module):
        """
        Called after weight loading is complete for any cleanup that
        needs to occur.
        """
        raise NotImplementedError()

    def convert_to_canonical(self, layer: torch.nn.Module) -> _ParamsT:
        """
        Optional: convert raw checkpoint params on ``layer`` into the
        scheme's canonical ``LinearParamsBase`` subclass. Schemes that
        have migrated to the typed-params pipeline should override this
        and call it from ``process_weights_after_loading``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} has not adopted convert_to_canonical"
        )
