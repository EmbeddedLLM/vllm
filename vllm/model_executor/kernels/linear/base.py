# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar, Generic, TypeVar, overload

import torch


@dataclass
class MMLinearLayerConfig: ...


@dataclass
class Params: ...


@dataclass
class Int8Params(Params):
    """Int8 layer parameters with typed fields"""

    weight: torch.Tensor
    weight_scale: torch.Tensor
    input_scale: torch.Tensor | None

    # Attribute names on the layer
    WEIGHT: ClassVar[str] = "weight"
    WEIGHT_SCALE: ClassVar[str] = "weight_scale"
    INPUT_SCALE: ClassVar[str] = "input_scale"

    input_zero_point: torch.Tensor | None
    azp_adj: torch.Tensor | None

    INPUT_ZERO_POINT: ClassVar[str] = "input_zero_point"
    AZP_ADJ: ClassVar[str] = "azp_adj"

    @classmethod
    def from_layer(cls, layer: torch.nn.Module) -> "Int8Params":
        """Extract parameters from layer"""
        return cls(
            weight=getattr(layer, cls.WEIGHT),
            weight_scale=getattr(layer, cls.WEIGHT_SCALE),
            input_scale=getattr(layer, cls.INPUT_SCALE, None),
            input_zero_point=getattr(layer, cls.INPUT_ZERO_POINT, None),
            azp_adj=getattr(layer, cls.AZP_ADJ, None),
        )


_ParamsT = TypeVar("_ParamsT")
_ConfigT = TypeVar("_ConfigT", bound=MMLinearLayerConfig)


class MMLinearKernel(ABC, Generic[_ConfigT, _ParamsT]):
    """Abstract base class for quantized matrix multiplication kernels.

    This class provides the interface for implementing custom quantized linear layer
    kernels in vLLM. Subclasses should implement specific quantization strategies
    (e.g., FP8, INT8) and their corresponding compute kernels.

    Generic Type Parameters:
        _ConfigT: Configuration type for the kernel (subclass of MMLinearLayerConfig).
                  Contains kernel-specific settings like quantization keys, dtypes, etc.
        _ParamsT: Parameter type for the kernel (subclass of Params).
                  Defines the quantized weights and scales needed by the kernel.

    Typical Usage:
        1. Define a config dataclass inheriting from MMLinearLayerConfig
        2. Define a params dataclass inheriting from Params (or FP8Params/Int8Params)
        3. Subclass MMLinearKernel with your config and params types
        4. Implement all abstract methods
        5. Register the kernel with the quantization method

    Example:
        ```python
        @dataclass
        class MyKernelConfig(MMLinearLayerConfig):
            static: bool
            output_dtype: torch.dtype


        @dataclass
        class MyKernelParams(FP8Params):
            custom_scale: torch.Tensor
            CUSTOM_SCALE: ClassVar[str] = "custom_scale"


        class MyKernel(MMLinearKernel[MyKernelConfig, MyKernelParams]):
            @classmethod
            def is_supported(cls, compute_capability=None):
                if compute_capability and compute_capability < 90:
                    return False, "Requires compute capability >= 9.0"
                return True, None

            @classmethod
            def can_implement(cls, config):
                if not config.static:
                    return False, "Only static quantization supported"
                return True, None

            def process_weights_after_loading(self, layer):
                # Preprocess weights for the kernel
                params = self._get_layer_params(layer)
                processed = preprocess_weights(params.weight)
                replace_parameter(layer, params.WEIGHT, processed)

            def _get_layer_params(self, layer, **kwargs):
                return MyKernelParams.from_layer(layer)

            def apply_weights(self, layer, x, bias=None, **kwargs):
                params = self._get_layer_params(layer)
                # Call your custom kernel
                output = my_custom_kernel(x, params.weight, params.weight_scale)
                if bias is not None:
                    output += bias
                return output
        ```

    Lifecycle:
        1. Kernel selection: is_supported() and can_implement() check compatibility
        2. Initialization: __init__() creates kernel instance with config
        3. Weight loading: process_weights_after_loading() preprocesses weights
        4. Inference: apply_weights() executes the quantized matmul
    """

    @classmethod
    @abstractmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        """Check if this kernel is supported on the current hardware.

        This method checks hardware-level compatibility (e.g., GPU architecture,
        compute capability, available instructions). It's called during kernel
        selection to filter out kernels that cannot run on the current device.

        Args:
            compute_capability: GPU compute capability (e.g., 80 for A100, 90 for H100).
                               If None, should check the current device.

        Returns:
            A tuple of (is_supported, reason):
                - is_supported: True if the kernel can run on this hardware
                - reason: If not supported, a string explaining why; otherwise None
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def can_implement(cls, config: _ConfigT) -> tuple[bool, str | None]:
        """Check if this kernel can implement the given configuration.

        This method checks configuration-level compatibility (e.g., quantization
        scheme, group sizes, static vs dynamic quantization). It's called after
        is_supported() to determine if this kernel can handle the specific
        quantization configuration.

        Args:
            config: The kernel configuration to check

        Returns:
            A tuple of (can_implement, reason):
                - can_implement: True if this kernel supports the config
                - reason: If not supported, a string explaining why; otherwise None
            ```
        """
        raise NotImplementedError

    def __init__(self, config: _ConfigT) -> None:
        """Initialize the kernel with the given configuration.

        Args:
            config: Kernel-specific configuration containing settings like
                   quantization keys, output dtypes, etc.
        """
        self.config = config

    @overload
    def process_weights_after_loading(self, arg: _ParamsT) -> None: ...
    @overload
    def process_weights_after_loading(self, arg: torch.nn.Module) -> None: ...

    def process_weights_after_loading(self, arg: _ParamsT | torch.nn.Module) -> None:
        """Optional hook called once after weights are loaded.

        Two signatures are accepted during the migration to canonical
        ``LinearParamsBase`` flow:

        * ``(self, params: _ParamsT)`` — preferred. Mutate fields on
          ``params`` in place; the LinearMethod writes them back to the
          layer.
        * ``(self, layer: torch.nn.Module)`` — legacy. Kernels (e.g. Int8)
          may still take the layer directly.
        """
        return

    # return a covariant type in the subclass
    @abstractmethod
    def _get_layer_params(self, layer: torch.nn.Module, **kwargs: Any) -> _ParamsT:
        """Extract typed parameters from the layer module.

        This internal method retrieves the quantized weights and scales from
        the layer as a typed parameter object. Subclasses should typically
        delegate to ParamsClass.from_layer().

        Args:
            layer: The layer module containing the parameters
            **kwargs: Additional arguments

        Returns:
            A typed parameter object containing weights, scales, and other
            quantization parameters

        Example:
            ```python
            def _get_layer_params(self, layer, **kwargs):
                return MyKernelParams.from_layer(layer)
            ```
        """
        raise NotImplementedError

    def get_output_padding(self) -> int | None:
        """Get the number of output tokens to pad for this kernel.

        Some kernels require input padding for optimal performance.
        Override this method to specify padding requirements.

        Returns:
            Number of tokens to pad, or None for no padding (default)
        """
        return None

    @abstractmethod
    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Apply the quantized weights to the input tensor.

        This is the main inference method that performs the quantized matrix
        multiplication. It should handle input quantization (if needed), call
        the underlying kernel, and apply bias.

        Args:
            layer: The layer module containing the quantized weights
            x: Input tensor of shape [..., in_features]
            bias: Optional bias tensor of shape [out_features]
            **kwargs: Additional kernel-specific arguments

        Returns:
            Output tensor of shape [..., out_features]
        """
        raise NotImplementedError
