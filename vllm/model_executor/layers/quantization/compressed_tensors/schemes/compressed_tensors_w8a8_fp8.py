# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationStrategy
from torch.nn import Module

from vllm._aiter_ops import rocm_aiter_ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    init_fp8_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    STRATEGY_TO_PARAMETER_TYPE,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    create_fp8_input_scale,
    create_fp8_scale_parameter,
    create_fp8_weight_parameter,
    process_fp8_weight_block_strategy,
    process_fp8_weight_channel_strategy,
    process_fp8_weight_tensor_strategy,
    validate_fp8_block_shape,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    cutlass_block_fp8_supported,
)
from vllm.model_executor.linear_params import Fp8LinearParams

__all__ = ["CompressedTensorsW8A8Fp8"]

STATIC_QUANT = True
DYNAMIC_QUANT = False
activation_quant_key_mapping = {
    STATIC_QUANT: kFp8StaticTensorSym,
    DYNAMIC_QUANT: kFp8DynamicTokenSym,
}
weight_quant_key_mapping = {
    QuantizationStrategy.CHANNEL: kFp8StaticChannelSym,
    QuantizationStrategy.TENSOR: kFp8StaticTensorSym,
}
logger = init_logger(__name__)


class CompressedTensorsW8A8Fp8(CompressedTensorsScheme[Fp8LinearParams]):
    def __init__(self, weight_quant: QuantizationArgs, is_static_input_scheme: bool):
        self.weight_quant = weight_quant
        self.strategy = weight_quant.strategy
        self.out_dtype = torch.get_default_dtype()
        self.input_dtype = get_current_vllm_config().model_config.dtype
        self.is_static_input_scheme = is_static_input_scheme
        self.weight_block_size = self.weight_quant.block_structure

        if self.weight_block_size is not None:
            self.cutlass_block_fp8_supported = cutlass_block_fp8_supported()
            self.use_aiter_and_is_supported = rocm_aiter_ops.is_linear_fp8_enabled()
            assert not self.is_static_input_scheme
            self.act_q_group_shape = GroupShape(1, self.weight_block_size[0])
            self.weight_quant_key = create_fp8_quant_key(
                static=True, group_shape=GroupShape(*self.weight_block_size)
            )
            self.activation_quant_key = create_fp8_quant_key(
                static=False, group_shape=self.act_q_group_shape
            )
        else:
            self.activation_quant_key = activation_quant_key_mapping[
                self.is_static_input_scheme
            ]
            self.weight_quant_key = weight_quant_key_mapping[self.strategy]

    @classmethod
    def get_min_capability(cls) -> int:
        # lovelace and up
        return 89

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.weight_block_size = None
        layer.orig_dtype = params_dtype

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            # Validate block quantization shapes
            validate_fp8_block_shape(
                layer,
                input_size,
                output_size,
                input_size_per_partition,
                output_partition_sizes,
                self.weight_block_size,
            )

        # WEIGHT
        weight = create_fp8_weight_parameter(
            output_size_per_partition, input_size_per_partition, weight_loader
        )

        # WEIGHT SCALE
        weight_scale = create_fp8_scale_parameter(
            STRATEGY_TO_PARAMETER_TYPE[self.strategy],
            output_partition_sizes,
            input_size_per_partition,
            layer.weight_block_size,
            weight_loader,
        )

        # INPUT SCALE
        input_scale = None
        if self.is_static_input_scheme:
            input_scale = create_fp8_input_scale(output_partition_sizes, weight_loader)

        if self.strategy == QuantizationStrategy.BLOCK:
            Fp8LinearParams.register_params_in_layer(
                layer,
                weight=weight,
                weight_scale_inv=weight_scale,
            )
        else:
            Fp8LinearParams.register_params_in_layer(
                layer,
                weight=weight,
                weight_scale=weight_scale,
                input_scale=input_scale,
            )

        self.fp8_linear = init_fp8_linear_kernel(
            activation_quant_key=self.activation_quant_key,
            weight_quant_key=self.weight_quant_key,
            input_dtype=self.input_dtype,
            out_dtype=self.out_dtype,
            weight_shape=(output_size_per_partition, input_size_per_partition),
            module_name=self.__class__.__name__,
        )

    def convert_to_canonical(self, layer: Module) -> Fp8LinearParams:
        params = Fp8LinearParams.read_params_from_layer(layer)

        if self.strategy == QuantizationStrategy.BLOCK:
            assert self.is_static_input_scheme is False
            new_weight, new_weight_scale_inv = process_fp8_weight_block_strategy(
                params.weight, params.weight_scale_inv
            )
            return params.evolve_and_verify(
                weight=new_weight,
                weight_scale_inv=new_weight_scale_inv,
            )

        if self.strategy == QuantizationStrategy.TENSOR:
            new_weight, new_weight_scale, new_input_scale = (
                process_fp8_weight_tensor_strategy(
                    params.weight,
                    params.weight_scale,
                    layer.logical_widths,
                    params.input_scale,
                )
            )
        elif self.strategy == QuantizationStrategy.CHANNEL:
            new_weight, new_weight_scale, new_input_scale = (
                process_fp8_weight_channel_strategy(
                    params.weight, params.weight_scale, params.input_scale
                )
            )
        else:
            raise ValueError(
                f"Unknown quantization strategy {self.strategy}: "
                f"should be one of {list(QuantizationStrategy)}"
            )

        if self.is_static_input_scheme:
            assert new_input_scale is not None
            new_input_scale = new_input_scale.max()
        else:
            new_input_scale = None

        new_weight = new_weight.t()
        return params.evolve_and_verify(
            weight=new_weight,
            weight_scale=new_weight_scale,
            input_scale=new_input_scale,
        )

    def process_weights_after_loading(self, layer) -> None:
        linear_params = self.convert_to_canonical(layer)
        self.fp8_linear.process_weights_after_loading(linear_params)
        linear_params.update_params_in_layer(layer)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.fp8_linear.apply_weights(layer, x, bias)
