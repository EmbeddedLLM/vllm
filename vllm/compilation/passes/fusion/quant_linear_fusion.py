# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fuses ``rocm_aiter_group_fp8_quant`` followed by
``rocm_aiter_gemm_a8w8_blockscale`` into a single
``vllm::fused_block_fp8_quant_gemm`` triton kernel.
"""

import torch

import vllm.kernels.triton.fused_block_fp8_quant_gemm  # noqa: F401
from vllm._aiter_ops import rocm_aiter_ops  # noqa: F401
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmPatternReplacement,
)

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()


class AiterGroupQuantBlockscaleGemmPattern(VllmPatternReplacement):
    """Match ``rocm_aiter_group_fp8_quant`` + ``rocm_aiter_gemm_a8w8_blockscale``
    and replace with the fused triton kernel.
    """

    FUSED_OP = torch.ops.vllm.fused_block_fp8_quant_gemm.default

    def __init__(self, group_size: int, out_dtype: torch.dtype) -> None:
        self.group_size = group_size
        self.out_dtype = out_dtype

    def get_inputs(self) -> list[torch.Tensor]:
        M, N, K = 4, 128, self.group_size
        a = self.empty(M, K, dtype=self.out_dtype)
        b_q = self.empty(N, K, dtype=FP8_DTYPE)
        b_scale = self.empty_fp32(N // 128 or 1, K // self.group_size)
        return [a, b_q, b_scale]

    @property
    def pattern(self):
        def _pattern(
            a: torch.Tensor,
            b_q: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            a_q, a_scale = torch.ops.vllm.rocm_aiter_group_fp8_quant(a, self.group_size)
            return torch.ops.vllm.rocm_aiter_gemm_a8w8_blockscale(
                a_q, b_q, a_scale, b_scale, self.out_dtype
            )

        return _pattern

    @property
    def replacement(self):
        def _replacement(
            a: torch.Tensor,
            b_q: torch.Tensor,
            b_scale: torch.Tensor,
        ) -> torch.Tensor:
            return self.FUSED_OP(
                a, b_q, b_scale, self.group_size, FP8_DTYPE, self.out_dtype
            )

        return _replacement


class AiterQuantLinearFusionPass(VllmFusionPatternMatcherPass):
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "quant_linear_fusion_pass")

        for group_size in [64, 128]:
            for out_dtype in [torch.float16, torch.bfloat16]:
                self.register(
                    AiterGroupQuantBlockscaleGemmPattern(
                        group_size=group_size,
                        out_dtype=out_dtype,
                    )
                )

        self.dump_patterns(config, self.pm_pass)
