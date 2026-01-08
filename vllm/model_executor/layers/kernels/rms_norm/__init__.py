# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.logger import init_logger
from vllm.platforms import PlatformEnum, current_platform

from .aiter import AiterRMSNormKernel
from .cuda import CudaRMSNormKernel
from .ipex import XPURMSNormKernel
from .pytorch import PytorchRMSNormKernel
from .RMSNormKernel import RMSNormKernel, RMSNormLayerConfig

logger = init_logger(__name__)

# in priority/performance order (when available)
_PLATFORM_SPECIFIC_KERNELS: dict[PlatformEnum, type[RMSNormKernel]] = {
    PlatformEnum.CPU: PytorchRMSNormKernel,
    PlatformEnum.CUDA: CudaRMSNormKernel,
    PlatformEnum.ROCM: AiterRMSNormKernel,
    PlatformEnum.XPU: XPURMSNormKernel,
}


def choose_rms_norm_kernel(weight_dtype: torch.dtype) -> RMSNormKernel:
    layer_config = RMSNormLayerConfig(weight_dtype=weight_dtype)
    platform_spec_kernel = _PLATFORM_SPECIFIC_KERNELS[current_platform._enum]
    is_supported, reason = platform_spec_kernel.is_supported(
        current_platform, layer_config
    )

    if not is_supported:
        fallback_kernels = platform_spec_kernel.supported_fallback_kernels()
        for fallback_kernel in fallback_kernels:
            is_fallback_supported, _ = fallback_kernel.is_supported(
                current_platform, layer_config
            )
            if is_fallback_supported:
                logger.warning_once(
                    f"{platform_spec_kernel.__name__} is not supported. \
                    due to {reason}. \
                    falling back to {fallback_kernel.__name__}"
                )
                return fallback_kernel()

        # If we get here, no fallback kernels were supported
        raise RuntimeError(
            f"No supported RMSNorm kernel found for platform {current_platform._enum} "
            f"with dtype {weight_dtype}. Tried {platform_spec_kernel.__name__} and "
            f"fallback kernels: {[k.__name__ for k in fallback_kernels]}"
        )

    return platform_spec_kernel()
