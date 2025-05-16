# SPDX-License-Identifier: Apache-2.0
from typing import Optional

import pytest
import torch

from tests.kernels.utils import baseline_scaled_mm, to_int8
from vllm.platforms import current_platform

MNK_FACTORS = [
    (1, 16384, 1024),
    (1, 24576, 496),
    (16, 256, 496),
    (16, 24576, 4096),
    (32, 8192, 4096),
    (32, 16384, 4096),
    (33, 1024, 1024),
    (64, 2048, 496),
    (64, 16384, 1024),
    (100, 8192, 496),
    (128, 32768, 4096),
    (256, 4096, 4096),
    (512, 256, 1024),
    (512, 8192, 4096),
    (512, 16384, 128),
    (512, 24576, 128),
]


def to_fp8_e4m3fnuz(tensor: torch.Tensor) -> torch.Tensor:
    finfo = torch.finfo(torch.float8_e4m3fnuz)
    return torch.round(tensor.clamp(
        min=finfo.min, max=finfo.max)).to(dtype=torch.float8_e4m3fnuz)


def aiter_scaled_mm_helper(a: torch.Tensor, b: torch.Tensor,
                           a_scale: torch.Tensor, b_scale: torch.Tensor,
                           bias: Optional[torch.Tensor],
                           out_dtype: torch.dtype) -> torch.Tensor:
    from aiter import gemm_a8w8_CK
    return gemm_a8w8_CK(a, b.t(), a_scale, b_scale, bias).to(out_dtype)


class AiterLayer(torch.nn.Module):

    def __init__(self, b, scale_a, scale_b, out_dtype):
        super().__init__()
        self.b = b
        self.scale_a = scale_a
        self.scale_b = scale_b
        self.out_dtype = out_dtype

    def forward(self, a):
        return aiter_scaled_mm_helper(a, self.b, self.scale_a, self.scale_b,
                                      None, self.out_dtype)


def assert_rocm_aiter_w8a8_gemm_close_to_baseline(a: torch.Tensor,
                                                  b: torch.Tensor,
                                                  a_scale: torch.Tensor,
                                                  b_scale: torch.Tensor,
                                                  bias: Optional[torch.Tensor],
                                                  out_dtype: torch.dtype,
                                                  rtol: float = 1e-2,
                                                  atol: float = 1e-1):
    out = aiter_scaled_mm_helper(a, b, a_scale, b_scale, bias, out_dtype)
    baseline = baseline_scaled_mm(a, b, a_scale, b_scale, out_dtype, bias)

    torch.testing.assert_close(out, baseline, rtol=rtol, atol=atol)


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(not current_platform.has_device_capability(90),
                    reason="FP8 is not supported on this GPU type.")
@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="Aiter W8A8 GEMM is only supported on ROCM ")
def test_rocm_aiter_fp8_gemm(m: int, n: int, k: int, use_bias: bool):

    out_dtype = torch.bfloat16
    a = to_fp8_e4m3fnuz(torch.randn((m, k), device="cuda"))
    b = to_fp8_e4m3fnuz(torch.randn((n, k), device="cuda").t())

    scale_a = (torch.randn((m, 1), device="cuda", dtype=torch.float32))
    scale_b = (torch.randn((1, n), device="cuda", dtype=torch.float32))

    bias = torch.rand(
        (n, ), device="cuda", dtype=out_dtype) * 10 if use_bias else None

    assert_rocm_aiter_w8a8_gemm_close_to_baseline(
        a,
        b,
        scale_a,
        scale_b,
        bias,
        out_dtype,
    )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="Aiter W8A8 GEMM is only supported on ROCM ")
def test_rocm_aiter_int8_gemm(m: int, n: int, k: int, use_bias: bool):

    out_dtype = torch.bfloat16
    a = to_int8(torch.randn((m, k), device="cuda") * 5)
    b = to_int8(torch.randn((n, k), device="cuda").t() * 5)

    a_scale = (torch.randn((m, 1), device="cuda", dtype=torch.float32))
    b_scale = (torch.randn((1, n), device="cuda", dtype=torch.float32))

    bias = torch.rand(
        (n, ), device="cuda", dtype=out_dtype) * 10 if use_bias else None

    assert_rocm_aiter_w8a8_gemm_close_to_baseline(
        a,
        b,
        a_scale,
        b_scale,
        bias,
        out_dtype,
    )


@pytest.mark.skipif(not current_platform.is_rocm(),
                    reason="Aiter W8A8 GEMM is only supported on ROCM ")
def test_rocm_aiter_gemm_cuda_graph():
    m, n, k = 512, 512, 512

    a = to_int8(torch.randn((m, k), device="cuda"))
    b = to_int8(torch.randn((n, k), device="cuda").t())

    scale_a = (torch.randn((m, 1), device="cuda", dtype=torch.float32) / 10)
    scale_b = (torch.randn((1, n), device="cuda", dtype=torch.float32) / 10)

    # Construct a trivial model with a single layer that calls a Aiter kernel
    model = AiterLayer(b, scale_a, scale_b, torch.bfloat16)

    # Run the model with a cuda graph
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = model(a)
    out.zero_()
    g.replay()

    baseline = torch.mm(scale_a * a.to(dtype=torch.float32),
                        scale_b * b.to(dtype=torch.float32)).to(torch.bfloat16)
    torch.testing.assert_close(out, baseline, rtol=1e0, atol=1e0)
