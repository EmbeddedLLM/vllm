# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# This is a test for the RMS norm AITER ops.
# It tests if the AITER RMS norm ops are
# 1. correctly registered as custom ops
# 2. correctly defined the relationship between
#    implementation and fake function
# 3. can be used with torch.compile
# This file will be skipped if AITER is not installed
# and the platform is not ROCm.

import importlib.util
import pytest
import torch
import torch.nn.functional as F

# Import to ensure the ops are registered
import vllm.model_executor.layers.layernorm  # noqa: F401
from vllm.model_executor.layers.layernorm import RMSNorm, GemmaRMSNorm
from vllm.platforms import current_platform

import unittest.mock as mock
from vllm.model_executor.layers.layernorm import (
    dispatch_cuda_rmsnorm_func, 
    rms_norm,
    fused_add_rms_norm
)


# Check if aiter package is installed
aiter_available = importlib.util.find_spec("aiter") is not None
pytestmark = pytest.mark.skipif(
    not (current_platform.is_rocm() and aiter_available),
    reason="AITER RMS norm ops are only available on ROCm with aiter package installed"
)


def test_rocm_aiter_rms_norm_custom_op_registration():
    """Test that the RMS norm custom op is correctly registered."""
    # Check if the op exists in torch.ops.vllm
    assert hasattr(torch.ops.vllm, 'rocm_aiter_rms_norm')
    # Check if the op is callable
    assert callable(torch.ops.vllm.rocm_aiter_rms_norm)


def test_rocm_aiter_rmsnorm2d_fwd_with_add_custom_op_registration():
    """Test that the fused add RMS norm custom op is correctly registered."""
    # Check if the op exists in torch.ops.vllm
    assert hasattr(torch.ops.vllm, 'rocm_aiter_rmsnorm2d_fwd_with_add')
    # Check if the op is callable
    assert callable(torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add)


def test_rocm_aiter_rms_norm_torch_compile_compatibility():
    """Test that the RMS norm op can be used with torch.compile."""
    device = "cuda"
    dtype = torch.bfloat16
    hidden_size = 2048
    batch_size = 8
    eps = 1e-6
    
    # Create test tensors
    x = torch.randn((batch_size, hidden_size), dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    
    # Define a function that uses the op
    def rms_norm_fn(x, weight, eps):
        return torch.ops.vllm.rocm_aiter_rms_norm(x, weight, eps)
    
    # Compile the function
    compiled_fn = torch.compile(
        rms_norm_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False
    )
    
    # Run both compiled and uncompiled versions
    original_output = rms_norm_fn(x, weight, eps)
    compiled_output = compiled_fn(x, weight, eps)
    
    # Verify results match
    assert torch.allclose(original_output, compiled_output, rtol=1e-3, atol=1e-4), \
        "Compiled and uncompiled RMS norm results don't match"


def test_rocm_aiter_rmsnorm2d_fwd_with_add_torch_compile_compatibility():
    """Test that the fused add RMS norm op can be used with torch.compile."""
    device = "cuda"
    dtype = torch.bfloat16
    hidden_size = 2048
    batch_size = 8
    eps = 1e-6
    
    # Create test tensors
    x = torch.randn((batch_size, hidden_size), dtype=dtype, device=device)
    residual = torch.randn((batch_size, hidden_size), dtype=dtype, device=device)
    weight = torch.randn(hidden_size, dtype=dtype, device=device)
    
    # Define a function that uses the op
    def fused_add_rms_norm_fn(x, residual, weight, eps):
        return torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add(x, residual, weight, eps)
    
    # Compile the function
    compiled_fn = torch.compile(
        fused_add_rms_norm_fn,
        fullgraph=True,
        backend="inductor",
        mode="reduce-overhead",
        dynamic=False
    )
    
    # Run both compiled and uncompiled versions
    original_output, original_residual = fused_add_rms_norm_fn(
        x.clone(), residual.clone(), weight, eps
    )
    compiled_output, compiled_residual = compiled_fn(
        x.clone(), residual.clone(), weight, eps
    )
    
    # Verify results match
    assert torch.allclose(original_output, compiled_output, rtol=1e-3, atol=1e-4), \
        "Compiled and uncompiled fused add RMS norm output don't match"
    assert torch.allclose(original_residual, compiled_residual, rtol=1e-3, atol=1e-4), \
        "Compiled and uncompiled fused add RMS norm residual don't match"


@pytest.mark.parametrize("add_residual", [True, False])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("aiter_enabled", [True, False])
def test_dispatch_cuda_rmsnorm_func(add_residual, dtype, aiter_enabled):
    """Comprehensive test of dispatcher function with all combinations."""
    
    with mock.patch('vllm.model_executor.layers.layernorm.is_rocm_aiter_rmsnorm_enabled', return_value=aiter_enabled):
        func = dispatch_cuda_rmsnorm_func(add_residual=add_residual, dtype=dtype)
        
        # Determine expected function based on logic
        if aiter_enabled and dtype != torch.float32:
            if add_residual:
                expected_func = torch.ops.vllm.rocm_aiter_rmsnorm2d_fwd_with_add
            else:
                expected_func = torch.ops.vllm.rocm_aiter_rms_norm
        else:
            if add_residual:
                expected_func = fused_add_rms_norm
            else:
                expected_func = rms_norm
        
        assert func == expected_func, \
            f"Dispatcher returned wrong function for add_residual={add_residual}, dtype={dtype}, aiter_enabled={aiter_enabled}"
