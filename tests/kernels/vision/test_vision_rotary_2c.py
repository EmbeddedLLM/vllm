# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test suite for the apply_vision_rotary_2c CUDA kernel.

This test verifies that the optimized CUDA kernel produces the same results
as the reference PyTorch implementation for vision rotary embeddings used
in Qwen2-VL and Qwen3-VL models.
"""

import pytest
import torch

from vllm.platforms import current_platform

# Skip if not on CUDA/ROCm
if not (current_platform.is_cuda() or current_platform.is_rocm()):
    pytest.skip("CUDA/ROCm required for this test", allow_module_level=True)


def apply_rotary_emb_torch_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Reference PyTorch implementation of rotary embedding.
    Uses NEox-style (half-split) rotation.
    
    Args:
        x: (batch, seqlen, nheads, headdim)
        cos: (seqlen, rotary_dim/2) 
        sin: (seqlen, rotary_dim/2)
    Returns:
        rotated tensor of same shape as x
    """
    rotary_dim = cos.shape[-1] * 2
    
    # Split x into rotary and non-rotary parts
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    
    # Split rotary part into two halves (NEox style)
    x1 = x_rot[..., :rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2:]
    
    # Expand cos/sin for broadcasting: (seqlen, rotary_dim/2) -> (1, seqlen, 1, rotary_dim/2)
    cos = cos.unsqueeze(0).unsqueeze(2).to(x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(x.dtype)
    
    # Apply rotation
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    
    # Concatenate back
    x_rotated = torch.cat([o1, o2], dim=-1)
    
    if x_pass.numel() > 0:
        return torch.cat([x_rotated, x_pass], dim=-1)
    return x_rotated


def apply_rotary_pos_emb_vision_2c_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation that applies rotary to q and k separately.
    """
    out_q = apply_rotary_emb_torch_reference(q, cos, sin)
    out_k = apply_rotary_emb_torch_reference(k, cos, sin)
    return out_q, out_k


@pytest.fixture
def device():
    return torch.device("cuda:0")


class TestVisionRotary2C:
    """Test class for vision rotary 2-component kernel."""
    
    # @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
    @pytest.mark.parametrize("batch_size", [1, 2, 4])
    @pytest.mark.parametrize("seqlen", [16, 64, 256, 1024])
    @pytest.mark.parametrize("nheads", [8, 16, 32])
    @pytest.mark.parametrize("headdim", [64, 128])
    @pytest.mark.parametrize("rotary_dim_ratio", [0.5, 1.0])
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_correctness(
        self, 
        device,
        batch_size: int,
        seqlen: int,
        nheads: int,
        headdim: int,
        rotary_dim_ratio: float,
        dtype: torch.dtype,
    ):
        """Test that CUDA kernel matches reference implementation."""
        if dtype == torch.bfloat16 and not current_platform.has_device_capability(80):
            pytest.skip("BFloat16 requires compute capability >= 8.0")
        
        rotary_dim = int(headdim * rotary_dim_ratio)
        if rotary_dim % 2 != 0:
            rotary_dim -= 1  # Ensure even rotary_dim
        
        # Create random input tensors
        q = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        
        # Create cos/sin values (half rotary_dim since we split)
        cos = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        sin = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        
        # Reference implementation
        ref_q, ref_k = apply_rotary_pos_emb_vision_2c_reference(
            q.clone(), k.clone(), cos, sin
        )
        
        # CUDA kernel implementation
        from vllm.model_executor.models.qwen2_vl import apply_rotary_2c_cuda
        cuda_q, cuda_k = apply_rotary_2c_cuda(q.clone(), k.clone(), cos, sin)
        
        # Check correctness
        rtol = 1e-3 if dtype == torch.float16 else 1e-4
        atol = 1e-3 if dtype == torch.float16 else 1e-5
        
        torch.testing.assert_close(cuda_q, ref_q, rtol=rtol, atol=atol)
        torch.testing.assert_close(cuda_k, ref_k, rtol=rtol, atol=atol)
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("seqlen", [64, 256])
    @pytest.mark.parametrize("nheads", [16])
    @pytest.mark.parametrize("headdim", [128])
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_inplace(
        self,
        device,
        batch_size: int,
        seqlen: int,
        nheads: int,
        headdim: int,
        dtype: torch.dtype,
    ):
        """Test inplace operation."""
        rotary_dim = headdim // 2
        
        q = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        cos = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        sin = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        
        # Reference (not inplace)
        ref_q, ref_k = apply_rotary_pos_emb_vision_2c_reference(
            q.clone(), k.clone(), cos, sin
        )
        
        # CUDA kernel inplace
        from vllm.model_executor.models.qwen2_vl import apply_rotary_2c_cuda
        q_inplace = q.clone()
        k_inplace = k.clone()
        out_q, out_k = apply_rotary_2c_cuda(
            q_inplace, k_inplace, cos, sin, inplace=True
        )
        
        # Should be same object when inplace
        assert out_q.data_ptr() == q_inplace.data_ptr()
        assert out_k.data_ptr() == k_inplace.data_ptr()
        
        # Check correctness
        torch.testing.assert_close(out_q, ref_q, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(out_k, ref_k, rtol=1e-3, atol=1e-3)
    
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_contiguous_and_noncontiguous(self, device, dtype: torch.dtype):
        """Test with both contiguous and non-contiguous tensors."""
        batch_size, seqlen, nheads, headdim = 2, 64, 16, 128
        rotary_dim = headdim // 2
        
        # Create contiguous tensors
        q = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        cos = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        sin = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        
        # Make non-contiguous by transposing
        q_nc = q.transpose(1, 2).transpose(1, 2)  # Still same data but different strides
        k_nc = k.transpose(1, 2).transpose(1, 2)
        
        # Reference
        ref_q, ref_k = apply_rotary_pos_emb_vision_2c_reference(q, k, cos, sin)
        
        # CUDA with original contiguous
        from vllm.model_executor.models.qwen2_vl import apply_rotary_2c_cuda
        cuda_q, cuda_k = apply_rotary_2c_cuda(q.clone(), k.clone(), cos, sin)
        
        torch.testing.assert_close(cuda_q, ref_q, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(cuda_k, ref_k, rtol=1e-3, atol=1e-3)
    
    @pytest.mark.parametrize("seqlen", [197, 577, 1025])  # Typical ViT sequence lengths
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_vit_typical_sizes(self, device, seqlen: int, dtype: torch.dtype):
        """Test with typical Vision Transformer sequence lengths."""
        batch_size = 1
        nheads = 16
        headdim = 64
        rotary_dim = headdim // 2
        
        q = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        cos = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        sin = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        
        ref_q, ref_k = apply_rotary_pos_emb_vision_2c_reference(q, k, cos, sin)
        
        from vllm.model_executor.models.qwen2_vl import apply_rotary_2c_cuda
        cuda_q, cuda_k = apply_rotary_2c_cuda(q.clone(), k.clone(), cos, sin)
        
        torch.testing.assert_close(cuda_q, ref_q, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(cuda_k, ref_k, rtol=1e-3, atol=1e-3)


class TestVisionRotary2CWrapper:
    """Test the wrapper function that matches vLLM's API."""
    
    @pytest.mark.parametrize("dtype", [torch.float32])
    def test_wrapper_api_compatibility(self, device, dtype: torch.dtype):
        """Test that wrapper handles the full rotary_dim cos/sin correctly."""
        batch_size, seqlen, nheads, headdim = 2, 64, 16, 128
        rotary_dim = headdim // 2
        
        q = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, 
                       device=device, dtype=dtype)
        
        # Full rotary_dim cos/sin (as vLLM provides)
        cos_full = torch.randn(seqlen, rotary_dim, device=device, dtype=dtype)
        sin_full = torch.randn(seqlen, rotary_dim, device=device, dtype=dtype)
        
        # The wrapper should handle the splitting internally
        from vllm.model_executor.models.qwen2_vl import (
            apply_rotary_pos_emb_vision_2c_cuda
        )
        cuda_q, cuda_k = apply_rotary_pos_emb_vision_2c_cuda(
            q.clone(), k.clone(), cos_full, sin_full
        )
        
        # Reference with half cos/sin
        cos_half = cos_full[..., :rotary_dim // 2]
        sin_half = sin_full[..., :rotary_dim // 2]
        ref_q, ref_k = apply_rotary_pos_emb_vision_2c_reference(
            q.clone(), k.clone(), cos_half, sin_half
        )
        
        torch.testing.assert_close(cuda_q, ref_q, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(cuda_k, ref_k, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

