#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script for vision rotary embedding kernels.

Compares:
1. Reference PyTorch implementation (with concatenation)
2. flash_attn Triton kernel (ROCm path when torch.compile disabled)
3. Optimized CUDA kernel (processes q and k simultaneously)

Usage:
    python benchmarks/kernels/benchmark_vision_rotary_2c.py

Requirements:
    - vLLM built with CUDA support
    - CUDA/ROCm device available
    - flash_attn package installed (for Triton rotary benchmark)
"""

import argparse
from collections import defaultdict
from typing import Callable

import torch

# Check CUDA availability
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark")

# Import flash_attn triton rotary (ROCm path)
from flash_attn.ops.triton.rotary import apply_rotary as flash_attn_apply_rotary


def apply_rotary_emb_torch_reference(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reference PyTorch implementation of rotary embedding."""
    rotary_dim = cos.shape[-1] * 2
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    
    x1 = x_rot[..., :rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2:]
    
    cos = cos.unsqueeze(0).unsqueeze(2).to(x.dtype)
    sin = sin.unsqueeze(0).unsqueeze(2).to(x.dtype)
    
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    
    x_rotated = torch.cat([o1, o2], dim=-1)
    
    if x_pass.numel() > 0:
        return torch.cat([x_rotated, x_pass], dim=-1)
    return x_rotated


def apply_rotary_concat_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reference implementation that concatenates q and k before applying rotary.
    This is similar to what vLLM currently does.
    """
    # Concatenate q and k along batch dimension
    qk = torch.cat([q, k], dim=0)
    
    # Apply rotary to concatenated tensor
    qk_rotated = apply_rotary_emb_torch_reference(qk, cos, sin)
    
    # Split back
    q_rot, k_rot = torch.chunk(qk_rotated, 2, dim=0)
    return q_rot, k_rot


def apply_rotary_separate_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference implementation that applies rotary to q and k separately."""
    q_rot = apply_rotary_emb_torch_reference(q, cos, sin)
    k_rot = apply_rotary_emb_torch_reference(k, cos, sin)
    return q_rot, k_rot


def apply_rotary_flash_attn_triton_concat(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    flash_attn Triton kernel implementation (ROCm path).
    Uses concatenation similar to vLLM's current approach.
    
    flash_attn.ops.triton.rotary.apply_rotary signature:
        apply_rotary(x, cos, sin, interleaved=False, inplace=False, ...)
    
    Where:
        x: (batch, seqlen, nheads, headdim)
        cos: (seqlen, rotary_dim) or (batch, seqlen, rotary_dim)
        sin: (seqlen, rotary_dim) or (batch, seqlen, rotary_dim)
    """
    # Concatenate q and k along batch dimension
    qk = torch.cat([q, k], dim=0)
    
    # flash_attn expects cos/sin with full rotary_dim, not half
    # cos_half is [seqlen, rotary_dim/2], we need [seqlen, rotary_dim]
    rotary_dim_half = cos.shape[-1]
    cos_full = torch.cat([cos, cos], dim=-1)  # [seqlen, rotary_dim]
    sin_full = torch.cat([sin, sin], dim=-1)  # [seqlen, rotary_dim]
    
    # Apply rotary using flash_attn triton kernel
    # interleaved=False means NEox style (split in half)
    qk_rotated = flash_attn_apply_rotary(
        qk, cos_full, sin_full, interleaved=False, inplace=False
    )
    
    # Split back
    q_rot, k_rot = torch.chunk(qk_rotated, 2, dim=0)
    return q_rot, k_rot


def apply_rotary_flash_attn_triton_separate(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    flash_attn Triton kernel applied to q and k separately.
    This avoids the concatenation overhead.
    """
    # flash_attn expects cos/sin with full rotary_dim
    cos_full = torch.cat([cos, cos], dim=-1)
    sin_full = torch.cat([sin, sin], dim=-1)
    
    q_rot = flash_attn_apply_rotary(
        q, cos_full, sin_full, interleaved=False, inplace=False
    )
    k_rot = flash_attn_apply_rotary(
        k, cos_full, sin_full, interleaved=False, inplace=False
    )
    return q_rot, k_rot


def apply_rotary_cuda_kernel(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimized CUDA kernel implementation."""
    from vllm.model_executor.models.qwen2_vl import apply_rotary_2c_cuda
    return apply_rotary_2c_cuda(q, k, cos, sin)


def benchmark_function(
    fn: Callable,
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    warmup_iters: int = 10,
    bench_iters: int = 100,
) -> tuple[float, float]:
    """
    Benchmark a rotary embedding function.
    
    Returns:
        tuple of (mean_time_ms, std_time_ms)
    """
    # Warmup
    for _ in range(warmup_iters):
        _ = fn(q.clone(), k.clone(), cos, sin)
    
    torch.cuda.synchronize()
    
    # Benchmark
    times = []
    for _ in range(bench_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        _ = fn(q.clone(), k.clone(), cos, sin)
        end.record()
        
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    
    mean_time = sum(times) / len(times)
    std_time = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return mean_time, std_time


def run_benchmarks(
    batch_sizes: list[int],
    seqlens: list[int],
    nheads: list[int],
    headdims: list[int],
    rotary_dim_ratios: list[float],
    dtypes: list[torch.dtype],
    warmup_iters: int = 10,
    bench_iters: int = 100,
):
    """Run comprehensive benchmarks."""
    device = torch.device("cuda:0")
    
    results = defaultdict(list)
    
    implementations = {
        "PyTorch Concat": apply_rotary_concat_reference,
        "PyTorch Separate": apply_rotary_separate_reference,
        "FlashAttn Triton Concat": apply_rotary_flash_attn_triton_concat,
        "FlashAttn Triton Separate": apply_rotary_flash_attn_triton_separate,
        "CUDA Kernel": apply_rotary_cuda_kernel,
    }
    
    print("=" * 120)
    print("Vision Rotary Embedding Benchmark")
    print("=" * 120)
    print(f"Warmup iterations: {warmup_iters}")
    print(f"Benchmark iterations: {bench_iters}")
    print("Implementations:")
    print("  - PyTorch Concat: Reference PyTorch with Q/K concatenation")
    print("  - PyTorch Separate: Reference PyTorch processing Q/K separately")
    print("  - FlashAttn Triton Concat: flash_attn.ops.triton.rotary with Q/K concatenation (ROCm path)")
    print("  - FlashAttn Triton Separate: flash_attn.ops.triton.rotary processing Q/K separately")
    print("  - CUDA Kernel: Optimized kernel processing Q/K simultaneously")
    print("=" * 120)
    
    total_configs = (len(batch_sizes) * len(seqlens) * len(nheads) * 
                    len(headdims) * len(rotary_dim_ratios) * len(dtypes))
    config_idx = 0
    
    for dtype in dtypes:
        dtype_name = str(dtype).split('.')[-1]
        
        for batch_size in batch_sizes:
            for seqlen in seqlens:
                for nhead in nheads:
                    for headdim in headdims:
                        for rotary_ratio in rotary_dim_ratios:
                            config_idx += 1
                            rotary_dim = int(headdim * rotary_ratio)
                            if rotary_dim % 2 != 0:
                                rotary_dim -= 1
                            
                            config_str = (
                                f"[{config_idx}/{total_configs}] "
                                f"B={batch_size}, S={seqlen}, H={nhead}, "
                                f"D={headdim}, R={rotary_dim}, dtype={dtype_name}"
                            )
                            print(f"\n{config_str}")
                            print("-" * 100)
                            
                            # Create input tensors
                            q = torch.randn(batch_size, seqlen, nhead, headdim,
                                          device=device, dtype=dtype)
                            k = torch.randn(batch_size, seqlen, nhead, headdim,
                                          device=device, dtype=dtype)
                            cos = torch.randn(seqlen, rotary_dim // 2,
                                            device=device, dtype=dtype)
                            sin = torch.randn(seqlen, rotary_dim // 2,
                                            device=device, dtype=dtype)
                            
                            times = {}
                            for name, fn in implementations.items():
                                try:
                                    mean_ms, std_ms = benchmark_function(
                                        fn, q, k, cos, sin,
                                        warmup_iters=warmup_iters,
                                        bench_iters=bench_iters,
                                    )
                                    times[name] = mean_ms
                                    print(f"  {name:28s}: {mean_ms:8.3f} ms ± {std_ms:6.3f} ms")
                                except Exception as e:
                                    print(f"  {name:28s}: ERROR - {e}")
                                    times[name] = float('inf')
                            
                            # Calculate speedups
                            print()
                            if "CUDA Kernel" in times and times["CUDA Kernel"] != float('inf'):
                                for baseline in ["PyTorch Concat", "FlashAttn Triton Concat"]:
                                    if baseline in times and times[baseline] != float('inf'):
                                        speedup = times[baseline] / times["CUDA Kernel"]
                                        print(f"  Speedup vs {baseline:24s}: {speedup:6.2f}x")
                            
                            results["config"].append(config_str)
                            for name, t in times.items():
                                results[name].append(t)
    
    print("\n" + "=" * 120)
    print("Summary")
    print("=" * 120)
    
    # Print summary statistics
    for impl in implementations:
        if impl in results and results[impl]:
            valid_times = [t for t in results[impl] if t != float('inf')]
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                print(f"{impl:28s}: avg = {avg_time:.3f} ms")
    
    # Calculate average speedups
    print("\nAverage Speedups (CUDA Kernel vs baselines):")
    for baseline in ["PyTorch Concat", "FlashAttn Triton Concat", "FlashAttn Triton Separate"]:
        if baseline in results and "CUDA Kernel" in results:
            valid_pairs = [
                (b, c) for b, c in zip(results[baseline], results["CUDA Kernel"])
                if b != float('inf') and c != float('inf')
            ]
            if valid_pairs:
                speedups = [b / c for b, c in valid_pairs]
                avg_speedup = sum(speedups) / len(speedups)
                max_speedup = max(speedups)
                min_speedup = min(speedups)
                print(f"  vs {baseline:28s}: avg={avg_speedup:.2f}x, min={min_speedup:.2f}x, max={max_speedup:.2f}x")
    
    return results


def run_qwen3_vl_benchmark():
    """Run benchmark with Qwen3-VL typical configurations."""
    print("\n" + "=" * 120)
    print("Qwen3-VL Typical Configurations Benchmark")
    print("=" * 120)
    
    # Qwen3-VL-72B vision encoder configurations
    # embed_dim=1536, num_heads=16, head_dim=96, rotary_dim=48
    configs = [
        # (batch, seqlen, nheads, headdim, rotary_dim_ratio)
        (1, 256, 16, 96, 0.5),    # Small image
        (1, 1024, 16, 96, 0.5),   # Medium image  
        (1, 4096, 16, 96, 0.5),   # Large image
        (1, 16384, 16, 96, 0.5),  # Very large image / video
        (4, 1024, 16, 96, 0.5),   # Batch of medium images
        (8, 256, 16, 96, 0.5),    # Larger batch of small images
    ]
    
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    
    implementations = {
        "PyTorch Concat": apply_rotary_concat_reference,
        "FlashAttn Triton": apply_rotary_flash_attn_triton_concat,
        "CUDA Kernel": apply_rotary_cuda_kernel,
    }
    
    header = f"{'Config':<40} {'PyTorch (ms)':<15} {'FlashAttn (ms)':<15} {'CUDA (ms)':<15} {'Speedup vs PT':<15} {'Speedup vs FA':<15}"
    print(header)
    print("-" * len(header))
    
    for batch_size, seqlen, nheads, headdim, rotary_ratio in configs:
        rotary_dim = int(headdim * rotary_ratio)
        
        q = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seqlen, nheads, headdim, device=device, dtype=dtype)
        cos = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        sin = torch.randn(seqlen, rotary_dim // 2, device=device, dtype=dtype)
        
        config_str = f"B={batch_size}, S={seqlen}, H={nheads}, D={headdim}"
        
        times = {}
        for name, fn in implementations.items():
            try:
                mean_ms, _ = benchmark_function(fn, q, k, cos, sin, 
                                               warmup_iters=10, bench_iters=50)
                times[name] = mean_ms
            except Exception as e:
                times[name] = float('inf')
        
        pytorch_time = times.get("PyTorch Concat", float('inf'))
        flashattn_time = times.get("FlashAttn Triton", float('inf'))
        cuda_time = times.get("CUDA Kernel", float('inf'))
        
        speedup_pt = pytorch_time / cuda_time if cuda_time > 0 else 0
        speedup_fa = flashattn_time / cuda_time if cuda_time > 0 else 0
        
        print(f"{config_str:<40} {pytorch_time:<15.3f} {flashattn_time:<15.3f} {cuda_time:<15.3f} {speedup_pt:<15.2f}x {speedup_fa:<15.2f}x")


def main():
    parser = argparse.ArgumentParser(description="Benchmark vision rotary embeddings")
    parser.add_argument("--quick", action="store_true", 
                       help="Run quick benchmark with fewer configurations")
    parser.add_argument("--qwen3-vl", action="store_true",
                       help="Run Qwen3-VL specific configurations")
    parser.add_argument("--warmup", type=int, default=10,
                       help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=100,
                       help="Number of benchmark iterations")
    args = parser.parse_args()
    
    if args.qwen3_vl:
        run_qwen3_vl_benchmark()
        return
    
    if args.quick:
        batch_sizes = [1, 4]
        seqlens = [256, 1024]
        nheads = [16]
        headdims = [64, 128]
        rotary_dim_ratios = [0.5]
        dtypes = [torch.float16]
    else:
        batch_sizes = [1, 2, 4, 8]
        seqlens = [64, 256, 1024, 4096]
        nheads = [8, 16, 32]
        headdims = [64, 96, 128]
        rotary_dim_ratios = [0.5, 1.0]
        dtypes = [torch.float16, torch.bfloat16]
    
    run_benchmarks(
        batch_sizes=batch_sizes,
        seqlens=seqlens,
        nheads=nheads,
        headdims=headdims,
        rotary_dim_ratios=rotary_dim_ratios,
        dtypes=dtypes,
        warmup_iters=args.warmup,
        bench_iters=args.iters,
    )
    
    # Always run Qwen3-VL specific benchmarks
    run_qwen3_vl_benchmark()


if __name__ == "__main__":
    main()
