# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

FP8_DTYPE = torch.float8_e4m3fnuz

TORCH_TRITON_DTYPE_MAPPING = {torch.float8_e4m3fnuz: tl.float8e4b8}

Config = tuple[int, int, int, int]

_HW_ID: str = (
    current_platform.get_device_name(0) if current_platform.is_cuda_alike() else "cpu"
)

_TUNED_CONFIGS: dict[tuple[str, int, int, int], Config] = {
    ("AMD_Instinct_MI300X", 1, 1536, 7168): (16, 16, 4, 2),
    ("AMD_Instinct_MI300X", 1, 2560, 2048): (16, 16, 4, 2),
    ("AMD_Instinct_MI300X", 1, 4096, 4096): (16, 16, 4, 2),
    ("AMD_Instinct_MI300X", 1, 4096, 7168): (16, 8, 4, 2),
    ("AMD_Instinct_MI300X", 1, 7168, 16384): (16, 8, 4, 2),
    ("AMD_Instinct_MI300X", 1, 8192, 8192): (16, 16, 4, 2),
    ("AMD_Instinct_MI300X", 1, 11008, 4096): (16, 16, 4, 2),
    ("AMD_Instinct_MI300X", 2, 4096, 7168): (16, 8, 4, 2),
    ("AMD_Instinct_MI300X", 4, 4096, 7168): (16, 8, 4, 2),
    ("AMD_Instinct_MI300X", 8, 4096, 7168): (16, 8, 4, 2),
    ("AMD_Instinct_MI300X", 16, 4096, 7168): (16, 8, 4, 2),
    ("AMD_Instinct_MI300X", 16, 7168, 2048): (16, 8, 4, 2),
    ("AMD_Instinct_MI300X", 32, 4096, 7168): (16, 16, 4, 2),
    ("AMD_Instinct_MI300X", 64, 4096, 7168): (16, 16, 4, 2),
    ("AMD_Instinct_MI300X", 128, 4096, 7168): (16, 16, 4, 2),
    ("AMD_Instinct_MI300X", 128, 7168, 2048): (32, 16, 4, 2),
    ("AMD_Instinct_MI300X", 256, 4096, 7168): (16, 16, 4, 2),
    ("AMD_Instinct_MI300X", 512, 4096, 7168): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 1024, 4096, 7168): (128, 8, 8, 2),
    ("AMD_Instinct_MI300X", 2048, 4096, 7168): (128, 16, 4, 1),
    ("AMD_Instinct_MI300X", 4096, 1536, 7168): (64, 16, 4, 1),
    ("AMD_Instinct_MI300X", 4096, 2048, 2048): (64, 16, 8, 1),
    ("AMD_Instinct_MI300X", 4096, 2560, 2048): (32, 16, 4, 2),
    ("AMD_Instinct_MI300X", 4096, 4096, 4096): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 4096, 7168): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 4096, 11008): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 4096, 14336): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 7168, 16384): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 7168, 18432): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 8192, 8192): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 8192, 28672): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 11008, 4096): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 14336, 4096): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 24576, 1536): (64, 16, 4, 1),
    ("AMD_Instinct_MI300X", 4096, 28672, 8192): (64, 16, 8, 2),
    ("AMD_Instinct_MI300X", 4096, 32768, 512): (64, 16, 4, 1),
    ("AMD_Instinct_MI300X", 4096, 36864, 7168): (64, 16, 4, 1),
}


_DEFAULT_CONFIG_BY_HW: dict[str, Config] = {
    "AMD_Instinct_MI300X": (64, 16, 8, 2),
}


_GLOBAL_FALLBACK_CONFIG: Config = (64, 8, 8, 2)


def _pick_config(M: int, N: int, K: int) -> Config:
    if M >= 4096:
        M_bucket = 4096
    elif M <= 1:
        M_bucket = 1
    else:
        M_bucket = 1 << (M - 1).bit_length()

    cfg = _TUNED_CONFIGS.get((_HW_ID, M_bucket, N, K))
    if cfg is not None:
        return cfg
    return _DEFAULT_CONFIG_BY_HW.get(_HW_ID, _GLOBAL_FALLBACK_CONFIG)


@triton.jit
def _fused_block_quant_gemm(
    a_ptr,
    b_ptr,
    c_ptr,
    bs_ptr,
    M,
    N,
    K,
    stride_a_m,
    stride_a_k,
    stride_b_n,
    stride_b_k,
    stride_c_m,
    stride_c_n,
    stride_bs_n,
    stride_bs_k,
    TILE_SIZE_M: tl.constexpr,
    TILE_SIZE_N: tl.constexpr,
    TILE_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    FP8_DTYPE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BOUNDARY_CHECK: tl.constexpr,
) -> None:
    tl.assume(M >= 1)
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, TILE_SIZE_M)
    num_pid_n = tl.cdiv(N, TILE_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % num_pid_in_group) % group_size_m
    pid_n = (pid % num_pid_in_group) // group_size_m

    offset_m = pid_m * TILE_SIZE_M + tl.arange(0, TILE_SIZE_M)
    offset_n = pid_n * TILE_SIZE_N + tl.arange(0, TILE_SIZE_N)
    offset_k = tl.arange(0, TILE_SIZE_K)

    a_ptrs = a_ptr + offset_m[:, None] * stride_a_m + offset_k[None, :] * stride_a_k
    b_ptrs = b_ptr + offset_k[:, None] * stride_b_k + offset_n[None, :] * stride_b_n

    num_k_blocks = tl.cdiv(K, TILE_SIZE_K)

    acc = tl.zeros((TILE_SIZE_M, TILE_SIZE_N), dtype=tl.float32)

    for k_block in range(num_k_blocks):
        k_start = k_block * TILE_SIZE_K

        if BOUNDARY_CHECK:
            a_mask = (offset_m[:, None] < M) & (k_start + offset_k[None, :] < K)
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        else:
            a = tl.load(a_ptrs)

        amax = tl.max(tl.abs(a), axis=1)
        amax = tl.maximum(amax, 1e-10)
        s_a = amax / FP8_MAX
        s_a_inv = FP8_MAX / amax
        a_q = a * s_a_inv[:, None]
        a_q = tl.minimum(tl.maximum(a_q, -FP8_MAX), FP8_MAX)
        a_q = a_q.to(FP8_DTYPE)

        if BOUNDARY_CHECK:
            b_mask = (offset_n[None, :] < N) & (k_start + offset_k[:, None] < K)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        else:
            b = tl.load(b_ptrs)
        s_w = tl.load(bs_ptr + pid_n * stride_bs_n + k_block * stride_bs_k)

        group_acc = tl.dot(a_q, b)
        acc += group_acc * s_a[:, None] * s_w

        a_ptrs += TILE_SIZE_K * stride_a_k
        b_ptrs += TILE_SIZE_K * stride_b_k

    c_ptrs = c_ptr + offset_m[:, None] * stride_c_m + offset_n[None, :] * stride_c_n
    if BOUNDARY_CHECK:
        c_mask = (offset_m[:, None] < M) & (offset_n[None, :] < N)
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=c_mask)
    else:
        tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty))


def _fused_block_fp8_quant_gemm_impl(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    group_size: int,
    fp8_dtype: torch.dtype,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    assert out_dtype in [torch.float16, torch.bfloat16]
    assert group_size in [64, 128]

    M, K = A.shape
    N = B.shape[0]
    C = torch.empty(M, N, dtype=out_dtype, device=A.device)

    tile_m, group_m, num_warps, num_stages = _pick_config(M, N, K)
    boundary_check = not (
        M % tile_m == 0 and N % group_size == 0 and K % group_size == 0
    )

    grid = (triton.cdiv(M, tile_m) * triton.cdiv(N, group_size),)

    _fused_block_quant_gemm[grid](
        A,
        B,
        C,
        Bs,
        M,
        N,
        K,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        C.stride(0),
        C.stride(1),
        Bs.stride(0),
        Bs.stride(1),
        TILE_SIZE_M=tile_m,
        TILE_SIZE_N=group_size,
        TILE_SIZE_K=group_size,
        GROUP_SIZE_M=group_m,
        FP8_DTYPE=TORCH_TRITON_DTYPE_MAPPING[fp8_dtype],
        FP8_MAX=224.0,
        BOUNDARY_CHECK=boundary_check,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return C


def _fused_block_fp8_quant_gemm_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    Bs: torch.Tensor,
    group_size: int,
    fp8_dtype: torch.dtype,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    M = A.shape[0]
    N = B.shape[0]
    return torch.empty(M, N, dtype=out_dtype, device=A.device)


direct_register_custom_op(
    "fused_block_fp8_quant_gemm",
    _fused_block_fp8_quant_gemm_impl,
    fake_impl=_fused_block_fp8_quant_gemm_fake,
)
