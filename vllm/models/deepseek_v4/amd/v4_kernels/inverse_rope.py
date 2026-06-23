# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ATOM-style in-place inverse RoPE for DeepSeek-V4 ROCm attention output."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _inverse_rope_gptj_inplace_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    positions_ptr,
    stride_x_t,
    stride_x_h,
    stride_x_d,
    stride_cos_t,
    stride_cos_d,
    T: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_RD: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    head_id = tl.program_id(0)
    token_block = tl.program_id(1)

    token_offsets = token_block * BLOCK_T + tl.arange(0, BLOCK_T)
    dim_offsets = tl.arange(0, BLOCK_RD)
    token_mask = token_offsets < T
    half_offsets = dim_offsets // 2
    dim_mask = dim_offsets < ROPE_HEAD_DIM

    positions = tl.load(positions_ptr + token_offsets, mask=token_mask, other=0)
    cos_offsets = (
        positions[:, None] * stride_cos_t
        + half_offsets[None, :] * stride_cos_d
    )
    cos_mask = token_mask[:, None] & dim_mask[None, :]
    cos = tl.load(cos_ptr + cos_offsets, mask=cos_mask, other=1.0)
    sin = tl.load(sin_ptr + cos_offsets, mask=cos_mask, other=0.0)

    x_offsets = (
        token_offsets[:, None] * stride_x_t
        + head_id * stride_x_h
        + dim_offsets[None, :] * stride_x_d
    )
    x = tl.load(x_ptr + x_offsets, mask=cos_mask, other=0.0).to(tl.float32)

    # GPT-J inverse RoPE over adjacent pairs:
    # out[2i] = x[2i] * cos + x[2i + 1] * sin
    # out[2i + 1] = x[2i + 1] * cos - x[2i] * sin
    x_sin = x * sin
    even_dim = (dim_offsets % 2 == 0)[None, :]
    rotated = tl.where(even_dim, -x_sin, x_sin)
    rotated = tl.reshape(rotated, (BLOCK_T, BLOCK_HALF, 2))
    rotated = tl.flip(rotated, 2)
    rotated = tl.reshape(rotated, (BLOCK_T, BLOCK_RD))
    out = x * cos + rotated

    tl.store(x_ptr + x_offsets, out, mask=cos_mask)


def inverse_rope_inplace(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    """Apply ATOM-style in-place inverse GPT-J RoPE to ``x``.

    ``x`` is the trailing RoPE slice of the attention output with shape
    ``[num_tokens, num_heads, rope_head_dim]``. ``cos`` and ``sin`` may be
    either flattened ``[max_position, rope_head_dim // 2]`` tables or higher
    rank tables whose last dimension is ``rope_head_dim // 2``.
    """
    if x.numel() == 0:
        return
    if x.dim() != 3:
        raise RuntimeError(f"inverse_rope_inplace expects [T, H, RD], got {x.shape}")
    if positions.dim() != 1:
        raise RuntimeError(
            f"inverse_rope_inplace expects 1-D positions, got {positions.shape}"
        )
    T, H, rope_head_dim = x.shape
    if int(positions.shape[0]) < T:
        raise RuntimeError(
            "inverse_rope_inplace positions shorter than token dimension: "
            f"{positions.shape[0]} < {T}"
        )
    if rope_head_dim % 2 != 0:
        raise RuntimeError(
            f"inverse_rope_inplace expects even rope_head_dim, got {rope_head_dim}"
        )

    half = rope_head_dim // 2
    cos_flat = cos.reshape(cos.shape[0], -1)
    sin_flat = sin.reshape(sin.shape[0], -1)
    if cos_flat.shape[-1] < half or sin_flat.shape[-1] < half:
        raise RuntimeError(
            "inverse_rope_inplace cos/sin tables are too narrow: "
            f"cos={cos.shape}, sin={sin.shape}, required half={half}"
        )

    block_t = 32
    block_rd = triton.next_power_of_2(rope_head_dim)
    _inverse_rope_gptj_inplace_kernel[(H, triton.cdiv(T, block_t))](
        x,
        cos_flat,
        sin_flat,
        positions,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        cos_flat.stride(0),
        cos_flat.stride(1),
        T,
        ROPE_HEAD_DIM=rope_head_dim,
        BLOCK_T=block_t,
        BLOCK_RD=block_rd,
        BLOCK_HALF=block_rd // 2,
        num_warps=4,
    )
