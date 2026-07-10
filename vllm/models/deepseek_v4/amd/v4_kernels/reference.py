# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reference helpers for ROCm DeepSeek-V4 paged ATOM kernels.

These helpers are intentionally torch-only and are used by unit/reference
paths. Production decode/prefill dispatches through the paged kernels in this
package, not the old standalone unpaged sparse-attention kernel.
"""

from __future__ import annotations

import torch


def sparse_attn_ragged_torch(
    q: torch.Tensor,
    kv: torch.Tensor,
    attn_sink: torch.Tensor,
    topk_idxs: torch.Tensor,
    softmax_scale: float,
) -> torch.Tensor:
    """Torch reference for sparse MQA with per-head attention sink.

    Args:
        q: [num_tokens, H, D]
        kv: [total_kv, D]
        attn_sink: [H]
        topk_idxs: [num_tokens, K], with -1 entries skipped.
    """
    T, H, D = q.shape
    K = topk_idxs.shape[-1]
    assert kv.dim() == 2
    assert kv.shape[-1] == D
    assert attn_sink.shape == (H,)
    assert topk_idxs.shape == (T, K)

    out_dtype = q.dtype
    device = q.device

    valid = topk_idxs != -1
    safe_idxs = topk_idxs.clamp(min=0).long()
    kv_gathered = kv[safe_idxs]  # [T, K, D]

    kv_f32 = kv_gathered.float()
    kv_f32 = torch.where(
        valid.unsqueeze(-1),
        kv_f32,
        torch.zeros((), dtype=kv_f32.dtype, device=device),
    )

    scores = torch.einsum("thd,tkd->thk", q.float(), kv_f32) * float(softmax_scale)
    scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))

    sink = attn_sink.float().view(1, H, 1).expand(T, H, 1)
    combined = torch.cat([scores, sink], dim=-1)
    cmax = combined.amax(dim=-1, keepdim=True)
    cmax = torch.where(
        cmax == float("-inf"),
        torch.zeros((), dtype=cmax.dtype, device=device),
        cmax,
    )
    weights = (combined - cmax).exp()
    weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    weights_kv = weights[..., :K]

    out = torch.einsum("thk,tkd->thd", weights_kv, kv_f32)
    return out.to(out_dtype)
