# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ATOM-style DeepSeek-V4 ROCm kernels vendored for vLLM integration."""

from vllm.models.deepseek_v4.amd.v4_kernels.paged_decode import (
    sparse_attn_v4_paged_decode,
    sparse_attn_v4_paged_decode_kv_splits,
    sparse_attn_v4_paged_decode_reference,
    sparse_attn_v4_paged_decode_split_workspace_mode,
    sparse_attn_v4_paged_decode_split_kv,
    sparse_attn_v4_paged_decode_split_kv_reference,
)
from vllm.models.deepseek_v4.amd.v4_kernels.paged_decode_indices import (
    write_v4_paged_decode_indices,
    write_v4_paged_decode_indices_reference,
)
from vllm.models.deepseek_v4.amd.v4_kernels.paged_prefill import (
    sparse_attn_v4_paged_prefill,
    sparse_attn_v4_paged_prefill_reference,
    sparse_attn_v4_paged_prefill_split_kv,
)
from vllm.models.deepseek_v4.amd.v4_kernels.paged_prefill_indices import (
    write_v4_paged_prefill_indices,
    write_v4_paged_prefill_indices_reference,
)
from vllm.models.deepseek_v4.amd.v4_kernels.compress_plan import (
    CompressPlan,
    make_compress_plans,
)
from vllm.models.deepseek_v4.amd.v4_kernels.csa_translate_pack import (
    csa_translate_pack,
    csa_translate_pack_reference,
)
from vllm.models.deepseek_v4.amd.v4_kernels.fused_compress import (
    fused_compress_attn,
    fused_compress_attn_reference,
)
from vllm.models.deepseek_v4.amd.v4_kernels.inverse_rope import (
    inverse_rope_inplace,
)
from vllm.models.deepseek_v4.amd.v4_kernels.state_writes import (
    swa_write,
    swa_write_reference,
    update_compressor_states,
    update_compressor_states_reference,
)

__all__ = [
    "sparse_attn_v4_paged_decode",
    "sparse_attn_v4_paged_decode_kv_splits",
    "sparse_attn_v4_paged_decode_reference",
    "sparse_attn_v4_paged_decode_split_workspace_mode",
    "sparse_attn_v4_paged_decode_split_kv",
    "sparse_attn_v4_paged_decode_split_kv_reference",
    "sparse_attn_v4_paged_prefill",
    "sparse_attn_v4_paged_prefill_reference",
    "sparse_attn_v4_paged_prefill_split_kv",
    "write_v4_paged_decode_indices",
    "write_v4_paged_decode_indices_reference",
    "write_v4_paged_prefill_indices",
    "write_v4_paged_prefill_indices_reference",
    "CompressPlan",
    "make_compress_plans",
    "csa_translate_pack",
    "csa_translate_pack_reference",
    "fused_compress_attn",
    "fused_compress_attn_reference",
    "inverse_rope_inplace",
    "swa_write",
    "swa_write_reference",
    "update_compressor_states",
    "update_compressor_states_reference",
]
