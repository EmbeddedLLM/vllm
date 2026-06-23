# DeepSeek V4 ROCm KV Cache And Workspace Plan

Date: 2026-06-20

This note describes how the ROCm DeepSeek-V4 ATOM integration should use
vLLM's KV-cache and workspace systems while preserving the existing CUDA path.

## Current vLLM Contracts

### KV Cache

vLLM's scheduler owns request-to-block allocation through generic KV-cache
specs and block tables. Model code should describe its cache shape through
`KVCacheSpec` fields; worker code should not need to know DeepSeek-V4 types.

Current ROCm DSV4 custom fields:

- `fixed_prefix_size_bytes`
- `requires_strided_kv_cache_view`
- `inner_block_stride_bytes`
- `DeepseekV4AtomMLAAttentionSpec.atom_swa_prefix_bytes`
- `DeepseekV4AtomMLAAttentionSpec.atom_swa_pages`
- `DeepseekV4AtomMLAAttentionSpec.atom_compressed_layout`
- `DeepseekV4AtomMLAAttentionSpec.atom_compressed_scale_bytes_per_page`

The current vLLM-owned ROCm packed layout is:

- fixed SWA prefix before the paged tail;
- compressed tail pages shaped as `uint8 [num_blocks, k_per_block, 584]`;
- layout string `fp8_ds_mla`;
- no sidecar scale tensor because each 584-byte slot embeds 8 UE8M0 scale
  bytes.

`kv_cache_utils._get_kv_cache_config_deepseek_v4` already reserves the fixed
prefix and emits a per-layer KV tensor with `fixed_prefix_size`. The generic
worker reshaping path consumes the generic spec fields above, not
DeepSeek-V4-specific classes.

### ModelState

`DeepseekV4RocmAtomModelState` is the right ownership layer for persistent
per-request DSV4 state because it sees model structure, request metadata, and
ROCm-only feature flags without requiring GPU worker changes.

State that belongs in ModelState:

- SWA ring metadata and views;
- compressor state rings;
- compressor plans;
- indexer state and decode/prefill metadata derived from vLLM scheduler state;
- ROCm-only binding from vLLM-owned KV storage into model-facing views such as
  `atom_split_kv_swa`, `atom_split_kv_compressed`, and `atom_unified_kv` when a
  homogeneous native ABI exists.

ModelState must not silently fall back to a side allocation when
`VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1` was requested. That flag is meant
to test the vLLM-owned KV integration path.

### WorkspaceManager

`WorkspaceManager.get_simultaneous(...)` is a scratch allocator. It returns
views into a per-ubatch persistent buffer, but the lifetime of those views is
only until the next `get_simultaneous(...)` call for the same ubatch. The
manager is useful for temporary gather/dequant/top-k workspaces and for
avoiding hot-path `torch.empty` allocations that would interact badly with CUDA
graphs.

State that does not belong in WorkspaceManager:

- per-request SWA rings;
- compressor state rings;
- compressed KV cache;
- any state that must survive across layers, requests, scheduler iterations, or
  multiple nested helper calls.

The current AMD attention code may use WorkspaceManager for temporary sparse
attention workspaces. The ROCm ModelState code must not import
`vllm.v1.worker.workspace` or call `current_workspace_manager()`.

## Practical Split

No GPU worker changes are required for persistent request state. Use
model-specific ROCm ModelState for SWA/compressor/indexer state, and use the
generic KV-cache spec fields for allocation shape.

Core/attention changes are still required for true native ATOM benefit:

- a ROCm-only DSV4 KV spec/allocation/binding that exposes the native packed
  layout required by the ATOM kernels, or
- native ATOM/aiter kernels that consume the current vLLM-owned split packed
  layout directly.

CUDA should stay untouched:

- return `DeepseekV4RocmAtomModelState` only for ROCm plus the DSV4 ATOM state
  flag;
- return `DeepseekV4AtomMLAAttentionSpec` only for ROCm plus the vLLM-owned
  ATOM KV flag;
- keep NVIDIA DeepSeek-V4 model files free of ROCm ATOM imports.

## Native Integration Choices

### Choice A: Keep vLLM Split Packed Layout

This is the lower-risk path for vLLM. Keep the current allocation:

```text
[fixed BF16 SWA prefix][paged uint8 fp8_ds_mla compressed tail]
```

Then add native attention/compressor entry points that consume:

- `atom_split_kv_swa`
- `atom_split_kv_compressed`
- `atom_split_kv_layout="fp8_ds_mla"`
- `swa_pages`
- vLLM block tables
- CSA/HCA physical slot metadata

This avoids a scheduler or worker rewrite, but requires native kernels with a
split-source ABI.

### Choice B: Expose A Homogeneous Native View

Change only the ROCm DSV4 KV spec and binding so packed deployment exposes the
native homogeneous view expected by ATOM kernels. This path is only viable if
the native kernel ABI can express the packed 584-byte token layout in one
contiguous logical tensor without hot-path reshaping.

Acceptance requirements:

- `atom_unified_kv` exists for packed ROCm deployment;
- decode, prefill, and compressor all read/write the same vLLM-owned
  allocation;
- CUDA/NVIDIA specs and model files are unchanged;
- generic worker code still operates through `KVCacheSpec` fields only.

## Current Verdict

The practical vLLM split is in place, but full native ATOM kernel benefit is
not yet available. The missing piece is a native packed DSV4 attention and
compressor ABI that matches either Choice A or Choice B.
