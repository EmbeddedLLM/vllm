# DeepSeek-V4 ROCm KV Cache And Workspace Notes

Date: 2026-06-18

This note summarizes how the current vLLM KV-cache and workspace systems work,
where DeepSeek-V4 currently plugs into them, and how a ROCm-only unified DSV4
cache could be introduced without changing the CUDA/NVIDIA path.

## Current vLLM KV-Cache Model

The vLLM scheduler owns logical token/block lifetime. Model code does not
allocate request cache directly. It declares cache needs through
`KVCacheSpec` objects:

- `KVCacheSpec`: base contract. Important properties are `block_size`,
  `page_size_bytes`, `storage_block_size`, `max_memory_usage_bytes`, and
  `merge`.
- `AttentionSpec` / `FullAttentionSpec`: normal K/V attention pages.
- `MLAAttentionSpec`: MLA cache pages. DeepSeek-V4 adds
  `cache_dtype_str`, `alignment`, `compress_ratio`, and `model_version`.
- `SlidingWindowMLASpec`: sliding-window MLA cache pages. DeepSeek-V4 uses
  this for SWA and compressor state.
- `UniformTypeKVCacheSpecs`: wrapper used when layers have the same lifetime
  semantics but different page sizes.
- `KVCacheSpecRegistry`: maps a spec type to a manager class. Built-ins are
  registered in `single_type_kv_cache_manager.register_all_kvcache_specs`;
  platforms can register custom specs through
  `current_platform.register_custom_kv_cache_specs(vllm_config)`.

The engine flow is:

1. Workers return `dict[layer_name, KVCacheSpec]`.
2. `get_kv_cache_configs` merges worker specs and calls `get_kv_cache_groups`.
3. `get_kv_cache_groups` groups compatible specs and handles DeepSeek-V4's
   current special case through `group_and_unify_kv_cache_specs`.
4. `get_kv_cache_config_from_groups` produces a `KVCacheConfig`:
   `num_blocks`, `kv_cache_groups`, and raw `KVCacheTensor(size, shared_by)`
   allocations.
5. `GPUModelRunner._allocate_kv_cache_tensors` allocates raw `int8` buffers.
6. `GPUModelRunner._reshape_kv_cache_tensors` reshapes each raw buffer by
   calling the attention backend's `get_kv_cache_shape` and stride-order hook.
7. `bind_kv_cache` writes the reshaped tensor into each module's
   `kv_cache` field through `static_forward_context[layer_name].kv_cache`.

This means a backend can change the physical tensor view by changing the
cache spec page size and `get_kv_cache_shape`, while still using vLLM's
scheduler/block table/slot mapping.

## Current DeepSeek-V4 vLLM Layout

DSV4 currently uses multiple vLLM cache modules rather than one unified cache.
Each module registers itself in `static_forward_context` and declares an
independent spec:

- Main compressed MLA cache:
  `DeepseekV4Attention.get_kv_cache_spec` returns `MLAAttentionSpec` when
  `compress_ratio > 1`.
- SWA cache:
  `DeepseekV4SWACache.get_kv_cache_spec` returns `SlidingWindowMLASpec` with
  block size `64`.
- Main compressor state:
  `CompressorStateCache.get_kv_cache_spec` returns `SlidingWindowMLASpec` with
  block size `4` for CSA and `8` for HCA.
- Indexer K cache:
  `DeepseekV4IndexerCache.get_kv_cache_spec` returns `MLAAttentionSpec`.
- Indexer compressor state:
  another `CompressorStateCache`.

The current DSV4 cache grouping code is already special:

- `group_and_unify_kv_cache_specs` detects DSV4 by finding
  `SlidingWindowMLASpec`.
- It groups full MLA and SWA-like MLA specs into `UniformTypeKVCacheSpecs`.
- `_get_kv_cache_groups_uniform_groups` pads page sizes to align layer tuples.
- `_get_kv_cache_config_deepseek_v4` then buckets layers by page size and
  emits raw tensors shared by `(page_size, slot_idx)`.

So vLLM already has a DSV4-specific page-size packing path, but the runtime
semantics still remain paged/ragged and split across several cache tensors.

## Baseline ROCm Runtime Pain Points

The baseline ROCm DSV4 path adapts ATOM-style kernels to vLLM's current cache
structure:

- Decode builds dense or ragged index lists for SWA and extra compressed cache.
- Prefill gathers paged FP8 cache rows into a bf16 workspace before calling the
  sparse attention kernel.
- Indexer prefill gathers FP8 K and scale rows into workspace buffers before
  scoring.
- Compressor state is stored through vLLM paged state cache and block tables,
  not request-ring slots.

This creates extra metadata and memory movement that ATOM/SGLang avoid with a
unified cache plus per-request ring buffers.

## Current Branch Status

This branch has moved part of the proposed design into code while keeping the
CUDA/NVIDIA path untouched:

- `DeepseekV4RocmAtomModelState` is selected only when ROCm plus
  `VLLM_ROCM_DSV4_ATOM_STATE=1` are active. This keeps request-lived DSV4 state
  inside a model-specific `ModelState`; no GPU-worker changes are required.
- `DeepseekV4RocmAtomModelState` owns the persistent `state_slot_mapping`
  tensors that bridge vLLM request indices to ATOM-style request slots.
- Optional persistent request-state allocation behind
  `VLLM_ROCM_DSV4_ATOM_STATE_ALLOC=1` binds SWA and compressor state rings to
  attention/compressor modules.
- The vLLM-owned unified-KV path behind
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1` uses
  `DeepseekV4AtomMLAAttentionSpec` to allocate one raw DSV4 attention buffer
  with an SWA prefix and compressed tail.
- `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1` selects the packed `fp8_ds_mla` compressed
  tail format: uint8 `[num_blocks, k_per_block, 584]` with embedded UE8M0 scale
  bytes. This is accuracy-correct, but it is currently a vLLM compatibility
  path rather than a native aiter entry point.

The current branch therefore proves the practical split:

- persistent request rings can live in model-specific `ModelState`;
- ROCm-only cache spec/allocation/binding changes can express the DSV4
  unified-KV preview layout;
- CUDA remains on the normal DSV4 specs, model state, and attention path.

What remains unresolved is full ATOM-kernel benefit. Installed
`aiter==0.1.15.post1` does not expose a native packed DSV4 sparse
attention/compressor contract for
`BF16 SWA prefix + uint8 fp8_ds_mla 584-byte compressed tail + ATOM index
metadata`. The local packed reader/writer paths are correctness-compatible
adapters, not the final native kernel contract.

## WorkspaceManager

`WorkspaceManager` is a global per-worker scratch allocator initialized by the
GPU/XPU worker. It owns one `uint8` workspace tensor per active DBO ubatch.

Important behavior:

- `get_simultaneous((shape, dtype), ...)` returns typed views into one shared
  byte buffer.
- Returned buffers are valid only until the next `get_simultaneous` call for
  the same ubatch.
- Requested regions are 256-byte aligned.
- The buffer grows during warmup/profiling. After graph capture,
  `lock_workspace()` prevents further growth.
- If DBO is enabled, each microbatch has its own workspace slot through
  `dbo_current_ubatch_id`.

This is a scratch replacement for hot-path `torch.empty`, not a persistent
cache model. It is appropriate for temporary gathers and logits workspaces. It
is not appropriate for DSV4 request-state rings because those must survive
across forwards and across layers.

Current DSV4 usage follows that boundary:

- `DeepseekV4RocmAtomModelState` stores persistent `state_slot_mapping`,
  optional SWA rings, compressor rings, and unified-KV buffer bundles. These
  tensors are not borrowed from `WorkspaceManager`.
- `DeepseekV4ROCMAiterMLAAttention` still uses `WorkspaceManager` for temporary
  fallback/gather buffers in the legacy or split paths. Those tensors must not
  be retained after another `get_simultaneous` call.
- The packed mixed-KV path should use workspace only for temporary adapter
  scratch. If a tensor must be visible to a later layer or later forward, it
  belongs in `ModelState` or vLLM KV-cache allocation, not `WorkspaceManager`.
- Guard:
  `test_deepseek_v4_atom_model_state_does_not_use_workspace_manager` asserts
  that `model_state.py` does not import `vllm.v1.worker.workspace`, call
  `current_workspace_manager`, or call `get_simultaneous`.

## ATOM/SGLang Structural Target

ATOM's local DSV4 backend documents the target split:

1. Per-request state cache:
   - SWA ring: `[num_slots, window_size + max_spec_steps, head_dim]`.
   - Compressor state rings: `kv_state` and `score_state`, fp32, indexed by
     `state_slot_mapping[batch]` and `position % ring_size`.
2. Classical compressed KV cache:
   - Global block-table-addressed compressed pages.
   - Block size is `lcm(4, 128) = 128` original tokens.
   - CSA stores `128 / 4 = 32` compressed entries per block.
   - HCA stores `128 / 128 = 1` compressed entry per block.
   - CSA indexer uses a separate FP8/scaled cache.
3. A per-layer unified KV tensor:
   - SWA ring region at the front.
   - Compressed classical region after `swa_pages`.
   - Sparse attention kernels read one base pointer and offsets distinguish
     SWA versus compressed regions.

ATOM's compressor ordering is also tied to this structure: fused compression
reads previous-forward compressor state, then state update writes current
forward tokens into the per-request state ring for the next forward.

## Proposed vLLM Direction For ROCm

The cleanest vLLM-compatible direction is to add ROCm-only DSV4 specs and
metadata while preserving the existing CUDA specs/backends.

### 1. Keep CUDA Unchanged

Do not modify the existing CUDA DSV4 path's cache specs or backend shape
contracts. Add all new behavior behind ROCm platform checks and/or a config
gate such as `VLLM_ROCM_DSV4_UNIFIED_KVCACHE`.

CUDA should continue to use the current:

- `MLAAttentionSpec`
- `SlidingWindowMLASpec`
- current DSV4 grouping path
- FlashMLA/FlashInfer sparse attention metadata

### 2. Add A ROCm DSV4 Unified Cache Spec

Add a new spec type, for example `DeepseekV4ROCmUnifiedKVSpec`, registered only
from the ROCm platform hook.

The spec should describe the physical bytes needed for one DSV4 layer or layer
type under the unified layout:

- semantic scheduler block size: preferably `128` original tokens for DSV4,
  matching ATOM's `lcm(4, 128)`.
- page size for compressed classical blocks.
- per-layer `compress_ratio` and type: dense/SWA-only, CSA, HCA, CSA-indexer.
- dtype/layout: `fp8_ds_mla` for current MI355X FP8 path, plus future FP4/MXFP4
  indexer if needed.

There are two possible designs:

- Spec-per-layer unified tensor:
  each DSV4 layer gets one tensor containing SWA prefix plus compressed tail.
  This matches ATOM most closely but needs request-slot storage outside normal
  page allocation because `swa_pages = num_slots * ring`.
- Spec-per-pool:
  separate specs for request state rings and compressed block pages. This fits
  vLLM's current block manager more naturally but loses ATOM's single-base
  pointer unless a binder creates per-layer unified views.

The first design is better for ATOM kernels. The second is easier to stage in
vLLM. A practical implementation can start with separate specs and bind a
logical `DeepseekV4ROCmCacheBundle` object to the modules.

### 3. Add A Request-State Cache Manager Or Companion Allocator

The missing vLLM abstraction is request-scoped cache slots.

Current vLLM KV managers allocate block IDs as a function of token count. SWA
rings and compressor rings need one persistent slot per live request, with
fixed bytes independent of sequence length. Mamba's `align` mode is the
closest existing manager, but DSV4 also needs classical block pages in the same
attention layer, so reusing `MambaSpec` directly would be misleading.

Recommended approach:

- Add a DSV4 request-state manager, or a DSV4 companion allocator owned by the
  ROCm DSV4 metadata builder.
- Allocate `max_num_seqs` state slots, not token blocks.
- Emit `state_slot_mapping` in metadata from request id to slot id.
- Free slots when requests finish, independent of block-table eviction.
- Disable or explicitly define prefix-cache behavior for request-state rings;
  ATOM currently disables prefix caching because SWA state is not restorable
  from the classical KV pool.

This should not use `WorkspaceManager`; it must be persistent.

### 4. Bind A ROCm Cache Bundle In `static_forward_context`

Current vLLM modules expect `module.kv_cache` tensors. For unified ROCm DSV4,
bind a structured object or multiple named fields on the ROCm attention module,
similar to ATOM:

- `unified_kv[layer]`
- `swa_kv` view into the SWA prefix
- `compressor.kv_state`
- `compressor.score_state`
- `compressor.kv_cache` view into compressed tail
- `indexer.kv_cache`
- indexer cache scale view when FP8 layout needs it

This can be done in a ROCm-specific initializer after raw tensors are allocated
and before forward execution. The existing `bind_kv_cache` path assumes one
tensor per layer name, so a custom DSV4 cache tensor reshape/bind path will
likely be needed.

### 5. Build ROCm Metadata Once Per Forward

The metadata builder should produce ATOM-like metadata from vLLM scheduler data:

- `state_slot_mapping`
- `batch_id_per_token`
- `CompressPlan` equivalents for ratio 4 and 128
- packed/ragged indices for decode and prefill, or direct offset metadata if
  kernels read unified cache without gather
- `swa_pages`
- committed compressed counts per request

Inputs already available from vLLM:

- `CommonAttentionMetadata.block_table_tensor`
- `slot_mapping`
- `query_start_loc`
- `query_start_loc_cpu`
- `seq_lens`
- `positions`

The important change is to stop materializing temporary paged-gather buffers in
prefill once kernels can read the unified cache directly.

### 6. WorkspaceManager Use After Unified Cache

With a unified ROCm cache:

- Keep `WorkspaceManager` for temporary tensors such as top-k scratch, logits,
  or fallback gather buffers.
- Do not use it for SWA rings, compressor state, or unified KV storage.
- Avoid nested `get_simultaneous` calls where child calls can invalidate parent
  buffers. If nesting is unavoidable, request all needed tensors at the highest
  level and pass views downward.
- During dummy/warmup paths, call `get_simultaneous` with worst-case shapes so
  the workspace locks at a stable size before graph capture.

## Minimal Implementation Plan

1. Done for the current preview gate set: use
   `VLLM_ROCM_DSV4_ATOM_STATE=1`,
   `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`, and
   `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`.
2. Done for the vLLM-owned preview layout:
   `DeepseekV4AtomMLAAttentionSpec` carries SWA-prefix and compressed-tail
   metadata and is selected only on ROCm.
3. Done and guarded by tests: CUDA/non-ROCm stays on regular
   `MLAAttentionSpec`, `DefaultModelState`, and no ATOM post-bind views.
4. Partially done:
   - per-layer unified KV tensors are bound from vLLM-owned KV allocation;
   - optional side-allocated SWA/compressor rings exist in `ModelState`;
   - CSA indexer cache remains partly on existing vLLM/indexer structures.
5. Partially done: `DeepseekV4RocmAtomModelState` emits
   `state_slot_mapping`, compression plans, and decode/prefill buffers, but
   there is still adapter metadata around packed tails and index translation.
6. Partially done:
   - `swa_write`, compressor state update, read-before-update fused
     compressor, split/unified decode, and split/unified prefill paths exist;
   - packed `fp8_ds_mla` decode/prefill/compressor are compatibility kernels,
     not native aiter packed DSV4 entry points.
7. Still required: keep old ROCm paths as fallback until the packed native
   path beats the current best C32 run and continues to pass unchanged
   `lmeval.sh`.

## ATOM Kernel Readiness Audit

The current vLLM ROCm DSV4 path contains several ATOM-inspired operations, but
it is not yet structurally equivalent to ATOM. In particular, matching a wrapper
name is not sufficient: the important contract is the physical cache layout and
the order in which state is read, written, and consumed.

### Components Already Present In vLLM

- **vLLM scheduler and request lifecycle:** model runner v2 already exposes
  stable per-request state indices through `RequestState` and `InputBatch`.
  These indices can act as ATOM's `state_slot_mapping`.
- **Model-specific state hook:** model runner v2 calls
  `model.get_model_state_cls()` and routes `add_request`, `remove_request`,
  `prepare_inputs`, and `prepare_attn` through the selected `ModelState`.
  This means request-lived ROCm DSV4 state can be added without modifying
  `gpu_worker.py`.
- **Fused MoE:** vLLM's ROCm model path already uses vLLM `FusedMoE`/`GateLinear`
  and the existing loader path.
- **Q/K ROCm fused path:** `DeepseekV4ROCMAiterMLAAttention` has optional
  ATOM-style q/r norm, RoPE, KV insert, and inverse-RoPE output projection
  helpers.
- **Compressor kernels:** vLLM has compressor state-cache, partial-state save,
  and fused compress/norm/RoPE/cache-store kernels, but they operate over
  vLLM's paged cache/state metadata rather than ATOM's request-state rings.
- **ROCm sparse attention wrappers:** vLLM has ROCm decode/prefill wrappers and
  ragged-index helpers, but their cache contract differs from ATOM.

### Components Missing For Full ATOM Benefit

- **Persistent request-state buffers:** ATOM has SWA and compressor state rings
  indexed by `state_slot_mapping`. vLLM currently stores SWA and compressor
  state as ordinary paged KV-cache groups.
- **Unified per-layer KV pool:** ATOM's sparse kernels read one per-layer
  `unified_kv` where `[0, swa_pages)` is SWA ring storage and
  `[swa_pages, ...)` is compressed classical KV. vLLM currently keeps separate
  SWA, compressed MLA, compressor-state, and indexer cache tensors.
- **ATOM index generation:** ATOM builds indices directly into `unified_kv`
  (`paged_decode_indices`, `paged_prefill_indices`, `csa_translate_pack`).
  vLLM currently builds block-table/global slot indices for separate cache
  tensors.
- **ATOM compressor ordering over request rings:** ATOM's fused compressor reads
  previous state first, writes compressed KV, then updates compressor state.
  vLLM's current compressor writes partial state into a paged state cache before
  compressed cache store.
- **ATOM sparse prefill/decode kernels:** vLLM wrappers are not equivalent yet.
  They must be ported or adapted to read ATOM-style unified KV offsets.

### Sparse Attention Wrapper Comparison

ATOM decode:

- Function: `sparse_attn_v4_paged_decode(q, unified_kv, kv_indices, kv_indptr,
  attn_sink, softmax_scale, kv_scales=None, ...)`.
- Cache contract: one flat `[total_pages, D]` `unified_kv`.
- Index contract: ragged `kv_indices/kv_indptr` already point into unified KV.
- FP8 contract: optional separate `kv_scales` for `[total_pages, D // 64]`
  1x64 block scales.
- Performance intent: no separate SWA/compressed gather, one kernel reads the
  selected pages directly.

vLLM ROCm decode:

- Function: `rocm_sparse_attn_decode(q, kv_cache, swa_k_cache, swa_only,
  topk_indices, topk_lens, swa_indices, swa_lens, ...)`.
- Cache contract: separate SWA cache and compressed cache, both in vLLM
  `fp8_ds_mla` paged layout.
- Index contract: separate dense/ragged SWA indices and top-k compressed indices.
- Current consequence: useful fallback, but not ATOM unified-cache decode.
  Its `uint8 fp8_ds_mla` rows are not ATOM's modeling-file BF16 unified rows,
  nor ATOM decode's optional `[fp8 unified_kv, fp32 kv_scales]` layout.

ATOM prefill:

- Function: `sparse_attn_v4_paged_prefill(q, unified_kv, kv_indices_prefix,
  kv_indptr_prefix, kv, kv_indices_extend, kv_indptr_extend, ...)`.
- Cache contract: prefix reads from unified KV, current extend reads from local
  dense `kv`.
- Index contract: prefix and extend are two ragged sources.
- Performance intent: avoid materializing a large gathered BF16 prefix workspace.

vLLM ROCm prefill:

- Function path: `DeepseekV4ROCMAiterMLAAttention._forward_prefill`.
- Cache contract: dequantize/gather separate compressed and SWA paged caches into
  a temporary BF16 workspace from `WorkspaceManager`.
- Index contract: sparse attention then reads indices into that workspace.
- Current consequence: this does not get ATOM's unified-cache prefill benefit.

### First Integrated Slice

The first code slice adds `DeepseekV4RocmAtomModelState`, gated by
`VLLM_ROCM_DSV4_ATOM_STATE=1`, and wires it through
`DeepseekV4ForCausalLM.get_model_state_cls()`.

This gives the ROCm DSV4 path an ATOM-compatible metadata anchor:

- `state_slot_mapping`: GPU int32 request slot mapping.
- `state_slot_mapping_cpu`: CPU mirror for planning.
- `win_with_spec`: `sliding_window + num_speculative_tokens`.
- `swa_pages`: `max_num_reqs * win_with_spec`.
- per-forward `positions`, `query_start_loc`, and `seq_lens` references.

This is not the complete unified-cache implementation. It is the prerequisite
that lets the next slices bind persistent SWA/compressor rings and switch
attention/compressor kernels to request-ring/unified-KV addressing.

### Second Integrated Slice

The second code slice adds persistent ATOM-style state buffers behind a second
gate: `VLLM_ROCM_DSV4_ATOM_STATE_ALLOC=1`.

Allocated buffers:

- `swa_kv`: `[active_layers, max_num_reqs, win_with_spec, head_dim]`.
- `csa_main_kv_state` / `csa_main_score_state`:
  `[csa_layers, max_num_reqs, 8 + spec_tokens, 2 * head_dim]`.
- `csa_idx_kv_state` / `csa_idx_score_state`:
  `[csa_layers, max_num_reqs, 8 + spec_tokens, 2 * index_head_dim]`.
- `hca_main_kv_state` / `hca_main_score_state`:
  `[hca_layers, max_num_reqs, 128 + spec_tokens, head_dim]`.

Binding is deliberately non-invasive:

- Attention modules receive `atom_swa_kv`, `atom_win_with_spec`, and
  `atom_swa_pages`.
- Main compressors receive `atom_kv_state` and `atom_score_state`.
- Indexer-inner compressors receive their own CSA indexer state slices.
- Existing vLLM cache attributes are not replaced yet, so current ROCm/CUDA
  behavior stays unchanged unless future slices explicitly consume these
  `atom_*` attributes.

Remaining work after this slice:

- Add unified per-layer KV tails using `KVCacheConfig.num_blocks`.
- Bind `atom_unified_kv` as the SWA prefix plus compressed tail.
- Port ATOM `swa_write`, `update_compressor_states`, and
  `fused_compress_attn` to consume the new state buffers.
- Replace vLLM's current ROCm sparse decode/prefill wrappers with kernels that
  read ATOM unified-KV offsets directly.

### Third Integrated Slice

The third code slice adds ATOM-shaped per-layer unified KV pools behind
`VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`.

Allocation is lazy in `prepare_attn`, because the number of vLLM physical KV
blocks is only available through `KVCacheConfig.num_blocks`.  For each active
DSV4 attention layer it allocates:

- Dense/SWA-only layer: `[swa_pages, head_dim]`.
- CSA layer: `[swa_pages + num_blocks * 32, head_dim]`.
- HCA layer: `[swa_pages + num_blocks * 1, head_dim]`.

Binding remains deliberately inert and uses only `atom_*` attributes:

- `attn.atom_unified_kv`: full per-layer ATOM-style pool.
- `attn.atom_swa_kv`: SWA prefix view
  `[max_num_reqs, win_with_spec, head_dim]`.
- `attn.atom_compressed_kv_cache`: compressed tail view
  `[num_blocks, k_per_block, head_dim]` for CSA/HCA layers.
- `attn.compressor.atom_kv_cache`: same compressed tail view.

The existing vLLM attributes (`kv_cache`, `swa_cache_layer.kv_cache`,
`compressor.kv_cache`) are not replaced yet.  This is important because the
active vLLM ROCm sparse wrappers still expect the old split fp8 paged layout.

### Sparse MLA Wrapper Deep Dive

The sparse MLA decode/prefill wrappers are not interchangeable today.

ATOM modeling-file dataflow:

1. `qk_norm_rope_maybe_quant(..., quant_q=False, quant_k=False)` produces BF16
   `q_sa` and BF16 `kv`.
2. Decode calls `swa_write(kv, ..., self.swa_kv, ...)` before attention so the
   current token is visible in the SWA ring.
3. Compressor writes CSA/HCA main compressed rows into the compressed tail view
   of `self.unified_kv`.
4. Decode calls
   `sparse_attn_v4_paged_decode(q_sa, self.unified_kv, kv_indices, kv_indptr,
   attn_sink, softmax_scale)`.
5. Prefill calls
   `sparse_attn_v4_paged_prefill(q_sa, self.unified_kv, prefix_indices,
   prefix_indptr, kv, extend_indices, extend_indptr, ...)`, then writes SWA.

vLLM ROCm dataflow:

1. `_fused_qnorm_rope_kv_insert` writes `kv_out` into
   `self.swa_cache_layer.kv_cache` through `quantize_and_insert_k_cache`.
2. The current ROCm path raises if the SWA cache is not `torch.uint8`, because
   it expects the `fp8_ds_mla` vLLM cache layout.
3. Decode calls `rocm_sparse_attn_decode` with two cache bases:
   `swa_k_cache` and optional compressed `kv_cache`.
4. Prefill dequantizes and gathers both caches into a BF16 workspace, combines
   top-k and SWA indices into that workspace coordinate system, then calls
   `rocm_sparse_attn_prefill`.

Required replacement to reach ATOM equivalence:

1. Build ATOM decode indices where every entry is an offset into
   `atom_unified_kv`, including SWA ring offsets below `swa_pages` and
   CSA/HCA compressed offsets above `swa_pages`.
2. Build ATOM prefill indices as two ragged sources: prefix offsets into
   `atom_unified_kv`, and extend offsets into the current forward's dense `kv`.
3. Port or vendor the ATOM paged decode/prefill wrappers into vLLM without an
   `atom` package dependency.
4. Switch DSV4 ROCm attention to write BF16 `atom_swa_kv` and BF16 compressed
   main rows before calling those wrappers.
5. Keep CSA indexer cache separate, because ATOM also keeps the indexer FP8
   cache outside `unified_kv`.

### Fourth Integrated Slice

The fourth code slice vendors ATOM's paged sparse-attention wrappers into vLLM
under `vllm.models.deepseek_v4.amd.v4_kernels`.

Vendored files:

- `paged_decode.py`: ATOM-style unified-KV decode wrapper.
- `paged_prefill.py`: ATOM-style unified-KV prefill wrapper.
- `paged_decode_indices.py`: ATOM decode SWA-prefix index writer.
- `paged_prefill_indices.py`: ATOM prefill prefix/extend index writer.
- `compress_plan.py`: ATOM/SGLang-style packed compressor plan builder.
- `state_writes.py`: ATOM `swa_write` and `update_compressor_states`.
- `fused_compress.py`: ATOM fused compress + norm + RoPE + cache scatter.
- `csa_translate_pack.py`: ATOM CSA top-k to unified-KV offset packer.
- `reference.py`: torch-only ragged sparse-attention reference helper used by
  paged decode/prefill reference paths.

Local vLLM changes made during vendoring:

- Removed all imports from the external `atom` package.
- `paged_decode.py` imports `sparse_attn_ragged_torch` from the local
  torch-only reference helper.
- `paged_prefill.py` reads `ATOM_FORCE_ATTN_TRITON` from `os.environ` at module
  import time instead of depending on `atom.utils.envs`.
- `fused_compress.py` imports `CompressPlan` from the local vLLM vendored
  module and reads `ATOM_FUSED_COMPRESS_USE_FLYDSL` from `os.environ` at module
  import time.
- `__init__.py` exports the paged attention and index-writer public APIs.

This slice still does not change active forward execution.  The wrappers now
exist in vLLM, but active execution requires replacing vLLM's current metadata
builders with the vendored ATOM-style index writers.

Index coordinate systems:

- ATOM decode SWA offset:
  `state_slot_per_seq[bid] * win_with_spec + (abs_pos % win_with_spec)`.
- ATOM compressed HCA offset:
  `swa_pages + block_id`.
- ATOM compressed CSA offset:
  `swa_pages + block_id * 32 + slot_in_block`; the SWA segment is written by
  `write_v4_paged_decode_indices`, while the CSA top-k segment still needs the
  CSA translate/pack step.
- vLLM decode top-k offset today:
  `block_number * compressed_block_size + block_offset`, relative to the
  separate compressed `kv_cache`, not relative to `unified_kv`.
- vLLM prefill offset today:
  index into a temporary BF16 workspace assembled by `dequantize_and_gather`.

Therefore the current vLLM indices cannot be reused as-is.  Decode can reuse
parts of vLLM's existing top-k and block-table data, but it must add
`swa_pages` for compressed unified-KV tails and must generate SWA ring offsets
from request-state slots instead of vLLM SWA block tables.  Prefill must switch
from workspace coordinates to ATOM's two-source coordinates:
`prefix -> atom_unified_kv`, `extend -> current dense kv`.

Verification completed for this slice:

- Python bytecode compilation of the vendored package.
- Import of the public paged decode/prefill APIs with `PYTHONPATH=previewdsv4`.
- CPU reference sanity check for decode and prefill output shape/dtype.
- CPU reference sanity check for decode/prefill index-writer output shape and
  offset formulas.
- CPU reference sanity check for compression-plan generation, SWA ring writes,
  and compressor-state ring updates.
- `rg` confirmed no remaining `atom` package imports in the vendored package.

### Runnable Gate

The code is not ready for meaningful lmeval/perf yet.  A runnable ATOM-layout
path requires all of these to be true in the same forward path:

1. `qk_norm_rope_maybe_quant` produces BF16 `q_sa` and BF16 `kv`.
2. Decode writes current-token KV to `atom_swa_kv` with local `swa_write`
   before sparse attention.
3. Prefill calls sparse attention before `swa_write`, matching ATOM's
   read-before-update ordering for chunked prefill.
4. Compressor forward calls local `fused_compress_attn` against
   `atom_kv_state` / `atom_score_state` and scatters BF16 main rows into the
   `atom_unified_kv` compressed tail.
5. Compressor forward then calls local `update_compressor_states`.
6. CSA indexer top-k is translated with local `csa_translate_pack`.
7. Decode calls local `sparse_attn_v4_paged_decode` with local
   `write_v4_paged_decode_indices` outputs.
8. Prefill calls local `sparse_attn_v4_paged_prefill` with local
   `write_v4_paged_prefill_indices` outputs.

Only after those conditions are wired under a ROCm gate should we run
`launchdeepseekgraph.sh`, `lmeval.sh`, and then `benchmarkvllm.sh`.  Running
accuracy before that would only measure the old split-cache vLLM path or an
invalid mixed coordinate system.

### Fifth Integrated Slice

The fifth code slice adds two pieces needed by the compressor side of the
ATOM path:

- `DeepseekV4RocmAtomModelState` can build ATOM `CompressPlan` objects in
  `prepare_attn`, gated by `VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN=1`.
- `DeepseekCompressor.forward` has a ROCm-only, off-by-default main-compressor
  hook gated by `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`.

The main-compressor hook is intentionally strict.  If enabled, it requires:

- `VLLM_ROCM_DSV4_ATOM_STATE=1`
- `VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN=1`
- `VLLM_ROCM_DSV4_ATOM_STATE_ALLOC=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`
- bound `atom_kv_state`, `atom_score_state`, and `atom_kv_cache`

When those are present and the compressor is a main BF16 compressor
(`head_dim == 512`), it bypasses vLLM's paged `state_cache` path and runs:

1. local `fused_compress_attn` against request-state rings and the
   `atom_unified_kv` compressed tail,
2. local `update_compressor_states` after fused compression.

This preserves ATOM's read-before-update ordering for the main CSA/HCA
compressors.  It is still not a complete runnable ATOM path because:

- the CSA indexer-inner compressor still uses the existing vLLM cache path,
- SWA writes still target vLLM's split SWA cache in active attention,
- sparse attention still dispatches through vLLM's split-cache wrappers.

### Sixth Integrated Slice

The sixth code slice prepares the attention side without switching dispatch:

- `DeepseekV4RocmAtomStateMetadata` now carries:
    - `batch_id_per_token`
    - `batch_id_per_token_cpu`
    - `n_committed_csa_per_seq`
    - `n_committed_csa_per_seq_cpu`
    - `n_committed_hca_per_seq`
    - `n_committed_hca_per_seq_cpu`
- These are built in `DeepseekV4RocmAtomModelState.prepare_attn` from vLLM's
  existing `InputBatch` scheduler state.
- `DeepseekV4ROCMAiterMLAAttention._fused_qnorm_rope_kv_insert` preserves the
  rotated BF16 `kv_out` as `self._atom_last_kv` when
  `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`.

This prepares the inputs needed by local ATOM kernels:

- `swa_write` needs `batch_id_per_token`, `state_slot_mapping`, positions, and
  rotated `kv`.
- `csa_translate_pack` needs `batch_id_per_token`, committed CSA counts, and
  block tables.
- paged prefill needs rotated `kv` as the extend source.

The existing vLLM SWA cache insert still runs.  The path remains non-disruptive
until the next slice replaces active sparse attention dispatch with the local
ATOM paged decode/prefill wrappers and the local ATOM index writers.

### Seventh Diagnostic Slice

The seventh slice focused on normal cudagraph safety for the first HCA
attention integration attempt.

Code changes:

- Removed the temporary `VLLM_ROCM_DSV4_ATOM_SYNC_DEBUG` forward-path
  synchronization. `torch.cuda.synchronize()` inside the model forward
  invalidated normal graph capture and cannot remain in the non-eager target
  path.
- Added defensive bounds to `swa_write`:
    - `src_id` must be in `[0, total_tokens)`;
    - request state slot must be in `[0, num_slots)`.
- Added safe slot address formation in local paged decode:
    - invalid slots are still masked out of attention math;
    - pointer arithmetic uses a clamped `safe_slot` to avoid masked OOB pointer
    formation on ROCm.
- Added diagnostic flag `VLLM_ROCM_DSV4_ATOM_DISABLE_SWA_WRITE=1`, default
  off, to isolate ATOM SWA ring writes from ATOM decode.

Runs:

- `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS=128`,
  `VLLM_ROCM_DSV4_ATOM_HCA_FORCE_SWA_ONLY=1`, normal cudagraph:
  graph capture completed, but a 128-request high-concurrency flood still
  failed with async illegal memory access.
- Same run with `VLLM_ROCM_DSV4_ATOM_DISABLE_SWA_WRITE=1`:
  graph capture completed, but the same flood still failed. This rules out the
  ATOM SWA ring write as the only crash trigger.
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=0` while leaving the launch defaults for
  ATOM state/unified-KV/compressor enabled:
  the same 128-request flood completed `128 / 128`.

Current conclusion:

- Persistent `ModelState` allocation and unified-KV side allocation are not by
  themselves causing the stress crash.
- The crash is gated by `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`.
- Because disabling SWA writes does not fix it, the next investigation should
  focus on the local ATOM decode dispatch and the metadata/index buffers it
  consumes under cudagraph replay. In the failed runs, logs still showed native
  sparse decode JIT warnings, so the next slice should add a graph-safe
  dispatch counter or one-time startup log outside the captured forward to
  prove whether the local `sparse_attn_v4_paged_decode` kernels are actually
  captured/replayed for HCA layers.

### Eighth Diagnostic Slice

The eighth slice narrowed the HCA force-SWA crash further under normal
cudagraph. It added graph-safe diagnostic flags in the ROCm attention hook:

- `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_DECODE=1`: run ATOM routing/index setup, but
  zero the output instead of calling local `sparse_attn_v4_paged_decode`.
- `VLLM_ROCM_DSV4_ATOM_PROBE_INDICES_ONLY=1`: run the ATOM attention hook and
  then return `False`, allowing native vLLM sparse attention to produce real
  model outputs.
- `VLLM_ROCM_DSV4_ATOM_SKIP_DECODE_INDEX_WRITE=1`: with probe mode, enter the
  ATOM hook but skip `write_v4_paged_decode_indices` before falling back to
  native attention.

Runs:

- `ATOM_SKIP_PAGED_DECODE=1`, HCA force-SWA, normal cudagraph:
  graph capture completed, but the 128-request flood failed `0 / 128`. This
  means the local paged-decode kernel is not the only possible trigger.
- `ATOM_SKIP_PAGED_DECODE=1` plus `ATOM_DISABLE_SWA_WRITE=1`:
  the same flood still failed `0 / 128`. This rules out the SWA write plus
  paged decode as the complete failure set.
- `ATOM_PROBE_INDICES_ONLY=1`, `ATOM_DISABLE_SWA_WRITE=1`:
  the same flood still failed `0 / 128` while native attention produced the
  output. This means corrupt zero attention output was not the explanation.
- `ATOM_PROBE_INDICES_ONLY=1`, `ATOM_SKIP_DECODE_INDEX_WRITE=1`,
  `ATOM_DISABLE_SWA_WRITE=1`:
  the same flood still failed `0 / 128`. Logs showed only native vLLM sparse
  attention JIT warnings before the illegal memory access. No local ATOM decode,
  SWA write, or decode-index Triton JIT appeared in the failure window.
- Fresh current control with `VLLM_ROCM_DSV4_ATOM_ATTENTION=0` and the same
  state/unified-KV/compressor defaults:
  the same flood completed `128 / 128`.

Updated conclusion:

- The current crash is still gated by `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`, but it
  is not proven to be caused by local ATOM paged decode, ATOM SWA write, or the
  local decode-index writer.
- The minimal crashing delta is now entering the ATOM attention hook for enabled
  HCA layers and falling back to native attention. The hook body in the
  skip-index probe only checks enablement, fetches `ModelState`, verifies
  `atom_unified_kv`, slices persistent decode buffers, and returns `False`.
- Next analysis should inspect graph capture/replay aliasing and side effects
  around returning from the model-specific attention hook: whether the extra
  buffer views change graph input/output alias assumptions, whether the hook is
  reached for empty decode slices and returns `True`, and whether any metadata
  prepared in `DeepseekV4RocmAtomModelState.prepare_attn` differs when
  `ATOM_ATTENTION=1` even if local kernels are skipped.

### Ninth Diagnostic Slice

The ninth slice tested whether the crash was caused by the ATOM attention hook
body or by another `VLLM_ROCM_DSV4_ATOM_ATTENTION=1` side effect.

It added `VLLM_ROCM_DSV4_ATOM_RETURN_FALSE_AT_ENTRY=1`, which returns `False`
immediately after the ATOM attention ratio/layer gate and before reading
`ModelState`, unified KV, decode buffers, or launching any local ATOM index or
attention kernel.

Runs:

- `ATOM_RETURN_FALSE_AT_ENTRY=1`, `ATOM_PROBE_INDICES_ONLY=1`,
  `ATOM_SKIP_DECODE_INDEX_WRITE=1`, `ATOM_DISABLE_SWA_WRITE=1`, HCA
  force-SWA, normal cudagraph, default `ATOM_MAIN_COMPRESSOR=1`:
  graph capture completed, but the 128-request flood failed `0 / 128`. Logs
  again showed native sparse attention JIT before `hipErrorIllegalAddress`.
- Same run, but with `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=0`:
  the same flood completed `128 / 128`.
- After adding a guard so the ATOM main compressor does not replace the native
  compressor when diagnostic flags force native attention fallback, the original
  `ATOM_MAIN_COMPRESSOR=1` entry-return run completed `128 / 128`.

Updated conclusion:

- The immediate illegal memory access in these probe configurations was a
  writer/reader mismatch. The ATOM main compressor wrote the ROCm unified KV
  cache and returned early, while diagnostic flags forced the attention path
  back to native vLLM sparse attention. Native attention then read the native
  compressed KV cache, which was not updated for those decode steps.
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=1` is not by itself the corrupting operation.
  The compressor and attention reader must be treated as a matched pair: if the
  ATOM attention reader is not actually consuming unified KV, the native
  compressor must still run.
- This also explains why the crash looked like a native sparse attention or GEMM
  failure in logs. The visible failing kernel was downstream of the real
  inconsistency.

### Tenth Diagnostic Slice

The tenth slice moved from diagnostic fallback to a matched ATOM writer/reader
configuration for HCA layers:

- `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS=128`
- `VLLM_ROCM_DSV4_ATOM_HCA_FORCE_SWA_ONLY=1`
- `VLLM_ROCM_DSV4_ATOM_HCA_FLAT_CACHE=1`
- default `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- normal cudagraph, V2 runner, no `--enforce-eager`

Runs:

- Matched HCA path, 32 concurrent requests:
  completed `32 / 32`. Logs showed `_swa_write_kernel` JIT, confirming the
  persistent SWA-ring writer was active.
- Same server, 128 concurrent requests:
  failed `0 / 128` with `hipErrorIllegalAddress`.
- Matched HCA path with `VLLM_ROCM_DSV4_ATOM_DISABLE_SWA_WRITE=1`, 128
  concurrent requests:
  still failed `0 / 128`. Logs no longer showed `_swa_write_kernel`, so the
  remaining high-concurrency crash does not require the SWA write kernel.

Updated conclusion:

- The writer/reader mismatch from the ninth slice is fixed for diagnostic
  native-fallback runs, but the real matched HCA ATOM path is only stable at the
  smaller 32-request stress level.
- At 128-way pressure, the remaining fault is in the ATOM decode reader/index
  side or its buffer sizing/metadata assumptions, not solely in the SWA ring
  write.
- The next useful slice should instrument or harden
  `sparse_attn_v4_paged_decode`, `write_v4_paged_decode_indices`, and the
  `DecodeBuffers` capacities against actual `T`, per-token SWA lengths, and
  maximum generated slot IDs at high concurrency.

### Eleventh Diagnostic Slice

The eleventh slice checked whether the 128-way failure was caused by using
active-batch size as the SWA ring capacity.

Patch:

- `write_v4_paged_decode_indices` now accepts `max_pages`.
- The ROCm attention path passes `atom_state.swa_pages`, i.e.
  `max_num_reqs * win_with_spec`, instead of deriving capacity from
  `state_slot_per_seq.shape[0]`.
- Rationale: `state_slot_per_seq` is indexed by active batch row, but its
  values are persistent request-state slots. At high concurrency, slot ids are
  not guaranteed to be dense in the current active row count.

Validation:

- `py_compile` passed for the touched ROCm attention, model-state, compressor,
  and decode-index modules.
- A standalone 128-slot `write_v4_paged_decode_indices` GPU launch completed.
- Standalone `sparse_attn_v4_paged_decode` launches completed for representative
  `T=128, H=16, D=512` in both fused (`kv_splits=1`) and split
  (`kv_splits=4`) modes.
- A standalone shrunk-batch launch matching the fatal scheduler shape also
  completed: `T=120`, position `17`, SWA length `18`, and non-dense persistent
  state slots up to `127`.

Server runs, all with normal cudagraph, V2 model runner, no `--enforce-eager`,
HCA-only ATOM attention, `ATOM_HCA_FORCE_SWA_ONLY=1`, and 128 concurrent
completion requests:

- With `ATOM_DISABLE_SWA_WRITE=1`, `max_tokens=8` still failed after
  `13 / 128` completed. This remains a bad semantic configuration because the
  HCA reader consumes a zero/stale SWA ring, but it proves the page-capacity
  change alone does not make the disabled-writer diagnostic stable.
- With `ATOM_DISABLE_SWA_WRITE=1` and `ATOM_SKIP_PAGED_DECODE=1`, `max_tokens=8`
  still failed after `8 / 128` completed. Because the HCA layer output is
  intentionally zeroed in this mode, this is not a clean decode-index writer
  failure proof.
- With the real SWA writer and ATOM HCA reader enabled:
    - `max_tokens=1`: completed `128 / 128`
    - `max_tokens=2`: completed `128 / 128`
    - `max_tokens=4`: completed `128 / 128`
    - `max_tokens=8`: failed after `8 / 128` completed

The fatal scheduler dump for the failing `max_tokens=8` run showed:

- `num_running_reqs=120`
- `num_computed_tokens=[17, ...]`
- `num_output_tokens=[7, ...]`
- 4 requests had already finished and been removed
- `num_scheduled_tokens` was 1 per remaining request

Updated conclusion:

- Initial unified-KV binding, persistent SWA writes, decode indptr construction,
  and the first several ATOM HCA decode replays are stable at 128-way pressure.
- The remaining crash appears only after partial request completion/shrink in a
  later decode step. That points at request-slot lifecycle and graph replay
  metadata under a shrinking active batch, not just initial allocation.
- The local standalone index writer and paged reader do not reproduce the
  `T=120` failure shape, so the next pass should instrument the full model path
  around request removal/reuse, output sampling, and any per-layer persistent
  buffer aliasing that is not present in the standalone kernels.
- The page-capacity fix is still correct and should stay: ATOM SWA ring
  addressing must be bounded by persistent slot capacity, not active-batch row
  count.
- Next useful slice: compare the per-step `idx_mapping`, `state_slot_mapping`,
  `query_start_loc`, padded token count, and SWA/HCA indptr tails when the batch
  shrinks from 128 to 120. In particular, verify that full-cudagraph padded rows
  have zero-length indptr slices and that no stale request slot survives in a
  graph-replayed ATOM metadata buffer.

### Twelfth Diagnostic Slice

This slice isolated the remaining `max_tokens=8` failure to the ATOM paged
decode split-K path rather than vLLM request-slot metadata.

Patch/experiment:

- Added a model-state `req_id -> slot` map so the ROCm DSV4 ATOM model state can
  track request slots without changing the V2 GPU worker.
- Tried clearing ATOM state on `remove_request`, but this is unsafe because vLLM
  removes requests while output handling is asynchronous. The final patch only
  removes the Python map entry at removal time and keeps the existing add-time
  slot reset as the reuse boundary.
- Forced `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`, which sends
  `sparse_attn_v4_paged_decode` through its fused single-pass kernel and avoids
  the split/reduce partial-workspace path.
- Added that as an overridable launch default in `launchdeepseekgraph.sh`.

Validation:

- `py_compile` passed for `vllm/models/deepseek_v4/amd/model_state.py`.
- Baseline all-HCA ATOM attention with default split heuristic still failed:
  normal cudagraph, V2 runner, no `--enforce-eager`, 128-way
  `max_tokens=8` smoke completed only `8 / 128`, then hit an async
  `hipErrorIllegalAddress`.
- Layer-scoped diagnostic with `VLLM_ROCM_DSV4_ATOM_ATTENTION_LAYERS=0` and the
  default split heuristic completed the first 8 requests, then wedged the next 8
  with zero throughput. This proves a single HCA layer is enough to reproduce the
  second-wave failure.
- The same layer-0 run with `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1` passed:
    - `16 / 16`, `max_tokens=8`
    - `128 / 128`, `max_tokens=8`
- All HCA layers with `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1` passed:
    - `128 / 128`, `max_tokens=8`
- `ATOM_USE_TRITON_ATTN=0` is not a valid normal-cudagraph diagnostic for this
  path: the PyTorch reference calls `.item()` during capture and fails startup
  with `hipErrorStreamCaptureUnsupported`.

Updated conclusion:

- vLLM V2 model-state integration is sufficient to carry request-lived ATOM SWA
  rings and compressor state without changing GPU workers for this preview.
- The immediate normal-cudagraph blocker was the ATOM paged decode split/reduce
  path under graph replay and request reuse, not the scheduler request-slot
  mapping itself.
- The fused single-pass ATOM paged decoder is currently the stable path for the
  all-HCA ATOM attention preview. It may be slower for small batches than the
  intended split-K path, but it is the first path that survives the shrink/reuse
  smoke under normal cudagraph.
- The split-K path still needs a focused kernel/workspace audit before it can be
  made the default: likely areas are WorkspaceManager lifetime under graph
  replay, partial buffer aliasing across repeated layer calls, and reduce-kernel
  assumptions for short SWA-only windows.

Follow-up audit:

- ATOM allocates paged-decode split-K partials with per-call `torch.empty`
  buffers: `m_partial`, `l_partial`, and `acc_partial`.
- The vLLM port had replaced those buffers with `WorkspaceManager` views. That
  changes the lifetime contract from normal tensor allocation to shared ubatch
  scratch that is only valid until the next `get_simultaneous(...)` call.
- Added a diagnostic allocator switch,
  `VLLM_ROCM_DSV4_ATOM_DECODE_SPLIT_WORKSPACE=workspace|torch_empty`, cached at
  module import. After larger-shape validation, the default is `torch_empty`;
  `workspace` remains an explicit diagnostic mode for reproducing the vLLM
  shared-scratch behavior.
- This switch is only relevant when the unified paged-decode path uses
  `kv_splits > 1`. Current stable deployment still forces
  `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`, so it bypasses split/reduce partials
  entirely.

Initial validation:

- Launched graph mode without `--enforce-eager` using:
    - `MAX_NUM_SEQS=16`
    - `MAX_NUM_BATCHED_TOKENS=2048`
    - `MAX_MODEL_LEN=2048`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=4`
    - `VLLM_ROCM_DSV4_ATOM_DECODE_SPLIT_WORKSPACE=torch_empty`
- A 16-way `/v1/completions` smoke with `max_tokens=8` completed `16 / 16`.
- A 128-request reuse smoke at concurrency 16 completed `128 / 128`.
- Server/client logs:
    - `runlogs/splitk-torch-empty-smoke-server.log`
    - `runlogs/splitk-torch-empty-smoke-client.log`
- Log scan found no `hipError`, illegal address, traceback, runtime error, or
  assertion.
- Repeated the same smoke with the default
  `VLLM_ROCM_DSV4_ATOM_DECODE_SPLIT_WORKSPACE=workspace`:
    - server log: `runlogs/splitk-workspace-smoke-server.log`
    - client log: `runlogs/splitk-workspace-smoke-client.log`
    - `16 / 16` first-wave smoke passed;
    - `128 / 128` reuse smoke at concurrency 16 passed;
    - log scan found no `hipError`, illegal address, traceback, runtime error, or
    assertion.

Interpretation:

- This does not prove full split-K readiness; both allocator modes were tested
  only at `MAX_NUM_SEQS=16`, `MAX_MODEL_LEN=2048`, and short `max_tokens=8`.
- It does show that `kv_splits=4` is not intrinsically broken under graph mode
  for the reduced shape.
- Since both `torch_empty` and `workspace` passed the reduced shape, the old
  split-K failure is not explained by allocator choice alone. Reproduce it next
  at the historical larger shape, for example `MAX_NUM_SEQS=128` and the
  original request-reuse smoke, then compare allocator modes there.

Larger-shape validation:

- Repeated the split-K smoke at the historical larger shape:
    - `MAX_NUM_SEQS=128`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=2048`
    - `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=4`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- With `VLLM_ROCM_DSV4_ATOM_DECODE_SPLIT_WORKSPACE=workspace`:
    - one `128 / 128` first-wave `/v1/completions` smoke passed;
    - the next four-wave reuse run completed wave 0 (`128 / 128`) then stalled;
    - server log showed `Running: 5 reqs`, zero throughput, then
    `No available shared memory broadcast block found in 60 seconds`;
    - logs:
        - `runlogs/splitk-workspace-maxseq128-smoke-server.log`
        - `runlogs/splitk-workspace-maxseq128-smoke-client.log`
- With `VLLM_ROCM_DSV4_ATOM_DECODE_SPLIT_WORKSPACE=torch_empty`:
    - one `128 / 128` first-wave smoke passed;
    - the same four-wave reuse run completed all waves:
    `512 / 512` requests, no client errors;
    - log scan found no `hipError`, illegal address, traceback, runtime error,
    assertion, or shared-memory hang;
    - logs:
        - `runlogs/splitk-torch-empty-maxseq128-smoke-server.log`
        - `runlogs/splitk-torch-empty-maxseq128-smoke-client.log`

Updated interpretation:

- The split-K kernel path is usable under graph mode at `MAX_NUM_SEQS=128` when
  its partials use ATOM-style allocation.
- The vLLM WorkspaceManager partial-buffer path is the current larger-shape
  replay/reuse blocker for forced split-K decode.
- Making split-K a production default should not use the global
  `WorkspaceManager.get_simultaneous(...)` partials as-is. It needs either:
    - ATOM-style per-call graph-pool tensors; or
    - a graph-safe persistent split-K partial allocator with lifetime isolated
    from unrelated workspace users and from repeated layer calls.
- Updated `launchdeepseekgraph.sh` so
  `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS` defaults to an empty string instead of
  `1`. The Python wrapper treats an empty override as absent, so default launch
  now follows the ATOM/vendored heuristic and explicit overrides remain
  available for diagnostics.
- Heuristic launch validation:
    - launched without a split override at `MAX_NUM_SEQS=128`,
    `MAX_NUM_BATCHED_TOKENS=8192`, `MAX_MODEL_LEN=2048`, graph mode, V2 runner;
    - one `128 / 128` first-wave smoke passed;
    - the four-wave reuse smoke completed `512 / 512` requests;
    - log scan found no `hipError`, illegal address, traceback, runtime error,
    assertion, or shared-memory hang;
    - logs:
        - `runlogs/splitk-heuristic-maxseq128-smoke-server.log`
        - `runlogs/splitk-heuristic-maxseq128-smoke-client.log`
- Accuracy launch fix:
    - an attempted default lmeval launch failed before serving because
    `MAX_MODEL_LEN` defaulted empty, so vLLM tried DeepSeek-V4-Pro's full
    `1048576` context and rejected KV-cache allocation;
    - updated `launchdeepseekgraph.sh` defaults to the prior accuracy-proven
    launch shape: `MAX_MODEL_LEN=8192` and `GPU_MEMORY_UTILIZATION=0.9`.
- Accuracy validation after enabling heuristic split-K:
    - launch: default `launchdeepseekgraph.sh` after the above fixes,
    graph mode, V2 runner, no `--enforce-eager`, max seqs 256;
    - command: unchanged `lmeval.sh`;
    - result: GSM8K flexible exact match `0.9530`, strict exact match `0.9538`;
    - this is inside the requested `0.95 ± 0.01` band;
    - logs:
        - `runlogs/heuristic-splitk-accuracy-server.log`
        - `runlogs/heuristic-splitk-lmeval.log`
- Fresh C32 benchmark after restarting the server:
    - launch: `MAX_NUM_SEQS=32 bash launchdeepseekgraph.sh`;
    - command: `RESULT_PREFIX=heuristic-splitk-current-default CONCURRENCIES=32 bash benchmarkvllm.sh`;
    - successful requests: `320 / 320`;
    - failed requests: `0`;
    - output throughput: `882.5892 tok/s`;
    - total token throughput: `1768.6260 tok/s`;
    - mean TPOT: `35.2812 ms`;
    - logs/results:
        - `runlogs/heuristic-splitk-benchmark-server.log`
        - `runlogs/heuristic-splitk-benchmark-c32.log`
        - `bench-sparsemla/heuristic-splitk-current-default-C32.json`

### Thirteenth Diagnostic Slice

This slice answered whether breakable cudagraph should be used for the target
run and isolated the remaining long-decode failure.

Decision:

- Historical note: the runs in this slice explicitly used
  `VLLM_USE_BREAKABLE_CUDAGRAPH=0` to test the stricter normal compile/graph
  path.
- Current decision after checking `vllm/config/vllm.py`: use breakable
  cudagraph for practical DeepSeek-V4 accuracy/perf runs unless specifically
  testing torch.compile support.
- vLLM auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1` for DeepSeek-V4 when the
  env var is unset because the DSV4 model classes do not carry
  `@support_torch_compile`.

ATOM-attention run:

- Launch:
    - `MAX_NUM_SEQS=128`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `ENFORCE_EAGER=0`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
    - `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`
- `lmeval.sh` was unchanged.
- Result: failed late in GSM8K generation. The server accepted and completed
  many requests, then `VllmWorker-0` died at `2026-06-18 15:36:25`.
- Failure state:
    - `num_running_reqs=118`
    - `num_waiting_reqs=0`
    - `kv_cache_usage=0.2726404963608161`
    - decode step was scheduling one token per request
    - engine then raised `RuntimeError: cancelled` from SHM broadcast after the
    worker death
- Interpretation: this is not KV exhaustion or a graph-capture startup issue.
  It is a late long-decode worker death in the ATOM attention path.

vLLM-attention isolation:

- Launch was identical except `VLLM_ROCM_DSV4_ATOM_ATTENTION=0`.
- This kept:
    - V2 model runner
    - normal cudagraph
    - no `--enforce-eager`
    - ATOM request-state allocation
    - ATOM unified-KV allocation
    - ATOM compressor planning/main compressor flags
    - aiter fused MoE
    - vLLM sparse attention path
- Unchanged `lmeval.sh` passed:
    - GSM8K flexible exact match: `0.9560 +/- 0.0056`
    - GSM8K strict exact match: `0.9568 +/- 0.0056`

Benchmark after a clean server restart:

- Launch:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `ENFORCE_EAGER=0`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_ROCM_DSV4_ATOM_ATTENTION=0`
- `benchmarkvllm.sh` result for random `1024/1024`, concurrency 32,
  320 prompts:
    - successful requests: `320`
    - failed requests: `0`
    - benchmark duration: `384.46 s`
    - output throughput: `852.30 tok/s`
    - total throughput: `1707.93 tok/s`
    - mean TPOT: `36.60 ms`
    - median TPOT: `36.59 ms`
    - P99 TPOT: `37.40 ms`

Updated conclusion:

- The current stable accuracy-passing ROCm DSV4 preview still depends on vLLM's
  existing sparse attention path.
- Breakable cudagraph should not be enabled for target measurements.
- The next blocker for reaching ATOM-like performance is not scheduler
  integration or general request-state allocation; it is the ATOM attention
  decode path under long GSM8K-style generation.
- The remaining attention audit should focus on the ATOM HCA/CSA decode
  wrappers, index construction, and lifetime/aliasing of decode workspaces under
  normal cudagraph replay.

## Main Risks

- Prefix caching: request-state rings are not recoverable from the classical KV
  block cache unless explicit state transfer/restore is implemented.
- Scheduler block size: ATOM uses one DSV4 block size of 128 original tokens.
  vLLM currently uses several block sizes plus compressed `storage_block_size`.
  Changing this globally must be ROCm-only and validated against admission,
  hashing, and prefix-cache behavior.
- MTP/spec decode: ring sizes need `max_spec_steps` slack to avoid draft token
  aliasing.
- KV transfer/disaggregation: vLLM's transfer system expects block regions; a
  unified cache adds slot regions and possibly staging for compressor state.
- CUDA graph capture: all persistent tensors and worst-case workspace sizes
  must be allocated before workspace lock/capture.

## Conclusion

vLLM's existing KV-cache spec system is flexible enough to host a DSV4 ROCm
unified cache, but the current built-in specs are block-table/token-page
oriented. ATOM's DSV4 layout additionally requires request-scoped ring slots.
The recommended path is not to replace vLLM's scheduler, but to add a ROCm-only
DSV4 cache spec plus request-state-slot allocator/binder, then adapt ROCm DSV4
metadata and kernels to read a unified cache directly. CUDA should remain on
the current `MLAAttentionSpec` / `SlidingWindowMLASpec` path.

## Fourteenth Diagnostic Slice

HCA decode index translation had one confirmed integration bug:

- ATOM's HCA compressed cache assumes one compressed entry per physical block.
- vLLM can use a larger scheduler block size, for example `block_size=256`.
- With DSV4 HCA `compress_ratio=128`, that gives two compressed entries per
  physical vLLM block.
- The ATOM-style HCA writer was using the compressed entry offset directly as a
  block-table column, which is only valid when the compressed block capacity is
  one.

The ROCm HCA compressed-head writer now computes:

- `hca_block_capacity = attn_metadata.block_size // compress_ratio`
- `block_offsets = hca_offsets // hca_block_capacity`
- `slot_offsets = hca_offsets % hca_block_capacity`
- physical page = `block_table[request, block_offsets]`
- final compact page = `physical_page * hca_block_capacity + slot_offsets`

Syntax validation passed for:

- `vllm/models/deepseek_v4/amd/rocm.py`
- `vllm/models/deepseek_v4/amd/model_state.py`
- `vllm/models/deepseek_v4/compressor.py`

Post-fix short smoke:

- Launch:
    - `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
    - `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
    - `ENFORCE_EAGER=0`
    - `MAX_NUM_SEQS=128`
    - `MAX_NUM_BATCHED_TOKENS=8192`
- Graph capture completed with `VLLM_USE_BREAKABLE_CUDAGRAPH=0`, which was the
  stricter diagnostic graph mode used at that point in the investigation.
- Random `1024/128`, concurrency 128, 128 prompts completed from the client
  perspective:
    - successful requests: `128`
    - failed requests: `0`
    - output throughput: `959.72 tok/s`
    - mean TPOT: `87.47 ms`
- The worker still died at the tail of the run:
    - `VllmWorker-1 died unexpectedly`
    - scheduler dump showed about 120 running requests and no waiting requests
    - KV cache usage was around 21.9%
    - output tokens were in the `112-127` range

Interpretation:

- The HCA block-capacity fix removed a concrete indexing mismatch with vLLM's
  scheduler block table.
- It did not make the ATOM attention path stable enough for full lm-eval.
- The next isolation should split ATOM attention by compression ratio:
    - CSA-only: `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS=4`
    - HCA-only: `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS=128`
- If HCA-only fails, compare native vLLM HCA index construction with
  `VLLM_ROCM_DSV4_ATOM_HCA_NATIVE_INDICES=1`.

## Fifteenth Diagnostic Slice

The next slice split the HCA failure between attention metadata, ATOM main
compressor writes, and vLLM native compressor side effects. All runs used:

- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- `ENFORCE_EAGER=0`
- `MAX_NUM_SEQS=128`
- `MAX_NUM_BATCHED_TOKENS=8192`
- HCA-only ATOM attention:
  `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS=128`
- decode output bypass:
  `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_DECODE=1`
- decode index write bypass:
  `VLLM_ROCM_DSV4_ATOM_SKIP_DECODE_INDEX_WRITE=1`
- random `128/128`, concurrency 128, 1280 main prompts after 128 warmups.

Observed runs:

- `ATOM_MAIN_COMPRESSOR=1`, fused-compress enabled, state-update disabled:
  warmup completed, then main run returned `1280/1280` internal errors.
  Worker died at the tail with scheduler rows around `num_computed_tokens=258`
  and `num_output_tokens=127`.
- `ATOM_MAIN_COMPRESSOR=1`, fused-compress disabled, state-update disabled:
  same failure shape. This proves the crash is not solely the HCA fused
  compressor cache scatter and not solely the HCA state-update kernel.
- Added diagnostic
  `VLLM_ROCM_DSV4_ATOM_NATIVE_AFTER_MAIN_COMPRESSOR=1`, which runs the ATOM
  compressor branch and then falls through to vLLM's native compressor.
  With fused/state both disabled, this still failed in the same way. Therefore
  the failure is caused by side effects that happen before those skips or by
  the HCA ATOM attention/metadata path itself, not by missing native compressor
  population alone.
- Changed the ATOM compressor RoPE split to use non-copying strided views when
  `rotary_emb.cos_sin_cache.dtype` already matches the compressor dtype, and
  relaxed the fused-compress stride validation. This removes a large per-layer
  contiguous RoPE cache allocation hazard, but the same diagnostic run still
  failed. So the repeated RoPE allocation was a real integration risk, but not
  the immediate HCA worker-death trigger in this smoke.

Important code-path detail:

- Even when both `VLLM_ROCM_DSV4_ATOM_SKIP_FUSED_COMPRESS=1` and
  `VLLM_ROCM_DSV4_ATOM_SKIP_COMPRESS_STATE_UPDATE=1` are set, the current
  ATOM main-compressor wrapper still:
    - fetches ATOM state metadata and compression plans,
    - fetches the vLLM compressed-cache metadata,
    - builds or reuses the HCA flattened block table when
    `VLLM_ROCM_DSV4_ATOM_HCA_FLAT_CACHE=1`,
    - prepares RoPE views.

Current interpretation:

- The HCA block-capacity fix is necessary but insufficient.
- HCA `update_compressor_states` is sufficient to kill the worker in prior
  isolations, but it is not the only unsafe component.
- The new tight suspect set is:
    - HCA flattened block-table construction/replay,
    - HCA ATOM decode metadata lifetime under vLLM graph replay,
    - ATOM HCA attention path bookkeeping even when paged decode and decode
    index writes are bypassed.
- The next highest-signal run is the same native-fallback/skip-both diagnostic
  with `VLLM_ROCM_DSV4_ATOM_HCA_FLAT_CACHE=0`. If that survives, the flattened
  block-table kernel/lifetime is the immediate crash boundary. If it still
  fails, the issue is outside HCA compressor cache-table flattening and likely
  in ATOM state metadata or the HCA attention branch entry/graph replay.

## Sixteenth Diagnostic Slice

The no-flat-cache-table diagnostic used the exact Fifteenth Slice setup, plus:

- `VLLM_ROCM_DSV4_ATOM_HCA_FLAT_CACHE=0`
- `VLLM_ROCM_DSV4_ATOM_NATIVE_AFTER_MAIN_COMPRESSOR=1`
- `VLLM_ROCM_DSV4_ATOM_SKIP_FUSED_COMPRESS=1`
- `VLLM_ROCM_DSV4_ATOM_SKIP_COMPRESS_STATE_UPDATE=1`

Result:

- random `128/128`, concurrency 128, 128 warmups + 1280 main prompts
- successful requests: `1280`
- failed requests: `0`
- benchmark duration: `71.30 s`
- output throughput: `2297.80 tok/s`
- total throughput: `4667.40 tok/s`
- mean TPOT: `48.27 ms`
- server remained alive after the run
- logs showed first-shape JIT warnings only, with no worker death or scheduler
  dump

Interpretation:

- The stricter `VLLM_USE_BREAKABLE_CUDAGRAPH=0` path was not the blocker for
  this narrowed configuration, but it is no longer the default target for
  practical DSV4 benchmarking.
- Breakable cudagraph is not the same as `--enforce-eager`: it disables the
  torch.compile pipeline while keeping a cudagraph serving path.
- The immediate HCA worker-death boundary is the flattened compressed-cache
  block-table path enabled by `VLLM_ROCM_DSV4_ATOM_HCA_FLAT_CACHE=1`.
- The next code fix should focus on making the flattened HCA cache-table buffer
  graph-stable and scheduler-stable, or replacing it with direct block/slot
  addressing inside the ATOM compressor kernels so no transient flattened table
  has to be captured or replayed.

## Seventeenth Diagnostic Slice

The HCA compressor was re-tested with packed cache addressing instead of the
flattened block table. This uses the fused compressor kernel's native address
resolution:

- `ci = position // 128`
- `block_in_seq = ci // k_per_block`
- `slot_in_block = ci % k_per_block`
- `physical_block = block_table[batch_id, block_in_seq]`

This matches the ATOM decode HCA index writer, which emits:

- `swa_pages + physical_block * hca_block_capacity + slot_in_block`

Two normal-cudagraph runs were tested:

- shared settings:
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
    - `ENFORCE_EAGER=0`
    - `MAX_NUM_SEQS=128`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - HCA-only ATOM attention:
    `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS=128`
    - `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
    - `VLLM_ROCM_DSV4_ATOM_HCA_FLAT_CACHE=0`
    - `VLLM_ROCM_DSV4_ATOM_NATIVE_AFTER_MAIN_COMPRESSOR=1`
    - decode output bypass:
    `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_DECODE=1`
    - decode index write bypass:
    `VLLM_ROCM_DSV4_ATOM_SKIP_DECODE_INDEX_WRITE=1`
    - random `128/128`, concurrency 128, 128 warmups + 1280 main prompts
- fused HCA compressor enabled, HCA state update disabled:
    - successful requests: `1280`
    - failed requests: `0`
    - output throughput: `2271.77 tok/s`
    - total throughput: `4614.53 tok/s`
    - mean TPOT: `48.77 ms`
- fused HCA compressor enabled, HCA state update enabled:
    - successful requests: `1280`
    - failed requests: `0`
    - output throughput: `2262.01 tok/s`
    - total throughput: `4594.71 tok/s`
    - mean TPOT: `49.00 ms`

Code decision:

- `VLLM_ROCM_DSV4_ATOM_HCA_FLAT_CACHE` now defaults to `0`.
- The flattened HCA block table remains only as an explicit diagnostic branch
  via `VLLM_ROCM_DSV4_ATOM_HCA_FLAT_CACHE=1`.
- Packed block/slot addressing is the production default because it is already
  supported by the fused compressor and agrees with the HCA decode index layout.

Interpretation:

- The fused HCA compressor path is graph-stable with vLLM's packed KV blocks.
- The HCA compressor state-update kernel is also graph-stable once the
  flattened cache-table path is removed.
- The previous worker-death pattern is now isolated to
  `VLLM_ROCM_DSV4_ATOM_HCA_FLAT_CACHE=1`, not to breakable cudagraph, fused HCA
  compression, or HCA compressor state writes.
- The next boundary is the ATOM HCA decode index writer and then the actual
  ATOM paged-decode attention kernel, with packed HCA indices.

## Eighteenth Diagnostic Slice

The next two runs enabled the HCA decode-index writer and then the actual ATOM
HCA paged-decode attention kernel. Both used:

- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- `ENFORCE_EAGER=0`
- `MAX_NUM_SEQS=128`
- `MAX_NUM_BATCHED_TOKENS=8192`
- HCA-only ATOM attention:
  `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS=128`
- `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- `VLLM_ROCM_DSV4_ATOM_NATIVE_AFTER_MAIN_COMPRESSOR=1`
- packed HCA cache addressing by default
- random `128/128`, concurrency 128, 128 warmups + 1280 main prompts

Run A enabled HCA index writes but still bypassed paged decode:

- `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_DECODE=1`
- successful requests: `1280`
- failed requests: `0`
- output throughput: `2256.88 tok/s`
- total throughput: `4584.30 tok/s`
- mean TPOT: `49.19 ms`

Run B enabled the actual ATOM HCA paged-decode attention kernel:

- no decode-index bypass
- no paged-decode bypass
- successful requests: `1280`
- failed requests: `0`
- output throughput: `2197.38 tok/s`
- total throughput: `4463.42 tok/s`
- mean TPOT: `50.71 ms`
- server remained alive
- logs had no worker-death or runtime-error entries

Interpretation:

- The packed HCA decode-index writer is graph-stable.
- The ATOM HCA paged-decode attention kernel is graph-stable for this random
  smoke under vLLM V2 runner and normal cudagraph.
- The remaining broadening step is enabling both CSA and HCA ATOM attention
  paths together, then running accuracy before treating the integration as
  usable.

## Nineteenth Diagnostic Slice

The all-ratios ATOM decode smoke enabled both CSA and HCA ATOM attention paths:

- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- `ENFORCE_EAGER=0`
- `MAX_NUM_SEQS=128`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
- no `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS` filter
- `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- `VLLM_ROCM_DSV4_ATOM_NATIVE_AFTER_MAIN_COMPRESSOR=1`
- packed HCA cache addressing by default
- random `128/128`, concurrency 128, 128 warmups + 1280 main prompts

Result:

- graph capture completed in normal vLLM FULL_AND_PIECEWISE mode
- successful requests: `1280`
- failed requests: `0`
- output throughput: `2241.08 tok/s`
- total throughput: `4552.20 tok/s`
- mean TPOT: `50.54 ms`
- server remained alive
- logs had no worker-death or runtime-error entries

Interpretation:

- CSA and HCA ATOM decode paths can run together under the vLLM scheduler,
  V2 model runner, and normal cudagraph for this random smoke.
- The next required gate is unchanged `lmeval.sh` accuracy. Runtime stability
  alone is not enough because the ATOM unified-cache path can still be
  numerically wrong while producing valid tokens.

Accuracy gate:

- The unchanged `lmeval.sh` command was run against this all-ratios ATOM decode
  server.
- GSM8K flexible exact match: `0.9560 +/- 0.0056`
- GSM8K strict exact match: `0.9568 +/- 0.0056`
- This is inside the required `0.95 +/- 0.01` target band.
- The run used `VLLM_ROCM_DSV4_ATOM_NATIVE_AFTER_MAIN_COMPRESSOR=1`, so the
  native compressor cache remained populated as a transition/debug side effect.
  The attention path itself still used the ATOM unified-cache decode path for
  both CSA and HCA.

## Twentieth Diagnostic Slice

After the all-ratios accuracy pass, the server was restarted to clear KV and
prefix-cache state before collecting the C32 performance number. The benchmark
server used:

- `MAX_NUM_SEQS=32`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `ENFORCE_EAGER=0`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
- `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- no `VLLM_ROCM_DSV4_ATOM_NATIVE_AFTER_MAIN_COMPRESSOR`
- no `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS` filter
- packed HCA cache addressing by default

The benchmark command was `benchmarkvllm.sh` with its target C32 workload:

- input length: `1024`
- output length: `1024`
- concurrency: `32`
- prompts: `320`
- warmups: `32`

Result:

- successful requests: `320`
- failed requests: `0`
- benchmark duration: `388.74 s`
- output throughput: `842.93 tok/s`
- total throughput: `1689.15 tok/s`
- mean TPOT: `37.09 ms`
- median TPOT: `37.06 ms`
- P90 TPOT: `37.52 ms`
- P99 TPOT: `37.77 ms`
- logs had no worker-death or runtime-error entries

Interpretation:

- The all-ratios ATOM decode path is now stable enough to pass GSM8K accuracy
  and complete the requested C32 benchmark without `--enforce-eager` or
  breakable cudagraph.
- This performance does not yet approach the ATOM recipe target for C32
  (`1145.71 tok/s` output throughput, `26.90 ms` mean TPOT). It is also close
  to the earlier stable vLLM sparse-attention baseline rather than a clear
  ATOM-kernel win, so the remaining work is performance-oriented rather than
  runtime stability-oriented.

## Twenty-first Diagnostic Slice

The no-native-after accuracy gap was closed with the target deployment config:

- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- `ENFORCE_EAGER=0`
- `MAX_NUM_SEQS=256`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
- `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- no `VLLM_ROCM_DSV4_ATOM_NATIVE_AFTER_MAIN_COMPRESSOR`
- no `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS` filter
- packed HCA cache addressing by default

The unchanged `/app/atomdsv4/lmeval.sh` command was run against a fresh server.
Result artifact:

- `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-18T17-58-16.432667.json`

Accuracy result:

- GSM8K flexible exact match: `0.9583 +/- 0.0055`
- GSM8K strict exact match: `0.9591 +/- 0.0055`
- sample count: `1319`

Interpretation:

- The all-ratios ATOM compressor plus ATOM decode path no longer depends on the
  native-after compressor diagnostic to satisfy the requested GSM8K accuracy
  band.
- At the time of this slice, the validated config used
  `VLLM_USE_BREAKABLE_CUDAGRAPH=0` and did not use `--enforce-eager`.
- Current practical DSV4 runs should allow breakable cudagraph unless the
  specific experiment is testing torch.compile/normal-graph compatibility.
- The remaining gap is not correctness for this gate. The remaining gap is that
  C32 performance is still at `842.93 tok/s` output throughput, slightly below
  the previous stable vLLM sparse-attention baseline of `852.30 tok/s`, and well
  below the ATOM recipe C32 target of `1145.71 tok/s`.

## Twenty-second Diagnostic Slice

The next integration slice starts moving ATOM unified KV from a model-state side
allocation into vLLM-owned KV-cache storage. This is opt-in and does not change
the validated default path.

New opt-in flag:

- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`

Code added:

- `DeepseekV4AtomMLAAttentionSpec` in `vllm/v1/kv_cache_interface.py`
    - subclass of `MLAAttentionSpec`
    - keeps the normal compressed MLA page-size contract
    - adds `atom_swa_prefix_bytes`, `atom_swa_pages`, and `atom_swa_dtype`
- DeepSeek-V4 allocator support in `vllm/v1/core/kv_cache_utils.py`
    - subtracts fixed ATOM SWA prefix bytes before computing `num_blocks`
    - emits per-layer `KVCacheTensor` storage for ATOM-prefixed layers
    - leaves the existing DeepSeek-V4 bucketed sharing path for regular layers
- reshape support in `vllm/v1/worker/gpu/attn_utils.py`
    - skips the SWA prefix before forming the normal compressed `kv_cache` tensor
    - preserves the existing attention-module `kv_cache` contract
- ROCm DSV4 spec emission in `vllm/models/deepseek_v4/attention.py`
    - only on ROCm
    - only when `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
    - CUDA still emits the original `MLAAttentionSpec`
- model-state binding in `vllm/models/deepseek_v4/amd/model_state.py`
    - attempts to alias ATOM unified views from vLLM-owned raw storage
    - binds `atom_swa_kv`, `atom_unified_kv`, and compressor `atom_kv_cache`
    - falls back to the known side allocation if the dtype contract is not
    compatible

Static check:

- `python -m py_compile vllm/v1/kv_cache_interface.py vllm/v1/core/kv_cache_utils.py vllm/v1/worker/gpu/attn_utils.py vllm/models/deepseek_v4/attention.py vllm/models/deepseek_v4/amd/model_state.py`
- result: pass

Current limitation:

- The current ATOM decode kernel call takes one `atom_unified_kv` tensor.
- The deployment cache tail can be `uint8`/fp8 layout while the SWA ring is
  model dtype.
- Therefore the fp8 deployment cannot fully use vLLM-owned unified storage
  until the ATOM ROCm kernels accept either:
    - split typed views: `swa_kv` plus compressed tail, or
    - a raw unified byte allocation plus layout metadata.

Interpretation:

- vLLM has enough scheduler/model-state structure to host persistent ATOM
  request rings without GPU-worker changes.
- vLLM also has a viable ROCm-only KV-cache planning hook for ATOM-prefixed
  per-layer storage.
- The missing component for full ATOM benefit is now narrower: the attention
  and compressor kernel interface must stop assuming one homogeneous
  `atom_unified_kv` tensor when the target deployment uses fp8 compressed KV.

## Twenty-third Diagnostic Slice

The vLLM-owned ATOM unified-KV path was advanced from planning-only to an
allocation/binding smoke.

Additional code changes:

- `KVCacheTensor.fixed_prefix_size`
    - lets vLLM carry tensors with a fixed SWA prefix plus a scalable compressed
    KV tail
    - fixes the final cross-rank `min_num_blocks` shrink step so only the
    per-block tail is scaled
- per-spec cache dtype in `vllm/v1/worker/gpu/attn_utils.py`
    - the reshape path now passes `kv_cache_spec.cache_dtype_str` to the backend
    shape helper when the spec provides one
    - this avoids using global `fp8_ds_mla` shape rules for the opt-in bf16 ATOM
    tail
- opt-in ATOM vLLM-owned spec now advertises `cache_dtype_str="bf16"`
    - this is not the final fp8 mixed-layout target
    - it is a mechanical validation path for vLLM-owned storage with the current
    homogeneous-`atom_unified_kv` kernels

Smoke command:

- `MAX_NUM_SEQS=16`
- `MAX_NUM_BATCHED_TOKENS=2048`
- `ENFORCE_EAGER=0`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
- `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
- no `VLLM_ROCM_DSV4_ATOM_NATIVE_AFTER_MAIN_COMPRESSOR`

Observed successful boundary:

- KV-cache config was generated successfully.
- GPU KV cache size was logged as `1,601,640` tokens for this small smoke.
- All 8 workers logged:
    - `Bound ROCm DSV4 ATOM unified KV views from vLLM-owned KV storage`
    - `active_layers=61`
    - `num_blocks=7486`
    - `swa_pages=2048`

Observed failure after binding:

- Engine failed during worker warmup/compile.
- Root error came from the native fp8 cache dequant path:
    - `Cannot bitcast data-type of size 16 to data-type of size 8`
    - the failing code attempted to load fp8 token bytes and bitcast them to fp8
    - the opt-in ATOM cache tail was bf16, so this native path was reading the
    wrong layout

Interpretation:

- vLLM can now allocate and bind the ATOM SWA-prefix + compressed-tail storage
  without GPU-worker changes.
- The remaining gap is no longer the KV-cache allocator itself.
- The remaining gap is attention/backend cache-op routing:
    - with `--kv-cache-dtype fp8`, some native DeepSeek-V4 cache update/gather
    paths still assume global `fp8_ds_mla` layout during warmup/prefill
    - the ATOM path needs either a split typed kernel contract or backend metadata
    that routes ATOM-owned layers away from native fp8 dequant/update helpers
    - the final target should be mixed-layout-aware rather than the temporary
    homogeneous-bf16 validation path

## Twenty-fourth Diagnostic Slice

The warmup failure above was fixed by adding a ROCm-local plain BF16 K-cache
gather path.

Additional code changes:

- `vllm/models/deepseek_v4/amd/rocm.py`
    - added `_gather_plain_k_cache_kernel`
    - added `_gather_plain_k_cache`
    - added `_gather_k_cache`
    - native prefill now dispatches:
        - `uint8` cache: existing `dequantize_and_gather_k_cache`
        - BF16/model-dtype cache: direct plain gather

Smoke command:

- same as the Twenty-third slice, except the launch path now defaults to
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- this follows vLLM's DeepSeek-V4 default: DSV4 classes do not carry
  `@support_torch_compile`, so breakable cudagraph is the supported graph path
  unless explicitly disabled
- `--enforce-eager` is still not used

Observed successful boundary:

- Server reached readiness.
- `/v1/models` returned HTTP 200.
- All workers bound vLLM-owned ATOM unified KV views.
- A tiny completion request completed without crashing.

Observed quality signal:

- The tiny prompt output was nonsensical, so this smoke proves runtime
  viability only.
- It is not an accuracy result.
- GSM8K/lm-eval is still required before treating this path as usable.

Important implementation gap:

- ATOM's modeling file uses `sparse_attn_v4_paged_prefill` for prefill and
  `sparse_attn_v4_paged_decode` for decode over `unified_kv`.
- The current vLLM ROCm adapter only routes decode through the ATOM paged
  decode path.
- Prefill still uses the native ROCm sparse prefill flow:
    - gather compressed cache rows into a workspace
    - gather SWA rows into a workspace
    - combine top-k and SWA indices
    - call `rocm_sparse_attn_prefill`
- The paged-prefill index kernel already exists in
  `vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill_indices.py`, but the
  model state does not yet expose persistent prefill indptr/index buffers, and
  `DeepseekV4ROCMAiterAttention._forward_prefill` does not call
  `sparse_attn_v4_paged_prefill`.

Current component status:

- Fused MoE: enabled through vLLM `--moe-backend aiter`.
- MHC/HC/QK norm-RoPE: ATOM/aiter paths are enabled in the ROCm DSV4 adapter.
- Classical KV block size: ATOM uses `128` original tokens
  (`lcm(4, 128) = 128`). The launch script now defaults to
  `BLOCK_SIZE=128`, and the ROCm ATOM spec path raises early if a different
  vLLM block size is used.
- Main compressor: ATOM-style `fused_compress_attn` followed by
  `update_compressor_states` is wired behind
  `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`.
- Request state rings: vLLM V2 `ModelState` owns persistent SWA and compressor
  state slots without GPU-worker changes.
- Unified KV allocation:
    - side allocation path is runnable and accuracy-validated in previous runs
    - vLLM-owned allocation path can bind, run, pass GSM8K, and complete C32
    benchmark validation in later runs
- ATOM decode attention: wired through `sparse_attn_v4_paged_decode`.
- ATOM prefill attention: wired through `sparse_attn_v4_paged_prefill` for the
  validated path, but still carries significant vLLM-side metadata/index
  preparation overhead.

## Twenty-fifth Diagnostic Slice

ATOM's DeepSeek-V4 model uses a classical KV block size of 128 original tokens,
not vLLM's previous DSV4 default of 256. Moving the launch default to
`--block-size 128` exposed two vLLM block-size assumptions:

- `DeepseekV4FlashMLABackend.get_supported_kernel_block_sizes()` advertised
  only `256`.
- `DeepseekV4IndexerBackend.get_supported_kernel_block_sizes()` advertised
  only `256`.

The V2 worker selects one kernel block size per KV group. With a KV-manager
block size of 128, those declarations made `select_common_block_size()` fail
with `ValueError: No common block size for 128`.

Fix applied:

- Under cached ROCm ATOM feature gates only, both DSV4 sparse MLA and DSV4
  indexer backends now advertise `[128, 256]`.
- CUDA/NVIDIA and non-ATOM paths keep the original `256` declaration.
- ROCm ATOM SWA pages are scaled as `cache_config.block_size // 4`, so
  `128 -> 32`.
- ROCm ATOM compressor state pages are scaled relative to the classical block:
  `ratio=4 -> block_size // 64`, `ratio=128 -> block_size // 32`.

Validation:

- Small V2 smoke launched with `--block-size 128`, breakable cudagraph enabled,
  and no `--enforce-eager`.
- KV-cache allocation succeeded:
    - available KV memory: 30.27 GiB
    - GPU KV cache size: 4,594,363 tokens
- Breakable cudagraph capture completed in both PIECEWISE and FULL phases.
- `/v1/models` returned HTTP 200.
- Tiny completion request returned the expected arithmetic answer prefix:
  `Question: What is 2+2? Answer:` -> `4`.

Next validation:

- Restart with lmeval-sized server capacity (`max_num_seqs=256`) and run the
  unchanged `lmeval.sh` to verify GSM8K accuracy after the block-size change.

Results:

- GSM8K with unchanged `lmeval.sh`, server launched with
  `MAX_NUM_SEQS=256`, `MAX_NUM_BATCHED_TOKENS=8192`, `BLOCK_SIZE=128`,
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, no `--enforce-eager`:
    - flexible-extract exact match: `0.9530 ± 0.0058`
    - strict-match exact match: `0.9538 ± 0.0058`
    - no lm-eval or server errors
- C32 benchmark with fresh server restart, `MAX_NUM_SEQS=32`,
  `MAX_NUM_BATCHED_TOKENS=8192`, `BLOCK_SIZE=128`,
  `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, no `--enforce-eager`:
    - output throughput: `840.57 tok/s`
    - total throughput: `1684.43 tok/s`
    - request throughput: `0.82 req/s`
    - mean TPOT: `37.07 ms`
    - median TPOT: `37.02 ms`
    - mean TTFT: `1054.92 ms`
    - zero failed requests

Notes:

- A `MAX_NUM_BATCHED_TOKENS=32768` benchmark launch did not fit with the current
  ROCm ATOM side allocations and graph profile; vLLM reported
  `Available KV cache memory: -10.24 GiB`.
- `MAX_NUM_BATCHED_TOKENS=8192` is the validated runnable profile for both
  lmeval and C32 benchmark in this slice.
- This C32 benchmark is essentially tied with the earlier validated
  side-allocation run (`842.93 tok/s`, `37.09 ms` mean TPOT) and still trails
  the earlier vLLM sparse baseline (`852.30 tok/s`) and ATOM recipe C32 target
  (`1145.71 tok/s`).

## Twenty-sixth Diagnostic Slice

Goal: test whether the copied ATOM paged-prefill attention path is correctness
ready, and keep the default launch on an accuracy-valid configuration.

Paged prefill experiment:

- Added persistent prefill indptr/index buffers to the ROCm ATOM model state.
- Routed prefill through `sparse_attn_v4_paged_prefill` when
  `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_PREFILL=0`.
- Fixed one real stability bug: CSA/SWA layers were calling the prefill index
  writer with HCA writes enabled, so the HCA section could write beyond the
  intentionally tiny HCA buffer. The writer now has a `write_hca` flag.
- Compared the vLLM metadata formulas against ATOM commit
  `e95ef5d74a860e04a6219dfff319535bc19449dd`; CSA valid length formula matches
  ATOM: `min((position + 1) // 4, committed_csa, index_topk)`.
- Restored the CSA prefill SWA placement to match that ATOM commit's index
  writer. Note: the ATOM comments and CSA translator comments disagree about
  whether SWA occupies the head or tail of the CSA slice, so this remains a
  risk area.

Paged prefill validation:

- Small prompt smoke passed.
- Long prompt smoke passed.
- Full unchanged `lmeval.sh` failed badly with paged prefill enabled:
    - CSA tail-placement attempt: flexible `0.1016`, strict `0.0781`
    - ATOM-commit head-placement attempt: flexible `0.0917`, strict `0.0728`
- Conclusion: the paged-prefill sparse attention path is runtime-stable but not
  correctness-valid. Do not use it for performance claims.

Compressor-first experiment:

- ATOM's modeling file launches compressor/indexer before sparse attention so
  `unified_kv` compressed pages are populated before attention reads them.
- Enabling `VLLM_ROCM_DSV4_ATOM_COMPRESS_FIRST=1` in this partial vLLM port did
  not pass even a tiny smoke: `Question: What is 2+2?` returned garbage.
- Conclusion: full ATOM ordering is not ready in vLLM yet. The likely missing
  pieces are exact compressor/indexer state/cache binding and ordering across
  native fallback paths, not just the attention kernel call itself.

Default restored:

- `launchdeepseekgraph.sh` now defaults to:
    - `VLLM_ROCM_DSV4_ATOM_COMPRESS_FIRST=0`
    - `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_PREFILL=1`
- This keeps the server on the accuracy-valid path while leaving the
  experimental paged-prefill and compressor-first paths opt-in.

Validation after restoring defaults:

- Server: `MAX_NUM_SEQS=256`, `MAX_NUM_BATCHED_TOKENS=8192`,
  `BLOCK_SIZE=128`, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, no `--enforce-eager`.
- Unchanged `lmeval.sh`:
    - flexible-extract exact match: `0.9530 ± 0.0058`
    - strict-match exact match: `0.9538 ± 0.0058`
    - zero lm-eval/server errors
- C32 benchmark after fresh server restart:
    - server: `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=8192`,
    `BLOCK_SIZE=128`, no `--enforce-eager`
    - successful requests: `320`
    - failed requests: `0`
    - benchmark duration: `389.35 s`
    - output throughput: `841.60 tok/s`
    - total throughput: about `1686 tok/s`
    - request throughput: `0.82 req/s`
    - mean TTFT: `1097.20 ms`
    - mean TPOT: `36.98 ms`
    - median TPOT: `36.82 ms`

Current interpretation:

- We do not yet have all necessary components to get the benefit of all ATOM
  kernels under vLLM's scheduler.
- The model has enough components for a correct ROCm preview using vLLM
  scheduler, V2 model runner, vLLM KV allocation, ATOM-side request state
  buffers, and ATOM decode/state kernels.
- The missing correctness-valid pieces for full ATOM benefit are:
    - exact prefill unified-KV population/read ordering
    - paged-prefill CSA/SWA packed layout parity
    - compressor-first execution parity without breaking native fallback paths
    - less Python/NumPy metadata preparation in the hot path
    - eventually a ROCm-only unified DSV4 KV-cache spec/allocation path instead
    of side allocation plus vLLM paged-cache adaptation

## Twenty-seventh Diagnostic Slice: ATOM Decode Overhead Attribution

Question:

- Could conversion logic and metadata preparation hide the benefit of a faster
  ATOM attention kernel?

Code added:

- `vllm/models/deepseek_v4/amd/rocm.py` now has opt-in profiling flags:
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=0` by default; set `-1` for all layers.
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=200` by default.
- The values are cached at module import, matching the rest of the ROCm ATOM
  flags and avoiding repeated `os.environ.get` in the hot path.

What it measures:

- `ATOM_PROFILE_DECODE` synchronizes around the ATOM decode wrapper segments:
    - `index_ms`: ATOM decode indptr/index writes.
    - `translate_ms`: CSA/HCA dense/native index translation into ATOM page ids.
    - `kernel_ms`: `sparse_attn_v4_paged_decode` only.
    - `total_ms`: wrapper total for the profiled layer call.
- `ATOM_PROFILE_METADATA` synchronizes around the ROCm metadata builders:
    - `base_ms`: inherited vLLM sparse MLA/SWA metadata construction.
    - `ragged_ms`: dense-to-ragged conversion and graph-stable buffer copy added
    for the ATOM decode wrapper.
    - `total_ms`: builder total.

Important caveat:

- Profiling mode intentionally synchronizes the device and prints from Python,
  so it must not be used for final throughput numbers. Use it only in short
  diagnostic runs to determine whether the kernel itself is fast while wrapper
  preparation is expensive.

Suggested diagnostic run:

```bash
VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=0 \
VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=50 \
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
BLOCK_SIZE=128 \
ENFORCE_EAGER=0 \
bash /app/atomdsv4/launchdeepseekgraph.sh
```

Interpretation:

- If `kernel_ms` is lower than vLLM sparse attention but `total_ms` is not, the
  current integration is losing the ATOM benefit in metadata/index conversion.
- If `ragged_ms` or `base_ms` is large, the next optimization target is moving
  those conversions into persistent request state, backend metadata, or a
  ROCm-only unified KV-cache/index layout instead of rebuilding dense-to-ragged
  views every decode step.
- If `index_ms` dominates, the ATOM ring/index state is not yet being maintained
  in the same incremental form ATOM expects, and the decode path is paying an
  avoidable compatibility translation cost.

Profiler implementation note:

- The profiler must not call `torch.cuda.synchronize()` while CUDA graph capture
  is active on ROCm. The hook now checks `torch.cuda.is_current_stream_capturing`
  and suppresses synchronized timing during capture. This keeps normal
  no-`--enforce-eager` startup valid.

Short profiled run:

- Server config:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `BLOCK_SIZE=128`
    - `ENFORCE_EAGER=0`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=0`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=50`
- Smoke request:
    - prompt: `Question: What is 2+2? Answer:`
    - `max_tokens=16`, `temperature=0`
    - first answer token was correct: `4`
- Log: `/app/atomdsv4/dsv4_profile_overhead_server.log`

Observed decode timings:

- Layer 0, ratio 128, T=32, per-rank steady samples were roughly:
    - `index_ms`: about `2.1 ms`
    - `translate_ms`: about `1.5 ms`
    - `kernel_ms`: about `2.9 ms`
    - `total_ms`: about `6.5 ms`
- T=24 samples:
    - `index_ms`: about `1.04 ms`
    - `translate_ms`: about `0.05 ms`
    - `kernel_ms`: about `0.08 ms`
    - `total_ms`: about `1.16 ms`
- T=16 samples:
    - `index_ms`: about `0.065 ms`
    - `translate_ms`: about `0.036 ms`
    - `kernel_ms`: about `0.10 ms`
    - `total_ms`: about `0.20 ms`

Observed metadata timings:

- Warm-ish T=32 HCA metadata samples after cold setup:
    - MLA ratio 128 base metadata: about `0.3-0.4 ms`
    - dense-to-ragged/copy: about `0.1-0.17 ms`
    - total: about `0.44-0.53 ms`
- Cold/setup samples are much larger:
    - MLA ratio 128 T=32 total around `7 ms`
    - SWA first T=32 metadata around `16 ms`
    - These appear during graph warmup/capture setup and should not be mixed with
    steady runtime estimates.

Conclusion:

- Yes, conversion and metadata preparation can hide the benefit of the ATOM
  attention kernel in the current vLLM integration.
- For T=32 HCA decode in the current wrapper, non-kernel work is roughly
  `3.6 ms` (`index + translate`) versus about `2.9 ms` in the ATOM decode
  kernel for layer 0. The wrapper is therefore spending more time preparing
  ATOM-compatible indices than running the attention kernel.
- For smaller T samples, index preparation can dominate even more sharply.
- This supports the earlier architecture conclusion: to get ATOM-level benefit
  under vLLM, we need to remove compatibility translation from the hot path by
  maintaining ROCm DSV4 request-state rings/unified-KV indices in the format the
  ATOM kernels consume, not rebuild/translate from vLLM dense/block-table
  metadata every decode step.

HCA native-index experiment:

- Tested `VLLM_ROCM_DSV4_ATOM_HCA_NATIVE_INDICES=1`, which avoids the current
  ATOM-side HCA head writer and copies prebuilt vLLM HCA ragged indices into the
  ATOM unified-KV index space.
- Profiled smoke config:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `BLOCK_SIZE=128`
    - `ENFORCE_EAGER=0`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - profile layer 0
- Smoke request still produced a correct first answer token for `2+2`.

Profile result versus default:

- Default HCA writer, T=32 median:
    - `index_ms`: `2.111`
    - `translate_ms`: `1.502`
    - `kernel_ms`: `2.902`
    - `total_ms`: `6.514`
- HCA native indices, T=32 median:
    - `index_ms`: `2.083`
    - `translate_ms`: `0.010`
    - `kernel_ms`: `2.912`
    - `total_ms`: `4.998`
- HCA native indices, T=24 median:
    - default total: `1.163 ms`
    - native-index total: `1.128 ms`
- HCA native indices, T=16 median:
    - default total: `0.197 ms`
    - native-index total: `0.164 ms`

Accuracy validation:

- Started a clean no-profile server:
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `BLOCK_SIZE=128`
    - `ENFORCE_EAGER=0`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - `VLLM_ROCM_DSV4_ATOM_HCA_NATIVE_INDICES=1`
- Ran unchanged `/app/atomdsv4/lmeval.sh`.
- Log: `/app/atomdsv4/lmeval_hca_native.log`
- Result:
    - flexible-extract exact match: `0.8908 ± 0.0086`
    - strict-match exact match: `0.8893 ± 0.0086`

Conclusion for HCA native-index path:

- The timing improvement is real, especially for T=32, where it removes about
  `1.5 ms` of HCA translation overhead in the profiled layer-0 wrapper.
- It is not correctness-valid. GSM8K drops far below the required `0.95 ± 0.01`.
- Do not enable or benchmark this path for performance claims.
- The likely issue is that the vLLM prebuilt HCA ragged slots are not equivalent
  to the ATOM HCA committed-head layout/read ordering expected by
  `sparse_attn_v4_paged_decode`, even after offsetting into the unified-KV HCA
  address range.

Twenty-eighth Diagnostic Slice: Fused HCA committed-head index fill
------------------------------------------------------------------

Implemented an opt-in decode index writer path behind:

- `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=1`

What changed:

- `write_v4_paged_decode_indices` still writes the SWA ring tail for SWA, CSA,
  and HCA index buffers.
- When the optional HCA arguments are provided, the same Triton launch also
  writes the HCA committed-head prefix:
    - source: `attn_metadata.block_table`
    - committed length: `atom_state.n_committed_hca_per_seq`
    - destination: `hca_indices[hca_indptr[t] : hca_indptr[t] + n_hca]`
    - value formula:
    `swa_pages + physical_block * hca_block_capacity + slot_offset`
- This is intended to be equivalent to the existing
  `_write_hca_compress_head` helper, but avoids the second HCA-head kernel
  launch in the ratio-128 decode path.
- The old helper remains the default path when
  `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=0`.

Why this is different from the failed native-index experiment:

- It does not reuse vLLM's top-k HCA ragged indices.
- It preserves ATOM's committed-HCA prefix semantics and only changes where the
  index fill is performed.
- Therefore it should not have the semantic mismatch that caused GSM8K to fall
  to about `0.89` with `VLLM_ROCM_DSV4_ATOM_HCA_NATIVE_INDICES=1`.

Validation performed so far:

- `python -m py_compile` passed for:
    - `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode_indices.py`
    - `vllm/models/deepseek_v4/amd/rocm.py`
- A small CPU reference fixture passed, verifying that the helper writes:
    - SWA tails at each ragged slice tail.
    - HCA committed heads before the SWA tail.
    - HCA values using the same block-table formula as `_write_hca_compress_head`.

Validation still required before enabling by default:

- Profile decode timing to confirm `translate_ms` drops without increasing
  `index_ms` enough to cancel the benefit.

Runtime validation:

- Runtime smoke:
    - `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=1`
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `BLOCK_SIZE=128`
    - `ENFORCE_EAGER=0`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - result: server started, graph capture completed, minimal `2+2` completion
    returned the correct first answer token, and no Triton/runtime error was
    observed.
- Full unchanged `/app/atomdsv4/lmeval.sh`:
    - server config:
        - `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=1`
        - `MAX_NUM_SEQS=256`
        - `MAX_NUM_BATCHED_TOKENS=8192`
        - `BLOCK_SIZE=128`
        - `ENFORCE_EAGER=0`
    - log: `/app/atomdsv4/lmeval_fused_hca_index.log`
    - result:
        - flexible-extract exact match: `0.9515 +/- 0.0059`
        - strict-match exact match: `0.9522 +/- 0.0059`
    - conclusion: accuracy passes the required GSM8K `0.95 +/- 0.01` band.
- C32 benchmark after a fresh server restart:
    - server config:
        - `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=1`
        - `MAX_NUM_SEQS=32`
        - `MAX_NUM_BATCHED_TOKENS=8192`
        - `BLOCK_SIZE=128`
        - `ENFORCE_EAGER=0`
    - benchmark log: `/app/atomdsv4/benchmark_fused_hca_index_c32.log`
    - successful requests: `320`
    - failed requests: `0`
    - benchmark duration: `387.06 s`
    - output throughput: `846.58 tok/s`
    - total throughput: `1696.46 tok/s`
    - mean TTFT: `913.70 ms`
    - mean TPOT: `36.93 ms`

Comparison with previous validated C32 runs:

- Default correct path before this change:
    - output throughput: `841.60 tok/s`
    - total throughput: about `1686 tok/s`
    - mean TPOT: `36.98 ms`
- Earlier vLLM sparse baseline:
    - output throughput: `852.30 tok/s`
- Fused HCA committed-head index fill:
    - output throughput: `846.58 tok/s`
    - total throughput: `1696.46 tok/s`
    - mean TPOT: `36.93 ms`

Conclusion:

- Fusing the HCA committed-head fill is correctness-valid and gives a small
  improvement over the immediately previous correct ATOM-index path
  (`+4.98 tok/s`, about `+0.6%` output throughput).
- It does not close the larger gap to the earlier vLLM sparse baseline or the
  ATOM recipe C32 target, so the remaining performance issue is not only the
  extra HCA-head kernel launch.
- The next likely bottleneck remains broader decode metadata/index preparation
  and the mismatch between vLLM's ragged/block-table flow and ATOM's persistent
  request-state/unified-KV layout.

Profile follow-up:

- Started a profiled C32 server with:
    - `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=0`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=20`
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `BLOCK_SIZE=128`
    - `ENFORCE_EAGER=0`
- Ran a short C32 random workload:
    - input length: `1024`
    - output length: `256`
    - warmups: `32`
    - prompts: `96`
    - log: `/app/atomdsv4/benchmark_fused_hca_index_profile_c32_short.log`
- Server log parsed from:
  `/app/atomdsv4/dsv4prographnomtp-aitermhc_nobreakablecudagraph.log`

Layer-0 HCA profile medians:

| Ratio | T  | Samples | Index ms | Translate ms | Kernel ms | Total ms |
| ----- | -- | ------- | -------- | ------------ | --------- | -------- |
| 128   | 32 | 8       | 2.673    | 0.005        | 2.898     | 5.548    |
| 128   | 24 | 8       | 1.008    | 0.004        | 0.085     | 1.102    |
| 128   | 16 | 8       | 0.081    | 0.003        | 0.074     | 0.165    |

Comparison to the previous default HCA writer profile:

| Path                 | T  | Index ms | Translate ms | Kernel ms | Total ms |
| -------------------- | -- | -------- | ------------ | --------- | -------- |
| Default HCA writer   | 32 | 2.111    | 1.502        | 2.902     | 6.514    |
| Fused HCA index fill | 32 | 2.673    | 0.005        | 2.898     | 5.548    |

Interpretation:

- The fused path does remove the separate HCA-head translation/fill segment:
  `translate_ms` drops from about `1.5 ms` to near zero for T=32.
- The index segment grows by about `0.56 ms` because the HCA-head fill now runs
  inside the decode-index kernel. Net layer-0 wrapper time still improves by
  about `0.97 ms` for the measured T=32 HCA path.
- End-to-end benchmark throughput only improved about `0.6%`, so this was not
  the dominant deployment bottleneck.
- Important caveat: with CUDA graphs enabled, these Python-side profile prints
  are observed during warmup/capture, not every graph replay. They remain useful
  for comparing wrapper construction work between paths, but should not be read
  as literal per-token replay overhead.
- The remaining work should target:
    - eliminating or reducing the decode-index kernel itself,
    - avoiding vLLM ragged/index conversion work where possible,
    - moving closer to ATOM's persistent request-state/unified-KV index layout
    instead of recreating compatible views from vLLM metadata.

Twenty-ninth Diagnostic Slice: Can the decode-index kernel be removed?
---------------------------------------------------------------------

Current sparse decode interface:

- vLLM port:
    - `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`
    - `_sparse_attn_v4_paged_decode_triton(q, unified_kv, kv_indices,
    kv_indptr, ...)`
- ATOM source:
    - `ATOM/atom/model_ops/v4_kernels/paged_decode.py`
    - same core contract: the attention kernel reads flat ragged
    `kv_indices[kv_indptr[t] : kv_indptr[t + 1]]`.

Implication:

- With the current ATOM sparse attention kernel contract, some producer must
  materialize flat `kv_indices` and `kv_indptr`.
- The fused-HCA change moved the HCA committed-head fill into the SWA-tail index
  writer, but it did not change the attention kernel contract.
- Therefore the remaining decode-index kernel cannot be removed purely inside
  `rocm.py` unless another part of vLLM/metadata already provides ATOM-native
  flat indices in the exact committed-HCA/SWA-tail order.

What ATOM's vLLM bridge does:

- `ATOM/atom/plugin/vllm/deepseek_v4_bridge.py` also builds per-forward V4
  decode indices before calling `sparse_attn_v4_paged_decode`.
- That bridge uses:
    - `write_v4_paged_decode_indices`
    - a separate HCA compress-tail writer in older code
- So the current vLLM integration is structurally consistent with ATOM's vLLM
  bridge: ATOM kernels still consume a flat ragged index view.

What would remove the remaining index writer:

1. Change the sparse decode kernel contract.
   - Inputs would become request-state oriented:
     - `state_slot_per_seq`
     - `batch_id_per_token`
     - `positions`
     - `n_committed_hca_per_seq`
     - `block_table`
     - `swa_pages`
     - `win_with_spec`
   - The attention kernel would compute SWA ring addresses and HCA committed
     addresses internally while iterating K.
   - This removes `kv_indices` materialization but makes the attention kernel
     more coupled to DSV4 ROCm layout and vLLM block tables.

2. Move ATOM-native flat index generation into vLLM metadata/scheduler output.
   - The model forward would receive already-populated ATOM-compatible
     `kv_indices_*` and `kv_indptr_*`.
   - This keeps the attention kernel unchanged but moves the work earlier.
   - It does not eliminate work unless it can be maintained incrementally across
     requests, graph captures, and scheduler steps.

3. Introduce a ROCm DSV4 unified cache/metadata spec.
   - The cache spec owns the persistent SWA ring and compressed-cache layout.
   - Backend metadata exposes ATOM-native decode descriptors.
   - CUDA keeps the existing path.
   - This is the cleanest match to the original goal, but it is a vLLM
     core/attention-backend integration, not a local modeling-file tweak.

Conclusion:

- We currently have the kernels needed to run the ATOM sparse attention path
  under vLLM and preserve accuracy.
- We do not yet have the structural vLLM metadata/cache integration needed to
  get the full ATOM benefit without per-forward index materialization.
- The next meaningful performance step is therefore not another small wrapper
  fusion; it is either:
    - a new request-state sparse decode kernel, or
    - a ROCm-only DSV4 cache/metadata path that maintains ATOM-compatible indices
    as persistent request state.

Cleanup performed:

- Removed the unused duplicate `_build_atom_decode_indptrs` helper from
  `vllm/models/deepseek_v4/amd/rocm.py`.
- Active decode indptr preparation now lives in
  `vllm/models/deepseek_v4/amd/model_state.py`.
- `python -m py_compile` passed for the edited ROCm files after cleanup.

Thirtieth Diagnostic Slice: Direct request-state HCA decode
-----------------------------------------------------------

Question:

- Could conversion logic and metadata preparation be slowing the HCA decode
  path more than the sparse attention kernel itself?

Prototype:

- Added an opt-in request-state HCA decode path behind
  `VLLM_ROCM_DSV4_ATOM_HCA_DIRECT_DECODE=1`.
- The direct path bypasses per-forward flat `kv_indices` materialization for
  ratio-128 HCA decode when `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`.
- The kernel computes the committed-HCA and SWA-ring addresses from:
    - `state_slot_per_seq`
    - `batch_id_per_token`
    - `positions`
    - `n_committed_hca_per_seq`
    - `block_table`
    - `swa_pages`
    - `win_with_spec`
- This keeps the vLLM scheduler active, but it avoids the flat ragged decode
  index writer for this narrow HCA path.

Validation:

- Full unchanged `lmeval.sh` passed with:
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `MAX_NUM_SEQS=256`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_ROCM_DSV4_ATOM_HCA_DIRECT_DECODE=1`
- GSM8K:
    - flexible-extract: `0.9507 +/- 0.0060`
    - strict-match: `0.9515 +/- 0.0059`

Deployment benchmark:

- Fresh server restart before benchmark.
- C32 random 1024/1024, `benchmarkvllm.sh` unchanged except result naming.
- Config:
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `VLLM_ROCM_DSV4_ATOM_HCA_DIRECT_DECODE=1`
- Result:
    - successful requests: `320`
    - failed requests: `0`
    - benchmark duration: `387.09 s`
    - output throughput: `846.52 tok/s`
    - total throughput: `1696.35 tok/s`
    - mean TTFT: `1097.14 ms`
    - mean TPOT: `36.76 ms`

Comparison:

| Path                         | Output tok/s | Total tok/s | Mean TPOT ms | Accuracy |
| ---------------------------- | ------------ | ----------- | ------------ | -------- |
| Fused HCA committed-head fill | 846.58       | 1696.46     | 36.93        | pass     |
| Direct request-state HCA      | 846.52       | 1696.35     | 36.76        | pass     |

Interpretation:

- Yes, the conversion/index-preparation path is measurable. In profiling, the
  direct request-state kernel removes the explicit HCA decode-index preparation
  segment and greatly reduces wrapper-side metadata work.
- However, the C32 deployment benchmark is effectively unchanged versus the
  fused flat-index path. The likely reason is that much of the removed work is
  not the dominant replay-time bottleneck under CUDA graphs, or the direct
  kernel's extra address arithmetic offsets the removed materialization work.
- This confirms that conversion and metadata preparation are real costs, but
  removing only this one HCA flat-index writer is not enough to move end-to-end
  C32 throughput toward ATOM's published number.
- The next useful analysis is to separate:
    - graph-capture/setup-only metadata work,
    - per-step CPU scheduler/metadata packing work,
    - graph-replayed GPU kernels,
    - address arithmetic inside the direct sparse attention kernel.

Thirty-first Diagnostic Slice: vLLM-owned unified KV allocation
---------------------------------------------------------------

Question:

- Can the practical split move from model-state side unified-KV allocation to a
  vLLM-owned KV-cache spec/allocation while still using vLLM scheduler state?

Implementation state:

- Added an opt-in vLLM cache spec path behind
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`.
- The spec adds a fixed per-layer SWA prefix before the compressed paged tail:
  `[SWA ring prefix][compressed cache blocks]`.
- `KVCacheTensor.fixed_prefix_size` keeps the SWA prefix out of block-count
  scaling.
- `_get_kv_cache_config_deepseek_v4` subtracts fixed prefixes from available
  memory before calculating `num_blocks`.
- `_reshape_kv_cache` now resolves the per-layer spec from
  `UniformTypeKVCacheSpecs` before computing:
    - `prefix_bytes`
    - page size
    - dtype
    - storage block size
  This matters for DSV4 because group-level specs can hide layer-specific ATOM
  prefix metadata.

Structural validation:

- Server starts with:
    - `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `MAX_NUM_SEQS=32`
- The model-state binder logs:
  `Bound ROCm DSV4 ATOM unified KV views from vLLM-owned KV storage`.
- Observed allocation:
    - `num_blocks=11053`
    - `swa_pages=4096`
    - GPU KV cache size: `801,182 tokens`
- Compared to model-state side allocation, the vLLM-owned homogeneous BF16
  unified tail greatly reduces available KV capacity:
    - side allocation run: about `2,352,349 tokens`
    - vLLM-owned BF16 tail run: about `801,182 tokens`

Functional validation:

- With ATOM attention enabled, a trivial completion starts but returns garbage:
  prompt `Question: What is 2+2? Answer:` did not produce `4`.
- With `VLLM_ROCM_DSV4_ATOM_RETURN_FALSE_AT_ENTRY=1`, forcing native attention
  fallback while keeping the vLLM-owned ATOM cache spec, graph capture fails:
  `ROCm Triton sparse decode expects uint8 fp8_ds_mla extra cache, got torch.bfloat16`.

Interpretation:

- The vLLM-owned allocation bridge exists and can bind views correctly enough to
  start the server.
- This early slice was not yet accuracy-valid, but that conclusion is
  superseded by later vLLM-owned BF16 unified-KV runs that passed GSM8K. The
  BF16 compressed tail is not itself a mismatch with ATOM's V4 modeling file;
  ATOM's main SWA/CSA/HCA sparse-attention `unified_kv` is also homogeneous
  BF16.
- The remaining native ROCm sparse paths still expect the existing
  `fp8_ds_mla` uint8 compressed cache. This matters because the current
  practical path must route compressed layers consistently to ATOM kernels when
  the vLLM-owned BF16 unified layout is active.
- Therefore, a true vLLM-owned unified KV path needs the ROCm DSV4
  attention/cache backend contract to keep native `fp8_ds_mla` paths from
  reading BF16 ATOM storage, while exposing the BF16 ATOM views needed by the
  local ATOM sparse-attention and compressor kernels.

Conclusion:

- V2 model runner and model-specific `ModelState` are sufficient for persistent
  request slots, SWA rings, compressor state rings, and side unified-KV buffers.
- They are not sufficient by themselves for true ATOM-style vLLM-owned unified
  KV. The missing piece is the attention/cache backend contract that consistently
  routes ROCm DSV4 compressed layers to the BF16 ATOM unified layout.

Thirty-second Diagnostic Slice: ATOM paged-prefill safety and metadata cost
---------------------------------------------------------------------------

Question:

- Is the ATOM paged-prefill path blocked by conversion/metadata overhead, or by
  a correctness/safety problem in mixed scheduler batches?

Implementation state:

- Cached several ROCm DSV4 ATOM environment switches at module import time so
  hot paths no longer repeatedly call `os.environ.get`.
- Forced the ATOM paged-prefill wrapper to use the Triton implementation with
  `ATOM_FORCE_ATTN_TRITON=1` to separate OPUS-wrapper behavior from index and
  scheduler integration behavior.
- Added `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED`, defaulting to `0`.
- With the default guard, pure prefill may use the ATOM paged-prefill path, but
  mixed decode+prefill batches fall back to the existing vLLM prefill path.

Validation:

- Pure smoke test with ATOM Triton paged prefill produced the expected answer
  for `Question: What is 2+2? Answer:`.
- Full `lmeval.sh` with forced Triton paged prefill and no mixed-batch guard
  crashed before scoring with a HIP illegal memory access.
- The crash happened during a scheduler step containing many decode requests
  plus one large prefill request. The Python stack surfaced near
  `write_v4_paged_prefill_indices`, but HIP errors are asynchronous, so the
  actual fault could be in the immediately preceding/following prefill index or
  attention sequence.
- Full `lmeval.sh` with the mixed-batch guard passed:
    - flexible `exact_match`: `0.9507 +/- 0.0060`
    - strict `exact_match`: `0.9515 +/- 0.0059`
    - log: `/app/atomdsv4/lmeval_atom_prefill_triton_guard.log`
- C32 benchmark with the same passing guarded configuration:
    - successful requests: `320`
    - failed requests: `0`
    - duration: `388.71 s`
    - output throughput: `843.00 tok/s`
    - total throughput: `1689.29 tok/s`
    - mean TTFT: `1150.96 ms`
    - mean TPOT: `36.86 ms`
    - log: `/app/atomdsv4/benchmark_atom_prefill_triton_guard_c32.log`

Comparison:

- Previous passing fused-HCA committed-head-fill C32 run:
    - output throughput: `846.58 tok/s`
    - total throughput: `1696.46 tok/s`
    - mean TTFT: `913.70 ms`
    - mean TPOT: `36.93 ms`
- Previous passing direct-HCA decode C32 run:
    - output throughput: `846.52 tok/s`
    - total throughput: `1696.35 tok/s`
    - mean TTFT: `1097.14 ms`
    - mean TPOT: `36.76 ms`
- Earlier native sparse baseline was about `852.30 tok/s` output throughput.
- The ATOM recipe C32 target remains `1145.71 tok/s` output throughput.

Interpretation:

- Conversion and metadata preparation can slow the end-to-end scheduler step,
  especially because the current port still builds some prefill state through
  CPU NumPy, host-to-device copies, GPU index-generation kernels, CSA pack, and
  contiguous tensor materialization.
- That overhead is not the same as the attention kernel being slow. It sits
  around the kernel and can reduce realized serving throughput even if the
  kernel body is faster in isolation.
- For the current ATOM paged-prefill port, the primary blocker is still
  correctness/safety under vLLM mixed decode+large-prefill batches. The guarded
  configuration is accuracy-valid, but it does not exercise full ATOM mixed
  prefill and therefore does not close the performance gap.
- The current measurement shows no C32 improvement from enabling pure guarded
  ATOM prefill on top of the existing passing decode path. The steady-state
  decode path still dominates the 1024/1024 C32 benchmark.

Next analysis ideas:

- Reproduce the unguarded mixed-prefill crash with HIP launch blocking or
  narrower per-layer enablement to identify whether the fault is in index write,
  CSA/HCA prefix packing, extend indices, or the paged-prefill kernel.
- Add lightweight timing around prefill metadata build, H2D copies, index-write
  kernels, CSA pack, `.contiguous()` materialization, and prefill attention to
  separate scheduler-side overhead from kernel execution time.
- Compare ATOM prefill only on all-prefill batches, mixed batches with one
  decode token, and mixed batches with large queued prefill to isolate the
  vLLM scheduler shape that diverges from ATOM/SGLang assumptions.

Thirty-third Diagnostic Slice: Mixed ATOM prefill profiling under S256
----------------------------------------------------------------------

Question:

- Can the mixed ATOM paged-prefill crash be reproduced with random serving
  workloads that match the high scheduler pressure of `lmeval.sh`?

Implementation state:

- Added opt-in ATOM prefill profiling in the ROCm DSV4 attention path:
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL_TRACE=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL_MIN_T=<tokens>`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL_MIN_TOKEN_OFFSET=<tokens>`
- The hook times:
    - CPU-side prefill indptr construction and H2D copies
    - ATOM prefill index-write kernels
    - CSA top-k packing
    - `.contiguous()` materialization of the extend KV slice
    - ATOM paged-prefill attention
    - output copy
    - SWA ring write
- The hook is fully disabled by default and does not change default execution.

Validation:

- Diagnostic server:
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `HIP_LAUNCH_BLOCKING=1`
    - `ATOM_FORCE_ATTN_TRITON=1`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`
- Random high-concurrency run, `input_len=1024`, `output_len=128`,
  `num_prompts=288`, `max_concurrency=256`:
    - successful requests: `288`
    - failed requests: `0`
    - output throughput: `942.97 tok/s`
    - total throughput: `8516.22 tok/s`
    - mean TTFT: `12240.24 ms`
    - mean TPOT: `151.47 ms`
    - log: `/app/atomdsv4/bench_atom_prefill_mixed_diag_s256_n288.log`
- Random high-concurrency run, `input_len=4096`, `output_len=64`,
  `num_prompts=288`, `max_concurrency=256`:
    - successful requests: `288`
    - failed requests: `0`
    - output throughput: `165.77 tok/s`
    - total throughput: `10785.19 tok/s`
    - mean TTFT: `48983.96 ms`
    - mean TPOT: `586.97 ms`
    - log: `/app/atomdsv4/bench_atom_prefill_mixed_diag_s256_i4096.log`
- No `Traceback`, HIP illegal access, or CUDA error signatures were found in
  the diagnostic server or benchmark logs.

Representative prefill timings:

- Mixed CSA layer, `layer=60`, `ratio=4`, `T=8191`, `token_offset=1`:
    - `extend_total=983424`
    - `prefix_csa_total=1046391`
    - `build_ms=0.219`
    - `index_ms=0.099`
    - `csa_pack_ms=0.097`
    - `kv_contig_ms=0.012`
    - `kernel_ms=0.746`
    - `output_ms=0.069`
    - `swa_write_ms=0.045`
    - `total_ms=1.288`
- Mixed HCA layer, `layer=59`, `ratio=128`, `T=8191`, `token_offset=1`:
    - `extend_total=983424`
    - `prefix_hca_total=64533`
    - `build_ms=0.208` to `0.212`
    - `index_ms=0.080` to `0.086`
    - `kernel_ms=0.469` to `0.480`
    - `total_ms=0.883` to `0.906`

Interpretation:

- Random serving workloads at S256 did not reproduce the previous full
  `lmeval.sh` unguarded mixed-prefill crash, even with 4096-token prompts and
  HIP launch blocking.
- The failure is therefore likely tied to a more specific lm-eval scheduler
  shape, request-length distribution, cancellation/finish pattern, or metadata
  corner case rather than generic high-concurrency mixed ATOM prefill.
- The timing data supports the earlier suspicion that conversion/metadata work
  is material. For the large CSA prefill slice above, the attention kernel is
  about `0.746 ms` of `1.288 ms`; the rest is metadata/index/pack/copy/write
  work around the kernel.
- This does not yet prove ATOM mixed prefill is accuracy-safe for production.
  The guard remains required for the default passing configuration because the
  full unchanged `lmeval.sh` previously crashed without it.

Next analysis ideas:

- Re-run full `lmeval.sh` with the prefill profiling hooks and
  `HIP_LAUNCH_BLOCKING=1`, preferably with narrower layer tracing after the
  first failure shape is identified.
- Use the focused profile filters for `token_offset > 0`, minimum `T`, and
  selected layers so the full lm-eval diagnostic log is manageable.
- Compare the exact failing lm-eval batch shape against the successful random
  S256 shapes: number of running requests, token offset, `T`, per-request prompt
  lengths, block-table span, and prefix totals.

Thirty-fourth Diagnostic Slice: Production unguarded prefill needs ordering
---------------------------------------------------------------------------

Question:

- Does unguarded mixed ATOM paged-prefill pass unchanged `lmeval.sh` in the
  real no-eager production shape, without `HIP_LAUNCH_BLOCKING` or profiling
  synchronization?

Implementation state:

- Added a default-off diagnostic fence:
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC=1`
- The fence calls `torch.cuda.synchronize()` after the ATOM prefill sub-stages
  that the profiler already serialized:
    - ATOM prefill index writes
    - CSA translate/pack
    - extend KV materialization
    - ATOM paged-prefill attention
    - output copy
    - SWA ring write
- This is intentionally a diagnostic mechanism. It is too coarse for the final
  performance path and should be replaced by precise stream/event dependencies.

Validation:

- Production-shaped unguarded mixed ATOM prefill without the fence:
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `ATOM_FORCE_ATTN_TRITON=1`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`
    - no `HIP_LAUNCH_BLOCKING`
    - no profiler flags
    - result: failed during unchanged `lmeval.sh`
    - first visible error:
        - `[aiter] Error in moe_sorting: CUDA error: an illegal memory access was encountered`
        - `topk_ids.shape=torch.Size([8192, 6])`
        - `max_num_tokens_padded=98298`
    - server log:
        - `/app/atomdsv4/server_lmeval_unguarded_prefill_prod.log`
- Production-shaped unguarded mixed ATOM prefill with the diagnostic fence:
    - same configuration, plus `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC=1`
    - unchanged `lmeval.sh` completed
    - GSM8K flexible exact match: `0.9515 +/- 0.0059`
    - GSM8K strict exact match: `0.9522 +/- 0.0059`
    - lm-eval log:
        - `/app/atomdsv4/lmeval_unguarded_prefill_sync.log`
    - server log:
        - `/app/atomdsv4/server_lmeval_unguarded_prefill_sync.log`
- Fresh C32 benchmark with the diagnostic fence:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `ATOM_FORCE_ATTN_TRITON=1`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC=1`
    - successful requests: `320`
    - failed requests: `0`
    - output throughput: `843.43 tok/s`
    - total throughput: `1690.15 tok/s`
    - mean TTFT: `1114.16 ms`
    - mean TPOT: `36.88 ms`
    - benchmark log:
        - `/app/atomdsv4/benchmark_unguarded_prefill_sync_c32.log`
    - server log:
        - `/app/atomdsv4/server_benchmark_unguarded_prefill_sync_c32.log`

Interpretation:

- The production failure does not prove that `aiter` MoE sorting is the root
  cause. The HIP error is asynchronous, and the same failure wave also reports
  errors in the sparse attention indexer path. The illegal access may have been
  caused by an earlier ATOM prefill/index/pack operation and surfaced later.
- The key new signal is that coarse synchronization around ATOM prefill makes
  the same S256 no-eager lm-eval shape pass accuracy. That strongly suggests a
  missing ordering dependency between the ATOM prefill staging kernels, the
  attention kernel, state writes, and later vLLM work.
- The C32 throughput with the fence, `843.43 tok/s`, is in the same band as the
  prior guarded ATOM prefill run (`843.00 tok/s`) and below the best measured
  run in this branch (`846.58 tok/s` with fused HCA index, and about `852 tok/s`
  for the native baseline). So the fence is not a performance feature.
- The practical conclusion is:
    - guarded mixed prefill remains the safe default for now;
    - unguarded mixed prefill is accuracy-capable only when serialized;
    - the next useful implementation step is replacing global sync with narrow
    event/stream waits around the exact producer-consumer boundaries.

Next analysis ideas:

- Split the diagnostic fence into per-stage flags to identify the minimal
  required boundary: after index write, after CSA pack, after attention, after
  output copy, or after SWA write.
- Check whether any ATOM/aiter prefill or pack kernel launches onto a non-current
  stream. If so, use explicit events instead of relying on default-stream order.
- Compare graph replay and eager execution ordering for the ATOM prefill path.
  The failure happened with no `--enforce-eager`, while profiling and
  `HIP_LAUNCH_BLOCKING` effectively serialized the path.

### Thirty-fifth Diagnostic Slice: Post-attention stream fence is accuracy-valid

Question:

- Is the full diagnostic device sync required, or is the missing dependency only
  around the lifetime of the prefill attention scratch/metadata buffers?

Code change:

- Replaced the single all-stage diagnostic switch with optional named stages:
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC=1` still synchronizes all stages.
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC_STAGES=<comma-separated stages>`
    synchronizes only selected stages.
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC_KIND=stream|device` chooses
    `torch.cuda.current_stream().synchronize()` or `torch.cuda.synchronize()`.
- Available stage labels in the ATOM mixed prefill path:
    - `post_index`
    - `post_pack`
    - `pre_attn`
    - `post_attn`
    - `post_output`
    - `post_swa`

Validation:

- Production-shaped unguarded mixed ATOM prefill with only a post-attention
  stream-local fence:
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `ATOM_FORCE_ATTN_TRITON=1`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC=0`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC_STAGES=post_attn`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC_KIND=stream`
    - unchanged `lmeval.sh` completed
    - GSM8K flexible exact match: `0.9530 +/- 0.0058`
    - GSM8K strict exact match: `0.9538 +/- 0.0058`
    - lm-eval log:
        - `/app/atomdsv4/lmeval_unguarded_prefill_postattn_stream.log`
    - server log:
        - `/app/atomdsv4/server_lmeval_unguarded_prefill_postattn_stream.log`
- Fresh C32 benchmark with the same post-attention stream-local fence:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `ATOM_FORCE_ATTN_TRITON=1`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC=0`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC_STAGES=post_attn`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC_KIND=stream`
    - successful requests: `320`
    - failed requests: `0`
    - output throughput: `844.12 tok/s`
    - total throughput: `1691.54 tok/s`
    - mean TTFT: `1047.86 ms`
    - mean TPOT: `36.91 ms`
    - benchmark log:
        - `/app/atomdsv4/benchmark_unguarded_prefill_postattn_stream_c32.log`
    - server log:
        - `/app/atomdsv4/server_benchmark_unguarded_prefill_postattn_stream_c32.log`

Interpretation:

- The all-stage device sync is not required for accuracy in this configuration.
  A stream-local wait immediately after `sparse_attn_v4_paged_prefill` is enough
  to make the production no-eager, V2-runner S256 lm-eval shape pass.
- This points to a scratch/metadata lifetime dependency around the prefill
  attention call. The global prefill buffers and translated/packed index tensors
  are reused by subsequent layers; if the attention kernel is still reading them,
  the next layer can overwrite them.
- The throughput remains in the same band as guarded/native runs:
    - guarded ATOM prefill: `843.00 tok/s`
    - all-stage diagnostic sync: `843.43 tok/s`
    - post-attention stream sync: `844.12 tok/s`
    - fused HCA index best in this branch: `846.58 tok/s`
    - native baseline observed earlier: about `852 tok/s`
- So the post-attention stream fence is a correctness improvement over the
  coarse sync, but it is not a performance breakthrough.

Metadata/conversion overhead hypothesis:

- The current ATOM mixed prefill path pays nontrivial setup cost outside the
  sparse attention math kernel:
    - sparse prefill index metadata build,
    - CSA translate/pack,
    - `kv_actual = kv_full[...].contiguous()`,
    - output copy,
    - SWA ring write,
    - and now a stream-local wait after the attention kernel.
- In the earlier profiled layer-60 S256 prefill sample, the attention kernel was
  about `1.597 ms` while the measured end-to-end prefill attention block was
  about `2.163 ms`. That means roughly one quarter of that measured block was
  preparation/copy/synchronization overhead even before considering other
  per-layer scheduling and metadata costs.
- This makes conversion and metadata preparation a credible reason that the
  faster ATOM-style prefill kernel path does not show higher end-to-end C32
  throughput in vLLM yet.

Next analysis ideas:

- Add per-stage timing under the same deployment flags for:
    - index metadata build,
    - CSA translate/pack,
    - `kv_actual.contiguous()`,
    - prefill attention kernel,
    - output copy,
    - SWA write.
- Replace the post-attention stream synchronize with an explicit event on the
  stream used by the ATOM prefill attention kernel once the actual producer
  stream is confirmed.
- Reduce or remove `kv_actual.contiguous()` by making the paged-prefill kernel
  consume the existing layout, or by writing compressor output directly into the
  layout expected by the kernel.
- Move reusable metadata work into persistent request-state/ring structures
  instead of rebuilding and repacking it for every layer.

### Thirty-sixth Diagnostic Slice: Component readiness for full ATOM benefit

Current component status:

- Present and usable without GPU worker changes:
    - ROCm DSV4 model-specific `ModelState`.
    - Persistent per-request SWA/compressor state buffers.
    - Per-forward request-slot metadata attached to vLLM attention metadata.
    - ATOM-style compress-plan buffers.
    - Side-allocated ATOM homogeneous `unified_kv` buffers.
    - ATOM decode/prefill index builders and attention kernels.
    - ATOM SWA ring writes.
    - ATOM fused compressor path for the side-allocated unified KV path.
    - vLLM weight loading remains in use.
- Present and accuracy-validated, but not yet a final performance path:
    - vLLM-owned ATOM unified KV via `DeepseekV4AtomMLAAttentionSpec`.
    - DeepSeek-V4-specific KV allocation that reserves a fixed SWA prefix before
    the scalable compressed tail.
    - GPU reshape logic that skips the fixed SWA prefix when binding the normal
    compressed tail view.
    - Model-state binding that can reconstruct ATOM `unified_kv`,
    `atom_swa_kv`, and compressed tail views from the same vLLM-owned storage.
- Missing or unresolved for full ATOM benefit:
    - A stricter ROCm DSV4 backend contract that prevents native `fp8_ds_mla`
    sparse paths from reading BF16 ATOM unified storage.
    - A CSA indexer integration cleanup that reduces the generic vLLM
    sparse-indexer metadata/wrapper overhead now that the FP8 cache path,
    scale view, gather, logits, and top-k kernel family have been audited
    against ATOM's modeling-file behavior.
    - A way to avoid or greatly reduce per-layer metadata/conversion overhead:
    index build, CSA translate/pack, `kv_actual.contiguous()`, output copy, and
    ordering fences.
    - A precise event dependency replacing the post-attention stream
    synchronization used to make unguarded mixed prefill accuracy-valid.
    - More C32 performance proof after reducing the request-state/indexer/overlap
    gap. Accuracy for the vLLM-owned BF16 unified-KV path has already passed
    GSM8K in later validation.

Code hardening added after this audit:

- `DeepseekV4AtomMLAAttentionSpec.max_memory_usage_bytes()` now includes the
  fixed SWA prefix. Before this, the spec page size described only the scalable
  compressed tail, so generic memory checks could undercount the experimental
  vLLM-owned unified layout.
- `DeepseekV4AtomMLAAttentionSpec.merge()` now preserves ATOM SWA prefix fields
  if the spec ever goes through a merge path.
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1` no longer silently falls back to
  the side-allocated ATOM unified KV once `kv_cache_config.num_blocks` exists.
  If binding from vLLM KV storage fails, the model state raises a clear error.
- The vLLM-owned binder now checks raw storage size against the ATOM view size
  and logs the bound layout: ratio counts, block count, SWA pages,
  `win_with_spec`, head dimension, and dtype.

Smoke validation after hardening:

- Command shape:
    - `MAX_NUM_SEQS=4`
    - `MAX_NUM_BATCHED_TOKENS=512`
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
    - `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_PREFILL=1`
    - V2 model runner and breakable CUDA graph enabled by launch defaults.
- Server log:
    - `/app/atomdsv4/server_from_vllm_bind_smoke.log`
- Startup result:
    - server reached readiness;
    - graph capture completed;
    - no runtime errors were logged;
    - all workers logged:
        - `Bound ROCm DSV4 ATOM unified KV views from vLLM-owned KV storage`
        - `ratio_counts={128: 31, 4: 30}`
        - `num_blocks=15489`
        - `swa_pages=512`
        - `win_with_spec=128`
        - `head_dim=512`
        - `dtype=torch.bfloat16`
- Tiny request result:
    - prompt: `What is 2+2? Answer briefly.`
    - request completed, but output was nonsensical.
    - This is a silent correctness failure, not a capacity or crash failure.
- Follow-up change:
    - Added a generic optional `post_bind_kv_cache(kv_cache)` hook in
    `vllm/v1/worker/utils.py`.
    - Implemented the hook on `DeepseekV4Attention` for the ROCm
    `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1` path.
    - The hook derives `atom_unified_kv`, `atom_swa_kv`, and
    `atom_compressed_kv_cache` immediately when vLLM binds the layer KV cache,
    before graph capture replays can depend on those attributes.
- Follow-up smoke:
    - server log: `/app/atomdsv4/server_from_vllm_prebind_smoke.log`
    - startup still succeeded;
    - graph capture still completed;
    - tiny request still returned nonsensical text.

Interpretation:

- We have enough components to continue integrating ATOM attention/compressor
  logic inside vLLM's scheduler without GPU worker changes for request state.
- We do not yet have all components needed to claim the full ATOM kernel benefit
  end to end. The remaining blocker is not simply adding another op call; it is
  the unified KV layout and metadata lifetime/overhead contract.
- This predated the later accuracy-valid vLLM-owned run. The bridge is now
  structurally reachable and accuracy-valid with BF16 homogeneous ATOM
  `unified_kv`. The remaining semantic-layout issue is narrower: native
  vLLM/ROCm sparse paths still carry global `fp8_ds_mla` assumptions, so the
  ATOM mode must consistently avoid those native compressed-layer paths or add
  an explicit backend contract for the BF16 ATOM layout.
- The follow-up pre-bind smoke makes graph-capture timing less likely as the
  primary cause of the bad output. The next likely causes are:
    - native prefill/routing still reading or writing a different semantic layout
    than ATOM decode expects;
    - BF16 homogeneous main cache matching ATOM's model file but not matching
    vLLM's global fp8 assumptions in every native fallback path;
    - a remaining mismatch between vLLM block-table metadata and ATOM's unified
    cache indices for the vLLM-owned allocation path.
- The practical split still holds:
    - no GPU worker changes are needed for persistent per-request state;
    - vLLM core/attention changes are needed for ROCm-only DSV4 unified KV cache
    layout, spec accounting, backend metadata, and kernels reading that layout;
    - CUDA should remain untouched because the custom spec is emitted only by the
    ROCm DSV4 path when the ATOM unified-KV experiment is enabled.

### vLLM-Owned Unified KV Accuracy And C32 Benchmark

The latest slice tested whether ATOM-shaped unified KV can be allocated through
vLLM's existing KV-cache allocator rather than side allocation.

Code changes:

- `DeepseekV4AtomMLAAttentionSpec` adds a fixed SWA prefix before the compressed
  tail and exposes `fixed_prefix_size` on `KVCacheTensor`.
- The DSV4 KV-cache allocator subtracts the fixed prefix when computing usable
  token capacity.
- `GPUModelRunner._reshape_kv_cache_tensors` skips fixed prefixes before
  reshaping the compressed tail for normal vLLM cache users.
- `DeepseekV4RocmAtomModelState` binds ATOM unified KV views from the
  vLLM-owned allocation when
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`.
- `post_bind_kv_cache` gives the model a generic hook after vLLM has allocated
  and bound KV cache tensors.

Important runtime details:

- The vLLM-owned ATOM layout currently uses BF16 unified rows, matching the
  active local ATOM sparse-attention path.
- Native ROCm sparse decode still expects the split `uint8 fp8_ds_mla` cache.
  Therefore, with `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`, mixed
  prefill+decode batches must keep using ATOM decode if an ATOM prefill path was
  used. Falling back to native decode is a writer/reader layout mismatch.
- A guard was added so native ROCm sparse decode raises immediately if it would
  consume the BF16 ATOM unified layout.

Validation:

- `MAX_NUM_SEQS=64`, `MAX_MODEL_LEN=8192`,
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`,
  `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`,
  no `--enforce-eager`, V2 runner:
  `lmeval.sh` passed GSM8K with flexible exact match `0.9560` and strict exact
  match `0.9568`.
- `MAX_NUM_SEQS=256` was not viable with this BF16 vLLM-owned layout:
  vLLM rejected the configuration before serving because the KV allocation
  exceeded available memory.
- `MAX_NUM_SEQS=32`, `MAX_MODEL_LEN=2048`,
  `GPU_MEMORY_UTILIZATION=0.9` served but rejected benchmark requests with HTTP
  400, because the chat-formatted 1024-token input plus 1024 requested output
  exceeded the 2048 model-length limit.
- `MAX_NUM_SEQS=32`, `MAX_MODEL_LEN=2304`,
  `GPU_MEMORY_UTILIZATION=0.9`, V2 runner, no `--enforce-eager`,
  `benchmarkvllm.sh` C32 completed:
    - successful requests: `320 / 320`
    - output throughput: `850.82 tok/s`
    - total throughput: `1704.97 tok/s`
    - mean TPOT: `36.58 ms`
    - mean TTFT: `1086.10 ms`
    - server-reported KV capacity: `22,656 tokens`
    - server-reported maximum full-length concurrency:
    `9.83x` for 2304-token requests

Comparison to known C32 runs:

- Side-allocation ATOM prefill/control runs were around `843-847 tok/s` output
  throughput at C32.
- The vLLM-owned BF16 unified-KV run reached `850.82 tok/s`, so ownership by
  vLLM did not materially improve throughput.
- ATOM's documented FP8 TP8 C32 target is `1145.71 tok/s` output throughput and
  `26.90 ms` mean TPOT. The current vLLM-owned BF16 unified-KV path is still
  about `25.7%` lower in output throughput and about `36.0%` higher in TPOT.

Current interpretation:

- Moving the ATOM unified cache into vLLM's KV allocator proves the scheduler
  integration is possible and can be accurate.
- It does not yet recover ATOM's performance because the active path still pays
  for conversion/index/metadata work and because the scheduling/state contract
  is still vLLM-style block-table preparation around ATOM kernels.
- The low server-reported `GPU KV cache usage` during C32 does not mean the
  allocation is cheap. It means the allocator reserved a very large BF16 cache
  pool up front, while the active workload used only a small percentage of those
  pages.

### ATOM FP8 KV Cache Clarification

Inspection of ATOM's current V4 source shows that the sparse-attention main
unified KV path is BF16, even when the recipe/server argument says
`--kv_cache_dtype fp8`:

- `atom/model_ops/attentions/deepseek_v4_attn.py` hard-codes
  `_swa_dtype = torch.bfloat16` and `_classical_dtype = torch.bfloat16`.
- `allocate_per_req_cache()` allocates each layer's `unified_kv` as a single
  BF16 tensor with layout:
  `[num_slots * win_with_spec + compressed_pages, head_dim]`.
- `build_kv_cache_tensor()` binds:
    - `attn.unified_kv` to that full BF16 tensor.
    - `attn.swa_kv` to the BF16 SWA prefix.
    - CSA/HCA main `compressor.kv_cache` to BF16 views into the compressed tail.
- `DeepseekV4Attention.forward_impl()` passes that same `self.unified_kv` to
  `sparse_attn_v4_paged_decode()` and `sparse_attn_v4_paged_prefill()`.
- The FP8 cache in the V4 path is the CSA indexer cache
  `v4_csa_idx_kv`, with a strided fp32 `cache_scale` view. That cache is used
  for index scoring/gather, not for the main sparse-attention paged decode or
  prefill kernels.

Implication for this vLLM integration:

- The current vLLM-owned BF16 ATOM unified-KV path matches ATOM's modeling-file
  contract for the main sparse-attention kernels.
- The remaining gap to ATOM's recipe C32 number should not be attributed to a
  BF16-vs-FP8 mismatch in the main `unified_kv` sparse-attention input.
- More likely causes are still the structural ones: vLLM block-table metadata
  preparation, index conversion/writes, lack of ATOM's full stream-overlap
  schedule, and any remaining differences in the CSA indexer/compressor path.

### ATOM Kernel Coverage Snapshot

Current ROCm DSV4 integration status against ATOM's modeling-file operation
sequence:

| Area | Current vLLM status | Notes |
| --- | --- | --- |
| MHC pre/post/fused post-pre | Gated on ROCm | Standalone `mhc_pre` and `mhc_post` exist in installed aiter but are accuracy-unsafe in the current vLLM MHC call contract, so they remain opt-in through `VLLM_ROCM_DSV4_USE_AITER_MHC=1`. Installed aiter does not expose `mhc_fused_post_pre`. |
| Q RMSNorm + Q quant | Enabled where the quant contract matches | `ATOM_USE_FUSED_Q_NORM_QUANT=1` routes through aiter fused RMSNorm/group-quant when the vLLM quant key is compatible. |
| Q/K RMSNorm + RoPE | Enabled for ROCm ATOM attention path | Calls local port of ATOM `qk_norm_rope_maybe_quant(..., quant_q=False, quant_k=False)`, matching ATOM's current sparse-attention BF16 consumer. |
| SWA ring write | Enabled for ATOM path | Uses local port of ATOM `swa_write` against per-request `atom_swa_kv` views. |
| Main CSA/HCA compressor | Enabled for ATOM path | Uses local port of ATOM `fused_compress_attn` before `update_compressor_states`, preserving ATOM's read-before-update ordering. |
| Main sparse paged decode | Enabled for ATOM path | Uses local port of ATOM `sparse_attn_v4_paged_decode` over BF16 `atom_unified_kv`. |
| Main sparse paged prefill | Enabled for ATOM path | Uses local port of ATOM `sparse_attn_v4_paged_prefill`, but vLLM still prepares prefix/extend metadata around it. |
| vLLM-owned unified KV allocation | Enabled behind `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1` | Allocates per-layer fixed SWA prefix plus compressed BF16 tail through vLLM KV cache specs and binds ATOM views in model state. |
| CSA indexer FP8 cache | Kernel family present, dispatch shape differs | The vLLM ROCm `SparseAttnIndexer` path calls the same classes of kernels: FP8 cache gather, `fp8_mqa_logits`/paged MQA logits, and aiter/vLLM top-k. It still uses vLLM `DeepseekV32IndexerMetadata`, chunking, and the generic custom-op wrapper rather than ATOM's model-local `Indexer.forward_batched()` / `indexer_score_topk()` metadata contract. |
| Metadata/request-state rings | Partially covered | Persistent ModelState buffers provide request slots, committed counts, decode/prefill indices, and pinned host mirrors, but vLLM still derives them from block-table scheduler state each step. |
| Aux-stream overlap | Not fully restored | Earlier aux stream changes were reverted. Current ROCm path is primarily sequential except for vLLM's existing graph/eager scheduling behavior. |

The practical conclusion is that the main ATOM attention/compressor kernels are
present and active in the validated path, but the full ATOM performance model is
not yet present. The missing pieces are not another main attention kernel; they
are the request-state/indexer/scheduling/overlap structure around the kernels.

### CSA Indexer Audit

ATOM's modeling-file indexer sequence is:

1. The indexer-inner `Compressor` writes the CSA indexer cache through
   `fused_compress_attn(..., quant=True)`, storing FP8 K rows plus a strided
   fp32 scale view in the same cache allocation.
2. `Indexer.forward_batched()` computes replicated indexer Q, applies RoPE and
   rotation, quantizes Q to FP8, folds the Q scale into the per-head weights,
   and dispatches `torch.ops.aiter.indexer_score_topk`.
3. `indexer_score_topk()` reads model-level V4 metadata:
   - decode: `deepgemm_fp8_paged_mqa_logits` over the paged FP8 indexer cache,
     then `top_k_per_row_decode`;
   - prefill: `cp_gather_indexer_k_quant_cache`, `fp8_mqa_logits`, then
     `top_k_per_row_prefill`, followed by global-to-seq-local conversion.

The current vLLM ROCm path is close at the kernel level:

- `DeepseekV4Indexer` owns an FP8 `DeepseekV4IndexerCache` with `head_dim + 4`
  bytes per row, matching the FP8 K plus fp32 scale layout.
- The indexer-inner `DeepseekCompressor` writes that cache and skips a second
  insert (`skip_k_cache_insert=True`).
- `fused_indexer_q_rope_quant()` performs the fused indexer-Q RoPE/FP8
  quantization and folds Q scale into weights for the FP8 path.
- `SparseAttnIndexer.forward_hip()` dispatches to
  `torch.ops.vllm.rocm_aiter_sparse_attn_indexer`.
- `rocm_aiter_sparse_attn_indexer()` calls:
    - `cp_gather_indexer_k_quant_cache` or the local Triton equivalent for
    prefill gather;
    - aiter `fp8_mqa_logits` for prefill when available;
    - aiter `deepgemm_fp8_paged_mqa_logits` for decode when available;
    - aiter `top_k_per_row_prefill` / `top_k_per_row_decode` when available.

The remaining difference is therefore not "missing aiter indexer ops". It is the
wrapper and metadata contract:

- ATOM builds V4 indexer metadata once per forward and gives the indexer module
  direct access to committed counts, sequence bases, and per-token bounds.
- vLLM still enters through the generic sparse-indexer custom op and
  `DeepseekV32IndexerMetadata`, including chunk metadata, workspace-manager
  allocation, optional packing/unpacking, and generic decode/prefill routing.
- This keeps correctness but leaves conversion and dispatch overhead around the
  indexer kernels. It also means the current code has two metadata worlds:
  ATOM ModelState metadata for the main unified-KV sparse attention, and vLLM
  sparse-indexer metadata for CSA top-k.

Next integration target:

- Either route ROCm DSV4 CSA indexer scoring through an ATOM-style
  model-local path that consumes `DeepseekV4RocmAtomStateMetadata`, or hoist the
  vLLM sparse-indexer metadata build so it produces the same pre-derived fields
  ATOM's `indexer_score_topk()` expects.
- Benchmark this specifically with indexer profiling before and after. The
  expected win is from less metadata work and fewer generic wrapper steps, not
  from swapping to a different FP8 logits/top-k kernel family.

Implementation step added:

- `DeepseekV4Indexer` now has an opt-in ROCm decode fast path behind
  `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`.
- The path is intentionally narrow:
    - ROCm only;
    - FP8 indexer cache only, not FP4;
    - pure decode only;
    - one token per sequence;
    - no packed/unpacked padding case;
    - requires `DeepseekV4RocmAtomStateMetadata` from the parent SWA metadata.
- It still reuses the same aiter-backed vLLM wrappers for
  `rocm_fp8_paged_mqa_logits` and `_top_k_per_row_decode`, but bypasses the
  generic `SparseAttnIndexer` custom-op body and uses ATOM ModelState's
  `n_committed_csa_per_seq` directly for decode bounds.
- It still reuses vLLM's indexer block table. This is deliberate: the first
  step removes generic wrapper/metadata work without changing cache allocation
  ownership or the scheduler's block table contract.

Validation needed before enabling by default:

- Run GSM8K with `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`.
- Run C32 benchmark with the same flag and compare against the current
  vLLM-owned BF16 unified-KV baseline of `853.60 tok/s`.
- If it is accurate and faster, extend the same idea to MTP/padded decode and
  then decide whether prefill should get an ATOM-local path or continue using
  the generic chunked sparse-indexer wrapper.

### Conversion And Metadata Cost Hypothesis

Conversion logic and metadata preparation can reduce end-to-end throughput even
when the sparse-attention kernel itself is fast.

Costs observed or still present in the current ROCm ATOM path:

- CPU-side plan construction for prefill/decode ragged indices and indptrs.
- CPU-to-GPU copies for indptr/plan tensors.
- Triton kernels for index writing, CSA top-k translation, compressor state
  update, and paged prefill/decode index generation.
- `.contiguous()` materialization of current dense `kv` slices before ATOM
  prefill.
- First-inference JIT latency for conversion/index kernels. The benchmark
  server logged JIT warnings for `_cp_gather_indexer_quant_cache_kernel`,
  `_gluon_fp8_mqa_logits_kernel`, `_update_compressor_states_kernel`,
  `_v4_paged_prefill_indices_kernel`, `_csa_translate_pack_kernel`, and a GEMM
  kernel during the first benchmark request.

This distinction matters:

- These costs do not make the device time of the final sparse-attention kernel
  slower.
- They do make the full vLLM scheduler step slower because the step includes
  metadata build, state updates, index conversion, temporary tensor creation,
  and synchronization/JIT effects around the kernel call.

Analysis ideas:

- Use `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1` to measure metadata-builder time
  separately from the attention kernel. The current ModelState hook logs
  `super`, `unified`, `plans`, `state`, `attach`, and `total` timings for the
  first 16 calls and then every 128th call.
- Use `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL=1` for prefill segment timings:
  `build_ms`, `index_ms`, `csa_pack_ms`, `kv_contig_ms`, `kernel_ms`,
  `output_ms`, and `swa_write_ms`.
- Warm all conversion/index shapes before serving benchmark traffic so JIT
  spikes do not contaminate TTFT.
- Reduce CPU preparation by caching environment flags, preallocating all
  metadata buffers, and moving repeated offset construction into persistent
  GPU-side builders where possible.
- For the real ATOM target, preserve the BF16 main unified-KV contract used by
  the V4 modeling file, and separately verify the FP8 CSA indexer cache path
  and scale view. The recipe's `fp8 KV cache` label should not be interpreted
  as FP8 main sparse-attention rows unless the ATOM source changes.

Latest C32 prefill attribution run:

- Server configuration matched the current fastest validated path:
  `VLLM_USE_V2_MODEL_RUNNER=1`, no `--enforce-eager`, block size `128`,
  `MAX_NUM_SEQS=32`, `MAX_MODEL_LEN=2304`,
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`,
  `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`, and
  `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`.
- Profiling used `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL=1`,
  `VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=-1`,
  `VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=1`, and a random C32
  `input_len=128`, `output_len=1` burst.
- Parsed rows from `/app/atomdsv4/server_prefill_profile_c32.log`: `1309`.

Large prefill chunk (`T=3696`) summary:

- CSA layers (`ratio=4`, 163 rows):
    - `total_ms`: mean `0.758`, p50 `0.767`
    - `kernel_ms`: mean `0.280`, p50 `0.277`
    - `build_ms`: mean `0.249`, p50 `0.249`
    - `index_ms`: mean `0.076`, p50 `0.074`
    - `csa_pack_ms`: mean `0.061`, p50 `0.060`
    - `kv_contig_ms`: mean `0.006`, p50 `0.006`
    - overhead excluding the attention kernel: mean `0.478`, p50 `0.480`
- HCA layers (`ratio=128`, 177 rows):
    - `total_ms`: mean `0.715`, p50 `0.660`
    - `kernel_ms`: mean `0.278`, p50 `0.269`
    - `build_ms`: mean `0.230`, p50 `0.232`
    - `index_ms`: mean `0.069`, p50 `0.065`
    - `kv_contig_ms`: mean `0.007`, p50 `0.007`
    - overhead excluding the attention kernel: mean `0.437`, p50 `0.390`

Smaller chunk (`T=264`) summary:

- CSA layers: `total_ms` mean `0.511`, `kernel_ms` mean `0.100`,
  overhead excluding the attention kernel mean `0.411`.
- HCA layers: `total_ms` mean `0.431`, `kernel_ms` mean `0.098`,
  overhead excluding the attention kernel mean `0.333`.

Interpretation:

- vLLM scheduler metadata preparation is not the main limiter in the validated
  C32 decode path; the previous metadata profile measured about `0.30 ms` for
  ATOM `plans + state`, or `1.27 ms` including inherited vLLM metadata.
- Per-layer ATOM prefill wrapper preparation is significant. For the large C32
  chunk, non-kernel work is about `0.44-0.48 ms` per layer while the attention
  kernel is about `0.28 ms`.
- `.contiguous()` materialization of the dense KV slice is negligible in this
  run (`~0.006-0.007 ms` per layer).
- The biggest prefill-side candidates are repeated `build_ms`, per-layer index
  generation, CSA translation packing, SWA writes, and missing overlap/fusion.
  Reusing or fusing HCA prefix indices across HCA layers looks more promising
  than optimizing dense KV contiguity.

### Prefill Index Reuse

A follow-up implementation added a per-forward
`DeepseekV4RocmAtomPrefillCache` to the ROCm DSV4 `ModelState` metadata.  The
cache is recreated with each scheduler-step metadata object, so it is scoped to
one model forward and does not persist across request scheduling steps.

The reuse path is controlled by:

```bash
VLLM_ROCM_DSV4_ATOM_PREFILL_INDEX_REUSE=1
```

This is the default.  Set it to `0` to return to the previous per-layer prefill
index build behavior.

What is reused:

- Prefill indptr CPU/GPU construction for the same `(T, token_offset,
  swa_only)` chunk.
- Common extend/SWA/CSA-prefix index writes for the same chunk totals.
- HCA prefix index writes only when the same HCA block-table tensor view is
  reused; the cache key includes data pointer, storage offset, strides, shape,
  and chunk totals.

What is still per-layer:

- CSA top-k translation/packing, because the top-k results are layer-specific.
- The actual ATOM paged prefill attention kernel.
- SWA writes after attention.

Correctness guard:

- The CSA prefix index buffer still receives a one-time `-1` initialization
  before the common index write for a chunk.  This preserves the existing
  behavior where unwritten CSA prefix tail entries stay sentinel-filled while
  each CSA layer overwrites only its valid top-k head.

Runtime smoke/profile with reuse enabled:

- Server: V2 model runner, no `--enforce-eager`, block size `128`,
  `MAX_NUM_SEQS=32`, `MAX_MODEL_LEN=2304`,
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`,
  `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`,
  `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`,
  `VLLM_ROCM_DSV4_ATOM_PREFILL_INDEX_REUSE=1`.
- Short random C32 burst: `input_len=128`, `output_len=1`,
  `num_prompts=32`.
- Requests: `32 / 32` successful.
- Profile rows parsed from
  `/app/atomdsv4/server_prefill_reuse_profile_c32.log`: `1289`.

Large chunk (`T=3696`) comparison:

- CSA (`ratio=4`):
    - previous: `total_ms` mean `0.758`, `build_ms` mean `0.249`,
    `index_ms` mean `0.076`, overhead excluding kernel mean `0.478`
    - reuse: `total_ms` mean `0.460`, `build_ms` mean `0.005`,
    `index_ms` mean `0.000`, overhead excluding kernel mean `0.171`
- HCA (`ratio=128`):
    - previous: `total_ms` mean `0.715`, `build_ms` mean `0.230`,
    `index_ms` mean `0.069`, overhead excluding kernel mean `0.437`
    - reuse: `total_ms` mean `0.461`, `build_ms` mean `0.017`,
    `index_ms` mean `0.005`, overhead excluding kernel mean `0.171`

Interpretation:

- The cache directly attacks the measured per-layer prefill conversion cost.
  On cache-hit layers, `build_ms` and `index_ms` are effectively removed.
- This validates that conversion/index preparation was materially slowing the
  prefill wrapper around the ATOM kernel.
- The short profiling run is not a final throughput benchmark. It still logs
  every layer and triggers first-request JIT warnings, so TTFT from this run is
  not comparable to the non-profiling C32 serving benchmark.
- Required next gates before treating this as the new default validated path:
  run unchanged `lmeval.sh` for GSM8K accuracy, then restart the server and run
  the normal C32 `benchmarkvllm.sh`.

Validation after enabling prefill index reuse:

- `python3 -m py_compile` passed for:
    - `vllm/models/deepseek_v4/amd/rocm.py`
    - `vllm/models/deepseek_v4/amd/model_state.py`
    - `vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill_indices.py`
    - `vllm/models/deepseek_v4/amd/v4_kernels/csa_translate_pack.py`
- GSM8K with unchanged `lmeval.sh`, V2 runner, no `--enforce-eager`,
  `MAX_NUM_SEQS=64`, `MAX_MODEL_LEN=8192`,
  `VLLM_ROCM_DSV4_ATOM_PREFILL_INDEX_REUSE=1`:
    - flexible exact match: `0.9530 +/- 0.0058`
    - strict exact match: `0.9538 +/- 0.0058`
    - result is within the required `0.95 +/- 0.01` band.
- Fresh C32 benchmark server, V2 runner, no `--enforce-eager`,
  `MAX_NUM_SEQS=32`, `MAX_MODEL_LEN=2304`,
  `GPU_MEMORY_UTILIZATION=0.9`,
  `VLLM_ROCM_DSV4_ATOM_PREFILL_INDEX_REUSE=1`:
    - result file:
    `/app/atomdsv4/bench-prefill-reuse-gpu90-len2304/ds-v4-pro-prefill-reuse-gpu90-len2304-C32.json`
    - completed requests: `320`
    - failed requests: `0`
    - output throughput: `855.67 tok/s`
    - total throughput: `1714.68 tok/s`
    - mean TPOT: `36.35 ms`
    - median TPOT: `36.23 ms`
    - mean TTFT: `1102.81 ms`

Comparison against the previous fastest validated C32 run with indexer
fastpath but without prefill index reuse:

- Previous: output throughput `857.47 tok/s`, total throughput `1718.30 tok/s`,
  mean TPOT `36.27 ms`.
- With reuse: output throughput `855.67 tok/s`, total throughput
  `1714.68 tok/s`, mean TPOT `36.35 ms`.
- The micro-profile improvement does not translate into a C32 decode-heavy
  throughput win. This is expected because the normal C32 1024/1024 benchmark
  is dominated by decode, while prefill index reuse attacks the prefill wrapper.
  Keep the reuse path because it is accuracy-safe and removes measured prefill
  overhead, but do not count it as a C32 throughput improvement.

### Metadata Upload Cleanup

A small follow-up cleanup targets the conversion path directly:

- Request-level CPU mirrors in `DeepseekV4RocmAtomModelState` now use pinned
  CPU tensors with NumPy views instead of plain NumPy arrays.
- Compressor-plan host buffers now use the same pinned CPU backing.
- Hot-path metadata uploads now copy directly from pinned CPU slices into
  persistent GPU tensors.
- The model-state file no longer contains `torch.from_numpy(...).to(device)` in
  the request metadata path.

Expected effect:

- This does not change the mathematical attention/compressor ordering.
- It removes avoidable temporary tensor creation and pageable-host copies from
  the per-forward metadata preparation path.
- The expected performance impact is likely modest, but it is aligned with the
  conversion/metadata overhead hypothesis and is safe to compare against the
  previous C32 baseline of `850.82 tok/s`.

Verification for the cleanup:

- `python3 -m py_compile` passed for:
    - `vllm/models/deepseek_v4/amd/model_state.py`
    - `vllm/models/deepseek_v4/amd/rocm.py`
    - `vllm/models/deepseek_v4/compressor.py`
    - `vllm/models/deepseek_v4/attention.py`
- `rg` found no remaining `torch.from_numpy` or `.to(self.device)` conversions
  in `vllm/models/deepseek_v4/amd/model_state.py`.

Runtime validation after the cleanup:

- Smoke request:
    - prompt: `What is 2+2? Answer briefly.`
    - output: `4`
- GSM8K with unchanged `lmeval.sh`, `MAX_NUM_SEQS=64`,
  `MAX_MODEL_LEN=8192`, V2 runner, no `--enforce-eager`:
    - flexible exact match: `0.9530`
    - strict exact match: `0.9538`
    - result is within the required `0.95 +/- 0.01` band.
- C32 benchmark with `MAX_NUM_SEQS=32`, `MAX_MODEL_LEN=2304`,
  `GPU_MEMORY_UTILIZATION=0.9`, V2 runner, no `--enforce-eager`:
    - successful requests: `320 / 320`
    - output throughput: `853.60 tok/s`
    - total throughput: `1710.54 tok/s`
    - mean TPOT: `36.44 ms`
    - mean TTFT: `1106.48 ms`
    - result file:
    `/app/atomdsv4/bench-from-vllm-unified-pinnedmeta-gpu90-len2304/ds-v4-pro-from-vllm-unified-pinnedmeta-gpu90-len2304-C32.json`

Comparison against the pre-cleanup vLLM-owned BF16 unified-KV C32 baseline:

- Output throughput improved from `850.82 tok/s` to `853.60 tok/s`, about
  `+0.33%`.
- Mean TPOT improved from `36.58 ms` to `36.44 ms`, about `-0.38%`.
- This confirms the pinned metadata cleanup removes some overhead, but it is
  not the main gap to ATOM's documented C32 target of `1145.71 tok/s` and
  `26.90 ms` TPOT.

Updated interpretation:

- Metadata upload cleanup is directionally correct and accuracy-safe.
- The remaining performance gap is dominated by the larger structural issues:
  conversion/index kernels around attention, missing or partial stream overlap,
  and the current vLLM block-table scheduling contract rather than host upload
  overhead or main unified-KV dtype alone.

### Indexer Decode Fastpath And Unused Kernel Cleanup

The follow-up slice added an opt-in CSA indexer decode fastpath behind
`VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`.

Intent:

- Reuse ATOM `ModelState` committed-count metadata for pure decode.
- Avoid the generic `SparseAttnIndexer` body when the scheduler shape is the
  normal one-token-per-request decode case.
- Still use vLLM-owned indexer cache and aiter-backed FP8 MQA logits/top-k
  wrappers, so this is not yet a full ATOM indexer-cache replacement.

Runtime validation:

- GSM8K with unchanged `lmeval.sh`, `MAX_NUM_SEQS=64`,
  `MAX_MODEL_LEN=8192`, V2 runner, no `--enforce-eager`,
  `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`:
    - flexible exact match: `0.9553`
    - strict exact match: `0.9560`
    - result is within the required `0.95 +/- 0.01` band.
- C32 benchmark with `MAX_NUM_SEQS=32`, `MAX_MODEL_LEN=2304`,
  `GPU_MEMORY_UTILIZATION=0.9`, V2 runner, no `--enforce-eager`,
  `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`:
    - successful requests: `320 / 320`
    - output throughput: `857.47 tok/s`
    - total throughput: `1718.30 tok/s`
    - mean TPOT: `36.27 ms`
    - mean TTFT: `1099.73 ms`
    - result file:
    `/app/atomdsv4/bench-from-vllm-unified-indexerfast-gpu90-len2304/ds-v4-pro-from-vllm-unified-indexerfast-gpu90-len2304-C32.json`

Comparison against the pinned-metadata C32 run:

- Output throughput improved from `853.60 tok/s` to `857.47 tok/s`, about
  `+0.45%`.
- Mean TPOT improved from `36.44 ms` to `36.27 ms`, about `-0.47%`.

Interpretation:

- The fastpath is accuracy-safe for the tested deployment configuration.
- The small gain confirms that generic indexer decode overhead exists, but it
  is not the primary reason this path remains far from ATOM's documented C32
  target.
- Remaining likely costs are still metadata/index construction, conversion
  kernels around prefill/decode, stream overlap, and the mismatch between
  ATOM's request-state/unified-cache contract and vLLM's block-table contract.

Cleanup:

- Removed the unused standalone unpaged sparse-attention module
  `v4_kernels/sparse_attn_v4.py`.
- Added `v4_kernels/reference.py` with only the torch ragged sparse-attention
  reference helper needed by the paged decode/prefill reference paths.
- Updated `paged_decode.py` and `paged_prefill.py` to import that reference
  helper directly.

Verification for the cleanup:

- `rg` found no remaining imports of the deleted module.
- `python3 -m py_compile` passed for:
    - `vllm/models/deepseek_v4/amd/v4_kernels/reference.py`
    - `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`
    - `vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill.py`
    - `vllm/models/deepseek_v4/amd/rocm.py`
    - `vllm/models/deepseek_v4/attention.py`
    - `vllm/models/deepseek_v4/compressor.py`

### Metadata Profiling Diagnostic

A short diagnostic run used the C32-style deployment server configuration plus
`VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`:

```bash
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_MODEL_LEN=2304 \
GPU_MEMORY_UTILIZATION=0.9 \
ENFORCE_EAGER=0 \
BLOCK_SIZE=128 \
VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1 \
VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_PREFILL=0 \
VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1 \
VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC=0 \
VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC_STAGES=post_attn \
VLLM_ROCM_DSV4_ATOM_PREFILL_SYNC_KIND=stream \
VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1 \
bash launchdeepseekgraph.sh
```

Then a short request burst was sent through `benchmarkvllm.sh` with
`INPUT_LEN=128`, `OUTPUT_LEN=16`, and `CONCURRENCIES=32`. This run is not a
target performance benchmark; it exists only to exercise live metadata paths.

Request-time profile summary from `/app/atomdsv4/server_metadata_profile_c32.log`:

- Live C32 decode samples (`reqs=32`, `tokens=32`, 16 samples):
    - `super`: mean `0.964 ms`, min `0.782 ms`, max `1.207 ms`
    - `unified`: mean `0.002 ms`
    - `plans`: mean `0.166 ms`, min `0.135 ms`, max `0.210 ms`
    - `state`: mean `0.129 ms`, min `0.110 ms`, max `0.154 ms`
    - `attach`: mean `0.011 ms`
    - `total`: mean `1.272 ms`, min `1.040 ms`, max `1.574 ms`
- Live C32 prefill-ish samples (`reqs=32`, `tokens=64`, 8 samples):
    - `super`: mean `2.435 ms`
    - `plans`: mean `0.198 ms`
    - `state`: mean `0.145 ms`
    - `total`: mean `2.792 ms`
- Capture samples are noisier, with `total` mean `5.302 ms` and a max
  `54.789 ms`; those include graph-capture warmup behavior and should not be
  used as steady-state evidence.

Interpretation:

- Host-side ATOM metadata construction is measurable but small in the C32
  decode path: about `0.30 ms` combined for `plans + state`, and about
  `1.27 ms` including inherited vLLM metadata.
- This is too small to explain the gap from the current `36.27 ms` TPOT to
  ATOM's documented `26.90 ms` C32 target.
- The stronger remaining suspects are GPU-side conversion/index kernels,
  JIT/warmup coverage, missing stream overlap, and the fact that vLLM is still
  adapting ATOM kernels to the block-table scheduler contract rather than using
  a native ATOM/SGLang-style request-state and unified-cache dataflow end to
  end.

### Decode Wrapper Profiling Diagnostic

A second short diagnostic run used the same C32-style server configuration but
enabled:

```bash
VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=-1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=1
```

Then a 32-request burst was sent with random `input_len=128`, `output_len=4`,
and `max_concurrency=32`. This run is intentionally noisy because every
profiled layer synchronizes and logs. It should be used only for attribution,
not as a throughput measurement.

Parsed profile rows from `/app/atomdsv4/server_decode_profile_c32.log`:

- Total `ATOM_PROFILE_DECODE` rows: `3817`.
- Rows with `T=32`: `486`.
- Rows with `T=1`: `488`.

For `T=32` rows:

- CSA (`ratio=4`, 238 rows):
    - `index_ms`: mean `0.0967`, p50 `0.0615`
    - `translate_ms`: mean `0.1020`, p50 `0.0380`
    - `kernel_ms`: mean `0.0872`, p50 `0.0530`
    - `total_ms`: mean `0.2862`, p50 `0.1550`
- HCA (`ratio=128`, 248 rows):
    - `index_ms`: mean `0.1481`, p50 `0.0600`
    - `translate_ms`: mean `0.0032`, p50 `0.0030`
    - `kernel_ms`: mean `0.1472`, p50 `0.0540`
    - `total_ms`: mean `0.2987`, p50 `0.1190`

For `T=1` rows:

- CSA (`ratio=4`, 240 rows):
    - `total_ms`: mean `0.1866`, p50 `0.1500`
- HCA (`ratio=128`, 248 rows):
    - `total_ms`: mean `0.1504`, p50 `0.1130`

Interpretation:

- Individual ATOM decode wrapper calls are not large enough by themselves to
  explain the C32 TPOT gap. Per-layer medians are roughly `0.15 ms` for CSA and
  `0.11-0.12 ms` for HCA, with the kernel itself around `0.05 ms`.
- The outlier-heavy means are expected because the profiler synchronizes and
  logs every layer; the medians are the more useful signal.
- The accumulated decode cost is still meaningful across 61 layers, but the
  larger target remains structural: reduce the number of separate index,
  translation, state-update, and attention launches, and recover ATOM's
  multistream overlap / native unified-cache dataflow rather than optimizing
  Python wrapper time.

### Decode Index Reuse Diagnostic

To test whether conversion/index preparation can hide the benefit of faster
attention kernels, a per-forward decode index cache was added around the ATOM
paged-decode index writer:

- The shared SWA/CSA/HCA tail indices are keyed by `(T, decode_swa_total,
  decode_csa_total, decode_hca_total)` and reused across all layers in the same
  forward.
- Fused HCA head indices are additionally keyed by the HCA block-table view,
  storage offset, strides, shape, and block capacity.
- CSA top-k translation is intentionally not reused because it depends on the
  layer-local top-k buffer.

The validation run used:

```bash
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_MODEL_LEN=2304 \
GPU_MEMORY_UTILIZATION=0.9 \
ENFORCE_EAGER=0 \
BLOCK_SIZE=128 \
VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1 \
VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_PREFILL=0 \
VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1 \
VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1 \
VLLM_ROCM_DSV4_ATOM_PREFILL_INDEX_REUSE=1 \
VLLM_ROCM_DSV4_ATOM_DECODE_INDEX_REUSE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=-1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=1 \
bash launchdeepseekgraph.sh
```

Then a diagnostic request burst was sent with random `input_len=128`,
`output_len=4`, `num_prompts=32`, and `max_concurrency=32`. This is not a
throughput benchmark because the profile mode synchronizes/logs every layer.

Comparison against `/app/atomdsv4/server_decode_profile_c32.log` at `T=32`:

| Path | Decode index reuse | `index_ms` p50 | `translate_ms` p50 | `kernel_ms` p50 | `total_ms` p50 |
| ---- | ------------------ | -------------- | ------------------ | --------------- | -------------- |
| HCA `ratio=128` | off | `0.060` | `0.003` | `0.054` | `0.119` |
| HCA `ratio=128` | on | `0.000` | `0.003` | `0.072` | `0.088` |
| CSA `ratio=4` | off | `0.061` | `0.038` | `0.053` | `0.155` |
| CSA `ratio=4` | on | `0.000` | `0.056` | `0.056` | `0.120` |

Short-burst result file:
`/app/atomdsv4/bench-decode-reuse-profile-short/decode-reuse-profile-short-C32.json`

- Completed requests: `32`
- Failed requests: `0`
- Output throughput: `100.37 tok/s`
- Total token throughput: `3412.72 tok/s`
- Mean TPOT: `45.22 ms`
- Median TPOT: `31.36 ms`
- Mean TTFT: `1133.96 ms`

Interpretation:

- Yes, conversion/index preparation can materially slow the apparent attention
  path. Before reuse, every layer paid about `0.06 ms` p50 just to rewrite
  decode indices; across 61 layers this is several milliseconds of serialized
  wrapper/kernel-launch work.
- The new cache removes that repeated per-layer decode-index cost. HCA p50
  wrapper total improves from `0.119 ms` to `0.088 ms`; CSA improves from
  `0.155 ms` to `0.120 ms`.
- CSA remains conversion-heavy because `csa_translate_pack` is still layer
  specific. In the cached run, CSA translation alone is about as expensive as
  the attention kernel (`~0.056 ms` p50 each).
- Host metadata preparation is smaller than this effect: the earlier metadata
  diagnostic measured about `0.30 ms` for ATOM-specific `plans + state` at C32
  decode and about `1.27 ms` including inherited vLLM metadata.
- The next structural target is not Python metadata but fusing or removing
  GPU-side conversion launches, especially CSA top-k translation and any
  remaining per-layer state/index adaptation from vLLM block tables to ATOM
  paged indices.

Validation after enabling decode index reuse:

- Accuracy command: unchanged `/app/atomdsv4/lmeval.sh`.
- Server configuration:
    - `MAX_NUM_SEQS=64`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `GPU_MEMORY_UTILIZATION=0.9`
    - `ENFORCE_EAGER=0`
    - `BLOCK_SIZE=128`
    - `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
    - `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_PREFILL=0`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`
    - `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`
    - `VLLM_ROCM_DSV4_ATOM_PREFILL_INDEX_REUSE=1`
    - `VLLM_ROCM_DSV4_ATOM_DECODE_INDEX_REUSE=1`
- Result from `/app/atomdsv4/lmeval_decode_reuse.log`:
    - `gsm8k` flexible-extract exact match: `0.9522 +/- 0.0059`
    - `gsm8k` strict-match exact match: `0.9530 +/- 0.0058`

Performance after enabling decode index reuse:

- Benchmark command: `/app/atomdsv4/benchmarkvllm.sh` with
  `CONCURRENCIES=32`, `INPUT_LEN=1024`, `OUTPUT_LEN=1024`, warmups enabled.
- Server was restarted after the accuracy run to clear KV/prefix state.
- Result file:
  `/app/atomdsv4/bench-decode-index-reuse-c32/ds-v4-pro-decode-index-reuse-C32.json`
- Completed requests: `320`
- Failed requests: `0`
- Output throughput: `861.85 tok/s`
- Total token throughput: `1727.07 tok/s`
- Mean TPOT: `36.15 ms`
- Median TPOT: `36.07 ms`
- Mean TTFT: `1035.27 ms`

Comparison against the recent vLLM-owned unified-KV ATOM path:

| Run | Output tok/s | Total tok/s | Mean TPOT ms | Notes |
| --- | ------------ | ----------- | ------------ | ----- |
| `from-vllm-unified-pinnedmeta` | `853.60` | `1710.54` | `36.44` | vLLM-owned unified KV baseline |
| `from-vllm-unified-indexerfast` | `857.47` | `1718.30` | `36.27` | indexer fastpath |
| `prefill-reuse` | `855.67` | `1714.68` | `36.35` | prefill index reuse only |
| `decode-index-reuse` | `861.85` | `1727.07` | `36.15` | decode index reuse enabled |

Comparison against all saved historical C32 JSONs under `/app/atomdsv4`:

- The new `decode-index-reuse` run is the best result in the current
  vLLM-owned unified-KV ATOM integration series.
- It is not the fastest historical C32 JSON in the workspace. Older
  `bench-sparsemla` runs remain higher, for example:
    - `/app/atomdsv4/bench-sparsemla/revert-compressor-aux-nomtp-C32.json`:
    `926.06 tok/s`, mean TPOT `33.50 ms`
    - `/app/atomdsv4/bench-sparsemla/ds-v4-pro-nomtp-compressor-order-off-C32.json`:
    `925.13 tok/s`, mean TPOT `33.50 ms`
    - `/app/atomdsv4/bench-sparsemla/fused-clamp-actmul-C32.json`:
    `922.73 tok/s`, mean TPOT `33.71 ms`
- Treat those as historical comparison points, not proof that the current
  ATOM unified-KV path has reached that level. They came from earlier
  configurations and need their exact feature set / accuracy assumptions checked
  before using them as the target baseline.

Current conclusion:

- Decode index reuse is accuracy-safe and produces a measurable but modest C32
  improvement in the current ATOM integration path: about `+4.38 tok/s` over
  indexer-fast and `-0.13 ms` mean TPOT.
- It still leaves a large gap to ATOM's documented C32 target and even to older
  local `bench-sparsemla` results. The remaining bottleneck is likely not
  host-side metadata. The stronger candidate is still per-layer GPU conversion
  and launch structure, especially `csa_translate_pack`, compressor/state
  update launches, and missing ATOM-style multistream overlap.

## ATOM Component Coverage Audit

Date: 2026-06-19.

Question: do we have all components needed to get the benefit of all ATOM
kernels inside vLLM while keeping vLLM's scheduler?

Short answer: not yet. The current tree has enough pieces to run an
accuracy-correct vLLM-owned unified-KV ATOM-style path, but it does not yet
remove all structural adaptation between vLLM's block-table/ragged metadata and
ATOM's request-state/ring/index model. That adaptation is now a credible
performance limiter because several conversion launches are comparable to the
attention kernel cost.

Current launch defaults were updated to run the most relevant path:

- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_ROCM_DSV4_ATOM_STATE=1`
- `VLLM_ROCM_DSV4_ATOM_STATE_ALLOC=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
- `VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN=1`
- `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
- `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_PREFILL=0`
- `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`
- `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`
- `VLLM_ROCM_DSV4_ATOM_PREFILL_INDEX_REUSE=1`
- `VLLM_ROCM_DSV4_ATOM_DECODE_INDEX_REUSE=1`
- `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`
- `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=1`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- `ATOM_USE_FUSED_Q_NORM_QUANT=1`
- `ATOM_USE_AITER_FUSED_CLAMP_ACT_MUL=1`

Component status:

| ATOM component | Current vLLM status | Evidence / caveat |
| --- | --- | --- |
| vLLM scheduler | Kept | V2 model runner path is selected via `VLLM_USE_V2_MODEL_RUNNER=1`; no GPU worker rewrite is required for the current path. |
| Per-request SWA/compressor state | Present | `DeepseekV4RocmAtomModelState` allocates/binds `swa_kv`, CSA/HCA compressor states, per-request slot maps, and prefill/decode buffers. |
| vLLM-owned unified KV | Present, ROCm-only | `DeepseekV4AtomMLAAttentionSpec` adds a SWA prefix, DSV4 KV planning reserves fixed prefix bytes, and `model_state.py` binds ATOM views from vLLM KV storage. CUDA path is untouched by the ROCm-only spec emission. |
| Main compressor | Present and active | `DeepseekCompressor._maybe_atom_main_compressor_forward` calls local ATOM `fused_compress_attn` then `update_compressor_states`, preserving ATOM read-before-state-update ordering. |
| Main compressor flydsl / aiter path | Partially active | `fused_compress_attn` can dispatch to aiter flydsl in supported shapes. HCA flat diagnostic layout is kept on Triton because the aiter entry assumes packed blocks. |
| Indexer compressor | Present through vLLM structure | Indexer owns a rotate=True `DeepseekCompressor` and skips the generic indexer K insert. The generic indexer path uses the vLLM cache/metadata wrapper, not ATOM's `torch.ops.aiter.indexer_score_topk` dispatcher. |
| Indexer scoring kernels | Mostly present | vLLM ROCm sparse indexer calls `cp_gather_indexer_k_quant_cache`, `fp8_mqa_logits`/`deepgemm_fp8_paged_mqa_logits`, and aiter `top_k_per_row_*` when available. Decode has a narrow `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH` over ModelState metadata. Prefill still uses the generic vLLM wrapper. |
| CSA top-k translation | Present but costly | `csa_translate_pack` is copied from ATOM and active for CSA decode/prefill. It remains per-layer and costs about the same as the decode attention kernel in profiling. |
| q norm / quant | Present | `_q_norm_maybe_quant` uses `get_rmsnorm_group_fused_quant_op()` when quant keys match; the launch script enables `ATOM_USE_FUSED_Q_NORM_QUANT=1`. |
| q/k RMSNorm + RoPE | Present | ROCm attention calls local ATOM `qk_norm_rope_maybe_quant`; quant outputs remain disabled for sparse attention (`quant_q=False`, `quant_k=False`) matching the validated BF16 path. |
| SWA write ordering | Present | Decode writes SWA before attention; prefill writes SWA after attention, matching ATOM's model-file ordering and preserving chunked prefill correctness. |
| Sparse paged decode | Present | `sparse_attn_v4_paged_decode` is copied locally and defaults to Triton, same as ATOM's `ATOM_USE_TRITON_ATTN=1` default. vLLM adds bounds checks and optional direct-HCA diagnostics. |
| Sparse paged prefill | Present | `sparse_attn_v4_paged_prefill` mirrors ATOM's OPUS-preferred fallback-to-Triton behavior. |
| Prefill/decode index reuse | Present, vLLM-specific optimization | Reuses common SWA/CSA/HCA index buffers across layers when metadata keys match. This reduces repeated vLLM-to-ATOM index writer launches but does not remove CSA translation. |
| MoE fused path | Present via vLLM | vLLM `FusedMoE` with `--moe-backend aiter` is used; dense MLP uses aiter `silu_and_mul`, optionally `fused_clamp_act_mul`. This is not the ATOM model-file MoE class, but it uses vLLM's fused MoE abstraction as requested. |
| MHC / HC | Present for available aiter ops, but gated | Current installed `aiter` exposes `mhc_pre` and `mhc_post`, but full GSM8K failed when they were enabled in the current vLLM path. They remain behind `VLLM_ROCM_DSV4_USE_AITER_MHC=1`; default keeps the validated non-aiter MHC path. This installed `aiter` does not expose `mhc_fused_post_pre`; ATOM's `getattr(aiter, "mhc_fused_post_pre", None)` would also resolve to `None` here. |
| Output inverse RoPE + LoRA | Present | ROCm path uses `rocm_inv_rope_einsum` then `wo_b`. It is not yet the FP8 grouped output LoRA optimization mentioned in ATOM comments. |
| Aux/multistream compressor overlap | Not active | ATOM's `maybe_compressors_async` uses side streams under hipgraph. The vLLM integration intentionally reverted/disabled aux stream logic earlier; current compressor execution is sequential with respect to the main path. |
| True ATOM/SGLang unified request-state KV layout | Not complete | vLLM now owns a unified storage allocation with ATOM views, but the scheduler still produces vLLM block tables and metadata. ATOM-style native ring/request metadata is reconstructed or translated at runtime. |

Implications:

- The current integration is not simply "vLLM sparse attention with aiter MoE".
  It uses ATOM-style compressor, q/k norm+RoPE, SWA write, paged decode/prefill,
  CSA translation, and vLLM-owned unified KV binding.
- The remaining gap is structural. The model still pays to adapt vLLM's
  scheduler/KV metadata into ATOM-style per-layer indices and request-state
  buffers.
- Host-side metadata is not the main measured issue. GPU-side conversion
  launches are: decode index writes were measurable until reuse, and
  `csa_translate_pack` is still comparable to sparse decode kernel time.
- `os.environ.get(...)` has been kept out of hot paths for the ATOM additions:
  the flags are cached at module import. Remaining environment reads found in
  this audit are import-time constants or non-hot backend setup paths.

Highest-impact next integration targets:

1. Remove or fuse CSA translation. The most direct path is an ATOM CSA attention
   entry that consumes raw seq-local top-k plus block tables and computes the
   physical unified-KV slot in the attention kernel, eliminating the separate
   `csa_translate_pack` launch.
2. Move prefill indexer closer to ATOM's `Indexer.forward_batched` contract:
   one ModelState-backed metadata object, fewer vLLM wrapper branches, and no
   generic sparse-indexer adaptation when ATOM attention is active.
3. Revisit multistream compressor overlap only after the current synchronous
   path is stable under graph capture. It is required for matching ATOM's model
   execution shape, but it is riskier than CSA translation because it affects
   stream dependencies and graph replay.
4. Evaluate grouped output LoRA FP8 optimization separately. It is independent
   from the KV/cache scheduler work and should be benchmarked behind a narrow
   flag.

Completion criteria for "all ATOM kernels benefit inside vLLM":

- `launchdeepseekgraph.sh` starts without `--enforce-eager`, using vLLM V2
  model runner and vLLM-owned unified KV by default.
- Unchanged `lmeval.sh` reaches GSM8K exact match `0.95 +/- 0.01`.
- C32 `benchmarkvllm.sh` is collected after a server restart.
- Profiles show CSA/index conversion is no longer a per-layer cost comparable to
  the attention kernel.
- CUDA DSV4 path remains on the existing KV-cache spec and attention backend.

## Experimental Direct CSA Decode

Date: 2026-06-19.

Implementation added behind:

```bash
VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1
```

Files:

- `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/__init__.py`
- `vllm/models/deepseek_v4/amd/rocm.py`
- `/app/atomdsv4/launchdeepseekgraph.sh`

Purpose:

- Remove the separate `csa_translate_pack` launch for ratio-4 pure decode.
- Consume the indexer's raw seq-local top-k directly in the attention kernel.
- Compute the physical ATOM unified-KV slot inline:
    - `block_idx = topk // csa_block_capacity`
    - `slot = topk % csa_block_capacity`
    - `physical_block = block_table[batch_id, block_idx]`
    - `csa_slot = swa_pages + physical_block * csa_block_capacity + slot`
    - append the SWA ring tail using `state_slot * win_with_spec + (abs_pos % win_with_spec)`

Scope of this first version:

- Decode only.
- CSA ratio 4 only.
- BF16/FP16 unified KV only.
- `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1` only.
- It still uses vLLM's indexer top-k result buffer. It removes CSA translation,
  not indexer scoring.

Why it is disabled by default:

- It is accuracy-safe in the first full run, but it regressed C32 throughput
  versus the known-good translated CSA path.
- The first benchmark still showed inference-time JIT for
  `_csa_translate_pack_kernel`, so the runtime did not fully eliminate the
  conversion/metadata path. Direct CSA decode was active, but at least one
  prefill/fallback shape still used the old translation machinery.

Static validation performed:

```bash
python3 -m py_compile \
  vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py \
  vllm/models/deepseek_v4/amd/v4_kernels/__init__.py \
  vllm/models/deepseek_v4/amd/rocm.py
```

Dynamic equivalence validation performed:

- A standalone HIP test compared the existing translated path
  (`csa_translate_pack` + SWA tail indices +
  `_sparse_attn_v4_paged_decode_triton`) against
  `sparse_attn_v4_csa_topk_paged_decode`.
- Result: `max_diff = 0.0`, `allclose = True`.

Accuracy validation:

- Server command:
  `MAX_NUM_SEQS=64 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.9 VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1 bash launchdeepseekgraph.sh`
- Unchanged `/app/atomdsv4/lmeval.sh`
- `gsm8k` flexible-extract exact match: `0.9515 +/- 0.0059`
- `gsm8k` strict-match exact match: `0.9522 +/- 0.0059`
- This passes the required `0.95 +/- 0.01` range.

C32 performance validation:

- Server was restarted after the accuracy run to clear KV/prefix state.
- Benchmark command:
  `RESULT_PREFIX=ds-v4-pro-nomtp-csa-direct CONCURRENCIES=32 bash benchmarkvllm.sh`
- Result file:
  `/app/atomdsv4/bench-sparsemla/ds-v4-pro-nomtp-csa-direct-C32.json`
- Completed requests: `320`
- Failed requests: `0`
- Output throughput: `834.82 tok/s`
- Total token throughput: `1672.91 tok/s`
- Mean TPOT: `37.30 ms`
- Median TPOT: `37.19 ms`
- Mean TTFT: `1086.65 ms`

Comparison:

| Run | Output tok/s | Total tok/s | Mean TPOT ms | Notes |
| --- | ------------ | ----------- | ------------ | ----- |
| `decode-index-reuse` | `861.85` | `1727.07` | `36.15` | best current vLLM-owned unified-KV run |
| `csa-direct-decode` | `834.82` | `1672.91` | `37.30` | accuracy-safe, but slower |
| `revert-compressor-aux-nomtp` | `926.06` | `1855.74` | `33.50` | older historical config |

Conclusion:

- Direct CSA decode is not useful in this first form. It removes one narrow
  decode-side translation in principle, but the end-to-end run is slower.
- The likely reason is exactly the conversion/metadata concern: the attention
  kernel may do less preparatory work, but the full path still pays for
  metadata preparation and at least some `csa_translate_pack` fallback work.
- A useful next attempt should not just move top-k-to-slot conversion into one
  decode kernel. It should remove the upstream conversion contract more
  broadly, especially prefill/mixed paths and per-layer metadata setup, or fuse
  metadata prep with the attention entry that actually consumes it.

## CSA Translation / Metadata Follow-Up

Date: 2026-06-19.

Question:

- Could conversion logic and metadata preparation make a faster kernel look
  slower end-to-end?

Short answer:

- Yes, but the latest evidence says the cost is not the standalone
  `csa_translate_pack` kernel by itself. The model-level `translate_ms` segment
  can be as expensive as, or more expensive than, the CSA decode attention
  kernel because the profiling sync waits for earlier queued indexer/conversion
  work as well.

Profile setup:

```bash
MAX_NUM_SEQS=32 \
MAX_NUM_BATCHED_TOKENS=8192 \
MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=2 \
VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=1 \
VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=0 \
bash launchdeepseekgraph.sh
```

Short diagnostic workload:

```bash
vllm bench serve \
  --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --dataset-name random \
  --input-len 1024 \
  --output-len 16 \
  --num-prompts 32 \
  --request-rate inf \
  --max-concurrency 32 \
  --num-warmups 0 \
  --random-range-ratio 0 \
  --ignore-eos
```

Evidence:

- Layer 2 is the first CSA layer (`layer_types[2] =
  compressed_sparse_attention`, ratio 4).
- With breakable CUDA graphs enabled, most Python wrapper timing prints occur
  during graph capture. Runtime graph replay does not necessarily execute the
  Python timing/print path. Therefore the C32 timings below are captured graph
  timings, not a no-graph eager runtime profile.
- Captured C32 CSA layer-2 samples:
    - `T=32`, `n=8`
    - `translate_ms` mean `2.264`, p50 `2.325`
    - `kernel_ms` mean `1.104`, p50 `1.098`
    - `total_ms` mean `3.385`, p50 `3.430`
    - translate/kernel mean ratio: `2.05x`
- Captured smaller-shape samples:
    - `T=16`: translate mean `0.057 ms`, kernel mean `0.064 ms`
    - `T=8`: translate mean `0.056 ms`, kernel mean `0.061 ms`
    - `T=4`: translate mean `0.058 ms`, kernel mean `0.055 ms`
- Runtime metadata samples from the same layer-2 profiling series showed
  ModelState metadata attach/planning overhead around `0.98 ms` at `tokens=32`
  (`super` about `0.73 ms`, plans about `0.125 ms`, state about `0.112 ms`).
- Inference JIT warnings still included:
    - `_build_prefill_chunk_metadata_kernel`
    - `_compute_prefill_metadata_kernel`
    - `_v4_paged_prefill_indices_kernel`
    - `_csa_translate_pack_kernel`

Standalone `csa_translate_pack` microbenchmark:

- A separate GPU microbenchmark used `T=32`, `index_topk=1024`,
  `csa_block_capacity=32`, `window_size=128`, and repeated the isolated
  `csa_translate_pack` call with CUDA events after warmup.
- Results:
    - effective valid K `512`: `0.0153 ms`
    - effective valid K `640`: `0.0159 ms`
    - effective valid K `768`: `0.0157 ms`
    - effective valid K `1024`: `0.0159 ms`
- This means the standalone translation kernel is not the 2 ms bottleneck.
  The model-level `translate_ms` bucket is a queue-synchronization segment that
  includes dependency/backlog from work launched before the translation call,
  especially indexer/top-k and metadata preparation.
- A trial graph-safe K-grid cap was considered and then reverted because the
  isolated kernel did not show a meaningful win and a dynamic per-forward cap
  is unsafe under CUDA graph replay.

Conclusion:

- Conversion and metadata preparation are now proven to be a first-order cost,
  not a rounding error.
- The direct CSA decode experiment did not help because it only changed one
  decode consumer. It did not remove the broader vLLM-to-ATOM metadata/indexer
  dependency chain.
- The next aligned integration target is not another narrow decode kernel or a
  micro-optimization of `csa_translate_pack`. It is a ROCm DSV4 metadata/KV
  contract that lets the indexer, compressor, and attention consume the same
  persistent request-state layout with fewer queued preparation kernels and
  fewer graph-captured compatibility steps.

## ATOM Indexer Dispatcher Audit

Date: 2026-06-19.

Question:

- Is vLLM missing an ATOM indexer op that would explain the remaining
  conversion/metadata gap?

ATOM model-file behavior:

- `ATOM/atom/models/deepseek_v4.py` imports:
    - `aiter.ops.topk.top_k_per_row_decode`
    - `aiter.ops.topk.top_k_per_row_prefill`
- Its `Indexer.forward_batched` computes:
    - replicated `wq_b`
    - RoPE + rotate activation
    - per-1x128 FP8 q quant
    - replicated `weights_proj`
    - scaled indexer weights
- Then it calls:
    - `torch.ops.aiter.indexer_score_topk(q_fp8, weights, self.prefix, self.index_topk)`
- ATOM's own comment says that dispatcher calls back into the Python module's
  `Indexer.indexer_score_topk(...)`, which then calls the actual kernels:
    - prefill: `cp_gather_indexer_k_quant_cache`, `fp8_mqa_logits`,
    `top_k_per_row_prefill`
    - decode: paged FP8 MQA logits, `top_k_per_row_decode`

Installed aiter 0.1.15.post1 runtime check:

```python
import aiter
[n for n in dir(aiter) if "index" in n.lower() or "top" in n.lower()]
```

Observed names:

- `cp_gather_indexer_k_quant_cache`
- `indexer_k_quant_and_cache`
- `indexer_qk_rope_quant_and_cache`
- `top_k_per_row_decode`
- `top_k_per_row_decode_fast`
- `top_k_per_row_prefill`
- `top_k_per_row_prefill_fast`
- grouped/top-k MoE helpers

Not observed:

- `indexer_score_topk`
- `torch.ops.aiter.indexer_score_topk`

Current vLLM status:

- `DeepseekV4Indexer.forward` already follows the same high-level math:
  replicated `wq_b`, fused indexer q/RoPE/quant, scaled weights, then indexer
  scoring/top-k.
- The narrow ROCm `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1` decode path bypasses
  the generic `SparseAttnIndexer` wrapper and directly uses:
    - `rocm_fp8_paged_mqa_logits`
    - `_top_k_per_row_decode`
    - ModelState `n_committed_csa_per_seq`
    - vLLM decode block table
- The generic path still exists for prefill and fallback cases.

Conclusion:

- There is no additional installed aiter `indexer_score_topk` kernel to enable.
  In ATOM, that name is a dispatcher/indirection point around the same lower
  level kernels, not a separate fused kernel exposed by aiter 0.1.15.post1.
- vLLM is already using the available aiter indexer kernels, but not through the
  ATOM dispatcher shape. The remaining difference is structural:
    - ATOM's dispatcher reads forward-context metadata shaped for ATOM's
    request-state/indexer contract.
    - vLLM still builds and adapts generic sparse-indexer metadata, then bridges
    it into the ModelState path for attention.
- Therefore, the next useful work is not "enable `indexer_score_topk`"; it is
  to reduce or replace the generic vLLM indexer metadata wrapper for ROCm DSV4
  with a ModelState-backed metadata object that the indexer, compressor, and
  attention all consume directly.

Follow-up implementation:

- `DeepseekV4RocmAtomStateMetadata` now carries:
    - `indexer_decode_block_table`
    - `indexer_decode_schedule_metadata`
    - `indexer_decode_requires_padding`
    - `indexer_decode_num_tokens`
- `DeepseekV4RocmAtomModelState.prepare_attn(...)` hoists these fields from the
  vLLM-built indexer decode metadata into the ATOM ModelState once per forward.
- `DeepseekV4Indexer._maybe_atom_decode_indexer_fastpath(...)` now consumes
  those fields from ModelState instead of importing/inspecting
  `DeepseekV32IndexerMetadata` in model forward.

What this does and does not solve:

- It reduces the model-forward dependency on the generic per-layer indexer
  metadata object. The ATOM decode indexer path now has a cleaner
  ModelState-facing contract.
- It does not yet remove the generic indexer metadata builder. The DeepGEMM
  paged MQA logits path still needs the block table and schedule metadata that
  builder prepares.
- The next deeper step is to build the required decode schedule metadata
  directly in `DeepseekV4RocmAtomModelState` from vLLM's common attention
  metadata/block tables, then bypass the generic `DeepseekV32IndexerMetadata`
  object entirely for the ROCm DSV4 ATOM path.

Validation:

- Static compile:
    - `python3 -m py_compile vllm/models/deepseek_v4/amd/model_state.py vllm/models/deepseek_v4/attention.py`
- Smoke server:
    - `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.9 bash launchdeepseekgraph.sh`
    - no `--enforce-eager`
    - V2 Model Runner
    - graph capture completed
    - vLLM-owned unified KV views were bound
- Smoke request:
    - prompt: `Question: What is 2+2? Answer:`
    - `max_tokens=16`, `temperature=0`
    - first answer token was `4`
- No runtime errors were present in `server_modelstate_indexer_smoke.log`.

## Direct ModelState Indexer Decode Metadata

Date: 2026-06-19.

Question:

- Can the ROCm DSV4 ATOM path stop depending on the generic
  `DeepseekV32IndexerMetadata` object for the common decode case?

Implemented slice:

- `DeepseekV4RocmAtomModelState` now owns a CUDAGraph-stable
  `indexer_decode_schedule_metadata` tensor.
- `prepare_attn(...)` tries to attach indexer decode metadata directly from
  ModelState inputs before falling back to the generic indexer metadata hoist.
- The direct path is intentionally narrow:
    - pure decode only,
    - one scheduled token per live request,
    - no padding expansion,
    - no mixed prefill/decode,
    - no speculative flattening.
- It locates the `DEEPSEEK_V4_INDEXER` attention group, reuses that group's
  vLLM block table, and attaches:
    - `indexer_decode_block_table`
    - `indexer_decode_schedule_metadata`
    - `indexer_decode_num_tokens`
    - `indexer_decode_requires_padding=False`

Important ROCm observation:

- `rocm_fp8_paged_mqa_logits(...)` currently accepts `schedule_metadata` for
  API compatibility but the aiter ROCm implementation does not consume it.
- Therefore, on ROCm the direct path mainly removes the dependency on the
  generic metadata object's decode wrapper. The important live inputs are the
  block table and `n_committed_csa_per_seq`.
- CUDA/DeepGEMM still needs real paged-MQA schedule metadata; the direct helper
  preserves that branch but it is not the ROCm hot path.

What this proves:

- ModelState already has enough information to prepare the deployment decode
  indexer metadata boundary for ROCm DSV4 without touching GPU workers.
- The indexer fastpath can consume ModelState-owned metadata while V2 model
  runner, vLLM scheduler, vLLM KV allocation, and graph capture remain active.

What this does not yet solve:

- `super().prepare_attn(...)` still runs first and still builds vLLM's generic
  attention/indexer metadata for all groups. This slice reduces the model
  forward dependency, but it does not yet remove the Python preparation cost.
- Compress plans still do CPU/NumPy construction and host-to-device copies each
  forward.
- ATOM-style unified prefill/decode still needs a deeper metadata path that can
  bypass vLLM's ragged/gather preparation for ROCm DSV4.

Validation:

- Static compile:
    - `python3 -m py_compile vllm/models/deepseek_v4/amd/model_state.py vllm/models/deepseek_v4/attention.py`
- Smoke server:
    - `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.9 bash launchdeepseekgraph.sh`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - no `--enforce-eager`
    - graph capture completed
    - vLLM-owned unified KV views were bound
- Smoke request:
    - prompt: `Question: What is 2+2? Answer:`
    - `max_tokens=16`, `temperature=0`
    - first answer token was `4`
- Log file: `server_direct_indexer_smoke.log`.
- No full `lmeval.sh` or C32 benchmark was run for this slice.

### Generic Indexer Metadata Skip Probe

Date: 2026-06-19.

Probe:

- Added an experimental flag:
  `VLLM_ROCM_DSV4_ATOM_SKIP_INDEXER_METADATA=1`.
- The probe filtered the `DEEPSEEK_V4_INDEXER` attention group out of
  `super().prepare_attn(...)` for the narrow pure-decode fastpath shape.
- Intent: avoid building generic `DeepseekV32IndexerMetadata` when
  `DeepseekV4Indexer._maybe_atom_decode_indexer_fastpath(...)` can use
  ModelState-owned metadata.

Result:

- First attempt was not runnable. Graph capture failed with:
  `KeyError: 'model.layers.2.attn.indexer.k_cache'`.
- Failure site was `vllm/models/deepseek_v4/compressor.py`, in
  `DeepseekCompressor.forward`, while the indexer-inner compressor was writing
  its compressed K into `model.layers.2.attn.indexer.k_cache`.
- That compressor reads `attn_metadata[self.k_cache_prefix]` to get the
  indexer K-cache slot mapping. This happens before the ATOM indexer fastpath
  can bypass `SparseAttnIndexer`.
- A second probe adds a minimal per-layer metadata object for the
  indexer-inner compressor K-cache write. It only provides `slot_mapping`,
  because `compress_norm_rope_store_triton(...)` only dereferences
  `k_cache_metadata.slot_mapping` for that write.
- With that minimal object, graph capture and a decode smoke request passed
  with `VLLM_ROCM_DSV4_ATOM_SKIP_INDEXER_METADATA=1`, no `--enforce-eager`, and
  Model Runner V2.

Conclusion:

- Full generic indexer metadata is not required for the narrow pure-decode
  ATOM indexer fastpath shape, but the indexer compressor cache write still
  needs a layer-keyed metadata entry that supplies `slot_mapping`.
- Therefore, removing vLLM's generic indexer metadata requires either this
  minimal compatibility object or moving the indexer compressor K-cache write
  fully to the ATOM/ModelState metadata contract.
- C32 deployment benchmark, 1024 input / 1024 output / 320 prompts, did not
  show a throughput win from this metadata skip:
    - default generic indexer metadata:
        - output throughput: 865.11 tok/s
        - total throughput: 1733.60 tok/s
        - mean TPOT: 36.07 ms
    - `VLLM_ROCM_DSV4_ATOM_SKIP_INDEXER_METADATA=1` with minimal indexer
    K-cache metadata:
        - output throughput: 862.96 tok/s
        - total throughput: 1729.30 tok/s
        - mean TPOT: 36.14 ms
    - delta: about -0.25% output throughput and +0.20% mean TPOT, which is
    within benchmark noise and does not support generic indexer metadata prep
    as the dominant C32 deployment bottleneck.
- The skip flag now defaults to off (`0`) and should remain an explicit
  development probe. It is useful for isolating metadata cost, but not yet a
  reason to change the default path.

Validation after disabling the probe by default:

- Static compile:
    - `python3 -m py_compile vllm/models/deepseek_v4/amd/model_state.py vllm/models/deepseek_v4/attention.py`
- Default smoke server:
    - `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.9 bash launchdeepseekgraph.sh`
    - no `--enforce-eager`
    - graph capture completed
    - vLLM-owned unified KV views were bound
- Smoke request:
    - prompt: `Question: What is 2+2? Answer:`
    - `max_tokens=16`, `temperature=0`
    - first answer token was `4`
- Log files:
    - failed probe: `server_skip_indexer_metadata_smoke.log`
    - restored default path: `server_default_after_skip_probe_smoke.log`
    - minimal metadata probe: `server_minimal_indexer_metadata_smoke2.log`
    - default C32 benchmark: `bench_default_metadata_client.log`,
    `bench_default_metadata_server.log`,
    `bench-metadata-compare/default_metadata-C32.json`
    - minimal-metadata C32 benchmark: `bench_minimal_metadata_client.log`,
    `bench_minimal_metadata_server.log`,
    `bench-metadata-compare/minimal_metadata-C32.json`

### Metadata And Conversion Prep Profile

Date: 2026-06-19.

Question:

- Could conversion logic or metadata preparation hide the benefit of the ATOM
  kernels?

Probe:

- Enabled the existing lightweight CPU-side metadata profiler:
  `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`.
- Server:
    - `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 GPU_MEMORY_UTILIZATION=0.9 bash launchdeepseekgraph.sh`
    - no `--enforce-eager`
    - Model Runner V2
    - block size 128
- Short load:
    - `vllm bench serve`
    - C32, 64 prompts
    - 1024 input / 128 output
    - 8 warmups

Observed `prepare_attn` profile:

- Capture calls:
    - first C32 calls include one-time unified KV binding and generic metadata
    setup; totals were about 50-55 ms on the first worker calls.
    - after first bind, capture calls settled around 1.1-2.0 ms for common C32
    decode-like shapes.
- Runtime calls:
    - all non-capture rows: mean total 1.49 ms, p50 0.99 ms.
    - C32 / 32-token runtime rows: mean total 1.21 ms, p50 0.99 ms.
    - C32 / 64-token runtime rows: mean total 2.86 ms, p50 2.89 ms.
- ATOM-specific pieces inside `ModelState.prepare_attn` were smaller:
    - compress plan build: about 0.15 ms mean for C32 / 32-token rows.
    - ATOM state build: about 0.14 ms mean for C32 / 32-token rows.
    - unified binding after startup: effectively 0.00 ms.
    - attach: about 0.01 ms.

Observed attention metadata helper profile:

- HCA/MLA ratio 128 was the larger metadata-side cost:
    - C32 rows averaged about 2.65 ms total in the helper profile, with about
    0.64 ms in ragged work.
- CSA/MLA ratio 4 was small:
    - C32 rows averaged about 0.16 ms total.

Interpretation:

- Generic indexer metadata is not the dominant deployment bottleneck at C32;
  the explicit skip benchmark showed no win.
- ATOM compress-plan/state preparation is measurable but small compared with
  36 ms TPOT at C32. It can matter for latency and graph replay overhead, but
  it does not explain the large gap to ATOM's target C32 figure by itself.
- The higher-signal conversion/prep suspects are now:
    - HCA ratio-128 attention metadata preparation.
    - Ragged/layout translation feeding ATOM paged attention.
    - Any remaining block-table to unified-KV conversion around HCA decode.
    - Device-side q norm/quant and top-k translate/pack kernels, which are not
    captured by the CPU-only `prepare_attn` profiler.

Artifacts:

- Server log: `profile_metadata_server.log`
- Client log: `profile_metadata_client.log`
- Result JSON: `bench-metadata-profile/profile_metadata-C32-O128.json`

### ROCm vLLM-Owned Unified KV Audit

Date: 2026-06-19.

Question:

- Do we already have the structural KV-cache pieces needed to get the full
  ATOM benefit, or are we still adapting vLLM's block-table layout at runtime?

Current vLLM-owned allocation path:

- `DeepseekV4Attention.get_kv_cache_spec(...)` emits
  `DeepseekV4AtomMLAAttentionSpec` only when the ROCm DSV4 ATOM unified-KV
  path is enabled.
- That spec adds a fixed SWA prefix:
  `max_num_seqs * (sliding_window + spec_tokens) * head_dim * dtype_size`.
- `_get_kv_cache_config_deepseek_v4(...)` subtracts that fixed prefix before
  computing the shared `num_blocks` and then allocates one tensor per ATOM
  sparse-attention layer:
  `atom_swa_prefix_bytes + page_size_bytes * num_blocks`.
- `_reshape_kv_cache(...)` skips the fixed prefix and returns the compressed
  tail to the attention layer in the normal vLLM shape:
  `[num_blocks, block_size // compress_ratio, head_dim]`.
- `bind_kv_cache(...)` calls `post_bind_kv_cache(...)`, and
  `DeepseekV4Attention.post_bind_kv_cache(...)` creates ATOM views over the
  same storage:
    - `atom_unified_kv = [swa_pages + num_blocks * k_per_block, head_dim]`
    - `atom_swa_kv = [max_num_seqs, win_with_spec, head_dim]`
    - `atom_compressed_kv_cache = [num_blocks, k_per_block, head_dim]`
- `DeepseekV4RocmAtomModelState._try_bind_atom_unified_kv_from_vllm(...)`
  validates and binds the same views again from model state, so graph capture
  and runtime use the same vLLM-owned storage.

What this proves:

- The current path is no longer a pure side allocation. The active ROCm DSV4
  sparse-attention KV tensor is owned by vLLM's KV-cache allocator and has an
  ATOM-readable SWA prefix plus compressed tail.
- CUDA remains untouched because the ATOM spec is only emitted by the ROCm DSV4
  model path when the ATOM unified-KV flag is enabled.
- This satisfies the first practical split requirement: no GPU worker rewrite
  is required for persistent request state and vLLM-owned unified KV views.

Remaining adapter layers:

- The tail is still allocated in vLLM block-table form. ATOM decode kernels
  must translate from request/block-table metadata to ATOM unified-KV page ids.
- The SWA ring is persistent per request, but the scheduler does not allocate
  request-state/ring slots as a first-class KV-cache resource. ModelState maps
  request indices to state slots and clears SWA rows on request removal.
- The indexer cache is separate from the main ATOM unified-KV tensor and still
  uses the vLLM indexer/cache metadata contract for prefill and fallback cases.
- The compressor/indexer/attention do not yet share one native ATOM metadata
  object end to end:
    - ModelState builds ATOM state and compress plans.
    - vLLM still builds common MLA/SWA/indexer metadata.
    - The ROCm wrapper adapts those into ATOM decode/prefill index buffers.
- The current homogeneous BF16 unified tensor is accuracy-valid, but it is not
  the final mixed FP8/BF16 ATOM storage contract. Supporting FP8 compressed
  tails inside the same logical unified layout would require either split-view
  kernels or a raw-layout kernel contract.

Conclusion:

- We have enough components for a correct preview using vLLM's scheduler,
  vLLM-owned KV allocation, ModelState request buffers, ATOM compressor,
  ATOM SWA writes, and ATOM paged attention kernels.
- We do not yet have the full structural setup needed to receive all ATOM
  performance benefit. The hot path still adapts vLLM block tables and generic
  metadata into ATOM's request-state layout.
- The next meaningful implementation boundary is a ROCm-only DSV4 metadata
  contract that makes request-state/ring slots and compressed-page addressing
  first-class for the indexer, compressor, and attention together. Short of
  that, the best narrow target remains fusing/removing CSA top-k translation
  and reducing generic indexer metadata use in prefill.

## ATOM Model-File Op Audit

Date: 2026-06-19.

Question:

- Do we have all components needed to benefit from the ops used by
  `ATOM/atom/models/deepseek_v4.py`, and where does the current vLLM ROCm path
  still differ?

Source of truth:

- ATOM model file:
  `ATOM/atom/models/deepseek_v4.py`
- vLLM ROCm model:
  `vllm/models/deepseek_v4/amd/model.py`
- vLLM ROCm attention adapter:
  `vllm/models/deepseek_v4/amd/rocm.py`
- vLLM DSV4 shared attention/indexer:
  `vllm/models/deepseek_v4/attention.py`
- vLLM ROCm ATOM kernels:
  `vllm/models/deepseek_v4/amd/v4_kernels/*`
- vLLM ROCm sparse/indexer wrappers:
  `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`

### Attention And Compressor

Active or available in the current vLLM ROCm path:

- `qk_norm_rope_maybe_quant`
    - ATOM calls this for per-head Q RMSNorm, KV RMSNorm, and RoPE.
    - vLLM imports the same local kernel wrapper from
    `amd/v4_kernels/qk_norm_rope_maybe_quant.py` and calls it from
    `DeepseekV4ROCMAiterMLAAttention._fused_qnorm_rope_kv_insert`.
    - The current active path sets `quant_q=False` and `quant_k=False`, matching
    ATOM's default sparse-attention path. `ATOM_USE_FUSED_Q_NORM_QUANT=1`
    affects the earlier q-lora RMSNorm/quant path, not this kernel's
    `quant_q` flag.
- `swa_write`
    - ATOM writes decode SWA before attention and prefill SWA after attention.
    - vLLM calls the same local `swa_write` in `_maybe_atom_swa_write` and after
    ATOM prefill attention.
- `fused_compress_attn`
    - ATOM runs this before `update_compressor_states`.
    - vLLM runs the same local wrapper in
    `DeepseekCompressor._maybe_atom_main_compressor_forward`.
- `update_compressor_states`
    - ATOM updates the per-request compressor rings after fused compression.
    - vLLM runs the same local wrapper after `fused_compress_attn`.
- `sparse_attn_v4_paged_decode`
    - ATOM's decode sparse attention kernel is present locally and used in
    vLLM's ATOM attention decode path.
- `sparse_attn_v4_paged_prefill`
    - ATOM's two-source prefill sparse attention kernel is present locally and
    used when `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`.
- `csa_translate_pack`
    - ATOM uses this to translate CSA indexer top-k rows into paged offsets.
    - vLLM uses the local kernel in both decode and prefill CSA paths.
- `inverse_rope_inplace`
    - ATOM uses an inverse-RoPE output step before grouped output LoRA.
    - vLLM uses a fused ROCm inverse-RoPE + cached BF16 `wo_a` path through
    `rocm_inv_rope_einsum`. This is equivalent in intent, but not the exact
    ATOM model-file call.

Important differences:

- ATOM's attention kernels read a native unified layout. vLLM now allocates a
  vLLM-owned ATOM-readable unified tensor, but it still adapts vLLM block-table
  metadata and request state into ATOM index buffers.
- ATOM's compressor, indexer, and attention all share one model-file metadata
  contract. vLLM splits this across ModelState metadata, common MLA/SWA
  metadata, indexer metadata, and adapter kernels.
- ATOM's optional asynchronous compressor overlap was intentionally reverted in
  this vLLM branch. The current ROCm model disables aux streams and runs the
  compressor path synchronously.

### Indexer

ATOM model-file sequence:

- `rope_rotate_activation`
- `get_hip_quant(QuantType.per_1x128)` for indexer Q
- `scale_indexer_weights`
- `torch.ops.aiter.indexer_score_topk(...)`
    - prefill internally uses `cp_gather_indexer_k_quant_cache`,
    `fp8_mqa_logits`, and `top_k_per_row_prefill`.
    - decode internally uses `deepgemm_fp8_paged_mqa_logits` and
    `top_k_per_row_decode`.

vLLM status:

- vLLM does not depend on ATOM's `torch.ops.aiter.indexer_score_topk`
  dispatcher. Installed `aiter==0.1.15.post1` does not expose that dispatcher.
- vLLM fuses the ATOM indexer Q-side sequence into
  `fused_indexer_q_rope_quant`: RoPE rotation, per-1x128 FP8/MXFP4 quant, and
  weight scaling are one local vLLM helper instead of ATOM's separate
  `rope_rotate_activation`, `get_hip_quant`, and `scale_indexer_weights` calls.
- vLLM implements the same lower-level indexer scoring pieces:
    - `indexer_k_quant_and_cache`
    - `cp_gather_indexer_k_quant_cache`
    - `rocm_fp8_mqa_logits`
    - `rocm_fp8_paged_mqa_logits`
    - `_top_k_per_row_prefill`
    - `_top_k_per_row_decode`
- The normal vLLM path still routes through `SparseAttnIndexer`, while the
  ROCm ATOM decode fastpath in `DeepseekV4Indexer._maybe_atom_decode_indexer_fastpath`
  bypasses the generic wrapper for pure decode and reuses ModelState metadata.

Important differences:

- The active decode fastpath is close to ATOM's lower-level kernel sequence,
  but it is not literally ATOM's `indexer_score_topk` dispatcher.
- Prefill still uses the generic vLLM sparse indexer metadata/chunking path.
- The indexer cache remains separate from the main unified KV tensor.
- This makes indexer metadata and top-k translation one of the remaining
  likely conversion-cost sources.

### MLP And MoE

Active or available:

- ATOM's `aiter_silu_and_mul` fallback is used by vLLM's
  `DeepseekV4MLP` when fused clamp/activation is disabled or unavailable.
- ATOM's `fused_clamp_act_mul` path is available in vLLM behind
  `ATOM_USE_AITER_FUSED_CLAMP_ACT_MUL=1` when the installed aiter module
  exposes it.
- vLLM uses its own `FusedMoE` abstraction and weight loader rather than
  ATOM's `FusedMoE` class. This preserves vLLM's loader strategies
  such as safetensors variants and runai-streamer.
- Hash MoE routing is handled through vLLM `FusedMoE` inputs rather than the
  ATOM model-file custom routing hook.

Missing or different:

- ATOM's optional `torch.ops.aiter.maybe_dual_stream_forward` shared/routed
  expert overlap is not active. Installed `aiter==0.1.15.post1` does not expose
  `maybe_dual_stream_forward`, and the ROCm vLLM model currently disables aux
  streams because of previous hang issues.
- This means vLLM has the routed/shared MoE compute components, but not ATOM's
  side-stream overlap behavior.

### mHC

Active or available:

- Installed `aiter==0.1.15.post1` exposes:
    - `mhc_pre`
    - `mhc_post`
- vLLM's ROCm model uses `MHCPreOp` and `MHCPostOp`; when `HAS_AITER_MHC` is
  true it selects the unfused aiter pre/post path.

Missing or different:

- Installed `aiter==0.1.15.post1` does not expose `mhc_fused_post_pre`.
- ATOM checks `getattr(aiter, "mhc_fused_post_pre", None)`, so in this
  environment ATOM would also not use that exact aiter fused post/pre kernel.
- vLLM has a TileLang fused post/pre implementation, but the current ROCm path
  prefers the aiter unfused pre/post path when aiter MHC exists. This is not
  ATOM's ideal fused path, but it avoids the older TileLang ROCm path.

### RoPE And Output Projection

Active or available:

- Installed aiter exposes `rope_cached_positions_2c_fwd_inplace` and
  `rope_cached_positions_fwd_inplace`.
- vLLM's core DSV4 path uses its existing RoPE cache plus fused q/k RoPE kernel
  and the ROCm fused inverse-RoPE/einsum output helper.
- `wo_a` is cached/dequantized to BF16 once in the ROCm helper, matching ATOM's
  practical BF16 grouped output LoRA intent and avoiding per-step FP8 dequant.

Different:

- The exact ATOM `_V4RoPE` wrapper is not copied wholesale into vLLM.
- vLLM keeps the existing model structure and weight loader while using fused
  kernels for the expensive parts.

### Components Present But Still Not Enough For Full ATOM Benefit

The current branch has the main individual components:

- vLLM scheduler and V2 ModelState request-state hook.
- vLLM-owned ATOM-readable unified KV allocation.
- persistent per-request SWA and compressor rings.
- local ATOM-style compressor kernels.
- local ATOM-style paged sparse attention kernels.
- CSA translation and HCA index generation.
- aiter MHC pre/post.
- vLLM fused MoE with DSV4 routing and fast vLLM weight loading.

The remaining performance gap is structural rather than simply missing one
operator:

- indexer prefill and fallback decode still use vLLM generic sparse-indexer
  metadata;
- CSA/HCA page addressing is adapted from vLLM block tables at runtime;
- compressor, indexer, and attention metadata are not yet one native
  ROCm DSV4 contract;
- side-stream compressor and shared-expert overlap are not active;
- the unified tensor is homogeneous BF16 today, not the final mixed-layout
  ATOM storage contract;
- some ATOM model-file calls are replaced by equivalent local vLLM helpers
  rather than literally using ATOM's dispatch wrappers.

Conclusion:

- Yes, we have enough necessary pieces to run a correct vLLM-scheduler preview
  that exercises the main ATOM attention and compressor kernels.
- No, we should not expect all ATOM performance benefit yet. The missing part
  is not just "enable another aiter op"; it is the native ROCm DSV4
  cache/metadata/work scheduling contract that removes adapter kernels,
  duplicate metadata, and lost overlap.
- The next useful implementation work should focus on making the ROCm DSV4
  metadata contract first-class for indexer, compressor, and sparse attention
  together. Narrow optimizations should target device-side CSA top-k
  translation, HCA page-index construction, and indexer prefill metadata before
  attempting aux-stream overlap again.

## Current Validation Run

Date: 2026-06-19.

Server configuration for accuracy:

- `MAX_NUM_SEQS=256`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `MAX_MODEL_LEN=8192`
- `GPU_MEMORY_UTILIZATION=0.9`
- `BLOCK_SIZE=128`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- no `--enforce-eager`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- `VLLM_ROCM_DSV4_ATOM_STATE=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
- `VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN=1`
- `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
- `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`
- `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=1`
- `ATOM_USE_FUSED_Q_NORM_QUANT=1`
- `ATOM_USE_AITER_FUSED_CLAMP_ACT_MUL=1`

Accuracy command:

- `bash lmeval.sh` unchanged.

Accuracy result:

- GSM8K 20-shot flexible exact match: `0.9530 +/- 0.0058`.
- GSM8K 20-shot strict exact match: `0.9538 +/- 0.0058`.
- This passes the target `0.95 +/- 0.01` window.
- Output log: `lmevaldeepseekprographmtp_aitermhc_nobreakablecudagraph.log`.

Server configuration for benchmark:

- Restarted the server after lmeval so KV/cache state was fresh.
- `MAX_NUM_SEQS=32`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `MAX_MODEL_LEN=8192`
- `GPU_MEMORY_UTILIZATION=0.9`
- same ROCm ATOM feature flags as the accuracy run.
- no `--enforce-eager`.

Benchmark command:

- `RESULT_DIR=./bench-current-atom-c32 RESULT_PREFIX=ds-v4-pro-atom-current CONCURRENCIES=32 INPUT_LEN=1024 OUTPUT_LEN=1024 bash benchmarkvllm.sh`

Benchmark result:

- Successful requests: `320`.
- Failed requests: `0`.
- Output throughput: `862.84 tok/s`.
- Total throughput: `1729.05 tok/s`.
- Mean TPOT: `36.18 ms`.
- Median TPOT: `36.21 ms`.
- Mean TTFT: `959.40 ms`.
- Result JSON: `bench-current-atom-c32/ds-v4-pro-atom-current-C32.json`.

Comparison notes:

- This is close to the previous current-branch best vLLM-owned unified-KV
  C32 measurements:
    - default metadata: `865.11 tok/s`, `36.07 ms` mean TPOT.
    - minimal metadata probe: `862.96 tok/s`, `36.14 ms` mean TPOT.
- It is below older pre-unified/adifferent-configuration C32 runs around
  `925-926 tok/s` and well below ATOM's documented C32 target
  (`1145.71 tok/s`, `26.90 ms` mean TPOT).
- The validation supports the earlier conclusion: correctness is good, the
  main ATOM compressor/attention preview is runnable under vLLM scheduler and
  CUDAGraph, but the remaining performance gap is still structural.

## Metadata And Conversion Profiling

Question: could conversion logic and metadata preparation be slowing the
kernel path?

Answer from the current evidence: yes, this overhead is measurable, but it
does not appear large enough to explain the full gap to ATOM's published C32
number by itself.

Existing profiling hooks:

- `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
    - profiles `DeepseekV4RocmAtomModelState.prepare_attn`.
    - breaks total metadata time into:
        - `super`: inherited vLLM attention metadata builder;
        - `unified`: unified-KV allocation/binding check;
        - `plans`: ATOM compressor plan construction;
        - `state`: ATOM request-state metadata construction;
        - `attach`: attaching ATOM state to per-layer metadata objects.
- `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`
    - profiles the ROCm ATOM decode wrapper for one selected layer.
    - breaks time into:
        - `index_ms`;
        - `translate_ms`;
        - `kernel_ms`;
        - `total_ms`.
- `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL=1`
    - profiles ATOM paged prefill with:
        - `build_ms`;
        - `index_ms`;
        - `csa_pack_ms`;
        - `kv_contig_ms`;
        - `kernel_ms`;
        - `output_ms`;
        - `swa_write_ms`.
- `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR=1`
    - profiles the ROCm ATOM main compressor wrapper.
    - use `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_LAYER=<layer>` to choose one
    layer, or `-1` for all layers.
    - use `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_EVERY=<N>` to reduce logging
    frequency.
    - use `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_START_AFTER=<N>` to skip the
    first N calls per compressor module. This is useful because graph warmup can
    otherwise consume the default first-three profile samples before serving
    traffic starts.
    - reports:
        - `prep_ms`: state lookup, metadata lookup, optional HCA block-table
      flattening, dtype conversion, and RoPE cache preparation;
        - `fused_ms`: `fused_compress_attn`;
        - `state_ms`: `update_compressor_states`;
        - `tail_ms`: return-path checks, including prefill/native fallback
      decision;
        - `total_ms`;
        - `num_compress`, `num_write`, `k_per_block`, `num_prefills`, and path
      label.

Observed metadata profile, steady C32 decode (`reqs=32`, `tokens=32`):

- Source logs:
    - `profile_metadata_server.log`.
    - `server_profile_csa_layer2.log`.
- Across sampled workers/steps:
    - inherited vLLM metadata builder: about `0.88-0.91 ms` mean,
    about `0.73-0.74 ms` median.
    - ATOM compressor plans: about `0.15 ms` mean.
    - ATOM request-state metadata: about `0.13 ms` mean.
    - ATOM unified-KV check/bind after startup: about `0.001 ms`.
    - metadata attach: about `0.011-0.013 ms`.
    - total metadata: about `1.17-1.21 ms` mean, about `0.99 ms` median.

Observed mixed/prefill-like C32 metadata profile (`reqs=32`, `tokens=64`):

- total metadata: about `2.8-2.9 ms`.
- most of that increase is still in inherited vLLM metadata construction.

Observed CSA decode wrapper profile for one ratio-4 layer:

- For normal decode sizes after capture:
    - `T=1`: total about `0.15 ms`, translate about `0.05 ms`, kernel about
    `0.09 ms`.
    - `T=16`: total about `0.12-0.13 ms`, translate about `0.05-0.06 ms`,
    kernel about `0.06 ms`.
    - `T=24`: total about `0.15-0.16 ms`, translate about `0.05-0.06 ms`,
    kernel about `0.08-0.09 ms`.
- During capture/profile-heavy paths, `T=32` shows a much larger
  synchronized sample:
    - translate about `2.0-2.3 ms`;
    - kernel about `1.1 ms`;
    - total about `3.1-3.4 ms`.
  This should be treated as a capture/profiling artifact unless reproduced
  during steady-state non-capture serving.

Performance comparison that isolates metadata-level toggles:

- default metadata C32: `865.11 tok/s`, `36.07 ms` mean TPOT.
- minimal metadata C32: `862.96 tok/s`, `36.14 ms` mean TPOT.
- current validation C32: `862.84 tok/s`, `36.18 ms` mean TPOT.

This says the metadata changes tested so far are roughly noise-level for the
full C32 benchmark. They are not zero-cost, but they are not the main
`862 tok/s` versus `1145 tok/s` explanation.

Likely useful next measurements:

- Reproduce decode profiling in steady-state, non-capture C32 with
  `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`,
  `VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY` set high enough to avoid distorting
  throughput.
- Profile HCA ratio-128 layers separately from CSA ratio-4 layers.
- Profile ATOM prefill for the deployment prompt phase with
  `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL=1` and a minimum token threshold, so the
  log captures only large prompt batches.
- Add a lower-overhead event-based timing mode if synchronized `print` profiling
  perturbs the benchmark too much.

Likely useful implementation targets:

- Reduce the inherited vLLM sparse metadata path for ROCm DSV4 pure decode.
  It is the largest measured metadata bucket.
- Avoid CPU/Numpy construction for decode indptrs where the scheduler already
  implies one token per live request.
- Make CSA/HCA page addressing a first-class ROCm DSV4 metadata contract rather
  than translating from generic vLLM block tables at layer time.
- Keep feature flags import-time cached. Current hot paths already mostly do
  this; repeated `os.environ.get` calls are not visible in the inner loops.

### Compressor Profile Probe

Date: 2026-06-19.

Server configuration:

- `MAX_NUM_SEQS=32`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `MAX_MODEL_LEN=8192`
- `GPU_MEMORY_UTILIZATION=0.9`
- `BLOCK_SIZE=128`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- no `--enforce-eager`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- normal ROCm ATOM feature flags from `launchdeepseekgraph.sh`
- profiling:
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_LAYER=-1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_EVERY=100000`

Short client workload:

- `RESULT_DIR=./bench-compressor-profile-short`
- `RESULT_PREFIX=compressor-profile-short`
- `CONCURRENCIES=4`
- `INPUT_LEN=128`
- `OUTPUT_LEN=32`
- `bash benchmarkvllm.sh`

Client result:

- Successful requests: `40`.
- Failed requests: `0`.
- Output throughput: `111.23 tok/s`.
- Mean TPOT: `29.00 ms`.
- Mean TTFT: `250.08 ms`.
- Result JSON:
  `bench-compressor-profile-short/compressor-profile-short-C4.json`.

Important limitation:

- The first-three samples were consumed during graph warmup/capture, before the
  short serving workload.
- The run still proves the profiler works and gives useful per-ratio warmup
  costs, but these numbers should not be reported as clean steady-state serving
  costs.
- After this run, `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_START_AFTER` was
  added so a follow-up run can skip warmup/capture samples.

Captured non-outlier compressor timing samples (`total_ms < 2`) from
`dsv4prographnomtp-aitermhc_nobreakablecudagraph.log`:

- CSA ratio 4:
    - all non-outlier samples: `n=712`, mean total `0.280 ms`, median
    `0.222 ms`, p90 `0.400 ms`.
    - `tokens=16`: mean total `0.218 ms`; `prep_ms=0.042`,
    `fused_ms=0.124`, `state_ms=0.048`.
    - `tokens=32`: mean total `0.375 ms`; `prep_ms=0.190`,
    `fused_ms=0.131`, `state_ms=0.050`.
- HCA ratio 128:
    - all non-outlier samples: `n=736`, mean total `0.217 ms`, median
    `0.157 ms`, p90 `0.334 ms`.
    - `tokens=16`: mean total `0.153 ms`; `prep_ms=0.041`,
    `fused_ms=0.069`, `state_ms=0.040`.
    - `tokens=32`: mean total `0.314 ms`; `prep_ms=0.192`,
    `fused_ms=0.076`, `state_ms=0.041`.

Warmup/JIT outliers:

- First CSA layer-2 samples at `tokens=32` hit about `56-60 ms`, almost all in
  `fused_ms`.
- First HCA layer-0 samples at `tokens=32` hit about `6 ms`, mostly
  `fused_ms` and `state_ms`.
- These align with server JIT warnings, including `_update_compressor_states`,
  and should be treated as first-use warmup cost, not steady serving cost.

Interpretation:

- Compressor-side wrapper/conversion prep is visible. For `tokens=32` warmup
  samples it is about `0.19 ms` per compressor layer, larger than the fused
  kernel time for HCA and comparable to CSA fused time.
- For `tokens=16`, prep falls to about `0.04 ms`, and fused/state kernels are
  the larger share.
- This is enough to justify a steady-state follow-up run with
  `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_START_AFTER` enabled, but the current
  evidence still does not show compressor metadata/conversion as the whole C32
  performance gap.

Follow-up start-after serving probe:

- Server added:
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_START_AFTER=20`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_EVERY=1`
- Short client workload:
    - `RESULT_DIR=./bench-compressor-profile-steady-short`
    - `RESULT_PREFIX=compressor-profile-steady-short`
    - `CONCURRENCIES=4`
    - `INPUT_LEN=128`
    - `OUTPUT_LEN=32`
- Client result:
    - Successful requests: `40`.
    - Failed requests: `0`.
    - Output throughput: `107.76 tok/s`.
    - Total throughput: `552.25 tok/s`.
    - Mean TPOT: `30.15 ms`.
    - Mean TTFT: `250.44 ms`.
    - Result JSON:
    `bench-compressor-profile-steady-short/compressor-profile-steady-short-C4.json`.
- Parsed profile lines:
    - `4392` rows, all `path=atom_prefill`.
    - The probe did not capture `path=atom_decode` samples. It therefore measures
    serving-time prefill compressor overhead after graph warmup, not decode
    compressor overhead.
    - CSA ratio 4: `n=2160`, mean total `0.230 ms`, p50 `0.230 ms`, p90
    `0.248 ms`, p99 `0.261 ms`, max `0.280 ms`.
        - Mean breakdown: `prep_ms=0.039`, `fused_ms=0.129`,
      `state_ms=0.058`, `tail_ms=0.004`.
    - HCA ratio 128: `n=2232`, mean total `0.222 ms`, p50 `0.221 ms`, p90
    `0.236 ms`, p99 `0.291 ms`, max `0.355 ms`.
        - Mean breakdown: `prep_ms=0.040`, `fused_ms=0.131`,
      `state_ms=0.047`, `tail_ms=0.004`.

Implication:

- Compressor wrapper preparation is real but small in the captured serving-time
  prefill path, around `0.04 ms` per compressor layer call.
- Fused compression plus compressor-state update remain the larger captured
  compressor buckets, around `0.18-0.19 ms` combined per layer call.
- This does not rule out metadata/conversion as an end-to-end performance issue.
  The larger suspects are the decode/indexer/attention preparation path:
  `DeepseekV4RocmAtomModelState._build_atom_state_metadata()` still builds
  `scheduled`, `computed`, `batch_id_per_token`, `positions`, committed counts,
  and decode indptrs through CPU/Numpy arrays followed by H2D copies; sparse MLA
  metadata still builds `req_id_per_token` with Numpy and launches helper kernels
  for compressed slot mappings and C128A/HCA index tables.

### Metadata Profile Probe

Date: 2026-06-19.

Purpose:

- Test the hypothesis that metadata preparation and conversion, not only the
  attention/compressor kernels, are a meaningful part of the remaining
  performance gap versus ATOM.
- Instrumented:
    - `DeepseekV4RocmAtomModelState.prepare_attn()`.
    - `DeepseekV4RocmAtomModelState._build_atom_state_metadata()`.
    - `DeepseekV4FlashMLAMetadataBuilder.build()`.
    - Existing ROCm `DeepseekV4ROCMAiterMLASparseMetadataBuilder` and
    `DeepseekV4ROCMAiterSparseSWAMetadataBuilder` profile rows.

Server configuration:

- `MAX_NUM_SEQS=32`
- `MAX_NUM_BATCHED_TOKENS=8192`
- `MAX_MODEL_LEN=8192`
- `GPU_MEMORY_UTILIZATION=0.9`
- `BLOCK_SIZE=128`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- no `--enforce-eager`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- profiling:
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_START_AFTER=20`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_EVERY=1`

Short client workload:

- `RESULT_DIR=./bench-metadata-profile-short`
- `RESULT_PREFIX=metadata-profile-short`
- `CONCURRENCIES=4`
- `INPUT_LEN=128`
- `OUTPUT_LEN=32`
- `bash benchmarkvllm.sh`

Client result:

- Successful requests: `40`.
- Failed requests: `0`.
- Output throughput: `111.27 tok/s`.
- Total throughput: `570.25 tok/s`.
- Mean TPOT: `29.74 ms`.
- Mean TTFT: `225.42 ms`.
- Result JSON:
  `bench-metadata-profile-short/metadata-profile-short-C4.json`.

Important limitation:

- `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1` enables synchronization in the
  existing ROCm backend metadata profilers. The client throughput from this run
  is diagnostic only and should not be compared to normal benchmark runs.

Parsed model-state rows from
`dsv4prographnomtp-aitermhc_nobreakablecudagraph.log`:

- Rows: `2944`.
- Likely decode rows (`tokens == reqs`): `2784`.
- Model-specific ATOM state conversion is small:
    - `_build_atom_state_metadata()` likely decode rows: mean `0.095 ms`, p50
    `0.094 ms`, p90 `0.101 ms`, p99 `0.109 ms`, max `0.169 ms`.
    - Breakdown: `map_batch` mean `0.021 ms`, `pos_commit` mean `0.019 ms`,
    `dataclass` mean `0.015 ms`, `decode_indptr` mean `0.040 ms`.
- The inherited `super().prepare_attn()` metadata path dominates:
    - likely decode rows: mean `0.894 ms`, p50 `0.807 ms`, p90 `0.842 ms`,
    p99 `0.987 ms`, max `28.869 ms`.
    - total `prepare_attn()` likely decode rows: mean `1.163 ms`, p50
    `1.073 ms`, p90 `1.116 ms`, p99 `1.302 ms`, max `29.152 ms`.
    - The `28-29 ms` outliers occur at call `200`, shape `(reqs=4,tokens=4)`,
    across workers, so treat them as first-use/JIT/synchronization artifacts
    until reproduced without synchronized profiling.

Parsed sparse MLA builder rows:

- Ratio 4:
    - `2944` rows.
    - total mean `0.060 ms`, p50 `0.059 ms`, p90 `0.063 ms`, max `0.104 ms`.
    - `req_ids` mean `0.029 ms`, compressed slot mean `0.029 ms`.
- Ratio 128:
    - `2944` rows.
    - total mean `0.099 ms`, p50 `0.097 ms`, p90 `0.106 ms`, p99
    `0.142 ms`, max `0.222 ms`.
    - `req_ids` mean `0.030 ms`, compressed slot mean `0.036 ms`,
    C128A metadata mean `0.030 ms`.

Parsed existing synchronized ROCm backend metadata rows, excluding rows with
`total_ms >= 5` as warmup/JIT outliers:

- HCA/MLA ratio 128:
    - clean rows: `23`.
    - total mean `0.887 ms`, p50 `0.333 ms`, p90 `2.169 ms`, max `2.360 ms`.
    - base mean `0.777 ms`; ragged conversion mean `0.109 ms`.
- CSA/MLA ratio 4:
    - rows: `32`.
    - total mean `0.150 ms`, p50 `0.104 ms`, p90 `0.277 ms`, max
    `0.334 ms`.
- SWA:
    - clean rows: `56`.
    - total mean `0.241 ms`, p50 `0.197 ms`, p90 `0.371 ms`, max
    `0.471 ms`.
    - base mean `0.162 ms`; ragged conversion mean `0.078 ms`.

Conclusion:

- Yes, metadata preparation/conversion can materially slow the end-to-end
  path even when the ATOM kernel itself is fast.
- The model-specific ATOM state conversion is not currently the dominant
  metadata cost. It is about `0.1 ms` per prepare call in this C4 probe.
- The inherited vLLM/ROCm sparse MLA + SWA metadata builders are the larger
  bucket. A likely-decode prepare call spends about `0.8-0.9 ms` in
  `super().prepare_attn()`, before ATOM model state is attached.
- Next integration target: for pure decode, avoid building full generic
  `DeepseekV4ROCMAiterMLASparseMetadata` and
  `DeepseekV4ROCMAiterSparseSWAMetadata` when ATOM decode only needs the
  scheduler block table/slot mapping, block size, ATOM state, and possibly the
  HCA block table. Build a lightweight ROCm DSV4 decode metadata object in
  `DeepseekV4RocmAtomModelState` and teach the ROCm attention path to consume
  it. This is closer to ATOM's unified request-state contract and should reduce
  the inherited ragged/dense index generation that is not needed by the direct
  ATOM decode kernels.

### Pure Decode Metadata Bypass

Date: 2026-06-19.

Change:

- Added a ROCm-only pure-decode metadata bypass in
  `DeepseekV4RocmAtomModelState`.
- New flag:
    - `VLLM_ROCM_DSV4_ATOM_SKIP_DECODE_METADATA`, default enabled.
- The bypass only triggers when all of these are true:
    - ROCm platform.
    - `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`.
    - `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`.
    - one scheduled token per live request.
    - no ATOM attention layer/ratio filtering.
    - no native HCA index fallback.
    - no probe/forced-native fallback flags.
- It removes these backend groups from the generic metadata build for that
  scheduler step:
    - `DEEPSEEK_SPARSE_SWA`
    - `FLASHMLA_SPARSE_DSV4`
    - `ROCM_FLASHMLA_SPARSE_DSV4`
- It then attaches lightweight metadata objects containing only the fields the
  ATOM decode path uses:
    - SWA: block table, slot mapping, SWA block size, sequence lens, query start,
    decode counts.
    - MLA: block table, slot mapping, compressed block size, query start,
    request id buffer, topk width.

Validation run:

- Server:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `GPU_MEMORY_UTILIZATION=0.9`
    - `BLOCK_SIZE=128`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - no `--enforce-eager`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - profiling:
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_START_AFTER=20`
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_EVERY=1`
- Client:
    - `RESULT_DIR=./bench-metadata-bypass-short`
    - `RESULT_PREFIX=metadata-bypass-short`
    - `CONCURRENCIES=4`
    - `INPUT_LEN=128`
    - `OUTPUT_LEN=32`
    - `bash benchmarkvllm.sh`
- Result:
    - Successful requests: `40`.
    - Failed requests: `0`.
    - Output throughput: `109.37 tok/s`.
    - Total throughput: `560.52 tok/s`.
    - Mean TPOT: `30.72 ms`.
    - Mean TTFT: `213.93 ms`.
    - Result JSON:
    `bench-metadata-bypass-short/metadata-bypass-short-C4.json`.

Important limitation:

- As with the previous metadata profile probe, this run used synchronized
  metadata profiling and is not a clean throughput benchmark.

Parsed metadata effect:

- ModelState profile rows: `2944`.
- `skip_decode=True` rows: `2784`.
- `skip_decode=False` rows: `160`, all prefill/mixed shapes.
- Generic sparse MLA builder profile rows dropped from the previous probe's
  `5888` rows to `48` rows because pure decode no longer builds those generic
  MLA metadata objects.
- Likely decode (`tokens == reqs`) with bypass:
    - `super().prepare_attn()` mean `0.257 ms`, p50 `0.252 ms`, p90 `0.265 ms`,
    p99 `0.383 ms`, max `0.642 ms`.
    - total `prepare_attn()` mean `0.559 ms`, p50 `0.547 ms`, p90 `0.568 ms`,
    p99 `0.881 ms`, max `1.221 ms`.
- Previous likely decode without bypass:
    - `super().prepare_attn()` mean `0.894 ms`, p50 `0.807 ms`, p90 `0.842 ms`,
    p99 `0.987 ms`, max `28.869 ms`.
    - total `prepare_attn()` mean `1.163 ms`, p50 `1.073 ms`, p90 `1.116 ms`,
    p99 `1.302 ms`, max `29.152 ms`.

Interpretation:

- The lightweight decode metadata bypass roughly halves profiled
  `prepare_attn()` time for pure decode in this short C4 diagnostic:
  p50 `1.073 ms -> 0.547 ms`.
- It also removes almost all pure-decode sparse MLA metadata builder work.
- The remaining likely-decode metadata buckets are now mostly compress-plan
  build, ATOM state metadata, and minimal indexer/cache attachment.
- The next verification step is a no-profiling C32 benchmark after a clean
  server restart, because synchronized metadata profiling distorts throughput.

No-profiling C32 benchmark:

- Server:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `GPU_MEMORY_UTILIZATION=0.9`
    - no metadata/compressor profiling
    - no `--enforce-eager`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- Client:
    - `RESULT_DIR=./bench-metadata-bypass-c32`
    - `RESULT_PREFIX=metadata-bypass-c32`
    - `CONCURRENCIES=32`
    - `INPUT_LEN=1024`
    - `OUTPUT_LEN=1024`
    - `bash benchmarkvllm.sh`
- Result:
    - Successful requests: `320`.
    - Failed requests: `0`.
    - Output throughput: `865.20 tok/s`.
    - Total throughput: `1733.78 tok/s`.
    - Mean TPOT: `35.98 ms`.
    - Mean TTFT: `1056.72 ms`.
    - Result JSON:
    `bench-metadata-bypass-c32/metadata-bypass-c32-C32.json`.
- Comparison:
    - Previous clean current-branch C32 validation:
    `862.84 tok/s`, total `1729.05 tok/s`, mean TPOT `36.18 ms`.
    - Previous best nearby metadata run:
    `865.11 tok/s`, total `1733.60 tok/s`, mean TPOT `36.07 ms`.
    - The bypass is runtime-safe in this benchmark and slightly improves or
    matches current clean throughput, but the gain is small at full C32 because
    GPU decode work dominates. The profiler still shows the CPU/metadata path
    is meaningfully reduced.

Accuracy validation:

- Server:
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `GPU_MEMORY_UTILIZATION=0.9`
    - `BLOCK_SIZE=128`
    - no metadata/compressor profiling
    - no `--enforce-eager`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- Client:
    - unchanged `bash lmeval.sh`
- Result:
    - GSM8K flexible exact match: `0.9507 +/- 0.0060`.
    - GSM8K strict exact match: `0.9515 +/- 0.0059`.
- Conclusion:
    - The pure-decode metadata bypass remains inside the required GSM8K
    `0.95 +/- 0.01` accuracy band.

Follow-up: main compressor metadata bypass:

- Date: 2026-06-19.
- Change:
    - Added `VLLM_ROCM_DSV4_ATOM_SKIP_COMPRESSOR_METADATA`, default enabled.
    - This only applies to pure one-token decode when:
        - ROCm platform.
        - ATOM attention is enabled.
        - ATOM unified KV is enabled.
        - ATOM main compressor is enabled.
        - native-after-main-compressor fallback is disabled.
        - no ATOM layer/ratio filtering or probe/fallback flags are active.
    - It removes only main compressor `CompressorBackend` groups from the
    generic metadata build. The discriminator is `kv_cache_spec.head_size >
    512`:
        - main CSA state-cache: `2048`
        - main HCA state-cache: `1024`
        - indexer-inner compressor state-cache: `512`, not skipped
    - Minimal `DeepseekV4RocmAtomCompressorDecodeMetadata` objects are attached
    for the skipped main compressor state-cache layer names, then annotated
    with `deepseek_v4_rocm_atom_state` like the other minimal metadata objects.

- Reasoning:
    - The ATOM main compressor path uses the compressor metadata object only as
    a carrier for `deepseek_v4_rocm_atom_state`.
    - Generic `CompressorMetadata` fields are still required by the native
    compressor fallback and by the indexer-inner compressor cache writer.
    - This removes one more vLLM conversion/metadata-prep layer in front of the
    ATOM decode path without changing the CUDA path.

- Validation status:
    - Syntax check passed with:
    `python3 -m py_compile vllm/models/deepseek_v4/amd/model_state.py`.
    - Short metadata-profile smoke:
        - Server:
            - `MAX_NUM_SEQS=32`
            - `MAX_NUM_BATCHED_TOKENS=8192`
            - `MAX_MODEL_LEN=8192`
            - `GPU_MEMORY_UTILIZATION=0.9`
            - metadata profiling enabled with start-after `20`, every `1`
            - no `--enforce-eager`
            - `VLLM_USE_V2_MODEL_RUNNER=1`
            - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
        - Client:
            - `RESULT_DIR=./bench-compressor-metadata-bypass-short`
            - `RESULT_PREFIX=compressor-metadata-bypass-short`
            - `CONCURRENCIES=4`
            - `INPUT_LEN=128`
            - `OUTPUT_LEN=32`
            - `bash benchmarkvllm.sh`
        - Result:
            - Successful requests: `40`.
            - Failed requests: `0`.
            - Output throughput: `109.91 tok/s`.
            - Total throughput: `563.28 tok/s`.
            - Mean TPOT: `28.84 ms`.
            - Result JSON:
        `bench-compressor-metadata-bypass-short/compressor-metadata-bypass-short-C4.json`.
        - Parsed metadata rows:
            - `skip_decode=True, skip_compressor=True`: `2784`.
            - `skip_decode=True, skip_compressor=False`: `0`.
            - `skip_decode=False`: `160`.
        - Pure decode timing:
            - `super().prepare_attn()` mean `0.199 ms`, min `0.183 ms`,
        max `0.550 ms`.
            - total `prepare_attn()` mean `0.505 ms`, min `0.470 ms`,
        max `1.180 ms`.
            - plan build mean `0.120 ms`.
            - ATOM state build mean `0.137 ms`.
            - indexer attach mean `0.039 ms`.
    - Full accuracy validation:
        - Server:
            - `MAX_NUM_SEQS=256`
            - `MAX_NUM_BATCHED_TOKENS=8192`
            - `MAX_MODEL_LEN=8192`
            - `GPU_MEMORY_UTILIZATION=0.9`
            - no metadata/compressor profiling
            - no `--enforce-eager`
            - `VLLM_USE_V2_MODEL_RUNNER=1`
            - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
        - Client:
            - unchanged `bash lmeval.sh`
        - Result:
            - GSM8K flexible exact match: `0.9522 +/- 0.0059`.
            - GSM8K strict exact match: `0.9530 +/- 0.0058`.
        - Conclusion:
            - The main compressor metadata bypass remains inside the required GSM8K
        `0.95 +/- 0.01` accuracy band.
    - No-profiling C32 benchmark after a fresh server restart:
        - Server:
            - `MAX_NUM_SEQS=32`
            - `MAX_NUM_BATCHED_TOKENS=8192`
            - `MAX_MODEL_LEN=8192`
            - `GPU_MEMORY_UTILIZATION=0.9`
            - no metadata/compressor profiling
            - no `--enforce-eager`
            - `VLLM_USE_V2_MODEL_RUNNER=1`
            - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
        - Client:
            - `RESULT_DIR=./bench-compressor-metadata-bypass-c32`
            - `RESULT_PREFIX=compressor-metadata-bypass-c32`
            - `CONCURRENCIES=32`
            - `INPUT_LEN=1024`
            - `OUTPUT_LEN=1024`
            - `bash benchmarkvllm.sh`
        - Result:
            - Successful requests: `320`.
            - Failed requests: `0`.
            - Output throughput: `869.40 tok/s`.
            - Total throughput: `1742.19 tok/s`.
            - Mean TPOT: `36.00 ms`.
            - Mean TTFT: `850.12 ms`.
            - Result JSON:
        `bench-compressor-metadata-bypass-c32/compressor-metadata-bypass-c32-C32.json`.
        - Comparison:
            - Sparse decode metadata bypass only:
        `865.20 tok/s`, total `1733.78 tok/s`, mean TPOT `35.98 ms`.
            - Current clean validation before these bypasses:
        `862.84 tok/s`, total `1729.05 tok/s`, mean TPOT `36.18 ms`.
            - The additional main-compressor metadata bypass produces a small
        end-to-end C32 throughput gain, but TPOT remains essentially flat. The
        profile improvement is real host-side work removal; it is not the main
        remaining gap to ATOM's documented C32 target.

### ROCm MHC Aiter Selection

Date: 2026-06-19.

Finding:

- ATOM's DSV4 model binds `aiter.mhc_pre`, `aiter.mhc_post`, and, when
  available, `aiter.mhc_fused_post_pre`.
- In this environment, `aiter==0.1.15.post1` exposes:
    - `aiter.mhc_pre`: present.
    - `aiter.mhc_post`: present.
    - `aiter.mhc_fused_post_pre`: not present.
- vLLM already had registered wrappers for the standalone aiter MHC kernels,
  but `vllm.model_executor.layers.mhc.HAS_AITER_MHC` was hardcoded to
  `False`. As a result, the AMD DSV4 layer selected the tilelang fused
  post+pre path whenever tilelang was installed, even though the comments said
  aiter standalone MHC was the preferred ROCm path.
- Direct standalone-kernel probes looked numerically close for a representative
  DSV4 shape, but the full model accuracy did not survive enabling that path.

Change:

- `HAS_AITER_MHC` now checks the installed `vllm._aiter_ops.rocm_aiter_ops`
  wrapper for both `mhc_pre` and `mhc_post` only when
  `VLLM_ROCM_DSV4_USE_AITER_MHC=1`.
- The default remains `HAS_AITER_MHC=False`, which preserves the previously
  validated MHC path.
- No attempt is made to enable aiter fused post+pre because the installed
  aiter package does not expose `mhc_fused_post_pre`.

Validation:

- Syntax:
    - `python3 -m py_compile vllm/model_executor/layers/mhc.py
    vllm/models/deepseek_v4/amd/model.py`
- Import probe:
    - default: `HAS_AITER_MHC=False`
    - opt-in with `VLLM_ROCM_DSV4_USE_AITER_MHC=1`: `HAS_AITER_MHC=True`
    - `HAS_TILELANG=True`
    - `aiter.mhc_pre=True`
    - `aiter.mhc_post=True`
    - `aiter.mhc_fused_post_pre=False`
- Direct DSV4-shaped kernel sanity check, `M=4`, `hc=4`, `hidden=7168`:
    - `mhc_pre_aiter` versus torch reference:
        - post max abs `7.22e-05`, mean abs `2.54e-05`
        - comb max abs `4.54e-05`, mean abs `1.05e-05`
        - layer input max abs `0.015625`, mean abs `7.07e-05`
    - `mhc_post_aiter` versus torch reference:
        - output max abs `0.015625`, mean abs `1.83e-05`
    - A smaller hidden-size probe with `hidden=1024` aborted inside aiter with
    `hidden_size must be >= residual_block * 2 stages prefetch`; that shape is
    not representative of DSV4.
- Serving smoke:
    - Server:
        - `MAX_NUM_SEQS=4`
        - `MAX_NUM_BATCHED_TOKENS=1024`
        - `MAX_MODEL_LEN=2048`
        - `GPU_MEMORY_UTILIZATION=0.6`
        - no `--enforce-eager`
        - `VLLM_USE_V2_MODEL_RUNNER=1`
        - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - Graph capture completed.
    - A short `/v1/completions` request completed successfully.
- Full GSM8K with standalone aiter MHC enabled:
    - Server:
        - `MAX_NUM_SEQS=256`
        - `MAX_NUM_BATCHED_TOKENS=8192`
        - `MAX_MODEL_LEN=8192`
        - `GPU_MEMORY_UTILIZATION=0.9`
        - no `--enforce-eager`
        - `VLLM_USE_V2_MODEL_RUNNER=1`
        - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
        - `VLLM_ROCM_DSV4_USE_AITER_MHC=1`
    - Client:
        - unchanged `bash lmeval.sh`
    - Result:
        - flexible exact match: `0.1296 +/- 0.0093`
        - strict exact match: `0.1168 +/- 0.0088`
    - Conclusion:
        - Standalone aiter `mhc_pre`/`mhc_post` are not accuracy-safe in the
      current vLLM AMD DSV4 MHC invocation path.
- Full GSM8K after restoring the default-gated MHC path:
    - Server:
        - `MAX_NUM_SEQS=256`
        - `MAX_NUM_BATCHED_TOKENS=8192`
        - `MAX_MODEL_LEN=8192`
        - `GPU_MEMORY_UTILIZATION=0.9`
        - no `--enforce-eager`
        - `VLLM_USE_V2_MODEL_RUNNER=1`
        - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
        - default `VLLM_ROCM_DSV4_USE_AITER_MHC=0`
    - Client:
        - unchanged `bash lmeval.sh`
    - Result:
        - flexible exact match: `0.9530 +/- 0.0058`
        - strict exact match: `0.9538 +/- 0.0058`
- C32 deployment benchmark after restoring the default-gated MHC path:
    - Server:
        - `MAX_NUM_SEQS=32`
        - `MAX_NUM_BATCHED_TOKENS=8192`
        - `MAX_MODEL_LEN=8192`
        - `GPU_MEMORY_UTILIZATION=0.9`
        - no `--enforce-eager`
        - `VLLM_USE_V2_MODEL_RUNNER=1`
        - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
        - default `VLLM_ROCM_DSV4_USE_AITER_MHC=0`
    - Client:
        - `RESULT_DIR=./bench-mhc-default-c32`
        - `RESULT_PREFIX=mhc-default-c32`
        - `CONCURRENCIES=32`
        - `INPUT_LEN=1024`
        - `OUTPUT_LEN=1024`
        - `bash benchmarkvllm.sh`
    - Result:
        - Successful requests: `320`.
        - Failed requests: `0`.
        - Output throughput: `869.70 tok/s`.
        - Total throughput: `1742.79 tok/s`.
        - Mean TPOT: `35.87 ms`.
        - Mean TTFT: `978.97 ms`.
        - Result JSON:
      `bench-mhc-default-c32/mhc-default-c32-C32.json`.

Status:

- Default MHC selection remains the accuracy-safe path.
- Standalone aiter MHC is kept as an explicit experiment gate, not a default
  optimization.
- `mhc_fused_post_pre` cannot be enabled with `aiter==0.1.15.post1` because
  that symbol is absent.

### Profiling Metadata And Conversion Overhead

Date: 2026-06-19.

Question:

- A faster sparse attention kernel can fail to improve deployment throughput if
  vLLM-side metadata preparation, index translation, KV layout conversion, or
  per-layer packing remains on the critical path.

Current instrumentation:

- `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
    - Logs model-state preparation cost in `DeepseekV4RocmAtomModelState`.
    - Splits the per-forward time into generic vLLM attention metadata
    construction, unified-KV allocation/binding, compress-plan construction,
    ATOM request-state metadata, minimal metadata attachment, and annotation.
    - The detailed state-metadata log splits request mapping, position/committed
    count preparation, padding, dataclass copies, and decode-indptr preparation.
- `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`
    - Logs selected-layer ATOM decode timing.
    - Splits decode into index construction, CSA/HCA translation/packing, kernel
    execution, and total time.
    - The log now also includes per-forward index-cache counters:
    `idx_hits`, `idx_writes`, `hca_hits`, and `hca_writes`.
- `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL=1`
    - Logs selected-layer ATOM prefill timing.
    - Splits prefill into indptr build, index construction, CSA packing,
    `kv_full.contiguous()`, attention kernel, output copy, SWA write, and total
    time.
    - The log now also includes per-forward prefill reuse counters:
    `indptr_hits`, `indptr_writes`, `idx_hits`, `idx_writes`, `hca_hits`, and
    `hca_writes`.

Smoke validation:

- Server:
    - `MAX_NUM_SEQS=4`
    - `MAX_NUM_BATCHED_TOKENS=1024`
    - `MAX_MODEL_LEN=2048`
    - `GPU_MEMORY_UTILIZATION=0.6`
    - no `--enforce-eager`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - profiling flags enabled for metadata, decode, and prefill.
- Request:
    - `/v1/completions`
    - prompt: `Q: What is 2+2?\nA:`
    - `max_tokens=8`
    - `temperature=0`
- Result:
    - Request completed successfully.
    - Example metadata profile after startup for one request:
    `super=5.468ms`, `plans=0.182ms`, `state=0.196ms`,
    `total=5.868ms`.
    - Example prefill profile:
    `build_ms=0.192`, `index_ms=0.117`, `kv_contig_ms=0.009`,
    `kernel_ms=0.122`, `swa_write_ms=1.152`, `total_ms=1.620`,
    `indptr_writes=1`, `idx_writes=1`, `hca_writes=1`.
    - Later decode-only metadata steps for the same request were about
    `0.49-0.84ms` total in this small smoke.
    - The smoke also showed runtime JIT warnings for shapes not covered by the
    small warmup; ignore those timings for final performance conclusions.

C32 short deployment probe:

- Server:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `GPU_MEMORY_UTILIZATION=0.9`
    - no `--enforce-eager`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - profiling flags:
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_START_AFTER=64`
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_EVERY=128`
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL=1`
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=0`
        - `VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=512`
- Client:
    - direct `vllm bench serve`
    - `--input-len 1024`
    - `--output-len 1024`
    - `--num-prompts 64`
    - `--max-concurrency 32`
    - `--num-warmups 32`
- Result:
    - Successful requests: `64`.
    - Failed requests: `0`.
    - Output throughput: `865.53 tok/s`.
    - Total throughput: `1734.44 tok/s`.
    - Mean TPOT: `35.75 ms`.
    - Result JSON:
    `bench-profile-c32-short/profile-c32-short-C32.json`.
- Parsed profile log:
    - Model-state metadata, sampled after startup:
        - `super`: avg `0.193ms`, max `0.222ms`.
        - `plans`: avg `0.118ms`, max `0.140ms`.
        - ATOM state metadata: avg `0.177ms`, max `0.262ms`.
        - total: avg `0.538ms`, max `0.654ms`.
    - State metadata detail:
        - request/batch mapping: avg `0.021ms`.
        - position/committed-count prep: avg `0.035ms`.
        - decode indptr prep: avg `0.039ms`.
        - total: avg `0.110ms`.
    - Backend metadata sampled in the same run:
        - `mla:128`: avg `2.522ms`, max `5.422ms`.
        - `mla:4`: avg `0.131ms`, max `0.159ms`.
        - `swa`: avg `3.612ms`, max `17.569ms`.
    - Layer-0 ATOM prefill samples:
        - `build_ms`: avg `0.253ms`.
        - `index_ms`: avg `1.213ms`.
        - `kernel_ms`: avg `1.693ms`.
        - `swa_write_ms`: avg `1.058ms`.
        - total: avg `4.283ms`.
        - `indptr_writes=1`, `idx_writes=1`, `hca_writes=1`, and no reuse hits
      in the sampled records.
    - Layer-0 ATOM decode split samples were emitted during graph capture, not
    during graph replay:
        - `index_ms`: avg `1.276ms`.
        - `kernel_ms`: avg `1.022ms`.
        - total: avg `2.304ms`.
        - This is still useful for shape-level cost comparison, but it should not
      be treated as a direct runtime decode measurement under CUDAGraph replay.

Caveat:

- With breakable CUDAGraph enabled, Python per-layer logging only runs while
  capturing graphs or on paths not replayed from graph. Runtime deployment
  metadata logs are valid because metadata is prepared outside graph replay.
  Runtime per-layer kernel split logs are not emitted for graph-replayed decode
  steps. To directly time replayed kernels, use external GPU profiling, CUDA/HIP
  events captured into graph-safe buffers, or a diagnostic no-graph run. Do not
  use a no-graph run as the final performance number.

Recommended deployment probe:

- Start the server with the same deployment configuration as C32 benchmark plus:
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_START_AFTER=16`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_EVERY=64`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=0`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=256`
- Run a short C32 random benchmark first, then a full C32 run if the log volume
  is manageable.

How to interpret:

- If `kernel_ms` is small but `index_ms`, `translate_ms`, `csa_pack_ms`, or
  `kv_contig_ms` are comparable or larger, the current bottleneck is outside
  the sparse attention kernel.
- If `super=...ms` dominates the metadata profile, the remaining cost is in
  generic vLLM metadata construction and can only be removed by skipping or
  replacing more backend metadata groups.
- If `idx_writes` stays high across many layers for the same forward, index
  reuse is not covering the current path. If `idx_hits` rises after the first
  layer, per-forward reuse is working and the remaining cost is likely kernel
  or layout conversion.

### ATOM Kernel Coverage Audit

Current default launch shape for these notes:

- ROCm-only DSV4 path.
- vLLM scheduler remains active.
- vLLM V2 model runner is enabled with `VLLM_USE_V2_MODEL_RUNNER=1`.
- vLLM-owned unified KV is enabled with
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`.
- No `--enforce-eager`.

Coverage status:

| ATOM modeling-file area | vLLM ROCm status | Evidence / caveat |
| --- | --- | --- |
| Main CSA/HCA compressor `fused_compress_attn` + `update_compressor_states` | Enabled | `DeepseekCompressor._maybe_atom_main_compressor_forward` dispatches the ATOM-style fused compressor before state update for `head_dim=512`, with ATOM compress plans and model-state rings. Full GSM8K passed in earlier runs. |
| Indexer-inner compressor `fused_compress_attn` quant path | Wired but gated off by default | The same ATOM fused compressor path accepts `head_dim=128` FP8 indexer caches behind `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`. Pure-decode ModelState metadata now carries the compressed block table and block size so this path can use block-table scatter and become eligible for the aiter/flydsl fused compressor wrapper. Mixed/prefill still falls back to direct `kv_slot_mapping` scatter. MXFP4 indexer cache still falls back to native. A large-batch lmeval attempt with this enabled previously crashed from ROCm out-of-resources/JIT pressure before accuracy could be measured, so the launch default is `0`. |
| Compressor state-update ordering | ATOM-style inside the ATOM compressor path | Fused compression reads previous state, then `update_compressor_states` writes current state. This is the read-before-update ordering validated for ROCm. |
| ATOM compressor-first attention ordering | Not default | User requested reverting compressor-order experiments. Current default uses the vLLM attention ordering with ATOM compressor side effects, not full ATOM async compressor-first scheduling. |
| `qk_norm_rope_maybe_quant` | Enabled | ROCm attention path calls the ATOM-style q/k norm + RoPE helper when `ATOM_DISABLE_QK_ROPE` is not set. |
| Fused Q norm / quant | Enabled by default | `ATOM_USE_FUSED_Q_NORM_QUANT=1` is default in the launch script. This is separate from compressor ordering; it does not mathematically require ATOM compressor ordering. |
| SWA write | Enabled for ATOM attention | ATOM-style `swa_write` is used for the unified SWA ring. Native SWA cache writes may still exist for compatibility in some paths. |
| Sparse paged decode / prefill | Enabled | `sparse_attn_v4_paged_decode` and `sparse_attn_v4_paged_prefill` wrappers are wired in `amd/rocm.py`. Current fastest validated run used this ATOM attention path, not the old vLLM sparse attention kernel. |
| CSA translate / pack | Enabled | `csa_translate_pack` is used before ATOM sparse attention for CSA layers, matching the ATOM modeling-file op sequence. |
| Direct CSA/HCA decode kernels | Removed | The vLLM-only direct CSA/HCA decode experiments were removed from the current code path because they bypassed `csa_translate_pack` / decode-index materialization and were not ATOM modeling-file ops. |
| Indexer Q-side op sequence | Partial | vLLM uses `fused_indexer_q_rope_quant`; ATOM modeling does `wq_b`, `rope_rotate_activation`, HIP FP8 quant, `scale_indexer_weights`, then `torch.ops.aiter.indexer_score_topk`. Math and downstream kernels are close, but the exact op sequence is not identical. |
| Indexer score/top-k kernels | Mostly aligned | vLLM ROCm path uses paged FP8 MQA logits and aiter top-k wrappers. The direct decode fastpath can bypass generic indexer metadata for pure decode. |
| MHC / HC | Not fully ATOM | Installed `aiter==0.1.15.post1` exposes standalone `mhc_pre`/`mhc_post` but not `mhc_fused_post_pre`. Opt-in standalone aiter MHC failed full GSM8K, so the validated default remains vLLM's existing MHC path. HC head is not switched to ATOM's `aiter.mhc_pre` path by default. |
| MoE | Partial | vLLM `FusedMoE` is used, with ROCm aiter backend behavior where available. ATOM's exact `atom.model_ops.moe.FusedMoE`, shared-expert dual-stream path, and TBO stream overlap are not ported. `fused_clamp_act_mul` is enabled for the MLP/shared-expert activation path when available. |
| Auxiliary streams / overlap | Not default | ROCm aux streams were disabled again after earlier experiments. ATOM's `maybe_compressors_async`, shared-expert stream overlap, and comm-stream overlap remain missing performance work. |
| True ATOM unified KV allocator | Partial | vLLM owns the KV allocation and the ROCm path binds ATOM unified views into it. This preserves vLLM scheduler/KV ownership but still pays some vLLM layout and metadata costs. A pure ATOM allocation/ring model would require deeper KV-cache spec/allocation/backend changes. |

Immediate validation needed after the indexer-inner compressor patch:

- Start the normal accuracy server with default ATOM flags, no
  `--enforce-eager`, and no standalone aiter MHC.
- Run unchanged `lmeval.sh`; GSM8K must remain `0.95 +/- 0.01`.
- If accuracy passes, rerun C32 performance after a clean server restart.
- To test the gated indexer-inner path, set
  `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`.
- If the gated path fails, isolate with:
    - `VLLM_ROCM_DSV4_ATOM_SKIP_FUSED_COMPRESS=1`
    - a temporary gate for `head_dim=128` only, if added
    - compressor profile logs on one CSA layer to verify indexer cache writes.

Local sanity completed for the indexer-inner compressor patch:

- `python3 -m py_compile` passed for:
    - `vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py`
    - `vllm/models/deepseek_v4/compressor.py`
- Synthetic ROCm launch passed for the new `head_dim=128`, `quant=True`,
  direct `kv_slot_mapping` scatter path.
- Synthetic ROCm launch passed for the existing `head_dim=512`, `quant=False`,
  block-table scatter path after using FP32 compressor state tensors, which
  matches the aiter/flydsl kernel contract.

Runtime lmeval attempt with indexer-inner ATOM compressor enabled:

- Server:
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `GPU_MEMORY_UTILIZATION=0.9`
    - no `--enforce-eager`
    - `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1` by code state before the gate
    was added.
- Smoke request:
    - `/v1/completions`
    - prompt `Q: What is 2+2?\nA:`
    - `max_tokens=8`, `temperature=0`
    - succeeded and returned `4`.
- Full unchanged `lmeval.sh`:
    - failed before producing GSM8K metrics.
    - First lmeval batch returned HTTP 500s, then the server stopped accepting
    connections.
    - Server log showed inference-time Triton JIT for several metadata/indexer
    kernels, then ROCm reported
    `HSA_STATUS_ERROR_OUT_OF_RESOURCES` with `Available Free mem : 0 MB`.
    - Later stack traces surfaced as `hipErrorUnknown` in unrelated downstream
    calls, which is consistent with an earlier asynchronous ROCm failure.
- Interpretation:
    - The new indexer compressor is compile/runnable for small requests, but not
    default-valid for the high-concurrency accuracy run.
    - Keep it gated until we either pre-warm the exact large-batch shapes, reduce
    memory pressure, or replace the Triton path with an aiter/flydsl indexer
    compressor path that does not trigger large inference-time JIT pressure.

Constrained graph-mode retry after the block-table/indexer fastpath work:

- Server:
    - `MAX_NUM_SEQS=16`
    - `MAX_NUM_BATCHED_TOKENS=4096`
    - `MAX_MODEL_LEN=4096`
    - `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - no `--enforce-eager`
    - graph capture completed
    - ATOM unified KV views were bound from vLLM-owned KV storage
- Server log:
    - `runlogs/indexer-compressor-smoke-server.log`
    - no `ERROR`, `Traceback`, illegal-memory, `RuntimeError`, or `EngineDead`
    marker was present before the controlled shutdown
- C4 smoke:
    - command shape: `CONCURRENCIES=4 INPUT_LEN=256 OUTPUT_LEN=64`
    - result file: `bench-sparsemla/indexer-compressor-smoke-C4.json`
    - successful requests: `40`
    - failed requests: `0`
    - output throughput: `125.18 tok/s`
    - total token throughput: `633.70 tok/s`
    - mean TTFT: `259.30 ms`
    - mean TPOT: `28.32 ms`
- C16 smoke:
    - command shape: `CONCURRENCIES=16 INPUT_LEN=512 OUTPUT_LEN=256`
    - result file: `bench-sparsemla/indexer-compressor-smoke512-C16.json`
    - successful requests: `160`
    - failed requests: `0`
    - output throughput: `483.66 tok/s`
    - total token throughput: `1458.52 tok/s`
    - mean TTFT: `437.93 ms`
    - mean TPOT: `31.48 ms`
- Interpretation:
    - The indexer-inner compressor is now stronger than a one-request smoke: it
    can run graph mode at reduced deployment shape and survive multi-request
    decode/prefill smoke workloads.
    - This still does not promote it to default. It has not passed the unchanged
    `lmeval.sh` gate or the full C32 1024/1024 benchmark shape.
    - The smoke logs prove the path is runnable under the flag, but they do not
    by themselves prove which `fused_compress_attn` dispatcher was selected.
    Use `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR=1` or a ROCm profile before
    making performance claims about the flydsl/aiter indexer-compressor path.

Follow-up indexer-inner compressor block-table path:

- Change:
    - `DeepseekV4RocmAtomIndexerKCacheMetadata` now carries:
        - compressed `slot_mapping`
        - compressed `block_table`
        - compressed `block_size`
    - The indexer-inner compressor prefers `block_table + block_size` when the
    metadata block size matches the actual FP8 indexer cache storage block.
    - This avoids the direct-slot scatter path for pure decode, making
    `fused_compress_attn(..., quant=True)` eligible for the aiter/flydsl fused
    compressor wrapper, whose public API only accepts block-table scatter.
    - If metadata is missing, non-contiguous, or mismatched, the code falls back
    to the existing direct `kv_slot_mapping` scatter path.
- Validation:
    - `python3 -m py_compile` passed for:
        - `vllm/models/deepseek_v4/amd/model_state.py`
        - `vllm/models/deepseek_v4/compressor.py`
        - `vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py`
    - Small smoke server:
        - `MAX_NUM_SEQS=4`
        - `MAX_NUM_BATCHED_TOKENS=1024`
        - `MAX_MODEL_LEN=2048`
        - `GPU_MEMORY_UTILIZATION=0.85`
        - `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`
        - `ATOM_FUSED_COMPRESS_USE_FLYDSL=auto`
        - no `--enforce-eager`
    - Smoke request:
        - `/v1/completions`
        - prompt `Q: What is 2+2?\nA:`
        - `max_tokens=16`, `temperature=0`
        - returned HTTP 200 and text beginning with `4`.
    - First request still showed inference-time JIT for metadata/indexer reader
    kernels such as `_cp_gather_indexer_quant_cache_kernel` and
    `_gluon_fp8_mqa_logits_kernel`; no `_fused_compress_attn_kernel` JIT warning
    appeared in the request log.
- Interpretation:
    - This is a useful step toward ATOM's block-table compressor contract for the
    pure-decode indexer path.
    - It does not yet prove full GSM8K accuracy or high-concurrency stability with
    `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`; that remains gated off until a
    full lmeval run passes.

Default-gated validation after adding `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR`:

- Launch default:
    - `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=0`
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `GPU_MEMORY_UTILIZATION=0.9`
    - no `--enforce-eager`
- Unchanged `lmeval.sh` passed GSM8K:
    - flexible exact match: `0.9507 +/- 0.0060`
    - strict exact match: `0.9515 +/- 0.0059`
- Fresh C32 benchmark server:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `GPU_MEMORY_UTILIZATION=0.9`
    - no `--enforce-eager`
- `benchmarkvllm.sh` result:
    - JSON:
    `/app/atomdsv4/bench-indexer-gate-default-c32/indexer-gate-default-c32-C32.json`
    - successful requests: `320`
    - failed requests: `0`
    - benchmark duration: `376.48 s`
    - output throughput: `870.39 tok/s`
    - total throughput: `1744.18 tok/s`
    - mean TPOT: `35.80 ms`
    - mean TTFT: `1013.67 ms`

Conversion and metadata-preparation interpretation:

- Conversion logic and metadata preparation can slow the end-to-end attention
  path even when the GPU sparse-attention kernel is faster in isolation.
- They usually do not make the launched kernel's own measured GPU duration
  longer. Instead, they add time before or after the kernel, may add extra
  temporary buffers/copies, and may reduce overlap or graph replay efficiency.
- In this tree, the likely costs are:
    - vLLM ragged/block metadata to ATOM-style compact index conversion.
    - CSA/HCA translate/pack work before ATOM attention.
    - generic sparse MLA/SWA/indexer metadata construction still required by some
    fallback paths.
    - first-inference JIT for conversion/index kernels if a benchmark shape was
    not warmed.
- The measured metadata-only probes so far show host metadata is real but not
  large enough by itself to explain the full C32 gap to ATOM's published
  `1145.71 tok/s` target. The stronger remaining suspect is the combined
  adapter layer: GPU-side index conversion, per-layer wrapper work, cache-layout
  translation, and missing ATOM-style stream overlap.

### Current ATOM Op Parity Check

This section records the current source-level parity check against
`ATOM/atom/models/deepseek_v4.py` and the vendored ATOM kernel files.

CSA decode sequence in ATOM:

1. `Indexer.forward_batched(...)` produces per-token seq-local top-k.
2. `_fill_csa_paged_compress(...)` calls `csa_translate_pack(...)`.
3. Decode calls `sparse_attn_v4_paged_decode(...)` with the translated
   unified-KV offsets.

CSA decode sequence in vLLM ROCm default:

1. vLLM indexer writes `self.topk_indices_buffer`.
2. `DeepseekV4ROCMAiterMLAAttention._maybe_forward_decode_atom` calls local
   `csa_translate_pack(...)` for ratio-4 layers.
3. It then calls local `sparse_attn_v4_paged_decode(...)` through
   `run_paged_decode(...)`.

Result:

- `vllm/models/deepseek_v4/amd/v4_kernels/csa_translate_pack.py` is currently
  identical to `ATOM/atom/model_ops/v4_kernels/csa_translate_pack.py`.
- The default vLLM CSA path therefore follows the ATOM modeling-file op
  sequence for CSA translation and sparse decode.
- The old `sparse_attn_v4_csa_topk_paged_decode(...)` direct-fusion
  experiment was not an ATOM modeling-file op because it skipped
  `csa_translate_pack`. It has been removed from the current code path.

Paged decode wrapper parity:

- The local `sparse_attn_v4_paged_decode(...)` is vendored from ATOM but has
  vLLM-specific additions:
    - bounds-safe slot address formation;
    - optional split SWA/compressed KV wrapper;
    - workspace-manager use for partial split-K buffers.
- The ATOM-equivalent default is the generic unified-KV paged decode wrapper.
  The direct CSA/HCA decode experiments are no longer present. The split-KV
  path remains an experiment, not a required ATOM op.
- The stable deployment default now leaves
  `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS` empty so the vendored ATOM-style
  heuristic chooses split-K from capture-time shape. The Python wrapper treats
  an explicit value as an override and the empty default as "use heuristic".
- With `VLLM_ROCM_DSV4_ATOM_DECODE_SPLIT_WORKSPACE=torch_empty`, the heuristic
  split-K path passed graph-mode lmeval and the C32 benchmark after a fresh
  restart. The older vLLM WorkspaceManager partial-buffer path is still a
  larger-shape replay/reuse blocker.
- A diagnostic flag now exists for that revisit:
  `VLLM_ROCM_DSV4_ATOM_DECODE_SPLIT_WORKSPACE=workspace|torch_empty`. It lets
  the split-K path use either vLLM's shared WorkspaceManager scratch or ATOM's
  original `torch.empty` partial-buffer allocation. The default is now
  `torch_empty`; the flag is cached at module import and does not add hot-path
  environment lookups.

Compressor wrapper parity:

- `fused_compress.py` is vendored from ATOM but has vLLM-specific changes:
    - local imports instead of `atom` package imports;
    - optional direct slot-mapping scatter for the vLLM indexer-inner path;
    - a guard to avoid aiter/flydsl HCA when vLLM flattens HCA blocks;
    - looser RoPE cache stride validation needed by vLLM tensor views.
- Main CSA/HCA compressor ordering now matches ATOM inside the enabled ATOM
  compressor path: fused compression reads previous request-ring state and
  writes compressed KV, then `update_compressor_states` writes current state.
- Indexer-inner compressor is still gated off by default. It has small-smoke
  evidence, but not full GSM8K/high-concurrency evidence.

Attention/modeling-file op sequence gaps still present:

- ATOM starts compressor work asynchronously with `maybe_compressors_async(...)`
  before Q/KV projection, then waits before indexer and sparse attention. vLLM
  currently does not reproduce that auxiliary-stream overlap by default.
- ATOM uses one modeling-file dataflow around its own fused MoE/shared-expert
  path. vLLM still uses vLLM `FusedMoE` with the ROCm aiter backend, so the MoE
  kernel family is close but not identical to ATOM's full stream-overlapped MoE
  schedule.
- MHC/HC is not ATOM-equivalent by default. The installed `aiter` version has
  the MHC accuracy fix, but earlier standalone aiter MHC attempts did not pass
  full GSM8K in this tree, so the validated default remains the vLLM path.
- Prefill still has more fallback/JIT surface than the ATOM modeling-file path.
  The paged prefill wrapper exists and is wired, but prefill/mixed batches still
  carry legacy metadata and warmup-shape sensitivity.

Practical next target:

1. Keep direct CSA/HCA diagnostic kernels default-off when measuring ATOM-op
   parity.
2. Keep the heuristic split-K default plus ATOM-style `torch.empty` partials
   for graph-mode correctness; compare against an explicit split=1 only when
   measuring whether split-K helps a given deployment workload.
3. Next performance work should not replace `csa_translate_pack` with direct
   CSA unless the goal changes from "ATOM-op parity" to "new vLLM fusion".
   Instead, focus on:
   - graph-safe split-K paged decode workspace/lifetime audit;
   - indexer-inner compressor high-concurrency stability;
   - auxiliary compressor/indexer stream overlap;
   - MHC/HC aiter path revalidation on `aiter==0.1.15.post1`;
   - prewarming or avoiding first-inference JIT for prefill/index conversion
     shapes.

## 2026-06-19 Decode Conversion/Profile Evidence

Change:

- Added diagnostic-only split-K fields to `ATOM_PROFILE_DECODE`:
  `kv_splits`, `kv_splits_source`, and `split_workspace`.
- The fields are computed only when decode profiling is enabled, using cached
  module-level environment decisions; normal serving does not add host work.

Short graph-mode profile:

- Launch:
  `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192
  VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1
  VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1
  VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=100000
  VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=-1 bash launchdeepseekgraph.sh`
- Traffic:
  `vllm bench serve`, random `input_len=1024`, `output_len=128`,
  `num_prompts=32`, `max_concurrency=32`, graph mode, V2 runner,
  no `--enforce-eager`.
- Logs:
  `runlogs/profile-kvsplit-server.log`,
  `runlogs/profile-kvsplit-client.log`,
  `bench-sparsemla/profile-kvsplit-c32-short.json`.

Observed decode shape decisions:

- All 1456 decode profile rows used `path=atom_kernel`.
- `kv_splits_source=heuristic` for all rows.
- `split_workspace=torch_empty` for all split rows.
- Shape distribution:
    - `T=16`: `kv_splits=32`
    - `T=24`: `kv_splits=16`
    - `T=32`: `kv_splits=16`

Trimmed per-layer decode timing (`kernel_ms < 10`, so first JIT/capture outliers
are excluded):

| Ratio | T  | Splits | index_ms | translate_ms | kernel_ms | total_ms | non-kernel overhead |
| ----- | -- | ------ | -------- | ------------ | --------- | -------- | ------------------- |
| 4     | 16 | 32     | 0.0000   | 0.0513       | 0.1205    | 0.1847   | 34.8%               |
| 4     | 24 | 16     | 0.0000   | 0.0551       | 0.0987    | 0.1667   | 40.8%               |
| 4     | 32 | 16     | 0.0000   | 0.1027       | 0.1275    | 0.2436   | 47.7%               |
| 128   | 16 | 32     | 0.0012   | 0.0031       | 0.1508    | 0.1708   | 11.7%               |
| 128   | 24 | 16     | 0.0343   | 0.0031       | 0.1071    | 0.1597   | 32.9%               |
| 128   | 32 | 16     | 0.0678   | 0.0032       | 0.2407    | 0.3304   | 27.2%               |

Metadata/profile aggregate from the same short run:

- 144 `ROCm DSV4 ATOM metadata profile` rows.
- Average total metadata prep was 3.638 ms, but p50 was 0.887 ms and max was
  33.211 ms due to capture/warmup outliers.
- Steady decode-state detail rows at call 256 were about 0.086-0.097 ms per
  worker for `reqs=32/32 tokens=32/32`.
- `prepare_attn` steady rows around call 256 were about 0.54-0.63 ms total per
  worker.

Interpretation:

- The suspicion that conversion/metadata work can hide kernel gains is
  supported. For ratio-4 decode at C32, the `csa_translate_pack`/conversion
  portion alone is about the same order as the paged decode kernel and reaches
  almost half of the profiled per-layer total at `T=32`.
- Ratio-128 has almost no translation cost in the default fused-HCA-index path,
  but index-write/cache behavior still contributes noticeable non-kernel time.
- This reinforces the structural conclusion: to get the full ATOM benefit, the
  next performance target is not only the sparse attention kernel. vLLM needs a
  ROCm DSV4 state/cache layout that avoids rebuilding/translating ATOM-style
  per-request/ring metadata from vLLM block-table metadata in the decode hot
  path.

## 2026-06-19 Generic Indexer Metadata Skip Default

Change:

- `launchdeepseekgraph.sh` now defaults
  `VLLM_ROCM_DSV4_ATOM_SKIP_INDEXER_METADATA=1`.
- The Python path already had the guard. It only skips generic
  `DEEPSEEK_V4_INDEXER` metadata when all of these are true:
    - ROCm;
    - ATOM indexer fastpath enabled;
    - no FP4 indexer cache;
    - pure decode, one scheduled token per request.
- Prefill/mixed/spec paths still build the generic indexer metadata and can
  fall back to the normal indexer path.

Short graph-mode validation:

- Launch:
  same as the decode conversion profile above, plus the new default skip flag.
- Traffic:
  `vllm bench serve`, random `input_len=1024`, `output_len=128`,
  `num_prompts=32`, `max_concurrency=32`.
- Logs/results:
  `runlogs/skip-indexer-profile-server.log`,
  `runlogs/skip-indexer-profile-client.log`,
  `bench-sparsemla/skip-indexer-profile-c32-short.json`.
- Completed 32/32 requests, no server/client errors.

Metadata comparison against the immediately previous short profile:

| Config | skip_indexer rows | steady super_ms avg | steady attach_ms avg | steady total_ms avg |
| ------ | ----------------- | ------------------- | -------------------- | ------------------- |
| generic indexer metadata | 0/144 | 0.3454 | 0.0529 | 0.7892 |
| skip generic indexer metadata | 136/144 | 0.2168 | 0.1089 | 0.7238 |

Interpretation:

- The skip works and moves the deployment path closer to the practical split:
  pure-decode indexer metadata is supplied by ROCm DSV4 ModelState instead of
  the generic vLLM indexer metadata builder.
- It saves a small amount of steady metadata preparation, but it is not the
  main C32 bottleneck. The profiled per-layer decode timings are within noise
  and the ratio-4 `csa_translate_pack` cost remains.
- Because this changes the default launch behavior, the unchanged GSM8K lmeval
  gate must be re-run before treating the default as validated.

Full validation:

- Accuracy launch:
  default `bash launchdeepseekgraph.sh` after adding
  `VLLM_ROCM_DSV4_ATOM_SKIP_INDEXER_METADATA=1` to the launch defaults.
- Accuracy command:
  unchanged `bash lmeval.sh`.
- Accuracy logs:
  `runlogs/skip-indexer-accuracy-server.log`,
  `runlogs/skip-indexer-lmeval.log`.
- Result:
    - GSM8K flexible exact match: `0.9530 ± 0.0058`
    - GSM8K strict exact match: `0.9538 ± 0.0058`
    - This is within the required `0.95 ± 0.01` band.

Fresh C32 benchmark after restarting the server:

- Launch:
  `MAX_NUM_SEQS=32 bash launchdeepseekgraph.sh`
- Benchmark:
  `RESULT_PREFIX=skip-indexer-current-default CONCURRENCIES=32
  bash benchmarkvllm.sh`
- Logs/results:
  `runlogs/skip-indexer-benchmark-server.log`,
  `runlogs/skip-indexer-benchmark-c32.log`,
  `bench-sparsemla/skip-indexer-current-default-C32.json`.
- Result:
    - completed requests: `320/320`
    - output throughput: `884.5629 tok/s`
    - total throughput: `1772.5811 tok/s`
    - mean TPOT: `35.1764 ms`
    - median TPOT: `35.1532 ms`
    - mean TTFT: `1051.1898 ms`

Comparison:

- Previous heuristic split-K default before skipping generic indexer metadata:
  `882.5892 tok/s`, mean TPOT `35.2812 ms`.
- New skip-indexer default:
  `884.5629 tok/s`, mean TPOT `35.1764 ms`.
- The effect is positive but small, consistent with the short profile:
  generic indexer metadata was not the dominant bottleneck. It is still worth
  keeping because it removes unnecessary generic vLLM metadata construction from
  the validated pure-decode ATOM path and moves the implementation closer to
  the ROCm DSV4 ModelState split.

## 2026-06-19 MHC / HC AITER Readiness Check

Question:

- Do we have all MHC/HC AITER pieces needed to match ATOM's model path?

Current vLLM state:

- `vllm/model_executor/layers/mhc.py` caches
  `VLLM_ROCM_DSV4_USE_AITER_MHC` at import time and only dispatches MHC pre/post
  to AITER when that flag is enabled.
- With the default launch, this flag is not set, so the AMD model uses the
  existing tilelang fused post-pre MHC path when tilelang is available.
- `aiter==0.1.15.post1` exposes `mhc_pre` and `mhc_post`, but does not expose
  `mhc_fused_post_pre` in this environment:
  `hasattr(aiter, "mhc_fused_post_pre") == False`.
- ATOM guards its direct fused post-pre call with `getattr`, so absence of this
  symbol is expected to fall back in ATOM as well.

Change tested:

- Added a vLLM custom op wrapper `torch.ops.vllm.hc_head_aiter` in
  `vllm/model_executor/kernels/mhc/aiter.py`.
- The wrapper calls `aiter.ops.mhc.mhc_pre(..., sinkhorn_repeat=0)`, matching
  ATOM's HC head reduction instead of going through tilelang/triton.
- The HIP HC head dispatcher can use this wrapper only when the separate
  `VLLM_ROCM_DSV4_USE_AITER_HC_HEAD=1` flag is set. The default remains off.

Smoke-test outcome:

- Import/registration with `VLLM_ROCM_DSV4_USE_AITER_MHC=1` succeeded:
  `HAS_AITER_MHC=True`, `torch.ops.vllm.hc_head_aiter` registered.
- `hc_head_aiter` with toy `hc_mult=2` failed because AITER only supports
  `hc_mult=4`.
- `hc_head_aiter` with `hc_mult=4, hidden=256` failed with
  `RuntimeError: hidden_size must be >= residual_block * 2 stages prefetch`.
- `hc_head_aiter` with DSV4-shaped `hc_mult=4, hidden=7168, M=1` crashed with
  a process-level floating-point exception.
- AITER MHC pre/post with DSV4-shaped `hc_mult=4, hidden=7168, M=1` also crashed
  with a process-level floating-point exception.

Interpretation:

- We do not yet have a validated "all AITER MHC/HC" path in vLLM.
- The installed AITER package has the fixed MHC symbols, but the decode-sized
  smoke tests above are not safe enough to enable by default.
- For the current validated serving path, keep MHC/HC on the existing
  tilelang/fallback route. Attention/compressor ATOM kernels remain the active
  integration focus.
- If MHC/HC AITER is revisited, test with representative captured batch sizes
  and real model weights first, then rerun the unchanged GSM8K gate before
  enabling either `VLLM_ROCM_DSV4_USE_AITER_MHC` or
  `VLLM_ROCM_DSV4_USE_AITER_HC_HEAD` in the launch defaults.

- A fresh import probe of the installed package in this environment showed:
    - top-level `aiter.mhc_pre`: present.
    - top-level `aiter.mhc_post`: present.
    - top-level `aiter.mhc_fused_post_pre`: absent.
    - `aiter.ops.triton.fusions.mhc.mhc_post_pre`: absent from the installed
    package.
- Therefore vLLM cannot currently call a true installed AITER fused
  post/pre MHC op. The only available AITER MHC/HC experiments are standalone
  pre/post and HC-head-through-`mhc_pre`, both still accuracy/stability gated.

## 2026-06-19 KV Cache Split Audit

Question:

- Are we still side-allocating ATOM KV state, or does vLLM have a first-class
  ROCm DSV4 unified KV cache spec?

Evidence from current vLLM code:

- `vllm/v1/kv_cache_interface.py` defines
  `DeepseekV4AtomMLAAttentionSpec`, a ROCm DSV4 opt-in MLA spec that adds:
    - `atom_swa_prefix_bytes`
    - `atom_swa_pages`
    - `atom_swa_dtype`
- `vllm/v1/core/kv_cache_utils.py::_get_kv_cache_config_deepseek_v4`
  recognizes that spec and emits one `KVCacheTensor` per ATOM layer with:
    - a fixed SWA prefix;
    - the normal compressed tail sized by `spec.page_size_bytes * num_blocks`;
    - `fixed_prefix_size=spec.atom_swa_prefix_bytes`.
- `vllm/v1/worker/gpu/attn_utils.py` slices the fixed prefix off before
  constructing the backend `kv_cache` view, so the normal backend still sees
  the compressed tail while the raw storage contains `[SWA prefix | compressed
  tail]`.
- `vllm/models/deepseek_v4/amd/model_state.py` defaults to
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1` through the launch script and
  binds ATOM views from vLLM-owned KV storage in
  `_try_bind_atom_unified_kv_from_vllm`.
- If that vLLM-owned binding fails while the flag is enabled, model state
  raises instead of silently falling back to a side allocation. This is good:
  the current default tests the intended vLLM KV-cache integration path.

Current split:

- First-class vLLM KV allocation: yes, for the homogeneous BF16 ATOM sparse
  attention tensor layout `[SWA prefix | CSA/HCA compressed tail]`.
- Model-specific request state: yes, via `DeepseekV4RocmAtomModelState`:
  state slot mapping, SWA request slots, compressor state rings, decode/prefill
  buffers, and compress plans.
- CUDA isolation: yes. The model returns the custom state class only when
  `current_platform.is_rocm()` and `VLLM_ROCM_DSV4_ATOM_STATE=1`; the ATOM MLA
  spec is ROCm DSV4 opt-in.
- True ATOM/SGLang scheduler-native cache model: not yet. vLLM still allocates
  scheduler KV blocks and block tables. Model state derives ATOM request/ring
  metadata from `InputBatch`, `block_tables`, and `slot_mappings`, then writes
  ATOM decode indices or attaches minimal metadata.

Implication:

- The current implementation is not merely an external side buffer; it does use
  vLLM's KV-cache system for the main ATOM unified KV storage.
- The remaining structural mismatch is metadata ownership. ATOM/SGLang expose
  unified request/ring state directly to the sparse MLA kernels, while vLLM
  still adapts from block-table scheduling to ATOM per-token/per-layer indices.
- This explains why conversion/metadata overhead remains visible even after
  the sparse decode kernel itself is active.

Next integration target:

- Keep the vLLM-owned `DeepseekV4AtomMLAAttentionSpec` path.
- Move more pure-decode metadata from generic attention builders into
  `DeepseekV4RocmAtomModelState`, but only where there is an ATOM equivalent
  and the unchanged GSM8K gate passes.
- The hard boundary for a true ATOM-style layout is scheduler/core block-table
  semantics: removing `csa_translate_pack`/index adaptation entirely would
  require kernels and metadata to consume scheduler request state without
  reconstructing ATOM slot lists from vLLM block tables.

## 2026-06-19 Direct Decode Experiment Removal

Change:

- Removed the current runtime wiring for:
    - `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE`
    - `VLLM_ROCM_DSV4_ATOM_HCA_DIRECT_DECODE`
- Removed the corresponding vLLM-only direct decode wrappers from the current
  exported kernel surface:
    - `sparse_attn_v4_csa_topk_paged_decode`
    - `sparse_attn_v4_hca_state_paged_decode`
- Removed the private Triton kernels backing those wrappers from
  `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`.

Reason:

- These direct paths were experiments that bypassed the ATOM modeling-file
  sequence. The validated CSA path is:
  `topk -> csa_translate_pack -> sparse_attn_v4_paged_decode`.
- Prior measurements showed the direct CSA path was not a performance win, and
  keeping it made the current implementation harder to reason about when the
  goal is ATOM op parity.

Current decode path:

- Ratio 4 CSA:
  `csa_translate_pack(...)` fills the ATOM CSA segment, then
  `sparse_attn_v4_paged_decode(...)` consumes the unified indices.
- Ratio 128 HCA:
  the default fused-HCA-index path fills the HCA index segment when enabled,
  then `sparse_attn_v4_paged_decode(...)` consumes the unified indices.
- Split-KV decode remains separately gated as a layout experiment; it is not
  part of the direct CSA/HCA removal.

## 2026-06-19 Indexer-Inner Fused Compressor Kernel Smoke

Question:

- Is the ATOM `fused_compress_attn(..., quant=True)` kernel available for the
  CSA indexer-inner compressor shape, or is the default-off gate hiding a basic
  missing-kernel problem?

Synthetic smoke:

- Shape:
    - `head_dim=128`
    - `rope_head_dim=64`
    - `ratio=4`
    - `overlap=True`
    - one request, four scheduled tokens, one compression boundary
    - FP8 KV cache shape `[1, 64, 128]`
    - FP32 scale cache shape `[1, 64]`
    - block-table scatter path, not direct slot mapping
- First run with BF16 `kv_state` / `score_state` reached the AITER flydsl
  wrapper and failed with:
  `TypeError: kv_state/score_state must be fp32`.
- Runtime model state is compatible with this requirement:
  `DeepseekV4RocmAtomModelState._allocate_atom_state_buffers` allocates
  `csa_idx_kv_state` and `csa_idx_score_state` with `state_dtype=torch.float32`.
- Rerun with FP32 state tensors succeeded:
    - output cache dtype: `torch.float8_e4m3fnuz`
    - `cache_scale` finite
    - example scale value: `0.0078125`

Interpretation:

- The CSA indexer-inner quant fused-compress kernel is present and can run for
  the deployment shape in isolation.
- The reason `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1` remains default-off is
  not a missing basic kernel. The unresolved issue is serving-scale stability:
  previous large-batch lmeval attempts with this enabled hit ROCm
  out-of-resources / JIT pressure before accuracy could be measured.
- This narrows the next validation step: use a small server/eval or targeted
  benchmark with the flag enabled to determine whether the instability comes
  from concurrent graph capture/JIT pressure, metadata shape, or long-context
  runtime behavior.

Follow-up server smoke:

- Command shape:
  `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`,
  `MAX_NUM_SEQS=16`, `MAX_NUM_BATCHED_TOKENS=4096`, `MAX_MODEL_LEN=4096`,
  model runner v2, breakable CUDA graph, no `--enforce-eager`.
- The server completed model load and graph capture, bound ATOM unified KV views
  from vLLM-owned KV storage, and served a small random benchmark.
- Benchmark:
    - `num_prompts=16`
    - `random_input_len=128`
    - `random_output_len=16`
    - `max_concurrency=4`
    - completed `16/16`, failed `0`
    - mean TPOT `34.74 ms`
    - output throughput `79.44 tok/s`
- The server log showed first-inference JIT for metadata/conversion/staging
  kernels including:
    - `_build_prefill_chunk_metadata_kernel`
    - `_compute_prefill_metadata_kernel`
    - `_update_compressor_states_kernel`
    - `_swa_write_kernel`
    - `_cp_gather_indexer_quant_cache_kernel`
    - `_gluon_fp8_mqa_logits_kernel`
    - `_build_c128a_topk_metadata_kernel`
    - `_pack_dense_prefix_to_ragged_kernel`
    - `_compute_swa_indices_and_lens_kernel`
    - `_v4_paged_prefill_indices_kernel`
    - `_csa_translate_pack_kernel`

Interpretation:

- This does not prove the indexer-inner fused compressor is accurate or fast at
  full lmeval/benchmark shape.
- It does prove the flag can survive startup, graph capture, and a small serving
  decode.
- The logs support the hypothesis that conversion and metadata preparation can
  dominate end-to-end step time even when the ATOM paged attention kernel is
  faster in isolation. Future profiling should break out:
    - vLLM scheduler/block-table to ATOM request metadata construction
    - host-to-device metadata copies
    - `csa_translate_pack`
    - paged prefill/decode index writes
    - the final sparse paged attention kernel

## 2026-06-19 Metadata and Compressor Plan Profiling

Question:

- Can conversion logic and metadata preparation hide the benefit of faster ATOM
  kernels?

Profile run:

- Server shape:
    - model runner v2
    - breakable CUDA graph
    - no `--enforce-eager`
    - `MAX_NUM_SEQS=16`
    - `MAX_NUM_BATCHED_TOKENS=4096`
    - `MAX_MODEL_LEN=4096`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`
- Client:
    - `num_prompts=16`
    - `random_input_len=128`
    - `random_output_len=16`
    - `max_concurrency=4`

Findings:

- The Python decode-layer timers do not emit during served pure-decode replay
  because the deployed path is graph-replayed. For non-eager deployment, the
  metadata outside the captured graph is the reliable timing boundary from the
  current hooks.
- Mixed/prefill metadata is still heavy. Examples from
  `runlogs/profile-overhead-server.log`:
    - `reqs=16 tokens=32`: total metadata around `42-50 ms` across ranks.
    - `reqs=2 tokens=133`: total metadata ranged around `9-38 ms`.
    - `reqs=4 tokens=266`: total metadata around `4.2-4.7 ms`.
- Pure decode uses the ROCm ATOM skip path for generic indexer/decode/main
  compressor metadata:
    - `skip_indexer=True`
    - `skip_decode=True`
    - `skip_compressor=True`
    - total metadata around `0.72-1.03 ms` for 4-16 live requests.
- In pure decode, the remaining model-state cost is mostly:
    - compressor plan construction/copy: around `0.17-0.20 ms`
    - state metadata and decode indptr construction: around `0.12-0.28 ms`
    - minimal indexer attachment: around `0.11-0.14 ms`

Interpretation:

- Yes, metadata/conversion can hide kernel gains, but the current validated
  pure-decode path has already bypassed the largest generic sparse MLA/SWA
  metadata builders.
- The remaining pure-decode metadata is not zero, but at this small shape it is
  closer to a sub-millisecond per-rank adapter cost than the dominant 10s of ms
  mixed/prefill cost.
- `csa_translate_pack` is still part of the ATOM op sequence used by the
  current path; removing it would be a different kernel contract, not simply a
  vLLM cleanup.

Follow-up change:

- Tightened pure-decode compressor plan capacity:
    - Before: decode-like forwards used prefill-sized `compress_plan_gpu`
    capacity from `max_num_batched_tokens`.
    - After: decode-like forwards use
    `max_num_reqs * ceil((1 + num_spec_tokens) / ratio)` for both compress and
    write plan views, while keeping the same persistent backing buffers.
    - Prefill behavior remains unchanged.
- Rationale:
    - This reduces host sentinel fill/copy surface.
    - More importantly, it reduces captured compressor-kernel plan capacity for
    pure decode, so fused-compress kernels do not have to launch over a
    prefill-sized sentinel tail.

Validation:

- Syntax:
  `python3 -m py_compile vllm/models/deepseek_v4/amd/v4_kernels/compress_plan.py vllm/models/deepseek_v4/amd/model_state.py`
- Plan-builder invariant smoke:
    - ratio 4 decode cap/write cap shape `[4, 4]`
    - ratio 128 decode cap/write cap shape `[4, 4]`
    - expected boundary counts passed
- Microbench with pinned host buffers and GPU copies:
    - full `8192` capacity: mean `0.0820 ms`
    - tight `256` capacity: mean `0.0769 ms`
    - host-side staging improvement is small, so the more meaningful expected
    benefit is reduced captured kernel grid work.
- Serving smoke after the change:
    - graph mode, no `--enforce-eager`
    - completed graph capture
    - random serving benchmark completed `16/16`, failed `0`
    - mean TPOT `31.28 ms`
    - output throughput `89.22 tok/s`

Open caveat:

- This is not yet an accuracy or deployment benchmark. It proves the tightened
  decode plan capacity is graph-runnable and request-runnable at the small
  smoke shape. The next required gates are unchanged: full `lmeval.sh` for
  GSM8K accuracy, then fresh-server `benchmarkvllm.sh` for C32 performance.

## 2026-06-19 Tight Decode Plan Accuracy And C32 Benchmark

Accuracy gate:

- Server:
    - default `launchdeepseekgraph.sh`
    - model runner v2
    - breakable CUDA graph
    - no `--enforce-eager`
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - block size `128`
- Command:
    - unchanged `/app/atomdsv4/lmeval.sh`
- Logs:
    - `runlogs/decode-plan-tight-accuracy-server.log`
    - `runlogs/decode-plan-tight-lmeval.log`
- Result:
    - GSM8K flexible exact match: `0.9545 +/- 0.0057`
    - GSM8K strict exact match: `0.9553 +/- 0.0057`
    - target `0.95 +/- 0.01` passed

Fresh C32 benchmark:

- The accuracy server was stopped before benchmarking.
- Server:
    - `MAX_NUM_SEQS=32 bash launchdeepseekgraph.sh`
    - model runner v2
    - breakable CUDA graph
    - no `--enforce-eager`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - block size `128`
    - graph capture completed
    - bound ATOM unified KV views from vLLM-owned KV storage
- Command:
    - `RESULT_PREFIX=decode-plan-tight-C32-run CONCURRENCIES=32 bash benchmarkvllm.sh`
- Logs/results:
    - `runlogs/decode-plan-tight-bench-server.log`
    - `runlogs/decode-plan-tight-benchmark.log`
    - `bench-sparsemla/decode-plan-tight-C32-run-C32.json`
- Result:
    - successful requests: `320`
    - failed requests: `0`
    - output throughput: `887.95 tok/s`
    - peak output throughput: `998.00 tok/s`
    - total token throughput: `1779.37 tok/s`
    - mean TTFT: `848.09 ms`
    - median TTFT: `866.76 ms`
    - mean TPOT: `35.23 ms`
    - median TPOT: `35.24 ms`
    - p90 TPOT: `35.67 ms`
    - p99 TPOT: `35.76 ms`

Comparison:

- Immediate previous skip-indexer default C32:
    - `bench-sparsemla/skip-indexer-current-default-C32.json`
    - output throughput `884.56 tok/s`
    - total throughput `1772.58 tok/s`
    - mean TPOT `35.18 ms`
- Tight decode plan is a very small throughput improvement over that immediate
  baseline, but TPOT is essentially unchanged.
- It is not the best saved C32 result in the workspace. The fastest saved
  historical run remains:
    - `bench-sparsemla/revert-compressor-aux-nomtp-C32.json`
    - output throughput `926.06 tok/s`
    - total throughput `1855.74 tok/s`
    - mean TPOT `33.50 ms`
- It remains below ATOM's documented FP8 TP8 C32 target:
    - output throughput `1145.71 tok/s`
    - mean TPOT `26.90 ms`

Interpretation:

- The tightened pure-decode compressor plan is accuracy-safe.
- Its deployment impact is small. The host staging microbench also showed only
  a small copy/fill improvement, so the dominant C32 gap is elsewhere.
- Conversion and metadata preparation are real end-to-end costs, especially for
  mixed/prefill steps. However, the current evidence does not support
  compressor plan capacity alone as the reason the ATOM attention/compressor
  kernels do not reach the published ATOM C32 number inside vLLM.

Follow-up cleanup:

- The ROCm DSV4 ATOM flags in the active code path are module-level cached; the
  scan did not find repeated `os.environ.get(...)` calls in the per-layer
  decode hot path.
- `_prepare_atom_decode_metadata` still allocated several NumPy temporaries
  every forward while constructing graph-stable SWA/CSA/HCA decode indptrs.
  This was changed to use reusable CPU scratch arrays in
  `DeepseekV4RocmAtomDecodeBuffers`.
- A standalone equivalence check over random valid and padded `batch_id=-1`
  inputs matched the previous formulas.
- Isolated CPU timing for the arithmetic portion:
    - `T=32`: old `12.37 us`, scratch `11.09 us`, `1.12x`
    - `T=64`: old `12.68 us`, scratch `11.35 us`, `1.12x`
    - `T=256`: old `13.89 us`, scratch `12.22 us`, `1.14x`
    - `T=1024`: old `18.42 us`, scratch `15.42 us`, `1.19x`
- This removes avoidable Python/NumPy allocator churn, but the expected
  deployment impact is small because prior C32 profiles put the whole
  decode-indptr slice in the sub-millisecond range.

Additional metadata-gating cleanup:

- `prepare_attn` previously evaluated the same pure one-token decode predicate
  while deciding whether to skip generic indexer metadata, generic sparse-MLA
  decode metadata, generic compressor metadata, and direct indexer attachment.
- The predicate is now computed once per `prepare_attn` call and passed through
  those helpers.
- Isolated timing for a representative NumPy predicate:
    - `n=32`: repeated `6.18 us`, cached `1.60 us`
    - `n=64`: repeated `6.10 us`, cached `1.58 us`
    - `n=256`: repeated `6.15 us`, cached `1.60 us`
- This is not expected to visibly move C32 throughput by itself, but it removes
  another unnecessary adapter cost from the vLLM-to-ATOM metadata path.

## 2026-06-19 ATOM Component Coverage Audit

Question: do we currently have every component needed to benefit from all ATOM
kernels inside vLLM?

Short answer: no, not yet. The current ROCm DSV4 path has many ATOM operations
ported and several enabled in the validated launch path, but it is still a
hybrid vLLM/ATOM adapter. The largest remaining gap is not just one kernel; it
is the structural difference between ATOM's request-ring/unified-cache
execution model and vLLM's block-table metadata model.

Component matrix:

| ATOM modeling-file component | vLLM current state | Default launch state | Notes |
| --- | --- | --- | --- |
| vLLM scheduler / model runner v2 | Integrated | Enabled | `VLLM_USE_V2_MODEL_RUNNER=1`; no GPU worker changes needed for request state. |
| Block size 128 | Integrated | Enabled | Launch uses `--block-size 128`, matching ATOM's DSV4 compressed-block alignment. |
| Request-lived SWA/compressor state rings | Integrated as `ModelState` side buffers | Enabled | Persistent slots come from v2 request indices. This is the right direction, but still not ATOM's original cache owner. |
| vLLM-owned unified KV storage | Integrated as ROCm-only binding/view path | Enabled | `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`; BF16 unified views are bound from vLLM KV storage. |
| ATOM `qk_norm_rope_maybe_quant` sequence | Ported and called | Enabled | vLLM calls the fused q/k norm+RoPE path with `quant_q=False`, `quant_k=False`, matching the observed ATOM call pattern. |
| Fused q RMSNorm/group quant | Uses aiter RMSNorm group fused quant | Enabled | `ATOM_USE_FUSED_Q_NORM_QUANT=1`; this is separate from compressor ordering and not mathematically tied to it. |
| `swa_write` request ring update | Ported and called | Enabled | Decode writes before attention; prefill writes after attention to preserve prior-prefix reads. |
| Main compressor `fused_compress_attn` | Ported/called through compressor fast path | Enabled | Uses ATOM-style read-before-update ordering with graph-safe decode-tight plans. |
| Main compressor `update_compressor_states` | Ported/called | Enabled | Tightened decode plan capacity passed accuracy, but perf impact was small. |
| Indexer compressor fused path | Ported and accuracy-validated only under lower server concurrency | Disabled | `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=0`; C4 256/64 and C16 512/256 graph-mode smoke passed, unchanged `lmeval.sh` passed at `MAX_NUM_SEQS=64`, and C32 1024/1024 completed at `MAX_NUM_SEQS=32`. Default `MAX_NUM_SEQS=256` lmeval still fails from worker/resource pressure in `update_compressor_states`, and the C32 run is slower than the default-off path, so it should remain gated. |
| Indexer prefill `cp_gather_indexer_k_quant_cache` + `fp8_mqa_logits` + topk | Available in vLLM ROCm sparse wrapper | Partly hybrid | vLLM has aiter/triton wrappers, but the active hot path skips generic decode metadata where possible and still relies on vLLM indexer cache semantics. |
| Indexer decode `deepgemm_fp8_paged_mqa_logits` + topk | Available | Partly hybrid | Direct metadata attachment avoids the generic builder for pure decode, but the cache layout is still vLLM-derived. |
| ATOM paged decode sparse attention | Ported as `sparse_attn_v4_paged_decode` | Enabled | The fastest validated run uses this ATOM-style paged decode path, not the old vLLM sparse attention fallback. |
| ATOM paged prefill sparse attention | Ported as `sparse_attn_v4_paged_prefill` | Enabled for allowed cases | Mixed/large-prefill edge cases are still guarded by flags; production correctness still needs fallback coverage. |
| CSA translate/pack | Ported and called | Enabled | Source is equivalent to ATOM; this is an ATOM operation cost, not a vLLM divergence. |
| HCA fused index writer | Ported/called | Enabled | `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=1`. |
| AITER MHC/HC kernels | vLLM wrappers exist | Disabled | Installed `aiter==0.1.15.post1` exposes `mhc_pre`/`mhc_post`; `mhc_fused_post_pre` is absent. Earlier DSV4-shaped smoke hit a floating-point exception, so this cannot be default-enabled yet. |
| `fused_clamp_act_mul` / fused MoE activation | Available through aiter path | Enabled by launch flag | This is an op-level alignment, but not the dominant attention/cache structural gap. |
| Dual-stream / auxiliary-stream overlap | Reverted/off | Disabled | Current validated path is single-stream/sequential. ATOM's multistream overlap is not represented. |
| Full ATOM package dependency | Not used | Satisfied | No vLLM runtime dependency on `atom` package was found; ports live under vLLM. |

Conclusion:

- We have enough ATOM components to run a meaningful vLLM preview using
  ATOM-style attention, compressor state, SWA rings, fused q/k path, and
  vLLM-owned unified KV views.
- We do not yet have all components needed to claim the full ATOM kernel
  benefit. The missing or disabled pieces are indexer fused compressor,
  reliable AITER MHC/HC, auxiliary-stream overlap, and a deeper removal of
  vLLM-side metadata/helper work around the unified KV layout.
- The next high-value validation should profile the deployment decode step as:
  metadata/adapter CPU time, helper GPU kernels, sparse attention kernel, and
  compressor state update. This will show whether the ATOM attention kernel is
  faster but hidden by conversion/index/pack work, or whether the kernel itself
  is still slower in this vLLM layout.

Additional conversion cleanup:

- Request-length conversion in `DeepseekV4RocmAtomModelState.prepare_attn` used
  fresh `np.ascontiguousarray` allocations for scheduled, computed, and context
  lengths. Those arrays feed both compressor plan generation and ATOM state
  metadata.
- The conversion now copies into persistent int32 scratch arrays once per
  scheduler step and reuses the views for both consumers.
- The pure one-token decode predicate now reads the same scratch scheduled view
  instead of scanning the original scheduler array separately.
- Standalone equivalence/timing for the request-length conversion:
    - `n=32`: old `1.103 us`, scratch `0.946 us`, `1.17x`
    - `n=64`: old `1.112 us`, scratch `0.954 us`, `1.17x`
    - `n=256`: old `1.193 us`, scratch `0.991 us`, `1.20x`
    - `n=1024`: old `1.507 us`, scratch `1.306 us`, `1.15x`
- Validation after the edit:
    - `python3 -m py_compile vllm/models/deepseek_v4/amd/model_state.py`
    - `git diff --check`

Environment flag caching audit:

- The ROCm DSV4 ATOM flags in the active model, attention, sparse MLA,
  compressor, and AMD kernel wrapper files are read into module-level constants
  or helper-derived constants.
- No repeated `os.environ.get(...)` calls were found inside the per-token
  attention/compressor forward loops during this audit.
- Therefore the current metadata/conversion overhead hypothesis should focus on
  metadata builders, array/view conversion, H2D copies, helper GPU kernels, and
  cache-layout translation rather than repeated environment lookups.

Accuracy and C32 validation after the request-length scratch change:

- Accuracy server:
    - default `bash launchdeepseekgraph.sh`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - breakable CUDA graph
    - no `--enforce-eager`
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - block size `128`
    - graph capture completed
    - ATOM unified KV views were bound from vLLM-owned KV storage
- Accuracy command:
    - unchanged `/app/atomdsv4/lmeval.sh`
- Accuracy logs:
    - `runlogs/metadata-scratch-accuracy-server.log`
    - `runlogs/metadata-scratch-lmeval.log`
- Accuracy result:
    - GSM8K flexible exact match: `0.9462 +/- 0.0062`
    - GSM8K strict exact match: `0.9469 +/- 0.0062`
    - target `0.95 +/- 0.01` passed
- Fresh C32 benchmark server:
    - the accuracy server was stopped first
    - `MAX_NUM_SEQS=32 bash launchdeepseekgraph.sh`
    - graph capture completed
    - ATOM unified KV views were rebound from vLLM-owned KV storage
- Benchmark command:
    - `RESULT_PREFIX=metadata-scratch-C32-run CONCURRENCIES=32 bash benchmarkvllm.sh`
- Benchmark files:
    - `runlogs/metadata-scratch-bench-server.log`
    - `bench-sparsemla/metadata-scratch-C32-run-C32.json`
- C32 result:
    - successful requests: `320`
    - failed requests: `0`
    - output throughput: `887.52 tok/s`
    - peak output throughput: `992.00 tok/s`
    - total token throughput: `1778.50 tok/s`
    - mean TTFT: `974.85 ms`
    - median TTFT: `941.04 ms`
    - mean TPOT: `35.13 ms`
    - median TPOT: `35.11 ms`
    - p90 TPOT: `35.70 ms`
    - p99 TPOT: `35.89 ms`
- Comparison:
    - immediate previous validated C32
    `bench-sparsemla/decode-plan-tight-C32-run-C32.json`:
    output `887.95 tok/s`, mean TPOT `35.23 ms`
    - new run is effectively flat: output `-0.05%`, mean TPOT slightly better
    - historical best saved run
    `bench-sparsemla/revert-compressor-aux-nomtp-C32.json`:
    output `926.06 tok/s`, mean TPOT `33.50 ms`
    - new run is `4.16%` below that historical best output throughput
    - ATOM C32 target remains output `1145.71 tok/s`, mean TPOT `26.90 ms`
    - new run is `22.54%` below ATOM C32 output throughput, a gap of
    `258.19 tok/s`

Interpretation:

- The request-length scratch cleanup is accuracy-safe.
- It does not move deployment C32 throughput in a meaningful way.
- This reinforces the earlier conclusion: Python/NumPy conversion cleanup helps
  remove adapter overhead, but the remaining C32 gap is dominated by larger
  structural or GPU-helper costs such as prefill/index metadata kernels,
  CSA/HCA packing, compressor state update, an indexer compressor that is only
  reduced-smoke validated, missing AITER MHC/HC, and lack of ATOM
  auxiliary-stream overlap.

## 2026-06-19 Indexer Compressor Accuracy And C32 Follow-Up

Question:

- Can the default-off indexer-inner ATOM compressor path pass accuracy, and is
  it useful for deployment C32 performance?

Configuration:

- `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`
- v2 model runner
- graph mode / breakable CUDA graph
- no `--enforce-eager`
- unchanged `lmeval.sh`
- unchanged `benchmarkvllm.sh`

Default server-concurrency check:

- Server:
    - default `bash launchdeepseekgraph.sh`
    - `MAX_NUM_SEQS=256`
    - `MAX_MODEL_LEN=8192`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - block size `128`
- Smoke:
    - `RESULT_PREFIX=indexer-compressor-default-C1-smoke CONCURRENCIES=1`
    - `INPUT_LEN=1024 OUTPUT_LEN=1024 bash benchmarkvllm.sh`
    - file:
    `bench-sparsemla/indexer-compressor-default-C1-smoke-C1.json`
    - completed `10`, failed `0`
    - output throughput `37.26 tok/s`
    - total throughput `74.67 tok/s`
    - mean TTFT `160.22 ms`
    - mean TPOT `26.71 ms`
- Single direct completion also succeeded:
    - `runlogs/indexer-compressor-default-smoke-completion.json`
- Unchanged `bash lmeval.sh` on this default server failed before an accuracy
  result:
    - `runlogs/indexer-compressor-default-lmeval-fail-server.log`
    - the visible stack lands in
    `vllm/models/deepseek_v4/amd/v4_kernels/state_writes.py`
    `update_compressor_states`
    - client observed HTTP 500 / connection refusal after the worker failure
    - this is a high-concurrency resource/stability failure, not a small-request
    functional failure

Lower server-concurrency accuracy check:

- Server:
    - `MAX_NUM_SEQS=64 VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`
    `bash launchdeepseekgraph.sh`
    - graph capture completed
    - vLLM-owned unified KV views were bound with `swa_pages=8192`
- Accuracy command:
    - unchanged `bash lmeval.sh`
- Logs:
    - `runlogs/indexer-compressor-maxseq64-lmeval.log`
    - `runlogs/indexer-compressor-maxseq64-accuracy-server.log`
- GSM8K result:
    - flexible exact match: `0.9492 +/- 0.006`
    - strict exact match: `0.9500 +/- 0.006`
    - target `0.95 +/- 0.01` passed

Fresh C32 benchmark:

- Server:
    - `MAX_NUM_SEQS=32 VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`
    `bash launchdeepseekgraph.sh`
    - graph capture completed
    - vLLM-owned unified KV views were bound with `swa_pages=4096`
- Benchmark command:
    - `RESULT_PREFIX=indexer-compressor-accuracypass-C32-run CONCURRENCIES=32`
    `bash benchmarkvllm.sh`
- Files:
    - `bench-sparsemla/indexer-compressor-accuracypass-C32-run-C32.json`
    - `runlogs/indexer-compressor-accuracypass-C32-server.log`
- C32 result:
    - completed `320`
    - failed `0`
    - output throughput `880.49 tok/s`
    - total token throughput `1764.41 tok/s`
    - mean TTFT `853.48 ms`
    - median TTFT `869.19 ms`
    - mean TPOT `35.53 ms`
    - median TPOT `35.56 ms`

Comparison:

- Default-off immediate baseline:
    - `bench-sparsemla/metadata-scratch-C32-run-C32.json`
    - output `887.52 tok/s`
    - total `1778.50 tok/s`
    - mean TPOT `35.13 ms`
- Indexer compressor on is slower than that baseline:
    - output throughput: `-0.79%`
    - mean TPOT: `+1.14%` slower
- Historical fastest saved C32 run:
    - `bench-sparsemla/revert-compressor-aux-nomtp-C32.json`
    - output `926.06 tok/s`
    - mean TPOT `33.50 ms`
- Indexer compressor on is `4.92%` below that historical best output
  throughput.
- ATOM recipe C32 target remains:
    - output `1145.71 tok/s`
    - mean TPOT `26.90 ms`
- Indexer compressor on is `23.15%` below the ATOM C32 output target.

Interpretation:

- The indexer-inner compressor path is now stronger than smoke-only: it can pass
  the unchanged GSM8K lmeval command when server concurrency is reduced to
  `MAX_NUM_SEQS=64`.
- It is not ready as a default path:
    - default `MAX_NUM_SEQS=256` lmeval still fails from worker/resource pressure;
    - C32 deployment throughput is lower than the default-off path;
    - the logs do not yet prove that the aiter/flydsl fused-compressor dispatcher
    is the selected implementation in the hot path.
- Keep `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=0` as the default until a
  compressor profile or ROCm profiler trace proves the selected kernel path and
  the resource pressure is fixed.

## 2026-06-19 Profiled Default C32 Attribution

Question:

- Can conversion logic and metadata preparation hide the benefit of the ATOM
  attention kernels in the current vLLM adapter path?

Run configuration:

- Default indexer compressor off:
    - `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=0`
- Server:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - v2 model runner
    - graph mode / breakable CUDA graph
    - no `--enforce-eager`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_START_AFTER=64`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_EVERY=128`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL=1`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=0`
    - `VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=256`
- Benchmark:
    - `RESULT_PREFIX=profile-default-C32-run`
    - `RESULT_DIR=./bench-profile-default-c32`
    - `CONCURRENCIES=32 bash benchmarkvllm.sh`
- Files:
    - `bench-profile-default-c32/profile-default-C32-run-C32.json`
    - `runlogs/profile-default-C32-run-server.log`

Benchmark result:

- completed `320`
- failed `0`
- output throughput `884.43 tok/s`
- total token throughput `1772.32 tok/s`
- mean TTFT `990.00 ms`
- median TTFT `1037.26 ms`
- mean TPOT `35.24 ms`
- median TPOT `35.15 ms`
- p90 TPOT `35.81 ms`
- p99 TPOT `36.03 ms`

Profile sample counts:

- `ATOM_PROFILE_DECODE`: `25`
- `ATOM_PROFILE_PREFILL`: `25`
- `ATOM_PROFILE_METADATA`: `99`
- `ROCm DSV4 ATOM state metadata detail`: `824`
- Triton JIT warnings during inference: `13`

Layer-0 HCA decode, `T=32`, averaged across 8 sampled rank rows:

| Segment | Avg ms | P50 ms | Max ms |
| --- | ---: | ---: | ---: |
| index/index-write | `2.875` | `2.897` | `3.029` |
| translate | `0.006` | `0.006` | `0.007` |
| ATOM paged-decode kernel | `5.486` | `5.491` | `5.686` |
| total wrapper | `8.367` | `8.383` | `8.724` |

Layer-0 HCA initial graph/warmup prefill, `T=64`, averaged across 8 sampled
rank rows:

| Segment | Avg ms | P50 ms | Max ms |
| --- | ---: | ---: | ---: |
| build indptr/meta | `0.216` | `0.215` | `0.235` |
| index/index-write | `2.574` | `2.590` | `2.644` |
| ATOM paged-prefill kernel | `4.429` | `4.475` | `4.718` |
| SWA write | `1.058` | `1.062` | `1.089` |
| total wrapper | `8.337` | `8.347` | `8.740` |

Large prefill samples during the C32 benchmark:

- `T=1028`, averaged across 8 rank rows:
    - build `0.270 ms`
    - index `0.130 ms`
    - ATOM kernel `0.201 ms`
    - SWA write `1.219 ms`
    - total `1.861 ms`
- `T=4112`, averaged across 8 rank rows:
    - build `0.257 ms`
    - index `1.188 ms`
    - ATOM kernel `0.369 ms`
    - SWA write `0.060 ms`
    - total `1.933 ms`

Backend metadata samples:

- MLA metadata, ratio 128, `tokens=64`:
    - average total `3.294 ms`
- MLA metadata, ratio 4, `tokens=64`:
    - average total `0.360 ms`
- SWA metadata, `tokens=64`:
    - average total across both SWA metadata objects `8.712 ms`
    - the first SWA object was around `16-18 ms` in the graph/warmup profile
    rows, while the second was around `0.25-0.30 ms`
- Model-state metadata detail during steady decode:
    - average total `0.088 ms`
    - p50 total `0.087 ms`
    - max total `0.142 ms`
    - decode indptr construction averaged `0.038 ms`

First-inference JIT warnings hit exactly the helper path under discussion:

- `_build_prefill_chunk_metadata_kernel`
- `_compute_prefill_metadata_kernel`
- `_update_compressor_states_kernel`
- `_swa_write_kernel`
- `_gemm_a8w8_blockscale_kernel`
- `_cp_gather_indexer_quant_cache_kernel`
- `_gluon_fp8_mqa_logits_kernel`
- `_build_c128a_topk_metadata_kernel`
- `_pack_dense_prefix_to_ragged_kernel`
- `_compute_swa_indices_and_lens_kernel`
- `_v4_paged_prefill_indices_kernel`
- `_csa_translate_pack_kernel`
- `_gemm_a16_w16_kernel`

Interpretation:

- Yes, conversion and metadata preparation can hide part of the ATOM kernel
  benefit, but it is not a single Python `os.environ.get`-style issue.
- Steady pure-decode model-state metadata is already small at C32
  (`~0.09 ms` sampled), so the remaining decode overhead is mostly GPU helper
  work and compatibility indexing:
    - HCA decode index/index-write averages `2.875 ms` for layer 0, compared with
    `5.486 ms` in the sampled ATOM paged-decode kernel.
- Prefill attribution is more severe:
    - For `T=4112`, index construction is `1.188 ms` while the sampled ATOM
    paged-prefill kernel is only `0.369 ms`.
    - For `T=1028`, SWA write is `1.219 ms` while the sampled ATOM paged-prefill
    kernel is only `0.201 ms`.
- This supports the structural plan: after the current vLLM-owned unified KV
  binding, the next useful performance work is not another small CPU conversion
  cleanup. It is to remove or fuse the GPU helper/index path around
  `v4_paged_prefill_indices`, `csa_translate_pack`, SWA write, and indexer
  FP8 gather/logits metadata, or move those descriptors into persistent
  request-state/backend metadata so the ATOM kernels consume them directly.

## CSA prefill layout checkpoint

The profiled prefill path exposed a layout mismatch in the fused
`csa_translate_pack` helper:

- `v4_paged_prefill_indices` writes the SWA/prefix portion of each prefill CSA
  slice at the slice head.
- Decode intentionally writes CSA topk at the slice head and keeps SWA at the
  tail.
- Prefill must therefore write CSA topk at `indptr + skip_prefix_len`, not
  always at `indptr`.

Candidate fix:

- `vllm/models/deepseek_v4/amd/v4_kernels/csa_translate_pack.py`
- Kernel write base is now:
    - decode / `INLINE_SKIP_FROM_POS=True`: `indptr`
    - prefill / `INLINE_SKIP_FROM_POS=False`: `indptr + skip`
- The torch reference path mirrors the same layout.

Validation performed:

- CPU reference sanity for one token with `skip=2`:
    - prefill preserves the two SWA-prefix cells and writes CSA topk after them.
    - decode writes CSA topk at the head and leaves the tail untouched.
- Reduced graph-mode server:
    - `MAX_NUM_SEQS=4`
    - `MAX_NUM_BATCHED_TOKENS=2048`
    - `MAX_MODEL_LEN=4096`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - no `--enforce-eager`
- Smoke workload:
    - `RESULT_DIR=./bench-csa-prefill-layout-smoke`
    - `RESULT_PREFIX=csa-prefill-layout-smoke`
    - `CONCURRENCIES=4`
    - `INPUT_LEN=2048`
    - `OUTPUT_LEN=64`
    - `bash benchmarkvllm.sh`
- Result:
    - `bench-csa-prefill-layout-smoke/csa-prefill-layout-smoke-C4.json`
    - completed `40`, failed `0`
    - output throughput `103.016 tok/s`
    - total throughput `3405.968 tok/s`
    - mean TTFT `518.624 ms`
    - mean TPOT `31.147 ms`
- Server log:
    - `runlogs/csa-prefill-layout-smoke-server.log`

This is only a reduced shape check. It does not replace the required unchanged
`lmeval.sh` GSM8K run and the fresh C32 `benchmarkvllm.sh` run.

Full validation after the CSA prefill layout change:

- Accuracy server:
    - default `launchdeepseekgraph.sh`
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - no `--enforce-eager`
- Accuracy command:
    - unchanged `bash lmeval.sh`
- Accuracy result:
    - GSM8K flexible-extract exact match: `0.9500 ± 0.006`
    - GSM8K strict-match exact match: `0.9507 ± 0.006`
    - samples file:
    `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/samples_gsm8k_2026-06-19T16-36-38.932353.jsonl`
    - server log: `runlogs/csa-prefill-layout-lmeval-server.log`
- Fresh performance server:
    - restarted after accuracy to clear KV/prefix state
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - no `--enforce-eager`
- Performance command:
    - `RESULT_DIR=./bench-csa-prefill-layout-c32`
    - `RESULT_PREFIX=csa-prefill-layout`
    - `CONCURRENCIES=32`
    - `bash benchmarkvllm.sh`
- Performance result:
    - `bench-csa-prefill-layout-c32/csa-prefill-layout-C32.json`
    - completed `320`, failed `0`
    - output throughput `885.850 tok/s`
    - total throughput `1775.160 tok/s`
    - mean TTFT `983.358 ms`
    - mean TPOT `35.189 ms`
    - server log: `runlogs/csa-prefill-layout-c32-server.log`

Comparison with nearby prior runs:

| Run | Output tok/s | Total tok/s | Mean TPOT ms | Notes |
| --- | ---: | ---: | ---: | --- |
| `csa-prefill-layout-C32.json` | `885.850` | `1775.160` | `35.189` | CSA prefill layout fix |
| `metadata-scratch-C32-run-C32.json` | `887.515` | `1778.498` | `35.131` | Prior default after request-length scratch |
| `revert-compressor-aux-nomtp-C32.json` | `926.061` | `1855.740` | `33.503` | Historical best saved C32 |

Interpretation:

- The CSA prefill layout fix preserves GSM8K accuracy and does not materially
  change C32 performance.
- The current C32 number remains close to the prior default run and below the
  historical best saved run.
- The fix should be treated as a correctness/layout alignment, not a throughput
  optimization.

Follow-up helper cleanup:

- Removed the CSA prefill `prefix_csa_indices[:prefix_csa_total].fill_(-1)`
  launch from `DeepseekV4ROCMAiterMLAAttention._maybe_forward_prefill_atom`.
- Reason:
    - `write_v4_paged_prefill_indices` writes the SWA prefix part of every
    consumed CSA prefill slice.
    - `csa_translate_pack` writes the CSA topk part of every consumed CSA
    prefill slice.
    - The attention kernel reads exactly the indptr-declared consumed slice, so
    no sentinel-filled hole remains after the layout fix.
- This should remove one GPU fill/memory-write launch from CSA prefill index
  construction. It is expected to be a small helper-path win, not a numerical
  change.

Additional decode helper cleanup:

- `DeepseekV4Indexer._maybe_atom_decode_indexer_fastpath` no longer fills
  `topk_indices_buffer` with `-1` when the default all-layer/all-ratio ATOM
  attention path is active.
- Reason:
    - The ATOM CSA decode packer derives `valid_k` from `csa_indptr` and only
    reads `topk_local[t, :valid_k]`.
    - aiter `top_k_per_row_decode` only needs to write the valid head of each
    row for that consumer.
    - Native vLLM ragged conversion still treats `-1` as the tail sentinel, so
    the fill remains enabled whenever ATOM attention is disabled, layer/ratio
    filtered, or not using the ATOM unified-KV path.
- This removes a per-indexer GPU fill from the default ATOM decode fastpath
  without changing native fallback semantics.
- Validation:
    - `python3 -m py_compile vllm/models/deepseek_v4/attention.py` passed.
    - `git diff --check` passed for the changed files.
    - Import probe with default ATOM attention env:
    `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
    `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
    produced `_ATOM_INDEXER_FASTPATH_NEEDS_SENTINEL_FILL=False`.
    - Import probe with ATOM attention disabled produced
    `_ATOM_INDEXER_FASTPATH_NEEDS_SENTINEL_FILL=True`.
    - Import probe with `VLLM_ROCM_DSV4_ATOM_ATTENTION_LAYERS=0` produced
    `_ATOM_INDEXER_FASTPATH_NEEDS_SENTINEL_FILL=True`.
    - Full unchanged accuracy gate passed:
        - server: default `launchdeepseekgraph.sh`, `MAX_NUM_SEQS=256`,
      `MAX_NUM_BATCHED_TOKENS=8192`, `MAX_MODEL_LEN=8192`, no
      `--enforce-eager`.
        - command: unchanged `bash lmeval.sh`.
        - GSM8K flexible-extract exact match: `0.9538 ± 0.0058`.
        - GSM8K strict-match exact match: `0.9545 ± 0.0057`.
        - samples file:
      `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/samples_gsm8k_2026-06-19T17-18-04.498685.jsonl`.
        - logs:
      `runlogs/indexer-fill-skip-lmeval.log`,
      `runlogs/indexer-fill-skip-lmeval-server.log`.
    - Fresh C32 performance benchmark after restarting the server:
        - server: `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=8192`,
      `MAX_MODEL_LEN=8192`, no `--enforce-eager`.
        - command:
      `RESULT_DIR=./bench-indexer-fill-skip-c32`
      `RESULT_PREFIX=indexer-fill-skip`
      `CONCURRENCIES=32`
      `bash benchmarkvllm.sh`.
        - result file:
      `bench-indexer-fill-skip-c32/indexer-fill-skip-C32.json`.
        - completed `320`, failed `0`.
        - output throughput `887.220 tok/s`.
        - total throughput `1777.906 tok/s`.
        - mean TTFT `1024.548 ms`.
        - mean TPOT `35.094 ms`.
        - logs:
      `runlogs/indexer-fill-skip-c32-benchmark.log`,
      `runlogs/indexer-fill-skip-c32-server.log`.

Validation for the helper cleanup:

- Static/reference checks:
    - `python3 -m py_compile` passed for `rocm.py`,
    `paged_prefill_indices.py`, and `csa_translate_pack.py`.
    - Reference sanity verified that a CSA prefill slice initially filled with a
    stale sentinel is fully overwritten by `write_v4_paged_prefill_indices`
    plus `csa_translate_pack`.
- Fresh C32 performance server:
    - restarted after the prior benchmark
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - no `--enforce-eager`
- Performance command:
    - `RESULT_DIR=./bench-csa-prefill-nofill-c32`
    - `RESULT_PREFIX=csa-prefill-nofill`
    - `CONCURRENCIES=32`
    - `bash benchmarkvllm.sh`
- Performance result:
    - `bench-csa-prefill-nofill-c32/csa-prefill-nofill-C32.json`
    - completed `320`, failed `0`
    - output throughput `887.922 tok/s`
    - total throughput `1779.313 tok/s`
    - mean TTFT `1010.334 ms`
    - mean TPOT `35.080 ms`
    - server log: `runlogs/csa-prefill-nofill-c32-server.log`

Updated C32 comparison:

| Run | Output tok/s | Total tok/s | Mean TPOT ms | Notes |
| --- | ---: | ---: | ---: | --- |
| `csa-direct-C32.json` | `891.152` | `1785.784` | `34.945` | CSA decode compressed head resolved inside paged decode; no per-CSA decode `csa_translate_pack` |
| `indexer-fill-skip-C32.json` | `887.220` | `1777.906` | `35.094` | CSA decode top-k sentinel fill skipped in default ATOM path |
| `csa-prefill-nofill-C32.json` | `887.922` | `1779.313` | `35.080` | CSA prefill sentinel fill removed |
| `csa-prefill-layout-C32.json` | `885.850` | `1775.160` | `35.189` | CSA prefill layout fix |
| `metadata-scratch-C32-run-C32.json` | `887.515` | `1778.498` | `35.131` | Prior default after request-length scratch |
| `revert-compressor-aux-nomtp-C32.json` | `926.061` | `1855.740` | `33.503` | Historical best saved C32 |

Interpretation:

- Removing the fill was safe for the benchmark path (`0` failed requests).
- The measured gain versus the layout-only run is about `+2.07 output tok/s`
  and `-0.109 ms` mean TPOT, but this is close to normal run noise and only
  returns the result to the earlier metadata-scratch band.
- The decode top-k fill skip is accuracy-safe, but the measured C32 result is
  slightly below the prefill no-fill run and effectively in the same run-noise
  band. It should be kept as a small helper cleanup, not treated as a major
  throughput lever.
- CSA direct decode is also accuracy-safe and gives a small measured C32 win
  over the recent default/no-fill runs, roughly `+3.2` to `+3.9` output tok/s
  and `-0.13 ms` mean TPOT. The result is still below the historical best
  saved C32 run, so this validates the direction but does not close the main
  performance gap.
- This confirms the helper cleanup is directionally aligned but not a major
  lever. Larger gains still need structural removal/fusion of the prefill
  index/pack/SWA-write path.

Guarded CSA direct decode prototype:

- Added `sparse_attn_v4_paged_decode_csa_direct` behind
  `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1`.
- The direct path removes the per-CSA-layer decode `csa_translate_pack`
  launch. Instead, paged decode resolves CSA compressed-head slots in-kernel
  from:
    - the indexer's seq-local `topk_indices_buffer`;
    - vLLM's compressed block table;
    - per-token positions and batch ids;
    - existing `csa_indptr` lengths.
- The existing `write_v4_paged_decode_indices` launch remains in place and
  still writes the SWA tail into `csa_indices`. Direct CSA decode reads that
  tail from `csa_indices` and only computes the CSA head inline.
- This is now enabled by default in `launchdeepseekgraph.sh` via
  `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=${...:-1}` after accuracy and C32
  validation passed. It can still be disabled with
  `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=0`.
- Validation so far:
    - `python3 -m py_compile` passed for `paged_decode.py`,
    `v4_kernels/__init__.py`, and `rocm.py`.
    - `git diff --check` passed.
    - Import check for `sparse_attn_v4_paged_decode_csa_direct` passed.
    - CPU reference slot construction sanity passed.
    - HIP/Triton parity check against the translated-index decode path passed
    for the split-K path with `max_diff=0.0`.
    - HIP/Triton parity check against the translated-index decode path passed
    for the fused `kv_splits=1` path with `max_diff=0.0`.
    - Full unchanged accuracy gate passed:
        - server: `launchdeepseekgraph.sh` with
      `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1`, `MAX_NUM_SEQS=256`,
      `MAX_NUM_BATCHED_TOKENS=8192`, `MAX_MODEL_LEN=8192`, no
      `--enforce-eager`.
        - command: unchanged `bash lmeval.sh`.
        - GSM8K flexible-extract exact match: `0.9545 ± 0.0057`.
        - GSM8K strict-match exact match: `0.9553 ± 0.0057`.
        - result file:
      `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-19T17-48-07.259107.json`.
        - log: `runlogs/csa-direct-lmeval.log`.
    - Fresh C32 performance benchmark after restarting the server:
        - server: `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=8192`,
      `MAX_MODEL_LEN=8192`, `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1`, no
      `--enforce-eager`.
        - command:
      `RESULT_DIR=./bench-csa-direct-c32`
      `RESULT_PREFIX=csa-direct`
      `CONCURRENCIES=32`
      `bash benchmarkvllm.sh`.
        - result file:
      `bench-csa-direct-c32/csa-direct-C32.json`.
        - completed `320`, failed `0`.
        - output throughput `891.152 tok/s`.
        - total throughput `1785.784 tok/s`.
        - mean TTFT `1015.014 ms`.
        - mean TPOT `34.945 ms`.
        - median TPOT `34.805 ms`.
        - p90 TPOT `35.486 ms`.
        - p99 TPOT `35.711 ms`.
        - log: `runlogs/csa-direct-c32-benchmark.log`.

Guarded CSA direct prefill prototype:

- Added `sparse_attn_v4_paged_prefill_csa_direct` behind
  `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_PREFILL=1`.
- The direct prefill path removes the per-CSA-layer prefill
  `csa_translate_pack` launch. It keeps using
  `write_v4_paged_prefill_indices` for the SWA prefix segment, then resolves
  the CSA top-k segment inside the prefill attention kernel from:
    - the indexer's seq-local `topk_indices_buffer`;
    - vLLM's compressed block table;
    - per-token batch ids;
    - per-token `skip_prefix_len_csa`;
    - existing `prefix_csa_indptr` lengths.
- Prefill layout differs from decode:
    - decode: CSA top-k segment is at the slice head and SWA is the tail.
    - prefill: SWA prefix is at the slice head and CSA top-k follows it.
  The direct path mirrors `csa_translate_pack` exactly by deriving
  `valid_k = indptr[t + 1] - indptr[t] - skip[t]` and never reading the
  uninitialized tail of the indexer top-k buffer.
- The direct prefill wrapper intentionally uses the Triton prefill kernel.
  OPUS currently consumes materialized prefix slot ids, so using direct CSA
  prefill bypasses OPUS for CSA prefill attention.
- Validation:
    - `python3 -m py_compile` passed for `paged_prefill.py`,
    `v4_kernels/__init__.py`, and `rocm.py`.
    - `git diff --check` passed.
    - CPU reference comparison against `csa_translate_pack_reference` passed:
    materialized indices matched and attention output `max_diff=0.0`.
    - HIP/Triton parity against the materialized translated-index prefill path
    passed with `max_diff=0.0`.
    - Full unchanged accuracy gate passed:
        - server: `launchdeepseekgraph.sh` with
      `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_PREFILL=1`,
      `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1`, `MAX_NUM_SEQS=256`,
      `MAX_NUM_BATCHED_TOKENS=8192`, `MAX_MODEL_LEN=8192`, no
      `--enforce-eager`.
        - command: unchanged `bash lmeval.sh`.
        - GSM8K flexible-extract exact match: `0.9545 ± 0.0057`.
        - GSM8K strict-match exact match: `0.9560 ± 0.0056`.
        - result file:
      `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-19T18-15-51.661787.json`.
        - log: `runlogs/csa-direct-prefill-lmeval.log`.
    - Fresh C32 performance benchmark after restarting the server:
        - server: `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=8192`,
      `MAX_MODEL_LEN=8192`,
      `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_PREFILL=1`, no `--enforce-eager`.
        - command:
      `RESULT_DIR=./bench-csa-direct-prefill-c32`
      `RESULT_PREFIX=csa-direct-prefill`
      `CONCURRENCIES=32`
      `bash benchmarkvllm.sh`.
        - result file:
      `bench-csa-direct-prefill-c32/csa-direct-prefill-C32.json`.
        - completed `320`, failed `0`.
        - output throughput `890.041 tok/s`.
        - total throughput `1783.559 tok/s`.
        - mean TTFT `991.613 ms`.
        - mean TPOT `35.013 ms`.
        - median TPOT `34.927 ms`.
        - p90 TPOT `35.573 ms`.
        - p99 TPOT `35.823 ms`.
        - log: `runlogs/csa-direct-prefill-c32-benchmark.log`.
- Interpretation:
    - Correctness is good and the path runs without `--enforce-eager`.
    - It is not a C32 performance win versus decode-only direct CSA:
    `890.041` output tok/s versus `891.152`, and `35.013` ms mean TPOT versus
    `34.945`.
    - The likely reason is that prefill direct CSA removes `csa_translate_pack`
    but bypasses OPUS for CSA prefill attention. The launch saved is smaller
    than the OPUS-to-Triton attention cost for this workload.
    - Keep the flag as an experimental probe. Do not enable it by default unless
    there is an OPUS direct-index variant or a faster direct-prefill kernel.

ATOM compress-first decode ordering probe:

- Tested `VLLM_ROCM_DSV4_ATOM_COMPRESS_FIRST=1` as a flag-only experiment.
- This path is intentionally limited to pure decode in
  `DeepseekV4ROCMAiterMLAAttention.attention_impl`:
    - mixed/prefill batches still use the existing validated path;
    - pure decode calls the main compressor before the Q/KV attention path,
    matching the order used by ATOM's modeling file more closely;
    - the path requires ATOM QK/RoPE, ATOM main compressor, ATOM attention, no
    aux streams, and an ATOM-enabled layer/ratio.
- Accuracy validation:
    - server: `MAX_NUM_SEQS=256`, `MAX_NUM_BATCHED_TOKENS=8192`,
    `MAX_MODEL_LEN=8192`, `ENFORCE_EAGER=0`,
    `VLLM_ROCM_DSV4_ATOM_COMPRESS_FIRST=1`, no `--enforce-eager`.
    - command: unchanged `bash lmeval.sh`.
    - result file:
    `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-19T18-38-39.854574.json`.
    - log: `runlogs/compress-first-lmeval.log`.
    - GSM8K flexible-extract exact match: `0.9545 ± 0.0057`.
    - GSM8K strict-match exact match: `0.9553 ± 0.0057`.
- Fresh C32 deployment benchmark after restarting the server:
    - server: `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=8192`,
    `MAX_MODEL_LEN=8192`, `ENFORCE_EAGER=0`,
    `VLLM_ROCM_DSV4_ATOM_COMPRESS_FIRST=1`, no `--enforce-eager`.
    - command:
    `RESULT_DIR=./bench-compress-first-c32`
    `RESULT_PREFIX=compress-first`
    `CONCURRENCIES=32`
    `bash benchmarkvllm.sh`.
    - result file: `bench-compress-first-c32/compress-first-C32.json`.
    - completed `320`, failed `0`.
    - output throughput `889.928 tok/s`.
    - total throughput `1783.332 tok/s`.
    - mean TTFT `1037.166 ms`.
    - mean TPOT `34.972 ms`.
    - median TPOT `34.943 ms`.
    - p90 TPOT `35.439 ms`.
    - p99 TPOT `35.804 ms`.
    - log: `runlogs/compress-first-c32-benchmark.log`.
- Interpretation:
    - Compress-first decode ordering is accuracy-safe and graph-compatible in
    this configuration.
    - It is not a throughput win versus the current decode-direct default:
    `889.928` output tok/s versus `891.152` for
    `bench-csa-direct-c32/csa-direct-C32.json`.
    - It is also below the historical best saved C32 run
    `bench-sparsemla/revert-compressor-aux-nomtp-C32.json`
    (`926.061` output tok/s, `33.503 ms` mean TPOT).
    - Keep `VLLM_ROCM_DSV4_ATOM_COMPRESS_FIRST` default-off for now. The result
    says ordering alone is not the missing performance lever; larger gains
    likely require reducing metadata/conversion overhead and/or introducing a
    true ROCm DSV4 unified cache/backend contract instead of reordering the
    already-integrated compressor call.

Scheduler visibility for ROCm ATOM unified KV prefix:

- Added a small scheduler-config preservation fix for the vLLM-owned ATOM
  unified-KV path.
- Background:
    - ROCm DSV4 ATOM unified mode emits `DeepseekV4AtomMLAAttentionSpec` for
    compressed attention layers when
    `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`.
    - That spec models the vLLM-owned tensor as:
    `[fixed SWA prefix][paged compressed tail]`.
    - Worker allocation and reshape already account for the fixed prefix through
    `KVCacheTensor.fixed_prefix_size` and by skipping the prefix before
    reshaping the paged tail.
    - `generate_scheduler_kv_cache_config()` still collapsed
    `UniformTypeKVCacheSpecs` by taking an arbitrary first per-layer spec.
    If that first spec was a regular `MLAAttentionSpec`, the scheduler-facing
    config forgot the ATOM fixed-prefix metadata.
- Change:
    - Register `DeepseekV4AtomMLAAttentionSpec` with the same KV-cache manager
    and uniform base as regular `MLAAttentionSpec`.
    - Add `_representative_scheduler_spec()` so collapsed scheduler specs prefer
    an ATOM spec when one exists in the per-layer group.
- Validation:
    - `python3 -m py_compile` passed for:
        - `vllm/v1/core/kv_cache_utils.py`
        - `vllm/v1/core/single_type_kv_cache_manager.py`
        - `vllm/v1/kv_cache_interface.py`
        - `vllm/v1/worker/gpu/attn_utils.py`
        - `vllm/v1/worker/utils.py`
    - Registry smoke:
        - regular `MLAAttentionSpec` and `DeepseekV4AtomMLAAttentionSpec` both
      resolve to uniform base `FullAttentionSpec`;
        - a collection containing one regular MLA spec and one ATOM MLA spec is
      accepted as uniform.
    - Scheduler smoke:
        - a synthetic `KVCacheConfig` with a mixed `UniformTypeKVCacheSpecs` group
      now collapses to `DeepseekV4AtomMLAAttentionSpec` and preserves
      `atom_swa_prefix_bytes`.
    - Real startup smoke:
        - server command: `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=8192`,
      `MAX_MODEL_LEN=8192`, `ENFORCE_EAGER=0`, default
      `launchdeepseekgraph.sh` ATOM flags.
        - log: `runlogs/scheduler-atom-prefix-smoke-server.log`.
        - scheduler reported KV cache size `45,496` tokens and max concurrency
      `5.55x` for `8192` tokens/request.
        - ATOM state allocation completed on all TP ranks.
        - vLLM-owned unified KV views bound with `active_layers=61`,
      `ratio_counts={128: 31, 4: 30}`, `num_blocks=57732`,
      `swa_pages=4096`, `win_with_spec=128`, `head_dim=512`, and
      `dtype=torch.bfloat16`.
        - graph capture finished and `/health` returned healthy.
- Interpretation:
    - This does not change the worker allocation that the successful accuracy and
    C32 benchmark already used.
    - It closes a correctness gap in the vLLM scheduler-facing cache metadata,
    moving the integration closer to a real ROCm DSV4 unified KV cache spec
    rather than a purely side-band model-state view.
    - CUDA remains unaffected because the ATOM spec is only emitted by the
    DeepSeek-V4 attention layer under ROCm plus the ATOM unified-KV flag.

Post scheduler-prefix fix accuracy and C32 benchmark:

- Accuracy command: unchanged `/app/atomdsv4/lmeval.sh`.
- Server command:
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `ENFORCE_EAGER=0`
    - default `/app/atomdsv4/launchdeepseekgraph.sh` ATOM flags.
- Accuracy log:
    - `runlogs/post-scheduler-prefix-lmeval.log`.
    - result file:
    `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-19T19-04-45.509823.json`.
- Accuracy result:
    - GSM8K flexible-extract exact_match: `0.9545 +/- 0.0057`.
    - GSM8K strict-match exact_match: `0.9553 +/- 0.0057`.
- Fresh benchmark server command:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `ENFORCE_EAGER=0`
    - default `/app/atomdsv4/launchdeepseekgraph.sh` ATOM flags.
- Benchmark command:
    - `RESULT_DIR=./bench-post-scheduler-prefix-c32`
    - `RESULT_PREFIX=post-scheduler-prefix`
    - `CONCURRENCIES=32`
    - `/app/atomdsv4/benchmarkvllm.sh`.
- Benchmark files:
    - server log: `runlogs/post-scheduler-prefix-c32-server.log`.
    - benchmark log: `runlogs/post-scheduler-prefix-c32-benchmark.log`.
    - result file:
    `bench-post-scheduler-prefix-c32/post-scheduler-prefix-C32.json`.
- Benchmark result:
    - completed `320`, failed `0`.
    - output throughput `890.326 tok/s`.
    - total throughput `1784.129 tok/s`.
    - mean TTFT `992.152 ms`.
    - mean TPOT `35.001 ms`.
    - median TPOT `34.904 ms`.
    - p90 TPOT `35.481 ms`.
    - p99 TPOT `35.623 ms`.
- Interpretation:
    - Accuracy remains inside the target range after preserving ATOM fixed-prefix
    metadata in the scheduler-facing KV-cache config.
    - C32 throughput is effectively flat against the previous CSA-direct decode
    default run (`891.152` output tok/s, `34.945 ms` mean TPOT), which is
    expected because this fix mainly corrects scheduler metadata visibility
    rather than changing the hot attention/compressor kernels.
    - The historical best saved C32 run remains
    `bench-sparsemla/revert-compressor-aux-nomtp-C32.json`
    (`926.061` output tok/s, `33.503 ms` mean TPOT).

Worker kernel-block-size propagation for ROCm ATOM unified KV:

- Follow-up gap:
    - `generate_scheduler_kv_cache_config()` now preserves the ATOM fixed-prefix
    metadata when collapsing a `UniformTypeKVCacheSpecs` group.
    - `prepare_kernel_block_sizes()` in the worker still picked an arbitrary
    layer spec from `UniformTypeKVCacheSpecs` to decide how to dispatch block
    splitting. If that arbitrary spec was a regular `MLAAttentionSpec`, the
    worker-side setup did not explicitly see the prefix-aware
    `DeepseekV4AtomMLAAttentionSpec`.
- Change:
    - Add `_representative_worker_spec()` in `vllm/v1/worker/utils.py`.
    - When a uniform group contains any `DeepseekV4AtomMLAAttentionSpec`, worker
    block-size preparation now uses that spec as the representative instead of
    relying on insertion order.
- Validation:
    - `python3 -m py_compile vllm/v1/worker/utils.py` passed.
    - Direct helper smoke forced a regular MLA spec to appear before an ATOM MLA
    spec and confirmed `_representative_worker_spec()` returned
    `DeepseekV4AtomMLAAttentionSpec` with the expected
    `atom_swa_prefix_bytes`.
    - `prepare_kernel_block_sizes()` smoke with the same mixed uniform group and
    a dummy backend returned `[128]`, proving the real worker helper executes
    correctly with the ATOM representative.
- Interpretation:
    - This is another integration hardening change for the vLLM-owned ATOM
    unified KV path.
    - It is not expected to move C32 throughput by itself; it prevents a metadata
    visibility mismatch as the custom ROCm DSV4 KV-cache spec is used in more
    worker/backend code.

Order-independent mixed ATOM/non-ATOM MLA grouping:

- Follow-up gap:
    - `DeepseekV4AtomMLAAttentionSpec` is a subclass of `MLAAttentionSpec`, and
    the registry intentionally lets it participate in full-attention uniform
    type grouping.
    - `is_kv_cache_spec_uniform()` probes concrete uniformity by calling
    `values[0].merge(values)`.
    - With a regular `MLAAttentionSpec` first and an ATOM spec second,
    `MLAAttentionSpec.merge()` accepted the mixed set and returned a regular
    MLA spec. That incorrectly classified the whole group as one concrete KV
    spec and could drop the per-layer ATOM SWA prefix metadata.
    - With the ATOM spec first, `DeepseekV4AtomMLAAttentionSpec.merge()` rejected
    the mixed set, so behavior was insertion-order dependent.
- Change:
    - `is_kv_cache_spec_uniform()` now returns `False` when a set mixes
    `DeepseekV4AtomMLAAttentionSpec` with non-ATOM specs.
    - Pure ATOM groups are still allowed to use the concrete uniform path.
    - Mixed regular/ATOM MLA groups now consistently fall through to
    `UniformTypeKVCacheSpecs`, where the per-layer ATOM spec and prefix fields
    are preserved.
- Validation:
    - Synthetic smoke before the fix:
        - regular-first mixed group: `is_kv_cache_spec_uniform(...) == True`;
        - atom-first mixed group: `False`.
    - Synthetic smoke after the fix:
        - regular-first mixed group: `False`;
        - atom-first mixed group: `False`;
        - `get_kv_cache_groups()` returns one `UniformTypeKVCacheSpecs` group and
      preserves `DeepseekV4AtomMLAAttentionSpec` for the ATOM layer.
    - All-ATOM control remains concrete-uniform: `True`.
- Interpretation:
    - This closes an order-dependent correctness hole in the ROCm unified-KV spec
    path.
    - CUDA remains unaffected unless the ROCm-only ATOM spec is present.

Fixed-prefix memory sizing for ATOM unified KV:

- Follow-up gap:
    - The ATOM unified-KV allocation has two parts:
    `[fixed SWA prefix][paged compressed tail]`.
    - The allocator path handled this layout for the main validated multi-group
    DeepSeek-V4 case, but related sizing paths still reasoned mostly in terms
    of scalable page bytes.
    - A single `UniformTypeKVCacheSpecs` group containing an ATOM spec could take
    the legacy single-group allocation path and allocate only
    `page_size * num_blocks`, dropping `atom_swa_prefix_bytes`.
    - `num_gpu_blocks_override` also adjusted effective memory using only
    `override * bytes_per_block`, which is incomplete when fixed prefixes are
    present.
- Change:
    - Add `_split_deepseek_v4_atom_layers()` and
    `_deepseek_v4_atom_layout_bytes()` so allocation, bytes-per-block, and
    max-memory estimation share the same ATOM-aware layout split.
    - Route any `UniformTypeKVCacheSpecs` group containing
    `DeepseekV4AtomMLAAttentionSpec` through the ATOM-aware allocator, including
    the single-group case.
    - `_max_memory_usage_bytes_from_groups()` now computes:
    `fixed_prefix_bytes + max_tail_pages * bytes_per_block`.
    - `num_gpu_blocks_override` effective memory now includes the fixed prefix:
    `fixed_prefix_bytes + override * bytes_per_block`.
- Validation:
    - Synthetic single-group mixed regular MLA + ATOM MLA config:
        - bytes per block: `128`;
        - max memory for `max_model_len=16`: `612`
      (`100 + 4 * 128`);
        - allocated `num_blocks=7` from `available_memory=1000`;
        - regular tensor: `448` bytes, fixed prefix `0`;
        - ATOM tensor: `548` bytes, fixed prefix `100`.
    - Synthetic `num_gpu_blocks_override=4` through `get_kv_cache_configs()`:
        - override log reports old block count as `77`, i.e.
      `(10000 - 100) // 128`, after subtracting the fixed prefix;
        - generated `num_blocks=4`;
        - regular tensor: `256` bytes;
        - ATOM tensor: `356` bytes, fixed prefix `100`.
- Interpretation:
    - This makes fixed-prefix SWA storage part of the KV cache sizing contract,
    not only a post-allocation model-state binding detail.
    - It is another prerequisite for a real ROCm DSV4 unified KV allocation while
    still leaving CUDA untouched unless the ATOM spec is emitted.

Capacity reporting for ATOM fixed-prefix KV cache:

- Follow-up gap:
    - `get_max_concurrency_for_kv_cache_config()` used a generic memory formula:
    `num_layer_per_group * max_memory_usage_bytes(...)`.
    - For ATOM unified KV this double-counted mixed per-layer uniform specs and
    treated fixed SWA prefix bytes as if they scaled with scheduler blocks.
    - Synthetic example before the fix:
        - `num_blocks=4`, semantic `block_size=4`, `max_model_len=16`;
        - one request needs four scheduler blocks;
        - old capacity reported `10` tokens and `0.67x` concurrency.
- Change:
    - Add an ATOM-only capacity path selected when a
    `DeepseekV4AtomMLAAttentionSpec` is present.
    - Capacity now computes required scheduler blocks per request from each KV
    group's scalable tail pages, subtracting `atom_swa_prefix_bytes` from ATOM
    specs.
    - Max concurrency is limited by the largest per-group scalable block demand.
    - Non-ATOM configs keep the existing generic capacity formula.
- Validation:
    - Synthetic mixed regular MLA + ATOM MLA config with
    `num_gpu_blocks_override=4` now logs:
        - GPU KV cache size `16` tokens;
        - maximum concurrency for `16` tokens/request: `1.00x`.
    - Synthetic two-group config with an ATOM-containing group plus a smaller
    block-size regular group also reports `16` tokens and `1.00x`, confirming
    the limiter is the max scalable block demand across groups.
- Interpretation:
    - This makes startup capacity reporting consistent with vLLM's scheduler
    block-table lifetime while excluding ATOM's fixed SWA prefix from scalable
    block accounting.
    - It does not affect CUDA or non-ATOM KV cache configs.

Real startup smoke after fixed-prefix sizing/capacity changes:

- Server command:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `ENFORCE_EAGER=0`
    - default `/app/atomdsv4/launchdeepseekgraph.sh` ATOM flags.
- Log:
    - `runlogs/post-kv-capacity-smoke-server.log`.
- Observed:
    - ATOM request-state buffers allocated on all TP ranks:
    `active_layers=61`, `csa_layers=30`, `hca_layers=31`,
    `win_with_spec=128`.
    - Engine reported KV capacity:
        - GPU KV cache size `115,435` tokens;
        - max concurrency for `8,192` tokens/request: `14.09x`.
    - vLLM-owned unified KV views bound on all TP ranks:
    `active_layers=61`, `ratio_counts={128: 31, 4: 30}`,
    `num_blocks=57732`, `swa_pages=4096`, `win_with_spec=128`,
    `head_dim=512`, `dtype=torch.bfloat16`.
    - Piecewise and full graph capture completed.
    - `/health` returned HTTP `200`.
- Interpretation:
    - The fixed-prefix KV sizing and ATOM capacity path are compatible with the
    real DSV4 C32 startup path, graph capture, and ModelState binding.
    - No `--enforce-eager` was used.

Projected empty-group robustness for ATOM capacity:

- Follow-up gap:
    - Pipeline-parallel projection can preserve KV cache groups whose layer list
    is empty on a particular worker.
    - The ATOM-specific capacity path originally used `max(...)` over each
    group's per-layer specs and would fail if an empty projected group was
    present before or beside an ATOM-containing group.
- Change:
    - Skip empty `KVCacheGroupSpec.layer_names` entries in the ATOM capacity
    helper.
- Validation:
    - Synthetic KV config with one empty `UniformTypeKVCacheSpecs` group and one
    ATOM-containing group reports capacity `(16, 1.0)` for
    `num_blocks=4`, `block_size=4`, `max_model_len=16`.
- Interpretation:
    - This keeps the ATOM capacity path compatible with vLLM's existing PP group
    projection behavior.

Unit coverage for ATOM fixed-prefix KV cache contract:

- Added focused CPU tests in `tests/v1/core/test_kv_cache_utils.py`.
- Coverage:
    - mixed regular MLA + ATOM MLA uniformity is insertion-order independent;
    - pure ATOM groups still remain concrete-uniform;
    - ATOM capacity excludes fixed SWA prefix bytes from scalable scheduler-block
    accounting;
    - ATOM capacity skips empty projected KV groups;
    - single `UniformTypeKVCacheSpecs` groups containing an ATOM spec allocate
    `KVCacheTensor.fixed_prefix_size`;
    - `num_gpu_blocks_override` keeps the fixed prefix and clamps only the
    scalable paged tail;
    - scheduler KV-cache config generation preserves the ATOM MLA spec instead
    of collapsing mixed regular/ATOM MLA groups to regular MLA;
    - worker kernel-block-size dispatch uses the ATOM MLA spec as the
    representative for mixed regular/ATOM MLA `UniformTypeKVCacheSpecs`.
- Validation:
    - `pytest -q tests/v1/core/test_kv_cache_utils.py -k 'atom_mla or scheduler_kv_cache_config_preserves_atom_spec'`:
    `6 passed`.
    - `pytest -q tests/v1/worker/test_utils.py -k 'representative_worker_spec or bind_kv_cache'`:
    `4 passed`.
    - `pytest -q tests/v1/core/test_kv_cache_utils.py`:
    `64 passed`.
    - `python3 -m py_compile vllm/v1/core/kv_cache_utils.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/kv_cache_interface.py vllm/v1/worker/gpu/attn_utils.py vllm/v1/worker/utils.py tests/v1/core/test_kv_cache_utils.py tests/v1/worker/test_utils.py`:
    passed.
    - `git diff --check`: passed.
- Interpretation:
    - The fixed-prefix KV cache behavior is now covered by durable unit tests,
    not just ad hoc smoke scripts.
    - The ATOM spec now survives the scheduler and worker representative-spec
    collapse points that previously could erase ROCm-only fixed-prefix layout
    metadata when a group also contained regular MLA.

## 2026-06-19 Current No-Eager Validation After ATOM KV Contract Tests

Runtime configuration:

- Server command:
    - `MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 ENFORCE_EAGER=0 bash ./launchdeepseekgraph.sh`
    - log: `/app/atomdsv4/runlogs/current-no-eager-server.log`
- Relevant launch defaults:
    - `VLLM_USE_V2_MODEL_RUNNER=1`
    - `VLLM_ROCM_DSV4_ATOM_STATE=1`
    - `VLLM_ROCM_DSV4_ATOM_STATE_ALLOC=1`
    - `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`
    - `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
    - `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
    - `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
    - `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
    - no `--enforce-eager`.
- Startup evidence:
    - `enforce_eager=False`
    - `GPU KV cache size: 114,218 tokens`
    - `Maximum concurrency for 8,192 tokens per request: 13.94x`
    - `Bound ROCm DSV4 ATOM unified KV views from vLLM-owned KV storage`
    with `ratio_counts={128: 31, 4: 30}`, `num_blocks=57123`,
    `swa_pages=32768`, `head_dim=512`, `dtype=torch.bfloat16`.
    - graph capture finished for all TP workers.

Accuracy validation:

- Command: unchanged `/app/atomdsv4/lmeval.sh`.
- Log: `/app/atomdsv4/runlogs/current-no-eager-lmeval.log`.
- Result file prefix:
  `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/`.
- GSM8K result:
    - flexible-extract exact match: `0.9507 +/- 0.0060`.
    - strict-match exact match: `0.9515 +/- 0.0059`.
- Interpretation:
    - This passes the requested `0.95 +/- 0.01` GSM8K accuracy band while
    running without `--enforce-eager`.
    - First-inference JIT warnings show the current path exercised ATOM-related
    metadata/compressor/prefill kernels including
    `_update_compressor_states_kernel`, `_v4_paged_prefill_indices_kernel`,
    and `_csa_translate_pack_kernel`.

Fresh C32 benchmark after server restart:

- The lmeval server was stopped first; `/health` went down before launching the
  benchmark server.
- Benchmark server command:
    - `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 ENFORCE_EAGER=0 bash ./launchdeepseekgraph.sh`
    - log: `/app/atomdsv4/runlogs/current-c32-benchmark-server.log`
- Startup evidence:
    - `GPU KV cache size: 115,435 tokens`.
    - `Maximum concurrency for 8,192 tokens per request: 14.09x`.
    - `Bound ROCm DSV4 ATOM unified KV views from vLLM-owned KV storage`
    with `ratio_counts={128: 31, 4: 30}`, `num_blocks=57732`,
    `swa_pages=4096`, `head_dim=512`, `dtype=torch.bfloat16`.
    - graph capture finished for all TP workers.
- Benchmark command:
    - `RESULT_PREFIX=ds-v4-pro-current-noeager-atomkv CONCURRENCIES=32 bash ./benchmarkvllm.sh`
    - log: `/app/atomdsv4/runlogs/current-c32-benchmark.log`
    - result:
    `/app/atomdsv4/bench-sparsemla/ds-v4-pro-current-noeager-atomkv-C32.json`
- C32 result:
    - successful requests: `320`
    - failed requests: `0`
    - output throughput: `890.587 tok/s`
    - total token throughput: `1784.653 tok/s`
    - mean TPOT: `34.982 ms`
    - median TPOT: `34.922 ms`
    - p99 TPOT: `35.797 ms`
    - mean TTFT: `998.798 ms`

Comparison with saved C32 runs:

| Run | Output tok/s | Total tok/s | Mean TPOT ms | Notes |
| --- | ---: | ---: | ---: | --- |
| `revert-compressor-aux-nomtp-C32.json` | `926.061` | `1855.740` | `33.503` | Historical best saved C32 |
| `ds-v4-pro-nomtp-compressor-order-off-C32.json` | `925.131` | `1853.875` | `33.502` | Previous no-compressor-order experiment |
| `post-scheduler-prefix-C32.json` | `890.326` | `1784.126` | `35.003` | Previous validated post scheduler-prefix run |
| `ds-v4-pro-current-noeager-atomkv-C32.json` | `890.587` | `1784.653` | `34.982` | Current run after ATOM KV contract tests |

Interpretation:

- Current accuracy is correct and the current no-eager C32 performance is
  effectively flat versus the previous post-scheduler-prefix validated run.
- The current run is not the best saved C32 result. It is about `3.83%` below
  `revert-compressor-aux-nomtp-C32.json` by output throughput.
- The remaining performance gap is therefore not explained by the fixed-prefix
  KV-cache contract work itself. The next useful profiling target is the
  recurring per-step/prefill metadata and state-update path shown by the JIT
  monitor: `_compute_prefill_metadata_kernel`, `_compute_swa_indices_and_lens_kernel`,
  `_update_compressor_states_kernel`, `_v4_paged_prefill_indices_kernel`, and
  `_csa_translate_pack_kernel`.

## 2026-06-19 AITER MHC Default Alignment

Follow-up gap:

- ATOM's DSV4 block sequence uses MHC around every attention and FFN sublayer:
  `hc_pre -> attn_norm -> attention/compressor/indexer/sparse_attn -> hc_post`
  and then `hc_pre -> ffn_norm -> moe/ffn -> hc_post`.
- MHC is not a direct dependency for the ATOM attention/compressor kernels:
  those depend on Q/KV layout, unified KV/SWA/compressor state layout, index
  metadata, compressor update ordering, and scheduler metadata.
- MHC is still relevant for matching the full ATOM modeling-file op sequence
  and performance profile because it changes the hidden activations entering
  attention and FFN.
- The installed `aiter==0.1.15.post1` exposes `mhc_pre` and `mhc_post`, but
  not `mhc_fused_post_pre`. With this aiter build, ATOM itself would fall back
  to separate aiter `hc_post` then `hc_pre` between sublayers rather than an
  aiter fused post-pre kernel.

Experiment:

- Temporarily defaulted `/app/atomdsv4/launchdeepseekgraph.sh` to:
    - `VLLM_ROCM_DSV4_USE_AITER_MHC=${VLLM_ROCM_DSV4_USE_AITER_MHC:-1}`
    - `VLLM_ROCM_DSV4_USE_AITER_HC_HEAD=${VLLM_ROCM_DSV4_USE_AITER_HC_HEAD:-1}`
- vLLM already selects `_forward_unfused_post_pre` when `HAS_AITER_MHC=True`,
  so enabling the flag switches the ROCm block path to separate aiter
  `MHCPreOp` and `MHCPostOp`, which is the closest ATOM-compatible path
  available in this aiter version.

Validation:

- Import check with both flags set:
    - `HAS_AITER_MHC=True`
    - `HAS_AITER_HC_HEAD=True`
    - `HAS_TILELANG=True`
- `python3 -m py_compile vllm/model_executor/layers/mhc.py vllm/models/deepseek_v4/amd/model.py` passed.
- Direct DSV4-shaped torch-op smoke on GPU passed:
    - `torch.ops.vllm.mhc_pre_aiter` on `(1, 4, 7168)` bf16 residual and
    `(24, 28672)` fp32 `hc_fn`.
    - `torch.ops.vllm.mhc_post_aiter` on the returned `post/comb`.
    - `torch.ops.vllm.hc_head_aiter` on the resulting `(1, 4, 7168)` bf16 HC
    residual and `(4, 28672)` fp32 final-head `hc_fn`.
    - Outputs:
        - `post`: `(1, 4, 1)`, fp32.
        - `comb`: `(1, 4, 4)`, fp32.
        - pre output: `(1, 7168)`, bf16.
        - post output: `(1, 4, 7168)`, bf16.
        - HC-head output: `(1, 7168)`, bf16.
- No-eager server startup with the aiter MHC/HC-head defaults passed:
    - command:
    `MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 ENFORCE_EAGER=0 bash ./launchdeepseekgraph.sh`
    - log: `/app/atomdsv4/runlogs/aiter-mhc-noeager-server.log`
    - `enforce_eager=False`
    - V2 model runner
    - ATOM state buffers allocated
    - ATOM unified KV views bound from vLLM-owned storage:
    `num_blocks=57180`, `swa_pages=32768`,
    `ratio_counts={128: 31, 4: 30}`.
    - graph capture completed and `/health` returned `200`.
- Accuracy with unchanged `/app/atomdsv4/lmeval.sh` failed:
    - log: `/app/atomdsv4/runlogs/aiter-mhc-noeager-lmeval.log`
    - result:
    `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-19T20-22-53.418611.json`
    - GSM8K flexible-extract exact match: `0.1357 +/- 0.0094`.
    - GSM8K strict-match exact match: `0.1228 +/- 0.0090`.
- Reference comparison:
    - `torch.ops.vllm.mhc_pre_tilelang` and `mhc_post_tilelang` match the
    vLLM torch reference closely on random `(8, 4, 7168)` DSV4-shaped inputs:
    `tile_y` max abs diff `0.0078125`, mean `3.57e-06`;
    `tile_out` max abs diff `0.03125`, mean `7.44e-06`.
    - `torch.ops.vllm.mhc_pre_aiter` and `mhc_post_aiter` differ materially on
    the same shape:
    `pre_y` max abs diff `0.453125`, mean `0.03897`;
    `post_out` max abs diff `0.9375`, mean `0.06889`.
- HC-head-only isolation:
    - Server command:
    `MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 ENFORCE_EAGER=0 VLLM_ROCM_DSV4_USE_AITER_MHC=0 VLLM_ROCM_DSV4_USE_AITER_HC_HEAD=1 bash ./launchdeepseekgraph.sh`
    - Server log:
    `/app/atomdsv4/runlogs/aiter-hchead-only-noeager-server.log`
    - Startup passed with no `--enforce-eager`, V2 runner, breakable CUDA graph,
    ATOM state allocation, vLLM-owned unified KV binding, and graph capture.
    - Accuracy command: unchanged `/app/atomdsv4/lmeval.sh`.
    - Accuracy log:
    `/app/atomdsv4/runlogs/aiter-hchead-only-noeager-lmeval.log`
    - Result:
    `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-19T20-38-12.415293.json`
    - GSM8K flexible-extract exact match: `0.9348 +/- 0.0068`.
    - GSM8K strict-match exact match: `0.9356 +/- 0.0068`.
    - Interpretation:
    final HC-head aiter alone is much less destructive than full aiter MHC,
    but still misses the requested `0.95 +/- 0.01` accuracy band.
- HC-head standalone observations:
    - `torch.ops.vllm.hc_head_aiter` with real final-head scale shape `[1]`
    runs in isolated calls for tested batch sizes `M=1,2,4,8,16,32,64`.
    - In a mixed-process comparison that calls tilelang HC-head and then aiter
    HC-head, the process hit a floating point exception. This suggests the
    compact aiter HC-head path is not safe to mix with the current tilelang
    HC-head test path without more isolation.
    - In an aiter-only comparison against the scalar PyTorch HC-head formula on
    `(16, 4, 7168)`, `hc_head_aiter` differed materially:
    max abs diff `0.4453125`, mean `0.02690`, relative max `0.0963`.

Interpretation:

- AITER MHC/HC-head is runnable, but it is not accuracy-safe in the current
  vLLM integration.
- The launch defaults were restored to:
    - `VLLM_ROCM_DSV4_USE_AITER_MHC=${VLLM_ROCM_DSV4_USE_AITER_MHC:-0}`
    - `VLLM_ROCM_DSV4_USE_AITER_HC_HEAD=${VLLM_ROCM_DSV4_USE_AITER_HC_HEAD:-0}`
- The flags remain available for explicit experiments, but the validated
  no-eager accuracy/performance path should keep using tilelang MHC until the
  AITER-vs-reference mismatch is understood.
- No C32 benchmark was collected for the failed aiter MHC default because the
  accuracy gate failed far outside the required `0.95 +/- 0.01` GSM8K band.
- No C32 benchmark was collected for the HC-head-only experiment because it
  also failed the accuracy gate.
