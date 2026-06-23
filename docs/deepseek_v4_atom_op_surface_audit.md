# DeepSeek V4 ATOM Op Surface Audit

Date: 2026-06-20

This audit compares `/app/atomdsv4/ATOM/atom/models/deepseek_v4.py` with the
current ROCm DeepSeek V4 path in vLLM. It answers a narrow question: do we have
the named ATOM components needed to benefit from the ATOM modeling file inside
vLLM's scheduler?

Short answer: vLLM now has enough components to run the ATOM-shaped
attention/compressor sequence with vLLM-owned KV and no `atom` package
dependency. It does not yet have every native ATOM kernel benefit. The remaining
gap is not one missing Python call; it is the native packed DSV4 sparse
attention/compressor ABI and metadata contract that would let kernels consume
vLLM scheduler state without split-layout adaptation.

## Component Verdict Matrix

| Requirement | Verdict | Evidence | Implication |
| --- | --- | --- | --- |
| vLLM scheduler and V2 model runner | Present | The ROCm DeepSeek-V4 path runs through vLLM's scheduler-facing model and KV-cache interfaces. | No GPU worker rewrite is needed for request-state experiments. |
| vLLM weight loading and model ownership | Present | The integrated model keeps vLLM module ownership and the audit verifies no runtime `atom` imports. | Weight loading can continue to use vLLM strategies such as fast safetensors and streamer paths. |
| ROCm ModelState request rings | Present | `DeepseekV4RocmAtomModelState` owns per-request SWA, compressor, and indexer state when the ROCm ATOM state flag is enabled. | Persistent request state can live in the model runner layer without changing CUDA. |
| CUDA and NVIDIA isolation | Present | Static contracts verify NVIDIA DeepSeek-V4 files do not import ROCm ATOM state, kernels, or custom KV-spec symbols. | CUDA can keep the existing KV-cache path. |
| vLLM-owned packed KV spec/allocation | Present, split layout | `DeepseekV4AtomMLAAttentionSpec` models a fixed BF16 SWA prefix plus packed `fp8_ds_mla` compressed tail. | vLLM can allocate the deployed ROCm cache, but it is not ATOM's homogeneous unified tensor ABI. |
| ATOM attention/compressor operation order | Mostly present | The ROCm path has ATOM-shaped Q/KV normalization, SWA write ordering, compressor ordering, CSA translation, decode, and prefill. | The sequence is good enough for accuracy/perf experiments with compatibility wrappers. |
| Native packed sparse attention ABI | Missing | Installed aiter/OPUS sparse attention APIs do not accept `packed_fp8_ds_mla`, split-KV, 584-byte slots, or embedded scale layout. | vLLM cannot yet get the full native ATOM attention benefit on vLLM-owned packed KV. |
| Native packed compressor ABI | Missing | Public flydsl compressor APIs do not expose the deployed `fp8_ds_mla` packed-tail scatter contract. | Packed main compressor writes use local compatibility kernels instead of a native ATOM/aiter packed-tail writer. |
| Full ATOM indexer dispatcher | Partial | vLLM exposes lower-level pieces and an opt-in decode fast path, but not the default ATOM prefill gather/logits/top-k sequence. | Indexer parity needs a deeper dispatcher integration or a native wrapper matching ATOM's compressed-cache contract. |
| aiter MHC/HC | Not production-ready | `aiter.mhc_pre` and `aiter.mhc_post` exist, but enabling them in vLLM failed GSM8K accuracy. | MHC is model-equivalence work, not the current attention/compressor ABI blocker. |
| aiter `mhc_fused_post_pre` | Missing | The installed `aiter==0.1.15.post1` does not export this symbol. | Exact ATOM aiter fused post/pre parity needs a new aiter export or a vLLM-owned equivalent. |
| ATOM auxiliary stream and MoE overlap | Missing | vLLM AMD disables ROCm attention aux streams and does not call `torch.ops.aiter.maybe_dual_stream_forward`. | Full ATOM overlap benefit is still absent even when core attention/compressor order matches. |

## Native ABI Integration Target

There are two viable ways to move from the current compatibility path to full
native ATOM attention/compressor benefit. Both keep vLLM's scheduler and keep
CUDA on the existing path.

### Target A: Native split-packed ABI

Keep the current vLLM-owned packed allocation:

- SWA prefix: BF16/model-dtype `[max_num_reqs, window + spec, 512]`.
- Compressed tail: `uint8 [num_blocks, k_per_block, 584]`.
- Layout name: `fp8_ds_mla`.
- Slot contract: 448 FP8 NoPE bytes, 64 BF16 RoPE values, and 8 embedded
  UE8M0 scale bytes per compressed token.

The missing native entry points would consume exactly the split views already
bound by `DeepseekV4RocmAtomModelState`:

- `atom_split_kv_swa`
- `atom_split_kv_compressed`
- `atom_split_kv_scales=None`
- `atom_split_kv_layout="fp8_ds_mla"`
- `swa_pages`
- vLLM block tables and CSA/HCA physical slot metadata

Acceptance criteria for Target A:

- packed `fp8_ds_mla` compressor dispatch no longer falls through the local
  compatibility writer when the native packed compressor is available;
- packed `fp8_ds_mla` decode and prefill no longer require the Triton split-KV
  compatibility readers when the native packed attention kernel is available;
- the native kernels load embedded UE8M0 scales from each 584-byte slot and do
  not require sidecar scale tensors;
- the vLLM KV manager remains generic: no DeepSeek-specific worker or core
  scheduler dependency is introduced;
- CUDA/NVIDIA files remain free of ROCm ATOM imports.

This target is the least invasive to vLLM because it preserves the current
`DeepseekV4AtomMLAAttentionSpec` allocation and only replaces the compatibility
kernel readers/writers with native packed ABI consumers.

### Target B: ROCm-only homogeneous native ABI

Change the ROCm-only KV spec/allocation/binding so the ATOM sparse attention
and compressor ABI can consume a homogeneous native tensor directly:

- `atom_unified_kv` must be present for packed deployment rather than deleted
  during `fp8_ds_mla` binding;
- the allocation must expose whatever homogeneous packed layout the native
  ATOM kernels require without reshaping in the hot path;
- the model-state binding must keep CUDA isolated and must refuse silent
  fallback to side allocations when vLLM-owned unified KV was requested.

Acceptance criteria for Target B:

- `sparse_attn_v4_paged_decode` and `sparse_attn_v4_paged_prefill` consume the
  native unified view directly for ROCm DSV4 packed mode;
- the compressor writes into that same vLLM-owned allocation through the native
  ABI;
- existing CUDA/NVIDIA MLA cache specs and bindings are unchanged;
- generic worker code still consumes only generic `KVCacheSpec` fields such as
  `fixed_prefix_size_bytes`, `requires_strided_kv_cache_view`, and
  `inner_block_stride_bytes`.

## ATOM Attention Order

The ATOM model's `DeepseekV4Attention.forward_impl` uses this order:

1. `maybe_compressors_async`
   - Main compressor and indexer compressor can launch on auxiliary streams.
   - vLLM coverage: compressor calls are present, but auxiliary stream overlap
     was reverted and remains disabled in the validated default.
2. `qk_norm_rope_maybe_quant`
   - Fused Q/KV normalization and RoPE.
   - vLLM coverage: present in
     `vllm.models.deepseek_v4.amd.v4_kernels.qk_norm_rope_maybe_quant`.
3. `swa_write before decode`
   - Decode writes current KV into the SWA ring before sparse attention.
   - vLLM coverage: present in the ROCm attention path.
4. `indexer_score_topk`
   - ATOM calls `torch.ops.aiter.indexer_score_topk` after the indexer
     compressor writes K.
   - vLLM coverage: default path decomposes this into vLLM/aiter pieces. The
     opt-in `VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH=1` registers a vLLM-local
     fallback op name without depending on `atom`.
5. `csa_translate_pack`
   - Translates raw CSA top-k rows into paged physical offsets.
   - vLLM coverage: present as a local Triton kernel.
6. `sparse_attn_v4_paged_decode`
   - Decode sparse attention over unified KV.
   - vLLM coverage: present. The deployed packed FP8 path uses
     `sparse_attn_v4_paged_decode_split_kv` because vLLM-owned packed KV is a
     BF16 SWA prefix plus packed `fp8_ds_mla` tail, not ATOM's homogeneous view.
7. `sparse_attn_v4_paged_prefill`
   - Prefill sparse attention over prefix unified KV plus in-flight extend KV.
   - vLLM coverage: present, with split-KV support for the packed layout.
8. `swa_write after prefill`
   - Prefill writes the SWA ring after attention so chunked prefill reads prior
     chunk state.
   - vLLM coverage: present in the ROCm attention path.
9. `inverse_rope_inplace`
   - ATOM applies inverse RoPE before output projection.
   - vLLM coverage: opt-in local primitive exists under
     `VLLM_ROCM_DSV4_ATOM_SEPARATE_INVERSE_ROPE=1`, but the faster validated
     default keeps vLLM's fused `rocm_inv_rope_einsum`.

## Imported ATOM V4 Kernels

| ATOM symbol | vLLM status | Notes |
| --- | --- | --- |
| `CompressPlan` | Covered | vLLM builds ROCm model-state compression plans in `amd/v4_kernels/compress_plan.py`. |
| `qk_norm_rope_maybe_quant` | Covered | Local ROCm kernel, with optional flydsl dispatch where available. |
| `fused_compress_attn` | Covered, partly native | Local path exists. aiter/flydsl can be used for compatible dense/HCA contracts, but packed `fp8_ds_mla` falls back to local compatibility code because public flydsl does not match the packed tail ABI. |
| `update_compressor_states` | Covered | Local state update kernel. Read-before-update order is enforced by tests. |
| `swa_write` | Covered | Local state write kernel for decode and prefill ordering. |
| `csa_translate_pack` | Covered, adapter cost remains | Local Triton kernel. Profiling shows this remains a real CSA per-layer adapter cost. |
| `sparse_attn_v4_paged_decode` | Covered, split layout for deployment | Homogeneous function exists. Packed FP8 deployment uses the split-KV wrapper. |
| `sparse_attn_v4_paged_prefill` | Covered, split layout for deployment | Homogeneous function exists. Packed FP8 deployment uses the split-KV wrapper. |
| `inverse_rope_inplace` | Covered, default-off | Accuracy safe but slower than fused vLLM inverse-RoPE/einsum in C32. |
| `scale_indexer_weights` | Covered | Local `common.ops` primitive used by the ATOM indexer sequence preview. |

## ATOM aiter Surface

| ATOM symbol | vLLM status | Outcome |
| --- | --- | --- |
| `get_hip_quant` | Covered differently | vLLM uses `per_token_group_quant_fp8` and local fused indexer Q/RoPE/quant by default. The ATOM explicit sequence is available as an opt-in parity path but was slower. |
| `rope_rotate_activation` | Covered differently | vLLM default fuses indexer Q/RoPE/quant; explicit ATOM-style sequence is available and accuracy-safe but slower. |
| `cp_gather_indexer_k_quant_cache` | Not used in default packed path | ATOM uses it for prefill indexer gather. vLLM's current indexer path uses vLLM sparse indexer/cache abstractions or the decode fast path. |
| `fp8_mqa_logits` | Partially covered | Lower-level aiter pieces exist, but vLLM does not run the full ATOM prefill `Indexer._score_topk_prefill` sequence by default. |
| `deepgemm_fp8_paged_mqa_logits` | Partially covered | The ATOM decode-style fast path is represented by vLLM's optional decode indexer fast path, but the default path still preserves vLLM indexer abstractions. |
| `top_k_per_row_decode` | Covered through current indexer flow | Used indirectly by the existing vLLM/aiter indexer implementation where applicable. |
| `top_k_per_row_prefill` | Covered through current indexer flow | Used indirectly by the existing vLLM/aiter indexer implementation where applicable. |
| `fused_clamp_act_mul` | Covered | vLLM AMD MLP has optional aiter fused activation; default launch enables it. |
| `mhc_pre` | Available but default-off | Installed `aiter==0.1.15.post1` exposes it; enabling vLLM aiter MHC failed GSM8K accuracy. |
| `mhc_post` | Available but default-off | Same as `mhc_pre`; graph smoke passed, accuracy failed badly. |
| `mhc_fused_post_pre` | Missing from installed aiter | vLLM falls back to tilelang for fused post/pre. Exact ATOM aiter parity is not possible without a new aiter export. |
| `maybe_dual_stream_forward` | Not integrated | ATOM uses it for MoE shared/routed expert overlap. vLLM currently uses vLLM fused MoE with aiter backend, not the ATOM dual-stream wrapper. |

## Current Answer

We do not yet have all necessary components to get the benefit of all ATOM
kernels.

Components present and validated:

- vLLM scheduler, V2 model runner, graph mode, and vLLM weight loading.
- ROCm-only ModelState for persistent per-request SWA/compressor/indexer state.
- ROCm-only packed `fp8_ds_mla` KV spec/allocation/binding from vLLM's KV cache.
- ATOM read-before-update compressor order.
- ATOM-shaped Q/KV normalization, SWA write, compression, index translation,
  paged decode, paged prefill, and inverse-RoPE parity switches.
- No runtime dependency on `atom` or `atom.*`.

Practical-split validation:

- `DeepseekV4ForCausalLM.get_model_state_cls()` returns
  `DeepseekV4RocmAtomModelState` only when ROCm and
  `VLLM_ROCM_DSV4_ATOM_STATE=1` are active; off-ROCm and disabled-ATOM paths
  return `DefaultModelState`.
- `DeepseekV4Attention.get_kv_cache_spec()` returns regular
  `MLAAttentionSpec` off-ROCm, and only emits `DeepseekV4AtomMLAAttentionSpec`
  for ROCm unified-KV mode.
- NVIDIA DeepSeek-V4 model files do not import the ROCm ATOM model-state,
  v4-kernel, or custom KV-spec symbols.
- Generic GPU worker files do not import `vllm.models.deepseek_v4` modules.
  Worker reshaping consumes generic `KVCacheSpec` fields such as
  `fixed_prefix_size_bytes`, `requires_strided_kv_cache_view`, and
  `inner_block_stride_bytes` instead of DeepSeek-specific types.
- `DeepseekV4AtomMLAAttentionSpec` is registered with vLLM's generic
  `FullAttentionManager`, grouped under `FullAttentionSpec`, so the custom spec
  does not require a DeepSeek-specific GPU worker or KV manager.
- `DeepseekV4AtomMLAAttentionSpec` extends the scheduler-facing MLA spec with
  a fixed SWA prefix plus a compressed paged tail. In packed `fp8_ds_mla` mode,
  the spec uses a `uint8` 584-byte compressed token layout and model-state
  binding deliberately exposes split SWA/compressed views rather than a
  homogeneous `atom_unified_kv` tensor.

Focused validation command:

```bash
pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py \
  tests/v1/worker/test_utils.py \
  -k 'deepseek_v4_kv_cache_spec or deepseek_v4_model_state_cls or generic_worker_code_does_not_import_deepseek_v4_model_modules or generic_worker_reshape_uses_kv_cache_spec_contract'
```

Result: `7 passed`.

Components still missing or not production-ready:

- Native packed DSV4 sparse attention ABI that consumes vLLM-owned packed KV
  directly without split-KV adaptation.
- Native packed DSV4 compressor ABI for the deployed `fp8_ds_mla` tail; current
  local writer is correct but not a public aiter/flydsl packed-tail entry point.
- Full ATOM indexer prefill/decode dispatcher sequence over ATOM's exact
  indexer compressed-cache contract. The lower-level aiter pieces are present,
  and vLLM has an opt-in decode fast path, but default vLLM model code still
  falls back through `SparseAttnIndexer` and does not run ATOM's prefill
  `cp_gather_indexer_k_quant_cache -> fp8_mqa_logits -> top_k_per_row_prefill`
  sequence directly.
- aiter `mhc_fused_post_pre` in the installed package.
- Accuracy-safe aiter MHC/HC enablement in vLLM.
- ATOM auxiliary stream overlap for compressors and MoE. ATOM uses
  `torch.ops.aiter.maybe_dual_stream_forward`, `alt_stream`, and
  `compress_stream`; the vLLM AMD path currently disables attention aux streams
  on ROCm and does not call `maybe_dual_stream_forward`.

## Installed aiter ABI Check

The installed `aiter==0.1.15.post1` was inspected directly. It has useful
pieces, but not the exact native ABI needed for the current vLLM packed
deployment layout.

`vllm.models.deepseek_v4.amd.atom_native_abi.probe_atom_native_abi()` now
records this as a runtime-inspectable capability status. With the installed
package it reports:

- `aiter_available=True`
- `packed_fp8_ds_mla_compressor=False`
- `packed_fp8_ds_mla_attention=False`
- `mhc_fused_post_pre=False`
- `maybe_dual_stream_forward=False`

Set `VLLM_ROCM_DSV4_REQUIRE_NATIVE_ATOM_ABI=1` to call
`require_atom_native_abi()` and fail fast during ROCm DSV4 ModelState
construction unless the installed package exposes both native packed main-path
capabilities: `packed_fp8_ds_mla_compressor=True` and
`packed_fp8_ds_mla_attention=True`. This guard is intentionally off by default
so the validated compatibility path can still run.

Relevant installed signatures:

- `aiter.ops.flydsl.kernels.fused_compress_attn.flydsl_fused_compress_attn`
  accepts `kv_cache`, `block_tables`, `k_per_block`, `quant`,
  optional `cache_scale`, `use_ue8m0`, and `preshuffle`.
- `aiter.ops.flydsl.kernels.fused_compress_attn_hca.flydsl_hca_compress_attn`
  accepts BF16 HCA `kv_cache`, `block_tables`, and optional
  `kv_compressed_scratch`.
- `aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_opus` accepts
  `q`, homogeneous `unified_kv`, prefix CSR indices, in-flight `kv`, extend CSR
  indices, `attn_sink`, and `softmax_scale`.
- `aiter.ops.triton.attention.pa_mqa_logits.deepgemm_fp8_paged_mqa_logits`
  and `_ragged_k` consume the indexer-style FP8 KV cache with scale data
  appended/sliced around `hidden_dim`.

What is missing from those ABIs:

- No `packed_fp8_ds_mla` or `compressed_kv_layout` argument.
- No 584-byte token contract.
- No split source contract with BF16 SWA prefix plus packed FP8 compressed
  tail.
- No main-attention reader that loads 448 FP8 no-RoPE bytes, 64 BF16 RoPE
  values, and 8 embedded UE8M0 scale bytes from one packed slot.
- No compressor scatter API that writes vLLM's deployed main packed tail
  directly through the public aiter/flydsl wrapper.

This is why the current vLLM path must use local compatibility kernels for
packed `fp8_ds_mla` main attention/compression, even though flydsl can cover
compatible dense BF16 and some indexer/HCA shapes.

Runtime guard evidence:

- `vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py` excludes
  `packed_fp8_ds_mla` from `_flydsl_use`, so the deployed packed tail cannot be
  routed through the public flydsl compressor wrapper by accident.
- `tests/kernels/test_deepseek_v4_fused_compress_contract.py` monkeypatches
  `flydsl_fused_compress_attn` to raise, calls `fused_compress_attn` with
  `packed_fp8_ds_mla=True`, and asserts the local kernel receives
  `PACKED_FP8_DS_MLA=1`.
- `vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill.py` keeps the OPUS
  homogeneous prefill path in `sparse_attn_v4_paged_prefill`, while the deployed
  packed layout calls `sparse_attn_v4_paged_prefill_split_kv`.
- `tests/kernels/attention/test_deepseek_v4_split_kv_contract.py` monkeypatches
  `pa_sparse_prefill_opus` to raise, calls the packed split-KV prefill wrapper,
  and reaches the split-KV CUDA/HIP guard instead of OPUS dispatch.
- `tests/kernels/test_deepseek_v4_atom_dependency_contract.py` checks the
  deployed ROCm attention branch: when the homogeneous `unified_kv` view is
  absent, decode and prefill route through the split-KV wrappers and pass
  `compressed_kv_layout=split_kv_layout`.
- The same dependency contract checks the packed KV spec/allocation invariant:
  packed `fp8_ds_mla` is represented as fixed SWA prefix plus compressed tail,
  and model-state binding removes any stale homogeneous `atom_unified_kv` view.
- The same dependency contract also checks that `DeepseekV4AtomMLAAttentionSpec`
  is handled by `FullAttentionManager`, preserving the generic scheduler/KV
  manager path.
- It also checks NVIDIA DeepSeek-V4 files do not import ROCm ATOM model-state,
  v4-kernel, or custom KV-spec symbols.
- It also checks the current vLLM DeepSeek-V4 model sources do not integrate
  ATOM's `maybe_dual_stream_forward` MoE overlap, and that the AMD model keeps
  ROCm attention auxiliary streams disabled.
- It also checks the ATOM full indexer prefill/decode dispatcher is only
  partially integrated: vLLM exposes the low-level ROCm/aiter pieces and an
  opt-in decode fast path, while default model code still falls back through
  `SparseAttnIndexer` and lacks the ATOM prefill gather/logits/top-k sequence.

Focused validation command:

```bash
pytest -q tests/kernels/test_deepseek_v4_fused_compress_contract.py \
  tests/kernels/test_deepseek_v4_atom_op_surface_audit.py \
  tests/kernels/test_deepseek_v4_atom_dependency_contract.py \
  tests/kernels/attention/test_deepseek_v4_split_kv_contract.py
```

Result: `53 passed`.

The next aligned implementation target is therefore not another env toggle. It
is either:

- add/find a native aiter/flydsl sparse attention and compressor entry point for
  vLLM's packed `fp8_ds_mla` layout, or
- change the ROCm-only vLLM KV spec/allocation/binding so the ATOM homogeneous
  sparse attention/compressor ABI can consume the cache directly while CUDA
  keeps the existing KV-cache path.
