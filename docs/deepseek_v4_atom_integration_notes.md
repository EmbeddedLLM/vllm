# DeepSeek V4 ATOM Integration Notes

Date: 2026-06-17

This note records what was tested while integrating ATOM-style DeepSeek V4
optimizations into vLLM's AMD path. It is written as a handoff for another LLM
or engineer: keep the deployment path, avoid known-bad branches, and focus
future work on the unresolved gaps.

For the static op-surface comparison against
`/app/atomdsv4/ATOM/atom/models/deepseek_v4.py`, see
`docs/deepseek_v4_atom_op_surface_audit.md`.

## Goal

Run DeepSeek-V4-Pro FP8/FP4 on vLLM with vLLM's scheduler, attention metadata,
KV-cache abstraction, and fused MoE abstraction, while bringing in safe ATOM
ops where they improve performance or preserve accuracy.

Validation target:

- Accuracy command: `launchdeepseekgraph.sh` plus unchanged `lmeval.sh`.
- GSM8K target: `0.95 +/- 0.01`; practical lower bound is `0.94`.
- Benchmark command: fresh server restart, then `benchmarkvllm.sh` at C32.
- Serving should run with graph capture, not `--enforce-eager`.
- Candidate V2 runs must keep async scheduling enabled
  (`ASYNC_SCHEDULING=1` / `--async-scheduling`). Runs with
  `--no-async-scheduling` are diagnostic only and should not be counted as
  deployment performance evidence.

## Current Deployment Default

As of 2026-06-21, `launchdeepseekgraph.sh` defaults to the validated async V2
ROCm ATOM path:

- `ENFORCE_EAGER=0`
- `ASYNC_SCHEDULING=1`
- `BLOCK_SIZE=128`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_ROCM_DSV4_ATOM_STATE=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
- `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`
- `VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN=1`
- `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
- `VLLM_ROCM_DSV4_ATOM_USE_AITER_PA_DECODE=0`
- `ATOM_USE_FUSED_Q_NORM_QUANT=1`
- `--kv-cache-dtype fp8`

The `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0` default is intentional for the current
vLLM integration. The mixed-KV path is accurate, but the latest fresh-server
C32 runs were slower than the dense no-mixed path. Mixed KV remains a useful
diagnostic for the ATOM recipe's FP8 KV-cache target, but it is not the current
deployment default.

Validated packed mixed-KV accuracy:

- Launch:
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1 bash launchdeepseekgraph.sh`
- `lmeval.sh` GSM8K flexible: `0.9545 +/- 0.0057`
- `lmeval.sh` GSM8K strict: `0.9553 +/- 0.0057`

Latest packed mixed-KV accuracy after the vLLM-owned KV reshape fixes:

- Result file:
  `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-20T04-10-37.051823.json`
- Launch used graph mode, V2 model runner, block size `128`, `--kv-cache-dtype fp8`,
  and `MAX_NUM_SEQS=256`.
- `lmeval.sh` GSM8K flexible: `0.9575435936315391 +/- 0.005553837749990046`
- `lmeval.sh` GSM8K strict: `0.9583017437452616 +/- 0.0055062050581757725`

Packed mixed-KV C32 benchmark:

- Result file:
  `bench-sparsemla/ds-v4-pro-nomtp-atom-mixedkv-noeager-maxseq256-C32-C32.json`
- Completed: `320`
- Failed: `0`
- Output throughput: `808.4598037391398 tok/s`
- Total throughput: `1620.0776535866357 tok/s`
- Mean TPOT: `38.55307477839084 ms`

Latest fresh-server C32 benchmark after the same reshape fixes:

- Result file:
  `bench-sparsemla/ds-v4-pro-atom-mixedkv-runtime-reshape-noeager-maxseq32-C32-C32.json`
- Launch used graph mode, V2 model runner, block size `128`, `--kv-cache-dtype fp8`,
  and `MAX_NUM_SEQS=32`.
- Completed: `320`
- Failed: `0`
- Output throughput: `807.7731461708663 tok/s`
- Total throughput: `1618.7016561939627 tok/s`
- Mean TPOT: `38.578050070810285 ms`
- Mean TTFT: `1091.112763369165 ms`

This is correct for the FP8 KV target, but slower than the best BF16-tail
compatibility run, which is why the remaining gap is a native packed DSV4
sparse-attention/compressor contract rather than another launch switch.

## Current Component Audit

As of 2026-06-20, the ROCm path has enough pieces to run an ATOM-ordered
attention/compressor sequence inside vLLM's scheduler, but not enough to claim
the full benefit of all ATOM kernels.

Present and validated:

- vLLM scheduler, V2 model runner, graph-mode serving, and vLLM weight loading.
- vLLM-owned DSV4 ATOM KV spec/binding for ROCm only, with CUDA untouched.
- Persistent per-request state through `DeepseekV4RocmAtomModelState`, not
  `WorkspaceManager`.
- ATOM-style compressor order: `fused_compress_attn` reads previous state, then
  `update_compressor_states` writes current state.
- ATOM-style Q/KV path: `qk_norm_rope_maybe_quant` feeds the ATOM SWA write and
  sparse MLA path.
- ATOM SWA writes, decode/prefill index writers, CSA top-k translation, and
  paged decode/prefill wrappers.
- Packed FP8 mixed-KV layout (`fp8_ds_mla`) is allocated and bound from vLLM KV
  storage; accuracy passes with graph mode.
- Core KV-cache spec accounting covers the packed DSV4 tail as
  `storage_block_size * 584` bytes plus the fixed SWA prefix. The focused
  registry/worker tests validate this without touching CUDA paths.

Partial or still performance-limited:

- Packed split-KV sparse attention works, but it is still mediated by vLLM
  metadata conversion and split-wrapper dispatch. This is the main remaining
  gap between "same logical op sequence" and "same ATOM kernel benefit".
- Prefill is wired through ATOM paged-prefill wrappers, but mixed prefill/decode
  behavior has required explicit gating in this branch. Keep
  `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1` with the current deployment
  defaults.
- CSA decode can either pre-translate top-k indices or use the fused decode
  path when available. The default passing path still pays visible metadata and
  wave scheduling overhead.
- MHC/HC is adjacent to full ATOM model equivalence, but it is not a dependency
  for the KV layout, compressor state order, SWA ring, or sparse attention
  reader. Treat it as a later end-to-end performance feature.

Missing for a true "all ATOM kernel benefits" claim:

- A native packed DSV4 sparse attention ABI that consumes the same vLLM-owned
  packed KV layout without extra split-layout adaptation.
- A tighter metadata contract so decode/prefill kernels can consume scheduler
  state directly, instead of repeated conversion into ATOM-specific ragged
  buffers.
- Stream overlap parity with ATOM's auxiliary streams. The current graph-mode
  path is correctness-safe without it, but it does not reproduce ATOM's overlap
  schedule.
- A proven large-batch path for optional indexer-inner ATOM compressor without
  ROCm out-of-resources or JIT pressure.

## Historical Fast Passing Configuration

The passing default launch uses graph mode:

- `ENFORCE_EAGER=0`
- `MAX_NUM_SEQS=256` for lmeval
- `MAX_NUM_SEQS=32` for C32 benchmark
- `ATOM_USE_ATOM_COMPRESSOR_ORDER=1`
- `ATOM_USE_FUSED_Q_NORM_QUANT=1`
- `ATOM_ENABLE_AITER_HC_HEAD=1`
- `ATOM_ENABLE_AUX_STREAMS=0`
- `ATOM_ENABLE_AITER_MHC_PRE=0`
- no ATOM-style sparse attention decode branch

Accuracy result:

- Result file:
  `results_deepseekprographmtp/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-16T20-34-37.693718.json`
- GSM8K flexible: `0.9446550416982562`
- GSM8K strict: `0.9454131918119788`

C32 benchmark result:

- Result file:
  `bench-sparsemla/ds-v4-pro-nomtp-current-default-noeager-C32.json`
- Completed: `320`
- Failed: `0`
- Output throughput: `916.2795284617761 tok/s`
- Total throughput: `1836.1382738316058 tok/s`
- Mean TPOT: `33.8685634798457 ms`

The C32 target from ATOM's DeepSeek-V4 recipe is around `1145.71 output tok/s`,
so this vLLM integration is still roughly 20% behind that reference point.

## Enabled Pieces That Helped Or Were Required

### vLLM Scheduler And Graph Mode

The final path keeps vLLM's scheduler and graph-captured serving path. Graph
capture is required for the final benchmark numbers. The passing server log
showed `enforce_eager=False` and successful graph capture.

### vLLM Sparse Decode Attention

The fastest passing run uses the existing vLLM ROCm sparse decode path:

- `_rocm_sparse_attn_decode_ragged_triton`
- split-K decode on gfx950

It does not use the ATOM-style sparse decode branch.

### AITer HC Head

The final head reduction uses the safe AITer HC/head path. ATOM's head path
calls `aiter.mhc_pre(..., sinkhorn_repeat=0)` for the final reduction. In this
integration the head-specific AITer path was accuracy-safe and gave a small
performance improvement.

### Fused MoE / MXFP4 Path

The vLLM fused MoE abstraction remains active, with AMD/aiter/oracle MXFP4
integration. This is necessary for useful DeepSeek-V4 performance, though the
MoE-specific delta was not isolated in this round.

### Fused Q Norm / Quant

`ATOM_USE_FUSED_Q_NORM_QUANT=1` enables the fused RMSNorm plus group quant path
for query/projection activation handling in the AMD attention path.

This is separate from compressor ordering. It acts on `qr` before attention
projection consumption; it does not define when compressor state is saved.

### ATOM-Style Compressor Ordering

`ATOM_USE_ATOM_COMPRESSOR_ORDER=1` is required for accuracy in the AMD path.

The two orderings are:

vLLM/CUDA-style write-before-compress:

```text
current kv/score -> save_partial_states -> state_cache
state_cache -> compress/norm/rope/quant -> compressed KV cache
```

ATOM-style read-before-update:

```text
old tokens read from state_cache
current in-flight tokens read from kv/score/ape directly
compress/norm/rope/quant -> compressed KV cache
then save_partial_states -> state_cache
```

Disabling ATOM order improved C32 throughput by about 1%, but failed accuracy:

- `ATOM_USE_ATOM_COMPRESSOR_ORDER=0`
- C32 output throughput: `925.1308471453252 tok/s`
- C32 total throughput: `1853.8754866623117 tok/s`
- C32 mean TPOT: `33.502449237068475 ms`
- GSM8K flexible: `0.9385898407884761`
- GSM8K strict: `0.9393479909021987`

Conclusion: keep ATOM-style compressor ordering for AMD unless another change
recovers accuracy.

## Removed Branch: ATOM-Style Sparse Decode Attention

The ATOM-style sparse decode branch was removed because it was not used in the
passing configuration and was slower in focused microbenchmarks.

Removed items:

- `ATOM_USE_ATOM_STYLE_DECODE_ATTN` launch/env switch
- `_sparse_attn_decode_atom_partial_kernel`
- `_sparse_attn_decode_atom_reduce_kernel`
- `_rocm_sparse_attn_decode_ragged_atom_triton`
- `_decode_atom_num_splits`

Microbenchmark setup:

- C32 decode
- TP8 local heads: `16`
- `head_dim=512`, `nope=448`, `rope=64`
- SWA window: `128`
- `index_topk=1024`
- ISL `1024` maps to SWA/main length `128` plus compressed C128A/extra length
  `256`

Results:

| Shape | vLLM sparse decode mean | ATOM-style mean | Delta |
| --- | ---: | ---: | ---: |
| C32, seq 1024, SWA128 + topk256 | `0.0566 ms` | `0.0732 ms` | `+29.3% slower` |
| C32, seq 4096, SWA128 + topk1024 | `0.0705 ms` | `0.0956 ms` | `+35.6% slower` |
| C32, seq 8192, SWA128 + topk1024 | `0.0702 ms` | `0.0958 ms` | `+36.4% slower` |
| C64, seq 1024 | `0.0573 ms` | `0.0758 ms` | `+32.3% slower` |
| C128, seq 1024 | `0.0729 ms` | `0.1004 ms` | `+37.8% slower` |

Correctness sanity for the microbench was bf16-scale acceptable:

- max diff about `0.001953125` to `0.00390625`

Conclusion: use vLLM's split-K sparse decode kernel on gfx950.

## Tested Variants And Outcomes

### Indexer-Inner ATOM Compressor

Result: accuracy-safe at reduced server concurrency, but not a performance win.

This variant enabled:

- `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`
- `ATOM_FUSED_COMPRESS_USE_FLYDSL=auto`

The first full-lmeval launch at `MAX_NUM_SEQS=256` failed before producing
metrics with ROCm out-of-resources errors after runtime JIT. Retrying at
`MAX_NUM_SEQS=64`, `MAX_NUM_BATCHED_TOKENS=8192`,
`MAX_MODEL_LEN=8192`, and `GPU_MEMORY_UTILIZATION=0.85` completed the
unchanged `lmeval.sh`:

- GSM8K flexible: `0.9492 +/- 0.006`
- GSM8K strict: `0.9500 +/- 0.006`

Fresh C32 benchmark with `MAX_NUM_SEQS=32`:

- Result file:
  `bench-indexer-compressor-c32/indexer-compressor-c32-C32.json`
- Completed: `320`
- Failed: `0`
- Output throughput: `861.9034074289726 tok/s`
- Total throughput: `1727.1736250432148 tok/s`
- Mean TPOT: `36.32073513339028 ms`
- Mean TTFT: `849.5995086035691 ms`

This is slower than the previous best default C32 result
(`916.2795284617761 tok/s`, `33.8685634798457 ms` TPOT). The run progressed in
visible 32-request waves with large stalls between waves. That behavior points
to scheduler/metadata/conversion/workspace overhead around the kernels, not
just the inner compressor or sparse attention kernel cost. Keep this path
experimental until those adapter costs are profiled and reduced.

### vLLM-Owned Unified KV Metadata Bundle

Result: integration plumbing improvement; not benchmarked independently.

When `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`, compressed main-MLA layers
emit `DeepseekV4AtomMLAAttentionSpec`. The DSV4 KV allocator reserves a fixed
SWA prefix in the same raw allocation as the compressed tail, and
`post_bind_kv_cache`/`DeepseekV4RocmAtomModelState` bind:

- `attn.atom_unified_kv`
- `attn.atom_swa_kv`
- `attn.atom_compressed_kv_cache`
- `attn.compressor.atom_kv_cache`

The vLLM-owned binding path now also publishes a
`DeepseekV4RocmAtomUnifiedKVBuffers` bundle into
`DeepseekV4RocmAtomStateMetadata.unified_kv_buffers`, matching the side
allocation path's metadata contract. This matters for future ATOM kernels that
should consume the model-state metadata directly instead of rediscovering
per-layer attributes.

### Removed Experiment: Direct CSA Decode Kernel

Result: accuracy-safe historically, but slower than the default deployed path.
The runtime hooks and helper kernels were removed after validation because the
path was not the ATOM modeling-file sequence and did not improve served C32.

`VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1` was meant to bypass the separate
`csa_translate_pack` launch during pure decode and call
`sparse_attn_v4_csa_topk_paged_decode` directly with raw indexer top-k,
request-state mapping, block tables, and the unified KV pool.

The branch previously returned before the direct kernel call, so it was not
actually testable. That early return was removed while preserving the
`VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_DECODE` debug gate.

Smoke validation:

- Server:
  `MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024 MAX_MODEL_LEN=2048`
  `GPU_MEMORY_UTILIZATION=0.85`
  `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1 bash launchdeepseekgraph.sh`
- Graph capture completed without `--enforce-eager`.
- A `/v1/completions` request with `max_tokens=32` returned HTTP 200.

Full accuracy validation:

- Server:
  `MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192`
  `GPU_MEMORY_UTILIZATION=0.9`
  `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1 bash launchdeepseekgraph.sh`
- Unchanged `lmeval.sh`
- GSM8K flexible: `0.9560 +/- 0.0056`
- GSM8K strict: `0.9568 +/- 0.0056`

Fresh C32 benchmark:

- Server:
  `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192`
  `GPU_MEMORY_UTILIZATION=0.9`
  `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1 bash launchdeepseekgraph.sh`
- Result file:
  `bench-csa-direct-c32/csa-direct-c32-C32.json`
- Completed: `320`
- Failed: `0`
- Output throughput: `843.1002223314714 tok/s`
- Total throughput: `1689.493804906425 tok/s`
- Mean TPOT: `36.972113092089565 ms`
- Mean TTFT: `1036.0968612425495 ms`

This is slower than the current default C32 result
(`916.2795284617761 tok/s`, `33.8685634798457 ms` TPOT) and slower than the
indexer-inner compressor C32 result (`861.9034074289726 tok/s`,
`36.32073513339028 ms` TPOT).

Important runtime signal: with the direct CSA accuracy server at
`MAX_NUM_SEQS=256`, vLLM reported only `11,978` GPU KV-cache tokens and max
concurrency `1.46x` for 8192-token requests after graph/profile allocation.
The C32 server reported `45,496` GPU KV-cache tokens and max concurrency
`5.55x`. The benchmark progressed in visible 32-request waves with long
first/wave latency. That means this deployed path is likely constrained by
KV/cache allocation pressure and adapter/layout work around the kernel, so the
end-to-end result should not be interpreted as pure CSA kernel speed.

### Metadata And Conversion Overhead

Result: plausible end-to-end bottleneck, but the first pure-decode metadata
cleanup did not materially improve C32 throughput.

Audit result:

- ATOM feature flags in the ROCm DSV4 path are already cached as module-level
  constants. There are no repeated `os.environ.get(...)` calls in the ATOM
  per-forward hot path.
- `test_deepseek_v4_atom_env_lookups_are_import_time_cached` now guards that
  invariant for the hot DSV4 attention/compressor/model-state/kernel wrapper
  files, allowing env helper use only at import time.
- The deployed path still does per-forward metadata/layout work: CPU
  `np.ascontiguousarray(...)` plan inputs, GPU copies of request metadata,
  decode indptr construction, decode index writer/translator kernels, HCA table
  flattening for the BF16 cache path, and occasional dtype/layout conversions
  before fused compression.
- The direct CSA C32 run showed scheduling/cache-capacity symptoms, so the
  current gap should be investigated as deployed-path overhead, not just kernel
  runtime.

Cleanup added after the direct CSA benchmark:

- `DeepseekV4RocmAtomModelState._build_atom_state_metadata` now special-cases
  pure one-token decode.
- It reuses a precomputed request arange buffer instead of allocating a fresh
  arange for `np.repeat`.
- It fills decode positions by slice assignment from `computed` instead of a
  Python loop.
- It skips per-forward numpy `.copy()` snapshots for CPU metadata during pure
  decode, because decode only needs those CPU arrays while constructing indptrs
  inside metadata preparation. Prefill/mixed paths still keep snapshots.

Validation:

- `python3 -m py_compile vllm/models/deepseek_v4/amd/model_state.py
  vllm/models/deepseek_v4/amd/rocm.py
  vllm/models/deepseek_v4/compressor.py`
- Default graph-mode accuracy after cleanup:
    - Server:
    `MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192`
    `GPU_MEMORY_UTILIZATION=0.9 bash launchdeepseekgraph.sh`
    - Unchanged `lmeval.sh`
    - GSM8K flexible: `0.9507 +/- 0.0060`
    - GSM8K strict: `0.9515 +/- 0.0059`
- Fresh default C32 benchmark after cleanup:
    - Server:
    `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192`
    `GPU_MEMORY_UTILIZATION=0.9 bash launchdeepseekgraph.sh`
    - Result file:
    `bench-metadata-cleanup-c32/metadata-cleanup-c32-C32.json`
    - Completed: `320`
    - Failed: `0`
    - Output throughput: `867.3439657962101 tok/s`
    - Total throughput: `1738.0759939588118 tok/s`
    - Mean TPOT: `35.89698607919532 ms`
    - Mean TTFT: `1050.674704555422 ms`

This is essentially tied with the recent default-gated C32 run
(`870.3875757954914 tok/s`, `35.80374276777991 ms` TPOT), slower than the
older saved default best (`916.2795284617761 tok/s`,
`33.8685634798457 ms` TPOT), and faster than the direct CSA run
(`843.1002223314714 tok/s`, `36.972113092089565 ms` TPOT). Conclusion:
small Python-side metadata allocation cleanup is not enough to explain or close
the end-to-end gap. The next bottleneck is more likely the deployed layout,
KV-capacity/graph memory pressure, GPU-side index translation/packing, or
missing ATOM stream/workspace structure.

Suggested next measurement:

- Run the default C32 path with `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1` and a
  large profile interval after warmup to quantify `super`, `plans`, `state`,
  `decode_indptr`, and `annotate` time.
- Repeat the same command before/after the pure-decode cleanup, then only run
  full lmeval/C32 if the metadata profile shows a measurable reduction.

### AITer Block MHC Pre

Result: rejected.

The installed `aiter==0.1.15.post1` exposes top-level `aiter.mhc_pre`, but the
block MHC pre path produced bad model accuracy in this environment.

Observed result:

- GSM8K flexible around `0.439`
- GSM8K strict around `0.437`

Investigation showed the direct GEMM portion matched, but divergence began in
the fused post-processing path (`mhc_pre_big_fuse` /
`mhc_pre_big_fuse_rmsnorm`). Keep block MHC pre disabled by default:

- `ATOM_ENABLE_AITER_MHC_PRE=0`
- `ATOM_DISABLE_AITER_MHC_PRE=1`

### AITer Triton Fused HC Post+Pre

Result: rejected as default, left only as an experimental path if still present.

The ATOM model calls top-level `aiter.mhc_fused_post_pre` when available and
skips it on layer 0. In this installed aiter source tree that exact top-level
symbol is not exported. The closest available implementation is:

- `aiter.ops.triton.fusions.mhc.mhc_post_pre`

The wrapper could run and matched small local references, but full lmeval was
slightly below target:

- GSM8K flexible: about `0.9393`
- GSM8K strict: about `0.9401`

Because the lower bound is `0.94`, the flexible result fails the acceptance
criterion.

### Tilelang Fused HC

Result: rejected for performance.

Earlier C32 output throughput was about `879 tok/s`, slower than the final
passing path.

### Auxiliary Streams

Result: rejected for performance.

Earlier C32 output throughput was about `601 tok/s`, much slower than default.

Keep:

- `ATOM_ENABLE_AUX_STREAMS=0`

### Forced Decode Splits

Result: rejected for the tested setting.

`ATOM_DECODE_NUM_SPLITS=16` produced about `890 output tok/s`, slower than the
default split heuristic. Keep the heuristic unless a new sweep shows otherwise.

### Decode Fused-Single Path

Result: rejected for performance.

Earlier C32 output throughput was about `914 tok/s`, slightly worse than the
final passing path.

## CUDA Compressor Ordering Observation

vLLM's CUDA/NVIDIA modeling path currently uses write-before-compress.

The shared `DeepseekCompressor` gates ATOM read-before-update to ROCm:

```python
self._use_atom_compressor_order = (
    current_platform.is_rocm()
    and os.environ.get("ATOM_USE_ATOM_COMPRESSOR_ORDER", "0") == "1"
)
```

For CUDA, the order is:

```text
save_partial_states(...)
compress_norm_rope_store_cutedsl(...)
```

The NVIDIA CuTe DSL compressor receives only `state_cache`; it does not receive
current `kv`, `score`, `ape`, or `query_start_loc`, and it has no
`read_before_update` mode. So CUDA was implemented and validated around
write-before-compress.

The AMD Triton compressor has an explicit `READ_BEFORE_UPDATE` mode, and the
AMD integration needed it to pass GSM8K. This is likely a state visibility and
chunk scheduling issue, not a q-norm/quant mathematical requirement.

## Analysis Ideas For Future Work

1. Reproduce ATOM's exact top-level `aiter.mhc_fused_post_pre` environment.
   The installed `0.1.15.post1` source here does not export the same top-level
   fused symbol that ATOM conditionally calls.

2. Isolate fused MoE performance. The final perf still trails ATOM's C32 table;
   MoE and scheduling overlap are likely larger contributors than sparse
   decode attention.

3. Investigate compressor ordering with a deterministic unit harness. Build a
   small test that compares read-before-update and write-before-compress for
   mixed prefill/decode batches, slot reuse, padding, and graph replay.

4. Profile full serving rather than only microbenching kernels. The sparse
   decode kernel itself is fast; the remaining gap may come from compressor,
   MoE, MHC, metadata building, or graph-captured host/device synchronization.

5. Revisit auxiliary streams only after the baseline op sequence is stable.
   The first attempt was significantly slower, likely due to synchronization or
   stream dependency placement.

6. Keep vLLM weight loading. The integration should not depend on ATOM as a
   Python package, and vLLM's loader supports faster/alternative strategies such
   as safetensors streaming paths.

## Operational Notes

Use these defaults for the currently passing AMD path:

```bash
MAX_NUM_SEQS=256 bash ./launchdeepseekgraph.sh
bash ./lmeval.sh

# Fresh server restart for benchmark:
MAX_NUM_SEQS=32 bash ./launchdeepseekgraph.sh
RESULT_PREFIX=ds-v4-pro-nomtp-current-default-noeager CONCURRENCIES=32 \
  bash ./benchmarkvllm.sh
```

Do not change `lmeval.sh` when validating accuracy.

## 2026-06-17 Latest Tilelang-MHC Baseline And Fused Activation

### Latest Tilelang-MHC / No-Breakable Baseline

Result: accuracy passed; performance was not the overall best saved C32 run.

Current launch state at measurement time:

- `VLLM_USE_BREAKABLE_CUDAGRAPH=0`
- no `--enforce-eager`
- `MAX_NUM_SEQS=256` for `lmeval.sh`
- fresh restart with `MAX_NUM_SEQS=32` for `benchmarkvllm.sh`

Accuracy:

- GSM8K flexible: `0.9537528431`
- GSM8K strict: `0.9545109932`

C32 random 1024/1024 benchmark:

- output throughput: `921.10 tok/s`
- total throughput: `1845.80 tok/s`
- mean TPOT: `33.76 ms`
- result file: `bench-sparsemla/latest-aitermhc-nobreakable-C32.json`

This is slower than the best saved run,
`bench-sparsemla/revert-compressor-aux-nomtp-C32.json`, which measured
`926.06 output tok/s`, `1855.74 total tok/s`, and `33.50 ms` mean TPOT.

### `aiter.ops.triton.fusions.fused_clamp_act_mul`

Result: kept as a gated experimental improvement over the latest baseline.

Integration:

- `DeepseekV4MLP` can call
  `aiter.ops.triton.fusions.fused_clamp_act_mul.fused_clamp_act_mul`.
- It is gated by `ATOM_USE_AITER_FUSED_CLAMP_ACT_MUL=1`.
- The launch script currently enables that gate by default for testing.
- Fallback remains the previous top-level `aiter.silu_and_mul` path.

Small BF16 tensor checks against `aiter.silu_and_mul` showed exact matches for
some shapes and maximum absolute differences of about `0.001-0.002` for other
shapes, so full task accuracy was required.

Accuracy:

- GSM8K flexible: `0.9537528431`
- GSM8K strict: `0.9545109932`

C32 random 1024/1024 benchmark:

- output throughput: `922.73 tok/s`
- total throughput: `1849.07 tok/s`
- mean TPOT: `33.71 ms`
- result file: `bench-sparsemla/fused-clamp-actmul-C32.json`

Compared with the latest tilelang-MHC baseline, this is a small improvement:
`+1.63 output tok/s` and `-0.05 ms` mean TPOT. It is still below the overall
best saved C32 result, so treat this as a tentative local improvement rather
than a solved performance gap.

### `aiter.cp_gather_indexer_k_quant_cache`

Result: rejected for performance and removed.

Experiment:

- Added a gfx950 opt-in path for top-level
  `aiter.cp_gather_indexer_k_quant_cache`.
- The current Triton cache writer uses the preshuffled layout, so the aiter
  gather call needed `preshuffle=True`; without that, a smoke test showed
  matching scales but incorrect gathered K values.

Accuracy with fused activation plus aiter cp-gather:

- GSM8K flexible: `0.9514783927`
- GSM8K strict: `0.9522365428`

C32 random 1024/1024 benchmark:

- output throughput: `922.16 tok/s`
- total throughput: `1847.92 tok/s`
- mean TPOT: `33.67 ms`
- result file: `bench-sparsemla/aiter-cpgather-C32.json`

This passed accuracy but was slower than the fused-activation-only result
(`922.73 output tok/s`, `1849.07 total tok/s`). The integration was removed
because the rule is to keep only changes that improve performance while
preserving accuracy.

### `aiter.rope_rotate_activation` + `aiter.get_hip_quant`

Result: rejected for performance and removed.

Experiment:

- Added an opt-in `DeepseekV4Indexer` path for the FP8 indexer-cache case.
- The path followed the ATOM indexer sequence:
  `wq_b -> rope_rotate_activation -> get_hip_quant(QuantType.per_1x128) ->
  weights * q_scale * softmax/head scale`.
- vLLM's existing path uses `fused_indexer_q_rope_quant`, which fuses RoPE,
  quantization, and weight-scale folding in one Triton kernel.
- `rope_rotate_activation` requires split `cos` and `sin` tensors with the
  same dtype as Q, while vLLM stores a combined `cos_sin_cache`; the experiment
  used a cached split/cast to avoid per-forward conversion.

Smoke result:

- Dtypes matched the current FP8 path: `torch.float8_e4m3fn` Q and `float32`
  folded weights.
- Folded weights were numerically close to the vLLM path, but Q values differed
  because `rope_rotate_activation` includes ATOM's activation rotation rather
  than being a bitwise replacement for vLLM's fused RoPE-only indexer kernel.

Accuracy:

- GSM8K flexible: `0.9514783927`
- GSM8K strict: `0.9522365428`
- result file:
  `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-17T19-12-07.602309.json`

C32 random 1024/1024 benchmark:

- output throughput: `888.72 tok/s`
- total throughput: `1780.92 tok/s`
- mean TPOT: `34.97 ms`
- result file: `bench-sparsemla/aiter-indexer-ropequant-C32.json`

This passed accuracy but was much slower than the kept fused-activation-only
result (`922.73 output tok/s`, `1849.07 total tok/s`, `33.71 ms` mean TPOT).
The likely reason is that the ATOM-style sequence adds separate kernels and
loses vLLM's fused indexer Q RoPE/quant/weight-fold kernel. The integration was
removed.

### `ATOM/atom/model_ops/v4_kernels/fused_compress.py`

Status: not integrated in the current kept code.

Inspection summary:

- ATOM's `fused_compress_attn` is not a simple replacement for vLLM's
  `compress_norm_rope_store_triton`.
- ATOM depends on `CompressPlan` metadata:
  `[ragged_id, batch_id, position, window_len]` rows, with fixed-capacity
  plan buffers for graph replay.
- ATOM also expects separate per-sequence compressor state slots:
  `kv_state`, `score_state`, and `state_slot_mapping`.
- vLLM's current compressor path is wired to vLLM scheduler/cache metadata:
  `slot_mapping`, `block_table`, a paged combined state cache, and the vLLM
  KV-cache abstraction.
- The ordering also differs. ATOM's fused compressor must run before the state
  update because it reads previous-forward compressor state. The current
  reverted vLLM path writes partial states first and then compresses through the
  paged state cache.

Implication:

- A direct copy would require adding an ATOM-style compression-plan builder and
  per-sequence ring-state abstraction inside vLLM, then adapting cache scatter
  to vLLM's KV-cache/indexer cache layout.
- Importing `atom.model_ops.v4_kernels.fused_compress` directly would violate
  the no-ATOM-package-dependency requirement.
- Reintroducing the previous compressor-order experiment is not appropriate as
  a shortcut; that code was explicitly reverted, and the current passing
  accuracy baseline uses the vLLM ordering.

Follow-up experiment:

- Added a vLLM-native FP8 indexer-only path behind
  `ATOM_USE_INDEXER_PLAN_COMPRESS=1`.
- The path kept vLLM's scheduler, paged state cache, and KV-cache metadata, but
  changed the order to read previous rows from the paged state cache and current
  rows directly from the current forward input before `save_partial_states`.
- This approximated ATOM's read-before-update compressor ordering without
  importing ATOM or adding ATOM's ring-state cache/`CompressPlan` abstraction.

Accuracy:

- GSM8K flexible: `0.9537528431`
- GSM8K strict: `0.9545109932`
- result file:
  `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-17T19-38-17.056777.json`

C32 random 1024/1024 benchmark:

- output throughput: `918.41 tok/s`
- total throughput: `1840.41 tok/s`
- mean TPOT: `33.83 ms`
- result file: `bench-sparsemla/indexer-plan-compress-C32.json`

Decision: rejected and removed. It preserved accuracy, but it was slower than
the kept fused-activation run (`922.73 output tok/s`, `1849.07 total tok/s`,
`33.71 ms` mean TPOT) and slower than the latest tilelang-MHC baseline
(`921.10 output tok/s`). The likely reason is that avoiding one state-cache
read in the compress kernel did not offset the extra direct current-input
loads and the additional metadata work in vLLM's paged layout.

Next analysis ideas:

- Build a vLLM-native `CompressPlan` from `CommonAttentionMetadata` using
  `query_start_loc`, per-request lengths, and `block_table`.
- Decide whether the ATOM ring-state layout can coexist with vLLM's paged state
  cache without duplicating memory.
- If a ring-state cache is added, test only the CSA/indexer compressor first
  because it has the most direct FP8 cache scatter analogue.
- Benchmark the compressor kernel in isolation before a full lmeval cycle;
  full benchmark should still be gated by GSM8K accuracy.

## 2026-06-19 Current ROCm ATOM Integration State

The active branch now contains a staged ROCm-only ATOM integration path while
keeping vLLM's scheduler:

- vLLM-owned ATOM unified KV spec/binding for DSV4 compressed MLA layers.
- Model-specific `DeepseekV4RocmAtomModelState` for persistent request state,
  SWA/compressor rings, decode/prefill buffers, and compression plans.
- ATOM-style paged decode/prefill kernels and CSA translate/pack kernels
  under `vllm/models/deepseek_v4/amd/v4_kernels`.
- Main CSA/HCA fused compressor path gated by
  `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`.
- Experimental indexer-inner fused compressor path gated by
  `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=1`.
- Pure-decode indexer fastpath gated by
  `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`.
- ATOM top-k reuse/skip logic, matching
  `ATOM/atom/models/deepseek_v4.py::_should_skip_v4_index_topk`, gated by
  `VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1` or config attributes
  `use_index_cache/index_topk_freq/index_topk_pattern`.

### ATOM Component Coverage Audit

Conclusion: we do not yet have every necessary component wired in a way that
can be expected to recover all ATOM kernel benefits. The branch has the
important structural hooks, but several high-value pieces are still either
gated, slower in this vLLM adaptation, or not yet the same execution sequence
as `ATOM/atom/models/deepseek_v4.py`.

Current state by component:

| Component | Current state | Evidence / caveat | Next action |
| --- | --- | --- | --- |
| vLLM scheduler / V2 runner | Active | `launchdeepseekgraph.sh` uses `VLLM_USE_V2_MODEL_RUNNER=1`; lmeval and C32 ran without `--enforce-eager`. | Keep. Do not move request lifecycle into model code. |
| ROCm-only model state | Active | `DeepseekV4ForCausalLM.get_model_state_cls()` returns `DeepseekV4RocmAtomModelState` when `VLLM_ROCM_DSV4_ATOM_STATE=1`. | Keep request-lived SWA/compressor rings here. |
| vLLM-owned unified KV binding | Active, BF16-only unified view | `DeepseekV4AtomMLAAttentionSpec` and `post_bind_kv_cache()` bind a homogeneous BF16 `atom_unified_kv` view over vLLM storage. | Needed next: mixed BF16+FP8 unified layout contract if we want true ATOM FP8 compressed-tail storage. |
| ATOM compression plans | Active | `model_state.py` builds `CompressPlan`; `DeepseekCompressor._maybe_atom_compressor()` consumes it. | Keep; reduce CPU/GPU metadata rebuild cost. |
| Main CSA/HCA fused compressor | Active/default | `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`; accuracy passed. | Keep; profile fused kernel versus state update and plan prep separately. |
| Indexer-inner fused compressor | Present, default-off | Accuracy passed at lower concurrency, but full `MAX_NUM_SEQS=256` hit OOM/JIT pressure and C32 was slower. | Revisit after reducing allocation/JIT pressure; do not enable by default yet. |
| ATOM `qk_norm_rope_maybe_quant` | Active in ROCm subclass for Q/K RoPE, no Q/K quant outputs | `DeepseekV4ROCMAiterMLAAttention._fused_qnorm_rope_kv_insert()` calls `qk_norm_rope_maybe_quant(..., quant_q=False, quant_k=False)`. | Next: decide whether ATOM's quant-output mode is required for an FP8 unified KV attention contract. |
| Fused Q RMSNorm + group quant for `qr` | Active when compatible | `_q_norm_maybe_quant()` dispatches `rocm_aiter_ops.get_rmsnorm_group_fused_quant_op()`. | Keep; independent from compressor ordering. |
| SWA write | Active in ATOM attention path | ROCm subclass stores `_atom_last_kv`; `_maybe_atom_swa_write()` / prefill path call vendored `swa_write`. | Keep; verify decode/prefill ordering whenever compressor order changes. |
| ATOM paged decode / prefill attention | Present and gated by ATOM attention flags | `amd/rocm.py` can call `sparse_attn_v4_paged_decode()` and `sparse_attn_v4_paged_prefill()`. Prior fastest C32 did not prove this is a win over vLLM sparse decode. | Microbenchmark wrapper overhead and kernel-only time separately. |
| ATOM CSA direct decode | Removed from runtime hooks | Accuracy passed historically, but C32 was slower than default and the path bypassed the ATOM `csa_translate_pack` sequence. `amd/rocm.py` no longer imports or dispatches it. | Keep out of the production path unless a new fused ATOM-compatible kernel replaces both translate and attention. |
| ATOM top-k reuse / skip | Present, default-off | With `VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1` and `INDEX_TOPK_FREQ=4`, lmeval passed but C32 was `894.56 tok/s`, slower than best `916.28 tok/s`. | Functionally validated; not a default perf win yet. |
| ATOM MHC / HC | Partial | aiter MHC op wrappers exist; prior standalone MHC path did not pass full GSM8K in this branch. HC/head path was safe. | Keep safe HC/head path; retest aiter MHC only with exact ATOM call sequence and aiter 0.1.15.post1. |
| MoE | vLLM fused MoE with aiter backend, not exact ATOM dual-stream sequence | `amd/model.py` uses vLLM `FusedMoE` with aiter MXFP4; ATOM dual-stream/shared-expert overlap is not fully replicated. | Isolate MoE throughput before changing scheduler/worker paths. |
| Aux stream / overlap | Mostly disabled on ROCm deployment | Current passing path uses graph mode and ROCm aux stream behavior remains limited/default-off. | Only re-enable after single-stream ATOM op sequence is stable and measured. |

Practical implication: the next integration work should not assume "all ATOM
kernels are enabled." The highest-value remaining gap is the runtime shape of
the sparse attention/indexer path: vLLM still pays metadata, packing, and
layout-adaptation costs around the ATOM kernels, and some ATOM-equivalent
kernels are default-off because they were slower in the current wrapper.

The next high-impact component is true FP8 unified KV. The current
`DeepseekV4AtomMLAAttentionSpec` can reserve a fixed SWA prefix before the
compressed tail, and the binding path can expose that as `atom_unified_kv`, but
the validated path forces the whole tensor to BF16 (`cache_dtype_str="bf16"`).
That is intentional: the current ATOM paged attention wrappers consume one
homogeneous `[swa_prefix + compressed_tail, head_dim]` tensor. The target ATOM
recipe is FP8 KV cache, which needs a mixed-layout contract:

- BF16 or model-dtype SWA ring prefix, addressed by request-state slot.
- FP8/scale compressed CSA/HCA tail, ideally without BF16 gather workspace.
- Attention kernels that accept either split pointers (`swa_kv`, `tail_fp8`,
  `tail_scale`) or a raw byte base plus layout metadata.
- Metadata builders that keep native vLLM `fp8_ds_mla` fallback paths from
  reading BF16 ATOM-owned tails, and keep ATOM kernels from assuming a
  homogeneous tensor when the tail is FP8.

Until that contract exists, enabling more ATOM sparse attention flags can prove
functional correctness but will not necessarily recover the FP8-KV performance
shown in ATOM's recipe.

Intermediate bridge now present:

- `sparse_attn_v4_paged_decode()` already supports an optional
  `kv_scales=[total_pages, D/64]` fp32 argument for a homogeneous all-FP8
  `atom_unified_kv` tensor.
- `sparse_attn_v4_paged_decode_split_kv_reference()` now defines the target
  mixed-layout contract in a slow reference path: BF16/FP16 SWA pages live in
  `[0, swa_pages)`, compressed-tail pages live after `swa_pages`, and the tail
  may be BF16/FP16 or FP8 plus 1x64 fp32 scales.
- `sparse_attn_v4_paged_decode_split_kv()` now provides the fused
  production-shaped Triton decode path for that split layout when
  `kv_splits == 1`, matching the current deployment flag
  `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`.
- `DeepseekV4Attention.post_bind_kv_cache()` now publishes split-view aliases
  for the vLLM-owned BF16 allocation: `atom_split_kv_swa`,
  `atom_split_kv_compressed`, and `atom_split_kv_scales=None`.
- `DeepseekV4ROCMAiterMLAAttention._maybe_forward_decode_atom()` can route the
  generic paged-decode path through the split-load kernel with
  `VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1`, while default behavior stays on the
  previously validated homogeneous path.
- `DeepseekV4ROCMAiterMLAAttention._maybe_forward_decode_atom()` now forwards a
  future `self.atom_unified_kv_scales` attribute into the generic paged-decode
  calls.
- The old direct CSA decode path is removed from the runtime. Specialized
  direct HCA remains experimental/default-off and still assumes BF16/FP16
  `unified_kv`.
- `DeepseekV4AtomMLAAttentionSpec` now carries mixed-layout metadata for the
  future FP8 compressed tail:
  `atom_compressed_kv_dtype`, `atom_compressed_scale_dtype`, and
  `atom_compressed_scale_bytes_per_page`. The extra scale bytes participate in
  allocator page-size math and are preserved by scheduler spec generation.

This does not yet solve the desired mixed layout end-to-end. It makes the
generic decode path ready for a homogeneous-FP8 experiment if a binder later
attaches both `atom_unified_kv` and `atom_unified_kv_scales`, and it gives the
mixed BF16-SWA/FP8-tail design both a reference oracle and a fused
`kv_splits=1` split-load kernel. The current binder publishes BF16 split views
over the homogeneous allocation, so the real missing allocation piece is FP8
tail storage plus scale storage and a binder that publishes those views to the
attention/compressor kernels. A split-K/reduce split-layout variant is also
still missing if we want the same adaptive decode scheduling outside the
current forced `kv_splits=1` deployment mode.

Split-layout decode validation:

- CPU BF16 split wrapper versus existing homogeneous reference:
  `max_abs_diff = 0.0`.
- GPU BF16-tail fused Triton split-load versus homogeneous reference:
  `max_abs_diff = 0.0078125`.
- GPU FP8-tail fused Triton split-load plus 1x64 scales versus split-layout
  reference: `max_abs_diff = 0.0078125`.

Top-k reuse is default-off because the HF config loaded by vLLM for
`deepseek-ai/DeepSeek-V4-Pro` exposes `compress_rates` and `index_topk`, but
does not expose ATOM's `use_index_cache`, `index_topk_freq`, or
`index_topk_pattern` knobs. To preview ATOM's cached-index behavior without
changing model config files, launch with:

```bash
VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1 \
VLLM_ROCM_DSV4_ATOM_INDEX_TOPK_FREQ=4 \
bash launchdeepseekgraph.sh
```

or provide an explicit per-layer pattern:

```bash
VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1 \
VLLM_ROCM_DSV4_ATOM_INDEX_TOPK_PATTERN=R,S,S,S,R,S,S,S \
bash launchdeepseekgraph.sh
```

The pattern string uses ATOM's convention: `S` means skip top-k on that layer
and reuse the shared per-forward CSA top-k buffer from an earlier refresh CSA
layer. Any non-`S` character refreshes.

Validation with top-k reuse enabled:

```bash
MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1 \
VLLM_ROCM_DSV4_ATOM_INDEX_TOPK_FREQ=4 \
bash launchdeepseekgraph.sh
```

then unchanged `bash lmeval.sh`:

- GSM8K flexible-extract exact match: `0.9545 +/- 0.0057`
- GSM8K strict-match exact match: `0.9553 +/- 0.0057`

Fresh C32 benchmark after restarting the server with
`MAX_NUM_SEQS=32` and the same top-k reuse flags:

- result file:
  `bench-sparsemla/ds-v4-pro-nomtp-sparsemlahiptritonhybrid-C32.json`
- output throughput: `894.5569409118277 tok/s`
- total throughput: `1792.6082448740924 tok/s`
- mean TPOT: `34.8380172284985 ms`
- failed requests: `0`

This is accuracy-safe but not the best saved C32 run. It is slower than the
earlier `916.2795 tok/s` default saved result and faster than the
`867.3440 tok/s` metadata-cleanup rerun. Treat top-k reuse as functionally
validated but not yet a clear performance win in the current vLLM/ROCm
integration shape.

### Metadata / Conversion Profiling

A short C32-shaped profile run was collected with:

```bash
MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_EVERY=64 \
bash launchdeepseekgraph.sh
```

and:

```bash
vllm bench serve --backend openai-chat \
  --base-url http://127.0.0.1:8000 \
  --endpoint /v1/chat/completions \
  --model deepseek-ai/DeepSeek-V4-Pro \
  --dataset-name random --input-len 1024 --output-len 1024 \
  --num-prompts 64 --request-rate inf --max-concurrency 32 \
  --num-warmups 32 --random-range-ratio 0 --ignore-eos
```

Result:

- output throughput: `863.93 tok/s`
- total throughput: `1731.24 tok/s`
- mean TPOT: `35.82 ms`

Steady pure-decode metadata was measurable but not dominant:

- total ATOM metadata path: roughly `0.49-0.58 ms`
- detailed ATOM state metadata: roughly `0.086-0.093 ms`
- vLLM/super metadata: roughly `0.19-0.21 ms`
- compress/index plans: roughly `0.115-0.121 ms`

Mixed/prefill metadata remains a larger cost. In the same log, SWA metadata for
`tokens=64, reqs=32` reached roughly `16-17 ms`, while ragged prefill-like MLA
metadata reached roughly `3.8-4.2 ms`.

Conclusion: CPU metadata/conversion overhead exists, but steady pure decode
metadata alone cannot explain the remaining C32 gap. The higher-priority
suspects are GPU-side index translation/packing, compressor state writes,
layout adaptation between vLLM block tables and ATOM unified KV, missing exact
ATOM MHC/HC/indexer/MoE sequencing, and missing ATOM stream overlap.

Follow-up decode profiling from `server_decode_profile_c32.log` gives a more
specific split for steady pure-decode attention. These numbers were collected
before the `rocm.py` profiling-accounting fix that keeps decode-index cache
hits from being charged to `index_ms`, so treat `index_ms` as the measured
profile boundary cost in that run, not proof that the shared index writer
launched for every layer. Median per-layer timings:

- HCA/r128 layers at C32: `index_ms ~= 0.060`, `translate_ms ~= 0.003`,
  `kernel_ms ~= 0.054`, `total_ms ~= 0.119`.
- CSA/r4 layers at C32: `index_ms ~= 0.061`, `translate_ms ~= 0.038`,
  `kernel_ms ~= 0.053`, `total_ms ~= 0.155`.
- At smaller active decode batches (`T=4`), HCA stays around
  `index_ms ~= 0.051`, `kernel_ms ~= 0.060`; CSA stays around
  `index_ms ~= 0.053`, `translate_ms ~= 0.037`, `kernel_ms ~= 0.060`.

Interpretation: the CPU-side metadata preparation is not the main steady decode
cost, but the GPU-side conversion/adaptation boundary is large enough to
matter. For CSA layers, `csa_translate_pack` remains a real per-CSA-layer
adapter launch and is often comparable to the attention kernel itself. For HCA
layers, the current numbers need a post-fix rerun before separating cache-hit
sync overhead from actual index-writer work. The next useful experiment is
therefore not another broad metadata cleanup; it is to remove or fuse the
per-layer adapter work by making the attention kernels consume ATOM-native
request state/top-k/block-table metadata directly, or by moving more of the
index translation into the sparse attention kernel.

### Split-KV Decode Experiment

`VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1` adds an opt-in decode path that keeps
the vLLM-owned unified allocation split into fixed SWA pages from the ATOM SWA
ring view and compressed tail pages from the existing vLLM KV tail view. The
goal was to avoid requiring a homogeneous physical KV layout before the
ATOM-style paged decode kernel can run.

Unit/smoke validation:

- CPU BF16 split wrapper versus homogeneous reference: max absolute diff `0.0`
- GPU BF16-tail fused split-load versus homogeneous reference: max absolute
  diff `0.0078125`
- GPU FP8-tail fused split-load plus 1x64 scales versus split-layout reference:
  max absolute diff `0.0078125`
- no-`--enforce-eager` smoke request returned successfully under graph capture

Full accuracy with:

```bash
MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1 \
bash launchdeepseekgraph.sh
```

then unchanged `bash lmeval.sh`:

- GSM8K flexible-extract exact match: `0.9553 +/- 0.0057`
- GSM8K strict-match exact match: `0.9560 +/- 0.0056`

Fresh C32 benchmark after restarting the server with `MAX_NUM_SEQS=32` and the
same split-KV flag:

- result file:
  `bench-sparsemla/ds-v4-pro-nomtp-sparsemlahiptritonhybrid-C32.json`
- log file: `runlogs/split-kv-benchmark-c32.log`
- output throughput: `837.5775964609244 tok/s`
- total throughput: `1678.4269804080243 tok/s`
- mean TPOT: `37.26076639912507 ms`
- failed requests: `0`

This confirms the split-load decode path is accuracy-safe, but it is slower
than both the earlier best saved default C32 run (`916.2795284617761 tok/s`)
and the metadata-cleanup rerun (`867.3439657962101 tok/s`). The benchmark
progress showed repeated wave-level pauses at the C32 boundary, and the first
request in warmup/main still paid a large setup cost. That supports the
current analysis: a faster isolated decode kernel can be hidden or outweighed
by conversion, metadata, index translation, JIT/shape setup, and scheduler
wave costs at the served benchmark level.

Post-fix decode profiling was collected with:

```bash
MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=-1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=100000 \
bash launchdeepseekgraph.sh
```

and a 64-prompt C32 random 1024/1024 workload. The short profile run measured
`836.43 output tok/s`, matching the full split-KV benchmark closely enough for
timing diagnosis. Median sampled `T=32` decode timings:

| Path | Ratio | Index ms | Translate ms | Kernel ms | Total ms |
| ---- | ----- | -------- | ------------ | --------- | -------- |
| old homogeneous profile | 4 | `0.061` | `0.038` | `0.053` | `0.155` |
| homogeneous with index-reuse accounting fixed | 4 | `0.000` | `0.056` | `0.056` | `0.120` |
| split-KV decode | 4 | `0.000` | `0.057` | `0.066` | `0.138` |
| old homogeneous profile | 128 | `0.060` | `0.003` | `0.054` | `0.119` |
| homogeneous with index-reuse accounting fixed | 128 | `0.000` | `0.003` | `0.072` | `0.088` |
| split-KV decode | 128 | `0.000` | `0.003` | `0.085` | `0.101` |

Interpretation:

- The accounting fix matters: cache-hit layers now report `index_ms=0`, so the
  old `~0.060 ms` index time was boundary/sync accounting, not necessarily an
  index-writer launch.
- Split-KV decode is slower at the kernel level in this implementation:
  roughly `+0.010 ms` on CSA/r4 and `+0.013 ms` on HCA/r128 medians versus the
  homogeneous index-reuse profile.
- CSA still pays `~0.056-0.057 ms` of translation/packing per sampled decode
  layer, comparable to the attention kernel itself. That adapter launch is a
  more actionable target than CPU metadata cleanup.

Conclusion: conversion/layout logic can absolutely slow served performance,
but for this experiment the main measurable cost is not Python metadata. It is
the GPU-side adapter boundary and the extra split-load kernel work. The next
aligned integration target is to remove or fuse CSA translation by making the
decode kernel consume ATOM-native top-k/block-table/request-state metadata
directly, or to move that translation into the attention kernel. Keep
`VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE` default-off unless a future true unified
KV allocation requires it and the kernel is improved.

### Direct CSA Decode Profile

The existing experimental direct CSA decode path removes `csa_translate_pack`
by consuming raw seq-local top-k plus vLLM block tables inside
`sparse_attn_v4_csa_topk_paged_decode()`. This is a vLLM-side fusion
experiment; ATOM's current modeling flow still calls `csa_translate_pack`
before `sparse_attn_v4_paged_decode`.

Short C32 profile command:

```bash
MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=-1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=100000 \
bash launchdeepseekgraph.sh
```

with the same 64-prompt C32 random 1024/1024 workload:

- output throughput: `834.66 tok/s`
- total throughput: `1672.58 tok/s`
- mean TPOT: `37.07 ms`
- failed requests: `0`

Median sampled `T=32` decode timings:

| Path | Ratio | Index ms | Translate ms | Kernel ms | Total ms |
| ---- | ----- | -------- | ------------ | --------- | -------- |
| homogeneous with index-reuse accounting fixed | 4 | `0.000` | `0.056` | `0.056` | `0.120` |
| direct CSA | 4 | `0.000` | `0.003` | `0.075` | `0.085` |
| homogeneous with index-reuse accounting fixed | 128 | `0.000` | `0.003` | `0.072` | `0.088` |
| direct-CSA run HCA path | 128 | `0.000` | `0.003` | `0.069` | `0.084` |

The direct CSA kernel does remove the translator cost and improves median CSA
layer time (`0.085 ms` versus `0.120 ms`). However, the served throughput stays
flat (`834-836 tok/s` in these short profile runs, and the earlier full C32
direct-CSA run was also below the best default). First-use samples also show
large JIT outliers on layer 2 (`~300 ms` on several ranks), but warmup absorbs
those for steady benchmarks.

Conclusion: fusing CSA translation is directionally correct but not sufficient
alone. Even a roughly `0.035 ms` median saving on each of 30 CSA layers is only
about `1 ms` per decode step, while served TPOT remains around `37 ms`. The
next larger bottlenecks are likely outside CSA translation: MoE/MHC/HC
sequence, HCA/indexer work, stream overlap, and scheduler/graph wave behavior.
The direct-CSA experiment was removed from the runtime; any future CSA bypass
should be a new ATOM-compatible fusion of `csa_translate_pack` with paged
attention, not a revival of the old raw-top-k direct path.

### Direct HCA Decode Profile

The experimental direct HCA decode path bypasses the homogeneous packed decode
adapter for HCA/r128 layers by consuming the persistent HCA state and vLLM block
tables inside `sparse_attn_v4_hca_state_paged_decode()`. This is also a
vLLM-side fusion experiment, not the exact ATOM modeling sequence.

Accuracy command:

```bash
MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192 \
GPU_MEMORY_UTILIZATION=0.9 \
VLLM_ROCM_DSV4_ATOM_HCA_DIRECT_DECODE=1 \
bash launchdeepseekgraph.sh
```

with unchanged `lmeval.sh`:

- GSM8K flexible-extract exact match: `0.9553 +/- 0.0057`
- GSM8K strict-match exact match: `0.9560 +/- 0.0056`

Fresh C32 benchmark after restarting the server with `MAX_NUM_SEQS=32`:

- result file:
  `bench-sparsemla/ds-v4-pro-nomtp-sparsemlahiptritonhybrid-C32.json`
- log file: `runlogs/hca-direct-benchmark-c32.log`
- output throughput: `866.7232897996976 tok/s`
- total throughput: `1736.8322174501752 tok/s`
- mean TPOT: `35.95092337180271 ms`
- failed requests: `0`

Short profile run before the full benchmark measured `861.4577 output tok/s`,
`1726.2805 total tok/s`, and `35.8054 ms` TPOT. Median sampled `T=32` HCA/r128
decode timing improved from roughly `0.088 ms` total on the homogeneous
index-reuse path (`0.072 ms` kernel) to roughly `0.076 ms` total on direct HCA
(`0.064 ms` kernel).

Full C32 comparison:

| Run | Output tok/s | Total tok/s | Mean TPOT ms |
| --- | ------------ | ----------- | ------------ |
| revert compressor/aux | `926.0611` | `1855.7396` | `33.5030` |
| compressor-order off | `925.1308` | `1853.8755` | `33.5024` |
| current default no-eager | `916.2795` | `1836.1383` | `33.8686` |
| metadata cleanup | `867.3440` | `1738.0760` | `35.8970` |
| direct HCA | `866.7233` | `1736.8322` | `35.9509` |
| direct CSA copy | `843.1002` | `1689.4938` | `36.9721` |
| split-KV decode | `837.5776` | `1678.4270` | `37.2608` |

Conclusion: direct HCA is accuracy-safe, and it improves the isolated sampled
HCA/r128 layer timing, but it does not improve served C32 throughput. The result
is nearly identical to the metadata-cleanup run and remains well below the best
saved default/revert runs. This supports the current bottleneck analysis:
conversion and metadata preparation can slow the effective attention path, but
the measurable serving loss is not explained by Python metadata alone. GPU-side
adapter boundaries, layout conversion/packing, request-state updates, graph
shape behavior, and non-attention work can outweigh a faster individual decode
kernel. Keep `VLLM_ROCM_DSV4_ATOM_HCA_DIRECT_DECODE` default-off unless later
changes remove enough surrounding overhead to make the fused kernel visible at
the served benchmark level.

### Pure-Decode Legacy Metadata Skip

Follow-up after the direct CSA/HCA experiments: the deployed ATOM decode path
does not consume the legacy vLLM dense/ragged decode metadata generated by:

- `DeepseekV4FlashMLAMetadataBuilder._build_c128a_metadata()` for HCA/r128
  decode. Default ATOM HCA decode uses block tables plus model-state buffers,
  and direct HCA also consumes block tables directly.
- `DeepseekSparseSWAMetadataBuilder._compute_swa_indices_and_lens_kernel()` for
  SWA decode. ATOM decode writes SWA ring indices via
  `write_v4_paged_decode_indices()` from positions, state slots, and model
  state.

The code now skips those legacy pure-decode builders only when all of the
following are true:

- ROCm ATOM attention is enabled for the full model.
- No `VLLM_ROCM_DSV4_ATOM_ATTENTION_RATIOS` or
  `VLLM_ROCM_DSV4_ATOM_ATTENTION_LAYERS` filter is active.
- Debug fallback flags such as
  `VLLM_ROCM_DSV4_ATOM_RETURN_FALSE_AT_ENTRY` and
  `VLLM_ROCM_DSV4_ATOM_SKIP_DECODE_INDEX_WRITE` are not active.
- For C128A/HCA, `VLLM_ROCM_DSV4_ATOM_HCA_NATIVE_INDICES=1` is not active,
  because that bridge explicitly reads legacy C128A ragged metadata.

Mixed/prefill batches still build the legacy fields. This is intentional:
production keeps mixed-batch fallback available, and HCA prefill fallback still
uses `c128a_prefill_topk_indices`.

Validation:

- `python3 -m py_compile vllm/models/deepseek_v4/sparse_mla.py
  vllm/v1/attention/backends/mla/sparse_swa.py
  vllm/models/deepseek_v4/amd/rocm.py`
- Graph-mode smoke server:
  `MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024 MAX_MODEL_LEN=2048
  GPU_MEMORY_UTILIZATION=0.85 bash launchdeepseekgraph.sh`
- Graph capture completed without `--enforce-eager`.
- A `/v1/completions` request with `max_tokens=16` returned HTTP 200.
- Full accuracy server:
  `MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192
  GPU_MEMORY_UTILIZATION=0.9 bash launchdeepseekgraph.sh`
- Unchanged `lmeval.sh`:
    - GSM8K flexible-extract exact match: `0.9530 +/- 0.0058`
    - GSM8K strict-match exact match: `0.9538 +/- 0.0058`
- Fresh C32 benchmark after restarting the server with `MAX_NUM_SEQS=32`:
    - result file:
    `bench-sparsemla/ds-v4-pro-nomtp-skip-legacy-decode-metadata-C32.json`
    - log file: `runlogs/skip-legacy-decode-metadata-benchmark-c32.log`
    - output throughput: `867.4619146726486 tok/s`
    - total throughput: `1738.3123536264501 tok/s`
    - mean TPOT: `35.91196604007207 ms`
    - failed requests: `0`

Comparison:

| Run | Output tok/s | Total tok/s | Mean TPOT ms |
| --- | ------------ | ----------- | ------------ |
| revert compressor/aux | `926.0611` | `1855.7396` | `33.5030` |
| current default no-eager | `916.2795` | `1836.1383` | `33.8686` |
| skip legacy decode metadata | `867.4619` | `1738.3124` | `35.9120` |
| metadata cleanup | `867.3440` | `1738.0760` | `35.8970` |
| direct HCA | `866.7233` | `1736.8322` | `35.9509` |
| direct CSA copy | `843.1002` | `1689.4938` | `36.9721` |
| split-KV decode | `837.5776` | `1678.4270` | `37.2608` |

Conclusion: the skip is accuracy-safe and removes dead pure-decode metadata
work, but it is effectively tied with the prior metadata-cleanup run. It does
not recover the older `916-926 tok/s` results. This further supports that the
serving gap is not dominated by Python or builder-side legacy metadata alone;
the remaining costs are likely GPU-side adapter boundaries, request-state
updates, layout conversion/packing, graph/scheduler wave behavior, and
non-attention work.

### Adapter And Metadata Attribution Profile

Question tested: could conversion logic and metadata preparation make the
integrated decode path slower even when an individual ATOM attention kernel is
fast?

Profiling command:

```bash
MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=2048 MAX_MODEL_LEN=2048 \
GPU_MEMORY_UTILIZATION=0.85 \
VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_EVERY=16 \
VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_START_AFTER=20 \
VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_EVERY=128 \
VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_LAYER=-1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_START_AFTER=40 \
VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1 \
VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY=128 \
VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER=-1 \
bash launchdeepseekgraph.sh
```

Then a 16-prompt `/v1/completions` request with `max_tokens=64` was sent. The
server ran graph mode with V2 runner and no `--enforce-eager`. This is an
attribution run only, not a comparable C32 throughput benchmark, because
`MAX_MODEL_LEN=2048` and profiling syncs were enabled.

Current pure-decode ModelState profile with generic ATOM decode and main
compressor metadata skips active:

| Segment | Mean ms | P95 ms |
| ------- | ------- | ------ |
| `super().prepare_attn` | `0.2003` | `0.2140` |
| compress plans | `0.1147` | `0.1210` |
| ATOM state metadata | `0.1363` | `0.2020` |
| indexer attach | `0.0385` | `0.0440` |
| total ModelState metadata | `0.5000` | `0.5720` |

The ATOM state sub-breakdown is small: `map_batch ~= 0.0228 ms`,
`pos_commit ~= 0.0135 ms`, `decode_indptr ~= 0.0394 ms`, and total state
metadata `~= 0.0902 ms`.

Steady attention decode profile after dropping graph/JIT outliers
(`total_ms < 2`):

| Path | Ratio | T | Overhead ms | Kernel ms | Total ms | Overhead / Total |
| ---- | ----- | - | ----------- | --------- | -------- | ---------------- |
| `atom_kernel` | 4 | 16 | `0.0504` | `0.0542` | `0.1125` | `44.8%` |
| `atom_kernel` | 4 | 24 | `0.0545` | `0.0536` | `0.1163` | `46.9%` |
| `atom_kernel` | 4 | 32 | `0.0537` | `0.0557` | `0.1175` | `45.7%` |
| `atom_kernel` | 128 | 16 | `0.0052` | `0.0588` | `0.0744` | `7.0%` |
| `atom_kernel` | 128 | 24 | `0.0364` | `0.0610` | `0.1077` | `33.8%` |
| `atom_kernel` | 128 | 32 | `0.0031` | `0.0677` | `0.0832` | `3.7%` |

Interpretation:

- Yes, conversion and metadata preparation can hide kernel wins. The clearest
  example is CSA/r4 decode: `csa_translate_pack` plus related adapter work is
  roughly the same size as the decode kernel itself (`~0.05 ms` overhead versus
  `~0.054-0.056 ms` kernel). Any faster CSA kernel will not show its full
  benefit until the translator/adapter boundary is fused or removed.
- HCA/r128 adapter overhead is usually small when index reuse hits. The
  exception is an index-write/reuse boundary, visible around `T=24` in this
  profile.
- Current ModelState metadata is not free (`~0.5 ms` per worker scheduler step),
  but the actual ATOM state construction is only `~0.09 ms`; the remaining
  cost is vLLM common metadata plus compress-plan and attach bookkeeping.

### Packed `fp8_ds_mla` Status And Remaining Native-ATOM Gap

The branch now has an accuracy-correct packed ATOM compressed-tail path, but it
should not be described as "all ATOM kernels enabled" yet.

What is active and validated:

- `DeepseekV4AtomMLAAttentionSpec` can request vLLM-owned compressed-tail
  storage with `cache_dtype_str="fp8_ds_mla"` and 584 bytes per compressed
  token.
- `DeepseekV4Attention.post_bind_kv_cache()` and
  `DeepseekV4RocmAtomModelState._try_bind_atom_unified_kv_from_vllm()` bind
  the vLLM allocation as split ATOM views:
  BF16/model-dtype SWA prefix plus `uint8 [num_blocks, k_per_block, 584]`
  packed compressed tail.
- `DeepseekCompressor._maybe_atom_main_compressor_forward()` writes packed
  tail slots through `fused_compress_attn(..., packed_fp8_ds_mla=True)`.
- `sparse_attn_v4_paged_decode_split_kv()` and
  `sparse_attn_v4_paged_prefill_split_kv()` can read the packed tail directly:
  448 FP8 NoPE bytes, 64 BF16 RoPE values, and 8 UE8M0 scale bytes per
  compressed token.

Validation:

- Focused unit tests for packed allocation/binding passed together with the
  older mixed-tail tests:
  `tests/v1/worker/test_utils.py::{test_reshape_kv_cache_atom_packed_fp8_tail_keeps_584_byte_slots,test_deepseek_v4_post_bind_exposes_packed_atom_split_view}` and
  `tests/v1/core/test_kv_cache_utils.py::{test_atom_mla_packed_fp8_tail_uses_dsv4_584_byte_pages,test_scheduler_atom_mla_preserves_packed_fp8_tail_contract}`.
- GPU synthetic checks showed packed writer/read dequant differences around
  `0.109375-0.125` for FP8 NoPE and `0.0` for the BF16 RoPE tail.
- Graph-mode `lmeval.sh` with unchanged command passed GSM8K:
  flexible `0.9537528430629265 +/- 0.005784991662691855`, strict
  `0.954510993176649 +/- 0.005739657656722217`.
- Fresh C32 benchmark with packed FP8 ATOM KV:
  `bench-sparsemla/ds-v4-pro-packed-fp8-atomkv-C32-20260619-C32.json`,
  output throughput `808.2077176284764 tok/s`, total throughput
  `1619.572496653939 tok/s`, mean TPOT `38.542512715599834 ms`.

This is slower than the best BF16-tail saved run:
`bench-sparsemla/revert-compressor-aux-nomtp-C32.json`, output throughput
`926.0611 tok/s`, total throughput `1855.7396 tok/s`, mean TPOT `33.503 ms`.
So packed FP8 is functionally correct, but not yet a performance win in this
vLLM wrapper.

The main remaining gap is native-op coverage:

| ATOM operation area | Current vLLM state | Why this matters |
| --- | --- | --- |
| Main packed FP8 compressor writer | Compatibility Triton path | The wrapper explicitly disables aiter/flydsl when `packed_fp8_ds_mla=True`, because the public aiter/flydsl entry points assume the original dense/preshuffled cache contracts. |
| Packed sparse decode/prefill reader | Compatibility Triton split-KV path | It reads the right ATOM 584B layout, but it is not the same native ATOM/aiter sparse attention wrapper. The C32 result shows the extra split-load/dequant work is visible. |
| CSA translation | Separate adapter launch | CSA/r4 still pays `csa_translate_pack`-style adapter overhead that is often comparable to the attention kernel itself. |
| Indexer-inner compressor | Present but default-off | It passed accuracy only in constrained runs and was slower/OOM-prone at high concurrency. |
| MHC | Not part of the cache-layout contract | MHC can improve full-block performance, but it does not enable packed KV layout, compressor ordering, SWA rings, or sparse attention reads. Treat it as a later perf feature, not the blocker for ATOM attention/compressor correctness. |
| Auxiliary streams | Default-off | ATOM overlaps compressors with the main Q/KV path. The earlier vLLM aux-stream attempt was slower, so this should be revisited only after the single-stream op sequence is stable. |

Implication for the active goal: vLLM now has the scheduler, cache-spec,
binding, request-state, compressor-plan, packed-writer, and packed-reader
components needed to preview ATOM-style DSV4 integration without depending on
ATOM as a Python package. It still does not have all of the native ATOM kernel
benefit, because the packed path routes through vLLM compatibility wrappers
where ATOM uses its own aiter/flydsl/OPUS-oriented implementations and stream
structure.

Next aligned work:

1. Profile packed FP8 decode/prefill kernels against BF16 split/dense readers
   at deployment shapes and separate FP8 dequant time from softmax/dot time.
2. Add or locate a native aiter entry point that consumes ATOM's 584B
   `fp8_ds_mla` tail directly; do not route packed FP8 through the existing
   flydsl compressor unless its cache contract is proven compatible.
3. Fuse or eliminate the CSA translation adapter by making the attention
   kernel consume the ATOM top-k/request-state/block-table contract directly.
4. Only after the attention/compressor path is native and stable, re-test MHC
   fused post/pre and aux-stream overlap as independent performance features.

API audit against installed `aiter==0.1.15.post1`:

- `aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_opus` accepts only a
  homogeneous fp16/bf16 `unified_kv` with the same dtype as `q`; it cannot
  consume the split BF16-SWA plus packed-uint8 `fp8_ds_mla` compressed tail.
- `aiter.ops.flydsl.kernels.fused_compress_attn` accepts BF16 dense cache
  scatter, or FP8 dense/preshuffled cache plus a separate fp32
  `[num_blocks, k_per_block]` scale tensor. It does not expose the 584-byte
  packed DSV4 tail format.
- `aiter.ops.flydsl.kernels.fused_compress_attn_hca` is HCA-only and
  BF16-cache-only.
- Generic `aiter.mla_decode_fwd` supports dense paged MLA buffers, including
  fp8 variants, but its buffer contract is `[num_page, page_size, num_kv_heads,
  head_size]`, not the DSV4 sparse split-KV/top-k/indexer contract.

Therefore the current packed path cannot simply call an installed native aiter
entry point. Either aiter needs a packed DSV4 sparse attention/compressor entry
point, or vLLM must keep a local compatibility kernel for this layout.

Local compatibility optimization added after this audit:

- The packed split-KV decode/prefill kernels now load each UE8M0 scale byte
  once per token/group and broadcast it across the 64 NoPE dimensions, instead
  of loading the same scale byte once per token/dimension. This preserves the
  584-byte layout and dequant math, but should reduce packed-tail scale-load
  traffic and redundant `exp2` work.
- Validation run:
  `python3 -m pytest tests/v1/worker/test_utils.py::test_reshape_kv_cache_atom_packed_fp8_tail_keeps_584_byte_slots tests/v1/worker/test_utils.py::test_deepseek_v4_post_bind_exposes_packed_atom_split_view tests/v1/core/test_kv_cache_utils.py::test_atom_mla_packed_fp8_tail_uses_dsv4_584_byte_pages tests/v1/core/test_kv_cache_utils.py::test_scheduler_atom_mla_preserves_packed_fp8_tail_contract -q`
  returned `4 passed`.
- GPU synthetic validation after the scale-load change:
    - packed split-KV decode versus split-KV reference:
    max absolute diff `0.001953125`, finite outputs on both sides;
    - packed split-KV prefill versus materialized-unified prefill reference:
    max absolute diff `0.001953125`, finite outputs on both sides.
- Fresh C32 benchmark after restarting the server with packed FP8 ATOM KV and
  the scale-load optimization:
  `bench-sparsemla/ds-v4-pro-packed-fp8-scaleopt-C32-20260619-C32.json`
  measured output throughput `807.9484820158484 tok/s`, total throughput
  `1619.0530127895713 tok/s`, and mean TPOT `38.59318696322429 ms`.
  This is effectively unchanged from the previous packed FP8 ATOM KV run
  (`808.2077176284764 tok/s`, `38.542512715599834 ms`), so the optimization
  reduced redundant scale loads in the compatibility reader but did not move
  deployment-level throughput.
- Generic ATOM decode metadata and main compressor metadata are already skipped
  in pure decode (`skip_decode=True`, `skip_compressor=True`), so repeatedly
  optimizing the legacy sparse metadata builders is unlikely to recover the
  missing C32 throughput.
- The compressor Python profile hook did not emit decode records in this
  graph-mode run because graph replay bypasses the per-layer Python print path.
  Compressor decode needs either graph-capture instrumentation, device-side
  events, or an eager diagnostic run. The absence of compressor samples is not
  evidence that compressor cost is zero.
- Prefill/mixed batches still show legacy metadata and JIT costs. The profile
  captured first-use JIT warnings for `_build_prefill_chunk_metadata_kernel`,
  `_compute_prefill_metadata_kernel`, `_compute_swa_indices_and_lens_kernel`,
  `_update_compressor_states_kernel`, `_v4_paged_prefill_indices_kernel`, and
  `_csa_translate_pack_kernel`. Warmup coverage should include these shapes if
  latency spikes matter.

Next analysis direction:

- Focus on eliminating the CSA/r4 translation boundary in the production path,
  but do not use the current direct-CSA kernel as-is; previous full C32 results
  showed it was slower overall.
- Add graph-safe compressor timing if compressor state update or fused-compress
  is suspected. Python print timing is insufficient under breakable/full graph
  replay.
- If optimizing metadata further, target ModelState/common metadata and
  compress-plan bookkeeping, not the legacy sparse MLA/SWA pure-decode builders.

## 2026-06-19 Component Availability Follow-up

Question:

- Do we still have a missing ATOM attention/compressor component that prevents
  vLLM from benefiting from the ATOM kernels?

Live checks:

- Installed `aiter==0.1.15.post1` imports, but `torch.ops.aiter` does not expose
  `indexer_score_topk`.
- The local aiter source contains the constituent indexer primitives used by
  this branch: `cp_gather_indexer_k_quant_cache`, FP8 MQA logits, paged FP8 MQA
  logits, and `top_k_per_row_{prefill,decode}`.
- The ATOM modeling file uses `torch.ops.aiter.indexer_score_topk` as a
  dispatcher back into module-side Python logic; that dispatcher is not an
  installed aiter op here.

Implication:

- The default vLLM integration cannot exactly call ATOM's
  `torch.ops.aiter.indexer_score_topk` without a vLLM-local dispatcher or
  depending on ATOM as a package. A guarded opt-in local dispatcher now exists
  under `VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH=1`; it still needs full
  accuracy/performance validation.
- Functionally, vLLM has the pieces underneath that dispatcher: indexer Q
  projection/rope/quant, FP8 MQA score kernels, aiter top-k wrappers, compressed
  block-table metadata, and CSA translate/pack.
- Therefore the remaining indexer gap is not one absent aiter kernel. It is the
  vLLM wrapper shape around those kernels: metadata construction, temporary
  gathers, translation/packing, and graph-safe scratch ownership.

Current answer:

| Area | Available enough to run ATOM-equivalent logic? | Default status | Notes |
| --- | --- | --- | --- |
| Main compressor | Yes | Enabled | Uses ATOM-style fused-compress then state-update ordering. |
| Indexer-inner compressor | Partially | Disabled | Kernel path exists and passed lower-concurrency accuracy, but deployment-shape lmeval failed from resource pressure and C32 was slower. |
| Indexer score/top-k | Yes, decomposed; exact op name available as opt-in local fallback | Enabled for current path, dispatcher preview disabled by default | Installed aiter dispatcher is absent; vLLM can now register a local fallback and otherwise calls the constituent kernels directly. |
| Sparse attention decode/prefill | Yes | Enabled | ATOM paged sparse attention wrappers are wired; prior profiling shows CSA adapter overhead can hide kernel wins. |
| MHC/HC | Not required | Disabled for aiter MHC | MHC is relevant to ATOM perf parity, but not required for ATOM attention/compressor correctness. Standalone aiter MHC was not accuracy-safe in this branch. |
| Aux streams | Not required | Disabled | Needed for full ATOM overlap, not for proving the vLLM scheduler/KV integration. |

Next aligned integration target:

- Validate the opt-in vLLM-local `aiter::indexer_score_topk` dispatcher with
  graph-mode lmeval and C32 benchmark before considering it for default use.
- Keep the exact ATOM dispatcher default-off unless it demonstrably removes
  Python or metadata overhead.
- Focus on the CSA/r4 boundary that profiling identified. The direct-CSA
  experiment is now removed from `amd/rocm.py` and the package export surface,
  so any future bypass must be a real ATOM-compatible fusion of
  `csa_translate_pack` with paged attention, not the previous default-off
  direct path.

## 2026-06-19 No-Direct-CSA Validation

Change validated:

- Removed `VLLM_ROCM_DSV4_ATOM_CSA_DIRECT_DECODE` from
  `launchdeepseekgraph.sh`.
- Removed direct-CSA imports/dispatch from `amd/rocm.py` and exports from
  `amd/v4_kernels/__init__.py`.
- CSA/r4 now follows the ATOM modeling-file sequence in the active runtime path:
  indexer top-k -> `csa_translate_pack` -> paged sparse attention.

Static checks:

- `python3 -m py_compile vllm/models/deepseek_v4/amd/rocm.py
  vllm/models/deepseek_v4/amd/v4_kernels/__init__.py`
- `git diff --check -- vllm/models/deepseek_v4/amd/rocm.py
  vllm/models/deepseek_v4/amd/v4_kernels/__init__.py
  docs/deepseek_v4_atom_integration_notes.md`
- `bash -n launchdeepseekgraph.sh`

Small graph-mode smoke:

- Server:
    - `MAX_NUM_SEQS=4`
    - `MAX_NUM_BATCHED_TOKENS=1024`
    - `MAX_MODEL_LEN=2048`
    - `ENFORCE_EAGER=0`
- Health passed.
- One `/v1/completions` request succeeded.
- Logs confirmed V2 runner, breakable CUDA graph, and vLLM-owned ATOM unified
  KV binding.

Full accuracy:

- Server:
    - `MAX_NUM_SEQS=256`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `ENFORCE_EAGER=0`
- Accuracy command:
    - unchanged `bash lmeval.sh`
- Result file:
    - `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-19T20-59-27.515513.json`
- GSM8K:
    - flexible exact match: `0.9553 +/- 0.0057`
    - strict exact match: `0.9568 +/- 0.0056`
- This passes the requested `0.95 +/- 0.01` accuracy band.

Fresh C32 benchmark after restarting the server:

- Server:
    - `MAX_NUM_SEQS=32`
    - `MAX_NUM_BATCHED_TOKENS=8192`
    - `MAX_MODEL_LEN=8192`
    - `ENFORCE_EAGER=0`
- Benchmark command:
    - `RESULT_PREFIX=no-direct-csa-translate CONCURRENCIES=32 bash benchmarkvllm.sh`
- Result file:
    - `bench-sparsemla/no-direct-csa-translate-C32.json`
- C32 result:
    - completed `320`
    - failed `0`
    - output throughput: `885.16 tok/s`
    - total throughput: `1773.77 tok/s`
    - mean TPOT: `35.17 ms`
    - median TPOT: `35.15 ms`
    - p99 TPOT: `35.96 ms`
    - mean TTFT: `1032.04 ms`

Comparison:

- Recent default-off/metadata-scratch C32:
    - output `887.52 tok/s`
    - mean TPOT `35.13 ms`
- No-direct-CSA translate path:
    - output `885.16 tok/s`
    - mean TPOT `35.17 ms`
- Historical fastest saved C32:
    - `bench-sparsemla/revert-compressor-aux-nomtp-C32.json`
    - output `926.06 tok/s`
    - mean TPOT `33.50 ms`

Interpretation:

- Removing the non-ATOM direct-CSA dispatch preserves accuracy.
- End-to-end C32 is effectively tied with the recent default translate-path
  baseline and remains below the historical best.
- The CSA/r4 opportunity remains a real fusion problem: the existing
  `csa_translate_pack` launch is ATOM-equivalent and validated, but still
  contributes adapter overhead before paged attention.

## 2026-06-19 Direct-CSA Helper Cleanup

Follow-up cleanup after the no-direct-CSA validation:

- Removed the unexported direct-CSA decode/prefill helper functions from
  `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py` and
  `vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill.py`.
- Simplified the paged decode/prefill slot loaders back to packed
  `kv_indices` consumption, matching the active ATOM sequence:
  indexer top-k -> `csa_translate_pack` -> paged sparse attention.

Static checks:

- `python3 -m py_compile vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py
  vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill.py
  vllm/models/deepseek_v4/amd/v4_kernels/__init__.py
  vllm/models/deepseek_v4/amd/rocm.py`
- `rg` found no remaining direct-CSA helper symbols in the active ROCm
  attention/kernel files.
- `git diff --check` passed for the touched ROCm attention/kernel files.

Small graph-mode smoke:

- Server:
    - `MAX_NUM_SEQS=4`
    - `MAX_NUM_BATCHED_TOKENS=1024`
    - `MAX_MODEL_LEN=2048`
    - `ENFORCE_EAGER=0`
- Health passed.
- One `/v1/completions` request succeeded.
- Server was stopped after validation.

## 2026-06-19 Mixed FP8 Tail Spec Contract

Change:

- Extended `DeepseekV4AtomMLAAttentionSpec` with ROCm-only compressed-tail
  metadata:
    - `atom_compressed_kv_dtype`
    - `atom_compressed_scale_dtype`
    - `atom_compressed_scale_bytes_per_page`
- `real_page_size_bytes` now includes
  `storage_block_size * atom_compressed_scale_bytes_per_page`, so future FP8
  tail scale storage is budgeted by the vLLM KV allocator instead of being
  hidden in side allocations. The field name says page, but it represents one
  compressed KV row/slot in the ATOM sparse-attention tail; a vLLM storage
  block contains `storage_block_size` such rows.
- Spec merging now requires every ATOM layer in a cache group to agree on the
  SWA prefix and compressed-tail layout, which protects scheduler/core from
  mixing incompatible ROCm DSV4 layouts.

Validation:

- `python3 -m pytest
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_single_uniform_group_allocates_fixed_prefix
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_num_gpu_blocks_override_keeps_fixed_prefix
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_mixed_tail_scale_bytes_participate_in_allocation
  tests/v1/core/test_kv_cache_utils.py::test_scheduler_atom_mla_preserves_mixed_tail_contract
  tests/v1/worker/test_utils.py::test_representative_worker_spec_prefers_atom_mla
  -q`
- Result: `5 passed`.
- `python3 -m py_compile vllm/v1/kv_cache_interface.py
  vllm/v1/core/kv_cache_utils.py vllm/v1/worker/gpu/attn_utils.py
  vllm/models/deepseek_v4/attention.py
  vllm/models/deepseek_v4/amd/model_state.py`

This was initially a contract step, not a runtime FP8-tail enablement. The
active default launch still uses the validated homogeneous BF16 vLLM-owned
ATOM unified KV path. A follow-up flag now emits and binds mixed views for
experimentation; the compressor writer still needs to populate those FP8
views before it can run end-to-end.

- BF16/model-dtype SWA prefix view.
- FP8 compressed-tail view.
- FP32 per-1x64 scale view.

After that, the compressor writer must write FP8 tail + scales and the decode
kernel must consume those split views by default before lmeval/C32 can validate
whether this recovers ATOM FP8-KV performance.

## 2026-06-19 Mixed FP8 Tail View Plumbing

Change:

- `_reshape_kv_cache()` now treats
  `DeepseekV4AtomMLAAttentionSpec.atom_compressed_scale_bytes_per_page > 0`
  like per-row sidecar storage for the KV data tensor. It sets both the block
  stride and the row stride so a raw vLLM allocation can hold
  `[compressed_kv_row][scale_sidecar]` for every compressed row while the
  worker exposes only the strided compressed KV data view to the attention
  layer.
- `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1` now makes
  `DeepseekV4Attention.get_kv_cache_spec()` emit a mixed ATOM spec with:
    - BF16/model-dtype SWA prefix.
    - `torch.float8_e4m3fnuz` compressed tail.
    - FP32 per-1x64 scale sidecars.
    - private `cache_dtype_str="atom_fp8_1x64"` so the DeepSeek V4 backend keeps
    the semantic `[num_blocks, block_size, head_dim]` shape instead of the
    `fp8_ds_mla` 584-byte format.
- `DeepseekV4Attention.post_bind_kv_cache()` now binds one raw allocation into
  `atom_swa_kv`, `atom_split_kv_compressed`, and `atom_split_kv_scales` for
  mixed mode. It does not create `atom_unified_kv` for mixed mode because no
  homogeneous tensor exists.
- `DeepseekV4ROCMAiterMLAAttention._maybe_forward_decode_atom()` no longer
  requires a homogeneous `atom_unified_kv` when split KV views are present and
  `VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1` with
  `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`. This is required for a true mixed
  BF16-SWA/FP8-tail decode path, where no single homogeneous tensor exists.
- `DeepseekV4RocmAtomModelState` now publishes `atom_split_kv_swa`,
  `atom_split_kv_compressed`, and `atom_split_kv_scales=None` for the existing
  homogeneous BF16 side-allocation and vLLM-owned binding paths. That keeps the
  split decode interface available before the FP8 scale binder lands.
- `DeepseekV4RocmAtomModelState` also accepts vLLM-owned mixed KV when
  `post_bind_kv_cache()` has already published split views, instead of
  rejecting the FP8 tail because it is not model dtype.

Validation:

- `python3 -m pytest
  tests/v1/worker/test_utils.py::test_deepseek_v4_post_bind_exposes_mixed_atom_split_views
  tests/v1/worker/test_utils.py::test_reshape_kv_cache_strides_atom_mixed_tail_scales
  tests/v1/worker/test_utils.py::test_representative_worker_spec_prefers_atom_mla
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_mixed_tail_scale_bytes_participate_in_allocation
  tests/v1/core/test_kv_cache_utils.py::test_scheduler_atom_mla_preserves_mixed_tail_contract
  -q`
- Result: targeted subsets passed (`3 passed` for the bind/stride/allocation
  subset, and `3 passed` for the corrected allocation/stride subset).
- `python3 -m py_compile vllm/v1/worker/gpu/attn_utils.py
  tests/v1/worker/test_utils.py vllm/v1/kv_cache_interface.py
  tests/v1/core/test_kv_cache_utils.py
  vllm/models/deepseek_v4/attention.py
  vllm/models/deepseek_v4/amd/rocm.py
  vllm/models/deepseek_v4/amd/model_state.py`

## 2026-06-19 Experimental Mixed FP8 Tail Compressor Writer

Change:

- `fused_compress_attn()` now supports quantized cache writes with
  `quant_group_size < head_dim`. The existing indexer-inner path keeps
  `quant_group_size=head_dim`, one fp32 scale per compressed row, UE8M0 scale
  rounding, and MFMA 16x16 preshuffled FP8 storage.
- The new mixed main-tail path uses `quant_group_size=64`, one fp32 scale per
  64-wide group, raw `amax / 224.0` fnuz scales, and linear FP8 storage. It is
  only selected when `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1` has produced a
  `torch.float8_e4m3fnuz` compressed tail plus bound `atom_kv_scales`.
- The aiter/flydsl fused-compressor wrapper is intentionally bypassed for this
  mixed group-quant path. The available wrapper supports the existing BF16 main
  and indexer FP8 contracts, not the split scale-sidecar layout.
- `DeepseekCompressor._maybe_atom_main_compressor_forward()` now distinguishes
  three output contracts:
    - 128-dim indexer FP8 cache: existing `torch.uint8` raw allocation with
    embedded scale region.
    - 512-dim mixed main compressed tail: vLLM-owned
    `torch.float8_e4m3fnuz` rows plus separate fp32 1x64 scale sidecars.
    - 512-dim BF16 main compressed tail: previous homogeneous ATOM unified KV
    path.
- If an FP8 compressed tail is present without a bound scale sidecar, the
  compressor now raises instead of falling through to the BF16 writer.

Validation:

- `python3 -m py_compile
  vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py
  vllm/models/deepseek_v4/compressor.py`
- `python3 -m pytest
  tests/v1/worker/test_utils.py::test_deepseek_v4_post_bind_exposes_mixed_atom_split_views
  tests/v1/worker/test_utils.py::test_reshape_kv_cache_strides_atom_mixed_tail_scales
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_mixed_tail_scale_bytes_participate_in_allocation
  tests/v1/core/test_kv_cache_utils.py::test_scheduler_atom_mla_preserves_mixed_tail_contract
  -q`
- Result: `4 passed`.
- `git diff --check` passed for the touched mixed-KV, compressor, and test
  files.

Remaining mixed-tail runtime work:

- Validate GSM8K accuracy before treating mixed FP8 tail as usable. The
  current evidence proves allocation/binding and Python contracts, not
  kernel-level numerical correctness.

## 2026-06-19 Mixed FP8 Tail Startup Smoke

First smoke result:

- Command shape: graph mode, `VLLM_USE_V2_MODEL_RUNNER=1`, block size 128,
  `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=1024`, `MAX_MODEL_LEN=2048`,
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1`,
  `VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1`, and
  `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`.
- Initial startup failed in `_get_kv_cache_groups_uniform_groups()` because the
  mixed full-MLA page set no longer dominated every SWA/state page size. Fixed
  by padding the largest full-MLA page upward when a DeepSeekV4 SWA group has a
  larger page. Regression:
  `tests/v1/core/test_kv_cache_utils.py::test_deepseek_v4_grouping_pads_full_mla_when_swa_page_is_larger`.
- Second startup failed during graph capture because ATOM prefill still
  required homogeneous `atom_unified_kv`. Added
  `sparse_attn_v4_paged_prefill_split_kv()` and wired
  `_maybe_forward_prefill_atom()` to use split SWA/compressed views when no
  homogeneous unified tensor exists.
- Third startup succeeded:
    - KV cache initialized.
    - Breakable graph capture completed.
    - `/health` returned 200.
    - A tiny `/v1/completions` request returned 200.
- Tiny completion output was corrupted: prompt
  `"Question: What is 2+2? Answer:"` with `max_tokens=8`, `temperature=0`
  returned text like `����姸. .`. Therefore mixed FP8 tail is runnable but not
  numerically correct.
- Server was stopped after the smoke. Log:
  `/app/atomdsv4/runlogs/mixed_kv_smoke_server.log`.

Validation after the changes:

- `python3 -m py_compile
  vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill.py
  vllm/models/deepseek_v4/amd/v4_kernels/__init__.py
  vllm/models/deepseek_v4/amd/rocm.py
  vllm/v1/core/kv_cache_utils.py
  tests/v1/core/test_kv_cache_utils.py`
- `python3 -m pytest
  tests/v1/core/test_kv_cache_utils.py::test_deepseek_v4_grouping_pads_full_mla_when_swa_page_is_larger
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_mixed_tail_scale_bytes_participate_in_allocation
  tests/v1/core/test_kv_cache_utils.py::test_scheduler_atom_mla_preserves_mixed_tail_contract
  tests/v1/worker/test_utils.py::test_deepseek_v4_post_bind_exposes_mixed_atom_split_views
  tests/v1/worker/test_utils.py::test_reshape_kv_cache_strides_atom_mixed_tail_scales
  -q`
- Result: `5 passed`.

Next analysis targets:

- Verify the FP8 scale convention for mixed tail. The code currently uses the
  existing ROCm fnuz convention (`224.0`) for consistency with the indexer
  writer, while PyTorch reports `torch.finfo(torch.float8_e4m3fnuz).max ==
  240.0` in this environment.
- Compare the split prefill/decode dequantization against a BF16 reference on a
  tiny synthetic input. The live smoke proves the path runs, but the corrupted
  text suggests either scale convention, row/sidecar stride, quantization order,
  or prefix index interpretation is wrong.
- Add a GPU reference test for `fused_compress_attn(... quant_group_size=64,
  preshuffle=False)` before running full `lmeval.sh`.

## 2026-06-19 Mixed FP8 Tail Accuracy Failure

Full lmeval after fixing the mixed writer row-stride bug:

- Server: graph mode, `VLLM_USE_V2_MODEL_RUNNER=1`, block size 128,
  `MAX_NUM_SEQS=256`, `MAX_NUM_BATCHED_TOKENS=8192`,
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1`,
  `VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1`,
  `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`, no `--enforce-eager`.
- Client: unchanged `bash lmeval.sh`.
- Result:
    - flexible-extract exact match: `0.8992 +/- 0.0083`
    - strict-match exact match: `0.8984 +/- 0.0083`

Conclusion: do not benchmark this run. It is runnable but outside the required
`0.95 +/- 0.01` band.

Root-cause direction:

- This is not evidence that MHC is required for compressor correctness. ATOM's
  MHC kernels wrap transformer-block pre/post hidden-state combination and are
  important for ATOM perf parity, but compressor state ordering and sparse
  attention can be validated without MHC.
- The current mixed-main-tail experiment is not ATOM's documented FP8 KV
  format. ATOM describes a 584-byte DSV4 KV slot: `448` FP8 NoPE bytes, `64`
  BF16 RoPE values (`128` bytes), and `8` UE8M0 scale bytes. The experiment
  writes all `512` channels as FP8 with fp32 1x64 scale sidecars. That is a
  different numerical contract and is the leading explanation for the accuracy
  drop.

Next implementation idea:

- Either keep the main CSA/HCA compressed tail BF16 for accuracy, or implement
  the actual ATOM mixed slot layout in vLLM storage:
    - compressed-tail raw storage as bytes, not a homogeneous
    `torch.float8_e4m3fnuz [pages, 512]` tensor;
    - quantize/dequantize only the first `448` dimensions;
    - preserve the last `64` RoPE dimensions as BF16;
    - store `8` UE8M0 scale bytes per page;
    - update split decode/prefill kernels to read this 584-byte layout directly.

## 2026-06-19 Practical ATOM Component Split Audit

Current component status against "all ATOM attention/compressor logic" on
ROCm while preserving the vLLM scheduler:

- vLLM scheduler: sufficient. No evidence that the GPU worker must own DSV4
  request state. V2 ModelState is the right place for persistent per-request
  SWA rings, compressor state rings, compress plans, and metadata workspaces.
- vLLM KV-cache core/spec: needed and now partially present. The ATOM unified
  allocation is not just "extra request rings"; the sparse-attention KV storage
  must expose a fixed SWA prefix plus a compressed-tail layout that ATOM kernels
  understand.
- CUDA path: can remain untouched. The custom spec/binding is selected only
  for ROCm + DSV4 ATOM unified mode.
- Attention backend/kernels: still required. The kernels need direct packed
  reads/writes; a homogeneous tensor view is not enough for true ATOM FP8 KV.

New aligned implementation step:

- `DeepseekV4AtomMLAAttentionSpec` now carries
  `atom_compressed_layout`. Existing BF16 and sidecar experiments keep
  `dense`; the real ATOM FP8 target uses `fp8_ds_mla`.
- `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1` now requests the documented ATOM DSV4
  packed tail instead of the inaccurate all-FP8/fp32-sidecar experiment:
  `torch.uint8`, `cache_dtype_str="fp8_ds_mla"`, `448` FP8 NoPE bytes,
  `64` BF16 RoPE values, and `8` UE8M0 scale bytes per compressed token.
- `post_bind_kv_cache()` accepts and exposes the packed
  `[num_blocks, k_per_block, 584]` tail as `atom_split_kv_compressed`, marks
  `atom_split_kv_layout="fp8_ds_mla"`, and does not create a homogeneous
  `atom_unified_kv` view.
- The DeepSeek-V4 ATOM fixed-prefix allocator now dispatches even when the
  ATOM spec is not wrapped in `UniformTypeKVCacheSpecs`, so the SWA prefix is
  preserved for single-spec/single-group layouts too.
- The compressor currently raises for `atom_kv_layout=="fp8_ds_mla"` instead
  of running the old BF16 writer against a byte-packed cache. This is
  intentional: packed layout runtime is not correct until the writer and
  decode/prefill readers handle the real block layout.

Validation:

- `python3 -m py_compile
  vllm/v1/kv_cache_interface.py
  vllm/v1/core/kv_cache_utils.py
  vllm/models/deepseek_v4/attention.py
  vllm/models/deepseek_v4/compressor.py
  tests/v1/worker/test_utils.py
  tests/v1/core/test_kv_cache_utils.py`
- `python3 -m pytest
  tests/v1/worker/test_utils.py::test_reshape_kv_cache_atom_packed_fp8_tail_keeps_584_byte_slots
  tests/v1/worker/test_utils.py::test_deepseek_v4_post_bind_exposes_packed_atom_split_view
  tests/v1/worker/test_utils.py::test_reshape_kv_cache_strides_atom_mixed_tail_scales
  tests/v1/worker/test_utils.py::test_deepseek_v4_post_bind_exposes_mixed_atom_split_views
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_packed_fp8_tail_uses_dsv4_584_byte_pages
  tests/v1/core/test_kv_cache_utils.py::test_scheduler_atom_mla_preserves_packed_fp8_tail_contract
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_mixed_tail_scale_bytes_participate_in_allocation
  tests/v1/core/test_kv_cache_utils.py::test_scheduler_atom_mla_preserves_mixed_tail_contract
  -q`
- Result: `8 passed`.
- `git diff --check` passed for the touched files.

Next kernel work:

- Add packed writer support to `fused_compress_attn`: write token data at
  `block * block_stride + slot * 576`, store FP8 bytes for dims `[0, 448)`,
  store BF16 RoPE dims `[448, 512)` at byte offset `448`, and store 8 UE8M0
  scales in the block scale region `block *block_stride + k_per_block* 576
    - slot * 8`.
- Add packed read support to split decode and prefill. They must compute the
  same block-packed offsets instead of treating `[block, slot, 584]` as
  per-token-interleaved 584-byte rows.
- Only after those packed kernels match a BF16 reference should we rerun
  unchanged `lmeval.sh`; do not benchmark a packed run before it passes
  accuracy.

## 2026-06-19 Packed fp8_ds_mla Writer/Reader Implementation

The packed ATOM DSV4 KV layout is now implemented for the ROCm ATOM split-KV
path:

- `fused_compress_attn(..., packed_fp8_ds_mla=True)` writes the compressed tail
  in the block-packed `fp8_ds_mla` format:
    - token data region: `block * block_stride + slot * 576`;
    - NoPE dims `[0, 448)`: FP8 e4m3 bytes with one UE8M0 scale byte per 64 dims;
    - RoPE dims `[448, 512)`: BF16 bytes at byte offset `448`;
    - scale region: `block * block_stride + k_per_block * 576 + slot * 8`.
- `DeepseekCompressor` now accepts `atom_kv_layout=="fp8_ds_mla"` when the
  bound cache is `uint8 [num_blocks, k_per_block, 584]`, and calls the packed
  compressor writer.
- ROCm split decode and split prefill wrappers pass `compressed_kv_layout` to
  their kernels. The packed branch keeps the `[block, slot, 584]` geometry and
  computes the ATOM block-packed offsets directly instead of flattening to a
  dense `[pages, 512]` view.
- The packed decode reference can dequantize the 584-byte layout for unit-test
  comparison.

GPU synthetic validation already completed:

- One-slot packed compressor writer vs BF16 writer:
    - BF16 reference finite: true
    - packed output nonzero: true
    - NoPE max diff after dequant: `0.109375`
    - RoPE max diff: `0.0`
- Split decode packed reader vs dense BF16 split reference:
    - one tail slot max diff: `0.1171875`
- Split prefill packed reader vs dense BF16 split reference:
    - one tail slot max diff: `0.125`
- `k_per_block=2`, slot 1 addressing:
    - slot 0 writer max diff: `0.125`
    - slot 1 writer max diff: `0.125`
    - slot 1 RoPE max diff: `0.0`
    - decode slot 1 max diff: `0.125`

Focused non-GPU validation also passes after the packed runtime changes:

- `python3 -m py_compile` on the modified cache/spec/model/kernel/test files.
- `git diff --check` on the modified cache/spec/model/kernel/test/doc files.
- `python3 -m pytest
  tests/v1/worker/test_utils.py::test_reshape_kv_cache_atom_packed_fp8_tail_keeps_584_byte_slots
  tests/v1/worker/test_utils.py::test_deepseek_v4_post_bind_exposes_packed_atom_split_view
  tests/v1/worker/test_utils.py::test_reshape_kv_cache_strides_atom_mixed_tail_scales
  tests/v1/worker/test_utils.py::test_deepseek_v4_post_bind_exposes_mixed_atom_split_views
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_packed_fp8_tail_uses_dsv4_584_byte_pages
  tests/v1/core/test_kv_cache_utils.py::test_scheduler_atom_mla_preserves_packed_fp8_tail_contract
  tests/v1/core/test_kv_cache_utils.py::test_atom_mla_mixed_tail_scale_bytes_participate_in_allocation
  tests/v1/core/test_kv_cache_utils.py::test_scheduler_atom_mla_preserves_mixed_tail_contract
  -q`
- Result: `8 passed`.

Remaining gate:

- Run a live server smoke with `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1` and the split
  decode path enabled.
- If the smoke is coherent, run unchanged `lmeval.sh`.
- Benchmark C32 only after GSM8K returns to the required `0.95 +/- 0.01` band.

## 2026-06-19 Packed fp8_ds_mla Runtime Validation

The first live packed-KV smoke revealed that `ModelState` still rebound
vLLM-owned KV storage as one homogeneous BF16 unified view, silently bypassing
the packed 584-byte tail. The bind path is now layout-aware:

- dense layout: creates the old BF16/model-dtype homogeneous `atom_unified_kv`;
- `fp8_ds_mla` layout: creates a BF16 SWA-prefix view from the underlying
  storage, keeps the vLLM-bound `uint8 [num_blocks, k_per_block, 584]`
  compressed tail, sets `atom_split_kv_layout="fp8_ds_mla"`, and binds the
  compressor with `atom_kv_layout="fp8_ds_mla"`.

Smoke evidence:

- Small server: `MAX_NUM_SEQS=4`, `MAX_MODEL_LEN=4096`,
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1`,
  `VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1`,
  `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`, graph mode.
- Startup log reported
  `layout_counts={'fp8_ds_mla': 61}`.
- Short completion and a longer 1,818-token prompt both returned successfully.

Full accuracy run:

- Server: default `launchdeepseekgraph.sh` deployment settings for lmeval
  (`MAX_NUM_SEQS=256`, `MAX_NUM_BATCHED_TOKENS=8192`, `MAX_MODEL_LEN=8192`),
  plus packed mixed KV and split-decode flags, graph mode, no
  `--enforce-eager`.
- Client: unchanged `bash lmeval.sh`.
- Result file:
  `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-19T23-11-37.065327.json`
- GSM8K:
    - flexible-extract exact match: `0.9537528430629265 +/- 0.005784991662691855`
    - strict-match exact match: `0.954510993176649 +/- 0.005739657656722217`

This passes the required `0.95 +/- 0.01` accuracy band.

C32 benchmark after a fresh server restart:

- Server: same packed flags, `MAX_NUM_SEQS=32`, graph mode, no
  `--enforce-eager`.
- Client: `RESULT_PREFIX=ds-v4-pro-packed-fp8-atomkv-C32-20260619
  CONCURRENCIES=32 bash benchmarkvllm.sh`.
- Result file:
  `bench-sparsemla/ds-v4-pro-packed-fp8-atomkv-C32-20260619-C32.json`
- Completed: `320`
- Failed: `0`
- Output throughput: `808.2077176284764 tok/s`
- Total throughput: `1619.572496653939 tok/s`
- Mean TPOT: `38.542512715599834 ms`

Comparison:

- Previous best BF16-tail run:
  `bench-sparsemla/revert-compressor-aux-nomtp-C32.json`
    - output throughput: `926.0610778396323 tok/s`
    - total throughput: `1855.7395817645756 tok/s`
    - mean TPOT: `33.50296501712195 ms`
- Previous no-direct-CSA translate run:
  `bench-sparsemla/no-direct-csa-translate-C32.json`
    - output throughput: `885.1578831441936 tok/s`
    - total throughput: `1773.7734142694192 tok/s`
    - mean TPOT: `35.170615295353684 ms`
- Previous current-default no-eager run:
  `bench-sparsemla/ds-v4-pro-nomtp-current-default-noeager-C32.json`
    - output throughput: `916.2795284617761 tok/s`
    - total throughput: `1836.1382738316058 tok/s`
    - mean TPOT: `33.8685634798457 ms`

Conclusion:

- The packed `fp8_ds_mla` path is now a real, accuracy-correct runtime path in
  vLLM-owned KV storage.
- It is not the fastest path yet. At C32 it is about `12.7%` slower than the
  previous best BF16-tail run by output throughput.
- Next analysis should focus on whether packed FP8 dequant/read overhead,
  metadata conversion, or the current split prefill/decode wrapper is erasing
  the memory-bandwidth benefit of the smaller compressed tail.

## 2026-06-20 Native ATOM Component Audit

Question:

- Do we have all of the necessary components to get the benefit of all ATOM
  kernels in vLLM?

Current answer:

- No. vLLM has enough scheduler/KV-cache/model-state integration to run an
  ATOM-style DSV4 attention and compressor path with correct GSM8K accuracy,
  but not enough native aiter/ATOM component coverage to claim the full ATOM
  kernel benefit.
- KV-cache/workspace architecture details are tracked in
  `docs/deepseek_v4_rocm_kvcache_workspace_design.md`. The important boundary:
  persistent request rings and unified-KV buffers belong in
  `DeepseekV4RocmAtomModelState` or vLLM KV-cache allocation, while
  `WorkspaceManager.get_simultaneous()` remains scratch-only and must not hold
  data across forwards/layers.
- Source audit evidence: ATOM's modeling file imports
  `fused_compress_attn`, `sparse_attn_v4_paged_decode`,
  `sparse_attn_v4_paged_prefill`, `swa_write`, and
  `update_compressor_states` from `atom.model_ops.sparse_attn_v4`.
  Installed `aiter==0.1.15.post1` exports MLA/MHC/cache primitives and
  `pa_sparse_prefill_opus`, but it does not export those ATOM sparse-attention
  or state-write APIs. Using the ATOM package directly would violate the
  integration constraint, so vLLM currently vendors/adapts the needed logic.
- Guard test:
  `tests/kernels/test_deepseek_v4_atom_dependency_contract.py` scans vLLM
  Python imports and asserts that runtime code does not import `atom` or
  `atom.*`. It also checks the imported `aiter` module surface for the missing
  ATOM sparse-attention/state-write names.

Current component status:

| Component | Current vLLM path | Native ATOM/aiter benefit status | Evidence / implication |
| --- | --- | --- | --- |
| vLLM scheduler integration | Uses V2 model runner and vLLM block-table lifetime | Sufficient | No GPU-worker rewrite is needed for persistent request rings; `ModelState` owns ROCm-only request state. |
| ROCm DSV4 unified KV allocation | `DeepseekV4AtomMLAAttentionSpec` adds SWA prefix bytes plus compressed tail bytes | Sufficient for preview | vLLM-owned packed FP8 path binds `layout_counts={'fp8_ds_mla': 61}` and passes GSM8K. CUDA path remains separate. |
| vLLM KV block zeroing | `KVBlockZeroer` now includes `AttentionSpec` layers and groups segments by page size | Sufficient for mixed ATOM layouts | The zeroer no longer skips ATOM MLA specs or assumes one page size across regular and ATOM compressed-tail tensors. |
| vLLM GPU runner KV reshape | `_reshape_kv_cache_tensors()` now handles ATOM fixed prefixes, layer `cache_dtype_str`, compressed `storage_block_size`, and sidecar-scale strides | Sufficient for runtime binding | The actual GPU model runner now matches the helper path for packed 584-byte FP8 tails and BF16 SWA prefixes. |
| Packed `fp8_ds_mla` compressed tail | Local vLLM compatibility writer/reader for `[num_blocks, k_per_block, 584]` | Missing native aiter component | Installed `aiter==0.1.15.post1` does not expose a packed DSV4 split-KV sparse attention or packed compressor API. Best packed C32 is `808.46 tok/s`, not a win. The ordered split-load experiment was disabled after regressing to `728.23 tok/s`. |
| Sparse attention decode | Local Triton split-KV reader for packed path; homogeneous BF16 path follows ATOM-style paged sparse kernel | Partially native-equivalent | ATOM's Python `sparse_attn_v4.py` is also Triton-backed by default, so Triton itself is not disqualifying. The mismatch is the split BF16-SWA plus packed-tail contract and extra adapter work. |
| Sparse attention prefill | OPUS can be used only for homogeneous `unified_kv`; split packed path uses local Triton | Missing packed OPUS path | `pa_sparse_prefill_opus` expects homogeneous KV and cannot consume split SWA plus 584-byte packed tail. |
| Main compressor | flydsl aiter path where supported; local Triton when packed FP8 or flattened HCA layout is requested | Partially native | `packed_fp8_ds_mla=True` explicitly disables flydsl because public flydsl cache contracts are dense/preshuffled or sidecar-scale layouts, not 584-byte tail. |
| HCA compressor | flydsl HCA for original BF16 HCA shape; local Triton for flattened/adapted layouts | Partially native | vLLM's packed/flattened layouts are scheduler adapters. They need either a matching aiter entry point or must remain compatibility kernels. |
| Indexer score/top-k | vLLM decomposes ATOM dispatcher into constituent aiter/vLLM ops by default; an opt-in local `aiter::indexer_score_topk` fallback now dispatches back to `DeepseekV4Indexer` | Sufficient but not identical | Installed `aiter==0.1.15.post1` does not provide the dispatcher. The lower-level score/top-k pieces exist; the fallback closes the op-name parity gap but not the metadata/translation cost by itself. |
| CSA translate/pack | Separate local adapter before attention | Correct but likely costly | Direct-CSA bypass was removed because it was slower and not ATOM-sequence faithful. Future work should fuse translate with attention instead of reviving that bypass. |
| Q/K norm + RoPE + optional quant | Vendored ATOM kernel with vLLM imports/style | Aligned | Diff against ATOM is mostly import/style and type-hint changes. |
| State writes / compressor ordering | Local kernels with extra bounds checks and graph-safe sizing | Aligned enough | Ordering remains read-before-update for ATOM compressor correctness. Extra checks adapt to vLLM scheduler padding. |
| MHC / HC | aiter MHC exists, but this branch keeps it out of the correctness-critical path | Independent perf feature | MHC affects residual/norm compute, not KV layout, compressor rings, sparse attention reads, or compressor ordering. It is not required to make attention/compressor kernels work. |
| Auxiliary streams | Default-off | Later perf feature | Previous vLLM attempt was slower. Revisit only after single-stream native component gaps are resolved. |

Conclusion for the active integration plan:

- Keep the current practical split:
    - no GPU-worker changes for request rings;
    - ROCm-only vLLM core/attention changes for DSV4 cache spec, binding, and
    split/packed readers;
    - CUDA untouched.
- Current guard coverage for the split:
    - `test_deepseek_v4_kv_cache_spec_stays_regular_mla_off_rocm` proves
    non-ROCm does not emit `DeepseekV4AtomMLAAttentionSpec`, even when the
    ATOM unified/mixed flags are patched on.
    - `test_deepseek_v4_model_state_cls_stays_default_off_rocm` and
    `test_deepseek_v4_model_state_cls_stays_default_without_atom_state` prove
    `DeepseekV4RocmAtomModelState` is selected only for ROCm plus the ATOM
    state flag.
    - `test_deepseek_v4_post_bind_stays_noop_off_rocm` proves the ATOM unified
    KV post-bind views are not created off ROCm.
    - Targeted validation command passed:
    `pytest -q tests/v1/worker/test_utils.py -k 'deepseek_v4_kv_cache_spec_stays_regular_mla_off_rocm or deepseek_v4_kv_cache_spec_uses_atom_mla_only_for_rocm_unified or deepseek_v4_model_state_cls_stays_default_off_rocm or deepseek_v4_model_state_cls_stays_default_without_atom_state or deepseek_v4_model_state_cls_uses_atom_state_only_for_rocm_atom or deepseek_v4_post_bind_stays_noop_off_rocm'`.
- Do not claim full ATOM-kernel benefit yet. The missing native component is
  not MHC. The blocking gap is native support for the packed DSV4 sparse
  attention/compressor contract:
  `BF16 SWA prefix + uint8 fp8_ds_mla 584-byte compressed tail + ATOM index
  metadata`.
- The next performance-focused implementation should either:
  1. add/find an aiter entry point for the packed split-KV DSV4 contract, or
  2. fuse vLLM's compatibility adapters so the local path stops paying separate
     `csa_translate_pack` and packed dequant/read overhead before attention.

## 2026-06-20 Experimental CSA Translate Fusion

Change:

- Added an opt-in decode-only fusion flag:
  `VLLM_ROCM_DSV4_ATOM_FUSE_CSA_TRANSLATE_DECODE=1`.
- Scope is intentionally narrow:
    - CSA/r4 decode only;
    - split-KV decode only;
    - `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`;
    - default remains the validated ATOM-sequence path:
    `indexer top-k -> csa_translate_pack -> paged sparse attention`.

Behavior:

- When the flag is enabled, vLLM skips the separate `csa_translate_pack` launch
  for CSA decode and passes raw indexer top-k plus block tables into
  `sparse_attn_v4_paged_decode_split_kv`.
- The split-KV attention kernel resolves CSA top-k rows directly:
  `topk_local -> block_tables -> swa_pages + physical_block * k_per_block + slot`.
- Existing SWA tail indices in `csa_indices` are still consumed from the same
  buffer, so this keeps the decode layout contract used by
  `write_v4_paged_decode_indices`.

Focused validation:

- `python3 -m py_compile vllm/models/deepseek_v4/amd/rocm.py
  vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`
- `git diff --check -- vllm/models/deepseek_v4/amd/rocm.py
  vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`
- Synthetic dense-tail GPU check:
    - baseline: `csa_translate_pack` then split-KV decode;
    - fused: split-KV decode resolves CSA top-k directly;
    - max absolute diff: `0.0`, finite outputs.
- Synthetic packed `fp8_ds_mla` GPU check:
    - same baseline/fused comparison with `[num_blocks, k_per_block, 584]`
    compressed tail;
    - max absolute diff: `0.0`, finite outputs.
- Small graph-mode server smoke:
    - server flags included `MAX_NUM_SEQS=4`, `MAX_NUM_BATCHED_TOKENS=1024`,
    `MAX_MODEL_LEN=2048`, `ENFORCE_EAGER=0`,
    `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1`,
    `VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1`,
    `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=1`, and
    `VLLM_ROCM_DSV4_ATOM_FUSE_CSA_TRANSLATE_DECODE=1`;
    - graph capture completed;
    - startup bound `layout_counts={'fp8_ds_mla': 61}`;
    - `/v1/models` returned `200`;
    - one short `/v1/completions` request returned `200`;
    - server was stopped afterward and no stale vLLM/lm-eval/benchmark process
    remained.

Full accuracy validation:

- Server: default `launchdeepseekgraph.sh` lmeval deployment shape
  (`MAX_NUM_SEQS=256`, `MAX_NUM_BATCHED_TOKENS=8192`, `MAX_MODEL_LEN=8192`),
  graph mode, no `--enforce-eager`, packed mixed KV, split-KV decode, and
  `VLLM_ROCM_DSV4_ATOM_FUSE_CSA_TRANSLATE_DECODE=1`.
- Client: unchanged `bash lmeval.sh`.
- Result file:
  `results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-20T00-03-01.986966.json`
- GSM8K:
    - flexible-extract exact match: `0.9514783927217589 +/- 0.0059184686189210885`
    - strict-match exact match: `0.9522365428354814 +/- 0.005874387536229305`

This passes the required `0.95 +/- 0.01` accuracy band.

C32 benchmark after a fresh server restart:

- Server: same fused CSA packed flags, `MAX_NUM_SEQS=32`, graph mode, no
  `--enforce-eager`.
- Client:
  `RESULT_PREFIX=ds-v4-pro-packed-fp8-fused-csa-C32-20260620 CONCURRENCIES=32 bash benchmarkvllm.sh`
- Result file:
  `bench-sparsemla/ds-v4-pro-packed-fp8-fused-csa-C32-20260620-C32.json`
- Completed: `320`
- Failed: `0`
- Output throughput: `774.60 tok/s`
- Total throughput: `1552.22 tok/s`
- Mean TPOT: `40.32 ms`

Comparison:

- Previous packed FP8 ATOM KV run:
  `bench-sparsemla/ds-v4-pro-packed-fp8-atomkv-C32-20260619-C32.json`
    - output throughput: `808.2077176284764 tok/s`
    - total throughput: `1619.572496653939 tok/s`
    - mean TPOT: `38.542512715599834 ms`
- Previous best BF16-tail run:
  `bench-sparsemla/revert-compressor-aux-nomtp-C32.json`
    - output throughput: `926.0610778396323 tok/s`
    - total throughput: `1855.7395817645756 tok/s`
    - mean TPOT: `33.50296501712195 ms`

Conclusion:

- The fused CSA decode path is accuracy-safe, but not a performance win.
- At C32 it is about `4.2%` slower than the previous packed FP8 baseline by
  output throughput and about `16.4%` slower than the best BF16-tail run.
- Do not enable `VLLM_ROCM_DSV4_ATOM_FUSE_CSA_TRANSLATE_DECODE=1` by default.
  The likely issue is that folding the CSA address resolution into the
  attention kernel increases decode-side index math/register pressure more than
  it saves by removing the separate `csa_translate_pack` launch.

## 2026-06-20 ROCm DSV4 Env Registry Cleanup

Change:

- Registered the Python-source-used `VLLM_ROCM_DSV4_*` integration flags in
  `vllm/envs.py`.
- Added small `env_bool` / `env_int` / `env_float` / `env_str` helpers to keep
  the registry compact.

Why:

- Recent lmeval and benchmark launches printed noisy
  `Unknown vLLM environment variable detected` warnings for the ROCm DSV4 ATOM
  flags.
- The model code already caches those flags at module import, so this change
  does not alter runtime behavior. It only makes the flags visible to vLLM's
  environment validation/cache machinery.

Verification:

- `python3 -m py_compile vllm/envs.py`
- `git diff --check -- vllm/envs.py docs/deepseek_v4_atom_integration_notes.md`
- Source-used flag audit:
    - scanned Python sources for `VLLM_ROCM_DSV4_*`;
    - found `62` unique flags;
    - direct `vllm.envs.validate_environ(hard_fail=True)` completed with
    `missing []`.
- Deployment-shape audit:
    - scanned Python and shell sources for `VLLM_ROCM_DSV4_*`;
    - found `62` unique flags;
    - `validate_environ(hard_fail=True)` and `enable_envs_cache()` completed with
    `missing []`;
    - launch-like empty `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=` resolves to
    default `0`, matching `launchdeepseekgraph.sh` when the flag is unset.

## 2026-06-20 ROCm-Only KV Spec Gating Test

Change:

- Added focused worker utility tests around
  `DeepseekV4Attention.get_kv_cache_spec`.

Why:

- The practical split requires CUDA/default paths to remain on vLLM's existing
  MLA KV-cache spec, while ROCm + DSV4 unified mode may return the custom ATOM
  spec with a fixed SWA prefix.

Coverage:

- Non-ROCm/default path:
    - even if the ATOM unified flag is monkeypatched on, `current_platform.is_rocm`
    false returns `MLAAttentionSpec`;
    - no `atom_vllm_unified_kv_*` attributes are attached to the attention stub.
- ROCm + unified path:
    - returns `DeepseekV4AtomMLAAttentionSpec`;
    - records `fp8_ds_mla` compressed-tail layout;
    - computes and stores the fixed SWA prefix bytes from
    `max_num_seqs * sliding_window * head_dim * dtype_size`.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'deepseek_v4_kv_cache_spec'`
- `pytest -q tests/v1/worker/test_utils.py`
- `python3 -m py_compile vllm/envs.py vllm/models/deepseek_v4/attention.py`
- `git diff --check -- tests/v1/worker/test_utils.py vllm/envs.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 ROCm-Only ModelState Gating Test

Change:

- Added focused worker utility tests around
  `DeepseekV4ForCausalLM.get_model_state_cls`.

Why:

- The practical split keeps persistent DSV4 SWA/compressor request state in a
  model-specific `ModelState`, but that state must not affect CUDA or default
  vLLM execution.

Coverage:

- Non-ROCm/default path:
    - even if the ATOM state flag is monkeypatched on, `current_platform.is_rocm`
    false returns vLLM's `DefaultModelState`.
- ROCm without ATOM state:
    - returns `DefaultModelState`.
- ROCm + ATOM state:
    - returns `DeepseekV4RocmAtomModelState`.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'model_state_cls'`
- `pytest -q tests/v1/worker/test_utils.py`
- `python3 -m py_compile vllm/models/deepseek_v4/amd/model.py vllm/models/deepseek_v4/amd/model_state.py tests/v1/worker/test_utils.py`
- `git diff --check -- tests/v1/worker/test_utils.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 Multi-Layer ATOM KV Prefix Allocation Test

Change:

- Added a core KV-cache utility test for one regular MLA layer plus multiple
  `DeepseekV4AtomMLAAttentionSpec` layers with distinct fixed SWA prefixes.

Why:

- The ROCm unified DSV4 path needs vLLM-owned allocation to reserve each
  ATOM layer's fixed SWA prefix while using one shared `num_blocks` value for
  the scalable compressed tails and any regular MLA layers.

Coverage:

- `get_kv_cache_configs` subtracts the sum of all ATOM fixed prefixes before
  computing scalable blocks.
- Each ATOM layer receives its own `KVCacheTensor` with
  `fixed_prefix_size=atom_swa_prefix_bytes`.
- The regular MLA layer allocation remains prefix-free and keeps the same
  block count as the ATOM compressed tails.

Verification:

- `pytest -q tests/v1/core/test_kv_cache_utils.py -k 'atom_mla_multiple_layers or atom_mla or deepseek_v4_grouping or scheduler_atom'`
- `pytest -q tests/v1/worker/test_utils.py`
- `python3 -m py_compile tests/v1/core/test_kv_cache_utils.py tests/v1/worker/test_utils.py vllm/v1/core/kv_cache_utils.py vllm/v1/kv_cache_interface.py vllm/v1/worker/utils.py`

## 2026-06-20 ROCm-Only Post-Bind View Gating Test

Change:

- Added negative worker utility tests around
  `DeepseekV4Attention.post_bind_kv_cache`.

Why:

- The vLLM-owned ATOM KV allocation creates bind-time SWA/tail views for ROCm
  graph capture, but that view creation must stay invisible to CUDA/default
  paths and to uncompressed SWA-only layers.

Coverage:

- Off-ROCm path:
    - even with ATOM unified-KV metadata present, `post_bind_kv_cache` returns
    before shape validation and does not attach `atom_swa_kv`,
    `atom_split_kv_compressed`, or `atom_unified_kv`.
- ROCm SWA-only path:
    - layers with `compress_ratio <= 1` also return without attaching ATOM
    split/unified views.
- Positive mixed and packed FP8 post-bind tests still verify that ROCm
  compressed MLA layers expose the expected SWA prefix, compressed tail, scale
  sidecar, and packed `fp8_ds_mla` layout.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'post_bind'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/attention.py`
- `git diff --check -- tests/v1/worker/test_utils.py vllm/models/deepseek_v4/attention.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 ModelState Packed vLLM-Owned KV Binding Test

Change:

- Added a worker utility test for
  `DeepseekV4RocmAtomModelState._try_bind_atom_unified_kv_from_vllm` on the
  packed `fp8_ds_mla` layout.

Why:

- The packed FP8 path is the closest current vLLM-owned storage shape to
  ATOM's desired split layout: BF16 SWA prefix in the same allocation as a
  vLLM-visible packed uint8 compressed tail. The `ModelState` metadata must
  publish that storage without pretending the whole allocation is a homogeneous
  unified tensor.

Coverage:

- The helper binds a BF16 `atom_swa_kv` view from byte offset zero of the
  vLLM-owned storage.
- The compressed tail remains the vLLM-bound uint8 tensor with 584-byte packed
  slots.
- `DeepseekV4RocmAtomUnifiedKVBuffers.compressed_kv_cache` points at that tail,
  while `unified_kv` stays empty for the non-homogeneous packed layout.
- Attention and compressor objects receive `atom_kv_cache`,
  `atom_kv_scales=None`, and `atom_kv_layout='fp8_ds_mla'`.
- No `atom_unified_kv` homogeneous tensor is attached for packed FP8.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'model_state_binds_packed'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/model_state.py`
- `git diff --check -- tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/model_state.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 Worker KV Post-Bind Hook Test

Change:

- Added a worker utility regression test proving `bind_kv_cache` invokes a
  layer's optional `post_bind_kv_cache(kv_cache)` hook after assigning
  `layer.kv_cache`.

Why:

- The ROCm DSV4 vLLM-owned KV path relies on this hook to expose SWA/tail views
  at bind time, before graph capture. Without the hook, the allocator could
  reserve the right storage but attention modules would not publish the ATOM
  views until later `ModelState` metadata preparation.

Coverage:

- Existing `runner_kv_caches` ordering is preserved.
- Each layer sees its assigned `kv_cache` before `post_bind_kv_cache` runs.
- The hook remains optional; layers without it keep the original behavior.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'bind_kv_cache_calls_post_bind_hook'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/v1/worker/utils.py`
- `git diff --check -- tests/v1/worker/test_utils.py vllm/v1/worker/utils.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 ModelState Unified-KV Fallback Guard Test

Change:

- Added worker utility tests around
  `DeepseekV4RocmAtomModelState._maybe_allocate_atom_unified_kv`.

Why:

- Benchmarks with `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1` must prove the
  vLLM-owned KV-cache path. If binding fails, silently falling back to the
  side allocation would produce misleading evidence.

Coverage:

- When vLLM-owned unified KV is requested and binding fails, `ModelState`
  raises instead of allocating side buffers.
- When vLLM-owned unified KV is not requested, the side-allocation path still
  allocates homogeneous `atom_unified_kv`, SWA views, compressed-tail views,
  and compressor `atom_kv_cache` metadata from the runtime KV-cache spec.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'model_state_refuses_vllm_owned_kv_fallback or model_state_side_allocates'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/model_state.py`
- `git diff --check -- tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/model_state.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 vLLM-Owned ATOM KV Block-Size Guard

Change:

- Tightened `DeepseekV4Attention.get_kv_cache_spec` so the
  vLLM-owned ATOM KV path also enforces `--block-size 128`, even when the
  broader `_ATOM_ROCM_DSV4_ENABLED` feature bundle is not active.

Why:

- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1` can emit
  `DeepseekV4AtomMLAAttentionSpec` by itself. That path still targets ATOM's
  DSV4 KV/block contract and must not run with vLLM's default block size.

Coverage:

- Off-ROCm/default still returns normal `MLAAttentionSpec`.
- ROCm + vLLM-owned ATOM KV with block size `128` returns
  `DeepseekV4AtomMLAAttentionSpec`.
- ROCm + vLLM-owned ATOM KV with block size `256` raises, even if
  `_ATOM_ROCM_DSV4_ENABLED` is false.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'vllm_owned_atom_kv_enforces_block_size or kv_cache_spec'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/attention.py`
- `git diff --check -- tests/v1/worker/test_utils.py vllm/models/deepseek_v4/attention.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 ATOM MLA Spec Kind Test

Change:

- Added explicit core KV-cache coverage that
  `DeepseekV4AtomMLAAttentionSpec` is classified as
  `KVCacheSpecKind.MLA_ATTENTION`.

Why:

- The ROCm ATOM spec should stay inside vLLM's existing MLA scheduler/backend
  category. It carries extra ROCm-only allocation metadata, but it is not a new
  CUDA/default cache kind.

Coverage:

- Plain `MLAAttentionSpec` and `DeepseekV4AtomMLAAttentionSpec` both classify
  as `MLA_ATTENTION`.
- Specialized MLA subclasses such as `SlidingWindowMLASpec` still keep their
  more specific kind because subclass checks remain ordered before base checks.

Verification:

- `pytest -q tests/v1/core/test_kv_cache_utils.py -k 'kv_cache_spec_kind or atom_mla_multiple_layers or atom_mla or deepseek_v4_grouping or scheduler_atom'`
- `python3 -m py_compile tests/v1/core/test_kv_cache_utils.py vllm/v1/kv_cache_interface.py`
- `git diff --check -- tests/v1/core/test_kv_cache_utils.py vllm/v1/kv_cache_interface.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 ATOM MLA Spec Merge Contract Test

Change:

- Added core KV-cache tests for `DeepseekV4AtomMLAAttentionSpec.merge`.
- Extended merge preservation coverage to the FP8+fp32 sidecar-scale layout.

Why:

- vLLM grouping/scheduler code may merge per-layer specs. For the ROCm ATOM
  layout, merging must preserve the packed/split layout metadata and reject
  incompatible SWA-prefix or compressed-tail contracts.

Coverage:

- Compatible packed `fp8_ds_mla` ATOM specs merge into a
  `DeepseekV4AtomMLAAttentionSpec` while preserving cache dtype, compression
  ratio, model version, SWA prefix/pages, compressed dtype, and layout.
- Compatible sidecar-scale ATOM specs merge while preserving compressed dtype,
  `"dense"` layout, fp32 scale dtype, and scale bytes per compressed page.
- Merge rejects mismatched SWA prefix bytes, SWA pages, compressed layout,
  compressed scale dtype, and compressed scale bytes per page.

Verification:

- `pytest -q tests/v1/core/test_kv_cache_utils.py -k 'merge_atom_mla_spec'`
- `python3 -m py_compile tests/v1/core/test_kv_cache_utils.py vllm/v1/kv_cache_interface.py`
- `git diff --check -- tests/v1/core/test_kv_cache_utils.py vllm/v1/kv_cache_interface.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 ATOM MLA Registry Contract Test

Change:

- Added `DeepseekV4AtomMLAAttentionSpec` to the KV-cache registry test matrix.

Why:

- The custom ROCm ATOM spec should use vLLM's existing full/MLA cache manager
  infrastructure. It should not require a new scheduler manager path or expose
  a new default/CUDA cache category.

Coverage:

- `KVCacheSpecRegistry.get_manager_class(DeepseekV4AtomMLAAttentionSpec(...))`
  resolves to `FullAttentionManager`.
- `KVCacheSpecRegistry.get_uniform_type_base_spec(...)` resolves to
  `FullAttentionSpec`, matching the production registration in
  `register_all_kvcache_specs`.
- The generic custom-spec registration test also covers subclasses of the ATOM
  spec through the same registry mechanism.

Verification:

- `pytest -q tests/v1/test_kv_cache_spec_registry.py`
- `python3 -m py_compile tests/v1/test_kv_cache_spec_registry.py vllm/v1/core/single_type_kv_cache_manager.py vllm/v1/kv_cache_spec_registry.py`
- `git diff --check -- tests/v1/test_kv_cache_spec_registry.py vllm/v1/core/single_type_kv_cache_manager.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 ATOM MLA Mixed Uniform-Type Test

Change:

- Added explicit core KV-cache coverage for mixed regular MLA and ATOM MLA
  specs.

Why:

- Mixed regular/ATOM layers should not be treated as a single homogeneous
  per-layer spec, because ATOM layers need fixed SWA prefixes and compressed
  tail metadata. They should still be accepted as one uniform type so vLLM can
  group them under the existing MLA/full-attention manager and let the
  DeepSeek-V4 allocator split the actual tensor layout.

Coverage:

- `is_kv_cache_spec_uniform({"regular": MLAAttentionSpec, "atom":
  DeepseekV4AtomMLAAttentionSpec})` returns false.
- `UniformTypeKVCacheSpecs.from_specs(...)` still returns a
  `UniformTypeKVCacheSpecs` containing both specs.

Verification:

- `pytest -q tests/v1/core/test_kv_cache_utils.py -k 'atom_mla_mixed_regular_is_uniform_type or atom_mla_mixed_uniformity or merge_atom_mla_spec or kv_cache_spec_kind or atom_mla_multiple_layers or atom_mla or deepseek_v4_grouping or scheduler_atom'`
- `python3 -m py_compile tests/v1/core/test_kv_cache_utils.py vllm/v1/core/kv_cache_utils.py vllm/v1/kv_cache_interface.py`
- `git diff --check -- tests/v1/core/test_kv_cache_utils.py vllm/v1/core/kv_cache_utils.py vllm/v1/kv_cache_interface.py docs/deepseek_v4_atom_integration_notes.md`

## 2026-06-20 Worker Reshape Fixed-Prefix Tail Alias Test

Change:

- Strengthened worker reshape coverage for `DeepseekV4AtomMLAAttentionSpec`.

Why:

- ATOM-style vLLM-owned KV uses one raw allocation with a fixed SWA prefix and
  compressed MLA tail pages. Attention backends should receive a view over the
  compressed tail, while the fixed prefix remains addressable for SWA/state
  handling from the same storage.

Coverage:

- Mixed compressed-tail layout with scale sidecars returns a view whose
  underlying storage is the original raw KV tensor and whose storage offset is
  exactly `atom_swa_prefix_bytes`.
- Packed `fp8_ds_mla` 584-byte tail layout has the same alias/offset contract.
- Existing stride/value checks still prove page and row strides skip ATOM scale
  sidecars and preserve 584-byte packed slots.

Implication:

- This proves the worker reshape layer can expose the ATOM compressed tail to
  attention without copying or losing the fixed SWA prefix in the owning
  allocation. It does not by itself prove native ATOM sparse attention kernels
  understand the full split-KV contract; that remains a kernel/backend contract.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'reshape_kv_cache'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/v1/worker/gpu/attn_utils.py`

## 2026-06-20 Side-Allocation Dense Layout Metadata Test

Change:

- The model-state side-allocation binder now explicitly sets
  `attn.atom_split_kv_layout = "dense"` and
  `compressor.atom_kv_layout = "dense"`.

Why:

- The vLLM-owned dense and packed KV binding paths already publish explicit
  layout metadata. The older side-allocation path relied on compressor defaults,
  which made allocation source part of the behavior. Keeping the layout field
  explicit lets attention/compressor code select dense versus packed behavior
  from the same contract regardless of whether storage came from vLLM KV-cache
  allocation or the model-state side allocation.

Coverage:

- `test_deepseek_v4_model_state_side_allocates_when_not_vllm_owned` now asserts
  both attention and compressor layout metadata are `"dense"` for side
  allocation.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'side_allocates or packed_vllm_owned_kv or refuses_vllm_owned_kv_fallback'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/model_state.py`

## 2026-06-20 Unified-KV Buffer Layout Metadata Test

Change:

- `DeepseekV4RocmAtomUnifiedKVBuffers` now carries
  `compressed_kv_scales` and `compressed_kv_layout` dictionaries alongside
  `compressed_kv_cache`.

Why:

- Future ATOM kernels should be able to consume model-state metadata directly.
  A compressed-tail tensor alone is ambiguous: dense BF16, mixed FP8+sidecar
  scales, and packed `fp8_ds_mla` all require different load/dequant behavior.
  Recording layout and optional scales in the bundle avoids rediscovering that
  contract from per-layer module attributes.

Coverage:

- Dense model-state side allocation records layout `"dense"` and no scales in
  both the buffer bundle and the attention/compressor attributes.
- Packed vLLM-owned KV binding records layout `"fp8_ds_mla"` and no sidecar
  scales in the buffer bundle, matching the attention/compressor attributes.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'side_allocates or packed_vllm_owned_kv or refuses_vllm_owned_kv_fallback'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/model_state.py`

## 2026-06-20 Sidecar vLLM-Owned KV Bundle Test

Change:

- Added worker coverage for the FP8+fp32 sidecar-scale vLLM-owned KV binding
  path in `DeepseekV4RocmAtomModelState`.

Why:

- Post-bind can expose sidecar-scale split views, but ModelState also needs to
  preserve the exact scale tensor and layout in
  `DeepseekV4RocmAtomUnifiedKVBuffers` so future kernels can consume metadata
  directly instead of rediscovering module attributes.

Coverage:

- A sidecar-scale tail bound by `DeepseekV4Attention.post_bind_kv_cache` is
  accepted by `_try_bind_atom_unified_kv_from_vllm`.
- The metadata bundle stores the same compressed tail tensor, the same fp32
  scale sidecar tensor, and layout `"dense"`.
- The compressor receives the same cache/scale/layout metadata.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'sidecar_vllm_owned_kv'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/model_state.py vllm/models/deepseek_v4/attention.py`

## 2026-06-20 Packed vLLM-Owned SWA Reset Test

Change:

- `_reset_atom_request_slot` now zeros split-view `atom_swa_kv` when the
  unified-KV bundle has no homogeneous `unified_kv` tensor.

Why:

- Packed `fp8_ds_mla` vLLM-owned KV stores the BF16 SWA prefix and packed FP8
  tail in one raw allocation, but the metadata bundle intentionally has
  `unified_kv == ()` because there is no single homogeneous tensor view.
  The previous reset logic saw the bundle and only iterated `unified_kv`, so it
  skipped SWA cleanup for packed split KV.

Coverage:

- `test_deepseek_v4_model_state_binds_packed_vllm_owned_kv` now calls
  `_reset_atom_request_slot(0)` after binding packed vLLM-owned KV.
- The test asserts the BF16 SWA prefix is zeroed while the packed compressed
  tail byte remains unchanged.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'packed_vllm_owned_kv or side_allocates'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/model_state.py`

## 2026-06-20 Forward Metadata Unified-KV Bundle Test

Change:

- Added worker coverage that `build_legacy_runner_metadata` preserves the exact
  `DeepseekV4RocmAtomUnifiedKVBuffers` object in
  `DeepseekV4RocmAtomStateMetadata.unified_kv_buffers`.

Why:

- The unified-KV bundle now carries compressed-tail layout and scale metadata.
  That contract only helps attention/compressor kernels if it reaches the
  per-forward metadata object consumed by the model path. The test keeps this
  explicit for packed `fp8_ds_mla` metadata.

Coverage:

- A minimal model-state instance with a packed `fp8_ds_mla` buffer bundle builds
  legacy-runner ATOM metadata.
- The returned metadata points to the same bundle object and preserves
  `compressed_kv_cache`, `compressed_kv_scales`, and `compressed_kv_layout`.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'legacy_metadata_carries_unified_kv_layout_bundle'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/model_state.py`

## 2026-06-20 Post-Bind Split Layout Metadata Test

Change:

- `DeepseekV4Attention.post_bind_kv_cache` now explicitly sets
  `atom_split_kv_layout` for every valid vLLM-owned split KV layout and rejects
  unknown compressed-tail layout strings during bind.

Why:

- Packed `fp8_ds_mla`, dense BF16, and the older FP8+fp32 sidecar layout all
  need different attention/compressor behavior. The sidecar-scale path
  previously relied on default `"dense"` lookups instead of publishing layout
  metadata on the attention module.

Coverage:

- Mixed FP8+sidecar bind publishes `attn.atom_split_kv_layout == "dense"` and
  the same layout/cache/scale metadata on the compressor.
- Unknown layout strings fail during post-bind before model-state metadata or
  kernels consume them.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'post_bind'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/attention.py`

## 2026-06-20 Split-Only KV Decode Selection Test

Change:

- ROCm ATOM decode now automatically selects the split-KV decode path when
  there is no homogeneous `atom_unified_kv` tensor but split SWA/compressed
  views are present.

Why:

- Packed `fp8_ds_mla` vLLM-owned KV has one raw allocation, but no valid
  homogeneous tensor view. Requiring the older
  `VLLM_ROCM_DSV4_ATOM_SPLIT_KV_DECODE=1` flag for that split-only layout made
  a correctly bound packed KV cache fail at attention time unless an unrelated
  experimental flag was also set.

Behavior:

- Dense/homogeneous layouts keep the previous opt-in split behavior.
- Split-only layouts use split decode automatically. The wrapper call passes
  `kv_splits=1` explicitly, so this does not depend on the deployment override
  used by the homogeneous split-K path.

Coverage:

- `test_deepseek_v4_split_only_kv_decode_auto_selects_split_path` asserts:
  split-only KV selects split decode without the opt-in flag, dense unified KV
  does not, the opt-in flag still requires `kv_splits=1` for dense split
  decode, and split-only KV remains selected when the deployment override is
  unset or different.

Verification:

- `pytest -q tests/v1/worker/test_utils.py -k 'split_only_kv_decode_auto_selects_split_path'`
- `python3 -m py_compile tests/v1/worker/test_utils.py vllm/models/deepseek_v4/amd/rocm.py`

## 2026-06-20 Split-KV Decode Layout Contract Test

Change:

- Added CPU-only tests for the public split-KV decode wrapper's compressed-tail
  layout validation.
- Split-KV prefill layout validation is now factored into a CPU-testable helper
  and runs before the CUDA/HIP-only launch guard.

Why:

- Packed `fp8_ds_mla` carries UE8M0 scales inside the 584-byte tail slot. It
  must not be treated like the older FP8+fp32 sidecar-scale layout. The wrapper
  should reject invalid layout/scale combinations before a deployment run
  reaches a kernel-specific failure.

Coverage:

- Decode and prefill reject packed `fp8_ds_mla` plus sidecar scales.
- Decode and prefill reject packed `fp8_ds_mla` with non-584-byte tail geometry.
- Decode and prefill reject unknown compressed-tail layout strings.
- Decode accepts a valid uint8 `[num_blocks, k_per_block, 584]` packed tail
  through the CPU reference path and returns finite output when the compressed
  page is indexed.
- Prefill accepts a valid uint8 `[num_blocks, k_per_block, 584]` packed tail
  through layout validation and, on CPU, then fails only at the expected
  CUDA/HIP launch guard.

Verification:

- `pytest -q tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
- `python3 -m py_compile tests/kernels/attention/test_deepseek_v4_split_kv_contract.py vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill.py`

## 2026-06-20 Packed FP8 Compressor Layout Contract Test

Change:

- Added explicit packed `fp8_ds_mla` compressor validation and CPU-only tests.

Why:

- The split attention wrappers already reject sidecar fp32 scales for packed
  `fp8_ds_mla`, because that layout embeds UE8M0 scales in each 584-byte tail
  slot. The compressor write path should enforce the same contract before it
  launches the fused compressor kernel.

Coverage:

- Packed compressor validation rejects missing `atom_kv_cache`.
- Packed compressor validation rejects sidecar scale tensors.
- Packed compressor validation rejects non-584-byte tail geometry.
- Valid uint8 `[num_blocks, k_per_block, 584]` tails without sidecar scales are
  accepted.

Verification:

- `pytest -q tests/kernels/test_deepseek_v4_compressor_contract.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
- `python3 -m py_compile tests/kernels/test_deepseek_v4_compressor_contract.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py vllm/models/deepseek_v4/compressor.py vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`

## 2026-06-20 Fused Compressor Packed Layout Contract Test

Change:

- Added explicit packed `fp8_ds_mla` validation inside the low-level
  `fused_compress_attn` wrapper and CPU-only tests for that validator.

Why:

- The high-level compressor path now rejects invalid packed tails, but future
  direct callers of the fused compressor wrapper should not be able to bypass
  the same contract. Packed `fp8_ds_mla` requires uint8
  `[num_blocks, k_per_block, 584]`, embedded UE8M0 scales, head_dim 512,
  RoPE dim 64, a positive block size, and a scatter map.

Coverage:

- Valid packed tails are accepted.
- Sidecar scales, missing scatter metadata, bad tail geometry, and wrong head
  dimensions are rejected before kernel launch.
- A direct `fused_compress_attn(..., packed_fp8_ds_mla=True)` call with block
  tables but no KV cache raises the packed-layout `RuntimeError` before the
  generic scatter assertions run.
- Packed `fp8_ds_mla` bypasses the FlyDSL fused compressor even when the
  shape is otherwise FlyDSL-supported. The public FlyDSL compressor contract is
  dense/preshuffled or sidecar-scale oriented, not the packed 584-byte tail
  layout, so this path must stay on the local compatibility writer until a
  native packed DSV4 entry point exists.

Verification:

- `pytest -q tests/kernels/test_deepseek_v4_fused_compress_contract.py tests/kernels/test_deepseek_v4_compressor_contract.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
- `pytest -q tests/kernels/test_deepseek_v4_fused_compress_contract.py tests/kernels/test_deepseek_v4_compressor_contract.py tests/kernels/test_deepseek_v4_atom_dependency_contract.py`
  passed: `14 passed`.
- `python3 -m py_compile tests/kernels/test_deepseek_v4_fused_compress_contract.py tests/kernels/test_deepseek_v4_compressor_contract.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py vllm/models/deepseek_v4/compressor.py vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`

## 2026-06-20 Runtime KV Metadata Bundle Precedence

Change:

- ROCm ATOM decode/prefill now resolve compressed KV tail, sidecar scales, and
  layout from `DeepseekV4RocmAtomStateMetadata.unified_kv_buffers` when that
  per-step metadata bundle is present.
- Layer attributes remain the SWA view source for packed split-only
  vLLM-owned KV, because that layout intentionally has no homogeneous
  `atom_unified_kv` tensor.
- `_bind_atom_unified_kv_buffers` is now layout-aware: a bundle may contain
  compressed split-only layers with `active_layer_ids` populated and
  `unified_kv=()`, as long as the layer already has an `atom_swa_kv` view.

Why:

- Packed `fp8_ds_mla` and older FP8+fp32-sidecar layouts must not silently fall
  back to the default `"dense"` layout string. The authoritative runtime
  contract is now the scheduler/model-state metadata object whenever it exists,
  not only mutable side effects on each attention layer.
- Split-only packed KV is a real vLLM-owned allocation mode, not an error
  case. Rebinding the metadata bundle must not assume every active layer has a
  homogeneous unified tensor.

Coverage:

- `test_deepseek_v4_atom_kv_views_prefer_metadata_bundle` verifies the runtime
  resolver prefers the metadata bundle over stale layer attributes for
  compressed tail, scales, and layout.
- `test_deepseek_v4_bind_split_only_unified_kv_bundle` verifies split-only
  packed bundles can be rebound without indexing `unified_kv`.

Verification:

- `python3 -m py_compile vllm/models/deepseek_v4/amd/rocm.py vllm/models/deepseek_v4/amd/model_state.py tests/v1/worker/test_utils.py`
- `pytest -q tests/v1/worker/test_utils.py -k 'split_only_kv_decode or atom_kv_views or bind_split_only or vllm_owned_kv or post_bind'`
- `pytest -q tests/v1/worker/test_utils.py`
- `pytest -q tests/kernels/test_deepseek_v4_fused_compress_contract.py tests/kernels/test_deepseek_v4_compressor_contract.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
- `pytest -q tests/v1/core/test_kv_cache_utils.py -k 'atom_mla_mixed_regular_is_uniform_type or atom_mla_mixed_uniformity or merge_atom_mla_spec or kv_cache_spec_kind or atom_mla_multiple_layers or atom_mla or deepseek_v4_grouping or scheduler_atom'`
- `pytest -q tests/v1/test_kv_cache_spec_registry.py`

Remaining gap:

- This proves the scheduler/model-state KV layout metadata is internally
  consistent through the Python runtime contracts. It does not prove end-to-end
  GSM8K accuracy or deployment throughput for the packed vLLM-owned ATOM KV
  path; that still requires a server run with the target deployment flags.

## 2026-06-20 Unified-KV Bundle Layer Map

Change:

- `DeepseekV4RocmAtomUnifiedKVBuffers` now carries
  `unified_kv_by_layer: dict[int, torch.Tensor]` in addition to the historical
  `unified_kv` tuple.
- `_bind_atom_unified_kv_buffers` and `_resolve_atom_kv_views` use the layer map
  for homogeneous unified tensors instead of deriving tuple position from
  `active_layer_ids`.
- Prefill SWA writes now use the resolved SWA view from `_resolve_atom_kv_views`
  so metadata-derived homogeneous views are still writable.

Why:

- `active_layer_ids` can contain split-only packed or sidecar layers that have
  no homogeneous `atom_unified_kv` tensor. In such a bundle, tuple position is
  not a valid layer lookup. A mixed bundle like layer 0 split-only and layer 1
  homogeneous would otherwise look for layer 1 at tuple index 1 even though the
  only homogeneous tensor is at tuple index 0.

Coverage:

- `test_deepseek_v4_bind_mixed_split_and_homogeneous_bundle` verifies mixed
  split-only plus homogeneous bundles bind by layer id.
- `test_deepseek_v4_atom_kv_views_use_layer_map_for_homogeneous_kv` verifies
  decode/prefill resolution uses the layer map when active IDs and tuple
  positions diverge.
- Existing vLLM-owned packed/sidecar tests verify split-only bundles keep an
  empty layer map; side-allocation tests verify homogeneous allocations populate
  the map.

Verification:

- `python3 -m py_compile vllm/models/deepseek_v4/amd/model_state.py vllm/models/deepseek_v4/amd/rocm.py tests/v1/worker/test_utils.py`
- `pytest -q tests/v1/worker/test_utils.py -k 'atom_kv_views or bind_split_only or bind_mixed_split or vllm_owned_kv or side_allocates or post_bind or split_only_kv_decode'`
- `pytest -q tests/v1/worker/test_utils.py`
- `pytest -q tests/v1/core/test_kv_cache_utils.py -k 'atom_mla_mixed_regular_is_uniform_type or atom_mla_mixed_uniformity or merge_atom_mla_spec or kv_cache_spec_kind or atom_mla_multiple_layers or atom_mla or deepseek_v4_grouping or scheduler_atom'`
- `pytest -q tests/kernels/test_deepseek_v4_fused_compress_contract.py tests/kernels/test_deepseek_v4_compressor_contract.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
- `pytest -q tests/v1/test_kv_cache_spec_registry.py`

Remaining gap:

- This fixes Python-side bundle identity and lookup for mixed layouts. It still
  does not prove the packed vLLM-owned ATOM KV path is end-to-end accurate or
  faster under the target server deployment flags.

## 2026-06-20 No-Eager Runtime Validation And C32 Benchmark

Runtime configuration:

- `bash launchdeepseekgraph.sh` with script defaults:
  `VLLM_USE_V2_MODEL_RUNNER=1`, no `--enforce-eager`, breakable CUDA graph
  enabled, TP8, `--block-size 128`, `--kv-cache-dtype fp8`,
  `VLLM_ROCM_DSV4_ATOM_STATE=1`,
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`,
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`,
  `VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN=1`,
  `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`, and
  `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`.
- `VLLM_ROCM_DSV4_ATOM_MIXED_KV` was not enabled, so the vLLM-owned ATOM
  unified sparse-attention storage intentionally used the homogeneous dense
  BF16 path even though the server-level KV cache dtype was `fp8`.

Accuracy result:

- `bash lmeval.sh` passed GSM8K in graph mode:
  flexible-extract exact_match `0.9530 +/- 0.0058`, strict-match exact_match
  `0.9538 +/- 0.0058`.

KV layout evidence:

- The server logs reported `Using DeepSeek's fp8_ds_mla KV cache format`, but
  the ATOM sparse-attention binding reported
  `layout_counts={'dense': 61}`, `dtype=torch.bfloat16`.
- Source reason: `DeepseekV4Attention.get_kv_cache_spec()` selects
  `DeepseekV4AtomMLAAttentionSpec` for ROCm vLLM-owned ATOM KV. When
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`, it sets `spec_cache_dtype="bf16"` and
  `atom_vllm_compressed_layout="dense"`. Packed `fp8_ds_mla` compressed tails
  are selected only when `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1`.

Benchmark results:

- Invalid crash-isolation run:
  `MAX_NUM_SEQS=32 RESULT_PREFIX=ds-v4-pro-nomtp-atom-runtime-current-noeager-maxseq32-C32 CONCURRENCIES=32 bash benchmarkvllm.sh`
  produced only `completed=160`, `failed=160`; the server log showed
  `Worker proc VllmWorker-2 died unexpectedly`. Treat its throughput as
  invalid.
- Valid deployment-shaped C32 run:
  `RESULT_PREFIX=ds-v4-pro-nomtp-atom-runtime-current-noeager-maxseq256-C32 CONCURRENCIES=32 bash benchmarkvllm.sh`
  completed `320/320` requests with `failed=0`.
  Result file:
  `bench-sparsemla/ds-v4-pro-nomtp-atom-runtime-current-noeager-maxseq256-C32-C32.json`.
  Output throughput `888.30 tok/s`, total throughput `1780.07 tok/s`,
  mean TPOT `35.08 ms`, median TPOT `35.07 ms`.

Comparison:

- Previous best valid BF16-tail run:
  `bench-sparsemla/revert-compressor-aux-nomtp-C32.json`:
  output `926.06 tok/s`, total `1855.74 tok/s`, mean TPOT `33.50 ms`.
- Previous valid current-default run:
  `bench-sparsemla/ds-v4-pro-nomtp-current-default-noeager-C32.json`:
  output `916.28 tok/s`, total `1836.14 tok/s`, mean TPOT `33.87 ms`.
- The new valid run is stable but not the best observed result. It is about
  `4.1%` below the previous best output throughput and about `4.7%` slower in
  mean TPOT.

Analysis:

- The max-seq-32 server cap is not a safe benchmark configuration for this
  path; the default `max_num_seqs=256` completed the same C32 workload cleanly.
- The currently verified runtime path gets correctness from vLLM-owned ATOM
  request state and homogeneous BF16 ATOM unified KV views. It does not yet
  exercise the packed FP8 compressed-tail path that should be needed to get
  closer to the FP8 numbers in `ATOM/recipes/DeepSeek-V4.md`.
- Next performance work should target enabling and validating
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1`, then re-running GSM8K and C32 benchmark.

## 2026-06-20 Packed FP8 Mixed-KV Runtime Validation

Runtime configuration:

- `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1 bash launchdeepseekgraph.sh` with script
  defaults: `VLLM_USE_V2_MODEL_RUNNER=1`, no `--enforce-eager`, breakable CUDA
  graph enabled, TP8, `--block-size 128`, `--kv-cache-dtype fp8`,
  `VLLM_ROCM_DSV4_ATOM_STATE=1`,
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`,
  `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`,
  `VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN=1`,
  `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`, and
  `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`.

KV layout evidence:

- Server startup bound the vLLM-owned packed ATOM KV layout:
  `layout_counts={'fp8_ds_mla': 61}`, `num_blocks=65693`,
  `swa_pages=32768`, `win_with_spec=128`, `head_dim=512`.
- vLLM reported GPU KV cache capacity `131,353` tokens for the packed path,
  compared with `114,218` tokens in the dense BF16 ATOM unified-KV validation.

Accuracy result:

- `bash lmeval.sh` passed GSM8K in graph mode:
  flexible-extract exact_match `0.9545 +/- 0.0057`, strict-match exact_match
  `0.9553 +/- 0.0057`.

Benchmark result:

- Fresh server restart before benchmark, then:
  `RESULT_PREFIX=ds-v4-pro-nomtp-atom-mixedkv-noeager-maxseq256-C32 CONCURRENCIES=32 bash benchmarkvllm.sh`.
- Result file:
  `bench-sparsemla/ds-v4-pro-nomtp-atom-mixedkv-noeager-maxseq256-C32-C32.json`.
- Completed `320/320` requests with `failed=0`.
- Output throughput `808.46 tok/s`, total throughput `1620.08 tok/s`,
  mean TPOT `38.55 ms`, median TPOT `38.50 ms`, mean TTFT `1082.85 ms`.

Comparison:

- Dense BF16 ATOM unified-KV validation:
  `bench-sparsemla/ds-v4-pro-nomtp-atom-runtime-current-noeager-maxseq256-C32-C32.json`
  reached output `888.30 tok/s`, total `1780.07 tok/s`, mean TPOT
  `35.08 ms`.
- Previous best valid run:
  `bench-sparsemla/revert-compressor-aux-nomtp-C32.json` reached output
  `926.06 tok/s`, total `1855.74 tok/s`, mean TPOT `33.50 ms`.
- Packed FP8 mixed-KV is accuracy-correct and increases available KV capacity,
  but it is not a throughput win in this implementation: about `8.99%` lower
  output throughput than the dense BF16 ATOM unified-KV run, and about `12.70%`
  below the previous best valid run.

MHC relevance:

- MHC is relevant to full ATOM model equivalence and end-to-end performance
  because it changes the hidden-state/norm path before attention and MoE.
- MHC is not the structural requirement that makes ATOM attention/compressor
  kernels work with vLLM. Those kernels depend on Q/K/V tensor semantics,
  compressor ordering, request-state rings, block tables, and the KV storage
  contract.
- The current blocking gap for ATOM-style packed FP8 performance is therefore
  not primarily MHC. The measured slowdown points more directly at split
  BF16-SWA plus packed-tail adaptation, packed dequant/read overhead,
  `csa_translate_pack`, metadata preparation, and missing native aiter entry
  points for the exact packed DSV4 sparse attention/compressor contract.

## 2026-06-20 Ordered Split-Load Packed Decode Experiment

Change tested:

- The packed `fp8_ds_mla` split-KV decode wrapper now passes decode positions
  and SWA window size even when fused CSA top-k is disabled.
- `_paged_decode_split_kv_fused_kernel` can derive the compressed-head versus
  SWA-tail boundary for the ordered sequence. It then skips one backing-store
  load on homogeneous tiles:
  compressed-only tiles skip SWA loads, SWA-only tiles skip packed-tail loads,
  and only the boundary tile reads both paths.
- Fused CSA top-k remained disabled. This only changed the packed split-KV
  reader specialization.

Focused validation:

- `python3 -m py_compile vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py vllm/models/deepseek_v4/amd/rocm.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
- `pytest -q tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
  passed: `9 passed`.
- Manual ROCm smoke compared ordered packed split-KV decode against the
  reference path for mixed compressed-plus-SWA input: max diff
  `3.0517578125e-05`, finite output.

Runtime accuracy:

- Fresh graph-mode server:
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1 bash launchdeepseekgraph.sh`.
- Server bound the packed vLLM-owned layout:
  `layout_counts={'fp8_ds_mla': 61}`, `num_blocks=65693`,
  `swa_pages=32768`.
- Unchanged `bash lmeval.sh` passed GSM8K:
  flexible-extract exact_match `0.9545 +/- 0.0057`,
  strict-match exact_match `0.9553 +/- 0.0057`.

C32 benchmark:

- Fresh server restart before benchmark, then:
  `RESULT_PREFIX=ds-v4-pro-nomtp-atom-mixedkv-ordered-load-noeager-maxseq256-C32 CONCURRENCIES=32 bash benchmarkvllm.sh`.
- Result file:
  `bench-sparsemla/ds-v4-pro-nomtp-atom-mixedkv-ordered-load-noeager-maxseq256-C32-C32.json`.
- Completed `320/320` requests with `failed=0`.
- Output throughput `728.23 tok/s`, total throughput `1459.30 tok/s`,
  mean TPOT `42.91 ms`, median TPOT `42.88 ms`, mean TTFT `1085.73 ms`.

Comparison:

- Previous packed mixed-KV path:
  `bench-sparsemla/ds-v4-pro-nomtp-atom-mixedkv-noeager-maxseq256-C32-C32.json`
  reached output `808.46 tok/s`, total `1620.08 tok/s`, mean TPOT
  `38.55 ms`.
- Dense BF16 ATOM unified-KV path:
  `bench-sparsemla/ds-v4-pro-nomtp-atom-runtime-current-noeager-maxseq256-C32-C32.json`
  reached output `888.30 tok/s`, total `1780.07 tok/s`, mean TPOT
  `35.08 ms`.
- Previous best valid run:
  `bench-sparsemla/revert-compressor-aux-nomtp-C32.json` reached output
  `926.06 tok/s`, total `1855.74 tok/s`, mean TPOT `33.50 ms`.

Outcome:

- This ordered split-load specialization is accuracy-correct, but it regresses
  packed mixed-KV throughput by about `9.92%` versus the previous packed run
  and about `21.36%` versus the previous best valid run.
- Runtime default after this measurement: disabled unless the fused CSA top-k
  branch explicitly needs the same position metadata. Do not treat ordered
  split-load as a performance fix. It may reduce some masked memory reads, but
  the extra per-tile control flow and layout handling appear more expensive
  under the deployment workload.
- The remaining packed-path bottlenecks are still likely outside this specific
  branch: packed compressor writer, packed tail dequant/read format,
  `csa_translate_pack`, metadata preparation, MoE/MHC/stream overlap, and the
  absence of a native aiter entry point for the exact packed DSV4 sparse
  attention/compressor contract.

Follow-up guard:

- Added
  `tests/kernels/test_deepseek_v4_atom_dependency_contract.py` to make the
  integration constraint executable. The test verifies that vLLM runtime Python
  files do not import `atom`/`atom.*`, and that the installed `aiter` module
  does not currently export the ATOM sparse attention/state-write API names
  needed to replace the local compatibility path.
- The same guard now checks that the runtime split-KV decode call only passes
  ordered split metadata (`csa_positions`, nonzero `csa_window_size`) when
  fused CSA top-k metadata is present. This keeps the slower ordered split-load
  experiment out of the default packed mixed-KV path.
- It also checks that `DeepseekV4RocmAtomModelState` does not import
  `WorkspaceManager` scratch APIs or call `get_simultaneous`; persistent DSV4
  request rings and unified-KV buffers must stay in `ModelState` or vLLM
  KV-cache allocation.
- It checks the ATOM main compressor sequence in
  `_maybe_atom_main_compressor_forward`: `fused_compress_attn` must appear
  before `update_compressor_states`, preserving read-before-update ordering.
- It checks that the ROCm DSV4 attention runtime still imports/calls the
  ATOM-style attention op surface: `qk_norm_rope_maybe_quant`, `swa_write`,
  `csa_translate_pack`, paged decode, split-KV paged decode, paged prefill, and
  split-KV paged prefill.
- It checks that the compressor runtime still calls the ATOM compressor op
  surface: `fused_compress_attn` and `update_compressor_states`.
- It checks that ATOM feature env lookups remain import-time cached and are not
  repeated from the hot DSV4 attention/compressor/model-state/kernel functions.
- It checks that `launchdeepseekgraph.sh` defaults to the ROCm ATOM FP8 KV path:
  graph mode, block size 128, V2 model runner, ATOM model state, vLLM-owned
  unified KV, mixed KV, compression plans, main compressor, ATOM attention,
  fused Q norm/quant, and `--kv-cache-dtype fp8`.
- Validation command:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
  passed: `12 passed`.
- Workspace-boundary-only validation:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py`
  passed: `4 passed`.
- Dependency/ordering/env-cache validation:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py`
  passed: `9 passed`.
- Compressor ordering validation:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py tests/kernels/test_deepseek_v4_compressor_contract.py tests/kernels/test_deepseek_v4_fused_compress_contract.py`
  passed: `16 passed`.
- Focused ATOM attention/compressor contract validation:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py tests/kernels/test_deepseek_v4_compressor_contract.py tests/kernels/test_deepseek_v4_fused_compress_contract.py`
  passed: `29 passed`.
- Focused vLLM KV-cache/ATOM contract validation:
  `pytest -q tests/v1/worker/test_utils.py tests/v1/core/test_kv_cache_utils.py -k 'atom or deepseek_v4 or reshape_kv_cache_atom or kv_block_zeroer or representative_worker_spec'`
  passed: `45 passed, 62 deselected`.
- Focused vLLM runtime KV reshape validation:
  `pytest -q tests/v1/worker/test_utils.py tests/v1/worker/test_gpu_model_runner.py tests/v1/core/test_kv_cache_utils.py -k 'atom or deepseek_v4 or reshape_kv_cache_atom or gpu_model_runner_reshape_atom or kv_block_zeroer or representative_worker_spec'`
  passed: `47 passed, 97 deselected`.
- Packed split-KV layout validation:
  `pytest -q tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
  passed: `10 passed`. This now includes an explicit ABI check that the
  `fp8_ds_mla` tensor shape `[num_blocks, k_per_block, 584]` is only a byte
  container: each block stores all `k_per_block * 576` token payload bytes
  first, then all `k_per_block * 8` UE8M0 scale bytes. Consumers must not read
  scales from `kv_cache[block, slot, 576:584]`, and tests write the BF16 RoPE
  tail via flat block offsets to match the compressor writer and attention
  readers.
- Producer/consumer packed-layout guard:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
  passed: `22 passed`. The dependency contract now also checks that
  `fused_compress.py`, `paged_decode.py`, and `paged_prefill.py` all use the
  same block-packed `fp8_ds_mla` offsets: token data at `slot * 576`, scale
  data at `PACKED_BLOCK_SIZE * 576 + slot * 8`, and BF16 RoPE tail at
  `token_data_base + 448`.
- Generic worker reshape contract:
  `KVCacheSpec` now exposes `fixed_prefix_size_bytes`,
  `requires_strided_kv_cache_view`, and `inner_block_stride_bytes`. The
  vLLM-owned ATOM spec implements these for the SWA prefix and legacy sidecar
  scale-tail layout, while `gpu_model_runner.py` and
  `worker/gpu/attn_utils.py` consume only the generic spec properties. This
  removes direct `DeepseekV4AtomMLAAttentionSpec` checks from generic worker
  reshape code while preserving packed-prefix and sidecar-stride behavior.
- Generic worker validation:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py tests/v1/test_kv_cache_spec_registry.py tests/v1/worker/test_utils.py tests/v1/worker/test_gpu_model_runner.py -k 'atom or deepseek_v4 or generic_worker_reshape or reshape_kv_cache_atom or gpu_model_runner_reshape_atom or representative_worker_spec or builtin or uniform'`
  passed: `70 passed, 51 deselected`.
- Generic core prefix accounting:
  `_representative_scheduler_spec`, `_representative_worker_spec`, and
  `_scalable_blocks_per_request` now use `KVCacheSpec.fixed_prefix_size_bytes`
  instead of checking `DeepseekV4AtomMLAAttentionSpec` directly.
  `is_kv_cache_spec_uniform` also rejects mixed fixed-prefix/non-prefix specs
  generically instead of keying on the DSV4 ATOM type. The DSV4-specific
  allocator split remains explicit, because it is the ROCm-only layout planner
  that emits one fixed-prefix tensor per ATOM layer.
- Generic core/worker validation:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py tests/v1/test_kv_cache_spec_registry.py tests/v1/worker/test_utils.py tests/v1/worker/test_gpu_model_runner.py tests/v1/core/test_kv_cache_utils.py -k 'atom or deepseek_v4 or generic_worker_reshape or core_prefix_sizing or reshape_kv_cache_atom or gpu_model_runner_reshape_atom or representative_worker_spec or builtin or uniform or scheduler_atom or kv_cache_spec_kind'`
  passed: `96 passed, 103 deselected`.
- Reduced graph-mode runtime smoke after the generic KV spec cleanup:
  `MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024 MAX_MODEL_LEN=2048 GPU_MEMORY_UTILIZATION=0.85 bash launchdeepseekgraph.sh`
  started successfully without `--enforce-eager`, using V2 model runner,
  breakable CUDA graph, `fp8_ds_mla` KV cache, and vLLM-owned packed ATOM KV.
  The log reported `layout_counts={'fp8_ds_mla': 61}`,
  `num_blocks=59466`, `swa_pages=512`, and graph capture completed on all TP
  workers. A `/v1/completions` request returned HTTP 200. Port 8000 was closed
  after shutdown.
- Smoke caveat:
  The first request still triggered Triton JIT warnings for metadata,
  compressor-state update, SWA write, GEMM, indexer cache, and MQA logits
  kernels. This does not invalidate correctness, but it means first-request
  latency is not warmed for all shapes in the reduced smoke configuration.
- Full graph-mode accuracy after generic KV spec cleanup:
  `bash launchdeepseekgraph.sh` with default graph-mode settings
  (`MAX_NUM_SEQS=256`, block size 128, V2 model runner, breakable CUDA graph,
  `--kv-cache-dtype fp8`, vLLM-owned packed ATOM KV) plus unchanged
  `bash lmeval.sh` passed GSM8K. Result file:
  `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-20T05-06-09.229612.json`.
  Scores: flexible exact match `0.9537528430629265 +/- 0.005784991662691852`,
  strict exact match `0.954510993176649 +/- 0.005739657656722221`.
- Fresh C32 benchmark after the accuracy pass:
  Server was restarted with `MAX_NUM_SEQS=32 bash launchdeepseekgraph.sh` to
  clear KV and prefix-cache state, then run with
  `RESULT_PREFIX=ds-v4-pro-atom-mixedkv-generic-spec-noeager-maxseq32-C32 CONCURRENCIES=32 bash benchmarkvllm.sh`.
  Result file:
  `/app/atomdsv4/bench-sparsemla/ds-v4-pro-atom-mixedkv-generic-spec-noeager-maxseq32-C32-C32.json`.
  Completed `320`, failed `0`, output throughput `810.6296220012367 tok/s`,
  total throughput `1624.4257659634159 tok/s`, mean TPOT
  `38.451398510847426 ms`, mean TTFT `1078.4290040144697 ms`, mean E2EL
  `40414.209680611384 ms`.
- MHC relevance note:
  MHC is not required for the ATOM attention/compressor memory contract. The
  ATOM attention and compressor path depends on q/k/v preparation, SWA state
  writes, compressor state ordering, packed KV layout, metadata translation,
  and the paged decode/prefill readers. MHC changes the hidden-state production
  before those paths and matters for full ATOM model-op parity/performance, but
  it does not by itself define the KV cache layout or compressor state update
  semantics.

## ATOM Kernel Coverage Audit

Current conclusion: we have enough components to run a validated
ATOM-shaped ROCm DSV4 path inside vLLM's scheduler, but not enough to claim
the benefit of all ATOM kernels. The current path is intentionally independent
of the ATOM Python package and has passed graph-mode accuracy/perf, but several
ATOM ops are represented by local vLLM ports or existing vLLM wrappers rather
than the exact ATOM runtime sequence.

Evidence from ATOM `atom/models/deepseek_v4.py`:

- ATOM imports/uses `qk_norm_rope_maybe_quant`, `swa_write`,
  `fused_compress_attn`, `update_compressor_states`, `csa_translate_pack`,
  `sparse_attn_v4_paged_decode`, `sparse_attn_v4_paged_prefill`,
  `inverse_rope_inplace`, and `scale_indexer_weights` from
  `atom.model_ops.v4_kernels`.
- ATOM indexer scoring uses `scale_indexer_weights`, `fp8_mqa_logits`,
  `deepgemm_fp8_paged_mqa_logits`, `top_k_per_row_prefill`, and
  `top_k_per_row_decode`.
- ATOM attention dispatch is wrapped by
  `torch.ops.aiter.v4_attention_with_output`, then internally runs the sequence:
  compressors first, q/k norm + RoPE, decode SWA write before attention,
  indexer top-k and CSA translation, paged sparse decode/prefill, prefill SWA
  write after attention, inverse RoPE on output, then output projection.
- ATOM has optional auxiliary stream overlap for main/indexer compressors and
  `maybe_dual_stream_forward`.
- ATOM MHC/HC uses aiter `mhc_pre`, `mhc_post`, and optionally
  `mhc_fused_post_pre`; those affect full block parity/perf but not the
  attention/compressor KV contract.

Current vLLM coverage:

- Persistent request state: covered. ROCm DSV4 uses model-specific
  `DeepseekV4RocmAtomModelState` for SWA rings, compressor state rings,
  compress plans, decode/prefill index buffers, and vLLM-owned packed KV
  binding. This avoids GPU worker changes for request-lived ATOM state.
- KV cache spec/allocation/binding: covered for the current deployment shape.
  `DeepseekV4AtomMLAAttentionSpec` adds a fixed SWA prefix plus compressed
  tail metadata. Generic worker/core reshape paths consume
  `KVCacheSpec.fixed_prefix_size_bytes`,
  `requires_strided_kv_cache_view`, and `inner_block_stride_bytes`. The ROCm
  path binds one vLLM-owned packed `fp8_ds_mla` KV tensor per ATOM layer while
  CUDA keeps the existing specs.
- Fused q/k norm + RoPE: covered by the local vLLM
  `qk_norm_rope_maybe_quant` port, with optional aiter/flydsl dispatch when
  available. Default launch uses this path with `quant_q=False` and
  `quant_k=False`, matching the ATOM attention call shape.
- Main compressor: covered for the validated default path.
  vLLM calls local `fused_compress_attn` and `update_compressor_states` in the
  ATOM read-before-update order. The packed `fp8_ds_mla` tail is supported and
  validated. The implementation can dispatch to aiter/flydsl for supported
  dense shapes, but the packed mixed-KV deployment still uses the local Triton
  path where the aiter/flydsl ABI does not match the vLLM-owned packed tail.
- SWA write: covered. Decode writes before sparse attention; prefill writes
  after sparse attention so chunked-prefill prefix reads see prior ring state.
- CSA translate/pack: covered by local `csa_translate_pack`; decode can also
  fuse CSA top-k translation into split-KV decode when the corresponding flag
  is enabled, but the default path keeps the explicit translation.
- Sparse paged decode/prefill: covered by local vLLM ATOM-shaped kernels over
  split SWA/compressed KV views, including packed `fp8_ds_mla` dequant. This
  is not the exact ATOM sparse attention ABI: ATOM consumes one homogeneous
  unified KV pool; the current vLLM deployment uses split views over
  vLLM-owned storage because the packed FP8 tail and BF16 SWA prefix are not a
  single homogeneous tensor.
- Indexer scoring/top-k: partially covered. Existing vLLM ROCm sparse indexer
  wrappers can use aiter `fp8_mqa_logits`, `deepgemm_fp8_paged_mqa_logits`,
  and aiter top-k kernels. The ATOM `scale_indexer_weights` formula is now
  available as local vLLM helper
  `vllm.models.deepseek_v4.common.ops.scale_indexer_weights` and is covered by
  a direct formula test. There is also an opt-in ROCm preview branch,
  `VLLM_ROCM_DSV4_ATOM_INDEXER_SEQUENCE=1`, that runs explicit RoPE,
  `per_token_group_quant_fp8`, and `scale_indexer_weights` before the existing
  indexer scoring path. Reduced graph-mode smoke with
  `MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024 MAX_MODEL_LEN=2048
  GPU_MEMORY_UTILIZATION=0.85 VLLM_ROCM_DSV4_ATOM_INDEXER_SEQUENCE=1
  bash launchdeepseekgraph.sh` completed graph capture and returned HTTP 200
  for a `/v1/completions` request. The default runtime is still the vLLM fused
  indexer flow, and even the opt-in branch is not yet the full ATOM
  `Indexer.forward_batched` sequence over the exact ATOM compressed indexer
  cache contract.
- Installed aiter indexer dispatcher surface:
  `torch.ops.aiter.indexer_score_topk` is not exposed by the installed
  `aiter==0.1.15.post1`. vLLM now has a guarded local fallback dispatcher under
  `VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH=1`, so the model can call the same op
  name without taking an ATOM package dependency. This is default-off until
  graph-mode lmeval and C32 benchmark results prove it is useful.
- Inverse RoPE/output projection: partially covered. vLLM uses
  `rocm_inv_rope_einsum` to combine inverse-RoPE behavior with output
  projection work. This is not a direct call to ATOM `inverse_rope_inplace`.
  An opt-in ROCm preview path,
  `VLLM_ROCM_DSV4_ATOM_SEPARATE_INVERSE_ROPE=1`, now runs a local
  `inverse_rope_inplace` primitive before the grouped `wo_a` projection so the
  ATOM output ordering can be tested without changing the validated default
  fused inverse-RoPE/einsum path.
- MoE and fused activation: covered by vLLM's aiter MoE path and the optional
  aiter `fused_clamp_act_mul` in `amd/model.py`, but this is orthogonal to
  the attention/compressor KV integration.
- MHC/HC: available but off by default in `launchdeepseekgraph.sh`
  (`VLLM_ROCM_DSV4_USE_AITER_MHC=0`,
  `VLLM_ROCM_DSV4_USE_AITER_HC_HEAD=0`). vLLM has aiter-backed `mhc_pre` and
  `mhc_post` wrappers, but `mhc_fused_post_pre` still falls back to tilelang in
  the generic MHC op stack. This is a full-model parity/performance gap, not a
  blocker for attention/compressor correctness.
- Installed aiter MHC surface:
  `aiter==0.1.15.post1` exposes `mhc_pre` and `mhc_post` both at the top level
  and under `aiter.ops.mhc`, but it does not expose `mhc_fused_post_pre` in
  either location. It also does not expose a top-level `hc_head`; the vLLM
  HC/head wrapper maps the head reduction through `aiter.ops.mhc.mhc_pre(...,
  sinkhorn_repeat=0)`. Because the user constraint is not to modify `aiter`,
  exact ATOM `mhc_fused_post_pre` parity is not available from this installed
  package. Enabling `VLLM_ROCM_DSV4_USE_AITER_MHC=1` uses aiter pre/post, but
  vLLM intentionally switches to the unfused post/pre model path when aiter MHC
  is enabled, since no aiter fused post-pre op exists.
- Aiter MHC/HC validation result:
  `MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024 MAX_MODEL_LEN=2048
  GPU_MEMORY_UTILIZATION=0.85 VLLM_ROCM_DSV4_USE_AITER_MHC=1
  VLLM_ROCM_DSV4_USE_AITER_HC_HEAD=1 bash launchdeepseekgraph.sh` loaded the
  model, bound vLLM-owned packed KV, captured graphs, and returned HTTP 200 for
  a `/v1/completions` smoke request. The unchanged full `lmeval.sh` accuracy
  gate then failed badly with the same MHC/HC flags at `MAX_NUM_SEQS=256`.
  Result file:
  `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-20T07-20-14.170862.json`.
  GSM8K flexible exact match was `0.12585291887793784 +/-
  0.009136212598406293`; strict exact match was `0.11751326762699014 +/-
  0.008870331256489955`. Because this misses the `0.95 +/- 0.01` target by a
  large margin, do not run C32 benchmark for this configuration and keep both
  `VLLM_ROCM_DSV4_USE_AITER_MHC=0` and
  `VLLM_ROCM_DSV4_USE_AITER_HC_HEAD=0` as the deployment defaults.
- Auxiliary streams: not covered after the revert. Current validated runs are
  graph mode with breakable CUDA graph and no ATOM compressor auxiliary stream
  overlap.

Remaining work to claim "all ATOM kernel benefit":

- Replace or extend the split-KV sparse attention wrapper with a native
  ROCm/aiter-compatible DSV4 sparse attention ABI that consumes the final
  vLLM-owned layout without extra metadata conversion, or change the vLLM KV
  spec/allocation so the ATOM sparse attention ABI can be used directly.
- Port the full ATOM `Indexer.forward_batched` sequence instead of relying on
  the existing vLLM sparse indexer path. The `scale_indexer_weights` primitive
  itself is now locally available, tested, and optionally callable from
  `DeepseekV4Indexer.forward`, but the remaining gap is the exact sequence
  around FP8 MQA logits/deepgemm dispatch, top-k contracts, and the ATOM
  indexer compressed cache.
- Decide whether inverse RoPE should stay fused with vLLM's output projection
  (`rocm_inv_rope_einsum`) or be changed to the ATOM `inverse_rope_inplace`
  ordering. The default choice is accurate in tested runs and remains the fast
  deployment path. The opt-in separate inverse-RoPE path is wired and has
  graph-mode smoke, GSM8K, and C32 benchmark evidence; it is accuracy-safe but
  slower than the fused default.
- Reintroduce auxiliary stream overlap only after a graph-mode-safe lifetime
  and dependency model is defined for vLLM's scheduler/model runner.
- Enable and validate aiter MHC/HC only as a separate full-model performance
  slice. It should not be treated as a prerequisite for attention/compressor
  kernel correctness.

Validated opt-in indexer sequence result:

- Configuration:
  `VLLM_USE_V2_MODEL_RUNNER=1 VLLM_ROCM_DSV4_ATOM_INDEXER_SEQUENCE=1`,
  graph mode, no `--enforce-eager`, default `launchdeepseekgraph.sh` for
  accuracy and `MAX_NUM_SEQS=32` for the C32 benchmark.
- Accuracy passed the GSM8K gate with unchanged `lmeval.sh`:
  `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-20T05-46-44.321788.json`
  reported flexible exact match `0.9507202426080363 ± 0.005962150655812466`
  and strict exact match `0.9514783927217589 ± 0.005918468618921092`.
- C32 benchmark after a fresh server restart:
  `/app/atomdsv4/bench-sparsemla/ds-v4-pro-atom-indexer-sequence-noeager-maxseq32-C32-C32.json`
  completed `320/320` requests with `0` failures, output throughput
  `803.1676751079132 tok/s`, total throughput `1609.4727239467165 tok/s`,
  mean TPOT `38.80958584313819 ms`, and mean TTFT
  `1087.5560981352464 ms`.
- Comparison against the prior default fused indexer C32 run:
  `/app/atomdsv4/bench-sparsemla/ds-v4-pro-atom-mixedkv-generic-spec-noeager-maxseq32-C32-C32.json`
  had output throughput `810.6296220012367 tok/s`, total throughput
  `1624.4257659634159 tok/s`, mean TPOT `38.451398510847426 ms`, and mean
  TTFT `1078.4290040144697 ms`.
- Conclusion: the explicit ATOM-style indexer q/scale sequence is
  accuracy-safe in graph mode, but it is not faster in the current vLLM
  scheduler integration. The default fused vLLM indexer q/RoPE/quant path
  should remain the deployment default while the larger ATOM indexer cache and
  sparse attention ABI gaps are investigated.

Validated separate inverse-RoPE wiring:

- Added local ROCm DSV4 kernel
  `vllm.models.deepseek_v4.amd.v4_kernels.inverse_rope_inplace`, matching the
  ATOM operation boundary of mutating the RoPE slice of the attention output
  before grouped output projection.
- Added opt-in model path
  `VLLM_ROCM_DSV4_ATOM_SEPARATE_INVERSE_ROPE=1`. The default remains
  `rocm_inv_rope_einsum`, which fuses inverse RoPE with the grouped `wo_a`
  input preparation and is the currently validated path.
- Validation run:
  `python3 -m py_compile vllm/envs.py vllm/models/deepseek_v4/amd/rocm.py
  vllm/models/deepseek_v4/amd/v4_kernels/__init__.py
  vllm/models/deepseek_v4/amd/v4_kernels/inverse_rope.py
  tests/kernels/test_deepseek_v4_atom_dependency_contract.py` passed.
- Env validation run:
  `VLLM_ROCM_DSV4_ATOM_SEPARATE_INVERSE_ROPE=1 python3 -c
  "import vllm.envs as envs; print(envs.validate_environ(hard_fail=True))"`
  accepted the new flag and exposed it as true.
- Contract validation run:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py`
  passed `18` tests.
- Reduced graph-mode smoke:
  `MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024 MAX_MODEL_LEN=2048
  GPU_MEMORY_UTILIZATION=0.85
  VLLM_ROCM_DSV4_ATOM_SEPARATE_INVERSE_ROPE=1 bash launchdeepseekgraph.sh`
  completed graph capture, bound vLLM-owned packed `fp8_ds_mla` KV, and a
  `/v1/completions` request returned HTTP 200. Runtime JIT logs included
  `_inverse_rope_gptj_inplace_kernel`, confirming the opt-in path was
  exercised.
- Accuracy passed the GSM8K gate with unchanged `lmeval.sh`:
  `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-20T06-15-34.945055.json`
  reported flexible exact match `0.9514783927217589 ± 0.005918468618921092`
  and strict exact match `0.9522365428354814 ± 0.005874387536229304`.
- C32 benchmark after a fresh server restart:
  `/app/atomdsv4/bench-sparsemla/ds-v4-pro-atom-separate-invrope-noeager-maxseq32-C32-C32.json`
  completed `320/320` requests with `0` failures, output throughput
  `802.3352535181467 tok/s`, total throughput `1607.8046291203486 tok/s`,
  mean TPOT `38.89213702041712 ms`, and mean TTFT
  `1045.898510797997 ms`.
- Comparison against the current default fused inverse-RoPE/einsum C32 run:
  `/app/atomdsv4/bench-sparsemla/ds-v4-pro-atom-mixedkv-generic-spec-noeager-maxseq32-C32-C32.json`
  had output throughput `810.6296220012367 tok/s`, total throughput
  `1624.4257659634159 tok/s`, mean TPOT `38.451398510847426 ms`, and mean
  TTFT `1078.4290040144697 ms`.
- Conclusion: the ATOM-style separate inverse-RoPE ordering is graph-mode and
  accuracy safe, but it is not faster in vLLM. Keep the fused
  `rocm_inv_rope_einsum` path as the deployment default and preserve
  `VLLM_ROCM_DSV4_ATOM_SEPARATE_INVERSE_ROPE=1` as a parity/debug switch.

ATOM-style indexer dispatch preview:

- ATOM registers `torch.ops.aiter.indexer_score_topk(q_fp8, weights, prefix,
  topk)` from its Python package and dispatches through
  `static_forward_context[prefix]`. Installed `aiter==0.1.15.post1` does not
  export that op, and vLLM must not depend on ATOM as a package.
- vLLM now has a guarded local fallback registration in
  `vllm.models.deepseek_v4.attention`: if `aiter::indexer_score_topk` is
  missing, a `torch.library.Library("aiter", "FRAGMENT")` op is registered and
  dispatches to the owning `DeepseekV4Indexer` from a vLLM-local prefix
  registry.
- The path is opt-in with `VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH=1`. When
  enabled, `DeepseekV4Indexer.forward` calls the same op name as ATOM after the
  compressor/indexer-cache write is joined. The method first tries the existing
  ATOM-style decode fast path over `ModelState`; otherwise it falls back to the
  vLLM sparse indexer op with `k=None` and `skip_k_cache_insert=True`, which is
  valid because the compressor has already written K into the indexer cache.
- This is a parity step, not a speedup. It preserves the validated default path
  because the full accuracy/performance run below is accuracy-safe but does not
  beat the current default C32 throughput.
- Reduced graph-mode smoke:
  `MAX_NUM_SEQS=4 MAX_NUM_BATCHED_TOKENS=1024 MAX_MODEL_LEN=2048
  GPU_MEMORY_UTILIZATION=0.85 VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH=1
  bash launchdeepseekgraph.sh` completed model load, bound vLLM-owned packed
  `fp8_ds_mla` KV, captured piecewise and full CUDA graphs, and a
  `/v1/completions` request returned HTTP 200. First-shape inference emitted
  expected Triton JIT warnings for metadata/compressor/indexer kernels; no
  dispatch error was observed.
- Full accuracy validation with unchanged `lmeval.sh`:
  `MAX_NUM_SEQS=256 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192
  GPU_MEMORY_UTILIZATION=0.9 VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH=1
  bash launchdeepseekgraph.sh`.
  Result file:
  `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-20T06-49-09.352132.json`.
  GSM8K flexible exact match was `0.954510993176649 ±
  0.005739657656722219`; strict exact match was `0.9552691432903715 ±
  0.005693886131407039`.
- Fresh-server C32 benchmark:
  `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=8192 MAX_MODEL_LEN=8192
  GPU_MEMORY_UTILIZATION=0.9 VLLM_ROCM_DSV4_ATOM_INDEXER_DISPATCH=1
  bash launchdeepseekgraph.sh`, then `RESULT_PREFIX=ds-v4-pro-atom-indexer-dispatch-noeager-maxseq32-C32
  CONCURRENCIES=32 bash benchmarkvllm.sh`.
  Result file:
  `/app/atomdsv4/bench-sparsemla/ds-v4-pro-atom-indexer-dispatch-noeager-maxseq32-C32-C32.json`.
  Completed `320/320` requests with `0` failures, output throughput
  `807.8246593484575 tok/s`, total throughput `1618.804883772495 tok/s`,
  mean TPOT `38.663176957417484 ms`, mean TTFT
  `1002.0076391723704 ms`.
- Comparison against the current default C32 run
  `/app/atomdsv4/bench-sparsemla/ds-v4-pro-atom-mixedkv-generic-spec-noeager-maxseq32-C32-C32.json`:
  default output throughput was `810.6296220012367 tok/s`, total throughput
  `1624.4257659634159 tok/s`, mean TPOT `38.451398510847426 ms`, and mean
  TTFT `1078.4290040144697 ms`. The dispatcher path slightly lowers TTFT in
  this run but regresses output throughput and TPOT, so keep it default-off.

2026-06-20 contract validation checkpoint:

- Command:
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py
  tests/kernels/attention/test_deepseek_v4_split_kv_contract.py
  tests/kernels/test_deepseek_v4_compressor_contract.py`
- Result: `33 passed`.
- Coverage relevant to the ATOM-benefit question: vLLM runtime files do not
  import `atom`/`atom.*`; packed `fp8_ds_mla` writer/reader geometry is shared;
  read-before-update compressor ordering is enforced; launch defaults select
  graph-mode ROCm ATOM FP8 KV; and DeepSeek V4 ATOM env reads are import-time
  cached rather than performed inside the hot forward path.
- The remaining validated gap is not an env lookup issue. It is still the
  native packed DSV4 sparse attention/compressor ABI and metadata contract:
  vLLM can run the ATOM-shaped op sequence, but the deployed packed-KV path
 still pays split-layout adaptation, CSA translate/pack, and scheduler
 metadata conversion that ATOM's native layout avoids.

2026-06-20 weight-swizzle checkpoint:

- The gated raw AITER MHC-pre experiment
  `VLLM_ROCM_DSV4_USE_AITER_MHC_BIG_FUSE=1
  VLLM_ROCM_DSV4_AITER_MHC_EVEN_ROW_WORKAROUND=1
  VLLM_ROCM_DSV4_AITER_MHC_BIG_FUSE_MIN_TOKENS=256` did not pass GSM8K:
  strict exact match was `0.9265`, flexible exact match was `0.9257`.
- MHC `hc_*_fn` must not be offline-swizzled for the installed
  `aiter==0.1.15.post1` kernel. The AITER GEMM source indexes global `fn`
  with `K ^ row_mask`, writes to LDS at the unmasked vector column, and later
  reads LDS with the same xor mask. With raw fp32 `[mix_hc, hc * hidden]`
  weights, `mhc_pre_gemm_sqrsum` matched a torch GEMM at max error
  `3.1e-7` in the local kernel test. Pre-swizzling `fn` by the apparent xor
  pattern double-applied the layout transform and produced max GEMM errors of
  about `1.35` to `1.79`.
- ATOM's modeling file also allocates `hc_attn_fn` and `hc_ffn_fn` as plain
  fp32 `[mix_hc, hc * dim]` parameters and calls `aiter.mhc_pre` directly. Its
  swizzle-related loader warning is for `wo_a` FP8 linear post-load handling,
  not for MHC `fn`.
- The active DeepSeek-V4-Pro server path selected
  `AITER_MXFP4_BF16` for MXFP4 MoE (`--moe-backend aiter`). That path already
  performs the relevant AITER CK weight preparation during
  `process_weights_after_loading`: de-interleave `w13`, view `w13/w2` as
  native FP4, then call `shuffle_weight_a16w4` and `shuffle_scale_a16w4` for
  both expert matrices. Do not add a second MoE swizzle without a failing
  kernel-level reference test.
- Current conclusion: the observed raw MHC-pre accuracy failure is not fixed by
  weight swizzling. Treat it as an AITER big-fuse/post-processing kernel or ABI
  contract issue; keep the accurate hybrid `aiter.mhc_pre_gemm_sqrsum` plus
  vLLM TileLang fuse path as the safe MHC-pre path until a kernel-side fix or a
  faster exact fuse replacement is available.

2026-06-20 follow-up after the `attn4/ffn4` raw MHC-pre run:

- Accuracy failed again with a narrower raw AITER MHC-pre gate:
  `VLLM_ROCM_DSV4_AITER_MHC_PRE_ATTN_MAX_LAYER=4` and
  `VLLM_ROCM_DSV4_AITER_MHC_PRE_FFN_MAX_LAYER=4`, plus the even-row workaround,
  produced GSM8K flexible exact match `0.9083` and strict exact match `0.9090`.
- A direct AITER op-test check on the installed `aiter==0.1.15.post1` source
  also fails for the DSV4 MHC shape:
  `PYTHONPATH=/app/atomdsv4/aiter python aiter/op_tests/test_mhc.py -m 64 -n 1280 -d bf16`.
  In that run, `mhc_pre` failed `post_mix`, `comb_mix`, and `layer_input`, while
  `mhc_post` passed. The `mhc_pre_fuse_rmsnorm` variant also failed. This
  reproduces outside vLLM, so the failure is not explained by vLLM scheduler,
  KV-cache, or loader integration.
- Generic AITER `shuffle_weight(fn, layout=(16, 16))` is not applicable to MHC
  `fn`: DSV4 MHC uses `mix_hc = 24` rows, and the helper asserts row divisibility
  by `16`. Padding and swizzling would also change the row-major
  `residual_flat @ fn.T` contract used by ATOM and by AITER's own Python
  reference.
- Practical implication: do not add an offline MHC `hc_attn_fn`/`hc_ffn_fn`
  swizzle in vLLM. The only weight swizzle that should be active for this model
  is the existing AITER MXFP4 MoE post-load path:
  `shuffle_weight_a16w4`/`shuffle_scale_a16w4` after `w13` de-interleave.

2026-06-20 async V2 fused-norm MHC checkpoint:

- Async scheduling remains mandatory for candidate V2 runs. Old saved runs that
  omit `async_scheduling=True` in server args must not be counted as valid
  async V2 performance evidence.
- Full AITER MHC fused-norm pre
  (`VLLM_ROCM_DSV4_USE_AITER_MHC_FUSE_NORM=1` with both attn/ffn pre enabled)
  starts in graph mode but crashes under async scheduling on the first
  benchmark main batch with HIP illegal memory access reported from
  `WorkerAsyncOutputCopy`. The actual failing kernel is asynchronous and not
  proven by that stack, but this is a real serving failure: all benchmark
  requests return HTTP 500 after the worker dies.
- Added `VLLM_ROCM_DSV4_AITER_MHC_FUSE_NORM_MAX_TOKENS`, default `64`, so fused
  MHC norm is restricted to decode-sized batches. Larger prefill/mixed batches
  fall back to the stable separate-norm path.
- Stability validation with the cap:
  `v2-async-fusenorm-cap64-smoke-o128-20260620-C32.json` completed `320/320`
  requests, `0` failures, output throughput `688.30 tok/s` for the short
  `1024 -> 128` smoke workload.
- Full C32 result with the cap:
  `v2-async-fusenorm-cap64-full-20260620-C32.json` completed `320/320`
  requests, `0` failures, output throughput `871.28 tok/s`, total throughput
  `1745.96 tok/s`, mean TPOT `35.47 ms`, mean TTFT `1322.35 ms`.
- Conclusion: the cap makes fused-norm MHC async-safe but is not a performance
  candidate. The best valid async V2 result in the current saved set is still
  `v2-async-mhc-bigfuse-20260620-C32.json` at `937.62 tok/s`, which remains
  below the V1 native baseline
  `native-v1-noatom-b256-mbt32768-20260620-C32.json` at `970.59 tok/s`.

2026-06-21 async V2 direct AITER MHC legacy-op checkpoint:

- The run kept the required V2 deployment shape:
  `VLLM_USE_V2_MODEL_RUNNER=1`, `ASYNC_SCHEDULING=1`, graph mode, no
  `--enforce-eager`. The server log confirms both `Asynchronous scheduling is
  enabled` and `Using V2 Model Runner`.
- C32 benchmark with
  `VLLM_ROCM_DSV4_USE_AITER_MHC=1
  VLLM_ROCM_DSV4_USE_AITER_MHC_LEGACY_OPS=1
  MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=32768 BLOCK_SIZE=128` completed
  `320/320` requests with `0` failures:
  `v2-async-mhc-legacyops-20260621-C32.json` reported output throughput
  `1053.93 tok/s`, total throughput `2111.98 tok/s`, mean TPOT `28.78 ms`,
  and mean TTFT `1652.72 ms`.
- Accuracy with unchanged `lmeval.sh` on the same legacy-op path failed badly:
  `results_2026-06-21T00-09-02.560428.json` reported GSM8K flexible exact
  match `0.1433` and strict exact match `0.1372`.
- Conclusion: async scheduling is not the failure mode here. The server was
  stable under async V2, but direct AITER legacy MHC ops are numerically invalid
  for this integration. Do not count the `1053.93 tok/s` run as a valid
  candidate and do not use `VLLM_ROCM_DSV4_USE_AITER_MHC_LEGACY_OPS=1` for
  accuracy or deployment.

2026-06-21 async V2 safe-path benchmark control:

- Re-ran the accuracy-passing safe MHC path with the deployment benchmark shape:
  `VLLM_USE_V2_MODEL_RUNNER=1`, `ASYNC_SCHEDULING=1`, graph mode,
  `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=32768`, `BLOCK_SIZE=128`,
  `VLLM_ROCM_DSV4_USE_AITER_MHC=1`, and no legacy MHC ops.
- Server log confirmed `Asynchronous scheduling is enabled`,
  `Using V2 Model Runner`, graph capture completion, and API startup. No server
  errors were logged during the benchmark.
- C32 result:
  `v2-async-mhc-safe-mbt32768-20260621-C32.json` completed `320/320` requests
  with `0` failures, output throughput `928.46 tok/s`, total throughput
  `1860.55 tok/s`, mean TPOT `33.00 ms`, and mean TTFT `1530.07 ms`.
- This remains below the V1 native baseline
  `native-v1-noatom-b256-mbt32768-20260620-C32.json` at `970.59 tok/s`, so
  increasing the V2 benchmark server to `MAX_NUM_BATCHED_TOKENS=32768` is not
  enough to close the gap.

2026-06-21 split-decode workspace experiment:

- Tested the existing split partial allocation switch with
  `VLLM_ROCM_DSV4_ATOM_DECODE_SPLIT_WORKSPACE=workspace` under the same async
  V2 graph-mode C32 setup as the safe-path control.
- The server started cleanly, graph capture completed, and no workspace lock or
  worker errors were logged.
- C32 result:
  `v2-async-mhc-safe-workspace-20260621-C32.json` completed `320/320` requests
  with `0` failures, output throughput `926.75 tok/s`, total throughput
  `1857.12 tok/s`, mean TPOT `33.20 ms`, and mean TTFT `1393.20 ms`.
- This is slightly slower than the default `torch_empty` safe control
  `v2-async-mhc-safe-mbt32768-20260621-C32.json` at `928.46 tok/s`. Keep
  `VLLM_ROCM_DSV4_ATOM_DECODE_SPLIT_WORKSPACE=torch_empty` as the default.

2026-06-21 unified decode split-count override:

- Tested `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=32` with the same safe async V2
  graph-mode C32 setup. The default heuristic uses `kv_splits=16` for the C32
  graph shape, so this tested whether more split-K parallelism can recover the
  V1 gap.
- The server started and graph capture completed, but the benchmark failed
  after the first completed wave: `v2-async-mhc-safe-kvsplit32-20260621-C32.json`
  completed `32/320` requests and failed `288/320`.
- Server log showed HIP illegal memory access from process-group watchdogs and
  EngineCore death. Do not use `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=32`.
- Together with the previous `kv_splits=8` crash, this leaves the built-in
 split heuristic as the only stable split-count setting found so far.

2026-06-21 async V2 mixed packed-KV split-decode checkpoint:

- Kept the required deployment shape throughout this pass:
  `VLLM_USE_V2_MODEL_RUNNER=1`, `ASYNC_SCHEDULING=1`, graph mode,
  no `--enforce-eager`, `MAX_NUM_SEQS=32` for benchmark, and
  `MAX_NUM_SEQS=64` for lmeval.
- The mixed packed-KV path was enabled with
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1`; server logs confirmed
  `layout_counts={'fp8_ds_mla': 61}`.
- Initial split-KV split-K implementation crashed under `1024 -> 128` smoke
  after `64/320` successful requests with HIP illegal memory access. Root
  cause was the split writer not mirroring the single-pass kernel's ordered
  compressed-head/SWA-tail tile guards. Passing decode `positions` into the
  split writer and adding the same ordered split boundary made the smoke
  stable:
  `v2-async-mixedkv-splitk-ordered-smoke-o128-20260621-C32.json` completed
  `320/320`, output throughput `676.24 tok/s`, mean TPOT `34.52 ms`.
- Unchanged `lmeval.sh` then passed on the ordered split-KV path:
  `results_2026-06-21T01-03-29.181268.json` reported GSM8K flexible exact
  match `0.9530` and strict exact match `0.9538`.
- Full `1024 -> 1024` benchmark with decode-index reuse enabled still crashed
  after warmup/main transition with HIP illegal memory access:
  `v2-async-mixedkv-splitk-ordered-fullbench-20260621-C32.json` completed
  `0/320` and failed `320/320`.
- Disabling `VLLM_ROCM_DSV4_ATOM_DECODE_INDEX_REUSE` by default made the full
  benchmark stable:
  `v2-async-mixedkv-splitk-ordered-noidxreuse-fullbench-20260621-C32.json`
  completed `320/320`, output throughput `864.99 tok/s`, total throughput
  `1733.35 tok/s`, mean TPOT `35.40 ms`, mean TTFT `1666.83 ms`.
- A narrower reuse policy that keeps common decode-index reuse but disables
  HCA-head reuse for the packed `fp8_ds_mla` split-KV layout was also stable:
  `v2-async-mixedkv-splitk-commonreuse-hcano-fullbench-20260621-C32.json`
  completed `320/320`, output throughput `861.61 tok/s`, total throughput
  `1726.58 tok/s`, mean TPOT `35.55 ms`, mean TTFT `1662.88 ms`. This is
  slightly slower than full decode-index reuse disabled, so it should be kept
  only as a safety guard for packed split-KV rather than treated as a
  performance win.
- Conclusion: async scheduling is not the blocker; this path runs under V2
  async and passes accuracy. However, packed-KV split decode is not a valid
  performance candidate yet. Reuse-on is unsafe for the full benchmark, and
  reuse-off is slower than both the safe BF16 async V2 control (`928.46 tok/s`)
  and the V1 native baseline (`970.59 tok/s`). The next useful optimization is
  to make decode-index reuse safe within the current forward, or replace the
  common-index rewrite with a cheaper graph-safe metadata path.

2026-06-21 async V2 metadata H2D skip rejection:

- Tried caching/skipping pure-decode H2D copies for small ATOM metadata buffers
  (`state_slot_mapping`, `batch_id_per_token`, and committed CSA/HCA counts).
  The O128 smoke completed `320/320` and reported
  `723.89 tok/s` in
  `v2-async-safe-h2dcache-o128-20260621-C32.json`, but the full C32 run
  `v2-async-safe-h2dcache-explore-20260621-C32.json` completed only `32/320`
  requests before EngineCore death.
- Retried an even narrower version that only skipped the pure-decode
  `chunk_start_per_seq` GPU copy. The fresh full C32 run
  `v2-async-safe-chunkcopy-20260621-C32.json` also completed only `32/320`
  requests before EngineCore death.
- Both changes were reverted. Under async scheduling, these metadata tensors
  must be treated as part of the per-step graph/replay contract even when a
  local inspection suggests a given decode kernel does not read one of them.
  Do not reintroduce H2D-skip optimizations without a stronger ownership/lifetime
  proof and a full `1024 -> 1024` C32 stability run.

2026-06-21 one-token decode compress-plan fast-path rejection:

- Tested a narrow `DeepseekV4RocmAtomModelState` fast path that bypassed the
  generic `make_compress_plans(...)` CPU planner when every scheduled request
  contributed exactly one decode token. The replacement filled the existing
  fixed-capacity pinned/GPU compress and write plan buffers directly, keeping
  the same H2D copies and graph-stable tensor shapes.
- Focused checks passed:
  `python3 -m py_compile vllm/models/deepseek_v4/amd/model_state.py`,
  `pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py -k 'metadata or decode'`,
  and `pytest -q tests/kernels/test_deepseek_v4_fused_compress_contract.py`.
- O128 smoke completed `320/320` with `717.70 output tok/s` and mean TPOT
  `32.20 ms` in
  `v2-async-decode-plan-fast-o128-20260621-C32.json`.
- Full C32 diagnostic completed `320/320` with `928.10 output tok/s`,
  total throughput `1859.83 tok/s`, and mean TPOT `33.20 ms` in
  `v2-async-decode-plan-fast-diagnostic-20260621-C32.json`.
- This is effectively identical to the safe async V2 control
  (`928.46 output tok/s`) and still below the V1 baseline
  (`970.59 output tok/s`). The code change was reverted. The CPU-side planner
 cleanup alone is not enough; the remaining gap is more likely in captured
 per-layer compressor/update/index/attention work or missing ATOM overlap.

2026-06-21 async V2 unified BF16 CSA-fused decode rejection:

- Extended the existing opt-in CSA translate fusion idea to the homogeneous
  BF16 unified-KV decode kernel. The experiment skipped the separate
  `csa_translate_pack` launch for CSA/r4 decode and had the unified paged
  decode kernel translate raw indexer top-k through the vLLM block table.
- Validation before serving:
  `python3 -m py_compile vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py
  vllm/models/deepseek_v4/amd/rocm.py`,
  `pytest -q tests/kernels/attention/test_deepseek_v4_split_kv_contract.py
  tests/kernels/test_deepseek_v4_atom_dependency_contract.py -k 'decode or metadata'`,
  and a small GPU equivalence check against
  `csa_translate_pack + sparse_attn_v4_paged_decode` passed with
  `max_abs_diff=0.0`.
- Server used the valid candidate shape:
  `VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling enabled,
  `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=32768`, `BLOCK_SIZE=128`,
  graph mode, no `--enforce-eager`, `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`,
  and `VLLM_ROCM_DSV4_ATOM_FUSE_CSA_TRANSLATE_DECODE=1`.
- O128 C32 smoke completed `320/320` with `702.68 output tok/s`,
  total throughput `6346.09 tok/s`, mean TPOT `32.68 ms`, and mean TTFT
  `1677.81 ms` in
  `v2-async-unified-fusedcsa-o128-20260621-C32.json`.
- This is slower than the safe async BF16 O128 controls
  (`719.49-723.89 output tok/s`) and therefore not worth a full C32 or
 accuracy run. The code change was reverted. As with the packed-KV CSA
 fusion, folding top-k translation into attention appears to add enough
 index math/register pressure to outweigh the saved launch.

2026-06-21 async V2 HCA flydsl two-kernel compressor rejection:

- Investigated whether the vLLM wrapper was incorrectly blocking ATOM's HCA
  compressor fast path. ATOM calls `fused_compress_attn` for HCA with
  `k_per_block = 128 // 128 = 1`; vLLM had a guard that treats the HCA
  `(512, 64, 128, False)` shape with `k_per_block == 1` as a flat-layout
  fallback and keeps it on the Triton single-kernel path.
- Standalone aiter sanity check was encouraging:
  `python3 /app/atomdsv4/aiter/op_tests/test_flydsl_compress_attn.py
  -s hca_main -b 32 -m 1 --modes decode` passed reference comparison and
  measured `13.54 us` for `path=2kernel` versus `55.38 us` for `path=single`.
- Temporarily removed the `not _hca_flat_layout` dispatch guard in
  `vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py` so the HCA
  shape could call `aiter.ops.flydsl.kernels.fused_compress_attn_hca`.
  Local vLLM checks passed:
  `python3 -m py_compile vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py`
  and
  `pytest -q tests/kernels/test_deepseek_v4_fused_compress_contract.py
  tests/kernels/test_deepseek_v4_atom_dependency_contract.py
  -k 'fused_compress or atom_compressor'`.
- Valid async V2 O128 smoke completed `320/320`, but was slightly slower:
  `v2-async-hca2kernel-o128-20260621-C32.json` reported `716.66 output tok/s`
  versus safe O128 controls around `719-724 output tok/s`.
- A chained full run after the O128 smoke crashed at main-run start
  (`v2-async-hca2kernel-full-20260621-C32.json`, `0/320`). Because benchmark
  validity requires a fresh server, this was treated as diagnostic only.
- A clean-server full C32 run using the valid deployment shape
  (`VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling enabled, graph mode, block
  size 128, `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=32768`) completed
  `320/320` but was slower:
  `v2-async-hca2kernel-freshfull-20260621-C32.json` reported
  `922.29 output tok/s`, total throughput `1848.18 tok/s`, mean TPOT
  `33.15 ms`, mean TTFT `1616.26 ms`.
- Conclusion: enabling aiter's HCA two-kernel compressor is not a usable win in
  the current vLLM async V2 integration. Despite the standalone kernel being
  much faster, end-to-end performance regresses versus the accepted async V2
  safe control (`928.46 output tok/s`) and remains below the V1 baseline
  (`970.59 output tok/s`). The dispatch change was reverted. A future retry
 needs a different hypothesis, likely around persistent scratch/workspace
 ownership or avoiding extra eager prefill/scheduler interactions, rather
 than simply unblocking the aiter HCA path.

2026-06-21 async V2 HCA compressor side-stream rejection:

- Tested a narrower version of ATOM's auxiliary compressor overlap without
  re-enabling vLLM's broader `aux_stream_list` GEMM overlap on ROCm. The
  temporary patch added an opt-in `VLLM_ROCM_DSV4_ATOM_AUX_COMPRESSOR=1` path
  in `amd/rocm.py` for pure-decode HCA layers only: launch the main ATOM
  compressor on a side stream, run `wq_b + _fused_qnorm_rope_kv_insert` on the
  main stream, then join before ATOM sparse attention.
- Validation before serving passed:
  `python3 -m py_compile vllm/models/deepseek_v4/amd/rocm.py` and
  `pytest -q tests/kernels/test_deepseek_v4_fused_compress_contract.py
  tests/kernels/test_deepseek_v4_atom_dependency_contract.py
  -k 'fused_compress or atom_compressor'`.
- Server used the valid deployment shape:
  `VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling enabled,
  `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=32768`, `BLOCK_SIZE=128`,
  graph mode, no `--enforce-eager`, `VLLM_ROCM_DSV4_USE_AITER_MHC=1`,
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`, and
  `VLLM_ROCM_DSV4_ATOM_AUX_COMPRESSOR=1`.
- O128 C32 smoke completed `320/320` with `721.82 output tok/s`, total
  throughput `6518.91 tok/s`, mean TPOT `33.87 ms`, and mean TTFT
  `1371.32 ms` in
  `v2-async-auxcompress-hca-o128-20260621-C32.json`.
- Full C32 completed `320/320` with `926.71 output tok/s`, total throughput
  `1857.05 tok/s`, mean TPOT `33.19 ms`, and mean TTFT `1405.26 ms` in
  `v2-async-auxcompress-hca-full-20260621-C32.json`.
- Conclusion: HCA-only compressor side-stream overlap is stable under async V2
  and graph mode, but is slower than the safe async V2 control
  (`928.46 output tok/s`) and still below the V1 baseline
  (`970.59 output tok/s`). The code change was reverted. The likely reason is
  that the added stream/event overhead is not offset by overlapping only HCA
  compressor work; a future auxiliary-stream retry should include CSA/indexer
  compressor overlap or reduce per-layer launch/metadata overhead first.

2026-06-21 async V2 attn-pre plus FFN-pre8 MHC reproduction rejection:

- User correction: async scheduling is a core Model Runner V2 optimization and
  must remain enabled for candidate runs. No-async runs are diagnostic only.
- Re-tested the historical above-baseline MHC-pre candidate with async V2,
  graph mode, ATOM attention/compressor defaults, vLLM-owned unified KV, and a
  fresh server before the full benchmark:
  `VLLM_USE_V2_MODEL_RUNNER=1`, `ASYNC_SCHEDULING=1`, `MAX_NUM_SEQS=32`,
  `MAX_NUM_BATCHED_TOKENS=32768`, `BLOCK_SIZE=128`,
  `VLLM_ROCM_DSV4_USE_AITER_MHC=1`,
  `VLLM_ROCM_DSV4_USE_AITER_MHC_POST=1`,
  `VLLM_ROCM_DSV4_USE_AITER_MHC_PRE=1`,
  `VLLM_ROCM_DSV4_USE_AITER_MHC_FUSE_NORM=1`,
  `VLLM_ROCM_DSV4_USE_AITER_MHC_PRE_ATTN=1`,
  `VLLM_ROCM_DSV4_USE_AITER_MHC_PRE_FFN=1`,
  `VLLM_ROCM_DSV4_AITER_MHC_PRE_FFN_MAX_LAYER=8`,
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`.
- Stability smoke:
  `v2-async-attnpre-ffnpre8-post-smoke-20260621-C32.json` completed
  `320/320`, output length `128`, output throughput `689.32 tok/s`.
- Fresh full benchmark:
  `v2-async-attnpre-ffnpre8-post-freshfull-20260621-C32.json` completed
  `320/320`, failed `0`, output throughput `869.12 tok/s`, total throughput
  `1741.63 tok/s`, mean TPOT `35.27 ms`, mean TTFT `1618.56 ms`.
- Outcome: reject this reproduction. It is stable but slower than the current
  safe async V2 ATOM control (`928.46 output tok/s`) and the V1 native baseline
  (`970.59 output tok/s`). It also does not reproduce the older
  `ds-v4-pro-v2-attnpre-ffnpre8-post-mbt32768-20260620-C32.json`
 result (`973.29 output tok/s`), so that older result should remain treated as
 historical/unexplained rather than a current candidate.

2026-06-21 async V2 no-mixed ATOM path acceptance:

- User correction: async scheduling is a core Model Runner V2 optimization and
  should not be disabled for accepted candidates. The accepted path keeps
  `VLLM_USE_V2_MODEL_RUNNER=1`, `ASYNC_SCHEDULING=1`, graph mode, no
  `--enforce-eager`, `BLOCK_SIZE=128`, `MAX_NUM_SEQS=32`,
  `MAX_NUM_BATCHED_TOKENS=32768`, `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`, and
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`.
- Accuracy was validated with unchanged `lmeval.sh` against a fresh server
  using `MAX_NUM_SEQS=64`, `MAX_NUM_BATCHED_TOKENS=16384`, `BLOCK_SIZE=128`,
  async V2, graph mode, and no legacy/bigfuse MHC. Result file:
  `results_2026-06-21T12-44-26.407661.json`.
- GSM8K passed: flexible exact match `0.953`, strict exact match `0.953`,
  which is within the target `0.95 +/- 0.01` band.
- Benchmark comparison before the final restart:
  `v2-async-atom-b128-nomixed-safe-rerun-20260621-C32.json` completed
  `320/320` with `932.16 output tok/s`, total throughput `1867.96 tok/s`, and
  mean TPOT `32.80 ms`.
- Final benchmark after restarting the server from the updated default launch:
  `v2-async-atom-b128-nomixed-safe-final-20260621-C32.json` completed
  `320/320` with `0` failures, `929.87 output tok/s`, total throughput
  `1863.38 tok/s`, mean TPOT `32.87 ms`, and mean TTFT `1614.66 ms`.
- This beats the current async V2 native/no-ATOM control
  (`v2-async-mhc-native-b256-current-20260621-C32.json`, `899.65 output
  tok/s`) and the older no-ATOM full control
  (`native-noatom-b256-full-C32.json`, `865.64 output tok/s`), but remains
  below the V1 native control
  (`native-v1-noatom-b256-mbt32768-20260620-C32.json`, `970.59 output tok/s`).
- Mixed KV remained accurate but slower in the current integration:
  `v2-async-atom-b128-mixed-nolegacy-full-20260621-C32.json` reported
 `881.33 output tok/s`. Therefore the launch default was changed to
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`.

2026-06-21 async V2 no-mixed block-size and decode-split follow-up:

- Tested the accepted no-mixed ATOM path with `BLOCK_SIZE=256` while keeping
  `VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling enabled, graph mode, no
  `--enforce-eager`, `MAX_NUM_SEQS=32`, `MAX_NUM_BATCHED_TOKENS=32768`, and
  `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`.
- Server log confirmed `block_size=256`, `Asynchronous scheduling is enabled`,
  `Using V2 Model Runner`, graph capture completion, and dense no-mixed ATOM
  unified KV views from vLLM-owned storage (`layout_counts={'dense': 61}`).
- C32 result:
  `v2-async-atom-b256-nomixed-safe-20260621-C32.json` completed `320/320`
  with `0` failures, `921.24 output tok/s`, total throughput `1846.09 tok/s`,
  mean TPOT `33.13 ms`, and mean TTFT `1675.25 ms`.
- Outcome: reject `BLOCK_SIZE=256` for the current no-mixed ATOM path. It is
  slower than the accepted `BLOCK_SIZE=128` final run (`929.87 output tok/s`)
  and the prior B128 rerun (`932.16 output tok/s`). Keep `BLOCK_SIZE=128`.
- Ran the standalone paged-decode split microbenchmark:
  `PYTHONPATH=/app/atomdsv4/previewdsv4 python3
  previewdsv4/benchmarks/kernels/bench_deepseek_v4_atom_paged_decode.py
  --tokens 32 --heads 16 --dim 512 --kv-lens 144,512,1024
  --block-ks 16,32,64 --kv-splits 1,2,4,8,16,32 --warmup 10 --rep 20`.
- The heuristic reports `kv_splits=16` for the deployment-like `T=32, H=16`
  shape. For dense unified BF16 decode, kernel-only timings support that
  region: `kv_len=512` was best at `kv_splits=16` (`0.0219-0.0223 ms` versus
  `0.0334-0.0337 ms` at `kv_splits=1`), and `kv_len=1024` was best around
  `kv_splits=8-16` (`0.0216-0.0226 ms`). The synthetic split-KV path was
  slower than dense unified BF16 for these shapes.
- Outcome: do not change `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS`. Prior live
  override tests for `8` and `32` crashed or regressed, and the standalone
  kernel evidence does not justify overriding the current heuristic.

2026-06-21 async V2 aiter direct paged-decode check:

- Tested the direct `aiter.pa_decode_sparse` path behind
  `VLLM_ROCM_DSV4_ATOM_USE_AITER_PA_DECODE=1` while keeping the accepted
  deployment shape: `VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling enabled,
  graph mode, no `--enforce-eager`, `BLOCK_SIZE=128`, `MAX_NUM_SEQS=32`,
  `MAX_NUM_BATCHED_TOKENS=32768`, and `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`.
- A synthetic BF16 equivalence smoke against the local paged-decode wrapper was
  close enough for a benchmark check: max absolute difference was `0.0078125`
  to `0.015625` for `T=16/24/32`, with finite outputs.
- Full C32 result:
  `v2-async-atom-b128-nomixed-aiterpa-20260621-C32.json` completed `320/320`
  with `0` failures, `922.41 output tok/s`, total throughput `1848.41 tok/s`,
  mean TPOT `33.34 ms`, and mean TTFT `1419.82 ms`.
- Outcome: keep `VLLM_ROCM_DSV4_ATOM_USE_AITER_PA_DECODE=0`. The direct aiter
  paged-decode path is slower than the accepted no-mixed B128 final run
  (`929.87 output tok/s`) and the B128 rerun (`932.16 output tok/s`), so it is
  not the current deployment path.

2026-06-21 async V2 no-scale decode dummy-pointer attempt:

- Tried removing the BF16 no-scale decode wrapper's `q.new_empty(1,
  dtype=torch.float32)` dummy scale allocation by passing the existing fp32
  `attn_sink` tensor as the unused `kv_scales` / `tail_scales` pointer.
- Focused validation was insufficient: `python3 -m py_compile
  vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py` and
  `pytest -q tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
  both passed.
- Full async V2 deployment benchmark failed during warmup/main transition:
  `v2-async-atom-b128-nomixed-dummyalloc-20260621-C32.json` completed `0/320`
  with `320` failures. The server log
  `server_v2_async_atom_b128_nomixed_dummyalloc_20260621.log` reported
  `torch.AcceleratorError: CUDA error: an illegal memory access was
  encountered` and the engine died.
- Outcome: reverted the dummy-pointer change. Keep the current `q.new_empty(1,
  dtype=torch.float32)` dummy tensors in `paged_decode.py` unless a safer
  graph-captured replacement is proven under the full async V2 server.

2026-06-21 async V2 CSA translate tile-size check:

- Investigated whether `csa_translate_pack` was over-fragmented at the
  deployment decode shape. ATOM/vLLM `index_topk` is `1024`, while the
  `1024 -> 1024` benchmark's steady CSA valid length is roughly `256-512`, so
  the existing `BLOCK_K=64` translator launches many masked-off K-block CTAs.
- A dynamic launch grid based on the current valid CSA length was rejected as
  unsafe for graph replay: the captured grid must remain large enough for later
  decode steps, and reducing it from per-step metadata could silently truncate
  later longer-context CSA rows.
- Temporarily made the translator K tile configurable and ran a synthetic
  steady decode microbenchmark with `T=32`, `index_topk=1024`, `valid_k=256`,
  `window_size=128`, and the same fused top-k translate/pack kernel. Results:
  `BLOCK_K=64` measured `0.014988 ms`, `BLOCK_K=128` measured `0.015301 ms`,
  and `BLOCK_K=256` measured `0.015058 ms`.
- Outcome: reject the tile-size change and keep the current `BLOCK_K=64`.
  Larger K tiles did not improve the isolated translator, and this path would
  not close the V2-vs-V1 gap. The temporary environment knob was reverted.

2026-06-21 async V2 benchmark max-model-len check:

- Tested whether constraining the server to the benchmark's nominal
  `1024 input + 1024 output` length could recover the V2-vs-V1 gap while
  keeping async scheduling, graph mode, `BLOCK_SIZE=128`, and the accepted
  no-mixed ATOM attention/compressor defaults.
- `MAX_MODEL_LEN=2048 MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=32768
  BLOCK_SIZE=128 ASYNC_SCHEDULING=1 ENFORCE_EAGER=0 bash
  launchdeepseekgraph.sh` loaded successfully, confirmed
  `Asynchronous scheduling is enabled`, used `V2 Model Runner`, captured
  graphs, and served `/v1/models`.
- The subsequent unchanged C32 benchmark failed immediately:
  `bench-sparsemla/v2-async-atom-b128-nomixed-maxlen2048-20260621-C32.json`
  completed `0/320` requests and failed `320/320`. The client saw connection
  refused after the server process exited; the server log ended after startup
  and did not contain a Python traceback. Treat `MAX_MODEL_LEN=2048` as too
  tight/invalid for this OpenAI-chat benchmark path.
- Older `MAX_MODEL_LEN=2304` experiments from earlier integration states were
  also not promising: saved C32 runs in `bench-from-vllm-*len2304` were only
  `850.82-857.47 output tok/s`, below the current accepted async V2 no-mixed
  ATOM path (`929.87-932.16 output tok/s`).
- Outcome: keep the default benchmark/accuracy server at `MAX_MODEL_LEN=8192`.
  Reducing max length is not a validated performance path for the current
  async V2 ATOM integration.
