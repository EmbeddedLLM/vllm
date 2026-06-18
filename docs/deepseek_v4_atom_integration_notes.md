# DeepSeek V4 ATOM Integration Notes

Date: 2026-06-17

This note records what was tested while integrating ATOM-style DeepSeek V4
optimizations into vLLM's AMD path. It is written as a handoff for another LLM
or engineer: keep the deployment path, avoid known-bad branches, and focus
future work on the unresolved gaps.

## Goal

Run DeepSeek-V4-Pro FP8/FP4 on vLLM with vLLM's scheduler, attention metadata,
KV-cache abstraction, and fused MoE abstraction, while bringing in safe ATOM
ops where they improve performance or preserve accuracy.

Validation target:

- Accuracy command: `launchdeepseekgraph.sh` plus unchanged `lmeval.sh`.
- GSM8K target: `0.95 +/- 0.01`; practical lower bound is `0.94`.
- Benchmark command: fresh server restart, then `benchmarkvllm.sh` at C32.
- Serving should run with graph capture, not `--enforce-eager`.

## Current Passing Configuration

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
