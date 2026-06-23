# DeepSeek V4 ROCm ATOM Integration Handoff

This note is intended to let another coding agent resume the current goal
without replaying the full chat history. It should be treated as the
high-signal resume point; the longer running log remains in
`docs/deepseek_v4_atom_integration_notes.md`.

## Bootstrap Goal For Next Agent

Continue the DeepSeek-V4-Pro ROCm ATOM integration in `/app/atomdsv4/previewdsv4`
using vLLM's scheduler and V2 model runner. Keep async scheduling enabled
(`VLLM_USE_V2_MODEL_RUNNER=1` and `--async-scheduling`) for all accepted
accuracy/performance runs. The immediate goal is to make the async V2 path using
ATOM attention/compressor kernels faster than the V1 native baseline while
preserving GSM8K accuracy at `0.95 +/- 0.01`.

Do not use `--enforce-eager` for accepted runs. Do not modify
`/app/atomdsv4/aiter`; the installed/source reference version is
`aiter==0.1.15.post1`. Do not change `lmeval.sh`.

## Non-Negotiable Constraints From The User

- Keep vLLM's scheduler. The ATOM modeling-file behavior should fit into
  vLLM's request scheduling, attention abstraction, KV cache abstraction, and
  fused MoE abstraction.
- Use async scheduling for accepted Model Runner V2 results. Disabling async
  scheduling is only a diagnostic, not an acceptable final path.
- Do not use `--enforce-eager` for accepted accuracy or benchmark runs.
- Do not change `lmeval.sh`.
- Do not modify `/app/atomdsv4/aiter`; use the installed/reference
  `aiter==0.1.15.post1`.
- vLLM must not depend on ATOM as a Python package. Any required ATOM-style
  logic should be vendored, wrapped, or reimplemented inside vLLM.
- Reuse vLLM weight loading logic. The user explicitly wants to preserve fast
  vLLM loader support such as safetensors strategies and related loader paths.
- Avoid modifying model runner / GPU worker code if possible. If it becomes
  necessary, keep changes generic and do not hard-import DeepSeek-V4 ROCm model
  modules into generic worker paths.
- Do not break CUDA/NVIDIA DeepSeek-V4 paths. CUDA should keep using the
  existing KV-cache/spec behavior unless explicitly guarded.
- Follow ATOM's op sequence where possible: fused MoE, MHC/HC only after the
  attention/compressor path is stable, attention backend, compressor/indexer,
  then auxiliary stream/overlap last.
- `os.environ.get(...)` in hot paths is slow; cache env-derived switches at
  import/config time.

## Important External Reference

Known ATOM commit that successfully ran an ATOM modeling-file path in vLLM:

```text
https://github.com/ROCm/ATOM/commit/e95ef5d74a860e04a6219dfff319535bc19449dd
```

Use it as a reference point for sequencing and integration choices, not as a
license to add ATOM as a package dependency.

## Workspace Layout

- vLLM repo under test: `/app/atomdsv4/previewdsv4`
- ATOM repo/reference: `/app/atomdsv4/ATOM`
- aiter source/reference only: `/app/atomdsv4/aiter`
- Launch script: `/app/atomdsv4/launchdeepseekgraph.sh`
- Accuracy script: `/app/atomdsv4/lmeval.sh`
- Benchmark script: `/app/atomdsv4/benchmarkvllm.sh`
- Benchmark outputs: `/app/atomdsv4/bench-sparsemla`
- Accuracy outputs: `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph`
- Accuracy tee log: `/app/atomdsv4/lmevaldeepseekprographmtp_aitermhc_nobreakablecudagraph.log`
- Secondary run/smoke logs: `/app/atomdsv4/runlogs`

## Working Tree Snapshot

At handoff the repo is intentionally dirty. Do not assume all changes are from
the most recent experiment. `git status --short` showed:

- 26 tracked files modified.
- Many new untracked docs, tests, and ROCm ATOM kernel files.
- `git diff --stat` showed roughly `10620 insertions` and `125 deletions`.

Important modified tracked files:

- `docs/deepseek_v4_atom_integration_notes.md`
- `tests/kernels/test_fused_indexer_q_rope_quant.py`
- `tests/v1/core/test_kv_cache_utils.py`
- `tests/v1/test_kv_cache_spec_registry.py`
- `tests/v1/worker/test_gpu_model_runner.py`
- `tests/v1/worker/test_utils.py`
- `vllm/envs.py`
- `vllm/model_executor/kernels/mhc/aiter.py`
- `vllm/model_executor/layers/mhc.py`
- `vllm/models/deepseek_v4/amd/model.py`
- `vllm/models/deepseek_v4/amd/rocm.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/__init__.py`
- `vllm/models/deepseek_v4/attention.py`
- `vllm/models/deepseek_v4/common/ops/__init__.py`
- `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py`
- `vllm/models/deepseek_v4/compressor.py`
- `vllm/models/deepseek_v4/sparse_mla.py`
- `vllm/v1/attention/backends/mla/indexer.py`
- `vllm/v1/attention/backends/mla/sparse_swa.py`
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/utils.py`

Important untracked files:

- `benchmarks/kernels/bench_deepseek_v4_atom_paged_decode.py`
- `docs/deepseek_v4_atom_agent_handoff.md`
- `docs/deepseek_v4_atom_op_surface_audit.md`
- `docs/deepseek_v4_rocm_kv_workspace_plan.md`
- `docs/deepseek_v4_rocm_kvcache_workspace_design.md`
- `tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
- `tests/kernels/test_deepseek_v4_atom_dependency_contract.py`
- `tests/kernels/test_deepseek_v4_atom_op_surface_audit.py`
- `tests/kernels/test_deepseek_v4_compressor_contract.py`
- `tests/kernels/test_deepseek_v4_fused_compress_contract.py`
- `vllm/models/deepseek_v4/amd/atom_native_abi.py`
- `vllm/models/deepseek_v4/amd/model_state.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/compress_plan.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/csa_translate_pack.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/inverse_rope.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode_indices.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill_indices.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/reference.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/state_writes.py`

## Current Process State At Handoff

No vLLM server, benchmark, or lmeval process was running when this file was
written. The interrupted `INDEX_TOPK_FREQ=8` launch did not leave a server
behind.

Use this exact check before starting another server:

```bash
pgrep -af 'vllm serve|launchdeepseekgraph|benchmarkvllm|lmeval|VLLM::EngineCore|VLLM::Worker_TP' || true
```

## Current Accepted Launch Shape

`/app/atomdsv4/launchdeepseekgraph.sh` currently defaults to:

- `MAX_NUM_SEQS=32`
- `MAX_NUM_BATCHED_TOKENS=32768`
- `ENFORCE_EAGER=0`
- `BLOCK_SIZE=128`
- `MAX_MODEL_LEN=8192`
- `ASYNC_SCHEDULING=1`
- `VLLM_USE_V2_MODEL_RUNNER=1`
- `VLLM_USE_BREAKABLE_CUDAGRAPH=1`
- `VLLM_ROCM_DSV4_ATOM_ATTENTION=1`
- `VLLM_ROCM_DSV4_ATOM_MAIN_COMPRESSOR=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV=1`
- `VLLM_ROCM_DSV4_ATOM_UNIFIED_KV_FROM_VLLM=1`
- `VLLM_ROCM_DSV4_ATOM_MIXED_KV=0`
- `VLLM_ROCM_DSV4_ATOM_COMPRESS_PLAN=1`
- `VLLM_ROCM_DSV4_ATOM_INDEXER_COMPRESSOR=0`
- `VLLM_ROCM_DSV4_ATOM_COMPRESS_FIRST=0`
- `VLLM_ROCM_DSV4_ATOM_SKIP_PAGED_PREFILL=0`
- `VLLM_ROCM_DSV4_ATOM_PREFILL_ALLOW_MIXED=1`
- `VLLM_ROCM_DSV4_ATOM_INDEXER_FASTPATH=1`
- `VLLM_ROCM_DSV4_ATOM_SKIP_INDEXER_METADATA=1`
- `VLLM_ROCM_DSV4_ATOM_PREFILL_INDEX_REUSE=1`
- `VLLM_ROCM_DSV4_ATOM_DECODE_INDEX_REUSE=1`
- `VLLM_ROCM_DSV4_ATOM_FUSED_HCA_INDEX=1`
- `VLLM_ROCM_DSV4_USE_AITER_MHC=1`
- `VLLM_ROCM_DSV4_USE_AITER_MHC_LEGACY_OPS=0`
- `AITER_BF16_FP8_MOE_BOUND=0`
- `AITER_LOG_LEVEL=WARNING`
- `ATOM_MOE_GU_ITLV=1`
- `ATOM_USE_FUSED_Q_NORM_QUANT=1`
- `ATOM_USE_AITER_FUSED_CLAMP_ACT_MUL=1`
- `--kv-cache-dtype fp8`
- `--moe-backend aiter`
- no prefix cache

Keep async scheduling on. Non-async runs are diagnostic only.

Launch side effects:

- The script removes `/root/.cache/vllm` on every start.
- Server output is tee'd to
  `/app/atomdsv4/dsv4prographnomtp-aitermhc_nobreakablecudagraph.log`.
- Accuracy output from `lmeval.sh` is tee'd to
  `/app/atomdsv4/lmevaldeepseekprographmtp_aitermhc_nobreakablecudagraph.log`.

## ATOM Recipe Target

The user's stated target is to move toward ATOM recipe performance, not only to
beat the current vLLM baseline. For iteration, use the C32 row from
`/app/atomdsv4/ATOM/recipes/DeepSeek-V4.md`:

- Workload: input sequence length `1024`, output sequence length `1024`.
- Concurrency: `32`.
- Num prompts: `320`.
- ATOM FP8 TP8 no-MTP output throughput: `1145.71 tok/s`.
- ATOM total throughput: `2287.81 tok/s`.
- ATOM mean TPOT: `26.90 ms`.

The current immediate engineering target remains: make async V2 beat the V1
native baseline (`970.59 output tok/s`) while passing GSM8K. The ATOM recipe
number is the larger target.

## Accuracy And Performance Baselines

Accepted current async V2 ATOM path, no top-k reuse:

- Result file:
  `/app/atomdsv4/bench-sparsemla/v2-async-atom-b128-nomixed-safe-rerun-20260621-C32.json`
- Output throughput: `932.1605860450228 tok/s`
- Total throughput: `1867.962424379284 tok/s`
- Mean TPOT: `32.797354056161005 ms`
- Completed/failed: `320/0`

Second accepted rerun:

- Result file:
  `/app/atomdsv4/bench-sparsemla/v2-async-atom-b128-nomixed-safe-final-20260621-C32.json`
- Output throughput: `929.8738141042135 tok/s`
- Total throughput: `1863.3799477947716 tok/s`
- Mean TPOT: `32.867308677032355 ms`
- Completed/failed: `320/0`

Current async V2 no-ATOM/native control:

- Result file:
  `/app/atomdsv4/bench-sparsemla/v2-async-mhc-native-b256-current-20260621-C32.json`
- Output throughput: `899.6479103845471 tok/s`
- Total throughput: `1802.8100704190338 tok/s`
- Mean TPOT: `34.07445619489062 ms`

V1 native baseline to beat:

- Result file:
  `/app/atomdsv4/bench-sparsemla/native-v1-noatom-b256-mbt32768-20260620-C32.json`
- Output throughput: `970.5858730690604 tok/s`
- Total throughput: `1944.963097204797 tok/s`
- Mean TPOT: `31.24914234224303 ms`

Accuracy pass for accepted no-mixed async V2 path:

- Result file:
  `/app/atomdsv4/results_deepseekprographmtp_aitermhc_nobreakablecudagraph/deepseek-ai__DeepSeek-V4-Pro/results_2026-06-21T12-44-26.407661.json`
- GSM8K flexible exact match: `0.9529946929492039`
- GSM8K strict exact match: `0.9529946929492039`

## Fast But Invalid Or Non-Reproducible Runs

These are tempting because they beat the V1 baseline, but they are not
acceptable evidence:

- `v2-async-atom-b256-legacy-mhc-full`: output throughput `1057.89 tok/s`;
  accuracy failed.
- `v2-async-mhc-legacyops-20260621-C32.json`: output throughput
  `1053.93 tok/s`, total throughput `2111.98 tok/s`, mean TPOT `28.78 ms`;
  GSM8K flexible exact match was around `0.14`.
- `v2-async-atom-b256-mixed-aiterbigfuse-full`: output throughput
  `994.29 tok/s`; accuracy failed around `0.16`.
- Historical `ds-v4-pro-v2-attnpre-ffnpre8-post-mbt32768`: output throughput
  around `973.29 tok/s`, but reproduction under async V2 was only
  `869.12 tok/s`.

Do not use these as proof that the goal is achieved. They are clues for later
kernel/debug work only.

## Latest Active Experiment

There is an uncommitted candidate to speed up ATOM top-k reuse by also reusing
translated CSA indices on skip-top-k CSA layers. It is default-inactive unless
`VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1` is set.

Files touched for this candidate:

- `vllm/models/deepseek_v4/amd/model_state.py`
  - Added `csa_translate_key`, `csa_translate_hits`, and
    `csa_translate_writes` to prefill/decode per-forward caches.
- `vllm/models/deepseek_v4/amd/rocm.py`
  - Decode CSA path skips `csa_translate_pack()` when `skip_topk=True` and the
    cached translate key matches.
  - Prefill CSA path does the analogous skip for `prefix_csa_indices`.

Validation already run:

```bash
python3 -m py_compile \
  vllm/models/deepseek_v4/amd/rocm.py \
  vllm/models/deepseek_v4/amd/model_state.py

pytest -q \
  tests/kernels/test_deepseek_v4_atom_dependency_contract.py \
  tests/kernels/attention/test_deepseek_v4_split_kv_contract.py \
  -k 'not launch_defaults_select_rocm_atom_benchmark_path'
```

Result: `41 passed, 1 deselected`.

The deselected test is a stale launch-default assertion expecting
`MAX_NUM_BATCHED_TOKENS=16384`; current launch default is `32768`.

The full focused invocation without deselection currently fails only because of
that stale assertion. It is unrelated to the CSA-translate reuse candidate.

Performance with candidate and `INDEX_TOPK_FREQ=4`:

- Launch env:
  `MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=32768 BLOCK_SIZE=128 ASYNC_SCHEDULING=1 ENFORCE_EAGER=0 VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1 VLLM_ROCM_DSV4_ATOM_INDEX_TOPK_FREQ=4`
- Result file:
  `/app/atomdsv4/bench-sparsemla/v2-async-atom-b128-topkreuse-csareuse-20260621-C32.json`
- Output throughput: `956.0223484758587 tok/s`
- Total throughput: `1915.7791592504514 tok/s`
- Mean TPOT: `32.27243741070906 ms`
- Completed/failed: `320/0`

Interpretation: this is a real improvement over accepted async V2 ATOM
(`~932 tok/s`) and async V2 native (`~900 tok/s`), but still below V1 native
(`970.59 tok/s`). It is not enough to call V2 better than V1. Accuracy was not
rerun after the CSA-translate-skip change, but earlier top-k reuse freq4 passed
GSM8K before this optimization. Rerun `lmeval.sh` unchanged before accepting it.

An `INDEX_TOPK_FREQ=8` run was about to be launched but was interrupted before a
server remained running. That experiment is still unmeasured and would need
accuracy validation if it beats V1.

CSA translate reuse safety assumptions:

- Refresh CSA layers must still run `csa_translate_pack()` and update the
  shared `csa_indices`/`prefix_csa_indices` buffer.
- Skip-top-k CSA layers may reuse translated indices only after a matching
  refresh in the same forward.
- The cache key includes token count, block-table pointer/shape/strides, top-k
  buffer pointer/shape, and common-index key. The top-k buffer pointer alone is
  not enough because refresh layers mutate its contents without changing the
  pointer.
- If a skip layer appears without a prior matching refresh, the key should miss
  and translation should run normally.
- `INDEX_TOPK_FREQ=8` is accuracy-risky because it reuses sparse top-k choices
  across more CSA layers than the already-validated freq4 behavior.

## Important Files For The Integration

Core ROCm DSV4 model/runtime:

- `vllm/models/deepseek_v4/amd/model.py`
- `vllm/models/deepseek_v4/amd/rocm.py`
- `vllm/models/deepseek_v4/amd/model_state.py`
- `vllm/models/deepseek_v4/attention.py`
- `vllm/models/deepseek_v4/compressor.py`
- `vllm/models/deepseek_v4/sparse_mla.py`

Vendored/preview ATOM-style kernels:

- `vllm/models/deepseek_v4/amd/v4_kernels/compress_plan.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/csa_translate_pack.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/inverse_rope.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/paged_decode_indices.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill_indices.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/reference.py`
- `vllm/models/deepseek_v4/amd/v4_kernels/state_writes.py`

vLLM KV/cache/scheduler integration:

- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/utils.py`
- `vllm/v1/attention/backends/mla/indexer.py`
- `vllm/v1/attention/backends/mla/sparse_swa.py`
- `vllm/v1/attention/ops/rocm_aiter_mla_sparse.py`

MHC/aiter wrappers, but do not focus here unless necessary:

- `vllm/model_executor/layers/mhc.py`
- `vllm/model_executor/kernels/mhc/aiter.py`

Environment flags:

- `vllm/envs.py`

Useful tests:

- `tests/kernels/test_deepseek_v4_atom_dependency_contract.py`
- `tests/kernels/attention/test_deepseek_v4_split_kv_contract.py`
- `tests/kernels/test_deepseek_v4_atom_op_surface_audit.py`
- `tests/kernels/test_deepseek_v4_compressor_contract.py`
- `tests/kernels/test_deepseek_v4_fused_compress_contract.py`
- `tests/kernels/test_fused_indexer_q_rope_quant.py`
- `tests/v1/core/test_kv_cache_utils.py`
- `tests/v1/test_kv_cache_spec_registry.py`
- `tests/v1/worker/test_gpu_model_runner.py`
- `tests/v1/worker/test_utils.py`

Useful docs already in this repo:

- `docs/deepseek_v4_atom_integration_notes.md`
- `docs/deepseek_v4_atom_op_surface_audit.md`
- `docs/deepseek_v4_rocm_kv_workspace_plan.md`
- `docs/deepseek_v4_rocm_kvcache_workspace_design.md`

Reference source in ATOM:

- `/app/atomdsv4/ATOM/atom/models/deepseek_v4.py`
- `/app/atomdsv4/ATOM/recipes/DeepSeek-V4.md`

## Current Feature Coverage

What is active in the accepted no-mixed async V2 path:

- vLLM Model Runner V2 with async scheduling.
- vLLM scheduler and request lifecycle.
- vLLM weight loading.
- vLLM fused MoE with aiter MXFP4 backend, not ATOM's full dual-stream MoE
  sequence.
- ROCm-only `DeepseekV4RocmAtomModelState`.
- vLLM-owned ATOM unified KV binding, homogeneous BF16 view over vLLM storage.
- ATOM-style paged decode/prefill wrappers.
- Local `csa_translate_pack` before CSA attention.
- ATOM main CSA/HCA fused compressor path.
- ATOM-style read-before-update compressor ordering in the ROCm path.
- Fused q norm/quant path where compatible.
- AITER fused clamp/activation/mul.
- Decode/prefill index reuse and HCA fused-index path.
- Breakable CUDA graph; this is expected for DSV4 because torch compile is not
  the current path.

What is present but default-off or not accepted:

- Packed mixed BF16-SWA + FP8 compressed-tail KV contract.
- Split-KV decode wrappers for mixed layout.
- Indexer-inner fused compressor.
- Top-k reuse (`VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1`).
- Fused CSA translate/decode.
- Direct aiter paged decode.
- AITER MHC legacy/raw/bigfuse variants.
- Aux stream compressor overlap.

What was explicitly removed or should not be revived as-is:

- Direct raw-top-k CSA sparse attention runtime path. It bypassed the ATOM
  sequence and did not improve served C32 throughput.

## KV Cache And Workspace Context

The user's broader design question is whether vLLM's existing KV-cache system
and workspace manager can support a unified DSV4 ROCm cache design without
breaking CUDA.

Current understanding:

- ATOM/SGLang use an attention structure with unified KV cache and an SWA ring
  buffer that differs from vLLM's current ragged paged pattern.
- A true ATOM-like ROCm path likely needs request-state allocation for
  persistent SWA rings and compressor state rings.
- The current branch models this with a ROCm-only
  `DeepseekV4RocmAtomModelState` rather than putting DeepSeek-specific logic in
  generic workers.
- CUDA/NVIDIA should keep using the existing KV-cache spec and path.
- Generic vLLM changes should stay contract-based: KV-cache specs, grouping,
  post-bind views, and worker utilities should not import DeepSeek-V4 ROCm
  model modules.

Workspace manager notes from the discussion:

- `WorkspaceManager.get_simultaneous(...)` is effectively a reusable
  preallocated-buffer replacement for `torch.empty`.
- The returned buffers live until the next `get_simultaneous(...)` call.
- It exists to reuse large workspaces across layers and across eager/graph
  pools, especially MoE, MLA/sparse-MLA/indexer, and DCP.
- It also handles micro-batch uniqueness for DBO.
- Nested calls in a deep stack can be tricky because lifetimes are manual.
- Removing it can increase memory usage by roughly `0.5-1.5 GB` for DSv3-like
  shapes according to the prior workspace-manager context.
- For the ATOM DSV4 path, use workspace only when lifetime and ownership are
  clear. Request-persistent SWA/compressor rings are better modeled as model
  state/KV-cache owned storage than as transient workspace.

Relevant design docs:

- `docs/deepseek_v4_rocm_kv_workspace_plan.md`
- `docs/deepseek_v4_rocm_kvcache_workspace_design.md`

## Detailed Experiment Ledger

Accepted async V2 no-mixed ATOM path:

- Accuracy passed GSM8K at `0.9529946929492039`.
- C32 output throughput is currently `929.87-932.16 tok/s`.
- This beats current async V2 native (`899.65 tok/s`) but not V1 native
  (`970.59 tok/s`).

Top-k reuse:

- Before CSA translate reuse, `VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1` with
  `INDEX_TOPK_FREQ=4` passed GSM8K but benchmarked only `894.56 tok/s`.
- The pre-CSA-translate-reuse top-k freq4 accuracy run reported GSM8K
  flexible exact match `0.9545 +/- 0.0057` and strict exact match
  `0.9553 +/- 0.0057`.
- After adding CSA translate reuse for skip-top-k CSA layers, the same freq4
  experiment benchmarked `956.02 tok/s`.
- Accuracy has not been rerun after this latest CSA translate reuse change.

Mixed packed KV:

- Multiple mixed-KV paths were accuracy-correct, including graph-mode runs.
- Latest current no-legacy mixed path:
  `v2-async-atom-b128-mixed-nolegacy-full-20260621-C32.json` was around
  `881.33 tok/s`.
- Earlier generic/mixed spec runs were around `807-811 tok/s`.
- Mixed KV increases capacity and moves toward ATOM FP8 tail design, but the
  current implementation pays split-layout adaptation and is not a throughput
  win.

Block size:

- `BLOCK_SIZE=128` is the accepted current default.
- `BLOCK_SIZE=256` for the no-mixed ATOM path benchmarked `921.24 tok/s`,
  slower than B128.
- If ATOM requires a different block size for a future true unified layout, test
  it, but do not change the default based on current evidence.

Max model length:

- `MAX_MODEL_LEN=8192` is the accepted current default.
- `MAX_MODEL_LEN=2048` with the accepted no-mixed ATOM path failed the C32
  benchmark with `0/320` completions and no useful Python traceback.
- Older `MAX_MODEL_LEN=2304` runs were `850.82-857.47 tok/s`.

Decode split count:

- Empty/default `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=` should remain.
- `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=32` crashed or killed EngineCore.
- `8` had prior instability.
- Split-KV `kv_splits=1` exists for mixed/split wrapper experiments, but it is
  not a default win.

Direct aiter paged decode:

- `VLLM_ROCM_DSV4_ATOM_USE_AITER_PA_DECODE=1` benchmarked `922.41 tok/s`.
- Keep it off by default.
- Decode-fused-single style paths are not accepted evidence of a win unless the
  actual deployment path reaches `num_splits == 1`, runs under async V2 graph
  mode, beats V1, and passes GSM8K. Prior attention-kernel probing did not make
  this a default-useful feature.

CSA translate fusion/direct CSA:

- Direct raw-top-k CSA decode removed translator cost in isolated timing but
  served throughput was worse.
- Fused CSA translate/decode for packed mixed KV passed accuracy but benchmarked
  only `774.60 tok/s`.
- A homogeneous/BF16 fused CSA translate/decode attempt had a misleading short
  benchmark and then failed accuracy; the code change was reverted.
- Future CSA work should fuse or eliminate translation using an ATOM-native
  metadata contract, not revive the old raw direct-CSA bypass.

HCA compressor experiments:

- HCA flydsl two-kernel compressor: `922.29 tok/s`, slower; reverted.
- HCA compressor side stream: `926.71 tok/s`, slower; reverted.
- Decode split workspace: `926.75 tok/s`, slower.

MHC:

- MHC is not structurally required for ATOM attention/compressor correctness.
- AITER MHC legacy ops are fast but numerically invalid in this branch.
- Raw/bigfuse/fused-norm MHC variants either failed GSM8K or were slower.
- AITER `mhc_fused_post_pre` availability in `aiter==0.1.15.post1` was checked
  during the work; do not assume all ATOM MHC kernels are usable without
  reproducing ATOM's exact call sequence and full accuracy.
- Offline swizzling using generic `shuffle_weight(fn, layout=(16,16))` is not
  applicable to DSV4 MHC `fn` because MHC uses `mix_hc=24` rows and the helper
  asserts incompatible divisibility.

Compressor ordering:

- CUDA and ROCm compressor ordering differ in practice. The accepted ROCm ATOM
  path uses ATOM-style read-before-update ordering for accuracy.
- Disabling ATOM compressor order gave a small throughput improvement in one
  test but failed accuracy.
- Fused q norm/quant and compressor ordering are not mathematically the same
  feature. The ordering is about which state the compressor reads/writes; fused
  q norm/quant is an upstream projection/norm/quant optimization.

Auxiliary streams:

- Aux stream compressor overlap was tried and was slower in the current graph
  deployment.
- Leave aux stream/overlap as a later optimization after the single-stream
  attention/compressor/indexer path is stable and above V1.

Metadata/CPU overhead:

- There is strong evidence that served throughput is limited by metadata and
  layout adaptation around the kernels, not just raw attention kernel time.
- Concrete targets to profile or reduce:
  `csa_translate_pack`, common decode-index reuse, HCA fused index path,
  indexer score/top-k dispatch, Python metadata preparation, H2D metadata
  copies, and prefill/mixed fallback metadata.
- `csa_translate_pack` has been comparable to CSA attention kernel time in
  profiles. Removing it in isolation helped layer timing but did not by itself
  recover served throughput.

Profiler flags already wired in `vllm/envs.py`:

- `VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_EVERY`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_LAYER`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_EVERY`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_METADATA_START_AFTER`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_EVERY`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_LAYER`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_COMPRESSOR_START_AFTER`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL_MIN_T`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL_MIN_TOKEN_OFFSET`
- `VLLM_ROCM_DSV4_ATOM_PROFILE_PREFILL_TRACE`

## Rejected Or Non-Accepted Paths

Do not accept any of these without new full accuracy/perf evidence:

- aiter MHC legacy/raw/bigfuse paths: some were fast but failed GSM8K badly.
- `VLLM_ROCM_DSV4_ATOM_MIXED_KV=1`: accurate in some runs but slower.
- `BLOCK_SIZE=256` no-mixed: slower than `BLOCK_SIZE=128`.
- `MAX_MODEL_LEN=2048`: C32 benchmark failed `0/320`; server exited after
  `/v1/models`.
- `MAX_MODEL_LEN=2304`: older runs were slow (`~850-857 tok/s`).
- `VLLM_ROCM_DSV4_ATOM_USE_AITER_PA_DECODE=1`: C32 `~922 tok/s`, slower.
- HCA side-stream/two-kernel compressor experiments: slower; reverted.
- decode split workspace experiment: slower.
- `VLLM_ROCM_DSV4_ATOM_DECODE_KV_SPLITS=32` and `8`: crashed or unstable.
- top-k reuse freq4 before CSA translate reuse: accurate but slow
  (`894.56 tok/s`) because it skipped top-k but still ran CSA translate/pack.
- direct ATOM CSA sparse attention kernel was removed from runtime because it
  was not used in accepted fastest paths.
- `VLLM_ROCM_DSV4_ATOM_FUSE_CSA_TRANSLATE_DECODE=1`: accuracy-safe for one
  packed mixed path but slow; another homogeneous attempt was reverted.
- raw/direct AITER MHC legacy ops: fast but wrong accuracy.
- generic offline MHC weight swizzle: not applicable to DSV4 MHC `fn`.

## Validation Gates

A candidate should not be called accepted until all of these hold:

- Server launches with `VLLM_USE_V2_MODEL_RUNNER=1`, async scheduling enabled,
  graph mode, and no `--enforce-eager`.
- C32 benchmark runs after a fresh server restart so KV/prefix caches are not
  contaminated by previous runs.
- `benchmarkvllm.sh` reports `320/320` successful requests and `0` failures.
- Output throughput beats the current V1 native baseline
  `970.5858730690604 tok/s`.
- Unchanged `bash lmeval.sh` reports GSM8K exact match inside `0.95 +/- 0.01`.
- CUDA/NVIDIA path tests or contract tests still pass if generic KV/cache code
  was touched.

Suggested quick checks before a full benchmark:

```bash
python3 -m py_compile \
  vllm/models/deepseek_v4/amd/rocm.py \
  vllm/models/deepseek_v4/amd/model_state.py

pytest -q \
  tests/kernels/test_deepseek_v4_atom_dependency_contract.py \
  tests/kernels/attention/test_deepseek_v4_split_kv_contract.py \
  -k 'not launch_defaults_select_rocm_atom_benchmark_path'
```

If the stale launch-default test is fixed, remove the `-k` deselection.

## Runbook

Start a clean async V2 C32 server:

```bash
cd /app/atomdsv4
MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=32768 BLOCK_SIZE=128 \
ASYNC_SCHEDULING=1 ENFORCE_EAGER=0 \
bash launchdeepseekgraph.sh
```

Benchmark C32:

```bash
cd /app/atomdsv4
RESULT_PREFIX=<name> CONCURRENCIES=32 bash benchmarkvllm.sh
```

Run lmeval unchanged after starting a server. For faster lmeval server-side
throughput, the launch script supports `MAX_NUM_SEQS=256`, but do not edit the
`lmeval.sh` command:

```bash
cd /app/atomdsv4
bash lmeval.sh
```

When testing top-k reuse plus CSA translate reuse:

```bash
cd /app/atomdsv4
MAX_NUM_SEQS=32 MAX_NUM_BATCHED_TOKENS=32768 BLOCK_SIZE=128 \
ASYNC_SCHEDULING=1 ENFORCE_EAGER=0 \
VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1 \
VLLM_ROCM_DSV4_ATOM_INDEX_TOPK_FREQ=4 \
bash launchdeepseekgraph.sh
```

Potential next experiment:

```bash
VLLM_ROCM_DSV4_ATOM_INDEX_TOPK_FREQ=8
```

Only keep it if it beats `970.59 tok/s` on C32 and then passes GSM8K accuracy.

## Suggested Next Priorities

1. Decide what to do with the latest CSA translate reuse candidate.
   - If keeping it, first run unchanged `lmeval.sh` with
     `VLLM_ROCM_DSV4_ATOM_USE_INDEX_CACHE=1` and `INDEX_TOPK_FREQ=4`.
   - If accuracy fails, revert only the `csa_translate_key` experiment in
     `amd/model_state.py` and `amd/rocm.py`.
   - If accuracy passes, try `INDEX_TOPK_FREQ=8` only as a measured experiment,
     then gate it by GSM8K.

2. Profile metadata and translation overhead under deployment shape.
   - Use C32, `1024/1024`, async V2, graph mode.
   - Focus on `csa_translate_pack`, common decode index writes/reuse, HCA fused
     index path, and Python/H2D metadata prep.
   - Avoid drawing conclusions from tiny standalone kernels unless they are
     confirmed in `benchmarkvllm.sh`.

3. Improve the ATOM attention/compressor path without touching model runner
   first.
   - Prefer metadata-cache/layout fixes inside DeepSeek-V4 ROCm model code.
   - Keep generic KV-cache changes contract-based and covered by tests.
   - Do not move request lifecycle into model code.

4. Revisit mixed FP8 compressed tail only after understanding the current
   split-layout cost.
   - Mixed KV is important for ATOM parity and capacity.
   - It is currently slower, so first isolate whether the loss comes from scale
     loads, split-load wrappers, metadata, or compressor writes.

5. Leave MHC and aux streams as later-stage work.
   - MHC is relevant for full ATOM parity, but it has repeatedly failed
     accuracy or regressed performance in this branch.
   - Aux stream overlap should wait until the single-stream ATOM op sequence is
     stable and above V1.

## Useful One-Liners

Summarize selected benchmark JSON files:

```bash
python3 - <<'PY'
import glob, json, os
for path in sorted(glob.glob('/app/atomdsv4/bench-sparsemla/*C32.json')):
    try:
        data = json.load(open(path))
    except Exception:
        continue
    name = os.path.basename(path)
    out = data.get('output_throughput')
    total = data.get('total_token_throughput')
    tpot = data.get('mean_tpot_ms')
    done = data.get('completed')
    failed = data.get('failed')
    if out is not None:
        print(f'{name}: output={out:.2f} total={total:.2f} tpot={tpot:.2f} completed={done} failed={failed}')
PY
```

Check whether ATOM is imported as a dependency:

```bash
pytest -q tests/kernels/test_deepseek_v4_atom_dependency_contract.py::test_vllm_runtime_does_not_import_atom_package
```

Check op-surface audit:

```bash
pytest -q tests/kernels/test_deepseek_v4_atom_op_surface_audit.py
```

Check KV-cache contract tests touched by the current design:

```bash
pytest -q \
  tests/v1/core/test_kv_cache_utils.py \
  tests/v1/test_kv_cache_spec_registry.py \
  tests/v1/worker/test_utils.py \
  -k 'atom or deepseek_v4 or kv_cache_spec or post_bind or reshape'
```

## Key Technical Interpretation

Async scheduling is part of the intended V2 path and should not be disabled for
accepted results. The current gap is not "V2 cannot run ATOM kernels"; it does
run ATOM attention/compressor pieces and beats current async V2 native. The
remaining gap to V1 appears to be overhead around the ATOM kernels: metadata
preparation, top-k/index translation, and vLLM paged-layout adaptation.

MHC is not structurally required to make ATOM attention/compressor kernels work
in vLLM. The MHC fast paths were attractive for throughput but have accuracy
risk in this branch, so focus on ATOM sparse attention/compressor/indexer
metadata first.

The next successful step is likely not a wholesale scheduler rewrite. It is
more likely a small reduction in the layout/metadata/translation overhead around
the already-integrated ATOM attention/compressor path, followed by unchanged
GSM8K validation.
