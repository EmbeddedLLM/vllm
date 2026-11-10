# PR #62004 (ReplaySSM for Gated DeltaNet standard decode) on ROCm / MI355X (gfx950)

Assessment of upstream PR **#62004** (`replayssm-gdn-standard`, head `3c8678368a`
"Merge main into replayssm-gdn-standard") applied on top of vLLM main
`73864f8360`, on **AMD Instinct MI355X (gfx950)**, torch `2.13.0+rocm10.0.0`
(HIP 7.15), GPUs 4/5 only.

* Worktree: `/app/local/yh/replayssm-gdn-wt`, branch `replayssm-gdn-rocm`
  (merge commit `ff8d623706`, one follow-up merge fix `d4899d8f5a`).
* Artifacts: `bench/` (harnesses), `logs/` (raw logs + json/csv), this file.
* Nothing was rebuilt or pip-installed: the compiled extensions from the main
  editable install were copied into the worktree
  (`cp /app/local/yh/vllm/vllm/*.so /app/local/yh/replayssm-gdn-wt/vllm/`) and
  every command ran with `PYTHONPATH=/app/local/yh/replayssm-gdn-wt`
  (`import vllm` verified to resolve to
  `/app/local/yh/replayssm-gdn-wt/vllm/__init__.py`).

## Headline

| | result |
|---|---|
| Milestone 1 (numerics vs baseline kernel on gfx950) | **PASS** - 80/80, 12/12, 82/82 tests. No failures, no platform skips. |
| Milestone 2 (kernel latency, CUDA-graph captured) | **PASS with caveats** - ReplaySSM beats the baseline only from batch >= 128; **1.63x** at bs 512 (bf16 model caches, best config), **1.49x** at bs 512 (fp32 caches). At bs <= 32 it is *slower* (0.68-0.99x). The shipped non-Blackwell config `(block_v=64, warps=1, stages=3, nk=2)` is badly tuned for gfx950: `(16,1,1,2)` is 1.55x/1.45x faster than it, and that swing is the whole difference between "no win" and "1.5-1.6x win". |
| Milestone 3 (e2e) | **DOES NOT START** on main. Exact error: `AttributeError: 'QwenGatedDeltaNetAttention' object has no attribute '_replayssm_ring_start'` raised from MRV2's `MambaHybridModelState.initialize_kv_cache`. |
| Kernel portability | The kernel itself is plain Triton and compiles/runs correctly on HIP. Both the fp16 and fp32 variants lower to real MFMA matrix ops on gfx950 (`v_mfma_f32_32x32x16_f16`, `v_mfma_f32_16x16x16_f16`, `v_mfma_f32_32x32x2_f32`). |

---

## 1. Merge / rebase notes

`git merge pr62004` produced **exactly one conflicted file**,
`tests/v1/e2e/test_replayssm_decode.py` (3 conflict blocks). Resolution kept
**main's structure** everywhere:

* conflict 1 (FlashInfer availability gates + model-name constants): took main,
  added the PR's `GDN_MODEL = "Qwen/Qwen3.5-4B"` next to `MAMBA2_MODEL`.
* conflict 2 (TP2 parity test): kept main's `monkeypatch.setenv(
  "VLLM_WORKER_MULTIPROC_METHOD", "spawn"); envs.disable_envs_cache()` body,
  added `GDN_MODEL` to the parametrize list.
* conflict 3 (prefix-cache tests): took main entirely. The PR side calls a
  helper (`_check_replayssm_prefix_caching_parity`) that main renamed, so the PR
  hunk was stale.

Everything else auto-merged. Beyond that conflict, three merge problems had to
be handled - two *silent* losses (the PR is behind main, so git took main's side
without reporting a conflict) and one auto-merged duplicate keyword:

1. **`CommonAttentionMetadata.replayssm_decode_base_cpu` disappeared.** PR commit
   adds the field; main **deleted** it in `cf31617992` *"[Mamba][MRV2] Plan
   Triton ReplaySSM decode on the GPU and support Model Runner V2 (#65033)"*.
   Because the merge base already contained the field, git treated it as "ours
   deleted it" and dropped it silently - which broke the PR's GDN decode
   planner (`vllm/v1/attention/backends/gdn_attn.py:673`).
   **Merge fix `d4899d8f5a`** restores the 6 lines (the field + its
   `maybe_slice_reqs` slice) in `vllm/v1/attention/backend.py` so the GDN
   builder exists at all. This is a local merge resolution, *not* a proposed
   upstream change - see "what does not work" for why main removed it.
   Verified non-invasive: `tests/v1/attention/test_replayssm_metadata_builder.py`
   (main's Mamba2 ReplaySSM builder) is 3 passed / 1 skipped (skip =
   `flashinfer` not installed).
2. **A duplicate keyword in the e2e test.** `_check_replayssm_prefix_caching`
   ended up with both main's `mamba_backend=mamba_backend` parameter and the
   PR's `mamba_backend="triton"` in the same `dict(...)` call (auto-merged, no
   conflict) - `ruff` reports `invalid-syntax: Duplicate keyword argument
   mamba_backend` and the call would `TypeError` at import. **Merge fix
   `54cb743bbf`** drops the PR's hard-coded key; main's parameter already lets
   every caller pass `mamba_backend="triton"` (which the Triton prefix-cache
   test does), so no behaviour is lost. `pre-commit run --files` on the merged
   files is clean afterwards (ruff, ruff-format, mypy, and the rest).
3. The PR's runner-side producers of that field
   (`vllm/v1/worker/gpu_model_runner.py`, `vllm/v1/worker/gpu_input_batch.py`)
   no longer exist in main, so **nothing in the tree fills the field** any more.
   That is milestone 3's blocker.

Compiled extensions: `.so` files copied from the main install; note
`vllm/__init__.py` prints `RuntimeWarning: Failed to read commit hash: No module
named 'vllm._version'` in the worktree - cosmetic (a source-tree-only install
artifact), it does not affect imports.

---

## 2. Milestone 1 - numerical correctness on gfx950

All three suites were run on `CUDA_VISIBLE_DEVICES=4` (MI355X, gfx950).

| suite | result | notes |
|---|---|---|
| `tests/kernels/test_replayssm_standard_decode_gdn.py -x -q` | **80 passed** in 71s | the full matrix ran: 6 precision pairs x 3 geometries (incl. Qwen3.5-4B `16/32/128/128` and `16/64/128/128`) x L in {4,16} x strided/contiguous = 72, plus 6 per-row-`write_pos` and 2 flush-beta tests. **0 skipped.** |
| `tests/kernels/test_replayssm_teacher_decode_equivalence_gdn.py -x -q` | **12 passed** in 102s | chunked-prefill teacher vs step decode vs ReplaySSM decode; fp32 and bf16 state, bf16 activations. **0 skipped.** |
| `tests/v1/attention/test_gdn_metadata_builder.py -q` (CPU) | **82 passed** in 48s | **only after the merge fix above**; before it, every ReplaySSM case died with `TypeError: CommonAttentionMetadata.__init__() got an unexpected keyword argument 'replayssm_decode_base_cpu'` (rebase artifact, category (c) - not a numerics issue). |

All three were re-run on the final tree (after the last merge fix) and gave the
same 80 / 12 / 82 (`logs/m1_final_tree_rerun.log`).

No test carries a platform gate that had to be lifted: the gates in these files
are `not torch.cuda.is_available()`, which is `True` on ROCm, so nothing was
skipped and nothing had to be un-skipped.

### Error magnitudes (independent float64 reference)

`bench/num_error_magnitudes.py` drives 33 decode steps (two flushes, L=16) at the
Qwen3.5-4B geometry (H=16, HV=32, K=V=128, batch 4, sparse state slots) through
both kernels and compares each to a **float64 recurrence** (not to each other),
so "who is more accurate" is answerable. The script aborts if the float64
reference fails to match the fp32 sequential baseline to 1e-3.
Raw: `logs/num_error_magnitudes.json`.

| state / act | d,k cache | out max abs err vs fp64: packed | out: replayssm | out: replayssm vs packed | state max abs err at flush: packed | state: replayssm |
|---|---|---|---|---|---|---|
| fp32 / fp32 | fp32 | 4.5e-08 | 5.7e-08 | 5.5e-08 | 1.5e-07 | 1.4e-06 |
| fp32 / bf16 | fp16 | 8.6e-04 | 8.6e-04 | 4.9e-04 | 1.8e-07 | 5.2e-04 |
| bf16 / bf16 | fp16 | 7.9e-04 | 7.2e-04 | 9.8e-04 | 6.0e-03 | 2.1e-03 |
| fp32 / fp16 | fp16 | 6.6e-05 | 6.6e-05 | 1.2e-04 | 1.8e-07 | 3.5e-04 |
| fp16 / fp16 | fp16 | 8.9e-05 | 8.8e-05 | 1.2e-04 | 7.3e-04 | 3.9e-04 |
| fp16 / bf16 | fp16 | 8.8e-04 | 8.8e-04 | 9.8e-04 | 7.4e-04 | 5.5e-04 |

Reading:

* **No ROCm-specific numerical defect.** In the production config (fp32 state,
  bf16 activations, fp16 ring caches) ReplaySSM is within **8.6e-4** of the
  float64 reference on the output - the *same* 8.6e-4 as the baseline kernel, so
  that error is the fp16 ring-storage rounding both kernels share, not a
  reconstruction bug. State at flush: 5.2e-4 (replayssm) vs 1.8e-7 (packed) -
  the expected cost of rounding the state once per flush instead of every token.
* ReplaySSM is *more* accurate than the baseline for reduced-precision state
  (bf16 state: 2.1e-3 vs 6.0e-3), matching the PR's premise.
* Relative error is not a useful metric here: several outputs are ~0, giving
  meaningless ratios up to 3e23 in `out_max_rel_err_replayssm_vs_packed`. Use
  the absolute errors.

### One precision note worth carrying upstream

The `tl.dot` operands are the ring-cache dtypes, so an **fp32** cache means an
fp32 dot. On gfx950 that lowers to `v_mfma_f32_32x32x2_f32` - an IEEE-fp32
matrix core (`logs/mfma_probe.log`) - and the measured error confirms it
(5.7e-08, i.e. no truncation). On NVIDIA the same dot defaults to **tf32**
(10-bit mantissa) unless the caller passes `input_precision="ieee"`, so an
fp32-cache ReplaySSM runs *more accurately* here than it would on CUDA - at the
(fp32 matrix core, not tf32) MAC rate, which is why the fp32-cache column in
3.1/3.2 is the slower of the two. The PR has no `input_precision` argument, so
this platform asymmetry is invisible in the source. (For the bf16-model case
the dot is fp16 x fp16 and lowers to `v_mfma_f32_32x32x16_f16` at the shipped
config and `v_mfma_f32_16x16x16_f16` at the tuned one.)

---

## 3. Milestone 2 - kernel latency on gfx950

Harness: `bench/bench_gdn_replayssm.py`. Qwen3.5-4B GDN geometry (H=16, HV=32,
K=V=128), L=16, fp32 state, sparse state slots (random permutation, slot 0
null), per-row `write_pos` staggered over `0..L-2` with **one flush row per L**
(`is_flush=1`, `write_pos=L-1`) so flush work is charged at its steady-state
1/16 rate. Each variant is captured in a `torch.cuda.CUDAGraph` and timed by
replaying it: >= 20 warmup replays, 100 timed replays, 3 trials, min reported.
Eager launching is CPU-bound here (`bench/eager_launch_overhead.py`,
`logs/eager_launch_overhead.log`: 26-34 us of
host time per call regardless of batch), so graph timing is the only meaningful
measure. The harness aborts a measurement if the host launch loop outruns the
GPU. Two full runs per table; cells agree to <2.2% except one outlier noted
below.

### 3.1 Shipped default launch config `(block_v=64, warps=1, stages=3, nk=2)`

`logs/bench_gdn_shipped2.*` (run 2; run 1 = `bench_gdn_shipped.*`).

| batch | (i) bf16 model: fp16 d/k + fp32 g | | | (ii) fp32 d/k + fp32 g | | |
|---|---|---|---|---|---|---|
| | packed_decode ms/step | replayssm ms/step | speedup | packed_decode ms/step | replayssm ms/step | speedup |
| 1 | 0.0057 | 0.0080 | **0.71x** | 0.0057 | 0.0117 | **0.49x** |
| 8 | 0.0083 | 0.0095 | **0.87x** | 0.0085 | 0.0134 | **0.64x** |
| 32 | 0.0239 | 0.0245 | 0.98x | 0.0243 | 0.0274 | 0.89x |
| 128 | 0.0845 | 0.0810 | 1.04x | 0.0885 | 0.0908 | 0.98x |
| 256 | 0.2144 | 0.2032 | 1.06x | 0.2207 | 0.2122 | 1.04x |
| 512 | 0.4336 | 0.4112 | 1.05x | 0.4316 | 0.4222 | 1.02x |

speedup = packed_decode / replayssm (derived). At bs 1-8 a kernel is 5-13 us,
i.e. essentially at the launch floor, so those ratios are dominated by fixed
cost. **With the shipped config the ROCm win is only ~1.05x.**

> Run-to-run outlier: in run 1 the `bs=1, bf16-model, packed_decode` cell
> measured 0.196 ms (trials 0.248/0.196/0.204) instead of 0.0057 ms; three
> isolated re-runs gave 0.0054-0.0069 ms, so run 1's value is external
> interference (shared GPU), not a kernel property. Every other cell matched
> run 2 within 2.2%.

### 3.2 Best launch config found, `(block_v=16, warps=1, stages=1, nk=2)`

Same harness, override via `replayssm_config._overrides["gdn_decode"]`
(`logs/bench_gdn_bestcfg2.*`).

| batch | (i) bf16 model: fp16 d/k | | | (ii) fp32 d/k | | |
|---|---|---|---|---|---|---|
| | packed_decode ms/step | replayssm ms/step | speedup | packed_decode ms/step | replayssm ms/step | speedup |
| 1 | 0.0059 | 0.0057 | 1.03x | 0.0062 | 0.0066 | 0.95x |
| 8 | 0.0083 | 0.0082 | 1.02x | 0.0086 | 0.0099 | 0.87x |
| 32 | 0.0241 | 0.0195 | 1.23x | 0.0243 | 0.0227 | 1.07x |
| 128 | 0.0829 | 0.0557 | 1.49x | 0.0882 | 0.0677 | 1.30x |
| 256 | 0.2142 | 0.1319 | **1.62x** | 0.2207 | 0.1453 | **1.52x** |
| 512 | 0.4336 | 0.2659 | **1.63x** | 0.4320 | 0.2901 | **1.49x** |

### 3.3 Launch-config sweep (bs 256 and 512)

Coordinate descent from the shipped default, `block_v {32,64,128}` x
`num_warps {1,2,4}` x `num_stages {1,2,3}` x `nk {1,2,4}`, 3 rounds (round 2+
also probes `block_v=16`, which is the minimum `tl.dot` tile). 18 distinct
configs measured per cache; raw `logs/bench_gdn_sweep.{json,csv,log}`.

Best per cache (both batch sizes must agree), against the shipped default
measured in the same process:

| cache | best config | bs 256 ms | bs 512 ms | shipped (64,1,3,2) bs 256 | bs 512 | gain vs shipped |
|---|---|---|---|---|---|---|
| (i) fp16 d/k + fp32 g | `(16,1,1,2)` | 0.1296 | 0.2627 | 0.2005 | 0.4067 | **1.55x / 1.55x** |
| (ii) fp32 d/k | `(16,1,1,2)` | 0.1436 | 0.2901 | 0.2104 | 0.4199 | **1.47x / 1.45x** |

Best **within the requested axis set** `block_v {32,64,128}` (no `16`):
`(32,1,1,2)` for both caches - 0.1531/0.3044 ms (fp16 d/k) and
0.1660/0.3268 ms (fp32 d/k), i.e. 1.31x/1.34x and 1.27x/1.28x over shipped.
The sweep converged after round 1 and did not move in rounds 2-3.

Full measured grid (ms/step, bs 256 / bs 512), best first:

```
(i) fp16 d/k + fp32 g                        (ii) fp32 d/k
bv  nw ns nk    bs256   bs512               bv  nw ns nk    bs256   bs512
16  1  1  2    0.1296  0.2627               16  1  1  2    0.1436  0.2901
16  1  1  1    0.1305  0.2627               16  1  1  1    0.1458  0.2933
16  1  2  2    0.1322  0.2641               16  1  2  2    0.1565  0.3145
16  1  3  2    0.1324  0.2669               16  1  1  4    0.1623  0.3221
16  1  1  4    0.1525  0.3013               32  1  1  2    0.1660  0.3268
32  1  1  2    0.1531  0.3044               32  1  1  1    0.1679  0.3311
32  1  1  1    0.1538  0.3057               32  1  2  2    0.1727  0.3417
32  1  3  2    0.1551  0.3063               32  1  3  2    0.1789  0.3541
32  1  2  2    0.1551  0.3086               16  1  3  2    0.1805  0.3615
16  2  1  2    0.1601  0.3122               32  1  1  4    0.1918  0.3793
32  2  3  2    0.1804  0.3558               16  2  1  2    0.1926  0.3815
32  1  1  4    0.1819  0.3590               64  1  1  2    0.1975  0.3914
64  1  1  2    0.1836  0.3640               32  2  3  2    0.2112  0.4188
128 1  1  2    0.1870  0.3712               64  1  3  2    0.2104  0.4203  <- shipped
128 1  3  2    0.1981  0.4018               16  4  1  2    0.2277  0.4498
64  1  3  2    0.2009  0.4070  <- shipped    128 1  3  2    0.2970  0.5859
16  4  1  2    0.2152  0.4246               128 1  1  2    0.3290  0.6617
32  4  3  2    0.3875  0.7692               32  4  3  2    0.6028  1.2146
```

Observations: smaller `block_v` (finer V split, 8 programs per (batch, head)
instead of 2) wins monotonically; `num_stages=1` beats 3 (the ring loads are
masked streaming loads, no pipelining benefit, and `BC` is small); `num_warps>1`
always loses; `nk=2` (BKT=64) is the best K-tile split, `nk=1` ties for the
fp16 case and loses for fp32. `block_v` also changes the MFMA shape
(`32x32x16_f16` at bv=64 vs `16x16x16_f16` at bv=16), which is why the launch
config matters far more than the 5-10% such a sweep moves on CUDA.

Caveat: single-process, single-GPU, one geometry (Qwen3.5-4B), `L=16`, batch
`<= 512`, and only one flush pattern (1 flush row per L). Numbers are derived
from `min` of 3x100 replays; two full repetitions agree within ~2% except the
noted outlier.

---

## 4. Milestone 3 - end-to-end (attempted, capped)

The Qwen3.5-4B snapshot in the local HF cache has config + tokenizer but **no
weights** (29 KB), so the probe used **Qwen3.5-0.8B** (1.7 GB, real weights,
same GDN hybrid family: 24 layers, 16/16 heads x K=V=128), `mamba_backend=triton`,
`mamba_ssm_cache_dtype=float32`, `--replayssm-buffer-len 16`, greedy, 2 prompts
(`bench/e2e_replayssm_gdn.py`).

* Baseline: **works** (`logs/m3_baseline.log`) - both prompts decode.
* `use_replayssm=True`: **fails at KV-cache initialisation**
  (`logs/m3_replayssm.log`):

```
File "vllm/v1/worker/gpu/model_states/mamba_hybrid.py", line 158, in initialize_kv_cache
    initialize_replayssm_ring_trackers(
File "vllm/v1/worker/replayssm_utils.py", line 37, in initialize_replayssm_ring_trackers
    layers[0]._replayssm_ring_start,
AttributeError: 'QwenGatedDeltaNetAttention' object has no attribute '_replayssm_ring_start'
RuntimeError: Engine core initialization failed.
```

Not attempted beyond this point (per the 15-minute cap): the engine cannot
reach a decode step, so there is no e2e speedup number on ROCm - and none can
exist until the PR is rebased onto main's MRV2 ReplaySSM plumbing.

---

## 5. What works / what does not / what is untested

### Works
* The new kernel `fused_recurrent_gated_delta_rule_replayssm` (pure Triton,
  `tl.cumsum/tl.sum/tl.sigmoid/tl.exp/tl.log` + one `tl.dot`) **compiles and
  runs correctly on gfx950**. No HIP-specific correctness bug found: 174 tests
  (80 kernel-matrix + 12 teacher-equivalence + 82 metadata), 0 failures,
  0 platform skips, and errors vs a float64 reference equal to the baseline's.
* Output/checkpoint numerics across all six precision pairs, L=4/16,
  strided/contiguous, per-row `write_pos`, sparse/padded state slots, and the
  chunked-prefill teacher comparison.
* The GDN host-side decode planner (`GDNAttentionMetadataBuilder`): 82 cases
  pass once main's removed metadata field is restored.
* Kernel-level speedup is real but batch-gated: **1.62-1.63x** at bs 256/512
  (bf16 model caches), **1.49-1.52x** (fp32 caches) at the best config;
  1.27-1.34x with the requested `block_v {32,64,128}` axis set; only ~1.05x with
  the config the PR actually ships for non-Blackwell.

### Does not work
1. **E2E / serving on ROCm with the current main.** Two independent blockers:
   * main deleted `CommonAttentionMetadata.replayssm_decode_base_cpu` in
     #65033 (MRV2 + GPU-planned ReplaySSM) *and deleted MRV1*
     (`vllm/v1/worker/gpu_model_runner.py`, `gpu_input_batch.py` and
     `VLLM_USE_V2_MODEL_RUNNER` are all gone). The PR's GDN decode path is
     host-planned: it needs that CPU field per decode row to derive
     `write_pos`/`is_flush`, so on main nothing fills it - the builder raises
     `ValueError("--use-replayssm requires exact CPU sequence lengths and
     decode-base counts ...")` even if it were reachable.
   * MRV2's `MambaHybridModelState.initialize_kv_cache` unconditionally calls
     `initialize_replayssm_ring_trackers` when `use_replayssm`, which assumes
     Mamba2 mixers (`_replayssm_ring_start`, `_plans_replayssm_decode`, GPU
     planning). The GDN layer sets `use_replayssm` (PR: `gdn/base.py`,
     `qwen_gdn_linear_attn.py`) but has none of those attributes, so init dies
     with the `AttributeError` above.
   * Consequence for the PR body's "requires V1": that requirement is now
     unmeetable - main has no V1 model runner. Rebasing this PR means porting
     the GDN `write_pos`/`is_flush` derivation onto main's GPU plan kernel
     (`vllm/model_executor/layers/mamba/ops/ssu_dispatch.py::_plan_replayssm_decode_kernel`)
     or adding a GDN variant of it. **Not attempted here** (out of scope by
     instruction).
2. **Shipped non-Blackwell launch config is wrong for gfx950**: `(64,1,3,2)` is
   1.45-1.55x slower than `(16,1,1,2)` at serving batch sizes, which is the
   difference between ~1.05x and ~1.6x over the baseline. `_is_blackwell()` is
   `False` on ROCm, so ROCm currently inherits a CUDA-shaped default.
3. Small-batch decode (bs <= 32) is *slower* than the baseline (0.49-0.98x):
   state reconstruction costs more than the per-step full-state store when
   there are few rows. ReplaySSM is a large-batch optimization.

### Untested
* e2e decode throughput/quality on ROCm (blocked, see above).
* Preemption, prefix caching / `mamba_cache_mode="align"`, speculative decode,
  TP2 and pipeline parallelism for the GDN path - the kernel tests cover
  decode-step numerics only, and the e2e tests that would cover the rest need a
  working engine (and, for the FlashInfer variants, `flashinfer`, which is not
  installed here).
* `L` other than 4/16, geometries other than the three in the test matrix, batch
  > 512, and non-uniform flush patterns beyond the 1-flush-per-L model.
* Correctness of the *serving* `write_pos`/`is_flush` derivation under
  continuous batching (only its unit-level planner is tested).
* Other AMD architectures (only gfx950 tested).

---

## 6. Reproduction

```bash
cd /app/local/yh/replayssm-gdn-wt            # branch replayssm-gdn-rocm
export PYTHONPATH=/app/local/yh/replayssm-gdn-wt
export CUDA_VISIBLE_DEVICES=4

# milestone 1
python3 -m pytest tests/kernels/test_replayssm_standard_decode_gdn.py -x -q
python3 -m pytest tests/kernels/test_replayssm_teacher_decode_equivalence_gdn.py -x -q
python3 -m pytest tests/v1/attention/test_gdn_metadata_builder.py -q

# milestone 1 error magnitudes
cd /app/local/yh/issue-workdirs/pr62004-replayssm-gdn-rocm/bench
python3 num_error_magnitudes.py --json ../logs/num_error_magnitudes.json

# milestone 2
python3 bench_gdn_replayssm.py --json ../logs/bench_gdn_shipped2.json --csv ../logs/bench_gdn_shipped2.csv
python3 bench_gdn_replayssm.py --config 16 1 1 2 --json ../logs/bench_gdn_bestcfg2.json --csv ../logs/bench_gdn_bestcfg2.csv
python3 bench_gdn_replayssm.py --sweep-only --sweep-config --sweep-rounds 3 \
    --sweep-batches 256 512 --json ../logs/bench_gdn_sweep.json --csv ../logs/bench_gdn_sweep.csv

# milestone 3
python3 e2e_replayssm_gdn.py --mode baseline
python3 e2e_replayssm_gdn.py --mode replayssm
```

Artifact index: `logs/bench_gdn_shipped{,2}.*`, `logs/bench_gdn_bestcfg{,2}.*`,
`logs/bench_gdn_sweep.*`, `logs/num_error_magnitudes.json`,
`logs/mfma_probe.log`, `logs/eager_launch_overhead.log`,
`logs/m1_test{1,2,3}_*.log`, `logs/m1_extra_mamba2_builder.log`,
`logs/m3_{baseline,replayssm}.log`.
