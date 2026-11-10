# SketchSSM PR #61124 - review + ROCm (MI355X / gfx950) feasibility

Workdir for https://github.com/vllm-project/vllm/pull/61124
("[SketchSSM] Add SketchSSM decode for Mamba-2 with Triton kernels").

Paper: *SketchSSM: Write to the Full State, Read from a Compact Sketch*,
Omin Kwon, JoongWon Shin, Minseo Kim, Kurt Keutzer, Sehoon Kim, Jae W. Lee -
arXiv:2609.33051 (v1 2026-09-27, v2 2026-10-06), NeurIPS 2026 LCFM workshop.
Project page: https://omin-kwon.github.io/project/SketchSSM/
Code: https://github.com/SNU-ARC/SketchSSM (fork of vLLM 0.30.0), PyPI `sketchssm`
(CUDA kernels), calibrations: https://huggingface.co/SketchSSM

Environment used for everything below: 8x AMD Instinct MI355X (gfx950),
ROCm 7.15, torch 2.13.0+rocm10.0.0, Triton 3.8.0, vLLM main @ 73864f8360.

---

## 1. What the PR does

Per request, decode runs in windows of W=16 steps (ReplaySSM's
`--replayssm-buffer-len`):

* **flush step** (1 in W): fold the window into the exact FP32 state, read it,
  rebuild a per-request sketch `U = S^T Omega` (leading rows of the state in a
  calibrated orthogonal frame per group) plus a coefficient map, in the same
  kernel.
* **other steps**: `S^T q` is replaced by `U^T c` where
  `c = C q` is rebuilt from the query. The window contribution (ring of x/dt/B)
  stays exact.

So writes stay exact (ReplaySSM's flush) and only *reads* are approximated.
Offline calibration gives the per-head frames `Omega` and per-head ranks
(`--sketchssm-mean-rank`, default 8, per-head allocation is a multiple-choice
knapsack over calibrated error curves).

New files: 2 Triton kernel modules (~1500 LOC), sketch layout/build, calibration
loader + rank allocation, a `--sketchssm` CLI, NemotronH wiring, a new CI job,
docs, one kernel test file, one e2e test, one benchmark.

---

## 2. ROCm verdict (short version)

**Not usable as-is, but only two one-line things stand in the way, and both are
ROCm-specific accidents rather than design problems.** After them, the whole
Mamba-2 kernel path is *numerically correct and already faster than the stock
decode* on gfx950.

Blocker 1 - config gate (`vllm/config/vllm.py::_validate_sketchssm`):

```python
if not current_platform.is_cuda() or not (current_platform.has_device_capability(80)):
    raise ValueError("--sketchssm requires a CUDA GPU with compute capability 8.0 or newer")
```

On ROCm `is_cuda()` is False (so it raises), while `has_device_capability(80)`
is *True* because gfx950 reports capability (9, 5) - i.e. the "compute
capability 8.0" clause is meaningless here and the real requirement is nothing
of the sort (the feature is 100% Triton; the sibling ReplaySSM feature has no
platform gate at all).

Blocker 2 - dot precision (`ops/sketchssm_mamba2_build.py`):

```python
SKETCH_DOT_PRECISION = "tf32x3"
```

The HIP backend's allowed dot input precisions are `("ieee", "bf16x3",
"bf16x6")` (`triton/backends/amd/compiler.py:93`); `tf32x3` and `tf32` only
compile on gfx942. Measured on gfx950 with a minimal kernel:

```
prec=ieee    cm=''     OK   relerr=3.7e-07
prec=ieee    cm='.cg'  OK   relerr=3.7e-07   <- cache_modifier is fine on ROCm
prec=bf16x3  cm=''     OK   relerr=4.9e-06
prec=tf32x3  cm=''     FAIL CompilationError
prec=tf32    cm=''     FAIL CompilationError
```

`selective_state_update_replayssm_output_only.py:586` already handles this with
`dot_input_precision = None if current_platform.is_rocm() else "tf32x3"`, so the
repository already has the convention; SketchSSM just needs to follow it.

Patch used here (2 files, 6 lines):

* `SKETCH_DOT_PRECISION = None if current_platform.is_rocm() else "tf32x3"`
  (None -> backend default = `ieee`), plus an env override used only for the
  bf16x3 measurement.
* the kernel test's `skipif(not current_platform.is_cuda())` ->
  `is_cuda_alike()`.

---

## 3. What was verified on gfx950

### 3.1 Kernel correctness (vs FP64 references)

```
$ cd /app/local/yh/vllm && python3 -m pytest tests/kernels/mamba/test_sketchssm_mamba2.py -q
2 passed
```

These tests are strong: a full W+1 window loop, rows at mixed ring positions,
dense (rank 0) + sketch heads (ranks 1,2,3,4,5,9,20,38,64), flush, sketch
rebuild after flush, padding rows, and the prefill rotation and its inverse -
all against FP64 references, with the state checked exactly at every step.
They pass unchanged on ROCm with only the precision port.

`tl.debug_barrier()` + `.cg` (used to rebuild the sketch from state the same
program just wrote inside the flush kernel) work on gfx950.

### 3.2 Module-level path, incl. the aux (multi-)stream

`Mamba2SketchSSM.decode()` runs the flush rows on an auxiliary stream
(`maybe_execute_in_parallel`, `aux_stream()` - both are `is_cuda_alike()`, so
ROCm is included). Not covered by the unit tests, so it was checked separately
(`smoke_module_rocm.py`): the multi-stream result is **bit-identical** to the
sequential one for both the outputs and the updated states, at batch 256,
H=96/P=80, mean rank 8. No missing event ordering on HIP.

### 3.3 Per-layer decode latency (the interesting part)

One Mamba-2 layer, FP32 state cache, W=16, mean sketch rank 8, DSTATE=128,
8 groups, 1 flush row in 16, **CUDA-graph captured** (the engine runs under
graphs; eager timing is CPU-launch-bound here - the sketch step is 6 Triton
launches and flat-lines at ~0.42 ms/step regardless of batch, while the stock
path scales with batch). `bench_graph_step.py`, raw numbers in
`bench_graph_step.json` / `bench_graph_bf16x3.json`.

Baseline = stock `selective_state_update` (vLLM's default Mamba-2 decode, full
state read + write every step). ms/step, MI355X, shipped configs (ieee dots):

| shape | batch | standard | SketchSSM (ieee) | speedup | SketchSSM (bf16x3) | speedup |
| --- | --- | --- | --- | --- | --- | --- |
| Nano-4B (H=96, P=80) | 128 | 0.326 | 0.169 | **1.94x** | 0.159 | **2.04x** |
| Nano-4B | 256 | 0.667 | 0.197 | **3.38x** | 0.187 | **3.56x** |
| Nano-4B | 512 | 1.323 | 0.281 | **4.70x** | 0.268 | **4.99x** |
| Nano-9B (H=128, P=80) | 128 | 0.418 | 0.210 | **2.00x** | 0.205 | **2.05x** |
| Nano-9B | 256 | 0.832 | 0.257 | **3.23x** | 0.251 | **3.33x** |
| Nano-9B | 512 | 1.671 | 0.344 | **4.86x** | 0.338 | **4.94x** |

Notes:
* The stock path scales ~linearly with batch (1.32 -> 1.67 ms at 512 rows for
  the two shapes): it is HBM-bound on the FP32 state read+write, as the paper
  says. SketchSSM does not scale with batch at these sizes (ring + sketch reads
  are ~8x smaller), so the win grows with batch.
* bf16x3 (BF16 MFMA on gfx950) buys only 4-6%: at serving batch sizes the
  kernels are memory-bound, not dot-bound, so the ROCm precision substitution
  costs ~nothing. Against the paper's B300 numbers (7.30x for Mamba-2 at mean
  rank 8) we are at ~4.9x, but that is a different model shape (Nemotron 3
  Super has much larger Mamba-2 layers), so the two are not directly comparable.
  At batch 128 we are latency-bound (~2x) - the sketch step is only 0.17 ms, so
  per-launch and small-kernel overheads dominate.
* Rough traffic check at batch 512 / 9B shape: the sketch step still reads
  ~0.5 GB/step, of which ~2/3 is the W=16 ring of x (FP32) that ReplaySSM
  already pays for. That is the remaining floor unless the ring is narrowed
  (e.g. BF16 x ring) - a ReplaySSM-level decision, not SketchSSM's.
* Fairness caveat: the stock baseline ran vLLM's untuned fallback launch config
  for `selective_state_update` (no `headdim=80,dstate=128` config exists for any
  device, so MI355X - and the paper's B300 - both use `BLOCK_SIZE_M=4,
  num_warps=8`). The comparison is therefore config-neutral, but a tuned stock
  config would shrink these ratios.

### 3.4 Launch-config tuning on gfx950 (the shipped configs are sm_120-tuned)

`ops/sketchssm_config.py` ships decode `(BLOCK_M, BLOCK_J, num_warps) =
(128, 16, 1)`, flush `(16, 4, 4, 32)`, build `(32, 2)`.

Decode coordinate sweep at batch 256/512 (`sweep_decode_gfx950.py`):

| decode cfg | batch 512 ms | speedup |
| --- | --- | --- |
| (128, 16, 1) shipped | 0.283 | 4.77x |
| (256, 16, 1) | 0.283 | 4.77x |
| (64, 16, 1) | 0.307 | 4.39x |
| (128, 8, 1) | 0.290 | 4.65x |
| (128, 16, 2) | 0.316 | 4.27x |
| (128, 16, 4) | 0.389 | 3.47x |
| (128, 16, 8) | 0.526 | 2.56x |

-> the shipped decode config is already right on gfx950 (BLOCK_M=256 ties);
`num_warps=1` (one 64-wide wave on CDNA4) is important, more warps hurt.

Flush sweep (`sweep_flush_gfx950.py`, batch 512, all-flush / mixed):

| flush cfg | all-flush ms | mixed ms |
| --- | --- | --- |
| (16, 4, 4, 32) shipped | 2.176 | 0.272 |
| (16, 1, 4, 32) | 1.914 (**-12%**) | 0.254 (**-7%**) |
| (16, 4, 16, 32) | 2.265 | 0.263 |
| (16, 4, 4, 16) | 2.637 (+21%) | 0.296 (+9%) |
| (16, 4, 4, 128) | 3.939 (+81%) | 0.392 (+44%) |

Profiling the flush kernel alone (`flush_cfg_small.py`, per-kernel times from
`torch.profiler`) shows why: on gfx950 `num_warps=4` costs almost 2x

| flush cfg | `_sketch_flush_kernel` at 2 flush rows | at 32 flush rows |
| --- | --- | --- |
| (16, 4, 4, 32) shipped | 167.6 us | 190.7 us |
| (16, 1, 4, 32) | **89.7 us** | **103.9 us** |
| (16, 1, 1, 32) | 94.1 us | 110.1 us |
| (32, 1, 4, 32) | 96.4 us | 109.8 us |

And the flush path has a large *fixed* cost that no row count amortises: 14.8 us
for 1 flush row, 167 us for 2 (per-kernel profile, batch 512 padded), then flat
up to ~32 rows (`flush_scaling.py`, `flush_grid_probe.py`, `flush_profile.py`).
In steady state a serving batch has `batch/W` flush rows *every* step (32 at
batch 512), so this is paid per step and is what caps the speedup at ~5x rather
than the ~13x an all-non-flush step shows (0.109 ms vs 1.32 ms standard).
`sketch_flush()` also launches `_sketch_flush_kernel` with
grid `(cdiv(padded_batch, rows_per_program), heads)` - 12288 programs at batch
512, all but `n_flush x heads` of which immediately exit; sizing that grid by the
flush-row count (or by an on-device count) would remove close to 1/2 of the
batch-dependent part (0.098 ms at padded batch 64 vs 0.175 ms at 512 for the same
2 flush rows).

Best config found on gfx950 (`num_warps=1` flush + `bf16x3` dots),
`bench_graph_best_gfx950.json`:

| shape | batch | standard | SketchSSM | speedup |
| --- | --- | --- | --- | --- |
| Nano-4B | 128 | 0.326 | 0.134 | **2.44x** |
| Nano-4B | 256 | 0.667 | 0.164 | **4.08x** |
| Nano-4B | 512 | 1.329 | 0.261 | **5.10x** |
| Nano-9B | 128 | 0.418 | 0.159 | **2.62x** |
| Nano-9B | 256 | 0.832 | 0.208 | **4.01x** |
| Nano-9B | 512 | 1.671 | 0.314 | **5.32x** |

That is +25% at batch 128 and +10% at batch 512 over the shipped configs, i.e.
the remaining tuning headroom is real but secondary: the two portable fixes
(section 2) plus a `num_warps=1` flush get most of it, and the biggest remaining
lever is the flush kernel's fixed cost.

All of 3.1-3.4 used the PR's own kernels, unmodified except the two-line
precision port; no CUDA-only construct had to be replaced.

---

## 4. What "usable on ROCm" still needs (same work as on CUDA)

The kernels are portable; the *engine* integration needs the rebase work the bot
is already asking for (the PR is ~2000 commits behind main):

1. `mamba_attn.py`: main replaced CPU-side ReplaySSM planning with a GPU planner
   (`plan_replayssm_decode`, planned inside the graph). The PR computes
   `write_pos_d`/`is_flush_d`/`sketch_indices_d`/`sketch_flush_rows_d` on the
   CPU with `.tolist()` + `torch.nonzero` + two `async_tensor_h2d` copies per
   step. A proper rebase should derive the sketch row list from the GPU plan
   (a small compaction kernel), which removes a per-step sync too.
2. `vllm/model_executor/warmup/mamba_triton_warmup.py` was deleted on main
   (JIT warmup moved); the `_warm_sketchssm` hook needs to move with it.
3. Then the 2-line ROCm patch from section 2 (gate + precision), plus
   `num_warps=1` for the flush on gfx950.
4. Model + calibration for a Mamba-2 hybrid: published calibrations exist for
   Nemotron Nano 9B v2, Nemotron 3 Super, Nemotron-3-Nano-4B (BF16), and for the
   GDN/KDA models (see below). No ROCm-specific calibration work is needed - the
   calibration is offline and precision-agnostic (it only produces frames and
   rank curves).

Nothing else in the feature is CUDA-specific: `torch.cuda.Event`,
`torch.cuda.Stream`, `aux_stream()`, `maybe_execute_in_parallel`,
`torch.get_default_device()` are all ROCm-safe (verified by running them).

---

## 5. GDN (Qwen3.5 / Qwen3.8 Flash-Next) - not in this PR

This PR is **Mamba-2 + NemotronH only**. GDN and KDA appear only as "PR
planned" in the reference repo, and the algorithm does extend to them: for GDN
and KDA the delta-rule erase is absorbed into the transition
(`M_t = (I - beta_t k_t k_t^T) D_t`, paper section 3.1), so the state read is
still `S_0^T q~_t` with an effective query. That costs more per rank
`K + V + W` instead of `K + V`, which is why the GDN/KDA rank tables depend on
the serving window `W`.

What exists today is in the fork (`SNU-ARC/SketchSSM`, vLLM 0.30.0):

* `vllm/model_executor/layers/mamba/ops/gdn_sketchssm_triton.py` (365 LOC) and
  `gdn_sketchssm_common.py` (364 LOC): fused GDN `step` + `flush` + `build`
  kernels (WY-based effective query, per-key-head rotation of q/k, beta ring).
* `kda_sketchssm_triton.py` (919 LOC) for GLM 5.3 Flash (KDA), plus
  `gdn_sketchssm.py` / `kda_sketchssm.py` module wrappers and
  `qwen_gdn_linear_attn.py` plumbing (state shape/dtype ring, rotation of
  prefill states, `sketchssm` metadata).
* ROCm handling is already partly there - their kernels take
  `DOT_PRECISION=None if current_platform.is_rocm() else "tf32x3"` - with one
  real bug: `gdn_sketchssm_common.py:140` (`_rotate_qk_kernel`) hardcodes
  `input_precision="tf32"`, which does **not** compile on gfx950 (gfx942 only).
  The KDA path passes `prec = None if rocm else "tf32x3"`, so it is fine.
* The CUDA-kernel package is gated by `envs.VLLM_SKETCHSSM_USE_CUDA and
  current_platform.is_cuda()`, so ROCm would fall back to Triton by design.

Published calibrations (HF `SketchSSM/*`), including GDN:

| model | arch | calibration |
| --- | --- | --- |
| Nemotron Nano 9B v2-BF16 | Mamba-2 | 11 MB |
| Nemotron 3 Super-NVFP4 | Mamba-2 | 13 MB |
| Nemotron-3-Nano-4B-BF16 | Mamba-2 | 8 MB (this PR's CI model) |
| Qwen3.8 Flash-Next-NVFP4 | GDN | 24 MB |
| Qwen3.5 9B-BF16 | GDN | 16 MB |
| GLM 5.3 Flash-NVFP4 | KDA | 68 MB |

So for "Qwen3.5 on AMD" the work is *not* ROCm work; it is upstreaming two
missing features:

1. GDN window rings (a ReplaySSM equivalent for GDN - does not exist upstream;
   `vllm/model_executor/layers/mamba/gdn/` has no replayssm path today),
2. the GDN sketch kernels + metadata + calibration wiring (~800-1500 LOC of
   Triton + attention-metadata plumbing, modelled on this PR),
3. plus the `tf32` -> `None if is_rocm()` fix and gfx950 configs.

Rough effort: Mamba-2 usable on ROCm ~1-2 days (rebase + the two-line patch +
flush warps + one accuracy run); GDN upstream port ~1-2 weeks by someone who
knows the GDN decode kernel, then the ROCm part is again the two-line class of
change.

---

## 6. PR review notes (tech, independent of ROCm)

Ordered by what would block or annoy a maintainer.

1. **Rebase is mandatory and non-trivial.** 2037 commits behind; the merge has
   conflicts in 7 files (I resolved them today to run the e2e attempt). The two
   substantive ones: (a) ReplaySSM per-step planning moved to the GPU
   (`plan_replayssm_decode`), which invalidates the PR's CPU-side planning
   approach; (b) `mamba_triton_warmup.py` no longer exists. Everything else is
   mechanical (`use_replayssm` refactors, test/config lists, buildkite yaml).
2. **`--sketchssm` is gated to CUDA + "compute capability >= 8.0"** although the
   feature is pure Triton and the sibling ReplaySSM has no gate. The CC check
   does not do what the message says on non-NVIDIA platforms. Use
   `is_cuda_alike()` or state the real requirement.
3. **`SKETCH_DOT_PRECISION = "tf32x3"` as a module constant** breaks every
   kernel on HIP; the codebase already has the `None if is_rocm()` convention in
   ReplaySSM. Also the kernel test skips on non-CUDA, so ROCm would never be
   covered - make it `is_cuda_alike()`.
4. **`ops/sketchssm_config.py` has no device detection** (hardcoded sm_120),
   unlike `replayssm_config.py` which auto-detects and documents an override
   path. Since the PR admits other GPUs are untuned, at least key on the device
   class; measurements here show the decode config does transfer and only the
   flush `num_warps` wants to differ, so this is cheap to get right.
5. **Per-step CPU work + H2D copies on the hot path** (`_sketch_decode_rows`:
   `.tolist()`, `torch.nonzero`, two `async_tensor_h2d`) - exactly what main
   just removed from ReplaySSM. Follow the GPU planner.
6. **Per-step allocations**: `sketch_flush` allocates a
   `(batch, groups, W, dstate)` FP32 `keys` buffer every step (16.8 MB at
   batch 256 for the 4B shape) although only `batch/W` rows flush;
   `sketch_query` allocates a `(batch, groups, dstate)` FP32 tensor per layer per
   step. Should be persistent scratch (flush rows x, padded batch).
6b. **The flush kernel is the steady-state bottleneck, and its grid is sized by
the padded batch.** In a serving batch every step has ~`batch/W` flush rows,
and on gfx950 the flush path costs 0.09-0.17 ms per step for *any* row count
>= 2 (1 row: 0.015 ms), i.e. it is paid every step and caps the per-layer
speedup at ~5x while a step with no flush rows runs 13x faster than the stock
decode (0.109 vs 1.41 ms at batch 512). Two concrete things measured here:
`num_warps=4 -> 1` nearly halves `_sketch_flush_kernel` (167.6 -> 89.7 us at
2 flush rows), and the launch grid `(cdiv(padded_batch, rows_per_program),
heads)` dispatches 12288 programs of which `n_flush x heads` do work (0.098 ms
at padded batch 64 vs 0.175 ms at 512 for the same 2 rows). Worth a profile on
B300 too - the paper's 7.30x window average suggests their flush is cheaper
relative to decode than what we see.
7. **`tl.debug_barrier()` + `.cg` is load-bearing** for the fused
   flush-and-rebuild (the sketch is built from the state the same program just
   wrote). It is correct in practice (verified bit-identical on gfx950), but it
   deserves a one-line comment on why no fence is needed (single workgroup,
   store-through to L2), because a future tiling change can silently break it.
8. **Two different views of the same buffer, undocumented at the call sites.**
   `sketch_build()` and `sketch_decode()/sketch_flush()` both take a
   `(slots, heads, dim, dstate)`-shaped view, while the module passes
   `ssm_state.transpose(-1, -2)` (the cache is key-major) and `sketch_query()`
   takes a per-layer `(groups, K, K)` `frames_t` while `SketchArgs.frames_t` is
   the same object. I got both wrong on the first try; assert the expected
   strides next to the existing `_check_inputs` asserts, or name the parameters
   `state_dim_major` / `frames_t_layer`.
9. **Naming drift**: `use_mamba_window_rings` (config) vs `use_window_rings`
   (local) vs `use_sketchssm` vs `use_replayssm`; and
   `replayssm_decode_base_cpu` on `MambaHybridAttnMetadata` now exists only for
   SketchSSM.
10. **Benchmark fairness**: the headline table's ReplaySSM row necessarily runs
    Model Runner V1 (the Triton ReplaySSM kernel needed V1 at that base) while
    standard and SketchSSM run V2, so "1.8x faster than ReplaySSM" mixes two
    runners; the 2.75x against the default decode is apples-to-apples. Main now
    supports the Triton ReplaySSM path under Model Runner V2
    (`v1/worker/gpu/model_states/mamba_hybrid.py`), so the ReplaySSM row can be
    re-measured on the same runner after the rebase.
11. **CI cost**: a new "(H100) SketchSSM E2E" job. The e2e test is two prompts x
    32 tokens of logprob parity; folding it into the existing Mamba e2e job as a
    parametrization would be cheaper. Also 32 tokens only exercises two windows;
    the accuracy claim that matters (long-generation drift) is untested there.
    A 4k-token generation comparison, or one RULER-style recall cell, would test
    the actual risk of the approximation.
12. **Small stuff**: `SketchTables` depends on `torch.get_default_device()`
    (works in the engine, surprising in a unit test - it silently produced CPU
    tables for me); `warmup()` fabricates a dummy row list to compile the build
    kernel (main now has JIT warmup infra); docs say "requires a CUDA GPU with
    compute capability 8.0" - if the intent is "NVIDIA-only for now because the
    configs are sm_120-tuned", say that instead of a hardware requirement.

Overall: the design is sound and conservative (exact updates, approximate reads,
opt-in, default path untouched, good FP64 kernel tests). The main risks are
integration-surface (rebased planner, per-step CPU work, per-step allocations)
and the hardcoded platform/precision/config assumptions, not the algorithm.

---

## 7. Files in this workdir

| file | what |
| --- | --- |
| `pr61124.diff` | the PR diff as fetched |
| `probe_triton.py` | Triton precision / cache-modifier probe on gfx950 (section 2) |
| `smoke_module_rocm.py` | module-level path incl. aux stream, sequential vs parallel bit-equality |
| `bench_graph_step.py`, `bench_graph_step.json`, `bench_graph_bf16x3.json` | CUDA-graph per-layer latency, ieee vs bf16x3 (section 3.3) |
| `sweep_decode_gfx950.py`, `sweep_decode_gfx950.json` | decode launch-config sweep (section 3.4) |
| `sweep_flush_gfx950.py`, `sweep_flush_gfx950.json` | flush launch-config sweep (section 3.4) |
| `quick_std_vs_sketch.py`, `quick_std_vs_sketch.json` | first (eager) timing - kept because it shows the CPU-launch-bound trap |
| `bench/`, `e2e/` | sub-agent benchmark + end-to-end attempt, see their own NOTES |

---

## 8. SketchSSM vs ReplaySSM on gfx950

Same layer/harness, `num_warps=1` flush, ieee dots, one Mamba-2 layer, batch
rows, 1 flush row in 16 (`bench_replayssm_vs_sketch.py`):

| batch | stock decode | ReplaySSM | SketchSSM | SketchSSM vs ReplaySSM |
| --- | --- | --- | --- | --- |
| 128 | 0.328 | 0.139 (2.36x) | 0.132 (2.48x) | 1.05x |
| 256 | 0.663 | 0.288 (2.30x) | 0.160 (4.16x) | **1.80x** |
| 512 | 1.350 | 0.534 (2.53x) | 0.244 (5.54x) | **2.19x** |

The PR's headline "1.8x faster than ReplaySSM" reproduces exactly at batch 256
on ROCm; it grows to 2.2x at batch 512 and collapses to 1.05x at batch 128,
where the flush path's fixed cost (~0.13-0.17 ms/step, section 3.4) is a large
share of a 0.13 ms step.

## 9. End-to-end on ROCm

A rebased copy of the PR (branch `sketchssm-rocm-e2e` in the worktree
`/app/local/yh/sketchssm-e2e-wt`) was run with two single-GPU servers:

```
vllm serve nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --trust-remote-code \
  --max-model-len 4096 --mamba-ssm-cache-dtype float32 \
  --enable-prefix-caching --mamba-cache-mode align          # baseline
  ... --sketchssm SketchSSM/Nemotron-3-Nano-4B-BF16 --sketchssm-mean-rank 8
```

with `VLLM_USE_V2_MODEL_RUNNER=1`, on gfx950, GPUs 6 (baseline) and 7
(SketchSSM). Both came up and served. The calibration loaded as published:

```
[calibration.py] SketchSSM calibration SketchSSM/Nemotron-3-Nano-4B-BF16:
mean rank 8 at window 16, 1926 sketch and 90 dense heads
[mamba2.py] SketchSSM sketch buffers: 4955.4 KiB per request over all 21
calibrated layers; each layer holds them for 1024 requests
```

(21 of the model's 30 Mamba-2 layers are calibrated; the rest run the default
decode. 4.8 MiB/request matches the PR description.)

Greedy generation, 8 prompts, 64 tokens, temperature 0, logprobs on
(`e2e/compare_mt64.json`, `e2e/compare_servers.py`):

* 5/8 prompts token-identical to the baseline;
* 3/8 diverge (greedy text), with the first logprob delta > 1e-3 at token 1-4;
* per-token logprob deltas on the matching prompts: 0.0-0.4 max, 0.009-0.03 mean.

That is the expected behaviour of an approximate state *read* (a rank-8 sketch
perturbs logits slightly and a greedy free-form continuation eventually flips),
not a ROCm bug: the kernel-level tests reproduce FP64 references, and the
sketch read is exact at flush steps by construction. It does mean the PR's e2e
guard (2 prompts x 32 tokens of `check_logprobs_close`) is thin - it would be
worth checking whether it is stable across seeds/prompts, since a 64-token run
here shows 3/8 free-form prompts diverging.

Caveat: this was one configuration (align mode, prefix caching on) on one box,
with an intermediate rebase; see `e2e/` for the logs. No throughput number was
produced end to end (the run stopped before the benchmark script).

Blockers hit during the rebase, for the record (all in the worktree, not in the
PR):
* `VllmConfig.use_v2_model_runner` no longer exists on main (moved to
  `ModelConfig`/parallel config), so `_validate_sketchssm`'s
  `if not self.use_v2_model_runner` had to be re-pointed;
* `mamba_triton_warmup.py` deleted on main -> the sketch warmup call dropped;
* ReplaySSM's metadata planning moved to `plan_replayssm_decode` (GPU), so the
  SketchSSM row lists were derived from the GPU plan instead of
  `replayssm_decode_base_cpu`.

### 9b. End-to-end model decode (offline `LLM`, no HTTP)

The PR's own benchmark, `benchmarks/sketchssm/e2e_decode_speedup.py`, on ROCm
(Nemotron-3-Nano-4B, batch 256, 500 greedy steps, CUDA graphs, FP32 SSM cache,
W=16, mean rank 8, `--modes standard sketchssm`; log in `e2e/e2e_speedup_b256.log`):

| mode | ms/step | output tok/s | speedup |
| --- | --- | --- | --- |
| standard (default decode) | 17.187 | 14,895 | 1.00x |
| SketchSSM | 7.570 | 33,820 | **2.27x** |

The PR reports 2.75x at the same batch on an RTX PRO 6000 (38.2 -> 13.9 ms/step);
MI355X is 2.27x (17.19 -> 7.57 ms/step) - both absolute times are ~2x lower
here, which compresses the ratio. Cross-check: one Mamba-2 layer is 4.16x faster
at batch 256 (section 8), so 2.27x end-to-end implies ~74% of the decode step was
linear attention on the baseline, inside the paper's 42-75% range.

## 10. Housekeeping notes

* CORRECTION: what looked like a foreign process holding ~219 GB was our own
  orphaned `VLLM::EngineCore` (pid 136598) from the first e2e attempt, whose
  parent died when that sub-agent timed out. The serving sub-agent found it,
  killed it and re-ran everything on clean GPUs (286 GiB free at startup). So
  the box was not shared; earlier measurements stand.
* The ROCm port used for all kernel numbers is `rocm_port.patch` here (scratch
  version, with an env override used only to measure bf16x3). A clean version
  that also relaxes the config gate lives in the worktree
  `/app/local/yh/sketchssm-e2e-wt`, branch `sketchssm-rocm-e2e` (merge commit
  `83e25b78ec`): `_validate_sketchssm` -> `is_cuda_alike()` with the CC check
  kept for CUDA only, `SKETCH_DOT_PRECISION` -> `"tf32x3" if is_cuda() else None`
  in `sketchssm_config.py`, and the kernel test gate -> `is_cuda_alike()`. That
  worktree is the one that served the e2e run in section 9, so it is the
  reference port.

## 11. End-to-end serving throughput on ROCm (HTTP, `vllm bench serve`)

Both servers identical apart from `--sketchssm` (MRV2, FP32 SSM cache,
`--mamba-cache-mode align` (a deprecation no-op on current main - Mamba prefix
caching follows `--enable-prefix-caching`, which was on), 1024-token prompts,
256 output tokens, `--ignore-eos`), GPU 6 baseline / GPU 7 SketchSSM,
cold prefix cache for both, 2 reps x distinct seeds. Note `--mamba-cache-mode`
is deprecated and inert on current main, so every "align mode" statement in this
document should be read as "prefix caching enabled". Details and provenance for
all 20 runs: `e2e/SERVING_ROCM.md`, `e2e/MANIFEST.md`.

c = 256 (saturated):

| metric | baseline | SketchSSM | speedup |
| --- | --- | --- | --- |
| req/s | 36.07 | 54.58 | **1.51x** |
| output tok/s | 9,234 | 13,972 | **1.51x** |
| TTFT p50 / p99 (ms) | 1262 / 2301 | 1262 / 2251 | 1.00x / 1.02x |
| TPOT p50 / p99 (ms) | 22.62 / 26.09 | 13.28 / 16.92 | **1.70x / 1.54x** |
| ITL p50 / p99 (ms) | 19.26 / 133.8 | 9.77 / 125.3 | 1.97x / 1.07x |

peak output tok/s 13,312 -> ~26,300 (**1.98x**)

c = 32 (small batch):

| metric | baseline | SketchSSM | speedup |
| --- | --- | --- | --- |
| req/s | 22.36 | 25.79 | 1.15x |
| output tok/s | 5,724 | 6,603 | 1.15x |
| TTFT p50 / p99 (ms) | 183.3 / 305.9 | 201.5 / 295.0 | 0.91x / 1.04x |
| TPOT p50 / p99 (ms) | 4.86 / 5.42 | 4.06 / 4.65 | 1.20x / 1.17x |

Reading:
* decode (TPOT) improves 1.70x at c=256, but end-to-end req/s improves only
  1.51x because prefill is unchanged and is ~27% of the wall clock at this
  input length (1.25 s of 262k prompt tokens for *both* servers), plus a
  ~5.6-5.8 ms/step serving overhead present on both paths (derived from
  offline-vs-serving, labelled as derived).
* TTFT does not regress at saturation, but does regress ~10% at c=32, where the
  prefill-side work SketchSSM adds (state rotation + sketch build after prefill)
  is not hidden by batching. That is the number to watch for latency-sensitive
  small-batch serving.
* The offline figure (2.27x, section 9 addendum) and the serving figure (1.51x)
  are both correct measurements of different things: offline = pure decode
  ms/step with no HTTP/detokenize overhead and no prefill in the window.

Open item raised by this run: the SketchSSM server logged
`Prefix cache hit rate: 0.0%` in every run, while the baseline climbed
0.4% -> 46% -> 74% on repeated identical prompts (`--dataset-name random`
regenerates the same prompts for a seed, so warm runs hit). So the comparison had
to be done cold-vs-cold. Cause is unexplained: either the PR's path does not
participate in align-mode prefix caching (it claims it does, token-for-token),
or the rebase lost it. Worth checking before quoting any cached-workload number,
because if real it means SketchSSM re-prefills every request.

## 12. Low-concurrency serving (c = 1, 2, 4, 8) - SketchSSM loses

Same two servers, 1024-in/256-out, fresh seed per run, per-run `/metrics`
prefix-cache deltas verified cold (details in `e2e/SERVING_LOWCONC_ROCM.md`):

| concurrency | baseline req/s | SketchSSM req/s | ratio | baseline TPOT p50 | SketchSSM TPOT p50 | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.81 | 1.50 | **0.83x** | 2.08 ms | 2.52 ms | 0.82x |
| 2 | 3.28 | 2.81 | **0.86x** | 2.24 ms | 2.61 ms | 0.86x |
| 4 | 5.81 | 5.21 | **0.90x** | 2.50 ms | 2.81 ms | 0.89x |
| 8 | 9.56 | 9.07 | **0.95x** | 2.96 ms | 3.12 ms | 0.95x |
| 32 | 22.36 | 25.79 | 1.15x | 4.86 ms | 4.06 ms | 1.20x |
| 256 | 36.07 | 54.58 | 1.51x | 22.62 ms | 13.28 ms | 1.70x |

Crossover between c=8 and c=32; SketchSSM is 5-17% slower at c<=8. TTFT is
0.93-0.95x throughout; ITL p99 is 1.4-1.6x worse at c=4/8 (reproducible) while
ITL p50 is flat. At c=1 the step is decode-bound, not prefill-bound: prefill is
only 3.5-3.9% of the 552/666 ms request (256 tokens at 2.08/2.52 ms decode), so
the whole c=1 gap is the +0.44 ms/token the extra kernels add. That matches the
kernel-level explanation: at batch 1 there is no state traffic to save (one
layer's state is ~3.9 MB, ~0.7 us of HBM time) while the flush path costs
0.015-0.17 ms/layer/step regardless of row count.

Practical reading: this is a throughput feature for saturating serving, not a
latency feature. Halving the flush kernel's fixed cost (`num_warps=1`, section
3.4) is the lever that would move the crossover down.

### Correction on the prefix-cache alarm (section 11)

The new run instrumented per-run `/metrics` deltas properly: 13 of 16 runs are
exactly 0 cache hits out of exactly 65,536 queried tokens, for **both** servers.
The 3 baseline runs showing hits (976 = 1024-48 tokens, 1.489%) are a dataset
artifact - `bench serve`'s RandomDataset builds prompts as slices of one 131072
token sequence, so two seeds can share one byte-identical prompt (verified for
seed 42001 vs 42003), which favours the baseline. So the earlier "SketchSSM logs
0.0% while the baseline climbs to 74%" was not evidence that SketchSSM skips
prefix caching - the 74% came from repeating the same seed on one server
(warm by construction, ~1024/1280 tokens of prompt matched). Both servers were
effectively cold in the tables above, so those numbers stand, and the "SketchSSM
may re-prefill everything" hypothesis is **withdrawn**.

A warm-cache A/B closed it out (`e2e/SERVING_LOWCONC_ROCM.md`, section
"Warm-cache A/B"). Repeating one 2670-token prompt four times:

| server (mamba block) | repeat-4 hit rate | same-seed rerun 1 | same-seed rerun 2 |
| --- | --- | --- | --- |
| baseline (block 976) | 5856/10680 = 54.8% (73.1% from the 2nd send) | 0% | 31,232/32,769 = 95.3% |
| SketchSSM (block 1056) | 6336/10680 = 59.3% (79.1% from the 2nd send) | 0% | 0% |

So SketchSSM **does** use the prefix cache - slightly better than the baseline on
a prompt that fully fills a block. The 0% on 1024-token inputs is a *block-size*
artifact, not a feature gap: the auto-selected mamba block size differs between
the two servers (1056 with `--sketchssm` vs 976 without), so a 1024-token random
prompt fills no complete block for SketchSSM while the baseline caches 976 of
them; at `--random-input-len 2048` run 2 hits exactly 33,792 = 32 x 1056.

That block-size difference is worth one question to the author, because it is a
user-visible side effect of enabling the flag: prompt lengths between 976 and
1056 tokens get cached without `--sketchssm` and not with it. Cause is
unverified (most likely the paged-state sizing feeding the align-mode block-size
derivation, which also depends on available KV memory), and it is
platform-independent.
