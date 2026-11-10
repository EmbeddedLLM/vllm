# PR #62004 (ReplaySSM for Gated DeltaNet) - ROCm trace + numerical analysis

PR: "[Kernel] ReplaySSM: cache SSM inputs for faster Gated DeltaNet standard
decode" (Johnny-Liou), +1838/-48, OPEN, branch `replayssm-gdn-standard`.
Second sub-PR of ReplaySSM; the Mamba-2 half (#48018) is already in main.
Relevance: this is exactly the missing upstream prerequisite for porting
SketchSSM to GDN (Qwen3.5/3.8) - SketchSSM builds its sketch on top of a
ReplaySSM-style window ring.

Environment: MI355X (gfx950), ROCm 7.15, torch 2.13+rocm, Triton 3.8,
vLLM main @ 73864f8360.

---

## 1. What it adds

* New Triton kernel `fused_recurrent_gated_delta_rule_replayssm`
  (`vllm/third_party/flash_linear_attention/ops/fused_recurrent_replayssm.py`,
  423 lines): non-flush steps reconstruct the delta-rule state from a
  checkpoint `h0` plus a ring of per-step `(k, d, g)` vectors and append the
  current step to the ring; flush steps fold the token into `h0` and store it.
* `_gdn_decode` launch config in `replayssm_config.py` (hardware-keyed).
* Five-tensor paged state: `(conv, ssm, d_cache, k_cache, g_cache)`; shapes and
  dtypes appended in `mamba_utils.py`.
* Dispatch in `QwenGatedDeltaNetAttention._forward_core`, metadata
  (`write_pos_d`, `is_flush_d`) built on the host in `gdn_attn.py`, plus
  align-mode re-anchoring for prefix caching.
* Reuses the existing `--use-replayssm` / `--replayssm-buffer-len` flags.

---

## 2. Portability trace, by file

### 2.1 The kernel - portable

Pure Triton: `tl.cumsum`, `tl.sum`, `tl.sigmoid`, `tl.exp`, `tl.log`,
`tl.where`, and exactly one `tl.dot(b_d_scaled_tc, b_k_all_c)`.

* No `input_precision` argument, no `allow_tf32`, no `cache_modifier`, no
  `tl.extra.cuda`, no inline asm, no TMA / warp specialization / `num_ctas`.
  Nothing that has a CUDA-only lowering.
* The dot operands are the *cache* dtype (`.to(p_d_main.dtype.element_ty)`), and
  `mamba_utils.append_gated_delta_net_replayssm_ring` makes `d_cache`/`k_cache`
  **fp16 when the activations are bf16** (`g_cache` is always fp32). So for the
  normal bf16 Qwen3.5 case the dot is fp16 x fp16 -> MFMA on gfx950, same
  instruction class as on NVIDIA.
* Caveat worth knowing: if someone runs a **float32** model, `d`/`k` caches
  become fp32 and that one `tl.dot` is then `tf32` on NVIDIA (Triton default
  there) but `ieee` on HIP (HIP has no tf32 dot). That is a *numerical
  divergence between platforms*, not a bug - HIP is the more accurate of the
  two - but any tolerance tuned on NVIDIA for fp32 state should be re-checked.
  The PR's own test tolerances are described as "tight enough to catch a TF32
  matmul-precision regression", so this is on their radar.
* Grid `(cdiv(V, BV), B, HV)`, early `return` for `state_idx <= 0` padded rows,
  `num_stages=3` pipelining: all fine on HIP.

### 2.2 Launch config - untuned but functional

`get_replayssm_config("gdn_decode", L=16)` uses `_is_blackwell()`; on ROCm that
is False, so it returns the non-Blackwell tuple `(block_v, num_warps,
num_stages, nk) = (64, 1, 3, 2)`, i.e. the H100-tuned config. Blackwell gets
`(128, 1, 3, 4)`. So ROCm runs, with an untuned (and possibly not ideal)
`block_v=64, nk=2`; `BKT = BK/nk = 64` satisfies the `BKT >= 16` assert for
K=128. No ROCm branch exists - same gap as `sketchssm_config.py`.

### 2.3 Paged state / TP

`append_gated_delta_net_replayssm_ring` appends
`d_cache (local_v_heads, L, head_v), k_cache (local_k_heads, L, head_k),
g_cache (local_v_heads, L)`, with `local_v_heads` from the already-TP-split ssm
shape and `local_k_heads = num_k_heads // tp_world_size`. Platform-neutral; TP
math is the same sharding the baseline uses, so no ROCm-specific risk beyond the
usual "tp must divide heads".

For Qwen3.5-4B (H=16, HV=32, K=V=128, L=16, 24 GDN layers) that is ~200 KB per
request per layer (~4.7 MB per request over the model) on top of the existing
~50 MB/request of fp32 state.

### 2.4 Metadata - platform-neutral, but the same CPU-planning pattern main removed

`gdn_attn.py` derives `write_pos = (num_computed - decode_base) mod L` and
`is_flush` **on the host** from `seq_lens_cpu` / `replayssm_decode_base_cpu`,
then `async_tensor_h2d`. All host-side torch ops - no ROCm issue.

But: this is the design main replaced for *Mamba-2* ReplaySSM with a GPU-side
planner (`plan_replayssm_decode` in `ops/ssu_dispatch.py`), and it reads
`m.replayssm_decode_base_cpu`, a `CommonAttentionMetadata` field from the V1
era. Expect the rebase to have to redo this part (see 2.6).

### 2.5 Integration - a ROCm-specific hole

On ROCm, GDN decode has **three** entry points
(`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`):

```
forward_hip      # GDN_AITER_TRITON_AVAILABLE -> qwen_gdn_attention_core(use_aiter=True)
                 #                              -> _forward_core_rocm -> _forward_core_decode_aiter
                 # else -> forward_cuda
forward_cuda     # -> qwen_gdn_attention_core(use_aiter=False) -> _forward_core   <-- PR hooks here
forward_aiter_flydsl  # active_backend == "aiter_flydsl"
```

The PR puts its dispatch inside `_forward_core`
(`if self.use_replayssm and is_non_spec_decode: ...`). `_forward_core_decode_aiter`
is a *separate* decode implementation that reads only `kv_cache[0]` (conv) and
`kv_cache[1]` (ssm) and never looks at `kv_cache[2:5]`.

So on a ROCm install where AITER's GDN Triton kernels are available
(`rocm_aiter_ops.are_gdn_triton_kernels_available()`), decode-only batches take
the aiter path and `--use-replayssm` becomes a **silent no-op**: the ring caches
are allocated (memory cost, 5-tensor page) but never read or written, and the
output stays baseline. There is no assert on `len(kv_cache) == 2`, so nothing
fails loudly. Mixed prefill+decode batches fall through
`_forward_core_rocm -> _forward_core`, so the feature would be active for those
and inactive for pure decode - the worst kind of inconsistent.

On *this* box AITER GDN is not installed, so
`are_gdn_triton_kernels_available() = False` (checked) and `forward_hip` falls
back to `forward_cuda` -> `_forward_core`, i.e. the PR's path **is** reached.
That is why the numerical tests below are meaningful here, and it is also the
first thing to fix if this lands: either hook `_forward_core_decode_aiter` too,
or force the generic path when `use_replayssm` is set, or assert loudly.

### 2.6 Model runner - stale, not platform-specific

The PR says "GDN ReplaySSM requires `VLLM_USE_V2_MODEL_RUNNER=0`". Current main
has **no** Model Runner V1 (`vllm/v1/worker/gpu_model_runner.py` is gone; the
only runner is `vllm/v1/worker/gpu/model_runner.py`, MRV2) and no
`VLLM_USE_V2_MODEL_RUNNER` env var. So the PR as written cannot run on main on
*any* platform until it is rebased onto MRV2 (the Mamba-2 ReplaySSM path already
was, see `v1/worker/gpu/model_states/mamba_hybrid.py`). This is the same
rebase wall #61124 hits.

---

## 3. Verdict on "does it work on AMD"

| layer | status |
| --- | --- |
| Triton kernel compiles / runs on gfx950 | expected yes (pure Triton, fp16 MFMA dot) - see numerical section |
| Numerical parity vs `..._packed_decode` on gfx950 | being measured (see `## 4`) |
| Launch config | runs with the H100 config; untuned for gfx950 |
| Paged state / TP / metadata | platform-neutral |
| ROCm integration with AITER GDN enabled | **silently inactive** - real gap, see 2.5 |
| Runs on current main at all | no - needs an MRV2 rebase first (2.6) |

## 4. Numerical analysis and performance on gfx950 (measured)

Full detail: `REPLAYSSM_GDN_ROCM.md`, harnesses in `bench/`, 26 logs in `logs/`.
Worktree: `/app/local/yh/replayssm-gdn-wt` branch `replayssm-gdn-rocm`.

### 4.1 Numerical correctness - PASS, nothing skipped

| suite | result |
| --- | --- |
| `tests/kernels/test_replayssm_standard_decode_gdn.py` | **80 passed** (71 s) - full matrix: 6 precision pairs x 3 geometries (incl. Qwen3.5-4B 16/32/128/128) x L in {4,16} x strided/contiguous, plus per-row `write_pos` and flush-beta cases |
| `tests/kernels/test_replayssm_teacher_decode_equivalence_gdn.py` | **12 passed** (102 s) |
| `tests/v1/attention/test_gdn_metadata_builder.py` (CPU) | **82 passed** (only after restoring `replayssm_decode_base_cpu`, which main removed with MRV1 - rebase artifact, not numerics) |

The gates are `torch.cuda.is_available()`, which is True on ROCm, so no test had
to be un-skipped. Against an *independent float64* recurrence
(`bench/num_error_magnitudes.py`):

| state / act | d,k cache | out err packed | out err ReplaySSM | state err at flush packed | ReplaySSM |
| --- | --- | --- | --- | --- | --- |
| fp32 / fp32 | fp32 | 4.5e-8 | 5.7e-8 | 1.5e-7 | 1.4e-6 |
| fp32 / bf16 | fp16 | 8.6e-4 | 8.6e-4 | 1.8e-7 | **5.2e-4** |
| bf16 / bf16 | fp16 | 7.9e-4 | 7.2e-4 | 6.0e-3 | 2.1e-3 |
| fp16 / fp16 | fp16 | 8.9e-5 | 8.8e-5 | 7.3e-4 | 3.9e-4 |

Two readings: (i) outputs match the baseline kernel to the same error level -
the fp16 ring dominates both equally, so ROCm introduces no numerical
regression; (ii) with the *shipping* configuration (bf16 model -> fp16 ring) the
**checkpoint state** is reconstructed through fp16 ring values every flush, so it
carries ~5e-4 error where the baseline's is ~2e-7. That error is rewrite-per-
window, i.e. it can drift across windows (bounded by the decay), and it is what
the fp32-ring option buys back at ~1.49x vs 1.63x. The PR's GSM8K parity
(0.917 vs 0.920) says it is benign in practice; it is still the accuracy knob to
know about.

### 4.2 Kernel latency on gfx950 (CUDA-graph captured, Qwen3.5-4B geometry, L=16)

Shipped non-Blackwell config `(block_v=64, warps=1, stages=3, nk=2)`:

| batch | fp16 d/k ms/step (packed -> replayssm) | speedup | fp32 d/k ms/step | speedup |
| --- | --- | --- | --- | --- |
| 1 | 0.0057 -> 0.0080 | **0.71x** | 0.0057 -> 0.0117 | **0.49x** |
| 8 | 0.0083 -> 0.0095 | **0.87x** | 0.0085 -> 0.0134 | **0.64x** |
| 32 | 0.0239 -> 0.0245 | 0.98x | 0.0243 -> 0.0274 | 0.89x |
| 128 | 0.0845 -> 0.0810 | 1.04x | 0.0885 -> 0.0908 | 0.98x |
| 256 | 0.2144 -> 0.2032 | 1.06x | 0.2207 -> 0.2122 | 1.04x |
| 512 | 0.4336 -> 0.4112 | 1.05x | 0.4316 -> 0.4222 | 1.02x |

Best config found on gfx950, `(block_v=16, warps=1, stages=1, nk=2)`:

| batch | fp16 d/k speedup | fp32 d/k speedup |
| --- | --- | --- |
| 1 | 1.03x | 0.95x |
| 8 | 1.02x | 0.87x |
| 32 | 1.23x | 1.07x |
| 128 | 1.49x | 1.30x |
| 256 | **1.62x** | **1.52x** |
| 512 | **1.63x** | **1.49x** |

So the shipped H100 config is the difference between "no win" (1.05x at bs 512)
and "1.6x": `(16,1,1,2)` is 1.55x faster than `(64,1,3,2)` for fp16 rings (1.45x
for fp32). `block_v=16` (8 programs in the value dim, more parallel waves) and
`num_stages=1` matter on gfx950; `nk=2` stays best. This is a bigger tuning
effect than anything seen for Mamba-2 ReplaySSM, whose non-Blackwell config
already transfers.

### 4.3 Precision seam confirmed from the generated ISA

Both ring dtypes lower to real matrix-core ops on gfx950
(`v_mfma_f32_32x32x16_f16`, `v_mfma_f32_16x16x16_f16`, `v_mfma_f32_32x32x2_f32`).
The fp32-ring case is the asymmetry predicted in 2.1: on gfx950 that `tl.dot`
becomes IEEE fp32 MFMA (error 5.7e-8 here), while the same kernel on NVIDIA takes
Triton's default `tf32` for fp32 operands. Same source, different numerics per
platform, and the PR passes no `input_precision` - worth a comment upstream.

### 4.4 End-to-end - does not start on main

`AttributeError: 'QwenGatedDeltaNetAttention' object has no attribute
'_replayssm_ring_start'`, raised from MRV2's
`MambaHybridModelState.initialize_kv_cache -> initialize_replayssm_ring_trackers`
(main's GPU-planned Mamba2 ReplaySSM plumbing, which assumes every
replayssm-enabled mamba layer has Mamba2's ring-tracker attributes). Behind that
sits the deletion of `CommonAttentionMetadata.replayssm_decode_base_cpu` with
MRV1 (#65033), which the GDN builder depends on. So no e2e number exists on main
- on any platform - until the GDN metadata is ported onto
`_plan_replayssm_decode_kernel` and the ring trackers are made Mamba2-only (or
GDN-aware).

## 5. MRV2 spike - the port is cheap, the e2e win is not yet visible

The MRV2 rebase was prototyped to answer "how hard is it to make this run on
main". Full notes: `MRV2_SPIKE.md`, patch: `mrv2_spike.patch` (base
`73864f8360`, needs rebasing onto the author's next push).

Cost: **2 files, +25/-1, mechanical**; the PR's three test suites stay green.

* `v1/worker/replayssm_utils.py` (+3): `initialize_replayssm_ring_trackers` now
  skips layers without `_replayssm_ring_start`, i.e. GDN cache groups. Clean
  upstream version would be an explicit capability flag rather than `hasattr`.
* `v1/worker/gpu/model_states/mamba_hybrid.py` (+23/-1): reheats
  `MambaHybridAttnMetadata.replayssm_decode_base_cpu`, forwarded through
  `get_extra_common_attn_kwargs` and filled from `input_batch.prefill_len_np`
  (CPU, no device sync) - the same field the SketchSSM rebase needed.

**The GPU planner does not transfer.** `plan_replayssm_decode` needs a `ring_pos`
slot counter that GDN's fused kernel has no index for, so reusing it would mean a
kernel extension or a second advance pass - a redesign, not a rebase item. The
spike therefore keeps host-side planning. On align mode: on current main
`--mamba-cache-mode` is a *deprecation no-op* ("has no effect: Mamba prefix
caching follows --enable-prefix-caching"), so there is no align-mode switch to
port - what remains unsolved is the PR's host-side re-anchoring logic, which
would have to key off `--enable-prefix-caching` instead. The spike ran with
prefix caching off.

Result: it decodes on main. Qwen3.5-0.8B (the local 4B snapshot is config-only,
no weights), 8 prompts x 64 greedy tokens, reproducible over two server starts:
**7/8 token-identical** to the baseline, the one difference a near-tie flip at
token 15 (0.4 nats between candidates), residual logprob noise 0.04-0.11 nats =
the fp16 ring reconstruction error.

Throughput, 128-in/128-out, `bench serve`, bs 64 / 256: baseline 21,499 / 36,184
output tok/s; ReplaySSM 20,641 / 36,208 (shipped config) and 20,973 / 35,691
(gfx950-tuned `(16,1,1,2)`). **Ratio 0.96-1.00x, inside run noise.**

That is not evidence against the feature, it is evidence about the model:
Qwen3.5-0.8B at bs <= 256 is not state-traffic-bound, and separately the shipped
kernel config caps the GDN kernel gain at 1.05x even at bs 512 (section 4.2). A
4B end-to-end run with the tuned config is the missing measurement
(`GDN_E2E_4B_ROCM.md` when it lands).

Verdict so far: **hand the author the patch as a starting point** - it buys
"runs on main", not "faster on AMD", and it is small enough to be useful to them
for their own rebase. The ROCm tuning ask stands on the kernel numbers, not on
this e2e run.

## 6. Qwen3.5-4B end to end (PR + MRV2 spike, ROCm)

Full setup, per-rep data and caveats: `GDN_E2E_4B_ROCM.md`. Real weights
(`Qwen/Qwen3.5-4B`, snapshot `851bf6e8...`, 9.3 GB safetensors - the local
snapshot had been config-only), 24 GDN layers of 32, bf16, FP32 SSM state, L=16,
MRV2 with CUDA graphs, prefix caching off, GPU 4, one server per config, 2 reps
per cell (spread <= 0.5%), 2048 prompts of 512-in/128-out.

| conc | baseline out tok/s | shipped `(64,1,3,2)` | tuned `(16,1,1,2)` | ratio shipped / tuned | TPOT p50 shipped |
| --- | --- | --- | --- | --- | --- |
| 64 | 7,820 | 8,504 | 8,553 | **1.087x / 1.094x** | 6.46 -> 5.78 ms |
| 256 | 11,295 | 12,815 | 12,823 | **1.135x / 1.135x** | 19.27 -> 16.63 ms |
| 512 | 12,328 | 14,105 | 14,168 | **1.144x / 1.149x** | 37.24 -> 32.22 ms |

TTFT is unchanged (p50 226 -> 229 ms at bs64, 488 -> 459 at bs512). So on ROCm
the feature delivers **1.09-1.15x end to end, TPOT -10 to -14%**, against the
PR's own H100/Qwen3.5-4B numbers of 1.14x / 1.27x / 1.30x at bs 64/256/512 -
i.e. ROCm gets roughly 80-88% of the H100 gain at serving batch.

Correctness: baseline-vs-baseline is 8/8 token-identical (0.0 logprob delta), which
proves the feature was on the decode path; ReplaySSM vs baseline is 6/8 identical
with mean |dlogp| 0.097 nats, both differences being greedy near-tie flips
(tuned: 7/8).

**Two things this measurement changes:**

1. **Launch-config tuning does not survive into e2e.** The isolated kernel is
   1.55x faster with `(16,1,1,2)`, yet e2e the difference is <= 0.6%. So the
   config branch is a kernel-level nice-to-have, not a serving win, and it should
   be presented that way to the author rather than as "1.55x sitting there".
2. **e2e beats the kernel numbers, which Amdahl cannot explain.** With the GDN
   kernel only 1.05x faster in isolation and 24 of 32 layers linear attention,
   e2e should be ~1.01-1.04x, not 1.144x. The likely reason is that removing the
   per-step state *store* relieves HBM pressure for the co-running attention/MLP
   kernels in the same step - an effect an isolated single-kernel benchmark
   cannot see. It is worth telling the author: their kernel microbenchmark
   understates the serving win, and (conversely) isolated config sweeps
   overstate it.

Verdict, updated: the feature **works on AMD and is worth 1.09-1.15x** at
serving batch on Qwen3.5-4B. The message to the author is now "it works here,
here are the numbers, and here is a 25-line MRV2 spike", with the config as a
footnote and the AITER path as the one real ROCm ask.
