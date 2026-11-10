# MRV2 spike: PR #62004 (GDN ReplaySSM) end-to-end on Model Runner V2

Time-boxed spike (90 min). Worktree `/app/local/yh/replayssm-gdn-wt`
(branch `replayssm-gdn-rocm`, 3 commits on main @73864f8360, PR #62004 merged).
GPUs 4 (ReplaySSM) and 5 (baseline), MI355X gfx950, ROCm 7.15, torch 2.13+rocm.

**Verdict: hand to the author as a patch.** The rebase is small (2 files,
+25/-1), mechanical, keeps the host-side plan (no device sync added to the
decode hot path), passes the existing CPU/GPU metadata and model-state tests,
and starts + decodes correctly on main. It is *not* too invasive. But the e2e
perf number on ROCm is ~1.00x, so the patch buys "runs on main", not "faster on
AMD" - the tuning gap (see 5) is a separate conversation with the author.

---

## 1. Blueprint: what Mamba-2 already does under MRV2 (and GDN would copy)

Mamba-2 ReplaySSM was rebased onto MRV2 in #65033 ("Plan Triton ReplaySSM
decode on the GPU"); SketchSSM's merge in
`/app/local/yh/sketchssm-e2e-wt` (branch `sketchssm-rocm-e2e`) is the worked
example of the same rebase.

* `ops/ssu_dispatch.py:_plan_replayssm_decode_kernel` / `plan_replayssm_decode`:
  one Triton launch per decode step per *cache group*. Reads `ring_pos[slot]`
  (per-slot token counter since the last full-state write), writes the step's
  per-row `write_pos` + `is_flush` into two persistent buffers, then advances
  `ring_pos`. `align_block_size>0` reads `seq_lens` and re-anchors at block
  boundaries (`pos = 0` on a block's first step, flush on the step completing a
  block).
* `v1/worker/replayssm_utils.py:initialize_replayssm_ring_trackers`: called from
  `MambaHybridModelState.initialize_kv_cache` when `cache_config.use_replayssm`;
  allocates `_replayssm_ring_start` (== the `ring_pos` tensor) and
  `_replayssm_prev_num_accepted` per cache group, shares them across the group's
  layers, and marks the group's lowest layer `_plans_replayssm_decode = True` so
  only it launches the planner.
* `v1/worker/gpu/model_states/mamba_hybrid.py`: builds
  `MambaHybridAttnMetadata` (CPU-side per-row tensors) and hands them to every
  builder through `get_extra_common_attn_kwargs` -> `CommonAttentionMetadata`;
  it no longer carries any ReplaySSM decode-base field (#65033 deleted it).
* `v1/attention/backends/mamba_attn.py`: the builder allocates persistent
  `decode_write_pos_d`/`decode_is_flush_d` buffers and returns *views* of them;
  the layer (`mamba_mixer2.py:1030-1062`) fills them by calling
  `plan_replayssm_decode(ring_start, state_indices, write_pos_d, is_flush_d,
  buffer_len, replayssm_seq_lens_d, replayssm_align_block_size)` and then runs
  `selective_state_update_replayssm_output_only` with those same tensors.
* Consequence: under MRV2 the planner owns `ring_pos` and no CPU seq-len value
  is read per step. `replayssm_seq_lens_d` / `replayssm_align_block_size` come
  from the Mamba-2 builder, which derives them from runner-provided tensors.

## 2. Blocker 1 - ring-tracker init assumed Mamba-2 attributes

`initialize_replayssm_ring_trackers` selected layers by `layer.use_replayssm`
alone and then read `layers[0]._replayssm_ring_start`, which only
`Mamba2Mixer` defines -> `AttributeError: 'QwenGatedDeltaNetAttention' object
has no attribute '_replayssm_ring_start'` for every GDN cache group.

Fix (`vllm/v1/worker/replayssm_utils.py`, +3/-0, one predicate):

```python
and hasattr(layer, "_replayssm_ring_start")   # after `layer.use_replayssm`
```

Correct for the spike because GDN never reads those trackers: it derives
`write_pos`/`is_flush` on the host and its kernel takes no `ring_pos`.
**Clean version:** an explicit capability on the layer/`MambaBase` (e.g.
`plans_replayssm_on_gpu: bool = False`, True in `Mamba2Mixer`) instead of a
`hasattr` probe, so the tracker group and the planner hand-off cannot drift
apart the way they just did. `get_replayssm_block_copy_tensors` needs the same
predicate if GDN ever enables `use_flashinfer_replayssm` (it filters on that
flag, so GDN is fine today).

## 3. Blocker 2 - GDN needs per-row `write_pos`/`is_flush` and the CPU field is gone

GDN's builder (`v1/attention/backends/gdn_attn.py:668-737`) computes
`write_pos = (num_computed - decode_base) % L` and `is_flush` on the host from
`CommonAttentionMetadata.replayssm_decode_base_cpu` + `seq_lens_cpu_upper_bound`
and `async_tensor_h2d`s two small int tensors, once per step (not per layer).
#65033 deleted the field with V1, so the builder raised
`ValueError: --use-replayssm requires exact CPU sequence lengths and decode-base
counts`.

**GPU planner reuse: NO (would not fit the box).** `plan_replayssm_decode` is
coupleable to GDN - the kernel is shape-agnostic - but it needs a `ring_pos`
tensor *and* something that advances it. Under Mamba-2 the planner advances it
and the layer needs nothing else; GDN's fused kernel takes no ring slot index at
all, so reusing the planner would mean allocating a per-slot ring counter in
GDN's cache group plus a second state-advance pass (or extending
`fused_recurrent_gated_delta_rule_replayssm` to own `ring_pos`). That is a
redesign of the PR's metadata, not a rebase line item - out of scope here.

**What I did instead: host-side fallback, no new device sync.** Two additions in
`v1/worker/gpu/model_states/mamba_hybrid.py` (+23/-1):

* `MambaHybridAttnMetadata.replayssm_decode_base_cpu: torch.Tensor | None` plus
  a branch in `get_extra_common_attn_kwargs` that forwards it (sketchssm's
  branch carries the identical field for its own keys - same shape of change).
* filled from `input_batch.prefill_len_np` (CPU numpy, already materialised by
  the runner) when `cache_config.use_replayssm`; padding rows get 0.
  `prefill_len` is "tokens the runner was given", so a request resumed after
  preemption re-anchors past the replayed prompt+generated tokens - exactly the
  semantics the PR's `decode_base` documents (`write_pos 0` after resume).

This is a **loudly-labelled spike shortcut**, not the design I would defend
upstream: it re-introduces a V1-era CPU tensor to a runner that deliberately
moved this state to the GPU, and it costs 2 small `async_tensor_h2d` calls per
decode step (no *sync*, but it is host work that grows with batch size). The
honest end state is either (a) GDN adopts the GPU planner (needs the ring slot
in the kernel), or (b) the field is accepted as a first-class runner output the
way `prefill_len`/`is_prefilling` already are.

**Align mode is not solved.** With `--mamba-cache-mode align` the block-boundary
re-anchoring in the host path (`gdn_attn.py:700-712`) interacts with
`MambaSpecDecodeGPUContext`'s pre-copy kernel, which GDN never exercised.
All measurements below run `--mamba-cache-mode none` (main's default).

## 4. e2e correctness (first numbers on main, any platform)

Server: `qwen3.5-0.8B`, `--use-replayssm --replayssm-buffer-len 16
--mamba-ssm-cache-dtype float32 --max-model-len 4096 --language-model-only
--trust-remote-code`, MRV2 (the only runner on main). Baseline = identical
command without `--use-replayssm`. 8 prompts x 64 greedy tokens, `logprobs=0`.

Note: `/models/models--Qwen--Qwen3.5-4B` has **config only, no weights** on this
box (29 KB snapshot), so the sweep uses 0.8B (16 k-heads / 16 v-heads x 128,
24 layers of which 18 GDN, conv kernel 4). Same arch, smaller; the kernel work
is geometry-independent at these sizes. Not a bandwidth-limited config - see 5.

| # | prompt | text | max abs dlogp |
| --- | --- | --- | --- |
| 0 | The capital of France is | match | 0.065 |
| 1 | Once upon a time, in a small village, | **DIFF** | 2.58 |
| 2 | def fibonacci(n): | match | 0.057 |
| 3 | Q: What is 17 * 23? A: | match | 0.044 |
| 4 | three laws of thermodynamics | match | 0.109 |
| 5 | In the year 2050, ... | match | 0.113 |
| 6 | Translate 'the cat sat on the mat' | match | 0.105 |
| 7 | A short poem about the sea at night | match | 0.100 |

**7/8 identical**, reproducible across two independent server starts. The one
divergence is a near-tie at token 15 (" named Bob" logp -0.47 vs " named Barn"
-0.06 in the two runs, i.e. the two candidates were ~0.4 nats apart), after
which the texts legitimately fork; the max dlogp 2.58 is downstream of that
flip, not an independent error. The residual 0.04-0.11 nats on matched tokens is
the fp16-ring reconstruction noise already measured at kernel level (~5e-4
state error with a bf16 model / fp16 ring), so this is "correct within the
shipped precision, with an occasional greedy near-tie flip" - the same
behaviour the PR reports on NVIDIA (GSM8K 0.917 vs 0.920).

The degenerate loops in some outputs (Paris x N, "The sea is a dark," x N) are
the 0.8B model, present identically in the baseline.

## 5. Throughput (bs 64 / 256, 128 in / 128 out, `vllm bench serve`, random)

| config | bs 64 out tok/s | bs 256 out tok/s |
| --- | --- | --- |
| baseline (`--mamba-cache-mode none`) | 21499 | 36632 / 35808 / 36111 (mean 36184) |
| ReplaySSM, shipped cfg `(block_v 64, warps 1, stages 3, nk 2)` | 20641 | 36208 |
| ReplaySSM, gfx950-tuned cfg `(16, 1, 1, 2)` | 20973 | 35613 / 35188 / 36273 (mean 35691) |

**Ratio vs baseline: 0.96-1.00x at bs 64, 0.97-1.00x at bs 256.** Run-to-run
spread on the same config is +-2%, so this is "no measurable e2e win", not a
regression. Two reasons, both expected and both already visible in the kernel
data: (i) 0.8B at bs 256 on MI355X is nowhere near HBM-bound on the SSM state,
which is the only traffic ReplaySSM removes; (ii) even at kernel level the
shipped non-Blackwell config is ~1.05x at bs 512 on gfx950 (the tuned
`(16,1,1,2)` reaches 1.62x there) - so there is no headroom for the model-level
number to show anything at these sizes. The tuned override was applied by
editing `_gdn_decode`'s non-Blackwell return (one line, reverted afterwards; it
is *not* in the saved patch) because no runtime config override is exposed.

A meaningful e2e ratio needs a model that is actually state-bound: 35B-A3B or
4B weights, larger batches (512+), and long generation. Not in the time box.

## 6. What the patch contains / what is left

`mrv2_spike_mrv2only.patch` (68 lines) = the two fixes above.
`mrv2_spike.patch` (2446 lines) = `git diff main`, i.e. PR #62004 + those fixes.

Done:
* starts and decodes end-to-end on MRV2 (both blockers clearance verified by
  running the server, not by reasoning);
* existing tests green: `tests/v1/worker/test_mamba_hybrid_model_state.py`
  22 passed / 1 skipped, `tests/v1/attention/test_gdn_metadata_builder.py`
  82 passed;
* `ruff check` + `ruff format --check` clean on both files.

Left for the author (in rough order):
1. replace the `hasattr` probe with a real capability flag (2 lines, above).
2. decide the CPU-vs-GPU plan for GDN: either extend
   `fused_recurrent_gated_delta_rule_replayssm` to own `ring_pos` and reuse
   `plan_replayssm_decode` (kills the `async_tensor_h2d`s and the revived CPU
   field), or accept `replayssm_decode_base_cpu` as runner output.
3. align-mode (`--mamba-cache-mode align`) for GDN: never exercised, likely
   needs the pre-copy kernel path.
4. the ROCm launch config: add a gfx9 branch to `_gdn_decode` (`16,1,1,2` is
   1.45-1.55x faster than the shipped non-Blackwell config at bs>=128).
5. the AITER-decode hole (`_forward_core_decode_aiter` ignores `kv_cache[2:5]`,
   so `--use-replayssm` is a silent no-op wherever AITER GDN kernels exist).

Artifacts in this dir: `bench/mrv2_spike_correctness.py`,
`mrv2_rs_correctness{,_final}.json`, `mrv2_base_correctness.json`,
`bench/mrv2_spike_*.json`, `logs/mrv2_spike_*.log`.
