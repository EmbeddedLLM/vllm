# GDN ReplaySSM under AITER on ROCm: is `--use-replayssm` a silent no-op?

**Verdict: silent no-op confirmed.** With `VLLM_ROCM_USE_AITER=1`, GDN decode
takes `_forward_core_decode_aiter` and the ReplaySSM kernel is never entered:
`--use-replayssm` buys **1.03x** instead of the **1.14x** it buys with AITER off.
The 1.03x is not the ReplaySSM kernel (the trace and a bit-exact correctness
result both say that block never runs); it is a secondary ~3% that comes with
the flag's KV-cache shape change, and it costs 8.4% of the KV cache.

Claim under test: *with AITER enabled, GDN `--use-replayssm` is silently inactive
on ROCm*. Timebox 75 min, one experiment.

## 1. Which core function each run entered (the decisive evidence)

Temporary env-guarded prints (`if os.environ.get("SPIKE_TRACE")`) at the top of
`_forward_core_decode_aiter` and inside the `if self.use_replayssm and
is_non_spec_decode:` branch of `_forward_core`
(`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`). Servers were
started with `SPIKE_TRACE=1 --enforce-eager` so every decode step runs Python
(`--enforce-eager` does not change the dispatch conditions, only whether the
step is inside a captured graph), then served one 32-token greedy generation
(`bench/gdn4b_trace.py`).

| aiter | `--use-replayssm` | function entered | log evidence |
| --- | --- | --- | --- |
| OFF | no | `_forward_core` -> generic kernel | 0 `_forward_core_decode_aiter` prints, 0 ReplaySSM-branch prints |
| OFF | yes | `_forward_core` -> **ReplaySSM branch** | 840 prints = 24 GDN layers x 35 steps (32 generated tokens + the engine's startup warmup steps), every one `_forward_core -> replayssm branch` |
| **ON** | no | **`_forward_core_decode_aiter`** | `SPIKE_TRACE: _forward_core_decode_aiter (...layers.0.linear_attn) kv_cache_tensors=2` |
| **ON** | **yes** | **`_forward_core_decode_aiter`** | `SPIKE_TRACE: _forward_core_decode_aiter (...layers.0.linear_attn) kv_cache_tensors=5`; **0** ReplaySSM-branch prints |

Verbatim, the two cells that matter (`logs/gdn4b_2x2_aon_{base,rs}_trace_server.log`):

```
SPIKE_TRACE: _forward_core_decode_aiter (language_model.model.layers.0.linear_attn) kv_cache_tensors=2 shapes=[(13508, 3, 8192), (13508, 32, 128, 128)]
SPIKE_TRACE: _forward_core_decode_aiter (language_model.model.layers.0.linear_attn) kv_cache_tensors=5 shapes=[(12382, 3, 8192), (12382, 32, 128, 128), (12382, 32, 16, 128), (12382, 16, 16, 128), (12382, 32, 16)]
```

So with AITER on the ReplaySSM ring **is allocated** by
`QwenGatedDeltaNetAttention.get_state_shape` -> `append_gated_delta_net_replayssm_ring`
(tensors 3-5 are the d/k/g ring), and the AITER decode function reads only
`kv_cache[0]` (conv) and `kv_cache[1]` (ssm) and calls aiter's
`gated_delta_rule`; `kv_cache[2:5]` is never touched. That is "inactive" in the
worst way: you pay the memory and the host-side `write_pos`/`is_flush`
bookkeeping (`gdn_attn.py:672-810`) for a kernel that never launches.

All four trace cells were run. The AITER-off ReplaySSM trace is the positive
control: the branch is reachable and fires for all 24 GDN layers when the AITER
path is skipped. (The AITER-on print fires once per process, hence count 1, so
the zero ReplaySSM prints there is the informative number: the branch is never
reached, not merely rarely reached.)

## 2. The 2x2 (GPU 6 exclusive, concurrency 512, 512-in / 128-out, 2048 prompts)

Same tree (`/app/local/yh/replayssm-gdn-wt`), same flags
(`--max-model-len 4096 --mamba-ssm-cache-dtype float32 --no-enable-prefix-caching
--gpu-memory-utilization 0.80`), one server at a time, one discarded warmup run
per server, fresh random seed per rep. All 18 runs completed 2048/2048 requests
with 0 failures.

| aiter | config | mean out tok/s | reps (out tok/s) | TPOT p50 (ms) | ratio vs its own baseline |
| --- | --- | --- | --- | --- | --- |
| OFF | baseline | 12,286 | 12287 / 12317 / 12260 / 12278 | 37.34 | 1.000 |
| OFF | `--use-replayssm` | **13,975** | 14044 / 13890 / 13988 / 13978 | 32.00 | **1.137x** |
| ON | baseline | 12,508 (12,542 median) | 12288 / 12650 / 12542* / 12514 / 12545 | 36.73 | 1.000 |
| ON | `--use-replayssm` | **12,921** | 12937 / 12924 / 12951 / 12936 / 12857 | 35.70 | **1.033x** |

\* the aiter-on baseline's first rep (12,288) is the only outlier in the set
(every other rep of that config is 12,514-12,650); against the 4-rep steady-state
mean 12,563 the aiter-on ratio is 1.029x. The aiter-on ratio is therefore
**1.03x, not 1.14x**, either way it is computed. TPOT p50 tells the same story:
36.73 -> 35.70 ms (1.029x) with AITER on, versus 37.34 -> 32.00 ms (1.167x) with
AITER off - a 14% decode-latency win collapsing to 3%.

Cross-checks that the two paths are not being mixed up:

* AITER off reproduces the earlier GPU-4 numbers on GPU 6: 12,286 vs 12,328
  baseline (+0.3%) and 13,975 vs 14,105 ReplaySSM (-0.9%), TPOT 37.34 vs 37.24
  and 32.00 vs 32.22 ms. GPU 6 is equivalent to GPU 4, so the aiter-on cells are
  comparable even though they were measured elsewhere.
* AITER on changes the baseline too: 12,508 vs 12,286 (1.018x) and TPOT
  36.73 vs 37.34 ms, i.e. the AITER decode kernel itself is ~1.7% faster than
  the generic path. The ReplaySSM win is measured *within* the same AITER
  setting, so that does not contaminate the ratio.

## 3. Correctness, aiter ON, ReplaySSM vs baseline

8 prompts x 64 greedy tokens, `temperature=0`, `logprobs=0`, same server config
as the bench (`bench/gdn4b_correctness.py`, diff with `bench/gdn4b_compare.py`,
raw in `aon_base_correctness.json` / `aon_rs_correctness.json` /
`aon_rs_vs_base_compare.json`).

| run | token-identical | max abs dlogp | mean abs dlogp |
| --- | --- | --- | --- |
| aiter ON, ReplaySSM vs baseline | **8 / 8** | **0.000** | **0.000** |

Not merely "still correct": **bit-exact**, every token and every logprob
identical. That is a second, independent proof that no ReplaySSM kernel ran
(contrast: with AITER off the same comparison is 6/8 with 0.02-0.10 nat
per-token deltas from fp16 ring reconstruction, see `GDN_E2E_4B_ROCM.md`). The
claim was "inactive", not "broken", and the output is in fact unchanged.

## 4. Cost side: the ring is allocated anyway

| aiter | config | GPU KV cache tokens | max concurrency @ 4096 |
| --- | --- | --- | --- |
| ON | baseline | 4,855,249 | 1185x |
| ON | `--use-replayssm` | 4,449,745 | 1086x |
| OFF | baseline | 4,855,249 | 1185x |
| OFF | `--use-replayssm` | 4,450,490 | 1086x |

Identical in both AITER settings, i.e. the ring costs **8.4% of KV cache
capacity** with AITER on while returning none of the kernel's benefit. Paged
mamba state therefore still has **5 tensors** with AITER on (see the trace
`kv_cache_tensors=5 shapes=[...]` line above) - `kv_cache[0]` conv,
`kv_cache[1]` ssm, `kv_cache[2:5]` the unused ring.

## 5. What the residual 1.03x is (and is not)

It is not the ReplaySSM kernel: the trace shows the branch is never entered and
the output is bit-exact. The only other difference the flag makes on the AITER
path is the state-shape change (5 state tensors per block, 2.10 MB ssm +
0.10 MB conv + 0.40 MB ring instead of 2.10 + 0.10 MB, hence 8.4% fewer blocks
for the same 217.6 GiB), plus host-side `write_pos`/`is_flush` derivation and two
extra H2D metadata copies. A plausible mechanism for a 3% *gain* from a
*fewer-blocks* layout is block-id locality for the 2 MB-per-block `ssm_state`
gather (smaller active page-id range), but this experiment did not test it, and
the host-side work would push the other way. Treat the 1.03x as an unexplained
secondary effect of the KV-cache shape, not as evidence that anything of the
ReplaySSM kernel path is running. It is 4x smaller than the real 1.14x, and it
costs 8.4% of KV capacity.

## 6. Caveats

* GPU 4 and GPU 5 were taken over by a foreign 235 GB/GPU job (PIDs 58791/58792)
  between the coordinator's check and this run; my server was stopped rather
  than sharing. Everything here was measured on **GPU 6, exclusively** (GPU 7
  idle, `amd-smi process` empty before the first start). Because the reference
  1.144x was measured on GPU 4, the aiter-OFF row above is *re-measured in this
  session on GPU 6* and agrees with it to 0.9%, so the ratio comparison is
  same-session and same-GPU.
* Cells: 4-5 reps per cell at bs 512 only (2 reps as briefed, extended to 4-5
  because the first aiter-on baseline rep was an outlier). No other batch size
  was re-measured with AITER on.
* `--enforce-eager` was used only for the trace servers; the timing and
  correctness cells use the normal `FULL_AND_PIECEWISE` cudagraph runner.
* The dispatch condition itself is batch-size independent
  (`spec_sequence_masks is None and num_prefills == 0 and num_decodes > 0 and
  not is_conv_state_dim_first()`), so the bs=1 trace generalises to the bs=512
  bench batches; mixed prefill+decode steps also miss the ReplaySSM branch
  because it additionally requires `num_prefills == 0`.
* The temporary prints were reverted; `git diff` on
  `qwen_gdn_linear_attn.py` is empty and the only modified files in the worktree
  are the two pre-existing MRV2 spike files (`mamba_hybrid.py`,
  `replayssm_utils.py`).

## Files

Results: `AITER_PATH_ROCM_summary.json` (means + per-rep), raw bench json
`bench/gdn4b_aiter_{aon,aoff}_{base,rs}_bs512_r*.json` (+ `_r0` for the aiter-off
reference cells), bench logs `logs/gdn4b_aiter_bench_*_bs512_r*.log`.
Server logs: `logs/gdn4b_2x2_{aon,aoff}_{base,rs}_server.log` (timing) and
`logs/gdn4b_2x2_{aon_base,aon_rs,aoff_base,aoff_rs}_trace_server.log` (trace +
KV shapes).
Correctness: `aon_base_correctness.json`, `aon_rs_correctness.json`,
`aon_rs_vs_base_compare.json`.
Scripts: `bench/gdn4b_2x2_server.sh` (DEV/AITER/EAGER/SPIKE_TRACE env-selected),
`bench/gdn4b_aiter_bench.sh`, `bench/gdn4b_trace.py`, `bench/wait_ready.sh`,
plus the reused `bench/gdn4b_correctness.py` / `bench/gdn4b_compare.py`.

Follow-up: `AITER_EXCLUDE_GDN_ROCM.md` excludes GDN from the AITER path with a
one-line gate, which makes ReplaySSM active again under
`VLLM_ROCM_USE_AITER=1` for 14,406 out tok/s (1.141x in-cell, +3.1% over the
AITER-off ReplaySSM cell).
