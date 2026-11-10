# AITER on, GDN excluded: does ReplaySSM come back, and is it the best cell?

**Verdict: yes, and it is the best configuration measured.** Excluding GDN from
the AITER path (one-line gate on `forward_hip`) makes the ReplaySSM kernel run
again with `VLLM_ROCM_USE_AITER=1` still set for everything else:
**12,628 -> 14,406 out tok/s (1.141x)** in-cell, and **+3.1% over the AITER-off
ReplaySSM cell** (13,975) and **+11.5% over the AITER-on ReplaySSM cell that was
a silent no-op** (12,921). Against the plain AITER-off baseline it is **1.173x**.

Two new server configs only (timebox 60 min, GPU 6 exclusive). Reference cells
are reused verbatim from `AITER_PATH_ROCM.md`, same GPU, same tree, same flags.

## 1. The change (experiment only, uncommitted)

`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `forward_hip`,
saved as `bench/gdn_aiter_exclude_gdn.patch`:

```python
if (
    GDN_AITER_TRITON_AVAILABLE
    and not self.enable_fused_gdn_spec_decode
    and not getattr(self, "use_replayssm", False)
):
    ...  # AITER fused projection + _forward_core_rocm
else:
    return self.forward_cuda(hidden_states)  # -> qwen_gdn_attention_core(use_aiter=False) -> _forward_core
```

With `--use-replayssm` the layer falls through to `forward_cuda` ->
`_forward_core`, where the PR's ReplaySSM branch lives; every other AITER
consumer (matmul/linear, rmsnorm, MoE, attention) is untouched - both new
servers log `ir_op_priority rms_norm=['aiter','native'],
fused_add_rms_norm=['aiter','native']`. The gate applies to all token counts, so
GDN **prefill** also leaves the AITER path and runs the generic kernel;
ReplaySSM itself only covers decode. Without `--use-replayssm`,
`use_replayssm` is False and the gate is a no-op.

## 2. Trace evidence: the ReplaySSM branch is entered, the AITER decode kernel is not

The two env-guarded `SPIKE_TRACE` prints (top of `_forward_core_decode_aiter`,
and inside `if self.use_replayssm and is_non_spec_decode:` of `_forward_core`)
were re-added, servers started with `SPIKE_TRACE=1 --enforce-eager`, and each
served one 32-token greedy generation (`bench/gdn4b_trace.py --tokens 32`).

| aiter | gate | `--use-replayssm` | `_forward_core_decode_aiter` prints | ReplaySSM-branch prints | log |
| --- | --- | --- | --- | --- | --- |
| ON | no (reference) | yes | 1 (once per process) | **0** | `logs/gdn4b_2x2_aon_rs_trace_server.log` |
| ON | **yes** | **yes** | **0** | **840** = 24 GDN layers x 35 steps | `logs/gdn4b_aexg_rs_trace_server.log` |
| ON | yes | no | 840 = 24 layers x 35 steps | 0 | `logs/gdn4b_aexg_base_trace_server.log` |

Verbatim from `logs/gdn4b_aexg_rs_trace_server.log` (and zero
`_forward_core_decode_aiter` lines in the whole file):

```
SPIKE_TRACE: _forward_core -> replayssm branch (language_model.model.layers.0.linear_attn)
SPIKE_TRACE: _forward_core -> replayssm branch (language_model.model.layers.1.linear_attn)
```

The third row is the control that the gate is conditional, not a blanket
"turn AITER off": without `--use-replayssm` the same patched tree still prints
840 `_forward_core_decode_aiter` lines and no ReplaySSM line. Prints were
reverted afterwards.

## 3. All cells (GPU 6, conc 512, 512-in / 128-out, 2048 prompts)

Same flags in every cell (`--max-model-len 4096 --mamba-ssm-cache-dtype float32
--mamba-cache-mode none --no-enable-prefix-caching
--gpu-memory-utilization 0.80`), one server at a time, one discarded warmup per
server, fresh seed per rep. All new runs: 2048/2048 requests, 0 failures.

| # | aiter | GDN on AITER | `--use-replayssm` | mean out tok/s | reps | TPOT p50 (ms) | ratio vs its own baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | OFF | - | no | 12,286 | 12287/12317/12260/12278 | 37.34 | 1.000 (reference) |
| 2 | OFF | - | yes | **13,975** | 14044/13890/13988/13978 | 32.00 | 1.137x (reference) |
| 3 | ON | yes | no | 12,508 | 12288/12650/12542/12514/12545 | 36.73 | 1.000 (reference) |
| 4 | ON | yes | yes | 12,921 | 12937/12924/12951/12936/12857 | 35.70 | 1.033x = **silent no-op** (reference) |
| 5 | ON | **no** (gate) | no | 12,628 | 12648.59/12607.67 | 36.65 | 1.000 (new, control) |
| 6 | ON | **no** (gate) | **yes** | **14,406** | 14390.65/14421.67 | **31.58** | **1.141x** (new) |

Key comparisons (all same-GPU, same-session for the new cells):

| comparison | out tok/s | ratio |
| --- | --- | --- |
| #6 vs #5 (new cell, in-cell) | 14,406 / 12,628 | **1.141x** |
| #6 vs #2 (new vs AITER-off ReplaySSM) | 14,406 / 13,975 | **1.031x** |
| #6 vs #1 (new vs AITER-off baseline) | 14,406 / 12,286 | **1.173x** |
| #6 vs #4 (new vs AITER-on ReplaySSM, the no-op cell) | 14,406 / 12,921 | **1.115x** |
| #5 vs #3 (control: same code path, reproduced) | 12,628 / 12,508 | 1.010x |
| TPOT p50, #5 -> #6 | 36.65 -> 31.58 ms | 1.161x |

Read: row 6 recovers the full ReplaySSM decode win (TPOT 36.65 -> 31.58 ms,
ratio 1.141x, indistinguishable from the 1.137x AITER-off pair) **and** keeps
AITER for everything that is not a GDN layer (the 8 full-attention layers plus
all matmul/rmsnorm work), which is worth the extra **+3.1%** out tok/s over
row 2. Rows 3/4 remain the statement of the bug: with GDN on AITER,
`--use-replayssm` is a 1.03x no-op.

## 4. Correctness: not bit-exact anymore (that is the point)

8 prompts x 64 greedy tokens, `temperature=0`, `logprobs=0`, same server config
as the bench; `bench/gdn4b_correctness.py` + `bench/gdn4b_compare.py`, raw in
`aexg_base_correctness.json` / `aexg_rs_correctness.json` /
`aexg_rs_vs_base_compare.json`.

| run | token-identical | max abs dlogp | mean abs dlogp |
| --- | --- | --- | --- |
| **#6 ReplaySSM vs #5 baseline (new)** | **4 / 8** | **2.474** | **0.157** |
| #4 vs #3 (AITER on, silent no-op, reference) | 8 / 8 | 0.000 | 0.000 |
| #2 vs #1 (AITER off, reference) | 6 / 8 | 1.744 | 0.097 |
| same-server rerun, #1 twice (determinism control) | 8 / 8 | 0.000 | 0.000 |

The 8/8-bit-exact signature of "ReplaySSM never ran" is gone: with the gate the
outputs change, first divergence at token 12-35 on 4 of 8 prompts, and even on
the prompts that stay on the same token sequence the per-prompt mean logprob
delta is 0.002-0.016 nat (single-token max up to 0.10). Greedy 8x64 is
flip-prone in this model - the AITER-off pair already flips 2/8, and even
#2-vs-#6 (both running the same ReplaySSM kernel, differing only in non-GDN
AITER numerics) is 3/8 - so the 4/8 here is evidence the kernel ran, not a
quality verdict. Two independent changes are in play at once (GDN kernel
aiter -> generic+fp16-ring, plus non-GDN aiter numerics), which is why it is
larger than the AITER-off 6/8 / 0.097 pair.

## 5. Cost side: the ring still costs 8.3% of KV cache

| config | GPU KV cache tokens | max concurrency @ 4096 |
| --- | --- | --- |
| #5 aiter-on, GDN excluded, no replay | 4,854,132 | 1185.09x |
| #6 aiter-on, GDN excluded, ReplaySSM | 4,450,490 | 1086.55x |
| #3 / #1 reference baselines | 4,855,249 | 1185.36x |

**-8.3% KV capacity** (1086 vs 1185 concurrent 4096-token requests) for +14.1%
decode throughput - the same trade the AITER-off cell makes (#2 = 4,450,490
tokens), and now it buys the kernel instead of just paying for the ring.

## 6. Caveats / not apples-to-apples

* **Row 5 is not a new code path.** Without `--use-replayssm`, `use_replayssm`
  is False, so the gate does not fire and GDN still uses the AITER kernel; row 5
  is a same-GPU reproduction of row 3 (12,628 vs 12,508, and 12,508's own rep
  range already spans 12,514-12,650). There is therefore no measured cell for
  "AITER off for GDN only, no replay" - the honest baseline for row 6 is its own
  row-5 reproduction plus row 3.
* New cells have 2 reps each (as briefed; the reference cells have 4-5 because
  the first AITER-on baseline rep was an outlier). Rep spread within the new
  cells is 0.3% (base) and 0.2% (rs), and the discarded warmups landed at 12,595
  and 14,442, i.e. inside the reported ranges.
* bs 512 only, one model (Qwen3.5-4B, 24 GDN of 32 layers), TP1, no speculative
  decoding, no prefix caching.
* The +3.1% of row 6 over row 2 is a whole-config statement: no per-op ablation
  of *which* non-GDN AITER consumer supplies it was run.
* This gate is an experiment patch (`getattr(self, "use_replayssm", False)`),
  not a proposal for how upstream should express it.
* `--enforce-eager` was used only for the three trace servers; all timing and
  correctness cells used the normal `FULL_AND_PIECEWISE` cudagraph runner.


## 7. Follow-up: the missing cell - "AITER on, GDN forced off AITER, no replay"

§6 said no measured cell existed for "aiter everywhere except GDN, no replay",
because the gate keys on `use_replayssm` and therefore no-ops without
`--use-replayssm` (row 5 was just a row-3 reproduction). This section measures
that cell by adding an unconditional override: one module-level flag plus one
clause in `forward_hip` (`bench/gdn_aiter_gdn_off_forcing.patch`, stacked on the
gate patch, uncommitted):

```python
SPIKE_GDN_NO_AITER = os.environ.get("SPIKE_GDN_NO_AITER", "0") == "1"
...
        if (
            GDN_AITER_TRITON_AVAILABLE
            and not self.enable_fused_gdn_spec_decode
            and not getattr(self, "use_replayssm", False)
            and not SPIKE_GDN_NO_AITER
        ):
```

With `SPIKE_GDN_NO_AITER=1`, GDN takes `forward_cuda` -> `_forward_core` in both
prefill and decode regardless of the flag; every other AITER consumer is
untouched (both servers still log
`ir_op_priority rms_norm=['aiter','native'], fused_add_rms_norm=['aiter','native']`).
Two new servers on GPU 6, same flags, same 2048-prompt/conc-512 bench, one
discarded warmup each; 2048/2048 requests, 0 failures in every rep.

| # | aiter | GDN on AITER | `--use-replayssm` | mean out tok/s | reps | TPOT p50 (ms) | KV cache tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A** | ON | **no** (env-forced) | no | **12,658** | 12686.71/12629.63 | 36.54 (36.41/36.66) | 4,854,877 |
| **B** | ON | **no** (env-forced) | **yes** | **14,393** | 14385.27/14400.15 | 31.72 (31.66/31.79) | 4,450,490 |

Rep spread 0.45% (A) and 0.10% (B). The discarded A warmup is 12,449.71, 1.6%
below A's reps (cold clocks, discarded per protocol); the B warmup, 14,419.71,
sits inside B's range. B's KV cache (4,450,490) is bit-identical to
row 6's and A's (4,854,877) matches rows 3/5, so the two cells really are the
row-3 and row-6 configurations.

| comparison | out tok/s | ratio | one-line read |
| --- | --- | --- | --- |
| **A vs #3** | 12,658 / 12,508 | **1.012x** | Taking GDN *off* AITER costs nothing in the baseline - if anything GDN-on-AITER is slightly negative (A vs #5's own reproduction of row 3: 1.002x). |
| **B vs #6** | 14,393 / 14,406 | **0.999x** | Control holds: forcing the env var changes nothing when `--use-replayssm` already routes GDN away from AITER, so A and B are the same code path as rows 3 and 6. |
| **B vs A** | 14,393 / 12,658 | **1.137x** | ReplaySSM's in-cell gain in this config (TPOT 36.54 -> 31.72 ms, 1.152x), matching the 1.141x of the row 5 -> row 6 pair. |

**Verdict: there is no measurable headroom in routing ReplaySSM through the
AITER path, so the one-line gate is sufficient.** AITER's GDN contribution is
1.012x at best against row 3 and 1.002x against row 5 - the row-3 rep range
(12,514-12,650, excluding its 12,288 outlier) already brackets A's 12,629-12,687,
so GDN-on-AITER is worth zero within noise for the no-replay baseline. A
few-percent win for keeping aiter's fused conv1d+reshape over the generic conv
cannot be hiding inside a total-GDN effect that is ~0 (this cell bounds the
whole GDN-on-AITER effect - decode recurrence plus conv - and does not
decompose it), so the bigger "route ReplaySSM through the AITER path" change is
not justified by this model/config; the gate buys the full 1.137x - 1.141x on
its own.

Files: forcing patch `bench/gdn_aiter_gdn_off_forcing.patch`; server wrapper
gained `SPIKE_GDN_NO_AITER` (`bench/gdn4b_2x2_server.sh`); raw bench json
`bench/gdn4b_aiter_goff_{base,rs}_bs512_r{1,2}.json` (+ `_warmup_`), bench logs
`logs/gdn4b_aiter_bench_goff_*`, server logs `logs/gdn4b_goff_{base,rs}_server.log`.

## Files

Gate: `bench/gdn_aiter_exclude_gdn.patch` (uncommitted; sections 1-6, prints
reverted). Section 7 stacks `bench/gdn_aiter_gdn_off_forcing.patch` on top of it,
and those two are the whole diff in `qwen_gdn_linear_attn.py`.
Results: `AITER_EXCLUDE_GDN_summary.json`; raw bench json
`bench/gdn4b_aiter_aexg_{base,rs}_bs512_r{1,2}.json` (+ `_warmup_`),
bench logs `logs/gdn4b_aiter_bench_aexg_*`.
Server logs: `logs/gdn4b_aexg_{base,rs}_server.log` (timing + KV cache) and
`logs/gdn4b_aexg_{base,rs}_trace_server.log` (trace).
Correctness: `aexg_base_correctness.json`, `aexg_rs_correctness.json`,
`aexg_rs_vs_base_compare.json`.

## 8. Follow-up: the hybrid cell (AITER fused conv + ReplaySSM recurrence)

§7 bounded the whole GDN-on-AITER effect at ~0 and could not decompose it. The
decomposition was then measured directly by keeping AITER's fused
conv1d+reshape in `_forward_core_decode_aiter` and swapping only the recurrence
for the ReplaySSM kernel: **14,300.7 vs 14,274.9 out tok/s, 1.002x in-cell, with
bit-identical outputs (8/8 prompts, max |dlogp| 0.000)**. AITER's conv is worth
nothing measurable, so the one-line gate remains the right shape of the fix and
routing GDN decode through AITER
is not worth writing upstream. Implementation, trace evidence and the full
timing table: `AITER_ROUTE_REPLAY_ROCM.md`; patch
`bench/gdn_aiter_route_replay.patch`.
