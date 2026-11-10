# AITER fused conv + ReplaySSM recurrence: the hybrid cell

**Verdict: the hybrid works, is bit-identical to the control, and is worth
nothing.** AITER's fused conv1d+reshape kept in front of the ReplaySSM
recurrence measures **14,300.7 out tok/s (mean of 6) vs 14,274.9 for the
generic conv + ReplaySSM control**, i.e. **1.002x** in-cell with the sign
flipping across rep pairs and both ranges overlapping. The control is itself
**~0.9% below the same config measured earlier** (#6 / B, 14,393-14,406), so the
session-to-session spread is larger than the effect being chased. Routing GDN
decode through the AITER path is not worth writing upstream; the one-line gate
of `AITER_EXCLUDE_GDN_ROCM.md` buys the same throughput.

GPU 6 exclusive, one server at a time, timebox 90 min. Flags identical to every
cell in `AITER_EXCLUDE_GDN_ROCM.md` (Qwen3.5-4B snapshot 851bf6e8, conc 512,
512-in/128-out, 2048 prompts, `--max-model-len 4096 --mamba-ssm-cache-dtype
float32 --mamba-cache-mode none --no-prefix-caching --gpu-memory-utilization
0.80 --use-replayssm --replayssm-buffer-len 16`, `VLLM_ROCM_USE_AITER=1`).

## 1. The change: `bench/gdn_aiter_route_replay.patch`

Stacked on the two existing uncommitted experiment patches (gate +
`SPIKE_GDN_NO_AITER` forcing); all three are the only diff in
`qwen_gdn_linear_attn.py`. Two edits:

1. The gate becomes env-controlled so one tree serves both cells:

```python
SPIKE_GATE_GDN = os.environ.get("SPIKE_GATE_GDN", "1") == "1"
...
        if (
            GDN_AITER_TRITON_AVAILABLE
            and not self.enable_fused_gdn_spec_decode
            and not (SPIKE_GATE_GDN and getattr(self, "use_replayssm", False))
            and not SPIKE_GDN_NO_AITER
        ):
```

`SPIKE_GATE_GDN=0` means "do not gate, take the AITER path even with
`--use-replayssm`". Unset keeps the previous behaviour exactly, so no earlier
cell is invalidated by this patch.

2. `_forward_core_decode_aiter` keeps step 1
(`gdn_aiter_fused_reshape_causal_conv1d_update_single_token`, which already
returns `mixed_qkv_non_spec, b, a`) and swaps step 2 for the ReplaySSM
recurrence:

```python
        # 2. Recurrent attention
        if getattr(self, "use_replayssm", False):
            num_actual_tokens = attn_metadata.num_actual_tokens
            fused_recurrent_gated_delta_rule_replayssm(
                mixed_qkv=mixed_qkv_non_spec, a=a, b=b,
                A_log=self.A_log, dt_bias=self.dt_bias,
                scale=self.head_k_dim**-0.5, initial_state=ssm_state,
                d_cache=self.kv_cache[2], k_cache=self.kv_cache[3],
                g_cache=self.kv_cache[4],
                out=core_attn_out[:num_actual_tokens].unsqueeze(1),
                ssm_state_indices=non_spec_state_indices_tensor[:num_actual_tokens],
                write_pos=attn_metadata.write_pos_d,
                is_flush=attn_metadata.is_flush_d,
                use_qk_l2norm_in_kernel=True,
            )
            return
        gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule(...)  # unchanged
```

Everything the brief flagged needed no adaptation, and **no assert or wrapper
check fired on the first attempt**:

* Step 1 already returns the *flat* `[q_all | k_all | v_all]` conv output
  (`out = torch.empty((num_actual_tokens, dim, 1))`, `qkvz_layout="flat"`), which
  is byte-for-byte the layout `_forward_core_decode_non_spec_cached` feeds the
  same kernel, so no `rearrange_mixed_qkv` is needed. It is contiguous with
  `stride(-1) == 1`, so the wrapper's stride checks pass.
* Step 1 also returns `b`, `a` as `(num_actual_tokens, num_v_heads)`
  `ba.dtype` tensors and copies `z` into `z_out`, so the output-gated RMSNorm in
  `_output_projection` still sees z - the recurrence kernel must *not* apply the
  z gate, and it does not.
* `core_attn_out` needs no reshape: the ReplaySSM `out` argument
  (`core_attn_out[:num_actual_tokens].unsqueeze(1)`, `(T,1,HV,V)`) addresses the
  same memory the AITER kernel wrote via `core_attn_out.reshape(-1)`.
* `ssm_state = self.kv_cache[1]` is already the 4-D `(slots, HV, V, K)` view the
  kernel's `initial_state.ndim != 4` check wants (dim-first/dim-second is handled
  by `is_conv_state_dim_first()` for `kv_cache[0]` only, and
  `_forward_core_rocm` only dispatches here when the conv state is *not*
  dim-first). `kv_cache[2:5]` are the same ring caches the cached path passes.
* I sliced `ssm_state_indices_tensor[:num_actual_tokens]`, mirroring the cached
  path, rather than passing the whole staged buffer; both are the same rows in a
  pure-decode batch.

Prefill is untouched: with `SPIKE_GATE_GDN=0` prefill still reaches
`_forward_core` through `_forward_core_rocm`, so prefill numerics are identical
in both cells and the comparison isolates the decode step.

## 2. Route evidence: the flag really changes the code path

`SPIKE_TRACE` prints (temporary, reverted) at the entry of
`_forward_core_decode_aiter`, in its new ReplaySSM branch, and at the entry of
`_forward_core_decode_non_spec_cached`; both servers `--enforce-eager`, one
32-token greedy generation each (`bench/gdn4b_trace.py`). Counts are layer-steps
(24 GDN layers), so they are directly comparable:

| `SPIKE_GATE_GDN` | `_forward_core_decode_aiter` | hybrid ReplaySSM branch in it | `_forward_core_decode_non_spec_cached` | log |
| --- | --- | --- | --- | --- |
| **0** (hybrid) | **96** | **96** | **0** | `logs/gdn4b_route_hybrid_trace_server.log` |
| 1 (control) | 0 | 0 | 96 | `logs/gdn4b_route_ctrl_trace_server.log` |

No crossover in either direction, and each `_forward_core_decode_aiter` line is
immediately followed by its `hybrid replayssm in aiter (...)` line for the same
layer (`language_model.model.layers.{0,1,...,30}.linear_attn`). The gate env var
is read at import time, so it cannot flip mid-run.

## 3. Correctness: bit-identical, which is the surprise

8 prompts x 64 greedy tokens, `temperature=0`, `logprobs=0`, same server config
as the bench, control written first then hybrid,
`bench/gdn4b_route_{ctrl,hybrid}_correctness.json` ->
`bench/gdn4b_route_hybrid_vs_ctrl_compare.json`.

| run | token-identical | max abs dlogp | mean abs dlogp | positions |
| --- | --- | --- | --- | --- |
| **hybrid vs control (this session)** | **8 / 8** | **0.000** | **0.000** | 512 |

Not "a small delta" - **zero**: all 8 logprob vectors are identical element for
element (`-0.06121179834008217` in both, full float32 print precision), so every
converted bf16 activation in every one of the 24 GDN layers is unchanged. Two
different conv kernels (AITER fused reshape kernel vs vLLM's
`causal_conv1d_update`) agreeing bit-for-bit means both accumulate the 4-tap
convolution in fp32 in the same order and the same bias/silu rounding, so the
bf16 store - and everything downstream - is identical. Useful consequence: this
rules out a layout, head-mapping or scale bug in the hybrid by construction; the
`qkvz_layout="flat"` reading of step 1 and the `b`/`a` split are provably the
same data the generic path builds.

Context for the reader: the earlier #6-vs-#5 pair (`AITER_EXCLUDE_GDN_ROCM.md`
§4) showed 4/8 and 2.474 because there the *recurrence* changed (ReplaySSM vs
the AITER delta-rule kernel). Here the recurrence is the same kernel in both
cells, so bit-exactness is the expected outcome of a correct plumbing, and a
non-zero delta would have been the alarm.

## 4. Timing (GPU 6, conc 512, 512-in/128-out, 2048 prompts, 6 reps each)

One discarded warmup per server; reps taken from 2 server starts per cell,
interleaved in time with the other cell so clock drift does not sit on one side.
2048/2048 requests and **0 failures** in all 12 reps.

| cell | `SPIKE_GATE_GDN` | GDN conv | recurrence | out tok/s (chronological) | mean | median | TPOT p50 ms (median) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **HYBRID** | 0 | **AITER fused** | ReplaySSM | 14342.0 / 14327.1 / 14330.8 / 14366.7 / 14208.2 / 14229.8 | **14,300.7** | 14,328.9 | **32.01** |
| **CONTROL** | 1 | generic | ReplaySSM | 14331.4 / 14279.2 / 14307.8 / 14159.9 / 14264.2 / 14306.9 | **14,274.9** | 14,293.1 | **32.00** |

Discarded warmups: 14,332.5 (hybrid), 14,342.4 (control) - both inside their
cells' ranges. Rep spread 1.1% (hybrid) and 1.2% (control); median absolute
deviation 0.18% / 0.15%. GPU KV cache tokens: 4,449,745 (hybrid) vs 4,450,490
(control), i.e. the same ring allocation, -0.017%.

Per-rep paired ratios, in measurement order:
1.0007, 1.0034, 1.0016, 1.0146, 0.9961, 0.9946 -> **median 1.0012**.

### The three comparisons asked for

| comparison | out tok/s | ratio | read |
| --- | --- | --- | --- |
| **HYBRID vs CONTROL** (in-cell) | 14,300.7 / 14,274.9 | **1.002x** | AITER's fused conv is worth ~0. TPOT p50 32.01 vs 32.00 ms. |
| **HYBRID vs A** (`AITER_EXCLUDE_GDN_ROCM.md` §7: aiter on, GDN generic, **no** replay, 12,658) | 14,300.7 / 12,658 | **1.130x** | Same statement as the gate's own 1.137x (#6 vs A, 14,406/12,658) measured one session earlier; the hybrid buys nothing on top of the gate. |
| **HYBRID vs #2** (aiter off, replay, 13,975) | 14,300.7 / 13,975 | **1.023x** | Keeps the +2-3% that AITER-for-everything-else supplies; the control scores the same cross-session ratio (14,274.9/13,975 = 1.021x). |

Cross-session drift is the honest caveat: CONTROL here is 14,274.9 against the
14,392.7 / 14,406.2 of the identical `goff_rs` / `aexg_rs` cells measured
earlier on the same GPU, i.e. **-0.6% to -0.9%**, and both of *my* cells carry
that offset. The in-cell ratio (1.002x) is therefore the only number in this
document that is trustworthy to better than ~1%, which is exactly why the
verdict is "no difference" rather than "0.2% faster".

## 5. Why the conv fusion buys nothing (mechanism, one paragraph)

At conc 512 a decode step is ~32 ms of TPOT across the whole 32-layer model
(~1 ms/step/layer), and the GDN layer's cost in that step is dominated by the
recurrence's HBM traffic: the ring caches (d/k/g) plus the checkpoint state.
What step 1 saves relative to the generic path - a separate
`prepare_gdn_attention_core_inputs` reshape/z-copy plus one `causal_conv1d_update`
launch - is a handful of microseconds of a 1 ms layer step, and it is inside a
CUDA graph in both cells, so the host-side Python saving never reaches the GPU
timeline. No per-op profile was taken; this is arithmetic on the measured
~0.0-0.3% delta, not a claim of an ablation.

## 6. Verdict

**Do not write the routing upstream.** Keeping AITER's fused conv1d+reshape in
front of the ReplaySSM recurrence changes nothing measurable (1.002x in-cell,
sign unstable across rep pairs, both cells' ranges overlapping and each cell's
own spread 5x the point estimate). The reason to route GDN decode through AITER
was never the conv - it was the projection+recurrence fusion, and the recurrence
has to leave AITER for ReplaySSM anyway. The one-line `use_replayssm` gate of
`AITER_EXCLUDE_GDN_ROCM.md` already collects the whole 1.13x-1.14x, is a smaller
change, and is the honest shape of the fix. Building the hybrid was still the
right experiment: it converts "aiter's conv over the generic conv cannot be
hiding more than a few percent" (a bound, §7 of the exclude doc) into "it is
0.2% +/- 0.5%", and it proves the plumbing is bit-exact, which is a good
regression story for whichever route is eventually upstreamed.

## 7. Caveats

* 6 reps per cell but over 2 server starts each, and the session drifted ~1%;
  treat the difference as <0.5%, not as 1.002x.
* bs 512 only, one model (Qwen3.5-4B, 24 GDN of 32 layers), TP1, greedy
  correctness, no spec decode, no prefix caching. A conv-heavy shape (small
  batch, short context, where a step is much cheaper than 32 ms) is where a
  fused conv *could* matter, and it was not measured.
* The trace counts are layer-steps from one 32-token generation; the print
  buffer truncates (96 = 24 layers x 4 steps flushed), so the counts are a lower
  bound on the number of decode steps, not the number of steps executed. The
  routing statement does not depend on the absolute count.
* `SPIKE_TRACE` prints were reverted; the tree contains only the three
  experiment patches, uncommitted.

## Files

Tree `/app/local/yh/replayssm-gdn-wt`, branch `replayssm-gdn-rocm`, third patch
uncommitted (`git status`: `qwen_gdn_linear_attn.py`, `mamba_hybrid.py`,
`replayssm_utils.py` modified, nothing staged).

* `bench/gdn_aiter_route_replay.patch` - this change (verified to apply on top
  of `gdn_aiter_exclude_gdn.patch` + `gdn_aiter_gdn_off_forcing.patch` and to
  reproduce the tested file bit-identically).
* `bench/gdn4b_2x2_server.sh` - added `SPIKE_GATE_GDN` passthrough (default 1 =
  unchanged for every earlier cell).
* `bench/gdn4b_route_bench_extra.sh` - reps 3-6 for an existing cell tag.
* Bench json: `bench/gdn4b_aiter_route_{hybrid,ctrl}_bs512_r{1..6}.json`,
  `bench/gdn4b_aiter_warmup_route_{hybrid,ctrl}.json`;
  logs `logs/gdn4b_aiter_bench_route_{hybrid,ctrl}_bs512_r{1..6}.log` and
  `logs/gdn4b_aiter_bench_warmup_route_*.log`.
* Server logs: `logs/gdn4b_route_{hybrid,ctrl}_server.log`,
  `logs/gdn4b_route_{hybrid,ctrl}_trace_server.log`,
  `logs/gdn4b_route_{hybrid,ctrl}_r34_server.log`,
  `logs/gdn4b_route_{hybrid,ctrl}_r56_server.log`.
* Correctness: `bench/gdn4b_route_{ctrl,hybrid}_correctness.json`,
  `bench/gdn4b_route_hybrid_vs_ctrl_compare.json`.
