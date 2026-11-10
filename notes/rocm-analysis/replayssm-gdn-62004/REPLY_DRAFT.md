# Reply draft for PR #62004 (ReplaySSM GDN) - copy-paste, not posted

Focus: (1) it works on AMD, here are the numbers, (2) two small changes, (3) the
rebase looks like the gate to landing this at all - is it waiting on something,
and do you want help getting it reviewed.

No AITER mention, no AMD config/CI offer, no launch-config sweep (deps: user's
call to raise those later - evidence kept under "Deferred").

Verified state of the PR as of writing: labels `performance, needs-rebase, qwen,
kimi, k3`, **no assignee, no human review** (only a bot comment), 3 comments.

---

## Draft A (recommended)

Hi @Johnny-Liou, I went through the ReplaySSM writeup when the Mamba-2 half
landed and really liked the idea - keeping the full update but amortising it via
a small input ring, and reconstructing the read from the checkpoint, is a neat
way to avoid the accuracy loss you get from compressing the state itself.

So I ran this on AMD (MI355X, gfx950) to see what it needs there. It works: your
three suites pass unchanged (80 + 12 kernel tests, 82 metadata builder, nothing
skipped), outputs match the baseline packed kernel to the same error level
(8.6e-4 max with the fp16 ring, 5.7e-8 with fp32 rings), and end to end on
Qwen3.5-4B it gives 1.09x / 1.14x / 1.14x at batch 64 / 256 / 512 (output tok/s,
TPOT p50 -10 to -14%, TTFT unchanged). That is roughly 80-88% of the H100 numbers
in your table, so the feature transfers.

Two small things from that. Worth pinning `input_precision` on the ring dot -
with fp32 `d`/`k` caches it defaults to tf32 on NVIDIA and IEEE fp32 on gfx950,
so the same source has different numerics per platform. And getting those numbers
meant porting to Model Runner V2, since main has no MRV1 any more: it came out at
about 25 lines and decodes end to end here, happy to send it as a patch to fold
into your rebase. One design detail worth knowing for it - `plan_replayssm_decode`
does not carry over to GDN as-is, it wants a `ring_pos` counter that GDN's fused
kernel has no index for, so the host-side cursor stays unless you want that
redesign.

More generally: with MRV1 gone, that rebase looks like the gate to this landing
at all. Is it waiting on something, or would it help if I made some noise about
it - the PR has no assignee and no reviewer yet, and I am happy to raise it in
#pr-reviews if that gets it picked up.

---

## Draft B (short)

Hi @Johnny-Liou, nice work on this - I liked the ReplaySSM idea enough to try it
on AMD (MI355X, gfx950), and it holds up: your suites pass there unchanged
(80 + 12 kernel, 82 metadata builder, no skips), numerics match the baseline
kernel (8.6e-4 fp16 ring, 5.7e-8 fp32), and end to end on Qwen3.5-4B it is worth
1.09x / 1.14x / 1.14x at bs 64 / 256 / 512 (TPOT p50 -10 to -14%).

Two things: pin `input_precision` on the ring dot (fp32 rings are tf32 on NVIDIA,
IEEE fp32 on gfx950); and I had to port it to MRV2 to measure anything, ~25 lines,
decoding end to end here - happy to send it. For that port, `plan_replayssm_decode`
needs a `ring_pos` counter GDN's kernel does not have, so the host-side cursor
stays.

With MRV1 gone the rebase looks like the gate to landing this - is it waiting on
something? Happy to raise it in #pr-reviews if that helps it get a reviewer.

---

## Numbers to attach only if asked

e2e, Qwen3.5-4B, 24 GDN layers of 32, bf16, FP32 SSM state, L=16, MRV2 CUDA
graphs, prefix caching off, 512-in/128-out, 2048 prompts, 2 reps (spread
<= 0.5%), MI355X:

| conc | baseline | ReplaySSM (shipped cfg) | ratio |
| --- | --- | --- | --- |
| 64 | 7,820 tok/s, TPOT 6.46 ms | 8,504, 5.78 ms | 1.087x |
| 256 | 11,295, 19.27 ms | 12,815, 16.63 ms | 1.135x |
| 512 | 12,328, 37.24 ms | 14,105, 32.22 ms | 1.144x |

* correctness: baseline-vs-baseline 8/8 identical, 0.0 dlogp (proves the feature
  was on the decode path); ReplaySSM 6/8 identical, mean |dlogp| 0.097 nats,
  differences are greedy near-tie flips.
* small batch is not the target: isolated kernel is 0.71x at bs 1 and 0.87x at
  bs 8, in line with the PR's own "break-even at bs 1 to 8".
* isolated kernel, one GDN layer, bs 512: 0.4336 -> 0.4112 ms (1.05x). A gfx950
  launch config `(16,1,1,2)` gives 1.63x at kernel level but <0.6% e2e, so not
  proposed.
* numerics vs float64: output 8.6e-4 (both kernels, fp16 ring); state at flush
  5.2e-4 ReplaySSM vs 1.8e-7 baseline; with fp32 rings output 5.7e-8.
* writeups: `issue-workdirs/pr62004-replayssm-gdn-rocm/` (`TRACE.md`,
  `REPLAYSSM_GDN_ROCM.md`, `GDN_E2E_4B_ROCM.md`, `mrv2_spike.patch`).

---

## Deferred - do NOT put in this reply (raise later)

* **AITER**: with `VLLM_ROCM_USE_AITER=1` (default off; the GDN kernels are
  importable in our image) `--use-replayssm` is a silent no-op - decode-only
  batches enter `_forward_core_decode_aiter`, which reads only `kv_cache[0:2]`,
  so rings are allocated for nothing (8.3% of KV) and output is bit-identical to
  baseline (12,921 vs 14,393 tok/s). One-line gate on the aiter branch in
  `forward_hip` (`and not self.use_replayssm`) fixes it: 1.137x in-cell,
  TPOT 36.5 -> 31.7 ms. Routing replay through the aiter kernel is NOT worth it:
  aiter's GDN path measures ~0 (forcing GDN off aiter = 1.012x; the hybrid
  aiter-conv + replay is bit-identical to generic conv and 1.002x). Evidence:
  `AITER_PATH_ROCM.md`, `AITER_EXCLUDE_GDN_ROCM.md`, `AITER_ROUTE_REPLAY_ROCM.md`,
  `bench/gdn_aiter_*.patch`.
* **AMD config + CI**: a gfx950 entry in `_gdn_decode` (worth <0.6% e2e) and a
  kernel test job on MI355X (worth doing: it would have caught the tf32x3 issue
  in #61124 and the aiter no-op here).

## Reviewers to consider pinging (for Slack #pr-reviews, not in the comment)

From 40 recently merged PRs whose titles touch mamba/GDN/ReplaySSM, the humans
reviewing most: `askliar`, `ZJY0516`, `benchislett`, `tomeras91`, `iseeyuan`,
`gau-nernst`. The PR currently has no assignee and no human review, so a short
note in #pr-reviews naming it (plus the fact that it needs a rebase) is the
normal way to get it picked up.

## Do not bundle

1. No AITER, no AMD config/CI, no launch-config sweep - all deferred by decision.
2. Do not offer a follow-up PR for the two fixes; they belong in their rebase.
3. The fp16-ring checkpoint drift (5.2e-4) stays out unless they ask about
   long-context accuracy.
4. Do not call the MRV2 spike a port. Say: ~25 lines, decodes end to end, not
   polished, prefix-cache re-anchoring not handled.
