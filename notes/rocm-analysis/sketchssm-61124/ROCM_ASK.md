# Talking to the author about ROCm (draft text, not posted)

RECOMMENDATION: raise it in THIS PR, but only as a two-line portability fix
("do not hard-fail on non-CUDA, follow the ReplaySSM precision convention"), and
explicitly hand the ROCm config branch + measurements + any CI job to a separate
follow-up PR. Do not ask for "ROCm support" as a feature in this PR, and do not
put AMD into its docs/CI surface.

Style reference: the user's own recent comments (#61486, #64874) are 2-3
sentences, no headers, no bullet lists, friendly, and offer a PR. Keep it that
way.

## Short version (PR comment, ~6 lines) - recommended

Hi @omin-Kwon, thanks for this - I tried it on ROCm (MI355X, gfx950) and it is
two lines away from working. `_validate_sketchssm` rejects non-CUDA
(`is_cuda()` is False on ROCm, while the compute-capability check actually
passes there since gfx950 reports (9, 5)), and `SKETCH_DOT_PRECISION =
"tf32x3"` does not compile on HIP, where the allowed dot precisions are
ieee/bf16x3/bf16x6 (ReplaySSM already does `None if is_rocm() else "tf32x3"`).
With those changed, plus `is_cuda_alike()` in the kernel test instead of
`is_cuda()`, both kernel tests pass unchanged on gfx950 and one Mamba-2 layer is
4.2x faster than the default decode at batch 256 (2.3x end to end, 1.5x req/s
through the serving path).

Would you be open to taking those two changes, and letting an AMD config branch
+ measurements come in a follow-up PR? Asking because the Triton kernels are
already portable and I would rather not add tuning scope here. One thing that
came out of profiling on gfx950 and may be useful to you anyway: the flush
kernel is the steady-state bottleneck (any flush-row count >= 2 costs
~0.09-0.17 ms/step in our runs, which caps the window average at ~5x while an
all-non-flush step is 13x), and `num_warps=4 -> 1` nearly halves it.

## Shorter version, if the maintainers want the PR minimal

Same first two sentences as above, then: "Happy to send a patch with the
measurements if you want to take it here, or to carry it as a separate `[ROCm]`
PR on top of yours - whichever is easier for review."

## If the author declines / wants it separate

Then the ask goes to the AMD side: a small follow-up PR that (a) relaxes the
gate to `is_cuda_alike()` or at least does not hard-fail, (b) branches
`SKETCH_DOT_PRECISION` on the platform, (c) adds a `gfx950` branch in
`ops/sketchssm_config.py` (decode config transfers as-is, flush wants
`num_warps=1`), (d) makes the kernel test run on `is_cuda_alike()`, (e) adds an
MI355X kernel job. Points to make in favour: pure Triton, no new backend, the
sibling feature (ReplaySSM) has no platform gate, and the author's own fork
already branches on `is_rocm()` for GDN/KDA, so this is not asking the author to
maintain AMD hardware.

## Do NOT bundle these into the ROCm ask

1. The 0% prefix-cache hit rate we saw on the SketchSSM server
   (`Prefix cache hit rate: 0.0%` in every run vs 0.4% -> 74% on the baseline).
   That is platform-independent and could be our rebase; ask it as a separate
   correctness question, with the log excerpt.
2. Greedy divergence (3/8 prompts within 64 tokens). Expected of an approximate
   read, not ROCm-specific; if raised at all, raise it as "is the e2e test's
   2-prompt/32-token logprob-close check stable?" and not as an AMD issue.

## Numbers to attach if asked

* kernel tests: 2 passed on gfx950 unchanged after the precision line.
* per Mamba-2 layer, batch 128/256/512, graph-captured:
  2.48x / 4.16x / 5.54x vs stock decode; 1.05x / 1.80x / 2.19x vs ReplaySSM.
* end to end, Nemotron-3-Nano-4B, batch 256, 500 steps, offline:
  17.19 -> 7.57 ms/step (2.27x).
* serving, c=256, cold prefix cache: 36.07 -> 54.58 req/s (1.51x), TPOT p50
  22.62 -> 13.28 ms (1.70x), TTFT unchanged; c=32: 1.15x, TTFT p50 -10%.
