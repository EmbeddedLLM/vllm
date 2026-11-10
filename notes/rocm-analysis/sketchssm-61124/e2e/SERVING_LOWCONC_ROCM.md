# SketchSSM at LOW concurrency through the OpenAI serving path (ROCm MI355X, gfx950)

Companion to `SERVING_ROCM.md` (which covers c=32 and c=256). This file measures
c = 1, 2, 4, 8: the regime where a single decode step has one to eight rows and the
per-step SketchSSM work is not amortised over a large batch.

- Hardware: 2x AMD MI355X (gfx950, 288 GiB HBM). Baseline on GPU 6 / port 8056,
  SketchSSM on GPU 7 / port 8057. Both GPUs showed 0% busy and no other tenants
  during the runs (`rocm-smi --showuse`).
- Software: vLLM worktree `/app/local/yh/sketchssm-e2e-wt` at `83e25b78ec`
  ("Merge PR #61124 (SketchSSM Mamba-2 decode) into main for ROCm porting"),
  torch 2.13.0+rocm10.0.0, ROCm-SMI 4.0.0 / lib 7.8.0.
- Model `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` served as `nemotron`
  (42 layers = 21 Mamba-2 + 4 attention + 17 MLP, 96 heads, head_dim 80, dstate 128,
  8 groups, vocab 131072).
- Runs: 2026-11-09 10:00 - 10:12 UTC. Both servers were already running and were
  left running. The two servers were never driven concurrently; every run is
  sequential, baseline and SketchSSM alternating per concurrency.

## Headline

SketchSSM is **slower** at every concurrency measured here (c=1 to c=8) on this
stack, and the penalty shrinks as the batch grows:

| concurrency | baseline req/s | sketchssm req/s | ratio (req/s) | baseline TPOT p50 | sketchssm TPOT p50 | ratio (TPOT) |
|---|---|---|---|---|---|---|
| 1 | 1.81 | 1.50 | **0.83x** | 2.08 ms | 2.52 ms | **0.82x** |
| 2 | 3.28 | 2.81 | **0.86x** | 2.24 ms | 2.61 ms | **0.86x** |
| 4 | 5.81 | 5.21 | **0.90x** | 2.50 ms | 2.81 ms | **0.89x** |
| 8 | 9.56 | 9.07 | **0.95x** | 2.96 ms | 3.12 ms | **0.95x** |

Each cell is the mean of two runs with different fresh seeds (64 prompts per run).
Ratio > 1 means SketchSSM better. Trend against the earlier high-concurrency numbers
in `SERVING_ROCM.md`: 0.83x (c=1), 0.86x (c=2), 0.90x (c=4), 0.95x (c=8),
1.15x (c=32), 1.51x (c=256) - the crossing point is between c=8 and c=32 for
request throughput, and coincidentally similar for TPOT (TPOT is already 1.20x better
at c=32 while it is 0.95x worse at c=8).

Not apples-to-apples caveats are collected in the last section; the one that matters
for the numbers above is a ~1.5%-of-prompt-tokens prefix-cache asymmetry that
**favours the baseline**, i.e. the SketchSSM penalty here is if anything slightly
understated.

## Per concurrency (baseline vs SketchSSM, mean of 2 runs)

### concurrency 1

| metric | baseline | sketchssm | sketchssm vs baseline (>1 = sketchssm better) |
|---|---|---|---|
| request throughput (req/s) | 1.81 | 1.50 | 0.83x |
| output token throughput (tok/s) | 463.9 | 384.5 | 0.83x |
| TTFT mean (ms) | 21.5 | 23.0 | 0.93x |
| TTFT p50 (ms) | 21.2 | 22.6 | 0.94x |
| TTFT p99 (ms) | 26.7 | 28.2 | 0.95x |
| TPOT mean (ms) | 2.08 | 2.52 | 0.82x |
| TPOT p50 (ms) | 2.08 | 2.52 | 0.83x |
| TPOT p99 (ms) | 2.11 | 2.56 | 0.83x |
| ITL p50 (ms) | 2.08 | 2.50 | 0.83x |
| ITL p99 (ms) | 2.21 | 2.97 | 0.74x |
| benchmark duration (s) | 35.32 | 42.62 | 0.83x |

### concurrency 2

| metric | baseline | sketchssm | sketchssm vs baseline (>1 = sketchssm better) |
|---|---|---|---|
| request throughput (req/s) | 3.28 | 2.81 | 0.86x |
| output token throughput (tok/s) | 838.7 | 720.6 | 0.86x |
| TTFT mean (ms) | 38.2 | 44.8 | 0.85x |
| TTFT p50 (ms) | 37.4 | 44.6 | 0.84x |
| TTFT p99 (ms) | 50.2 | 50.9 | 0.99x |
| TPOT mean (ms) | 2.24 | 2.61 | 0.86x |
| TPOT p50 (ms) | 2.24 | 2.61 | 0.86x |
| TPOT p99 (ms) | 2.27 | 2.64 | 0.86x |
| ITL p50 (ms) | 2.25 | 2.56 | 0.88x |
| ITL p99 (ms) | 2.34 | 3.06 | 0.76x |
| benchmark duration (s) | 19.53 | 22.74 | 0.86x |

### concurrency 4

| metric | baseline | sketchssm | sketchssm vs baseline (>1 = sketchssm better) |
|---|---|---|---|
| request throughput (req/s) | 5.81 | 5.21 | 0.90x |
| output token throughput (tok/s) | 1488.6 | 1333.2 | 0.90x |
| TTFT mean (ms) | 47.4 | 51.0 | 0.93x |
| TTFT p50 (ms) | 48.6 | 51.3 | 0.95x |
| TTFT p99 (ms) | 56.3 | 59.2 | 0.95x |
| TPOT mean (ms) | 2.51 | 2.81 | 0.89x |
| TPOT p50 (ms) | 2.50 | 2.81 | 0.89x |
| TPOT p99 (ms) | 2.57 | 2.86 | 0.90x |
| ITL p50 (ms) | 2.50 | 2.70 | 0.93x |
| ITL p99 (ms) | 2.83 | 4.14 | 0.68x |
| benchmark duration (s) | 11.01 | 12.29 | 0.90x |

### concurrency 8

| metric | baseline | sketchssm | sketchssm vs baseline (>1 = sketchssm better) |
|---|---|---|---|
| request throughput (req/s) | 9.56 | 9.07 | 0.95x |
| output token throughput (tok/s) | 2447.5 | 2322.2 | 0.95x |
| TTFT mean (ms) | 74.4 | 79.7 | 0.93x |
| TTFT p50 (ms) | 79.3 | 83.0 | 0.95x |
| TTFT p99 (ms) | 86.9 | 92.3 | 0.94x |
| TPOT mean (ms) | 2.99 | 3.14 | 0.95x |
| TPOT p50 (ms) | 2.96 | 3.12 | 0.95x |
| TPOT p99 (ms) | 3.12 | 3.27 | 0.95x |
| ITL p50 (ms) | 2.97 | 3.00 | 0.99x |
| ITL p99 (ms) | 3.12 | 4.97 | 0.63x |
| benchmark duration (s) | 6.69 | 7.06 | 0.95x |

### Per-run values (the 2 reps behind the means)

| mode | conc | rep | seed | duration (s) | req/s | out tok/s | TTFT mean | TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | run prefix-cache hit rate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | rep1 | 41001 | 35.35 | 1.81 | 463.5 | 21.5 | 21.0 | 28.3 | 2.08 | 2.08 | 2.13 | 2.08 | 2.25 | 0.000% |
| sketchssm | 1 | rep1 | 41002 | 42.64 | 1.50 | 384.3 | 23.2 | 23.1 | 26.6 | 2.52 | 2.52 | 2.56 | 2.50 | 2.97 | 0.000% |
| baseline | 1 | rep2 | 42001 | 35.28 | 1.81 | 464.3 | 21.6 | 21.4 | 25.1 | 2.08 | 2.08 | 2.10 | 2.08 | 2.16 | 0.000% |
| sketchssm | 1 | rep2 | 42002 | 42.60 | 1.50 | 384.6 | 22.9 | 22.1 | 29.7 | 2.52 | 2.52 | 2.55 | 2.50 | 2.96 | 0.000% |
| baseline | 2 | rep1 | 41003 | 19.51 | 3.28 | 839.6 | 38.4 | 37.0 | 59.9 | 2.24 | 2.24 | 2.27 | 2.24 | 2.35 | 0.000% |
| sketchssm | 2 | rep1 | 41004 | 22.74 | 2.81 | 720.4 | 44.9 | 44.6 | 51.5 | 2.61 | 2.61 | 2.64 | 2.56 | 3.06 | 0.000% |
| baseline | 2 | rep2 | 42003 | 19.55 | 3.27 | 837.8 | 37.9 | 37.8 | 40.6 | 2.25 | 2.25 | 2.27 | 2.25 | 2.33 | 1.489% |
| sketchssm | 2 | rep2 | 42004 | 22.73 | 2.82 | 720.8 | 44.7 | 44.6 | 50.4 | 2.61 | 2.61 | 2.64 | 2.56 | 3.06 | 0.000% |
| baseline | 4 | rep1 | 41005 | 11.03 | 5.80 | 1485.2 | 48.4 | 49.6 | 56.5 | 2.51 | 2.51 | 2.56 | 2.50 | 2.68 | 0.000% |
| sketchssm | 4 | rep1 | 41006 | 12.27 | 5.21 | 1335.0 | 50.0 | 50.5 | 59.1 | 2.81 | 2.81 | 2.84 | 2.70 | 4.14 | 0.000% |
| baseline | 4 | rep2 | 42005 | 10.98 | 5.83 | 1491.9 | 46.4 | 47.6 | 56.1 | 2.51 | 2.50 | 2.57 | 2.50 | 2.99 | 0.000% |
| sketchssm | 4 | rep2 | 42006 | 12.31 | 5.20 | 1331.4 | 52.0 | 52.1 | 59.3 | 2.81 | 2.81 | 2.88 | 2.70 | 4.13 | 0.000% |
| baseline | 8 | rep1 | 41007 | 6.69 | 9.56 | 2448.6 | 72.9 | 78.4 | 87.9 | 2.99 | 2.96 | 3.12 | 2.97 | 3.07 | 1.489% |
| sketchssm | 8 | rep1 | 41008 | 7.05 | 9.08 | 2325.2 | 79.7 | 83.3 | 91.4 | 3.14 | 3.12 | 3.27 | 2.99 | 4.97 | 0.000% |
| baseline | 8 | rep2 | 42007 | 6.70 | 9.56 | 2446.4 | 75.8 | 80.1 | 85.9 | 2.98 | 2.96 | 3.13 | 2.98 | 3.17 | 1.489% |
| sketchssm | 8 | rep2 | 42008 | 7.06 | 9.06 | 2319.3 | 79.6 | 82.7 | 93.3 | 3.15 | 3.13 | 3.28 | 3.00 | 4.98 | 0.000% |

Regenerate this whole document's tables with `python3 summarize_lowconc.py`.

Rep-to-rep spread is at most 0.03 req/s (c=4 baseline: 5.80 then 5.83) and at most
0.01 ms on TPOT p50 (c=1 baseline: 2.08 both times), so the smallest between-server
difference reported below (5% at c=8) is a full order of magnitude above the noise.

Two observations worth keeping:

- ITL p99 is 1.4-1.6x worse for SketchSSM at c=4 and c=8 (4.14/4.13 and 4.97/4.98 ms
  vs 2.83/2.99 and 3.12/3.17 ms), reproducibly across reps, while ITL p50 is
  essentially unchanged at c=8 (3.00 vs 2.97). It is not the W=16 flush cadence:
  a flush every 16 decode steps would move p50, not only the tail. `bench serve`
  keeps only percentile values here (the JSON has `latencies`/`queue_times`, no
  per-token list), so the tail could not be attributed further from this data.
- TTFT is slightly worse for SketchSSM at every concurrency (0.93-0.95x, i.e. 1.5-6.6 ms).
  Prefill uses the same attention/MLP path as baseline; this is consistent with the
  extra work per *layer* in the first forward pass rather than a prefill-specific cost.

## c=1: how much of the wall clock is prefill? (derived)

**Method.** vLLM reports `TPOT = (e2e - TTFT) / (output_len - 1)`, so a single request
occupies `TTFT + 255 * TPOT`. Both terms are taken from the client JSON means of the
same run; the prefill share is `TTFT / (TTFT + 255 * TPOT)`. Everything in this
section is **derived**, not directly measured. Cross-check: `64 * per_request /
concurrency` reproduces the measured benchmark duration to within 0.03 s in all 8
runs, so the decomposition is consistent.

| concurrency | mode | derived per-request wall | derived prefill share | predicted vs measured duration |
|---|---|---|---|---|
| 1 | baseline | 551.7 ms | **3.9%** | 35.31 s vs 35.32 s |
| 1 | sketchssm | 665.7 ms | **3.5%** | 42.61 s vs 42.62 s |
| 2 | baseline | 610.3 ms | 6.3% | 19.53 s vs 19.53 s |
| 2 | sketchssm | 710.4 ms | 6.3% | 22.73 s vs 22.74 s |
| 4 | baseline | 687.7 ms | 6.9% | 11.00 s vs 11.01 s |
| 4 | sketchssm | 767.9 ms | 6.6% | 12.29 s vs 12.29 s |
| 8 | baseline | 836.3 ms | 8.9% | 6.69 s vs 6.69 s |
| 8 | sketchssm | 881.4 ms | 9.0% | 7.05 s vs 7.06 s |

So at c=1 the wall clock is **decode-bound, not prefill-bound**: 256 output tokens at
2.08 ms (baseline) / 2.52 ms (sketchssm) is 530 ms / 643 ms of a 552 ms / 666 ms
request, while the 1024-token prefill is ~21 ms / ~23 ms (3.5-3.9%). The prefill share
grows with concurrency because batching makes the per-step decode time grow faster
than the prefill time. The c=1 gap between the two servers is therefore almost
entirely a per-decode-step effect: +0.44 ms/token, which is what the single-layer
kernel measurement below explains.

## Prefix-cache confirmation (was either server warm?)

Each run got its own fresh `--seed`, so no run can be served from prompts an earlier
run cached. The evidence per run is the **`/metrics` counter delta around the run**
(`vllm:prefix_cache_hits_total` / `vllm:prefix_cache_queries_total`), which isolates
that run; the engine log line is printed every 10 s and is a cumulative average since
process start, so on its own it cannot isolate one run (and it is further diluted by
the warm-cache control run below). Both are shown.

| mode | conc | rep | seed | /metrics queries | /metrics hits | run hit rate | last log line before the run | first log line after the run |
|---|---|---|---|---|---|---|---|---|
| baseline | 1 | rep1 | 41001 | 65536 | 0 | 0.000% | 11-09 00:25:27: 0.4% | 11-09 10:01:37: 0.3% |
| sketchssm | 1 | rep1 | 41002 | 65536 | 0 | 0.000% | 11-09 00:25:39: 0.0% | 11-09 10:02:30: 0.0% |
| baseline | 1 | rep2 | 42001 | 65536 | 0 | 0.000% | 11-09 10:06:37: 10.9% | 11-09 10:08:47: 9.8% |
| sketchssm | 1 | rep2 | 42002 | 65536 | 0 | 0.000% | 11-09 10:05:30: 0.0% | 11-09 10:09:40: 0.0% |
| baseline | 2 | rep1 | 41003 | 65536 | 0 | 0.000% | 11-09 10:01:47: 0.3% | 11-09 10:03:27: 0.2% |
| sketchssm | 2 | rep1 | 41004 | 65536 | 0 | 0.000% | 11-09 10:02:40: 0.0% | 11-09 10:04:00: 0.0% |
| baseline | 2 | rep2 | 42003 | 65536 | 976 | 1.489% | 11-09 10:08:57: 9.8% | 11-09 10:10:07: 9.0% |
| sketchssm | 2 | rep2 | 42004 | 65536 | 0 | 0.000% | 11-09 10:09:50: 0.0% | 11-09 10:10:40: 0.0% |
| baseline | 4 | rep1 | 41005 | 65536 | 0 | 0.000% | 11-09 10:03:37: 0.2% | 11-09 10:04:27: 0.2% |
| sketchssm | 4 | rep1 | 41006 | 65536 | 0 | 0.000% | 11-09 10:04:10: 0.0% | 11-09 10:04:50: 0.0% |
| baseline | 4 | rep2 | 42005 | 65536 | 0 | 0.000% | 11-09 10:10:17: 9.0% | 11-09 10:10:57: 8.3% |
| sketchssm | 4 | rep2 | 42006 | 65536 | 0 | 0.000% | 11-09 10:10:50: 0.0% | 11-09 10:11:20: 0.0% |
| baseline | 8 | rep1 | 41007 | 65536 | 976 | 1.489% | 11-09 10:04:37: 0.2% | 11-09 10:05:07: 0.4% |
| sketchssm | 8 | rep1 | 41008 | 65536 | 0 | 0.000% | 11-09 10:05:00: 0.0% | 11-09 10:05:20: 0.0% |
| baseline | 8 | rep2 | 42007 | 65536 | 976 | 1.489% | 11-09 10:11:07: 8.3% | 11-09 10:11:37: 7.8% |
| sketchssm | 8 | rep2 | 42008 | 65536 | 0 | 0.000% | 11-09 10:11:30: 0.0% | 11-09 10:12:00: 0.0% |

Read this as: 13 of 16 runs are exactly 0 hit tokens out of exactly 65536 queried
prompt tokens (= 64 prompts x 1024), i.e. fully cold. The three 1.489% rows are
explained below; they are a dataset property, not a warm cache. The baseline's
cumulative log figure is in the 8-11% range because of the warm-cache control run
(62464 of its 64416 cumulative hit tokens at the time of the c=1 rep2 run) plus the
earlier duplicate-prompt collisions, not because any of the 16 measured runs was
warm.

### Why three baseline runs show exactly 976 hit tokens

976 = 1024 - 48 = 61 of the 64 sixteen-token blocks, which is also the maximum
matchable prefix of a 1024-token prompt (verified: the warm-cache control below hits
exactly 976 tokens per request). So each of those runs had exactly *one* request whose
whole prompt was already cached.

Cause: `--dataset-name random` does not sample i.i.d. tokens. `vllm/benchmarks/
datasets/datasets.py::RandomDataset` builds prompt `i` as
`(offset_i + i + arange(input_len)) % vocab_size` (vocab_size 131072), so two
independently seeded datasets can contain a byte-identical prompt; a duplicate hits
the block hash chain of the first copy. Verified directly for the rep2 c=2 run:
`RandomDataset(seed=42001)` and `RandomDataset(seed=42003)` share exactly one
identical prompt, and 42001 (baseline c=1) ran immediately before 42003 (baseline
c=2) on the same server. The other two rows are the same birthday collision against
that server's older cache content (datasets from before this file's runs, which were
not regenerated for comparison).

Consequence: this is birthday-random noise that favours whichever server happens to
draw a collision - here the baseline, in 3 of its 8 runs (and 0 of SketchSSM's 8).
Bound on the effect: 976 of 65536 prompt tokens = 1.5% of prompt tokens in those
runs, i.e. one of 64 requests gets its prefill from cache. One request out of 64 can
move the run's mean TTFT by at most ~1/64 of a prefill (the warm control below saves
36.7 ms of TTFT p50 when *all* 64 requests are cached, so ~0.6 ms for one), which is within
the rep-to-rep spread of mean TTFT (0.1-2.9 ms across the eight pairs). The direction is against SketchSSM,
so it does not explain the reported SketchSSM slowdown.

### Warm-cache control (deliberately not apples-to-apples)

To show what a genuinely warm baseline looks like, and to measure the maximum
matchable prefix, baseline c=8 seed 41007 was re-run immediately after its own cold
run with the same seed:

| | cold (rep1) | warm (same seed re-run) |
|---|---|---|
| run prefix-cache hit rate | 1.489% (one duplicate prompt, see above) | **95.312%** (62464 / 65536 tokens) |
| TTFT p50 | 78.4 ms | **41.7 ms** |
| req/s | 9.56 | 10.01 |
| TPOT p50 | 2.96 ms | 2.97 ms |

(`serving_bench_baseline_c8_warmprobe.json`; excluded from every table above.)
This is the trap the earlier high-concurrency report hit: a warm baseline would have
made SketchSSM look ~2x worse on TTFT at low concurrency. None of the measured runs
are in this state.

## Warm-cache A/B

Settles the open item from the section above: does the SketchSSM server serve from the
prefix cache at all? Same measurement on both servers - a 4x repeat of one identical
2670-token prompt through `/v1/completions` (temperature 0, 64 tokens, `--ignore-eos`),
then the seed-777 `bench serve` run twice in a row with the same seed (run 2 is warm by
construction). Deltas of `vllm:prefix_cache_{hits,queries}_total` around each step.

| server (mamba block) | single prompt x4 (same 2670-token prompt) | seed-777 run 1 | seed-777 run 2 |
|---|---|---|---|
| baseline (976) | 5856 / 10680 = **54.83%** (per send: 0, 1952, 1952, 1952 -> 73.11%) | 0 / 32769 = 0.000% | 31232 / 32769 = **95.31%** |
| SketchSSM (1056) | 6336 / 10680 = **59.33%** (per send: 0, 2112, 2112, 2112 -> 79.10%) | 0 / 32769 = 0.000% | 0 / 32769 = **0.000%** |

(cells are hits / queries = hit rate; a 32 x 1024-token run reports 32768 or 32769
queries, the occasional extra token is engine bookkeeping and appears on both servers)

SketchSSM **does** participate in prefix caching - it reuses 79.10% of a repeated
2670-token prompt, more than the baseline's 73.11% - and the 0% on the same-seed rerun
is a block-size artifact, not a caching failure: a 1024-token prompt is shorter than
SketchSSM's 1056-token block, so it contains no complete block and there is nothing to
cache, while the baseline's 976-token block does fit one. Hit tokens are exactly
`32 * floor(input_len / block) * block` on both servers:

| server (block) | random-input-len | run 1 hits / queries | run 2, same seed | hits / (32 x block) |
|---|---|---|---|---|
| SketchSSM (1056) | 1024 | 0 / 32768 | 0 / 32768 | 0 |
| SketchSSM (1056) | 2048 | 0 / 65537 | 33792 / 65537 | 1 |
| baseline (976) | 2048 | 0 / 65536 | 62464 / 65536 | 2 |

(fresh seeds per row: SketchSSM 794 and 793, baseline 795; `warmcache_probe_inlen.log`)

So the 0.0% the SketchSSM server logged all session, and the 0 hits in all 8 of its
1024-token benchmark runs, were an artifact of the prompt length sitting below its block
size - its `block_size` / `mamba_block_size` is 1056 tokens vs the baseline's 976 (from
`vllm:cache_config_info`; both servers run `--mamba-cache-mode align`, the difference
is not investigated here). The 8-run
equal-treatment argument in (2) and the TTFT/req-s conclusions are unaffected (both
servers were cold within any run either way), but "SketchSSM gets no prefix-cache hits"
is not a property of SketchSSM.

## One-layer kernel-level data point (batch 1/2/4/8)

`bench_graph_step_lowbatch.py` (adapted from `../bench_graph_step.py`): one Mamba-2
layer with the Nano-4B shape (96 heads, head_dim 80, dstate 128, 8 groups, W=16,
mean sketch rank 8, 512 state slots), CUDA-graph captured so CPU launch overhead is
excluded, 200 timed replays after 20 warmups, on GPU 5 (idle, untouched by the
serving runs). `std` = stock `selective_state_update`. SketchSSM step =
`_rotate_groups` (query rotation) + `sketch_decode` + `sketch_flush`:

| batch | stock SSU (ms/step) | sketch, 1 flush row/step (ms) | sketch, 0 flush rows (ms) | 1-flush/std | 0-flush/std |
|---|---|---|---|---|---|
| 1 | 0.00662 | 0.01677 | 0.01634 | **2.53x** | 2.47x |
| 2 | 0.00729 | 0.02217 | 0.02122 | **3.04x** | 2.91x |
| 4 | 0.01149 | 0.02230 | 0.02153 | **1.94x** | 1.87x |
| 8 | 0.01945 | 0.02193 | 0.02163 | **1.13x** | 1.11x |

Two flush variants are needed because the flush has to be amortised over W=16 steps
and batch <= 8 is less than one row per step: "1 flush row/step" is the convention
`bench_graph_step.py` uses for large batches (batch/16 rows per step) and is a 16x
over-flush here, i.e. an upper bound; "0 flush rows" is what 15 of every 16 steps
cost, a lower bound. The realised cost for a rolling batch of b sequences is between
them at b/16 flushed rows/step. The flush is cheap either way at these sizes
(delta 0.4 us at batch 1, 0.3 us at batch 8), so the conclusion does not depend on
which variant you pick.

Mechanism, and the cross-check against the serving numbers (**derived**): the model
has 21 Mamba-2 layers, so the measured per-layer deltas predict
`21 * 10.15 us = +213 us` per decode step at batch 1 (upper-bound variant) against
`+440 us` measured in TPOT at c=1 (2.52 - 2.08 ms). At batch 8 they predict
`21 * 2.48 us = +52 us` against `+150 us` measured (3.14 - 2.99 ms). The microbench
covers the decode kernels only (not the rest of the layer's SketchSSM bookkeeping),
and it is a single layer in isolation, so it explains roughly half of the serving
delta - the same order, and it reproduces the *direction* and the *shrinkage with
batch* that the end-to-end numbers show (kernel 2.5x -> 1.1x, TPOT 0.82x -> 0.95x).
The reason is that the two curves have different shapes: the stock step cost grows
linearly with batch (6.6 us at batch 1, 19.5 us at batch 8, 326.2 us at batch 128 in
`../bench_graph_step.json`, i.e. ~2.56 us per row) while the sketch step is
near-affine with a large fixed part (21.9 us at batch 8, 168.6 us at batch 128, i.e.
~1.22 us per row plus ~12.2 us). A line through those two points each puts the
per-layer crossing just above batch 8 (**derived**: 2.56*b = 12.2 + 1.22*b ->
b ~ 9.1), which is the same story the serving numbers tell between c=8 (0.95x) and
c=32 (1.15x): the sketch step is only competitive once a step carries more than ~10
rows.

## Exact commands

Each run (`run_lowconc_bench.sh <mode> <conc> <rep> <seed> <num-prompts>`):

```
BEFORE=$(curl -s http://127.0.0.1:$PORT/metrics | <grab prefix_cache_hits/queries>)
cd /app/local/yh/sketchssm-e2e-wt && PYTHONPATH=/app/local/yh/sketchssm-e2e-wt \
python3 -m vllm.entrypoints.cli.main bench serve \
  --backend openai --endpoint /v1/completions --base-url http://127.0.0.1:$PORT \
  --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --served-model-name nemotron --trust-remote-code \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --ignore-eos --num-prompts 64 --max-concurrency $C \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 --seed $SEED \
  --save-result --result-dir . --result-filename serving_bench_<mode>_c$C_<rep>.json
AFTER=$(curl -s http://127.0.0.1:$PORT/metrics | <same>)
```

`PORT` is 8056 for baseline and 8057 for SketchSSM. Runs, in this order (all with 64
prompts, the two servers never driven concurrently):

```
run_lowconc_bench.sh baseline  1 rep1 41001 ; run_lowconc_bench.sh sketchssm 1 rep1 41002
run_lowconc_bench.sh baseline  2 rep1 41003 ; run_lowconc_bench.sh sketchssm 2 rep1 41004
run_lowconc_bench.sh baseline  4 rep1 41005 ; run_lowconc_bench.sh sketchssm 4 rep1 41006
run_lowconc_bench.sh baseline  8 rep1 41007 ; run_lowconc_bench.sh sketchssm 8 rep1 41008
run_lowconc_bench.sh baseline  1 rep2 42001 ; run_lowconc_bench.sh sketchssm 1 rep2 42002
run_lowconc_bench.sh baseline  2 rep2 42003 ; run_lowconc_bench.sh sketchssm 2 rep2 42004
run_lowconc_bench.sh baseline  4 rep2 42005 ; run_lowconc_bench.sh sketchssm 4 rep2 42006
run_lowconc_bench.sh baseline  8 rep2 42007 ; run_lowconc_bench.sh sketchssm 8 rep2 42008
run_lowconc_bench.sh baseline  8 warmprobe 41007   # warm-cache control, excluded
```

Kernel point:

```
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=/app/local/yh/vllm \
  python3 bench_graph_step_lowbatch.py bench_graph_step_lowbatch.json
```

Server processes (unchanged, still running after these measurements): baseline
pid 155436 with `CUDA_VISIBLE_DEVICES=6` / port 8056, SketchSSM pid 145582 with
`CUDA_VISIBLE_DEVICES=7` / port 8057 (read from `/proc/<pid>/environ` and
`/proc/<pid>/fd/1`), identical flags and `VLLM_USE_V2_MODEL_RUNNER=1` apart from
`--sketchssm SketchSSM/Nemotron-3-Nano-4B-BF16 --sketchssm-mean-rank 8` (full command
lines in `MANIFEST.md` / `SERVING_ROCM.md`). Both were verified healthy (`/health`
200) before and after, and both logs keep growing in
`server_baseline_cold5.log` / `server_sketchssm_serve3.log`.

## Artifacts

| file | contents |
|---|---|
| `serving_bench_<mode>_c<N>_<rep>.json` | `bench serve --save-result` client JSON (16 runs + 1 warm control) |
| `serving_bench_<mode>_c<N>_<rep>.console.log` | verbatim client output of that run |
| `serving_bench_<mode>_c<N>_<rep>.cache.json` | run-scoped cache evidence: `/metrics` counters before/after and the delta, run start/end, server log file + line range |
| `run_lowconc_bench.sh` | the runner above (one run per invocation) |
| `summarize_lowconc.py` | rebuilds every table in this file from those JSONs |
| `driver_rep1_c1.log`, `driver_rep1_c2.log`, `driver_rep1_c4_c8.log`, `driver_rep2.log` | raw driver output for the 16 runs |
| `bench_graph_step_lowbatch.py`, `bench_graph_step_lowbatch.json` | the batch 1/2/4/8 kernel measurement |
| `warmcache_ab.py`, `warmcache_ab_<mode>.json` | the warm-cache A/B above: per-step counter deltas for both servers, plus each completion's prompt/output token counts |
| `warmcache_probe_inlen.sh`, `warmcache_probe_inlen.log` | the block-size probe: same-seed reruns at `--random-input-len` 1024 vs 2048 |
| `serving_bench_<mode>_warmcache_<run>.json/.console.log` | client JSON and console for the 4 seed-777 reruns of the A/B |

## What is not apples-to-apples

1. **Two servers, two GPUs.** Baseline on GPU 6, SketchSSM on GPU 7, run as separate
   processes on separate devices, alternating in time rather than simultaneously.
   Same MI355X model, both idle, no co-tenancy; this is the same setup as
   `SERVING_ROCM.md`.
2. **Prefix cache: 3 of 8 baseline runs got one duplicate prompt (1.489%), 0 of 8
   SketchSSM runs did** (mechanism above). It favours the baseline, and only in those
   runs; the headline ratios would move toward SketchSSM, not away, if it were zero
   everywhere. So SketchSSM's low-concurrency penalty is not a cache artifact.
3. **The SketchSSM server recorded 0 prefix-cache hits over the whole session**: its
   `/metrics` hit counter is 0 with 2883588 tokens queried at the end of these runs,
   and its engine log printed 0.0% throughout, so on that server prefix caching
   returned nothing in any run. Investigated in `## Warm-cache A/B` below: it is a
   prompt-length vs block-size artifact (1024 tokens fits no full 1056-token block),
   not a SketchSSM caching failure. With fresh
   seeds per run its effect on these numbers is bounded by (2); what actually differs
   between the two servers here is only the block size (976 vs 1056 tokens).
4. **The engine log's `Prefix cache hit rate` cannot isolate a run**: it is a
   cumulative average since engine start, printed every 10 s, and the log is
   block-buffered, so lines from an earlier run can appear in the file during a later
   run. Both are reported above; the per-run number comes from counter deltas. The
   baseline's printed figure is also inflated by the deliberate warm-cache control run.
5. **Rewritten (non-i.i.d.) prompts.** The `random` dataset's arithmetic token
   sequences plus a decode/re-encode round trip mean the prompts are not i.i.d. random
   tokens; identical prompts across seeds are possible (see (2)). Each run has its own
   seed, so the two servers of one concurrency/rep pair saw *different* prompts; they
   are statistically equivalent (identical length: exactly 1024 in and 256 out per
   request), and rep-to-rep agreement (<=0.6%) shows the residual prompt-set effect
   is far below the measured differences.
6. **Short runs at c=8** (6.7 s). The engine's 10 s stats interval spans a whole c=8
   run, which is why the log lines are uninformative there and only the counter deltas
   are used; the client-side percentiles are computed over all 16384 ITL samples, so
   the reported latencies are not affected.
7. **`--ignore-eos`, default sampling (temperature 1.0, top_p 1.0), 256 output tokens
   always.** Identical for both servers, so the comparison is fair, but the numbers
   are for this sampling regime and for a 4B model on 2 GPUs - not a throughput
   ceiling measurement.
8. **Kernel numbers are one layer in isolation on a third GPU**, CUDA-graph captured,
   with synthetic states; they are a mechanism data point, not a model-level number
   (see the two bounds and the cross-check above).
