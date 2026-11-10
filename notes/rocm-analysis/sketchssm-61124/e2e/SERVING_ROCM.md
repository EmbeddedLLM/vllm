# SketchSSM end-to-end SERVING throughput on ROCm (MI355X, gfx950)

Online serving measurement of PR #61124 (SketchSSM) through the OpenAI-compatible
server, compared against the unmodified baseline. Complements the offline
`LLM.generate` result already in `e2e_speedup_b256.log` (2.27x at batch 256).

- Hardware: 2x AMD MI355X (gfx950), 288 GiB HBM each. Baseline on GPU 6, SketchSSM on GPU 7.
- Setup: vLLM worktree `/app/local/yh/sketchssm-e2e-wt` (branch `sketchssm-rocm-e2e`,
  ROCm port of PR #61124), model `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16` served as `nemotron`.
- Client: `python3 -m vllm.entrypoints.cli.main bench serve` from the same worktree.
- Date of runs: 2026-11-08 23:56 - 2026-11-09 00:26.

## Headline

| | baseline | SketchSSM | sketchssm vs baseline |
|---|---|---|---|
| c=256, request throughput | 36.07 req/s | 54.58 req/s | **1.51x** |
| c=256, output token throughput | 9234 out tok/s | 13972 out tok/s | **1.51x** |
| c=256, TPOT p50 | 22.62 ms | 13.28 ms | **1.70x faster** |
| c=256, TTFT p50 | 1262.1 ms | 1261.6 ms | 1.00x (unchanged) |
| c=32, request throughput | 22.36 req/s | 25.79 req/s | **1.15x** |
| c=32, TPOT p50 | 4.86 ms | 4.06 ms | **1.20x faster** |
| c=32, TTFT p50 | 183.3 ms | 201.5 ms | 0.91x (**10% worse**) |

Both tables are cold-prefix-cache measurements (see "Prefix caching" below);
each cell is the mean of two runs with different seeds (256 prompts each).

## Server configuration (identical apart from `--sketchssm`)

Both servers were launched by `start_servers.sh align <suffix>` (see that file):

```
VLLM_USE_V2_MODEL_RUNNER=1 PYTHONPATH=/app/local/yh/sketchssm-e2e-wt \
CUDA_VISIBLE_DEVICES=6 python3 -m vllm.entrypoints.openai.api_server \
  --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --served-model-name nemotron \
  --max-model-len 4096 --trust-remote-code --mamba-ssm-cache-dtype float32 \
  --mamba-cache-mode align --gpu-memory-utilization 0.75 --enable-prefix-caching \
  --port 8056
# SketchSSM = same command, GPU 7, --port 8057, plus:
#   --sketchssm SketchSSM/Nemotron-3-Nano-4B-BF16 --sketchssm-mean-rank 8
```

Logged non-default args (verbatim from `server_baseline_cold1.log` / `server_sketchssm_serve3.log`):

```
baseline  : {'port': 8056, 'model': 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16', 'trust_remote_code': True,
             'max_model_len': 4096, 'served_model_name': ['nemotron'], 'gpu_memory_utilization': 0.75,
             'enable_prefix_caching': True, 'mamba_ssm_cache_dtype': 'float32', 'mamba_cache_mode': 'align'}
sketchssm : {... identical ..., 'port': 8057,
             'sketchssm': 'SketchSSM/Nemotron-3-Nano-4B-BF16', 'sketchssm_mean_rank': 8.0}
```

## Sanity checks (all satisfied)

| check | evidence |
|---|---|
| SketchSSM path actually ran | `server_sketchssm_serve3.log`: `SketchSSM calibration SketchSSM/Nemotron-3-Nano-4B-BF16: mean rank 8 at window 16, 1926 sketch and 90 dense heads` and `SketchSSM sketch buffers: 4955.4 KiB per request over all 21 calibrated layers; each layer holds them for 1024 requests` |
| V2 model runner | Both logs: `VLLM_USE_V2_MODEL_RUNNER=1 is redundant and can be unset: Model Runner V1 has been removed.` (so V2 is the only runner) |
| align-mode prefix caching | Both logs: `Mamba cache mode is set to 'align' for NemotronHForCausalLM by default when prefix caching is enabled`; `--mamba-cache-mode` itself is deprecated and has no effect in this version |
| both GPUs clean at startup | baseline `Free memory on device (286.47/287.98 GiB)`; sketchssm `Free memory on device (286.47/287.98 GiB)` |

GPU memory, both servers after the final restart (measured with `torch.cuda.mem_get_info`):
74.2 GiB free on GPU 6, 73.5 GiB free on GPU 7.

Residual (honest) memory asymmetry, inherent to SketchSSM rather than a config difference:
weights + non-torch are 8.3 GiB (baseline) vs 13.19 GiB (SketchSSM, calibration buffers), so
the KV pool ends up 2,352,884 tokens (baseline) vs 2,189,125 tokens (SketchSSM), 7% smaller.
Both are far above the 327,680 tokens one 256-prompt run needs, so no preemption or
capacity-induced throttling occurred in either server.

## Benchmark command

```bash
cd /app/local/yh/sketchssm-e2e-wt
PYTHONPATH=/app/local/yh/sketchssm-e2e-wt python3 -m vllm.entrypoints.cli.main bench serve \
  --backend openai --endpoint /v1/completions --base-url http://127.0.0.1:8056 \
  --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --served-model-name nemotron --trust-remote-code \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --ignore-eos --num-prompts 256 --max-concurrency 256 \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 --seed 1234 \
  --save-result --result-dir <this dir> --result-filename serving_bench_<mode>_c256_<tag>.json
```

`--max-concurrency 32` replaces 256 for the second data point; nothing else changes.
Flag names were confirmed against `bench serve --help=all` on this version.
`--request-rate` stays at its `inf` default, so all 256 requests are submitted at t=0 and
the run is a saturation burst. `--ignore-eos` makes every request exactly 256 output
tokens; every run reports `total_input_tokens 262144`, `total_output_tokens 65536`,
`completed 256`, `failed 0`.

## Table 1 - concurrency 256 (cold prefix cache)

| metric | baseline | sketchssm | sketchssm vs baseline (>1 = sketchssm better) |
|---|---|---|---|
| req/s | 36.07 | 54.58 | 1.51x |
| out tok/s | 9234.2 | 13972.0 | 1.51x |
| total tok/s | 46170.9 | 69860.2 | 1.51x |
| TTFT p50 (ms) | 1262.1 | 1261.6 | 1.00x |
| TTFT p99 (ms) | 2300.5 | 2251.2 | 1.02x |
| TPOT p50 (ms) | 22.62 | 13.28 | 1.70x |
| TPOT p99 (ms) | 26.09 | 16.92 | 1.54x |
| ITL p50 (ms) | 19.26 | 9.77 | 1.97x |
| ITL p99 (ms) | 133.80 | 125.31 | 1.07x |

Peak output token throughput (`max_output_tokens_per_s`): 13,312 (baseline) vs
26,357 / 26,240 (SketchSSM) = **1.98x**.

Per-run values and provenance:

| mode | rep | seed | server-side prefix cache hit rate | req/s | out tok/s | mean TTFT | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | duration (s) | json |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 1234 | 0.4% | 36.08 | 9237.0 | 1254.1 | 1225.5 | 2285.9 | 22.75 | 26.17 | 19.25 | 134.42 | 7.09 | `serving_bench_baseline_c256_cold1.json` |
| baseline | 2 | 2024 | 0.0% | 36.06 | 9231.3 | 1266.3 | 1298.8 | 2315.2 | 22.49 | 26.02 | 19.26 | 133.17 | 7.10 | `serving_bench_baseline_c256_cold2.json` |
| sketchssm | 1 | 1234 | 0.0% | 54.50 | 13953.1 | 1261.8 | 1268.3 | 2260.0 | 13.28 | 16.91 | 9.76 | 125.17 | 4.70 | `serving_bench_sketchssm_c256.json` |
| sketchssm | 2 | 2024 | 0.0% | 54.65 | 13991.0 | 1244.7 | 1254.9 | 2242.4 | 13.28 | 16.93 | 9.77 | 125.45 | 4.68 | `serving_bench_sketchssm_c256_cold.json` |

Run-to-run spread is under 0.5% on throughput and under 6% on TTFT p50 for both servers.

## Table 2 - concurrency 32 (cold prefix cache)

| metric | baseline | sketchssm | sketchssm vs baseline (>1 = sketchssm better) |
|---|---|---|---|
| req/s | 22.36 | 25.79 | 1.15x |
| out tok/s | 5723.7 | 6603.4 | 1.15x |
| total tok/s | 28618.5 | 33016.9 | 1.15x |
| TTFT p50 (ms) | 183.3 | 201.5 | 0.91x |
| TTFT p99 (ms) | 305.9 | 295.0 | 1.04x |
| TPOT p50 (ms) | 4.86 | 4.06 | 1.20x |
| TPOT p99 (ms) | 5.42 | 4.65 | 1.17x |
| ITL p50 (ms) | 4.56 | 3.46 | 1.32x |
| ITL p99 (ms) | 5.15 | 5.63 | 0.91x |

Per-run values and provenance:

| mode | rep | seed | server-side prefix cache hit rate | req/s | out tok/s | mean TTFT | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | duration (s) | json |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 1 | 1234 | 0.0-0.4% | 22.33 | 5717.3 | 194.7 | 185.5 | 310.3 | 4.86 | 5.44 | 4.55 | 4.94 | 11.46 | `serving_bench_baseline_c32_cold3.json` |
| baseline | 2 | 2024 | 0.0-0.4% | 22.38 | 5730.1 | 189.2 | 181.1 | 301.6 | 4.87 | 5.40 | 4.56 | 5.36 | 11.44 | `serving_bench_baseline_c32_cold4.json` |
| sketchssm | 1 | 1234 | 0.0% | 25.81 | 6607.2 | 196.1 | 195.5 | 294.8 | 4.09 | 4.65 | 3.46 | 5.58 | 9.92 | `serving_bench_sketchssm_c32.json` |
| sketchssm | 2 | 2024 | 0.0% | 25.78 | 6599.6 | 199.3 | 207.4 | 295.2 | 4.03 | 4.65 | 3.46 | 5.68 | 9.93 | `serving_bench_sketchssm_c32_cold.json` |

At c=32 the server runs 8 waves of 32 requests, so every request pays a 1024-token prefill
(32,768 input tokens per wave) and then 256 decode steps. TPOT improves 1.20x, the prefill
step in front of it is ~10% slower (mean TTFT 192.0 vs 197.7 ms per the per-run means), and
the two effects nearly cancel: the throughput gain is only 1.15x.

## Greedy robustness check (c=256, `--temperature=0`)

`vllm bench serve` in this version no longer forces `temperature=0` (it prints
`WARNING: vllm bench serve no longer sets temperature==0 (greedy) ...` and lets the server
default apply). That warning is in every console log above, and it applies identically to
both servers. To confirm the result does not depend on it, one extra pair was run with
`--temperature=0` (cold cache, seed 1234):

| run | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|---|
| baseline c256 greedy (`serving_bench_baseline_c256_greedy.json`) | 36.76 | 9411.7 | 1232.1 | 2322.1 | 22.21 | 25.64 |
| sketchssm c256 greedy (`serving_bench_sketchssm_c256_greedy.json`) | 56.32 | 14417.2 | 1241.0 | 2234.0 | 12.80 | 16.40 |

Greedy: 1.53x req/s, 1.74x TPOT p50, TTFT unchanged. Same conclusion as the sampled runs.

## Prefix caching: the one measurement trap in this setup

`--enable-prefix-caching` is on in both servers, and the two servers do **not** behave the
same way with it:

- **baseline** serves identical prompts from its prefix cache on a repeat run. Its
  `Prefix cache hit rate` climbed 0.4% -> 19.4% -> 47.8% -> 55.3% -> 63.7% -> 73.8% across
  my repeated seeded runs.
- **SketchSSM logged `Prefix cache hit rate: 0.0%` in every single run**, including runs that
  re-sent byte-identical prompts (`serving_bench_sketchssm_c256_r2.json` is a re-run of
  `serving_bench_sketchssm_c256.json`: 54.61 vs 54.50 req/s, 1248.9 vs 1268.3 ms TTFT - no
  cache benefit at all). The mechanism was not investigated further; it is reported as
  observed. Consequence: SketchSSM numbers are cache-independent, baseline numbers are not.

My first pass at c=256/c=32 compared a warm-cache baseline against a cold SketchSSM and
looked like a 3x TTFT regression at c=32. That was the cache, not prefill. To keep the
comparison honest:

1. every baseline measurement was taken on a **freshly restarted** baseline server
   (`restart_baseline.sh`), verified cold by the server's own `Prefix cache hit rate`
   (0.0-0.4% in every measured run, see the provenance tables);
2. each rep uses a **different `--seed`** so the baseline cannot serve the previous rep's
   prompts out of cache;
3. both servers use the same seed within a rep, so they see identical prompts;
4. the SketchSSM server was left running because it never registers hits anyway (verified by
   re-running it, see above).

Warm-cache runs are kept as evidence and are **not** part of the headline tables:

| mode | concurrency | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | json |
|---|---|---|---|---|---|---|---|---|
| baseline (warm cache) | 256 | 48.06 | 12303.8 | 360.1 | 521.7 | 19.27 | 19.61 | `serving_bench_baseline_c256_r2.json` |
| baseline (warm cache) | 32 | 25.95 | 6643.5 | 67.0 | 103.3 | 4.56 | 4.67 | `serving_bench_baseline_c32_r2.json` |
| sketchssm (identical prompts, no hits) | 256 | 54.61 | 13981.2 | 1248.9 | 2242.6 | 13.31 | 16.94 | `serving_bench_sketchssm_c256_r2.json` |
| sketchssm (identical prompts, no hits) | 32 | 25.90 | 6631.3 | 204.6 | 293.3 | 4.04 | 4.64 | `serving_bench_sketchssm_c32_r2.json` |

This is worth stating in the PR: even when the baseline is handed a 63-74% prefix-cache hit
rate on repeated traffic and SketchSSM gets none, SketchSSM still wins decisively at c=256
(48.06 vs 54.61 req/s, 19.27 vs 13.31 ms TPOT p50). At c=32 warm cache makes it a wash
(25.95 vs 25.90 req/s). Note the baseline at c=256 warm-cache is still far below SketchSSM
cold, so the headline conclusion is robust either way.

## Why 1.51x here but 2.27x offline

Decomposing the c=256 run using the client's own numbers (derived, not independently measured):

- SketchSSM: 4.70 s total, mean TTFT 1.2618 s, so 3.44 s of decode / 256 steps = 13.4 ms/step
  (matches TPOT p50 13.28 ms).
- baseline: 7.09 s total, mean TTFT 1.2541 s, so 5.84 s of decode / 256 steps = 22.8 ms/step
  (matches TPOT p50 22.62 ms).

So the *decode step* speedup in serving is 22.8 / 13.4 = **1.70x**, and the end-to-end 1.51x
is what remains after both servers pay the same ~1.25 s prefill of 262,144 tokens (27% of the
SketchSSM wall clock). At c=32, prefill is a larger fraction of each wave (8 waves, ~190 ms of
prefill per ~1.4 s wave), which is why the end-to-end gain there is only 1.15x while TPOT
still improves 1.20x.

The offline loop measured 17.187 vs 7.570 ms/step (2.27x) at batch 256. Subtracting the offline
step times from the serving step times leaves a roughly constant ~5.6-5.8 ms/step
(22.8 - 17.19 = 5.61; 13.4 - 7.57 = 5.83), i.e. a serving-side per-token overhead (sampling,
detokenization, streaming, scheduler) that is added to both configs and therefore compresses
the ratio from 2.27x to 1.70x. The offline run used 1024-in/500-out and 500 greedy steps, so
this decomposition is indicative rather than exact.

## Caveats / not apples-to-apples

- **Prefix caching is on** in both servers and behaves differently between them (above). All
  headline numbers are cold-cache. A repeated-identical-traffic benchmark would favour the
  baseline in a way this measurement deliberately avoids.
- **Synthetic workload**: `--dataset-name random` generates uniform random token ids, 1024 in /
  256 out, `--ignore-eos`, `--request-rate inf` (saturation burst). No real trace, no
  prompt-length variation (`--random-range-ratio` left at 0, so every prompt is exactly 1024
  tokens), no shared prefixes between prompts.
- **Sampling**: `--temperature` was not passed (as specified), so the server-side default
  applied to both; see the greedy cross-check above.
- **No accuracy/idempotency check in this run.** Throughput and latency only. `compare_servers.py`
  in this directory is the existing greedy-output/logprob comparison harness
  (`compare_mt64.json`); it was not re-run here.
- **Sequential, single-tenant**: the two servers always live on separate GPUs (6 and 7) and were
  benchmarked one at a time, never concurrently. The other server stayed alive and idle, and
  each idle `VLLM::EngineCore` busy-polls ~100% of one CPU core; the machine has 236 CPUs and
  load average stayed ~4.8, so the 256-connection client was not CPU starved.
- **KV pool is 7% smaller on SketchSSM** (2,189,125 vs 2,352,884 tokens) because its weights +
  calibration buffers are 4.9 GiB larger under the same `--gpu-memory-utilization 0.75`. It made
  no difference at 327,680 tokens per run, but it would matter for a capacity-constrained test.
- **ITL p99 is ~125-134 ms at c=256 on both servers** (vs p50 9.8-19.3 ms) - a straggler/chunk
  boundary effect present in both; not analysed.
- **c=32 TTFT is 10% worse for SketchSSM** (183.3 -> 201.5 ms p50). TTFT p99 is slightly better
  (305.9 -> 295.0 ms). This is the one metric where SketchSSM does not win; it comes from the
  prefill path also rotating state and building the sketch. Only c=32 and c=256 were measured,
  so the crossover concurrency was not located.
- **One earlier attempt was invalid and was redone.** The servers the coordinator started had an
  orphaned `VLLM::EngineCore` (pid 136598, started 14:21:35, parent=1) left over from the earlier
  14:22 run, holding 204 GB on GPU 7. SketchSSM therefore started with only 79.28 GiB free and a
  54.5 GiB KV pool (629,294 tokens) while the baseline had 286.47 GiB free and 196.8 GiB KV
  (2,352,884 tokens). I stopped the orphan (`rocm-smi`/`amd-smi` showed it as an unattributed
  KFD process; `amd-smi process` mapped it to GPU 7), reloaded both GPUs to full free memory and
  restarted both servers (`start_servers.sh align serve3`). Every number reported above is from
  the clean restarts (`server_baseline_cold1.log` through `server_baseline_cold5.log`,
  `server_sketchssm_serve3.log`).
  Worth noting for future runs: killing the API server alone leaves its EngineCore alive.

## Artifacts in this directory

**`MANIFEST.md` is the authoritative index**: it lists every `serving_bench_*.json` with its
seed, server-side prefix-cache state, server log and role. Only rows marked `HEADLINE` feed the
two tables above. Watch out for the suffix-free names: `serving_bench_baseline_c32.json` is the
*discarded warm-cache* run (25.67 req/s, TTFT 73 ms) and must not be quoted - the c=32 baseline
cold-cache number is 22.36 req/s (suffix `cold3`/`cold4`).

| file | contents |
|---|---|
| `MANIFEST.md` | provenance for all 20 benchmark runs (seed, cache state, role) |
| `serving_bench_{baseline,sketchssm}_c{256,32}_cold*.json` | headline cold-cache results (`--save-result`) |
| `serving_bench_sketchssm_c{256,32}.json` | SketchSSM rep 1 (seed 1234) |
| `serving_bench_*_c*_greedy.json` | `--temperature=0` cross-check |
| `serving_bench_*_r2.json` | warm-cache runs, kept as evidence only |
| `serving_bench_*.console.log` | verbatim client output for every run |
| `server_baseline_cold1..cold5.log`, `server_sketchssm_serve3.log` | server logs for the measured runs |
| `run_serving_bench.sh` | one measurement + server-side cache-hit-rate capture |
| `restart_baseline.sh` | cold-cache baseline restart (also kills its EngineCore child) |
| `summarize_serving.py` | regenerates the tables above from the JSONs (`python3 summarize_serving.py`) |
| `manifest_serving.py` | regenerates `MANIFEST.md` (`python3 manifest_serving.py`) |
| `start_servers.sh` | launches both servers (unchanged) |
| `e2e_speedup_b256.log` | the offline `LLM.generate` comparison, for reference |

Reproduce:

```bash
cd /app/local/yh/issue-workdirs/pr61124-sketchssm-rocm/e2e
./start_servers.sh align serve3          # both servers, GPU 6 + 7
./restart_baseline.sh coldX              # cold baseline before each baseline run
./run_serving_bench.sh baseline 256 1234 rep1 coldX
./run_serving_bench.sh sketchssm 256 1234 rep1     # no restart needed, 0.0% hits always
python3 summarize_serving.py
python3 manifest_serving.py
```

## State at the end of this task

Both servers were **left running**: baseline pid 155436 on GPU 6 (`:8056`, log
`server_baseline_cold5.log`), SketchSSM pid 145582 on GPU 7 (`:8057`, log
`server_sketchssm_serve3.log`), both started from a clean GPU with 286.47/287.98 GiB free.
The orphaned EngineCore from the earlier run was killed and did not come back.
