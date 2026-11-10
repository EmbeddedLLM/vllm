# Manifest of serving benchmark runs

Provenance for every `serving_bench_*.json` in this directory. The seed and the
server-side prefix-cache state are not stored in the client JSON; they are
transcribed here from the commands and the server logs. `SERVING_ROCM.md` uses only
the rows marked HEADLINE.

| date | json | c | seed | prefix cache state | server log | role | req/s | TTFT p50 | TPOT p50 |
|---|---|---|---|---|---|---|---|---|---|
| 20261109-000846 | `serving_bench_baseline_c256.json` | 256 | 1234 | cold, 0.4% | `server_baseline_serve3.log` | first attempt, superseded by cold1 | 36.08 | 1263.3 | 22.60 |
| 20261109-000903 | `serving_bench_sketchssm_c256.json` | 256 | 1234 | 0.0% | `server_sketchssm_serve3.log` | HEADLINE c=256 sketchssm rep 1 | 54.50 | 1268.3 | 13.28 |
| 20261109-000926 | `serving_bench_baseline_c32.json` | 32 | 1234 | warm, 19.4% | `server_baseline_serve3.log` | DO NOT USE: warm cache bias | 25.67 | 73.1 | 4.57 |
| 20261109-000947 | `serving_bench_sketchssm_c32.json` | 32 | 1234 | 0.0% | `server_sketchssm_serve3.log` | HEADLINE c=32 sketchssm rep 1 | 25.81 | 195.5 | 4.09 |
| 20261109-001015 | `serving_bench_baseline_c32_r2.json` | 32 | 1234 | warm, 55.3% | `server_baseline_serve3.log` | DO NOT USE: warm cache bias, kept as evidence | 25.95 | 67.0 | 4.56 |
| 20261109-001033 | `serving_bench_sketchssm_c32_r2.json` | 32 | 1234 | 0.0% | `server_sketchssm_serve3.log` | repeat of identical prompts: shows sketchssm gets no cache benefit | 25.90 | 204.6 | 4.04 |
| 20261109-001117 | `serving_bench_baseline_c256_r2.json` | 256 | 1234 | warm, 63.7% | `server_baseline_serve3.log` | DO NOT USE: warm cache bias, kept as evidence | 48.06 | 360.1 | 19.27 |
| 20261109-001130 | `serving_bench_sketchssm_c256_r2.json` | 256 | 1234 | 0.0% | `server_sketchssm_serve3.log` | repeat of identical prompts: shows sketchssm gets no cache benefit | 54.61 | 1248.9 | 13.31 |
| 20261109-001228 | `serving_bench_baseline_c256_coldA.json` | 256 | 777 | near-cold | `server_baseline_serve3.log` | corroborating, superseded | 36.40 | 1279.9 | 22.31 |
| 20261109-001245 | `serving_bench_sketchssm_c256_coldA.json` | 256 | 777 | 0.0% | `server_sketchssm_serve3.log` | corroborating | 54.77 | 1257.7 | 13.23 |
| 20261109-001310 | `serving_bench_baseline_c32_coldA.json` | 32 | 4321 | decaying from warm | `server_baseline_serve3.log` | DO NOT USE: cache state unclear, superseded by cold3 | 22.44 | 174.4 | 4.86 |
| 20261109-001328 | `serving_bench_sketchssm_c32_coldA.json` | 32 | 4321 | 0.0% | `server_sketchssm_serve3.log` | corroborating | 25.82 | 208.7 | 4.03 |
| 20261109-001912 | `serving_bench_baseline_c256_cold1.json` | 256 | 1234 | cold, 0.4% | `server_baseline_cold1.log` | HEADLINE c=256 baseline rep 1 | 36.08 | 1225.5 | 22.75 |
| 20261109-002014 | `serving_bench_baseline_c256_cold2.json` | 256 | 2024 | cold, 0.0% | `server_baseline_cold2.log` | HEADLINE c=256 baseline rep 2 | 36.06 | 1298.8 | 22.49 |
| 20261109-002121 | `serving_bench_baseline_c32_cold3.json` | 32 | 1234 | cold, 0.0-0.4% | `server_baseline_cold3.log` | HEADLINE c=32 baseline rep 1 | 22.33 | 185.5 | 4.86 |
| 20261109-002229 | `serving_bench_baseline_c32_cold4.json` | 32 | 2024 | cold, 0.0-0.4% | `server_baseline_cold4.log` | HEADLINE c=32 baseline rep 2 | 22.38 | 181.1 | 4.87 |
| 20261109-002245 | `serving_bench_sketchssm_c256_cold.json` | 256 | 2024 | 0.0% | `server_sketchssm_serve3.log` | HEADLINE c=256 sketchssm rep 2 | 54.65 | 1254.9 | 13.28 |
| 20261109-002304 | `serving_bench_sketchssm_c32_cold.json` | 32 | 2024 | 0.0% | `server_sketchssm_serve3.log` | HEADLINE c=32 sketchssm rep 2 | 25.78 | 207.4 | 4.03 |
| 20261109-002513 | `serving_bench_baseline_c256_greedy.json` | 256 | 1234 | cold | `server_baseline_cold5.log` | cross-check: --temperature=0 | 36.76 | 1232.1 | 22.21 |
| 20261109-002525 | `serving_bench_sketchssm_c256_greedy.json` | 256 | 1234 | 0.0% | `server_sketchssm_serve3.log` | cross-check: --temperature=0 | 56.32 | 1241.0 | 12.80 |

## Low-concurrency runs (c=1/2/4/8), 2026-11-09 10:00-10:12

These 16 runs plus one warm-cache control are documented separately in
`SERVING_LOWCONC_ROCM.md`; they are not used by `SERVING_ROCM.md`. Every run used 64
prompts (`--random-input-len 1024 --random-output-len 256 --ignore-eos`) and its own
fresh `--seed`; the run-scoped prefix-cache state comes from the
`vllm:prefix_cache_hits_total`/`queries_total` delta around the run, stored in the
matching `serving_bench_<mode>_c<N>_<rep>.cache.json`:

| mode | concurrency | rep | seed | run prefix cache hit rate | json |
|---|---|---|---|---|---|
| baseline | 1 | rep1 | 41001 | 0.000% | `serving_bench_baseline_c1_rep1.json` |
| sketchssm | 1 | rep1 | 41002 | 0.000% | `serving_bench_sketchssm_c1_rep1.json` |
| baseline | 2 | rep1 | 41003 | 0.000% | `serving_bench_baseline_c2_rep1.json` |
| sketchssm | 2 | rep1 | 41004 | 0.000% | `serving_bench_sketchssm_c2_rep1.json` |
| baseline | 4 | rep1 | 41005 | 0.000% | `serving_bench_baseline_c4_rep1.json` |
| sketchssm | 4 | rep1 | 41006 | 0.000% | `serving_bench_sketchssm_c4_rep1.json` |
| baseline | 8 | rep1 | 41007 | 1.489% (one duplicate prompt) | `serving_bench_baseline_c8_rep1.json` |
| sketchssm | 8 | rep1 | 41008 | 0.000% | `serving_bench_sketchssm_c8_rep1.json` |
| baseline | 1 | rep2 | 42001 | 0.000% | `serving_bench_baseline_c1_rep2.json` |
| sketchssm | 1 | rep2 | 42002 | 0.000% | `serving_bench_sketchssm_c1_rep2.json` |
| baseline | 2 | rep2 | 42003 | 1.489% (one duplicate prompt) | `serving_bench_baseline_c2_rep2.json` |
| sketchssm | 2 | rep2 | 42004 | 0.000% | `serving_bench_sketchssm_c2_rep2.json` |
| baseline | 4 | rep2 | 42005 | 0.000% | `serving_bench_baseline_c4_rep2.json` |
| sketchssm | 4 | rep2 | 42006 | 0.000% | `serving_bench_sketchssm_c4_rep2.json` |
| baseline | 8 | rep2 | 42007 | 1.489% (one duplicate prompt) | `serving_bench_baseline_c8_rep2.json` |
| sketchssm | 8 | rep2 | 42008 | 0.000% | `serving_bench_sketchssm_c8_rep2.json` |
| baseline | 8 | warmprobe | 41007 | 95.312% (same prompts re-sent) | `serving_bench_baseline_c8_warmprobe.json` |

Only the warmprobe row is deliberately warm; it is a control, not a measurement.
