#!/bin/bash
# One serving measurement against an already-running server.
# Usage: run_serving_bench.sh <baseline|sketchssm> <concurrency> <seed> <tag> [server-log-suffix]
#
# Records:
#   serving_bench_<mode>_c<conc>_<tag>.json           (--save-result)
#   serving_bench_<mode>_c<conc>_<tag>.console.log    (verbatim client output)
# and prints the server-side "Prefix cache hit rate" lines produced by this run,
# so every number can be tied to the cache state it was measured in.
set -u
WT=/app/local/yh/sketchssm-e2e-wt
EW=/app/local/yh/issue-workdirs/pr61124-sketchssm-rocm/e2e
MODE=$1; CONC=$2; SEED=$3; TAG=$4; LOGSUFFIX=${5:-serve3}
if [ "$MODE" = "baseline" ]; then PORT=8056; else PORT=8057; fi
SRVLOG=$EW/server_${MODE}_${LOGSUFFIX}.log

BEFORE=$(wc -l < "$SRVLOG")

cd "$WT" || exit 1
PYTHONPATH=$WT timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
  --backend openai --endpoint /v1/completions --base-url http://127.0.0.1:$PORT \
  --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --served-model-name nemotron --trust-remote-code \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --ignore-eos --num-prompts 256 --max-concurrency "$CONC" \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 --seed "$SEED" \
  --save-result --result-dir "$EW" --result-filename "serving_bench_${MODE}_c${CONC}_${TAG}.json" \
  2>&1 | tee "$EW/serving_bench_${MODE}_c${CONC}_${TAG}.console.log" \
  | grep -E "Successful|Failed|Benchmark duration|Request throughput|Output token through|Peak output|Total token through|P50 TTFT|P99 TTFT|P50 TPOT|P99 TPOT|P50 ITL|P99 ITL"

echo "--- server-side cache state during this run ($MODE c=$CONC seed=$SEED) ---"
tail -n +$((BEFORE + 1)) "$SRVLOG" | grep -oE "Prefix cache hit rate: [0-9.]+%"
