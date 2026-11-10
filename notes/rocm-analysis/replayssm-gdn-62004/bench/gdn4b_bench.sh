#!/bin/bash
# Sweep one server config at bs 64/256/512, two reps each, fresh seed per run.
# usage: gdn4b_bench.sh <tag> <port>
set -uo pipefail
TAG=$1
PORT=$2
DIR=/app/local/yh/issue-workdirs/pr62004-replayssm-gdn-rocm
WT=/app/local/yh/replayssm-gdn-wt
MODEL=/models/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
cd "$WT"

bench() { # $1=conc $2=seed $3=filename
  env PYTHONPATH="$WT" python3 -m vllm.entrypoints.cli.main bench serve \
    --backend openai \
    --base-url "http://127.0.0.1:$PORT" \
    --endpoint /v1/completions \
    --model "$MODEL" \
    --served-model-name qwen35 \
    --tokenizer "$MODEL" \
    --dataset-name random \
    --random-input-len 512 \
    --random-output-len 128 \
    --ignore-eos \
    --num-prompts 2048 \
    --max-concurrency "$1" \
    --seed "$2" \
    --percentile-metrics ttft,tpot,itl,e2el \
    --metric-percentiles 50,99 \
    --save-result \
    --result-dir "$DIR/bench" \
    --result-filename "$3"
}

# warmup (discarded): clocks/page-cache, not part of the reported cells
bench 64 1 "gdn4b_warmup_${TAG}.json" > "$DIR/logs/gdn4b_bench_warmup_${TAG}.log" 2>&1

for REP in 1 2; do
  for C in 64 256 512; do
    SEED=$(( (RANDOM * 32768 + RANDOM) % 100000 + REP * 1000 + C ))
    echo "=== $TAG bs=$C rep=$REP seed=$SEED"
    bench "$C" "$SEED" "gdn4b_${TAG}_bs${C}_r${REP}.json" \
      > "$DIR/logs/gdn4b_bench_${TAG}_bs${C}_r${REP}.log" 2>&1
    tail -n 12 "$DIR/logs/gdn4b_bench_${TAG}_bs${C}_r${REP}.log" | grep -E "Output token|Request throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|P99 TPOT|Successful|Failed|Benchmark duration"
  done
done
