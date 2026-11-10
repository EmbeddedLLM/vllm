#!/bin/bash
# Confounder check for warmcache_ab.py: is the 0% on same-seed reruns a block-size
# artifact (prompt < block_size) rather than "SketchSSM does not cache"?
# Runs the seed-N random bench twice in a row at a chosen --random-input-len.
# Usage: warmcache_probe_inlen.sh <port> <mode> <seed> <random-input-len>
set -u
EW=/app/local/yh/issue-workdirs/pr61124-sketchssm-rocm/e2e
WT=/app/local/yh/sketchssm-e2e-wt
PORT=$1; MODE=$2; SEED=$3; INLEN=$4
m(){ curl -s "http://127.0.0.1:$PORT/metrics" | awk '/^vllm:prefix_cache_queries_total/{q=$2}/^vllm:prefix_cache_hits_total/{h=$2}END{printf "%d %d", q, h}'; }
bench(){
  TAG=$1
  read -r q0 h0 <<<"$(m)"
  ( cd "$WT" && PYTHONPATH=$WT python3 -m vllm.entrypoints.cli.main bench serve \
      --backend openai --endpoint /v1/completions \
      --base-url "http://127.0.0.1:$PORT" --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 \
      --served-model-name nemotron --trust-remote-code \
      --dataset-name random --random-input-len "$INLEN" --random-output-len 64 \
      --num-prompts 32 --max-concurrency 8 --ignore-eos --seed "$SEED" \
      --save-result --result-dir "$EW" --result-filename "serving_bench_${MODE}_${TAG}.json" ) \
    > "$EW/serving_bench_${MODE}_${TAG}.console.log" 2>&1
  read -r q1 h1 <<<"$(m)"
  printf '  %-22s seed=%s inlen=%s hits=%d queries=%d\n' "$TAG" "$SEED" "$INLEN" "$((h1-h0))" "$((q1-q0))"
}
bench "inlen${INLEN}_s${SEED}_r1"
bench "inlen${INLEN}_s${SEED}_r2"
