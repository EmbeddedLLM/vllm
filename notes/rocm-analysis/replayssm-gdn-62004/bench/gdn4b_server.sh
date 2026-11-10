#!/bin/bash
# Launch a Qwen3.5-4B server on GPU 4 for the GDN ReplaySSM e2e sweep.
# usage: gdn4b_server.sh <base|rs|rstuned> <port> <logfile>
set -euo pipefail
CFG=$1
PORT=$2
LOG=$(realpath -m "$3")
WT=/app/local/yh/replayssm-gdn-wt
MODEL=/models/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a

ARGS=(
  "$MODEL"
  --trust-remote-code
  --served-model-name qwen35
  --max-model-len 4096
  --mamba-ssm-cache-dtype float32
  --mamba-cache-mode none
  --gpu-memory-utilization 0.80
  --no-enable-prefix-caching
  --port "$PORT"
)
if [ "$CFG" != "base" ]; then
  ARGS+=(--use-replayssm --replayssm-buffer-len 16)
fi

cd "$WT"
exec env PYTHONPATH="$WT" CUDA_VISIBLE_DEVICES=4 \
  python3 -m vllm.entrypoints.cli.main serve "${ARGS[@]}" > "$LOG" 2>&1
