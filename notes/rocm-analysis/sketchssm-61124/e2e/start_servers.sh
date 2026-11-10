#!/bin/bash
# Start the baseline (GPU 6, port 8056) and SketchSSM (GPU 7, port 8057) servers.
# Usage: start_servers.sh <cache-mode: none|align> <log-suffix>
set -u
WT=/app/local/yh/sketchssm-e2e-wt
EW=/app/local/yh/issue-workdirs/pr61124-sketchssm-rocm/e2e
MODE=${1:-align}
SUFFIX=${2:-$MODE}
MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
PREFIX_CACHE=""
if [ "$MODE" = "align" ]; then PREFIX_CACHE="--enable-prefix-caching"; fi

COMMON=(--model "$MODEL" --served-model-name nemotron --max-model-len 4096
        --trust-remote-code --mamba-ssm-cache-dtype float32
        --mamba-cache-mode "$MODE" --gpu-memory-utilization 0.75 $PREFIX_CACHE)

cd "$WT" || exit 1
VLLM_USE_V2_MODEL_RUNNER=1 PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=6 \
  nohup python3 -m vllm.entrypoints.openai.api_server \
  "${COMMON[@]}" --port 8056 > "$EW/server_baseline_${SUFFIX}.log" 2>&1 &
echo "baseline pid $!"
VLLM_USE_V2_MODEL_RUNNER=1 PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=7 \
  nohup python3 -m vllm.entrypoints.openai.api_server \
  "${COMMON[@]}" --port 8057 \
  --sketchssm SketchSSM/Nemotron-3-Nano-4B-BF16 --sketchssm-mean-rank 8 \
  > "$EW/server_sketchssm_${SUFFIX}.log" 2>&1 &
echo "sketchssm pid $!"
