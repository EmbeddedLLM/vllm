#!/bin/bash
# Restart ONLY the baseline server (GPU 6, port 8056) with the exact args
# start_servers.sh uses for align mode, so its prefix cache starts empty.
# The SketchSSM server is left untouched: it logs "Prefix cache hit rate: 0.0%"
# on every run, so its measurements do not depend on cache state.
# Usage: restart_baseline.sh <log-suffix>
set -u
WT=/app/local/yh/sketchssm-e2e-wt
EW=/app/local/yh/issue-workdirs/pr61124-sketchssm-rocm/e2e
SUFFIX=${1:-serve4}
MODEL=nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16

# Stop the server and its EngineCore child (killing only the API server leaves
# an orphaned EngineCore holding the GPU - that is what happened earlier).
# Note: `ss -ltnp` does not list these listeners in this container, so the pid is
# found from the cmdline instead.
API=$(pgrep -f -- "--port 8056" | head -1)
if [ -n "${API:-}" ]; then
  EC=$(pgrep -P "$API" -f "VLLM::EngineCore" || true)
  kill -TERM "$API" ${EC:-} 2>/dev/null
  for _ in $(seq 1 30); do
    kill -0 "$API" 2>/dev/null || break
    sleep 2
  done
  kill -0 "$API" 2>/dev/null && kill -KILL "$API" ${EC:-} 2>/dev/null
  sleep 3
fi
pgrep -f "VLLM::EngineCore" | while read -r p; do
  grep -qa "CUDA_VISIBLE_DEVICES=6" /proc/"$p"/environ 2>/dev/null && kill -TERM "$p"
done
sleep 5
if pgrep -f -- "--port 8056" > /dev/null; then
  echo "ERROR: old baseline still running"; exit 1
fi

cd "$WT" || exit 1
VLLM_USE_V2_MODEL_RUNNER=1 PYTHONPATH=$WT CUDA_VISIBLE_DEVICES=6 \
  nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name nemotron --max-model-len 4096 \
  --trust-remote-code --mamba-ssm-cache-dtype float32 \
  --mamba-cache-mode align --gpu-memory-utilization 0.75 --enable-prefix-caching \
  --port 8056 > "$EW/server_baseline_${SUFFIX}.log" 2>&1 &
echo "baseline pid $! -> server_baseline_${SUFFIX}.log"

for _ in $(seq 1 60); do
  grep -q "Application startup complete" "$EW/server_baseline_${SUFFIX}.log" && { echo UP; exit 0; }
  sleep 3
done
echo "TIMEOUT waiting for baseline"; exit 1
