#!/bin/bash
# Launch a Qwen3.5-4B server for the GDN AITER-path 2x2 on an exclusive GPU.
# usage: [DEV=6] [AITER=0|1] [EAGER=1] [SPIKE_TRACE=1] [SPIKE_GDN_NO_AITER=1] \
#          [SPIKE_GATE_GDN=0|1] gdn4b_2x2_server.sh <base|rs> <port> <logfile>
# Same flags as gdn4b_server.sh; DEV chooses the GPU, AITER=1 exports
# VLLM_ROCM_USE_AITER=1 (enables the GDN AITER Triton decode path).
# SPIKE_GDN_NO_AITER=1 forces GDN off the AITER path unconditionally.
# SPIKE_GATE_GDN=0 keeps GDN on the AITER path even with --use-replayssm
# (hybrid: AITER fused conv + ReplaySSM recurrence).
set -euo pipefail
CFG=$1
PORT=$2
LOG=$(realpath -m "$3")
DEV=${DEV:-6}
AITER=${AITER:-1}
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
if [ "${EAGER:-0}" = "1" ]; then
  ARGS+=(--enforce-eager)
fi

cd "$WT"
exec env PYTHONPATH="$WT" CUDA_VISIBLE_DEVICES="$DEV" VLLM_ROCM_USE_AITER="$AITER" \
  SPIKE_GDN_NO_AITER="${SPIKE_GDN_NO_AITER:-0}" \
  SPIKE_GATE_GDN="${SPIKE_GATE_GDN:-1}" \
  python3 -m vllm.entrypoints.cli.main serve "${ARGS[@]}" > "$LOG" 2>&1
