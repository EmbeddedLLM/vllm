#!/bin/bash
# wait_ready.sh <logfile> [timeout_s] - poll until the vLLM server is serving.
LOG=$1
TMO=${2:-420}
for _ in $(seq 1 $((TMO/5))); do
  if grep -q "Application startup complete" "$LOG"; then echo READY; exit 0; fi
  if grep -qE "Engine core initialization failed|Traceback \(most recent" "$LOG"; then echo FAILED; exit 1; fi
  sleep 5
done
echo TIMEOUT; exit 1
