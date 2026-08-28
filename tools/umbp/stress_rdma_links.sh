#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

DURATION=${DURATION:-60}
MESSAGE_SIZE=${MESSAGE_SIZE:-1048576}
MODE=${MODE:-both}
READ_PORT_BASE=${READ_PORT_BASE:-18515}
WRITE_PORT_BASE=${WRITE_PORT_BASE:-18615}
LOG_DIR=${LOG_DIR:-rdma-stress-logs}
GID_INDEXES_CSV=${GID_INDEXES:-1,1,1,1,1,1,1,1}

RDMA_DEVICES=(ionic_0 ionic_1 ionic_2 ionic_3 ionic_4 ionic_5 ionic_6 ionic_7)
REMOTE_IPS=(
  fc01:800:5803:8a77:690:81ff:fe4c:5fd9
  fc01:700:5703:8a77:690:81ff:fe4c:61d1
  fc01:500:5503:8a77:690:81ff:fe4c:7d49
  fc01:600:5603:8a77:690:81ff:fe4c:dac1
  fc01:400:5403:8a77:690:81ff:fe4c:7f71
  fc01:300:5303:8a77:690:81ff:fe4c:6db9
  fc01:100:5103:8a77:690:81ff:fe4c:dd31
  fc01:200:5203:8a77:690:81ff:fe49:c7e9
)

IFS=, read -r -a GID_INDEXES <<<"$GID_INDEXES_CSV"
PIDS=()
LABELS=()

fail() {
  echo "error: $*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
Usage:
  stress_rdma_links.sh server
  stress_rdma_links.sh client

Run the server on 192.168.0.185 before running the client on 192.168.0.69.
All eight Ionic links run concurrently. Environment variables:

  MODE=read|write|both       Test direction; default: both
  DURATION=<seconds>         Duration of each test; default: 60
  MESSAGE_SIZE=<bytes>       Transfer size; default: 1048576
  GID_INDEXES=i0,...,i7      Per-device global RoCE v2 indexes; default: all 1
  LOG_DIR=<directory>        Result directory; default: rdma-stress-logs
  READ_PORT_BASE=<port>      ionic_0 read port; default: 18515
  WRITE_PORT_BASE=<port>     ionic_0 write port; default: 18615

MODE=both creates one read and one write process per link at the same time. It
is an aggressive bidirectional stress test, not an isolated peak-bandwidth
measurement.
EOF
}

cleanup() {
  local pid
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
}
trap cleanup INT TERM EXIT

validate_configuration() {
  [[ "$MODE" == read || "$MODE" == write || "$MODE" == both ]] ||
    fail "MODE must be read, write, or both"
  [[ "$DURATION" =~ ^[1-9][0-9]*$ ]] || fail "DURATION must be positive"
  [[ "$MESSAGE_SIZE" =~ ^[1-9][0-9]*$ ]] || fail "MESSAGE_SIZE must be positive"
  [[ ${#GID_INDEXES[@]} -eq 8 ]] || fail "GID_INDEXES must contain 8 indexes"
  if [[ "$MODE" == read || "$MODE" == both ]]; then
    command -v ib_read_bw >/dev/null 2>&1 || fail "ib_read_bw is unavailable"
  fi
  if [[ "$MODE" == write || "$MODE" == both ]]; then
    command -v ib_write_bw >/dev/null 2>&1 || fail "ib_write_bw is unavailable"
  fi
  mkdir -p "$LOG_DIR"
}

launch_test() {
  local role=$1
  local operation=$2
  local index=$3
  local device=${RDMA_DEVICES[$index]}
  local gid_index=${GID_INDEXES[$index]}
  local command_name port log_file

  if [[ "$operation" == read ]]; then
    command_name=ib_read_bw
    port=$((READ_PORT_BASE + index))
  else
    command_name=ib_write_bw
    port=$((WRITE_PORT_BASE + index))
  fi
  log_file="$LOG_DIR/${role}-${operation}-${device}.log"

  local args=(
    --ipv6
    --ipv6-addr
    -d "$device"
    -x "$gid_index"
    -p "$port"
    -D "$DURATION"
    -s "$MESSAGE_SIZE"
    --perform_warm_up
    --report_gbits
  )
  if [[ "$role" == client ]]; then
    args+=("${REMOTE_IPS[$index]}")
  fi

  echo "Starting $role $operation on $device, GID $gid_index, port $port"
  "$command_name" "${args[@]}" >"$log_file" 2>&1 &
  PIDS+=("$!")
  LABELS+=("$operation/$device")
}

launch_all() {
  local role=$1
  local index
  for index in "${!RDMA_DEVICES[@]}"; do
    if [[ "$MODE" == read || "$MODE" == both ]]; then
      launch_test "$role" read "$index"
    fi
    if [[ "$MODE" == write || "$MODE" == both ]]; then
      launch_test "$role" write "$index"
    fi
  done
}

wait_all() {
  local failures=0
  local position
  for position in "${!PIDS[@]}"; do
    if wait "${PIDS[$position]}"; then
      echo "PASS: ${LABELS[$position]}"
    else
      echo "FAIL: ${LABELS[$position]} (see $LOG_DIR)"
      failures=$((failures + 1))
    fi
  done
  PIDS=()
  ((failures == 0)) || fail "$failures RDMA stress processes failed"
}

print_client_results() {
  echo
  echo "Per-link results:"
  local log_file
  for log_file in "$LOG_DIR"/client-*.log; do
    [[ -e "$log_file" ]] || continue
    echo "$(basename "$log_file"):"
    awk '/#bytes.*#iterations/{getline; print}' "$log_file"
  done
  echo
  echo "Full logs: $LOG_DIR"
}

case ${1:-} in
  server | client)
    [[ $# -eq 1 ]] || fail "$1 mode takes no additional arguments"
    validate_configuration
    launch_all "$1"
    wait_all
    [[ "$1" == client ]] && print_client_results
    trap - INT TERM EXIT
    ;;
  *)
    usage
    exit 2
    ;;
esac
