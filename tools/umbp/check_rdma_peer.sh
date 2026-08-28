#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

REMOTE_MANAGEMENT_IP=${REMOTE_MANAGEMENT_IP:-192.168.0.185}
PING_COUNT=${PING_COUNT:-3}

RDMA_DEVICES=(ionic_0 ionic_1 ionic_2 ionic_3 ionic_4 ionic_5 ionic_6 ionic_7)
INTERFACES=(enP2p0s9 enP2p0s10 enP2p0s11 enP2p0s12 enP3p0s9 enP3p0s10 enP3p0s11 enP3p0s12)
LOCAL_IPS=(
  fc01:800:8803:8a50:690:81ff:fe47:a6a9
  fc01:700:8703:8a50:690:81ff:fe45:baf1
  fc01:500:8503:8a50:690:81ff:fe47:7e29
  fc01:600:8603:8a50:690:81ff:fe47:b0f9
  fc01:400:8403:8a50:690:81ff:fe47:78a1
  fc01:300:8303:8a50:690:81ff:fe47:c5b1
  fc01:100:8103:8a50:690:81ff:fe47:9e69
  fc01:200:8203:8a50:690:81ff:fe46:d9a1
)
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

fail() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

usage() {
  cat <<'EOF'
Usage:
  check_rdma_peer.sh ip
  check_rdma_peer.sh rdma-server <link-index> <gid-index>
  check_rdma_peer.sh rdma-client <link-index> <gid-index>

link-index is 0 through 7 and maps to ionic_0 through ionic_7.

Run `ip` on the local host to test management and routed RoCE addresses. For
an actual RDMA read test, run `rdma-server` on 192.168.0.185 first and then run
`rdma-client` with the same link and GID indexes on 192.168.0.69.
EOF
}

validate_index() {
  local index=$1
  [[ "$index" =~ ^[0-7]$ ]] || fail "link-index must be an integer from 0 to 7"
}

check_ip() {
  require_command ip
  require_command ping

  echo "Checking management address $REMOTE_MANAGEMENT_IP"
  ping -4 -c "$PING_COUNT" -W 2 "$REMOTE_MANAGEMENT_IP"

  local failures=0
  for index in "${!RDMA_DEVICES[@]}"; do
    local device=${RDMA_DEVICES[$index]}
    local interface=${INTERFACES[$index]}
    local local_ip=${LOCAL_IPS[$index]}
    local remote_ip=${REMOTE_IPS[$index]}

    echo
    echo "[$index] $device / $interface"
    if ! ip -6 addr show dev "$interface" | grep -Fq "$local_ip/64"; then
      echo "FAIL: expected local address $local_ip/64 is absent"
      failures=$((failures + 1))
      continue
    fi

    local route
    if ! route=$(ip -6 route get "$remote_ip" 2>&1); then
      echo "FAIL: no route to $remote_ip: $route"
      failures=$((failures + 1))
      continue
    fi
    echo "Route: $route"
    if [[ "$route" != *"dev $interface"* ]]; then
      echo "FAIL: route does not use $interface"
      failures=$((failures + 1))
      continue
    fi

    if ping -6 -c "$PING_COUNT" -W 2 -I "$interface" "$remote_ip"; then
      echo "PASS: IPv6 connectivity over $interface"
    else
      echo "FAIL: IPv6 connectivity over $interface"
      failures=$((failures + 1))
    fi
  done

  ((failures == 0)) || fail "$failures RDMA-facing IP connectivity checks failed"
  echo
  echo "All IP checks passed. Run the rdma-server/client modes next."
}

run_rdma_server() {
  local index=$1
  local gid_index=$2
  validate_index "$index"
  require_command ib_read_bw

  local device=${RDMA_DEVICES[$index]}
  echo "Listening for an RDMA read test on $device with GID index $gid_index"
  exec ib_read_bw --ipv6 --ipv6-addr -d "$device" -x "$gid_index" --report_gbits
}

run_rdma_client() {
  local index=$1
  local gid_index=$2
  validate_index "$index"
  require_command ib_read_bw

  local device=${RDMA_DEVICES[$index]}
  local remote_ip=${REMOTE_IPS[$index]}
  echo "Testing RDMA reads to $remote_ip on $device with GID index $gid_index"
  exec ib_read_bw --ipv6 --ipv6-addr -d "$device" -x "$gid_index" \
    --report_gbits "$remote_ip"
}

case ${1:-} in
  ip)
    [[ $# -eq 1 ]] || fail "ip mode takes no additional arguments"
    check_ip
    ;;
  rdma-server)
    [[ $# -eq 3 ]] || fail "rdma-server requires link-index and gid-index"
    run_rdma_server "$2" "$3"
    ;;
  rdma-client)
    [[ $# -eq 3 ]] || fail "rdma-client requires link-index and gid-index"
    run_rdma_client "$2" "$3"
    ;;
  *)
    usage
    exit 2
    ;;
esac
