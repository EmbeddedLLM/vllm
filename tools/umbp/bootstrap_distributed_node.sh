#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

set -euo pipefail

VLLM_REF=${VLLM_REF:-a90d83a602f8639b078af3f2c44b18ef1f546d7d}
MORI_REF=${MORI_REF:-dafdcfcf1e27b0c981b90903ab198b90d29e6867}
LLMD_REF=${LLMD_REF:-761016e68094b7a2b7e4f0e3d2160007468b3fa9}
SETUP_ROOT=${1:-"$PWD/umbp-node"}

VLLM_URL=${VLLM_URL:-https://github.com/EmbeddedLLM/vllm.git}
MORI_URL=${MORI_URL:-https://github.com/ROCm/mori.git}
LLMD_URL=${LLMD_URL:-https://github.com/EmbeddedLLM/llm-d-router.git}

fail() {
  echo "error: $*" >&2
  exit 1
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || fail "missing command: $1"
}

checkout_repo() {
  local url=$1
  local ref=$2
  local destination=$3
  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$url" "$destination"
  elif [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    fail "existing checkout has changes: $destination"
  fi
  git -C "$destination" fetch origin "$ref"
  git -C "$destination" checkout --detach "$ref"
  test "$(git -C "$destination" rev-parse HEAD)" = "$ref" ||
    fail "failed to check out $ref in $destination"
}

require_command git
require_command curl
require_command cmake
require_command ninja
require_command gcc
require_command g++
require_command hipcc
require_command ip
require_command ldconfig

[[ -e /dev/kfd ]] || fail "/dev/kfd is not available"
[[ -d /sys/class/infiniband ]] || fail "no RDMA devices are visible"
ldconfig -p 2>/dev/null | grep -q libibverbs ||
  fail "libibverbs is missing; install libibverbs-dev and ibverbs-utils"
ldconfig -p 2>/dev/null | grep -q libpci ||
  fail "libpci is missing; install libpci-dev"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
require_command uv

mkdir -p "$SETUP_ROOT"
SETUP_ROOT=$(cd "$SETUP_ROOT" && pwd)
VLLM_DIR="$SETUP_ROOT/vllmumbp"
MORI_DIR="$SETUP_ROOT/mori"
LLMD_DIR="$SETUP_ROOT/llm-d-router"

checkout_repo "$VLLM_URL" "$VLLM_REF" "$VLLM_DIR"
checkout_repo "$MORI_URL" "$MORI_REF" "$MORI_DIR"
checkout_repo "$LLMD_URL" "$LLMD_REF" "$LLMD_DIR"

uv venv --python 3.12 "$SETUP_ROOT/.venv"
UV_PYTHON="$SETUP_ROOT/.venv/bin/python"
export UV_PYTHON

uv pip install cmake ninja pybind11 'Cython>=3.0' setuptools setuptools-scm wheel
BUILD_UMBP=ON BUILD_UMBP_SPDK=OFF \
  uv pip install --no-build-isolation --editable "$MORI_DIR"
VLLM_USE_PRECOMPILED=1 \
  uv pip install --editable "$VLLM_DIR" --torch-backend=auto
uv pip install -r "$VLLM_DIR/requirements/test/cuda.in"

"$UV_PYTHON" - <<'PY'
import torch
from mori.cpp import UMBPClient, UMBPConfig

assert torch.cuda.is_available(), "ROCm GPU is not available through PyTorch"
assert UMBPClient is not None and UMBPConfig is not None
print(f"PyTorch GPUs: {torch.cuda.device_count()}")
print("MoRI UMBP Python bindings: OK")
PY

echo "RDMA devices:"
for device_path in /sys/class/infiniband/*; do
  device=$(basename "$device_path")
  for interface_path in "$device_path"/device/net/*; do
    [[ -e "$interface_path" ]] || continue
    interface=$(basename "$interface_path")
    echo "  $device -> $interface"
    ip -6 -o addr show dev "$interface" scope global || true
  done
done

if command -v go >/dev/null 2>&1 && [[ "$(go version)" == *"go1.26.6"* ]]; then
  (cd "$LLMD_DIR" && go test ./pkg/kvevents ./pkg/epp/framework/plugins/scheduling/scorer/restorecost)
else
  echo "llm-d source was cloned but not built: Go 1.26.6 is required."
fi

cat <<EOF

Setup complete:
  environment: $SETUP_ROOT/.venv
  vLLM:       $VLLM_DIR
  MoRI:       $MORI_DIR
  llm-d:      $LLMD_DIR

Activate with:
  source $SETUP_ROOT/.venv/bin/activate

This script did not change firewall rules, RDMA configuration, hugepages, or NVMe devices.
EOF
