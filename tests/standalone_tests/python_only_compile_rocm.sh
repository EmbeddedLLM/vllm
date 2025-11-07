#!/bin/bash
# This script tests if the Python-only compilation works correctly for ROCm
# for users who do not have any compilers installed on their system

set -e
set -x

cd /vllm-workspace/

# Uninstall vllm
pip3 uninstall -y vllm

# Restore the original files (if they were moved)
if [ -d src/vllm ]; then
    mv src/vllm ./vllm
fi

# Remove all compilers to ensure no compilation happens
apt remove --purge build-essential -y
apt autoremove -y

# Verify ROCm PyTorch is installed
python3 -c "import torch; assert torch.version.hip is not None, 'ROCm PyTorch is required'"
echo "ROCm PyTorch version: $(python3 -c 'import torch; print(torch.__version__)')"
echo "ROCm HIP version: $(python3 -c 'import torch; print(torch.version.hip)')"

# Add a test line to verify Python changes are reflected
echo 'import os; os.system("touch /tmp/rocm_changed.file")' >> vllm/__init__.py

# Install vLLM using precompiled binaries
# Use VLLM_ROCM_WHEEL_URL if set, otherwise fall back to nightly
if [ -z "$VLLM_ROCM_WHEEL_URL" ]; then
    VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL=1 VLLM_USE_PRECOMPILED=1 pip3 install -vvv -e .
else
    VLLM_USE_PRECOMPILED=1 pip3 install -vvv -e .
fi

# Run the script to import vllm
python3 -c 'import vllm'

# Check if the test file was created (verifies Python changes work)
if [ ! -f /tmp/rocm_changed.file ]; then
    echo "rocm_changed.file was not created, Python-only compilation failed"
    exit 1
fi

echo "✅ ROCm Python-only build test passed!"
echo "Python changes are reflected without recompilation."
