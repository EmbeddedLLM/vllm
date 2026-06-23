# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.compressor import (
    _validate_atom_packed_fp8_kv_cache,
)


def test_packed_fp8_compressor_rejects_sidecar_scales():
    packed_tail = torch.zeros((1, 4, 584), dtype=torch.uint8)
    sidecar_scales = torch.ones((4, 8), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="embedded UE8M0 scales"):
        _validate_atom_packed_fp8_kv_cache(packed_tail, sidecar_scales)


def test_packed_fp8_compressor_rejects_missing_cache():
    with pytest.raises(RuntimeError, match="atom_kv_cache is not bound"):
        _validate_atom_packed_fp8_kv_cache(None, None)


def test_packed_fp8_compressor_rejects_bad_geometry():
    bad_tail = torch.zeros((1, 4, 512), dtype=torch.uint8)

    with pytest.raises(RuntimeError, match=r"\[num_blocks, k_per_block, 584\]"):
        _validate_atom_packed_fp8_kv_cache(bad_tail, None)


def test_packed_fp8_compressor_accepts_packed_tail_without_sidecar_scales():
    packed_tail = torch.zeros((1, 4, 584), dtype=torch.uint8)

    _validate_atom_packed_fp8_kv_cache(packed_tail, None)
