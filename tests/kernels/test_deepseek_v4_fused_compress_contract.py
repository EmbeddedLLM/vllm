# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.deepseek_v4.amd.v4_kernels import CompressPlan
from vllm.models.deepseek_v4.amd.v4_kernels import (
    fused_compress as fused_compress_module,
)
from vllm.models.deepseek_v4.amd.v4_kernels.fused_compress import (
    _validate_packed_fp8_ds_mla_fused_compress_args,
    fused_compress_attn,
)


def _validate(**overrides):
    args = {
        "kv_cache": torch.zeros((1, 4, 584), dtype=torch.uint8),
        "cache_scale": None,
        "has_scatter_map": True,
        "head_dim": 512,
        "rope_head_dim": 64,
        "k_per_block": 4,
        "fp8_max": 448.0,
    }
    args.update(overrides)
    _validate_packed_fp8_ds_mla_fused_compress_args(**args)


def test_fused_compress_packed_fp8_accepts_valid_tail():
    _validate()


def test_fused_compress_packed_fp8_rejects_sidecar_scales():
    with pytest.raises(RuntimeError, match="embedded UE8M0 scales"):
        _validate(cache_scale=torch.ones((4, 8), dtype=torch.float32))


def test_fused_compress_packed_fp8_rejects_missing_scatter_map():
    with pytest.raises(RuntimeError, match="requires block_tables or kv_slot_mapping"):
        _validate(has_scatter_map=False)


def test_fused_compress_packed_fp8_rejects_bad_tail_geometry():
    with pytest.raises(RuntimeError, match=r"\[num_blocks, k_per_block, 584\]"):
        _validate(kv_cache=torch.zeros((1, 4, 512), dtype=torch.uint8))


def test_fused_compress_packed_fp8_rejects_wrong_head_dims():
    with pytest.raises(RuntimeError, match="head_dim=512"):
        _validate(head_dim=256)

    with pytest.raises(RuntimeError, match="rope_head_dim=64"):
        _validate(rope_head_dim=32)


def test_fused_compress_attn_packed_fp8_validates_before_generic_asserts():
    head_dim = 512
    ratio = 4
    plan = CompressPlan(
        compress_plan_gpu=torch.tensor([[0, 0, 0, 1]], dtype=torch.int32),
        write_plan_gpu=torch.empty((0, 4), dtype=torch.int32),
        num_compress=0,
        num_write=0,
        cu_compress_cpu=torch.zeros(2, dtype=torch.int32).numpy(),
    )

    with pytest.raises(RuntimeError, match="requires kv_cache"):
        fused_compress_attn(
            kv_in=torch.zeros((1, head_dim), dtype=torch.bfloat16),
            score_in=torch.zeros((1, head_dim), dtype=torch.bfloat16),
            kv_state=torch.zeros((1, ratio, head_dim), dtype=torch.bfloat16),
            score_state=torch.zeros((1, ratio, head_dim), dtype=torch.bfloat16),
            plan=plan,
            state_slot_mapping=torch.zeros((1,), dtype=torch.int32),
            ape=torch.zeros((ratio, head_dim), dtype=torch.bfloat16),
            rms_weight=torch.ones((head_dim,), dtype=torch.bfloat16),
            rms_eps=1e-6,
            cos_cache=torch.ones((1, 32), dtype=torch.bfloat16),
            sin_cache=torch.zeros((1, 32), dtype=torch.bfloat16),
            kv_cache=None,
            block_tables=torch.zeros((1, 1), dtype=torch.int32),
            k_per_block=4,
            overlap=False,
            ratio=ratio,
            head_dim=head_dim,
            rope_head_dim=64,
            fp8_max=448.0,
            packed_fp8_ds_mla=True,
        )


def test_fused_compress_attn_packed_fp8_bypasses_flydsl(monkeypatch):
    class _FakeKernel:
        def __init__(self):
            self.calls = []

        def __getitem__(self, grid):
            def _launch(*args, **kwargs):
                self.calls.append((grid, args, kwargs))

            return _launch

    def _unexpected_flydsl(*args, **kwargs):
        raise AssertionError("packed fp8_ds_mla must not dispatch to FlyDSL")

    fake_kernel = _FakeKernel()
    monkeypatch.setattr(
        fused_compress_module, "flydsl_fused_compress_attn", _unexpected_flydsl
    )
    monkeypatch.setattr(
        fused_compress_module, "_fused_compress_attn_kernel", fake_kernel
    )

    head_dim = 512
    ratio = 4
    dim_full = 2 * head_dim
    plan = CompressPlan(
        compress_plan_gpu=torch.tensor([[0, 0, -1, 0]], dtype=torch.int32),
        write_plan_gpu=torch.empty((0, 4), dtype=torch.int32),
        num_compress=0,
        num_write=0,
        cu_compress_cpu=torch.zeros(2, dtype=torch.int32).numpy(),
    )

    fused_compress_attn(
        kv_in=torch.zeros((1, dim_full), dtype=torch.bfloat16),
        score_in=torch.zeros((1, dim_full), dtype=torch.bfloat16),
        kv_state=torch.zeros((1, 2 * ratio, dim_full), dtype=torch.bfloat16),
        score_state=torch.zeros((1, 2 * ratio, dim_full), dtype=torch.bfloat16),
        plan=plan,
        state_slot_mapping=torch.zeros((1,), dtype=torch.int32),
        ape=torch.zeros((ratio, dim_full), dtype=torch.bfloat16),
        rms_weight=torch.ones((head_dim,), dtype=torch.bfloat16),
        rms_eps=1e-6,
        cos_cache=torch.ones((1, 32), dtype=torch.bfloat16),
        sin_cache=torch.zeros((1, 32), dtype=torch.bfloat16),
        kv_cache=torch.zeros((1, 4, 584), dtype=torch.uint8),
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        k_per_block=4,
        overlap=True,
        ratio=ratio,
        head_dim=head_dim,
        rope_head_dim=64,
        fp8_max=448.0,
        packed_fp8_ds_mla=True,
    )

    assert len(fake_kernel.calls) == 1
    _, _, kernel_kwargs = fake_kernel.calls[0]
    assert kernel_kwargs["PACKED_FP8_DS_MLA"] == 1
