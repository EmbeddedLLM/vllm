#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Conch W4A16 mixed-precision kernel (ROCm).

Covers can_implement's tile preconditions, that an explicit
--linear-backend fails loudly for shapes it cannot implement,
auto-selection, and numerics with the padded-N weight cache.

Run `pytest tests/kernels/quantization/test_conch_w4a16.py`.
"""

import pytest
import torch

import vllm.model_executor.kernels.linear as linear_kernels_pkg
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.conch import (
    ConchLinearKernel,
    _conch_tile_sizes,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

# This test module is ROCm/Conch specific. Avoid import-time failures on
# non-ROCm or environments without conch by skipping early.
if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

pytest.importorskip("conch")

device = "cuda"


def _make_config(
    K: int,
    N: int,
    group_size: int,
    weight_type=scalar_types.uint4b8,
    act_type=torch.bfloat16,
    zero_points: bool = False,
) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=weight_type,
        act_type=act_type,
        group_size=group_size,
        zero_points=zero_points,
    )


def test_conch_tile_sizes_match_tile_sanity():
    block_n, block_k = _conch_tile_sizes()
    for block in (block_n, block_k):
        assert isinstance(block, int)
        assert block > 0 and block % 32 == 0


@pytest.mark.parametrize(
    "K,N,group_size,expected,reason_substr",
    [
        (5120, 16384, 128, True, ""),
        # N=96 is padded to the tile width in process_weights_after_loading;
        # it must NOT be rejected (this is the issue-60581 shape).
        (5120, 96, 128, True, ""),
        # K-side violations cannot be padded away.
        (192, 1024, 128, False, "group size"),  # K % gs != 0
        (196, 1024, 128, False, "block_k"),  # K % block_k != 0
        # Channelwise: gs resolves to K, which is tile-aligned here.
        (5120, 16384, -1, True, ""),
    ],
)
def test_conch_can_implement_matrix(K, N, group_size, expected, reason_substr):
    config = _make_config(K, N, group_size)
    can_implement, reason = ConchLinearKernel.can_implement(config)
    assert can_implement is expected
    if not expected:
        assert reason is not None and reason_substr in reason


def test_explicit_backend_raises_for_unimplementable_shape(monkeypatch):
    """--linear-backend conch + K=196 (rejected by conch) raises ValueError with
    the rejection reason instead of silently running another backend."""
    monkeypatch.setattr(linear_kernels_pkg, "_get_linear_backend", lambda **_: "conch")
    config = _make_config(196, 96, 128)

    with pytest.raises(ValueError) as excinfo:
        choose_mp_linear_kernel(config)

    message = str(excinfo.value)
    assert "WNA16 linear layer" in message
    assert "ConchLinearKernel" in message
    assert "must be divisible" in message


def test_auto_backend_still_raises_when_nothing_implements(monkeypatch):
    """With backend "auto" the pre-fallback behavior is unchanged: every real
    kernel rejects K=196 (triton: K % group_size, conch: K % block_k,
    exllama: bf16 activations; RDNA3/Hybrid: not gfx1x) -> ValueError."""
    monkeypatch.setattr(linear_kernels_pkg, "_get_linear_backend", lambda **_: "auto")
    config = _make_config(196, 96, 128)

    with pytest.raises(ValueError) as excinfo:
        choose_mp_linear_kernel(config)

    message = str(excinfo.value)
    assert "WNA16 linear layer" in message
    assert "TritonW4A16LinearKernel" in message
    assert "ConchLinearKernel" in message


def test_uint8b128_auto_selects_conch():
    # triton rejects uint8 (triton_w4a16.py:33-36, only uint4b8/uint4) and
    # exllama rejects bf16 activations (exllama.py:43-44, fp16 only), so Conch
    # is the only candidate for uint8b128 + bf16 on gfx950 (N=96 is padded).
    config = _make_config(
        5120, 96, 128, weight_type=scalar_types.uint8b128, zero_points=False
    )
    assert choose_mp_linear_kernel(config) is ConchLinearKernel


def _build_quark_layer(monkeypatch, K, N, w_ckpt_kn8, scale_gn):
    """Create a QuarkW4A16Int4 layer (real parameter construction + process
    sequence) wired to ConchLinearKernel with random packed checkpoint data."""
    from vllm.model_executor.layers.quantization.quark.schemes import (
        quark_w4a16_int4 as quark_w4a16_int4_module,
    )
    from vllm.model_executor.layers.quantization.quark.schemes.quark_w4a16_int4 import (
        QuarkW4A16Int4,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4Static

    monkeypatch.setattr(
        quark_w4a16_int4_module,
        "choose_mp_linear_kernel",
        lambda config, compute_capability=None: ConchLinearKernel,
    )

    scheme = QuarkW4A16Int4(
        weight_quant_key=kInt4Static,
        activation_quant_key=None,
        pack_method="reorder",
        weight_config={"group_size": 128, "symmetric": True},
    )
    layer = torch.nn.Module()
    # The kernel instance locks in the config before weights exist.
    assert scheme.group_size == 128 and scheme.is_symmetric
    scheme.create_weights(
        layer=layer,
        output_partition_sizes=[N],
        input_size_per_partition=K,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
        input_size=K,
        output_size=N,
    )
    layer.weight.data = w_ckpt_kn8
    layer.weight_scale.data = scale_gn
    scheme.process_weights_after_loading(layer)
    return scheme, layer


def _quark_reference(x, w_q_kn8, scale_gn, *, group_size, weight_bias):
    """fp32 dequant reference reading the POST-transform [K/8, N] nibbles:
    nibble at shift 4*(k%8), minus weight_bias, times scale[k//gs, n]."""
    K, N = x.shape[1], w_q_kn8.shape[1]
    shifts = torch.arange(8, device=w_q_kn8.device, dtype=torch.int32) * 4
    nibbles = ((w_q_kn8.unsqueeze(-1) >> shifts) & 0xF).permute(0, 2, 1)
    w_fp = nibbles.reshape(K, N).to(torch.float32) - weight_bias
    w_fp = w_fp * scale_gn.to(torch.float32).repeat_interleave(group_size, dim=0)
    return (x.to(torch.float32) @ w_fp).to(x.dtype)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_conch_padded_n96_numerics_and_cross_config_invariance(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA/HIP device not available")

    # PackedvLLMParameter captures the TP rank at construction time.
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method="tcp://127.0.0.1:0",
            local_rank=0,
        )
        ensure_model_parallel_initialized(1, 1)

    set_random_seed(0)

    K, N, G = 5120, 96, 128
    w_ckpt = torch.randint(0, 16, (K, N // 8), device=device, dtype=torch.int32)
    scale = (0.05 * torch.rand((K // G, N), device=device, dtype=torch.float32)).to(
        torch.bfloat16
    )

    scheme96, layer96 = _build_quark_layer(monkeypatch, K, N, w_ckpt, scale)

    # Registered parameters keep their shapes; pads live in a plain attribute
    # that must not enter state_dict.
    assert tuple(layer96.weight.shape) == (K // 8, N)
    assert tuple(layer96.weight_scale.shape) == (K // G, N)
    w_q_pad = layer96._conch_padded_w_q
    w_s_pad = layer96._conch_padded_w_s
    w_zp_pad = layer96._conch_padded_w_zp
    assert w_q_pad is not None and w_s_pad is not None
    assert tuple(w_q_pad.shape) == (K // 8, 128)
    assert tuple(w_s_pad.shape) == (K // G, 128)
    assert w_zp_pad is None  # symmetric uint4b8: no zero points
    # non-persistent buffers: visible to reload snapshots, absent from state_dict
    assert not any(k.startswith("_conch_padded") for k in layer96.state_dict())

    M = 17
    x = (0.25 * torch.randn((M, K), device=device, dtype=torch.float32)).to(
        torch.bfloat16
    )
    out = scheme96.apply_weights(layer96, x)
    assert out.shape == (M, N)

    ref = _quark_reference(
        x,
        layer96.weight.data,
        layer96.weight_scale.data,
        group_size=G,
        weight_bias=scalar_types.uint4b8.bias,
    )
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    # Cross-config invariance: the same weights configured as N=64 (no padding)
    # must produce bit-identical first 64 columns - tile-identical math.
    scheme64, layer64 = _build_quark_layer(
        monkeypatch, K, 64, w_ckpt[:, : 64 // 8], scale[:, :64]
    )
    assert layer64._conch_padded_w_q is None
    out64 = scheme64.apply_weights(layer64, x)
    assert out64.shape == (M, 64)
    assert torch.equal(out[:, :64], out64), (
        "padded N=96 and tile-aligned N=64 must be bit-identical for cols 0..63"
    )
