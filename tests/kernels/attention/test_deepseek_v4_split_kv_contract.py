# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.amd.rocm as rocm_module
import vllm.models.deepseek_v4.amd.v4_kernels.paged_decode as paged_decode_module
import vllm.models.deepseek_v4.amd.v4_kernels.paged_prefill as paged_prefill_module
from vllm.models.deepseek_v4.amd.v4_kernels import (
    sparse_attn_v4_paged_decode_split_kv,
    sparse_attn_v4_paged_prefill_split_kv,
)
from vllm.models.deepseek_v4.amd.v4_kernels.paged_decode import (
    sparse_attn_v4_paged_decode_split_kv_reference,
)

_PACKED_TOKEN_DATA_SIZE = 576
_PACKED_TOKEN_SCALE_SIZE = 8
_PACKED_NOPE_BYTES = 448
_PACKED_ROPE_BYTES = 128


@pytest.mark.parametrize(
    ("bound", "num_prefills", "num_decodes", "tokens", "reqs", "expected"),
    [
        (True, 0, 32, 32, 32, True),
        (False, 0, 32, 32, 32, False),
        (True, 1, 31, 32, 32, False),
        (True, 0, 0, 0, 0, False),
        (True, 0, 32, 64, 32, False),
    ],
)
def test_atom_direct_swa_authority_contract(
    bound: bool,
    num_prefills: int,
    num_decodes: int,
    tokens: int,
    reqs: int,
    expected: bool,
):
    attn = SimpleNamespace(
        atom_direct_swa_bound=bound,
        compress_ratio=4,
        _atom_layer_id=0,
    )
    metadata = SimpleNamespace(
        num_prefills=num_prefills,
        num_decodes=num_decodes,
    )
    state = SimpleNamespace(
        num_actual_tokens=tokens,
        num_actual_reqs=reqs,
    )

    assert (
        rocm_module._atom_direct_swa_is_authoritative(
            attn,
            metadata,
            state,
            torch.empty(1),
        )
        is expected
    )


@pytest.mark.parametrize(
    ("direct_swa_bound", "expected_fused", "expected_native_inserts"),
    [(True, True, 0), (False, False, 1)],
)
def test_atom_direct_swa_decode_write_dispatch(
    monkeypatch,
    direct_swa_bound: bool,
    expected_fused: bool,
    expected_native_inserts: int,
):
    tokens, heads, dim, rope_dim = 2, 1, 4, 2
    state = SimpleNamespace(
        num_actual_tokens=tokens,
        num_actual_reqs=tokens,
        batch_id_per_token=torch.arange(tokens, dtype=torch.int32),
        state_slot_mapping=torch.arange(tokens, dtype=torch.int32),
        win_with_spec=8,
    )
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decodes=tokens,
        slot_mapping=torch.arange(tokens, dtype=torch.int64),
        block_size=128,
    )
    attn = SimpleNamespace(
        n_local_heads=heads,
        padded_heads=heads,
        head_dim=dim,
        rope_head_dim=rope_dim,
        eps=1.0e-6,
        compress_ratio=4,
        _atom_layer_id=0,
        atom_direct_swa_bound=direct_swa_bound,
        atom_swa_kv=torch.empty(tokens, 8, dim),
        kv_norm=SimpleNamespace(
            weight=SimpleNamespace(data=torch.ones(dim)),
        ),
        swa_cache_layer=SimpleNamespace(
            prefix="swa",
            kv_cache=torch.empty(1, 1, 584, dtype=torch.uint8),
        ),
        _atom_rotary_cos_sin=lambda dtype: (
            torch.ones(8, 1, 1, rope_dim // 2, dtype=dtype),
            torch.zeros(8, 1, 1, rope_dim // 2, dtype=dtype),
        ),
    )
    native_insert_calls = 0
    fused_calls = 0

    def fake_fused(q, kv, *args, **kwargs):
        nonlocal fused_calls
        fused_calls += 1
        return q.view(tokens, heads, dim)

    def fake_qk_norm(q, kv, *args, **kwargs):
        return q.view(tokens, heads, dim), kv, None, None

    def fake_native_insert(*args, **kwargs):
        nonlocal native_insert_calls
        native_insert_calls += 1

    monkeypatch.setattr(
        rocm_module,
        "get_deepseek_v4_rocm_atom_state",
        lambda _: state,
    )
    monkeypatch.setattr(
        rocm_module,
        "fused_reduce_qk_norm_rope_swa_write",
        fake_fused,
    )
    monkeypatch.setattr(
        rocm_module,
        "qk_norm_rope_maybe_quant",
        fake_qk_norm,
    )
    monkeypatch.setattr(
        rocm_module,
        "quantize_and_insert_k_cache",
        fake_native_insert,
    )

    q = torch.randn(tokens, heads, dim)
    kv = torch.randn(tokens, dim)
    positions = torch.arange(tokens, dtype=torch.int64)
    output = rocm_module.DeepseekV4ROCMAiterMLAAttention._fused_qnorm_rope_kv_insert(
        attn,
        q,
        kv,
        positions,
        {"swa": metadata},
    )

    assert output.shape == q.shape
    assert attn._atom_direct_swa_authoritative is direct_swa_bound
    assert attn._atom_swa_write_fused is expected_fused
    assert fused_calls == int(expected_fused)
    assert native_insert_calls == expected_native_inserts


def _packed_block_bytes(packed_tail: torch.Tensor) -> torch.Tensor:
    return packed_tail.reshape(packed_tail.shape[0], -1)


def _write_packed_rope_tail(
    packed_tail: torch.Tensor,
    logical_page: int,
    rope_tail: torch.Tensor,
) -> None:
    assert packed_tail.dtype == torch.uint8
    assert rope_tail.shape == (64,)
    assert rope_tail.dtype == torch.bfloat16
    block_size = packed_tail.shape[1]
    block = logical_page // block_size
    slot = logical_page % block_size
    start = slot * _PACKED_TOKEN_DATA_SIZE + _PACKED_NOPE_BYTES
    _packed_block_bytes(packed_tail)[block, start : start + _PACKED_ROPE_BYTES].copy_(
        rope_tail.view(torch.uint8)
    )


def _write_packed_scale_bytes(
    packed_tail: torch.Tensor,
    logical_page: int,
    encoded_scales: torch.Tensor,
) -> None:
    assert packed_tail.dtype == torch.uint8
    assert encoded_scales.shape == (8,)
    assert encoded_scales.dtype == torch.uint8
    block_size = packed_tail.shape[1]
    block = logical_page // block_size
    slot = logical_page % block_size
    start = block_size * _PACKED_TOKEN_DATA_SIZE + slot * _PACKED_TOKEN_SCALE_SIZE
    _packed_block_bytes(packed_tail)[
        block, start : start + _PACKED_TOKEN_SCALE_SIZE
    ].copy_(encoded_scales)


def _decode_inputs():
    q = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
    swa_kv = torch.zeros((1, 512), dtype=torch.bfloat16)
    kv_indices = torch.zeros((1,), dtype=torch.int32)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    attn_sink = torch.zeros((1,), dtype=torch.float32)
    return q, swa_kv, kv_indices, kv_indptr, attn_sink


def _prefill_inputs():
    q = torch.zeros((1, 1, 512), dtype=torch.bfloat16)
    swa_kv = torch.zeros((1, 512), dtype=torch.bfloat16)
    kv = torch.zeros((1, 512), dtype=torch.bfloat16)
    kv_indices_prefix = torch.zeros((1,), dtype=torch.int32)
    kv_indptr_prefix = torch.tensor([0, 1], dtype=torch.int32)
    kv_indices_extend = torch.zeros((1,), dtype=torch.int32)
    kv_indptr_extend = torch.tensor([0, 1], dtype=torch.int32)
    attn_sink = torch.zeros((1,), dtype=torch.float32)
    return (
        q,
        swa_kv,
        kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    )


def test_split_kv_decode_rejects_packed_fp8_sidecar_scales():
    q, swa_kv, kv_indices, kv_indptr, attn_sink = _decode_inputs()
    packed_tail = torch.zeros((1, 1, 584), dtype=torch.uint8)
    sidecar_scales = torch.ones((1, 8), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="embedded UE8M0 scales"):
        sparse_attn_v4_paged_decode_split_kv(
            q,
            swa_kv,
            packed_tail,
            kv_indices,
            kv_indptr,
            attn_sink,
            1.0,
            swa_pages=1,
            compressed_kv_scales=sidecar_scales,
            compressed_kv_layout="fp8_ds_mla",
        )


def test_split_kv_prefill_rejects_packed_fp8_sidecar_scales():
    (
        q,
        swa_kv,
        kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    ) = _prefill_inputs()
    packed_tail = torch.zeros((1, 1, 584), dtype=torch.uint8)
    sidecar_scales = torch.ones((1, 8), dtype=torch.float32)

    with pytest.raises(RuntimeError, match="embedded UE8M0 scales"):
        sparse_attn_v4_paged_prefill_split_kv(
            q,
            swa_kv,
            packed_tail,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            1.0,
            swa_pages=1,
            compressed_kv_scales=sidecar_scales,
            compressed_kv_layout="fp8_ds_mla",
        )


def test_split_kv_decode_rejects_bad_packed_fp8_geometry():
    q, swa_kv, kv_indices, kv_indptr, attn_sink = _decode_inputs()
    bad_tail = torch.zeros((1, 1, 512), dtype=torch.uint8)

    with pytest.raises(RuntimeError, match=r"\[num_blocks, block_size, 584\]"):
        sparse_attn_v4_paged_decode_split_kv(
            q,
            swa_kv,
            bad_tail,
            kv_indices,
            kv_indptr,
            attn_sink,
            1.0,
            swa_pages=1,
            compressed_kv_layout="fp8_ds_mla",
        )


def test_split_kv_prefill_rejects_bad_packed_fp8_geometry():
    (
        q,
        swa_kv,
        kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    ) = _prefill_inputs()
    bad_tail = torch.zeros((1, 1, 512), dtype=torch.uint8)

    with pytest.raises(RuntimeError, match=r"\[num_blocks, k_per_block, 584\]"):
        sparse_attn_v4_paged_prefill_split_kv(
            q,
            swa_kv,
            bad_tail,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            1.0,
            swa_pages=1,
            compressed_kv_layout="fp8_ds_mla",
        )


def test_split_kv_decode_rejects_unknown_layout():
    q, swa_kv, kv_indices, kv_indptr, attn_sink = _decode_inputs()
    dense_tail = torch.zeros((1, 1, 512), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="Unsupported compressed_kv_layout"):
        sparse_attn_v4_paged_decode_split_kv(
            q,
            swa_kv,
            dense_tail,
            kv_indices,
            kv_indptr,
            attn_sink,
            1.0,
            swa_pages=1,
            compressed_kv_layout="unexpected",
        )


def test_split_kv_prefill_rejects_unknown_layout():
    (
        q,
        swa_kv,
        kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    ) = _prefill_inputs()
    dense_tail = torch.zeros((1, 1, 512), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="Unsupported compressed_kv_layout"):
        sparse_attn_v4_paged_prefill_split_kv(
            q,
            swa_kv,
            dense_tail,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            1.0,
            swa_pages=1,
            compressed_kv_layout="unexpected",
        )


def test_split_kv_prefill_accepts_packed_fp8_layout_before_cuda_guard():
    (
        q,
        swa_kv,
        kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    ) = _prefill_inputs()
    packed_tail = torch.zeros((1, 1, 584), dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="requires CUDA/HIP tensors"):
        sparse_attn_v4_paged_prefill_split_kv(
            q,
            swa_kv,
            packed_tail,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            1.0,
            swa_pages=1,
            compressed_kv_layout="fp8_ds_mla",
        )


def test_split_kv_prefill_packed_fp8_bypasses_opus(monkeypatch):
    def _unexpected_opus(*args, **kwargs):
        raise AssertionError("packed split-KV prefill must not dispatch to OPUS")

    monkeypatch.setattr(
        paged_prefill_module, "pa_sparse_prefill_opus", _unexpected_opus
    )
    monkeypatch.setattr(paged_prefill_module, "_HAS_OPUS", True)

    (
        q,
        swa_kv,
        kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    ) = _prefill_inputs()
    packed_tail = torch.zeros((1, 1, 584), dtype=torch.uint8)

    with pytest.raises(RuntimeError, match="requires CUDA/HIP tensors"):
        sparse_attn_v4_paged_prefill_split_kv(
            q,
            swa_kv,
            packed_tail,
            kv_indices_prefix,
            kv_indptr_prefix,
            kv,
            kv_indices_extend,
            kv_indptr_extend,
            attn_sink,
            1.0,
            swa_pages=1,
            compressed_kv_layout="fp8_ds_mla",
        )


def test_split_kv_decode_accepts_packed_fp8_layout_on_cpu_reference():
    q, swa_kv, _kv_indices, kv_indptr, attn_sink = _decode_inputs()
    packed_tail = torch.zeros((1, 1, 584), dtype=torch.uint8)
    compressed_page_indices = torch.tensor([1], dtype=torch.int32)

    out = sparse_attn_v4_paged_decode_split_kv(
        q,
        swa_kv,
        packed_tail,
        compressed_page_indices,
        kv_indptr,
        attn_sink,
        1.0,
        swa_pages=1,
        compressed_kv_layout="fp8_ds_mla",
    )

    assert out.shape == q.shape
    assert out.dtype == q.dtype
    assert torch.isfinite(out).all()


def test_packed_fp8_ds_mla_is_block_packed_not_row_interleaved():
    packed_tail = torch.zeros((1, 2, 584), dtype=torch.uint8)
    rope0 = torch.arange(64, dtype=torch.float32).to(torch.bfloat16)
    rope1 = (torch.arange(64, dtype=torch.float32) + 128).to(torch.bfloat16)
    scales0 = torch.arange(8, dtype=torch.uint8)
    scales1 = torch.arange(8, dtype=torch.uint8) + 16

    _write_packed_rope_tail(packed_tail, 0, rope0)
    _write_packed_rope_tail(packed_tail, 1, rope1)
    _write_packed_scale_bytes(packed_tail, 0, scales0)
    _write_packed_scale_bytes(packed_tail, 1, scales1)

    block_bytes = _packed_block_bytes(packed_tail)
    slot0_rope = _PACKED_NOPE_BYTES
    slot1_rope = _PACKED_TOKEN_DATA_SIZE + _PACKED_NOPE_BYTES
    slot0_scales = 2 * _PACKED_TOKEN_DATA_SIZE
    slot1_scales = slot0_scales + _PACKED_TOKEN_SCALE_SIZE

    torch.testing.assert_close(
        block_bytes[0, slot0_rope : slot0_rope + _PACKED_ROPE_BYTES].view(
            torch.bfloat16
        ),
        rope0,
    )
    torch.testing.assert_close(
        block_bytes[0, slot1_rope : slot1_rope + _PACKED_ROPE_BYTES].view(
            torch.bfloat16
        ),
        rope1,
    )
    torch.testing.assert_close(
        block_bytes[0, slot0_scales : slot0_scales + _PACKED_TOKEN_SCALE_SIZE],
        scales0,
    )
    torch.testing.assert_close(
        block_bytes[0, slot1_scales : slot1_scales + _PACKED_TOKEN_SCALE_SIZE],
        scales1,
    )

    row_interleaved_slot1_rope = 584 + _PACKED_NOPE_BYTES
    assert row_interleaved_slot1_rope != slot1_rope
    assert not torch.equal(
        block_bytes[
            0,
            row_interleaved_slot1_rope : row_interleaved_slot1_rope
            + _PACKED_ROPE_BYTES,
        ],
        rope1.view(torch.uint8),
    )

    dense = paged_decode_module._dequantize_packed_fp8_ds_mla_reference(
        packed_tail, torch.bfloat16
    )
    torch.testing.assert_close(dense[0, 448:], rope0)
    torch.testing.assert_close(dense[1, 448:], rope1)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required for Triton packed split-KV decode",
)
def test_split_kv_decode_ordered_packed_fp8_matches_reference_rocm():
    device = "cuda"
    torch.manual_seed(0)
    q = torch.randn((2, 2, 512), device=device, dtype=torch.bfloat16) * 0.01
    swa_kv = torch.randn((4, 512), device=device, dtype=torch.bfloat16) * 0.01
    packed_tail = torch.zeros((1, 2, 584), device=device, dtype=torch.uint8)
    # Packed fp8_ds_mla stores all token payloads first, then all scale bytes.
    rope_tail = torch.randn((2, 64), device=device, dtype=torch.bfloat16) * 0.01
    _write_packed_rope_tail(packed_tail, 0, rope_tail[0])
    _write_packed_rope_tail(packed_tail, 1, rope_tail[1])
    # Per-token slices are ordered as compressed head followed by SWA tail.
    kv_indices = torch.tensor([4, 0, 4, 5, 1, 2], device=device, dtype=torch.int32)
    kv_indptr = torch.tensor([0, 2, 6], device=device, dtype=torch.int32)
    positions = torch.tensor([0, 1], device=device, dtype=torch.int64)
    attn_sink = torch.zeros((2,), device=device, dtype=torch.float32)

    out_kernel = sparse_attn_v4_paged_decode_split_kv(
        q,
        swa_kv,
        packed_tail,
        kv_indices,
        kv_indptr,
        attn_sink,
        1.0,
        swa_pages=4,
        compressed_kv_layout="fp8_ds_mla",
        kv_splits=1,
        csa_positions=positions,
        csa_window_size=128,
    )

    out_ref = sparse_attn_v4_paged_decode_split_kv_reference(
        q,
        swa_kv,
        packed_tail,
        kv_indices,
        kv_indptr,
        attn_sink,
        1.0,
        swa_pages=4,
        compressed_kv_layout="fp8_ds_mla",
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out_kernel, out_ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("prefix_len", [4, 20])
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required for Triton unified-KV prefill",
)
def test_unified_prefill_tuned_paths_match_reference_rocm(prefix_len: int):
    """Exercise the short-HCA and long-prefix CSA launch specializations."""
    device = "cuda"
    torch.manual_seed(prefix_len)
    tokens, heads, dim = 2, 2, 512
    q = torch.randn(
        (tokens, heads, dim), device=device, dtype=torch.bfloat16
    ) * 0.01
    unified_kv = torch.randn((64, dim), device=device, dtype=torch.bfloat16) * 0.01
    extend_kv = torch.randn((4, dim), device=device, dtype=torch.bfloat16) * 0.01
    prefix_indices = torch.arange(
        tokens * prefix_len, device=device, dtype=torch.int32
    ) % unified_kv.shape[0]
    prefix_indptr = torch.arange(
        0,
        (tokens + 1) * prefix_len,
        prefix_len,
        device=device,
        dtype=torch.int32,
    )
    extend_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    extend_indptr = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
    attn_sink = torch.zeros((heads,), device=device, dtype=torch.float32)
    out_buffer = torch.empty_like(q)

    out_kernel = paged_prefill_module._sparse_attn_v4_paged_prefill_triton(
        q,
        unified_kv,
        prefix_indices,
        prefix_indptr,
        extend_kv,
        extend_indices,
        extend_indptr,
        attn_sink,
        dim**-0.5,
        out=out_buffer,
    )
    out_ref = paged_prefill_module.sparse_attn_v4_paged_prefill_reference(
        q,
        unified_kv,
        prefix_indices,
        prefix_indptr,
        extend_kv,
        extend_indices,
        extend_indptr,
        attn_sink,
        dim**-0.5,
    )

    assert out_kernel.data_ptr() == out_buffer.data_ptr()
    torch.cuda.synchronize()
    torch.testing.assert_close(out_kernel, out_ref, atol=2e-2, rtol=2e-2)
