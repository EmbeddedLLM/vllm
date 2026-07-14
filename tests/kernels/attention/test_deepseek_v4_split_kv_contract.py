# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.models.deepseek_v4.amd.rocm as rocm_module
import vllm.models.deepseek_v4.amd.v4_kernels.paged_decode as paged_decode_module
import vllm.models.deepseek_v4.amd.v4_kernels.paged_prefill as paged_prefill_module
import vllm.models.deepseek_v4.amd.v4_kernels.qk_norm_rope_maybe_quant as qk_module
from vllm.models.deepseek_v4.amd.v4_kernels import (
    sparse_attn_v4_paged_decode_split_kv,
    sparse_attn_v4_paged_prefill_split_kv,
)
from vllm.models.deepseek_v4.amd.v4_kernels.paged_decode import (
    sparse_attn_v4_paged_decode_split_kv_reference,
)
from vllm.models.deepseek_v4.amd.v4_kernels.paged_decode_indices import (
    write_v4_paged_decode_indices_reference,
)
from vllm.models.deepseek_v4.amd.v4_kernels.qk_norm_rope_maybe_quant import (
    qk_norm_rope_maybe_quant,
    qk_norm_rope_maybe_quant_reference,
)
from vllm.models.deepseek_v4.amd.v4_kernels.state_writes import (
    swa_write,
    swa_write_reference,
)

_PACKED_TOKEN_DATA_SIZE = 576
_PACKED_TOKEN_SCALE_SIZE = 8
_PACKED_NOPE_BYTES = 448
_PACKED_ROPE_BYTES = 128


def test_qk_norm_rope_prefers_aiter_paged_abi_when_slot_mapping_is_present(
    monkeypatch,
):
    tokens, heads, dim, rope_dim, block_size = 2, 1, 512, 64, 2
    calls = []

    def fake_flydsl(q, kv, _weight, _cos, _sin, _positions, **kwargs):
        calls.append(kwargs)
        kwargs["q_out"].copy_(q.view(tokens, heads, dim))
        kwargs["kv_out"].copy_(kv)
        return kwargs["q_out"], kwargs["kv_out"], None, None

    monkeypatch.setattr(qk_module, "_FLYDSL_AVAILABLE", True)
    monkeypatch.setattr(qk_module, "_FLYDSL_HAS_PAGED_SWA", True)
    monkeypatch.setattr(qk_module, "flydsl_qk_norm_rope_quant", fake_flydsl)

    q = torch.randn(tokens, heads * dim, dtype=torch.bfloat16)
    kv = torch.randn(tokens, dim, dtype=torch.bfloat16)
    weight = torch.ones(dim, dtype=torch.bfloat16)
    positions = torch.arange(tokens, dtype=torch.int64)
    cos = torch.ones(tokens, rope_dim // 2, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)
    swa_kv = torch.zeros(2, block_size, dim, dtype=torch.bfloat16)
    block_tables = torch.tensor([[1, 0]], dtype=torch.int32)

    qk_module.qk_norm_rope_maybe_quant(
        q,
        kv,
        weight,
        cos,
        sin,
        positions,
        heads,
        dim,
        rope_dim,
        1.0e-6,
        swa_kv=swa_kv,
        batch_id_per_token=torch.zeros(tokens, dtype=torch.int32),
        swa_block_tables=block_tables,
        swa_block_size=block_size,
        swa_slot_mapping=torch.arange(tokens, dtype=torch.int64),
    )

    assert len(calls) == 1
    assert calls[0]["swa_kv"].shape == (4, dim)
    assert calls[0]["swa_kv"].is_contiguous()
    assert calls[0]["swa_block_tables"] is block_tables
    assert "swa_slot_mapping" not in calls[0]


@pytest.mark.parametrize(
    ("tokens", "cache_shape", "cache_dtype", "expected"),
    [
        (32, (4, 8, 512), torch.bfloat16, True),
        (0, (4, 8, 512), torch.bfloat16, False),
        (32, (4, 8, 512), torch.uint8, False),
        (32, (32, 512), torch.bfloat16, False),
    ],
)
def test_atom_paged_swa_authority_contract(
    tokens: int,
    cache_shape: tuple[int, ...],
    cache_dtype: torch.dtype,
    expected: bool,
):
    attn = SimpleNamespace(
        compress_ratio=4,
        _atom_layer_id=0,
    )
    metadata = SimpleNamespace(block_table=torch.zeros(2, 4, dtype=torch.int32))
    state = SimpleNamespace(num_actual_tokens=tokens)

    assert (
        rocm_module._atom_paged_swa_is_authoritative(
            attn,
            metadata,
            state,
            torch.empty(cache_shape, dtype=cache_dtype),
        )
        is expected
    )


@pytest.mark.parametrize(("num_prefills", "expected_fused"), [(0, True), (1, False)])
def test_atom_paged_swa_write_dispatch(monkeypatch, num_prefills, expected_fused):
    tokens, heads, dim, rope_dim = 2, 1, 4, 2
    state = SimpleNamespace(
        num_actual_tokens=tokens,
        num_actual_reqs=tokens,
        # Full CUDA graphs can pad this metadata beyond the model input.
        batch_id_per_token=torch.tensor([0, 1, -1, -1], dtype=torch.int32),
        state_slot_mapping=torch.arange(tokens, dtype=torch.int32),
        query_start_loc=torch.arange(tokens + 1, dtype=torch.int32),
        win_with_spec=8,
    )
    metadata = SimpleNamespace(
        num_prefills=num_prefills,
        num_decodes=0 if num_prefills else tokens,
        slot_mapping=torch.arange(tokens, dtype=torch.int64),
        block_table=torch.tensor([[2, 0], [1, 3]], dtype=torch.int32),
        block_size=8,
        max_query_len=1,
    )
    attn = SimpleNamespace(
        n_local_heads=heads,
        padded_heads=heads,
        head_dim=dim,
        rope_head_dim=rope_dim,
        eps=1.0e-6,
        compress_ratio=4,
        _atom_layer_id=0,
        kv_norm=SimpleNamespace(
            weight=SimpleNamespace(data=torch.ones(dim)),
        ),
        swa_cache_layer=SimpleNamespace(
            prefix="swa",
            kv_cache=torch.empty(4, 8, dim, dtype=torch.bfloat16),
        ),
        _atom_rotary_cos_sin=lambda dtype: (
            torch.ones(8, 1, 1, rope_dim // 2, dtype=dtype),
            torch.zeros(8, 1, 1, rope_dim // 2, dtype=dtype),
        ),
    )
    qk_kwargs = None

    def fake_qk_norm(q, kv, *args, **kwargs):
        nonlocal qk_kwargs
        qk_kwargs = kwargs
        return q.view(tokens, heads, dim), kv, None, None

    monkeypatch.setattr(
        rocm_module,
        "get_deepseek_v4_rocm_atom_state",
        lambda _: state,
    )
    monkeypatch.setattr(
        rocm_module,
        "qk_norm_rope_maybe_quant",
        fake_qk_norm,
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
    assert attn._atom_paged_swa_authoritative
    assert attn._atom_swa_write_fused is expected_fused
    assert qk_kwargs is not None
    if expected_fused:
        assert qk_kwargs["swa_kv"] is attn.swa_cache_layer.kv_cache
        assert qk_kwargs["swa_block_tables"] is metadata.block_table
        assert qk_kwargs["swa_block_size"] == metadata.block_size
        assert qk_kwargs["swa_write_per_batch"] == metadata.max_query_len
        # The block table is authoritative for content-addressed paged SWA.
        # Passing a direct slot mapping as well would make the local fallback
        # bypass the paged path and would not match the newer AITER ABI.
        assert qk_kwargs["swa_slot_mapping"] is None
        assert torch.equal(
            qk_kwargs["batch_id_per_token"],
            state.batch_id_per_token[:tokens],
        )
    else:
        assert qk_kwargs["swa_kv"] is None
        assert qk_kwargs["swa_block_tables"] is None
        assert qk_kwargs["swa_slot_mapping"] is None


def test_paged_swa_write_reference_uses_block_table():
    block_size, dim = 2, 4
    kv = torch.arange(3 * dim, dtype=torch.bfloat16).view(3, dim)
    positions = torch.tensor([0, 1, 2], dtype=torch.int64)
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32)
    state_slots = torch.tensor([7], dtype=torch.int32)
    block_tables = torch.tensor([[2, 0]], dtype=torch.int32)
    cache = torch.zeros(3, block_size, dim, dtype=torch.bfloat16)

    swa_write_reference(
        kv,
        positions,
        cu_seqlens,
        state_slots,
        cache,
        block_size,
        3,
        block_tables=block_tables,
        block_size=block_size,
    )

    torch.testing.assert_close(cache[2, 0], kv[0])
    torch.testing.assert_close(cache[2, 1], kv[1])
    torch.testing.assert_close(cache[0, 0], kv[2])


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required for strided paged-SWA kernels",
)
def test_strided_paged_swa_write_and_split_attention_rocm():
    device = "cuda"
    torch.manual_seed(17)
    block_size, dim = 2, 512
    # vLLM packs layer pages into one block-strided backing allocation.
    # Selecting one layer leaves dense token rows but a gap between blocks.
    backing = torch.zeros((2, 2, block_size, dim), device=device, dtype=torch.bfloat16)
    strided_swa = backing[:, 1]
    assert not strided_swa.is_contiguous()
    assert strided_swa.stride(1) == dim

    kv = torch.randn((4, dim), device=device, dtype=torch.bfloat16) * 0.01
    positions = torch.arange(4, device=device, dtype=torch.int64)
    cu_seqlens = torch.tensor([0, 4], device=device, dtype=torch.int32)
    state_slots = torch.tensor([0], device=device, dtype=torch.int32)
    block_tables = torch.tensor([[1, 0]], device=device, dtype=torch.int32)
    swa_write(
        kv,
        positions,
        cu_seqlens,
        state_slots,
        strided_swa,
        block_size,
        4,
        block_tables=block_tables,
        block_size=block_size,
    )
    dense_swa = strided_swa.clone()
    torch.testing.assert_close(strided_swa[1, 0], kv[0])
    torch.testing.assert_close(strided_swa[0, 1], kv[3])

    q = torch.randn((1, 2, dim), device=device, dtype=torch.bfloat16) * 0.01
    compressed_backing = (
        torch.randn((2, 2, 2, dim), device=device, dtype=torch.bfloat16) * 0.01
    )
    strided_compressed = compressed_backing[:, 1]
    dense_compressed = strided_compressed.clone()
    assert not strided_compressed.is_contiguous()
    prefix_indices = torch.tensor([0, 3, 4, 5], device=device, dtype=torch.int32)
    prefix_indptr = torch.tensor([0, 4], device=device, dtype=torch.int32)
    attn_sink = torch.zeros((2,), device=device, dtype=torch.float32)

    decode_strided = sparse_attn_v4_paged_decode_split_kv(
        q,
        strided_swa,
        strided_compressed,
        prefix_indices,
        prefix_indptr,
        attn_sink,
        dim**-0.5,
        swa_pages=4,
        kv_splits=1,
    )
    decode_dense = sparse_attn_v4_paged_decode_split_kv(
        q,
        dense_swa,
        dense_compressed,
        prefix_indices,
        prefix_indptr,
        attn_sink,
        dim**-0.5,
        swa_pages=4,
        kv_splits=1,
    )

    extend_kv = torch.randn((1, dim), device=device, dtype=torch.bfloat16) * 0.01
    extend_indices = torch.tensor([0], device=device, dtype=torch.int32)
    extend_indptr = torch.tensor([0, 1], device=device, dtype=torch.int32)
    prefill_strided = sparse_attn_v4_paged_prefill_split_kv(
        q,
        strided_swa,
        strided_compressed,
        prefix_indices,
        prefix_indptr,
        extend_kv,
        extend_indices,
        extend_indptr,
        attn_sink,
        dim**-0.5,
        swa_pages=4,
    )
    prefill_dense = sparse_attn_v4_paged_prefill_split_kv(
        q,
        dense_swa,
        dense_compressed,
        prefix_indices,
        prefix_indptr,
        extend_kv,
        extend_indices,
        extend_indptr,
        attn_sink,
        dim**-0.5,
        swa_pages=4,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(decode_strided, decode_dense, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(prefill_strided, prefill_dense, atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.hip is None,
    reason="ROCm GPU required for fused paged-SWA write",
)
def test_qk_norm_rope_fuses_strided_paged_swa_write_rocm():
    device = "cuda"
    torch.manual_seed(29)
    tokens, heads, dim, rope_dim, block_size = 4, 1, 512, 64, 2
    q = torch.randn(tokens, heads * dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(tokens, dim, device=device, dtype=torch.bfloat16)
    weight = torch.randn(dim, device=device, dtype=torch.bfloat16)
    positions = torch.arange(tokens, device=device, dtype=torch.int64)
    cos = torch.ones(tokens, rope_dim // 2, device=device, dtype=torch.bfloat16)
    sin = torch.zeros_like(cos)

    backing = torch.zeros((2, 3, block_size, dim), device=device, dtype=torch.bfloat16)
    strided_swa = backing[:, 1]
    block_tables = torch.tensor([[1, 0], [1, 0]], device=device, dtype=torch.int32)
    batch_ids = torch.tensor([0, 0, 1, 1], device=device, dtype=torch.int32)
    slot_mapping = torch.tensor([2, 3, 0, 1], device=device, dtype=torch.int64)

    q_out, kv_out, _, _ = qk_norm_rope_maybe_quant(
        q,
        kv,
        weight,
        cos,
        sin,
        positions,
        heads,
        dim,
        rope_dim,
        1.0e-6,
        swa_kv=strided_swa,
        batch_id_per_token=batch_ids,
        swa_block_tables=block_tables,
        swa_block_size=block_size,
        swa_slot_mapping=slot_mapping,
    )
    q_ref, kv_ref, _, _ = qk_norm_rope_maybe_quant_reference(
        q,
        kv,
        weight,
        cos,
        sin,
        positions,
        heads,
        dim,
        rope_dim,
        1.0e-6,
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(q_out, q_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(kv_out, kv_ref, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(strided_swa[1], kv_out[:2])
    torch.testing.assert_close(strided_swa[0], kv_out[2:])


def test_paged_decode_indices_reference_uses_swa_block_table():
    positions = torch.tensor([5], dtype=torch.int64)
    indptr = torch.tensor([0, 4], dtype=torch.int32)
    outputs = [torch.full((4,), -1, dtype=torch.int32) for _ in range(3)]

    write_v4_paged_decode_indices_reference(
        state_slot_per_seq=torch.tensor([9], dtype=torch.int32),
        batch_id_per_token=torch.tensor([0], dtype=torch.int32),
        positions=positions,
        swa_indptr=indptr,
        csa_indptr=indptr,
        hca_indptr=indptr,
        swa_indices=outputs[0],
        csa_indices=outputs[1],
        hca_indices=outputs[2],
        swa_block_tables=torch.tensor([[4, 1, 3]], dtype=torch.int32),
        swa_block_size=2,
        T=1,
        win=4,
        cs=4,
        max_pages=10,
    )

    expected = torch.tensor([2, 3, 6, 7], dtype=torch.int32)
    for output in outputs:
        torch.testing.assert_close(output, expected)


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
    q = torch.randn((tokens, heads, dim), device=device, dtype=torch.bfloat16) * 0.01
    unified_kv = torch.randn((64, dim), device=device, dtype=torch.bfloat16) * 0.01
    extend_kv = torch.randn((4, dim), device=device, dtype=torch.bfloat16) * 0.01
    prefix_indices = (
        torch.arange(tokens * prefix_len, device=device, dtype=torch.int32)
        % unified_kv.shape[0]
    )
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
