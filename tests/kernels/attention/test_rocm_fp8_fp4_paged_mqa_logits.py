# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import dataclasses
import inspect
import os
import random

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    calc_diff,
    fp8_fp4_paged_mqa_logits,
    get_num_sms,
    get_paged_mqa_logits_metadata,
)
from vllm.utils.math_utils import cdiv

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm(), reason="ROCm Triton kernel only"
)

MXFP4_BLOCK_SIZE = 32
FP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def test_rocm_aiter_sparse_attn_indexer_native_signature_preserved() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        rocm_aiter_sparse_attn_indexer_native,
    )

    signature = inspect.signature(rocm_aiter_sparse_attn_indexer_native)
    assert list(signature.parameters) == [
        "hidden_states",
        "k_cache_prefix",
        "kv_cache",
        "q_quant",
        "k",
        "weights",
        "quant_block_size",
        "scale_fmt",
        "topk_tokens",
        "head_dim",
        "max_model_len",
        "total_seq_lens",
        "topk_indices_buffer",
        "skip_k_cache_insert",
        "use_fp4_cache",
    ]


@dataclasses.dataclass(frozen=True)
class PagedMQACase:
    is_varlen: bool
    is_fp4: bool
    logits_dtype: torch.dtype
    block_kv: int
    batch_size: int
    next_n: int
    max_tokens_per_batch: int
    num_heads: int
    head_dim: int
    avg_kv: int

    def id(self) -> str:
        dtype = "bf16" if self.logits_dtype is torch.bfloat16 else "fp32"
        quant = "fp4" if self.is_fp4 else "fp8"
        varlen = f"varlen{self.max_tokens_per_batch}" if self.is_varlen else "dense"
        return (
            f"{varlen}-{quant}-{dtype}-blk{self.block_kv}-"
            f"b{self.batch_size}-n{self.next_n}-kv{self.avg_kv}"
        )


def _deepgemm_paged_mqa_cases() -> list[PagedMQACase]:
    cases: list[PagedMQACase] = []
    for is_varlen in (True, False):
        for is_fp4 in (True, False):
            for logits_dtype in (torch.float32, torch.bfloat16):
                for block_kv in (32, 64):
                    for batch_size in (256,):
                        next_ns = (1,) if is_varlen else (1, 2, 4, 5, 6)
                        for next_n in next_ns:
                            max_tpbs = (1, 4, 10) if is_varlen else (1,)
                            for max_tokens_per_batch in max_tpbs:
                                for num_heads, head_dim in [(32, 128), (64, 128)]:
                                    for avg_kv in (8192, 32768):
                                        cases.append(
                                            PagedMQACase(
                                                is_varlen=is_varlen,
                                                is_fp4=is_fp4,
                                                logits_dtype=logits_dtype,
                                                block_kv=block_kv,
                                                batch_size=batch_size,
                                                next_n=next_n,
                                                max_tokens_per_batch=(
                                                    max_tokens_per_batch
                                                ),
                                                num_heads=num_heads,
                                                head_dim=head_dim,
                                                avg_kv=avg_kv,
                                            )
                                        )
    return cases


DEEPGEMM_PAGED_MQA_CASES = _deepgemm_paged_mqa_cases()
FULL_DEEPGEMM_SHAPES = os.getenv("VLLM_ROCM_PAGED_MQA_FULL_DEEPGEMM_SHAPES", "0") == "1"


def _scaled_case_dims(case: PagedMQACase) -> tuple[int, int, int]:
    if FULL_DEEPGEMM_SHAPES:
        return case.batch_size, case.avg_kv, 111 * 1024
    # Preserve the DeepGEMM case matrix while keeping unit tests quick enough
    # for per-kernel validation.
    batch_size = 4 if not case.is_varlen else 3
    avg_kv = 192 if case.avg_kv == 8192 else 448
    max_model_len = int(1.4 * avg_kv) + case.block_kv * 2
    return batch_size, avg_kv, max_model_len


def _quantize_to_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_shape = x.shape
    head_dim = orig_shape[-1]
    assert head_dim % MXFP4_BLOCK_SIZE == 0
    n_blocks = head_dim // MXFP4_BLOCK_SIZE

    x_f32 = x.float().reshape(-1, n_blocks, MXFP4_BLOCK_SIZE)
    amax = x_f32.abs().amax(dim=-1, keepdim=True).clamp(min=6 * (2**-126))
    log2_ratio = (amax / 6.0).log2().ceil().clamp(-127.0, 127.0)
    scale = log2_ratio.exp2()
    scales = (log2_ratio + 127.0).to(torch.uint8)

    x_scaled = (x_f32 / scale).clamp(-6.0, 6.0)
    abs_x = x_scaled.abs()
    code = torch.zeros_like(abs_x, dtype=torch.int32)
    code = torch.where(abs_x > 0.25, 1, code)
    code = torch.where(abs_x >= 0.75, 2, code)
    code = torch.where(abs_x > 1.25, 3, code)
    code = torch.where(abs_x >= 1.75, 4, code)
    code = torch.where(abs_x > 2.5, 5, code)
    code = torch.where(abs_x >= 3.5, 6, code)
    code = torch.where(abs_x > 5.0, 7, code)
    sign = ((x_scaled.view(torch.int32) >> 31) & 1).to(torch.uint8)
    nibble = code.to(torch.uint8) | (sign << 3)

    nibble = nibble.reshape(-1, head_dim)
    packed = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).contiguous()
    packed = packed.reshape(*orig_shape[:-1], head_dim // 2)
    scales = scales.reshape(*orig_shape[:-1], n_blocks)
    return packed, scales


def _dequantize_mxfp4(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    table = FP4_VALUES.to(device=packed.device)
    bytes_i = packed.to(torch.int16)
    lo = bytes_i & 0xF
    hi = (bytes_i >> 4) & 0xF
    nibbles = torch.stack((lo, hi), dim=-1).flatten(-2).to(torch.long)
    mag = nibbles & 0x7
    sign = (nibbles & 0x8) != 0
    values = table[mag]
    values = torch.where(sign, -values, values)
    scale = (scales.to(torch.float32) - 127.0).exp2()
    scale = torch.repeat_interleave(scale, MXFP4_BLOCK_SIZE, dim=-1)
    return values * scale


def _kv_cache_cast_to_fp8(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / torch.finfo(current_platform.fp8_dtype()).max
    x_scaled = (x * (1.0 / sf)).to(current_platform.fp8_dtype())
    x_cast_back = x_scaled.float() * sf

    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(torch.uint8)
    x_fp8[:, block_size * head_dim :] = sf.view(num_blocks, block_size).view(
        torch.uint8
    )
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4), (
        x_cast_back.to(x.dtype)
    )


def _kv_cache_cast_to_fp4(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1 and head_dim == 128
    packed, scales = _quantize_to_mxfp4(x)
    x_cast_back = _dequantize_mxfp4(packed, scales).view_as(x).to(x.dtype)
    scales_i32 = scales.view(torch.int32).squeeze(-1)

    x_fp4 = torch.empty(
        (num_blocks, block_size * (head_dim // 2 + 4)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp4[:, : block_size * head_dim // 2] = packed.view(
        num_blocks, block_size * head_dim // 2
    )
    x_fp4[:, block_size * head_dim // 2 :] = scales_i32.view(torch.uint8).view(
        num_blocks, block_size * 4
    )
    return x_fp4.view(num_blocks, block_size, num_heads, head_dim // 2 + 4), (
        x_cast_back
    )


def _ref_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens_nextn: torch.Tensor,
    block_table: torch.Tensor,
    max_model_len: int,
) -> torch.Tensor:
    batch_size, next_n, _, _ = q.size()
    _, block_size, _, _ = kv_cache.size()
    logits = torch.full(
        [batch_size * next_n, max_model_len],
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_cpu = context_lens_nextn.cpu()
    for batch_idx in range(batch_size):
        for next_idx in range(next_n):
            row = batch_idx * next_n + next_idx
            context_len = int(context_lens_cpu[batch_idx, next_idx].item())
            qx = q[batch_idx, next_idx].float()
            weight = weights[row].float()
            for block_rk in range(cdiv(context_len, block_size)):
                block_idx = int(block_table[batch_idx, block_rk].item())
                kx = kv_cache[block_idx, :, 0].float()
                offsets = torch.arange(
                    block_rk * block_size,
                    (block_rk + 1) * block_size,
                    device=q.device,
                )
                valid = offsets < context_len
                score = torch.einsum("hd,kd->hk", qx, kx)
                score = torch.relu(score) * weight[:, None]
                score = score.sum(dim=0)
                logits[row, offsets] = torch.where(valid, score, float("-inf"))
    return logits


def _ref_fp4_mqa_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
) -> torch.Tensor:
    logits = torch.full(
        (q.shape[0], k.shape[0]),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    starts = cu_seqlen_ks.cpu().tolist()
    ends = cu_seqlen_ke.cpu().tolist()
    for row, (start, end) in enumerate(zip(starts, ends)):
        if end <= start:
            continue
        score = torch.einsum("hd,nd->hn", q[row].float(), k[start:end].float())
        score = torch.relu(score) * weights[row].float()[:, None]
        logits[row, start:end] = score.sum(dim=0)
    return logits


@torch.inference_mode()
def test_rocm_fp4_mqa_logits_matches_reference() -> None:
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp4_mqa_logits,
    )

    device = torch.device("cuda")
    torch.manual_seed(123)
    seq_len = 7
    seq_len_kv = 19
    num_heads = 8
    head_dim = 128
    q = (
        torch.randn(
            (seq_len, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    k = torch.randn((seq_len_kv, head_dim), device=device, dtype=torch.bfloat16)
    k = k * 0.125
    weights = torch.randn((seq_len, num_heads), device=device, dtype=torch.float32)
    cu_seqlen_ks = torch.tensor(
        [0, 0, 2, 4, 7, 9, 12], device=device, dtype=torch.int32
    )
    cu_seqlen_ke = torch.tensor(
        [3, 5, 8, 11, 13, 17, 19], device=device, dtype=torch.int32
    )

    q_packed, q_scales = _quantize_to_mxfp4(q)
    k_packed, k_scales = _quantize_to_mxfp4(k)
    q_dequant = _dequantize_mxfp4(q_packed, q_scales).view_as(q).bfloat16()
    k_dequant = _dequantize_mxfp4(k_packed, k_scales).view_as(k).bfloat16()

    actual = rocm_fp4_mqa_logits(
        (q_packed.view(torch.int8), q_scales.view(torch.int32).squeeze(-1)),
        (k_packed.view(torch.int8), k_scales.view(torch.int32).squeeze(-1)),
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=True,
    )
    expected = _ref_fp4_mqa_logits(
        q_dequant,
        k_dequant,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
    )

    finite = torch.isfinite(expected)
    diff = calc_diff(actual.masked_fill(~finite, 0), expected.masked_fill(~finite, 0))
    assert diff < 2e-2, f"non-paged fp4 MQA diff={float(diff):.6f}"
    torch.testing.assert_close(actual[~finite], expected[~finite])


@torch.inference_mode()
def test_rocm_fp4_mqa_logits_trivial_topk_skip_matches_full_topk() -> None:
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp4_mqa_logits,
    )

    device = torch.device("cuda")
    torch.manual_seed(234)
    seq_len = 9
    seq_len_kv = 160
    num_heads = 32
    head_dim = 128
    topk_tokens = 32
    q = (
        torch.randn(
            (seq_len, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    k = torch.randn((seq_len_kv, head_dim), device=device, dtype=torch.bfloat16) * 0.125
    weights = torch.randn((seq_len, num_heads), device=device, dtype=torch.float32)
    cu_seqlen_ks = torch.tensor(
        [0, 3, 11, 29, 37, 43, 61, 72, 88],
        device=device,
        dtype=torch.int32,
    )
    row_lens = torch.tensor(
        [0, 8, 31, 32, 33, 64, 72, 80, 72],
        device=device,
        dtype=torch.int32,
    )
    cu_seqlen_ke = cu_seqlen_ks + row_lens

    q_packed, q_scales = _quantize_to_mxfp4(q)
    k_packed, k_scales = _quantize_to_mxfp4(k)
    q_in = (q_packed.view(torch.int8), q_scales.view(torch.int32).squeeze(-1))
    k_in = (k_packed.view(torch.int8), k_scales.view(torch.int32).squeeze(-1))

    full_logits = rocm_fp4_mqa_logits(
        q_in,
        k_in,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
    )
    skipped_logits = rocm_fp4_mqa_logits(
        q_in,
        k_in,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
        trivial_topk_len=topk_tokens,
    )

    full_indices = torch.empty((seq_len, topk_tokens), dtype=torch.int32, device=device)
    skipped_indices = torch.empty_like(full_indices)
    torch.ops._C.top_k_per_row_prefill(
        full_logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        full_indices,
        full_logits.shape[0],
        full_logits.stride(0),
        full_logits.stride(1),
        topk_tokens,
    )
    torch.ops._C.top_k_per_row_prefill(
        skipped_logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        skipped_indices,
        skipped_logits.shape[0],
        skipped_logits.stride(0),
        skipped_logits.stride(1),
        topk_tokens,
    )

    starts_cpu = cu_seqlen_ks.cpu().tolist()
    row_lens_cpu = row_lens.cpu().tolist()
    for row, (start, row_len) in enumerate(zip(starts_cpu, row_lens_cpu)):
        selected = min(topk_tokens, row_len)
        assert torch.all(
            (skipped_indices[row] == -1) | (skipped_indices[row] < row_len)
        )
        if row_len <= topk_tokens:
            expected = torch.full(
                (topk_tokens,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            if row_len > 0:
                expected[:row_len] = torch.arange(
                    row_len, dtype=torch.int32, device=device
                )
            torch.testing.assert_close(skipped_indices[row], expected)
            continue

        full_positions = full_indices[row, :selected].clamp_min(0).to(torch.long)
        skipped_positions = skipped_indices[row, :selected].clamp_min(0).to(torch.long)
        full_values = full_logits[row, start + full_positions]
        skipped_values = full_logits[row, start + skipped_positions]
        torch.testing.assert_close(
            torch.sort(skipped_values).values,
            torch.sort(full_values).values,
            rtol=1e-5,
            atol=1e-5,
        )


@torch.inference_mode()
def test_rocm_fp4_mqa_logits_persistent_env_matches_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp4_mqa_logits,
    )

    device = torch.device("cuda")
    torch.manual_seed(235)
    seq_len = 10
    seq_len_kv = 192
    num_heads = 32
    head_dim = 128
    topk_tokens = 32
    q = (
        torch.randn(
            (seq_len, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    k = torch.randn((seq_len_kv, head_dim), device=device, dtype=torch.bfloat16) * 0.125
    weights = torch.randn((seq_len, num_heads), device=device, dtype=torch.float32)
    cu_seqlen_ks = torch.tensor(
        [0, 3, 11, 29, 37, 43, 61, 72, 88, 96],
        device=device,
        dtype=torch.int32,
    )
    row_lens = torch.tensor(
        [0, 8, 31, 32, 33, 64, 72, 80, 72, 96],
        device=device,
        dtype=torch.int32,
    )
    cu_seqlen_ke = cu_seqlen_ks + row_lens

    q_packed, q_scales = _quantize_to_mxfp4(q)
    k_packed, k_scales = _quantize_to_mxfp4(k)
    q_in = (q_packed.view(torch.int8), q_scales.view(torch.int32).squeeze(-1))
    k_in = (k_packed.view(torch.int8), k_scales.view(torch.int32).squeeze(-1))

    monkeypatch.delenv(
        "VLLM_ROCM_FP4_INDEXER_PREFILL_PERSISTENT_K_PROGS", raising=False
    )
    default_logits = rocm_fp4_mqa_logits(
        q_in,
        k_in,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
        trivial_topk_len=topk_tokens,
    )
    monkeypatch.setenv("VLLM_ROCM_FP4_INDEXER_PREFILL_PERSISTENT_K_PROGS", "4")
    persistent_logits = rocm_fp4_mqa_logits(
        q_in,
        k_in,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=False,
        trivial_topk_len=topk_tokens,
    )

    positions = torch.arange(seq_len_kv, device=device)[None, :]
    starts = cu_seqlen_ks[:, None]
    ends = cu_seqlen_ke[:, None]
    computed_rows = row_lens[:, None] > topk_tokens
    valid_mask = computed_rows & (positions >= starts) & (positions < ends)
    torch.testing.assert_close(
        persistent_logits.masked_fill(~valid_mask, 0),
        default_logits.masked_fill(~valid_mask, 0),
    )

    default_indices = torch.empty(
        (seq_len, topk_tokens), dtype=torch.int32, device=device
    )
    persistent_indices = torch.empty_like(default_indices)
    torch.ops._C.top_k_per_row_prefill(
        default_logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        default_indices,
        default_logits.shape[0],
        default_logits.stride(0),
        default_logits.stride(1),
        topk_tokens,
    )
    torch.ops._C.top_k_per_row_prefill(
        persistent_logits,
        cu_seqlen_ks,
        cu_seqlen_ke,
        persistent_indices,
        persistent_logits.shape[0],
        persistent_logits.stride(0),
        persistent_logits.stride(1),
        topk_tokens,
    )
    torch.testing.assert_close(persistent_indices, default_indices)


@torch.inference_mode()
def test_cp_gather_indexer_mxfp4_cache_triton_matches_compressor_layout() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        cp_gather_indexer_mxfp4_cache_triton,
    )

    device = torch.device("cuda")
    num_blocks = 3
    block_size = 4
    head_dim = 128
    head_bytes = head_dim // 2
    scale_bytes = head_dim // MXFP4_BLOCK_SIZE
    allocated_width = head_dim + 4
    k_cache = torch.zeros(
        (num_blocks, block_size, allocated_width),
        device=device,
        dtype=torch.uint8,
    )
    cache_flat = k_cache.flatten()

    value_rows = {}
    scale_rows = {}
    for block in range(num_blocks):
        block_base = block * k_cache.stride(0)
        for pos in range(block_size):
            value = (
                torch.arange(head_bytes, device=device, dtype=torch.uint8)
                + block * 17
                + pos * 3
            )
            scale = torch.tensor(
                [127 + block, 128 + pos, 129 + block + pos, 130 - pos],
                device=device,
                dtype=torch.uint8,
            )
            value_offset = block_base + pos * head_bytes
            scale_offset = block_base + block_size * head_bytes + pos * scale_bytes
            cache_flat[value_offset : value_offset + head_bytes] = value
            cache_flat[scale_offset : scale_offset + scale_bytes] = scale
            value_rows[(block, pos)] = value
            scale_rows[(block, pos)] = scale

    block_table = torch.tensor([[2, 0], [1, 0]], device=device, dtype=torch.int32)
    cu_seqlen = torch.tensor([0, 5, 8], device=device, dtype=torch.int32)
    token_to_seq = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1], device=device)
    token_to_seq = token_to_seq.to(torch.int32)
    k_fp4 = torch.empty((8, head_bytes), device=device, dtype=torch.uint8)
    k_scale = torch.empty((8, scale_bytes), device=device, dtype=torch.uint8)

    cp_gather_indexer_mxfp4_cache_triton(
        k_cache,
        k_fp4,
        k_scale,
        block_table,
        cu_seqlen,
        token_to_seq,
    )

    expected_values = []
    expected_scales = []
    for token in range(8):
        req = int(token_to_seq[token].item())
        local = token - int(cu_seqlen[req].item())
        block = int(block_table[req, local // block_size].item())
        pos = local % block_size
        expected_values.append(value_rows[(block, pos)])
        expected_scales.append(scale_rows[(block, pos)])
    torch.testing.assert_close(k_fp4, torch.stack(expected_values))
    torch.testing.assert_close(k_scale, torch.stack(expected_scales))


@pytest.mark.parametrize(
    "case",
    DEEPGEMM_PAGED_MQA_CASES,
    ids=[case.id() for case in DEEPGEMM_PAGED_MQA_CASES],
)
@torch.inference_mode()
def test_rocm_fp8_fp4_paged_mqa_logits_deepgemm_cases(case: PagedMQACase) -> None:
    device = torch.device("cuda")
    seed = DEEPGEMM_PAGED_MQA_CASES.index(case)
    torch.manual_seed(seed)
    random.seed(seed)

    raw_batch_size, avg_kv, max_model_len = _scaled_case_dims(case)
    raw_next_n = case.next_n
    if case.is_varlen:
        tokens_per_seq = torch.randint(
            1,
            case.max_tokens_per_batch + 1,
            (raw_batch_size,),
            device=device,
            dtype=torch.int32,
        )
        batch_size = int(tokens_per_seq.sum().item())
        next_n = 1
    else:
        tokens_per_seq = None
        batch_size = raw_batch_size
        next_n = raw_next_n

    q = torch.randn(
        (batch_size, next_n, case.num_heads, case.head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    q = q * 0.125
    weights = torch.randn(
        (batch_size * next_n, case.num_heads),
        device=device,
        dtype=torch.float32,
    )

    low = max(1, int(0.7 * avg_kv))
    high = max(low + 1, int(1.3 * avg_kv))
    context_lens = torch.randint(
        low,
        high,
        (raw_batch_size,),
        device=device,
        dtype=torch.int32,
    )
    context_lens.clamp_(max=max_model_len)
    if case.is_varlen:
        assert tokens_per_seq is not None
        max_ctx_len_per_seq = (context_lens + tokens_per_seq - 1).clamp(
            max=max_model_len
        )
    else:
        max_ctx_len_per_seq = context_lens

    num_blocks_per_query = torch.ceil(max_ctx_len_per_seq.float() / case.block_kv).to(
        torch.int32
    )
    total_used_blocks = int(num_blocks_per_query.sum().item())
    num_total_blocks = total_used_blocks + 8
    kv_cache = torch.randn(
        (num_total_blocks, case.block_kv, 1, case.head_dim),
        device=device,
        dtype=torch.bfloat16,
    )
    kv_cache = kv_cache * 0.125

    block_table = torch.empty(
        (raw_batch_size, int(num_blocks_per_query.max().item())),
        device=device,
        dtype=torch.int32,
    )
    block_idx_pool = torch.randperm(num_total_blocks, device=device, dtype=torch.int32)
    offset = 0
    for i, num_blocks in enumerate(num_blocks_per_query.tolist()):
        block_table[i, :num_blocks] = block_idx_pool[offset : offset + num_blocks]
        offset += num_blocks

    if case.is_varlen:
        assert tokens_per_seq is not None
        context_lens = context_lens.repeat_interleave(tokens_per_seq)
        offsets_within_seq = torch.cat(
            [
                torch.arange(int(n.item()), device=device, dtype=torch.int32)
                for n in tokens_per_seq
            ]
        )
        context_lens = (context_lens + offsets_within_seq).clamp(max=max_model_len)
        block_table = block_table.repeat_interleave(tokens_per_seq, dim=0)

    if case.is_varlen:
        context_lens_nextn = context_lens.view(-1, 1)
    else:
        rand = torch.rand(batch_size, next_n, device=device)
        context_lens_nextn = ((context_lens.unsqueeze(1) + 1) * rand).int()
        context_lens_nextn[:, -1] = context_lens
        context_lens_nextn.clamp_(min=1, max=max_model_len)
    context_lens_nextn = context_lens_nextn.contiguous().to(torch.int32)

    if case.is_fp4:
        q_packed, q_scales_u8 = _quantize_to_mxfp4(q)
        q_in = (
            q_packed.view(batch_size, next_n, case.num_heads, case.head_dim // 2).view(
                torch.int8
            ),
            q_scales_u8.view(torch.int32).squeeze(-1),
        )
        q_simulated = (
            _dequantize_mxfp4(q_packed, q_scales_u8).view_as(q).to(torch.bfloat16)
        )
        kv_in, kv_simulated = _kv_cache_cast_to_fp4(kv_cache)
    else:
        q_in = (q.to(current_platform.fp8_dtype()), None)
        q_simulated = q_in[0].to(torch.bfloat16)
        kv_in, kv_simulated = _kv_cache_cast_to_fp8(kv_cache)

    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens_nextn,
        case.block_kv,
        get_num_sms(),
    )
    assert schedule_metadata.shape == (get_num_sms() + 1, 2)
    assert schedule_metadata.dtype == torch.int32

    logits = fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens_nextn,
        block_table,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
        logits_dtype=case.logits_dtype,
    )
    assert logits.dtype == case.logits_dtype

    ref_logits = _ref_paged_mqa_logits(
        q_simulated,
        kv_simulated,
        weights,
        context_lens_nextn,
        block_table,
        max_model_len,
    )
    positions = torch.arange(max_model_len, device=device).unsqueeze(0)
    invalid = positions >= context_lens_nextn.view(-1, 1)
    actual = logits.float().masked_fill(invalid, 0)
    expected = ref_logits.masked_fill(invalid, 0)

    diff = calc_diff(actual, expected)
    tolerance = 2e-2 if case.is_fp4 or case.logits_dtype is torch.bfloat16 else 1e-3
    assert diff < tolerance, f"{case.id()} diff={float(diff):.6f}"


@pytest.mark.parametrize("seq_lens_2d", [False, True])
@torch.inference_mode()
def test_rocm_fp4_paged_mqa_logits_topk_chunked_matches_materialized(
    seq_lens_2d: bool,
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _fp4_decode_valid_lens_2d,
        _rocm_fp4_paged_mqa_logits_topk_chunked,
    )

    device = torch.device("cuda")
    torch.manual_seed(123)
    batch_size = 3
    next_n = 2
    num_heads = 32
    head_dim = 128
    block_size = 32
    max_model_len = 192
    topk_tokens = 32

    q = (
        torch.randn(
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    weights = torch.randn(
        (batch_size * next_n, num_heads), device=device, dtype=torch.float32
    )
    context_1d = torch.tensor([70, 133, 181], device=device, dtype=torch.int32)
    if seq_lens_2d:
        context_lens = torch.tensor(
            [[65, 70], [117, 133], [151, 181]],
            device=device,
            dtype=torch.int32,
        )
        max_ctx_per_seq = context_lens.max(dim=1).values
    else:
        context_lens = context_1d
        max_ctx_per_seq = context_lens

    num_blocks_per_query = torch.ceil(max_ctx_per_seq.float() / block_size).to(
        torch.int32
    )
    total_blocks = int(num_blocks_per_query.sum().item()) + 4
    kv_cache = (
        torch.randn(
            (total_blocks, block_size, 1, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    block_table = torch.empty(
        (batch_size, int(num_blocks_per_query.max().item())),
        device=device,
        dtype=torch.int32,
    )
    block_pool = torch.randperm(total_blocks, device=device, dtype=torch.int32)
    offset = 0
    for i, num_blocks in enumerate(num_blocks_per_query.tolist()):
        block_table[i, :num_blocks] = block_pool[offset : offset + num_blocks]
        offset += num_blocks

    q_packed, q_scales_u8 = _quantize_to_mxfp4(q)
    q_in = (
        q_packed.view(batch_size, next_n, num_heads, head_dim // 2).view(torch.int8),
        q_scales_u8.view(torch.int32).squeeze(-1),
    )
    kv_in, _ = _kv_cache_cast_to_fp4(kv_cache)
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms()
    )

    logits = fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
    )
    ref_indices = torch.empty(
        (batch_size * next_n, topk_tokens), dtype=torch.int32, device=device
    )
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        context_lens,
        ref_indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )

    actual_indices = torch.empty_like(ref_indices)
    _rocm_fp4_paged_mqa_logits_topk_chunked(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        actual_indices,
        max_model_len,
        topk_tokens,
    )

    safe_ref = ref_indices.clamp_min(0).to(torch.long)
    safe_actual = actual_indices.clamp_min(0).to(torch.long)
    ref_values = torch.gather(logits, 1, safe_ref).masked_fill(
        ref_indices < 0, float("-inf")
    )
    actual_values = torch.gather(logits, 1, safe_actual).masked_fill(
        actual_indices < 0, float("-inf")
    )
    row_valid_lens = _fp4_decode_valid_lens_2d(
        context_lens, next_n, max_model_len
    ).reshape(-1)
    for row in range(batch_size * next_n):
        valid_len = int(row_valid_lens[row].item())
        assert torch.all(
            (actual_indices[row] == -1) | (actual_indices[row] < valid_len)
        )
        selected = min(topk_tokens, valid_len)
        torch.testing.assert_close(
            torch.sort(actual_values[row, :selected]).values,
            torch.sort(ref_values[row, :selected]).values,
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("seq_lens_2d", [False, True])
@torch.inference_mode()
def test_rocm_fp4_paged_mqa_logits_streaming_topk_matches_materialized(
    seq_lens_2d: bool,
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _fp4_decode_valid_lens_2d,
    )
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp4_paged_mqa_logits_topk,
    )

    device = torch.device("cuda")
    torch.manual_seed(345)
    batch_size = 3
    next_n = 2
    num_heads = 32
    head_dim = 128
    block_size = 32
    max_model_len = 192
    topk_tokens = 32

    q = (
        torch.randn(
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    weights = torch.randn(
        (batch_size * next_n, num_heads), device=device, dtype=torch.float32
    )
    if seq_lens_2d:
        context_lens = torch.tensor(
            [[17, 31], [65, 129], [151, 181]],
            device=device,
            dtype=torch.int32,
        )
        max_ctx_per_seq = context_lens.max(dim=1).values
    else:
        context_lens = torch.tensor([31, 129, 181], device=device, dtype=torch.int32)
        max_ctx_per_seq = context_lens

    num_blocks_per_query = torch.ceil(max_ctx_per_seq.float() / block_size).to(
        torch.int32
    )
    total_blocks = int(num_blocks_per_query.sum().item()) + 4
    kv_cache = (
        torch.randn(
            (total_blocks, block_size, 1, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    block_table = torch.empty(
        (batch_size, int(num_blocks_per_query.max().item())),
        device=device,
        dtype=torch.int32,
    )
    block_pool = torch.randperm(total_blocks, device=device, dtype=torch.int32)
    offset = 0
    for i, num_blocks in enumerate(num_blocks_per_query.tolist()):
        block_table[i, :num_blocks] = block_pool[offset : offset + num_blocks]
        offset += num_blocks

    q_packed, q_scales_u8 = _quantize_to_mxfp4(q)
    q_in = (
        q_packed.view(batch_size, next_n, num_heads, head_dim // 2).view(torch.int8),
        q_scales_u8.view(torch.int32).squeeze(-1),
    )
    kv_in, _ = _kv_cache_cast_to_fp4(kv_cache)
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms()
    )

    logits = fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
    )
    ref_indices = torch.empty(
        (batch_size * next_n, topk_tokens), dtype=torch.int32, device=device
    )
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        context_lens,
        ref_indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )

    actual_indices = torch.empty_like(ref_indices)
    rocm_fp4_paged_mqa_logits_topk(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        actual_indices,
        max_context_len=max_model_len,
        topk_tokens=topk_tokens,
    )

    safe_ref = ref_indices.clamp_min(0).to(torch.long)
    safe_actual = actual_indices.clamp_min(0).to(torch.long)
    ref_values = torch.gather(logits, 1, safe_ref).masked_fill(
        ref_indices < 0, float("-inf")
    )
    actual_values = torch.gather(logits, 1, safe_actual).masked_fill(
        actual_indices < 0, float("-inf")
    )
    row_valid_lens = _fp4_decode_valid_lens_2d(
        context_lens, next_n, max_model_len
    ).reshape(-1)
    for row in range(batch_size * next_n):
        valid_len = int(row_valid_lens[row].item())
        assert torch.all(
            (actual_indices[row] == -1) | (actual_indices[row] < valid_len)
        )
        selected = min(topk_tokens, valid_len)
        if valid_len <= topk_tokens:
            expected = torch.full(
                (topk_tokens,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            if valid_len > 0:
                expected[:valid_len] = torch.arange(
                    valid_len, dtype=torch.int32, device=device
                )
            torch.testing.assert_close(actual_indices[row], expected)
            continue

        torch.testing.assert_close(
            torch.sort(actual_values[row, :selected]).values,
            torch.sort(ref_values[row, :selected]).values,
            rtol=1e-5,
            atol=1e-5,
        )


@pytest.mark.parametrize("seq_lens_2d", [False, True])
@torch.inference_mode()
def test_rocm_fp4_paged_mqa_logits_tile_topk_matches_materialized(
    seq_lens_2d: bool,
) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _fp4_decode_valid_lens_2d,
    )
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp4_paged_mqa_logits_tile_topk,
    )

    device = torch.device("cuda")
    torch.manual_seed(456)
    batch_size = 3
    next_n = 2
    num_heads = 32
    head_dim = 128
    block_size = 32
    max_model_len = 192
    topk_tokens = 32

    q = (
        torch.randn(
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    weights = torch.randn(
        (batch_size * next_n, num_heads), device=device, dtype=torch.float32
    )
    if seq_lens_2d:
        context_lens = torch.tensor(
            [[17, 31], [65, 129], [151, 181]],
            device=device,
            dtype=torch.int32,
        )
        max_ctx_per_seq = context_lens.max(dim=1).values
    else:
        context_lens = torch.tensor([31, 129, 181], device=device, dtype=torch.int32)
        max_ctx_per_seq = context_lens

    num_blocks_per_query = torch.ceil(max_ctx_per_seq.float() / block_size).to(
        torch.int32
    )
    total_blocks = int(num_blocks_per_query.sum().item()) + 4
    kv_cache = (
        torch.randn(
            (total_blocks, block_size, 1, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    block_table = torch.empty(
        (batch_size, int(num_blocks_per_query.max().item())),
        device=device,
        dtype=torch.int32,
    )
    block_pool = torch.randperm(total_blocks, device=device, dtype=torch.int32)
    offset = 0
    for i, num_blocks in enumerate(num_blocks_per_query.tolist()):
        block_table[i, :num_blocks] = block_pool[offset : offset + num_blocks]
        offset += num_blocks

    q_packed, q_scales_u8 = _quantize_to_mxfp4(q)
    q_in = (
        q_packed.view(batch_size, next_n, num_heads, head_dim // 2).view(torch.int8),
        q_scales_u8.view(torch.int32).squeeze(-1),
    )
    kv_in, _ = _kv_cache_cast_to_fp4(kv_cache)
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms()
    )

    logits = fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_model_len,
        clean_logits=False,
    )
    ref_indices = torch.empty(
        (batch_size * next_n, topk_tokens), dtype=torch.int32, device=device
    )
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        context_lens,
        ref_indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )

    actual_indices = torch.empty_like(ref_indices)
    rocm_fp4_paged_mqa_logits_tile_topk(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        actual_indices,
        max_context_len=max_model_len,
        topk_tokens=topk_tokens,
    )

    safe_ref = ref_indices.clamp_min(0).to(torch.long)
    safe_actual = actual_indices.clamp_min(0).to(torch.long)
    ref_values = torch.gather(logits, 1, safe_ref).masked_fill(
        ref_indices < 0, float("-inf")
    )
    actual_values = torch.gather(logits, 1, safe_actual).masked_fill(
        actual_indices < 0, float("-inf")
    )
    row_valid_lens = _fp4_decode_valid_lens_2d(
        context_lens, next_n, max_model_len
    ).reshape(-1)
    for row in range(batch_size * next_n):
        valid_len = int(row_valid_lens[row].item())
        assert torch.all(
            (actual_indices[row] == -1) | (actual_indices[row] < valid_len)
        )
        selected = min(topk_tokens, valid_len)
        if valid_len <= topk_tokens:
            expected = torch.full(
                (topk_tokens,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            if valid_len > 0:
                expected[:valid_len] = torch.arange(
                    valid_len, dtype=torch.int32, device=device
                )
            torch.testing.assert_close(actual_indices[row], expected)
            continue

        torch.testing.assert_close(
            torch.sort(actual_values[row, :selected]).values,
            torch.sort(ref_values[row, :selected]).values,
            rtol=1e-5,
            atol=1e-5,
        )


def test_rocm_fp4_paged_mqa_logits_streaming_topk_rejects_large_topk() -> None:
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp4_paged_mqa_logits_topk,
    )

    batch_size = 1
    next_n = 1
    num_heads = 32
    head_dim = 128
    block_size = 32
    topk_tokens = 1024
    q_values = torch.empty(
        (batch_size, next_n, num_heads, head_dim // 2), dtype=torch.int8
    )
    q_scales = torch.empty((batch_size, next_n, num_heads), dtype=torch.int32)
    kv_cache = torch.empty((1, block_size, 1, head_dim // 2 + 4), dtype=torch.uint8)
    weights = torch.empty((batch_size * next_n, num_heads), dtype=torch.float32)
    context_lens = torch.ones((batch_size, next_n), dtype=torch.int32)
    block_tables = torch.zeros((batch_size, 1), dtype=torch.int32)
    topk_indices = torch.empty((batch_size * next_n, topk_tokens), dtype=torch.int32)

    with pytest.raises(ValueError, match="validation-only"):
        rocm_fp4_paged_mqa_logits_topk(
            (q_values, q_scales),
            kv_cache,
            weights,
            context_lens,
            block_tables,
            topk_indices,
            max_context_len=topk_tokens,
            topk_tokens=topk_tokens,
        )


@torch.inference_mode()
def test_rocm_fp4_decode_topk_logits_view_matches_full_width() -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _fp4_decode_topk_logits_view,
    )

    device = torch.device("cuda")
    torch.manual_seed(321)
    batch_size = 64
    next_n = 1
    max_decode_seq_len = 192
    logits_width = 512
    topk_tokens = 32
    logits = torch.randn(
        (batch_size * next_n, logits_width), device=device, dtype=torch.float32
    )
    seq_lens = torch.randint(
        max_decode_seq_len - 32,
        max_decode_seq_len + 1,
        (batch_size, next_n),
        device=device,
        dtype=torch.int32,
    )
    seq_lens[:, -1] = max_decode_seq_len

    full_indices = torch.empty(
        (batch_size * next_n, topk_tokens), dtype=torch.int32, device=device
    )
    torch.ops._C.top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        full_indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        topk_tokens,
    )

    topk_logits = _fp4_decode_topk_logits_view(
        logits,
        topk_tokens,
        max_decode_seq_len,
    )
    view_indices = torch.empty_like(full_indices)
    torch.ops._C.top_k_per_row_decode(
        topk_logits,
        next_n,
        seq_lens,
        view_indices,
        topk_logits.shape[0],
        topk_logits.stride(0),
        topk_logits.stride(1),
        topk_tokens,
    )

    safe_full = full_indices.clamp_min(0).long()
    safe_view = view_indices.clamp_min(0).long()
    full_values = torch.gather(logits, 1, safe_full).masked_fill(
        full_indices < 0, float("-inf")
    )
    view_values = torch.gather(logits, 1, safe_view).masked_fill(
        view_indices < 0, float("-inf")
    )
    row_valid_lens = seq_lens.reshape(-1)
    for row in range(batch_size * next_n):
        selected = min(topk_tokens, int(row_valid_lens[row].item()))
        torch.testing.assert_close(
            torch.sort(view_values[row, :selected]).values,
            torch.sort(full_values[row, :selected]).values,
            rtol=1e-5,
            atol=1e-5,
        )


def test_rocm_fp4_decode_logits_width_env_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _fp4_decode_logits_width,
    )

    monkeypatch.delenv("VLLM_ROCM_FP4_INDEXER_DECODE_WIDTH_CAP", raising=False)
    assert (
        _fp4_decode_logits_width(
            active_paged_width=65536,
            topk_tokens=1024,
            max_decode_seq_len=8192,
            max_model_len=1048576,
        )
        == 65536
    )

    monkeypatch.setenv("VLLM_ROCM_FP4_INDEXER_DECODE_WIDTH_CAP", "1")
    assert (
        _fp4_decode_logits_width(
            active_paged_width=65536,
            topk_tokens=1024,
            max_decode_seq_len=8192,
            max_model_len=1048576,
        )
        == 8192
    )
    assert (
        _fp4_decode_logits_width(
            active_paged_width=65536,
            topk_tokens=2048,
            max_decode_seq_len=1536,
            max_model_len=1048576,
        )
        == 2048
    )
    monkeypatch.setenv("VLLM_ROCM_FP4_INDEXER_DECODE_WIDTH_CAP", "0")
    assert (
        _fp4_decode_logits_width(
            active_paged_width=65536,
            topk_tokens=1024,
            max_decode_seq_len=8192,
            max_model_len=1048576,
        )
        == 65536
    )


def test_rocm_fp4_fused_topk_env_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _should_use_fp4_fused_topk,
        _should_use_fp4_tile_topk,
    )

    monkeypatch.delenv("VLLM_ROCM_FP4_INDEXER_FUSED_TOPK", raising=False)
    monkeypatch.delenv("VLLM_ROCM_FP4_INDEXER_TILE_TOPK", raising=False)
    assert not _should_use_fp4_fused_topk(
        logits_width=65536, topk_tokens=1024, num_rows=64
    )
    assert not _should_use_fp4_fused_topk(
        logits_width=65536, topk_tokens=1024, num_rows=96
    )

    monkeypatch.setenv("VLLM_ROCM_FP4_INDEXER_FUSED_TOPK", "0")
    assert not _should_use_fp4_fused_topk(
        logits_width=65536, topk_tokens=1024, num_rows=96
    )
    monkeypatch.setenv("VLLM_ROCM_FP4_INDEXER_FUSED_TOPK", "1")
    assert _should_use_fp4_fused_topk(logits_width=8192, topk_tokens=2048, num_rows=16)
    assert not _should_use_fp4_tile_topk(logits_width=8192, topk_tokens=1024)
    monkeypatch.setenv("VLLM_ROCM_FP4_INDEXER_TILE_TOPK", "1")
    assert _should_use_fp4_tile_topk(logits_width=8192, topk_tokens=1024)
    assert not _should_use_fp4_tile_topk(logits_width=16384, topk_tokens=1024)


@torch.inference_mode()
def test_rocm_fp4_paged_mqa_logits_trivial_decode_topk_skip_matches_full_topk() -> None:
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp8_fp4_paged_mqa_logits,
    )

    device = torch.device("cuda")
    torch.manual_seed(432)
    batch_size = 4
    next_n = 2
    num_heads = 32
    head_dim = 128
    block_size = 32
    max_model_len = 128
    topk_tokens = 32

    q = (
        torch.randn(
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    weights = torch.randn(
        (batch_size * next_n, num_heads), device=device, dtype=torch.float32
    )
    context_lens = torch.tensor(
        [[7, 32], [33, 64], [1, 48], [80, 96]],
        device=device,
        dtype=torch.int32,
    )
    num_blocks = cdiv(max_model_len, block_size)
    kv_cache = (
        torch.randn(
            (num_blocks, block_size, 1, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).repeat(
        batch_size, 1
    )

    q_packed, q_scales_u8 = _quantize_to_mxfp4(q)
    q_in = (
        q_packed.view(batch_size, next_n, num_heads, head_dim // 2).view(torch.int8),
        q_scales_u8.view(torch.int32).squeeze(-1),
    )
    kv_in, _ = _kv_cache_cast_to_fp4(kv_cache)
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms()
    )

    full_logits = rocm_fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_model_len,
        clean_logits=False,
    )
    skipped_logits = rocm_fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_model_len,
        clean_logits=False,
        trivial_topk_len=topk_tokens,
    )

    full_indices = torch.empty(
        (batch_size * next_n, topk_tokens), dtype=torch.int32, device=device
    )
    skipped_indices = torch.empty_like(full_indices)
    torch.ops._C.top_k_per_row_decode(
        full_logits,
        next_n,
        context_lens,
        full_indices,
        full_logits.shape[0],
        full_logits.stride(0),
        full_logits.stride(1),
        topk_tokens,
    )
    torch.ops._C.top_k_per_row_decode(
        skipped_logits,
        next_n,
        context_lens,
        skipped_indices,
        skipped_logits.shape[0],
        skipped_logits.stride(0),
        skipped_logits.stride(1),
        topk_tokens,
    )

    row_valid_lens = context_lens.reshape(-1).cpu().tolist()
    for row, valid_len in enumerate(row_valid_lens):
        selected = min(topk_tokens, valid_len)
        assert torch.all(
            (skipped_indices[row] == -1) | (skipped_indices[row] < valid_len)
        )
        if valid_len <= topk_tokens:
            expected = torch.full(
                (topk_tokens,),
                -1,
                dtype=torch.int32,
                device=device,
            )
            if valid_len > 0:
                expected[:valid_len] = torch.arange(
                    valid_len, dtype=torch.int32, device=device
                )
            torch.testing.assert_close(skipped_indices[row], expected)
            continue

        full_positions = full_indices[row, :selected].clamp_min(0).to(torch.long)
        skipped_positions = skipped_indices[row, :selected].clamp_min(0).to(torch.long)
        full_values = full_logits[row, full_positions]
        skipped_values = full_logits[row, skipped_positions]
        torch.testing.assert_close(
            torch.sort(skipped_values).values,
            torch.sort(full_values).values,
            rtol=1e-5,
            atol=1e-5,
        )


@torch.inference_mode()
def test_rocm_fp4_paged_mqa_logits_active_launch_width_matches_full_launch() -> None:
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp8_fp4_paged_mqa_logits,
    )

    device = torch.device("cuda")
    torch.manual_seed(654)
    batch_size = 4
    next_n = 2
    num_heads = 32
    head_dim = 128
    block_size = 32
    max_model_len = 128
    launch_context_len = 96
    topk_tokens = 32

    q = (
        torch.randn(
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    weights = torch.randn(
        (batch_size * next_n, num_heads), device=device, dtype=torch.float32
    )
    context_lens = torch.tensor(
        [[7, 32], [33, 64], [1, 48], [80, 96]],
        device=device,
        dtype=torch.int32,
    )
    num_blocks = cdiv(max_model_len, block_size)
    kv_cache = (
        torch.randn(
            (num_blocks, block_size, 1, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).repeat(
        batch_size, 1
    )

    q_packed, q_scales_u8 = _quantize_to_mxfp4(q)
    q_in = (
        q_packed.view(batch_size, next_n, num_heads, head_dim // 2).view(torch.int8),
        q_scales_u8.view(torch.int32).squeeze(-1),
    )
    kv_in, _ = _kv_cache_cast_to_fp4(kv_cache)
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms()
    )

    full_logits = rocm_fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_model_len,
        clean_logits=False,
    )
    active_launch_logits = rocm_fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_model_len,
        clean_logits=False,
        launch_context_len=launch_context_len,
    )
    active_prefix = active_launch_logits[:, :launch_context_len]
    full_prefix = full_logits[:, :launch_context_len]
    positions = torch.arange(launch_context_len, device=device)[None, :]
    valid_mask = positions < context_lens.reshape(-1, 1)
    torch.testing.assert_close(
        active_prefix.masked_fill(~valid_mask, 0),
        full_prefix.masked_fill(~valid_mask, 0),
    )

    full_indices = torch.empty(
        (batch_size * next_n, topk_tokens), dtype=torch.int32, device=device
    )
    active_indices = torch.empty_like(full_indices)
    full_topk_logits = full_logits[:, :launch_context_len]
    active_topk_logits = active_launch_logits[:, :launch_context_len]
    torch.ops._C.top_k_per_row_decode(
        full_topk_logits,
        next_n,
        context_lens,
        full_indices,
        full_topk_logits.shape[0],
        full_topk_logits.stride(0),
        full_topk_logits.stride(1),
        topk_tokens,
    )
    torch.ops._C.top_k_per_row_decode(
        active_topk_logits,
        next_n,
        context_lens,
        active_indices,
        active_topk_logits.shape[0],
        active_topk_logits.stride(0),
        active_topk_logits.stride(1),
        topk_tokens,
    )
    torch.testing.assert_close(active_indices, full_indices)


@torch.inference_mode()
def test_rocm_fp4_paged_mqa_logits_segment_env_matches_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp8_fp4_paged_mqa_logits,
    )

    device = torch.device("cuda")
    torch.manual_seed(987)
    batch_size = 3
    next_n = 2
    num_heads = 32
    head_dim = 128
    block_size = 32
    max_model_len = 192
    topk_tokens = 32

    q = (
        torch.randn(
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    weights = torch.randn(
        (batch_size * next_n, num_heads), device=device, dtype=torch.float32
    )
    context_lens = torch.tensor(
        [[64, 96], [129, 160], [33, 192]],
        device=device,
        dtype=torch.int32,
    )
    num_blocks = cdiv(max_model_len, block_size)
    kv_cache = (
        torch.randn(
            (num_blocks, block_size, 1, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).repeat(
        batch_size, 1
    )

    q_packed, q_scales_u8 = _quantize_to_mxfp4(q)
    q_in = (
        q_packed.view(batch_size, next_n, num_heads, head_dim // 2).view(torch.int8),
        q_scales_u8.view(torch.int32).squeeze(-1),
    )
    kv_in, _ = _kv_cache_cast_to_fp4(kv_cache)
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms()
    )

    default_logits = rocm_fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_model_len,
        clean_logits=False,
        trivial_topk_len=topk_tokens,
    )
    monkeypatch.setenv("VLLM_ROCM_FP4_INDEXER_SEGMENT_GROUP_PROGS", "2")
    segment_logits = rocm_fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_model_len,
        clean_logits=False,
        trivial_topk_len=topk_tokens,
    )

    positions = torch.arange(max_model_len, device=device)[None, :]
    valid_mask = positions < context_lens.reshape(-1, 1)
    torch.testing.assert_close(
        segment_logits.masked_fill(~valid_mask, 0),
        default_logits.masked_fill(~valid_mask, 0),
    )

    default_indices = torch.empty(
        (batch_size * next_n, topk_tokens), dtype=torch.int32, device=device
    )
    segment_indices = torch.empty_like(default_indices)
    torch.ops._C.top_k_per_row_decode(
        default_logits,
        next_n,
        context_lens,
        default_indices,
        default_logits.shape[0],
        default_logits.stride(0),
        default_logits.stride(1),
        topk_tokens,
    )
    torch.ops._C.top_k_per_row_decode(
        segment_logits,
        next_n,
        context_lens,
        segment_indices,
        segment_logits.shape[0],
        segment_logits.stride(0),
        segment_logits.stride(1),
        topk_tokens,
    )
    torch.testing.assert_close(segment_indices, default_indices)


@torch.inference_mode()
def test_rocm_fp4_paged_mqa_logits_tile_chunks_env_matches_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.v1.attention.ops.rocm_fp8_fp4_paged_mqa_logits import (
        rocm_fp8_fp4_paged_mqa_logits,
    )

    device = torch.device("cuda")
    torch.manual_seed(988)
    batch_size = 3
    next_n = 2
    num_heads = 32
    head_dim = 128
    block_size = 32
    max_model_len = 192
    topk_tokens = 32

    q = (
        torch.randn(
            (batch_size, next_n, num_heads, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    weights = torch.randn(
        (batch_size * next_n, num_heads), device=device, dtype=torch.float32
    )
    context_lens = torch.tensor(
        [[64, 96], [129, 160], [33, 192]],
        device=device,
        dtype=torch.int32,
    )
    num_blocks = cdiv(max_model_len, block_size)
    kv_cache = (
        torch.randn(
            (num_blocks, block_size, 1, head_dim),
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.125
    )
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).repeat(
        batch_size, 1
    )

    q_packed, q_scales_u8 = _quantize_to_mxfp4(q)
    q_in = (
        q_packed.view(batch_size, next_n, num_heads, head_dim // 2).view(torch.int8),
        q_scales_u8.view(torch.int32).squeeze(-1),
    )
    kv_in, _ = _kv_cache_cast_to_fp4(kv_cache)
    schedule_metadata = get_paged_mqa_logits_metadata(
        context_lens, block_size, get_num_sms()
    )

    default_logits = rocm_fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_model_len,
        clean_logits=False,
        trivial_topk_len=topk_tokens,
    )
    monkeypatch.setenv("VLLM_ROCM_FP4_INDEXER_TILE_CHUNKS", "2")
    tiled_logits = rocm_fp8_fp4_paged_mqa_logits(
        q_in,
        kv_in,
        weights,
        context_lens,
        block_table,
        schedule_metadata,
        max_context_len=max_model_len,
        clean_logits=False,
        trivial_topk_len=topk_tokens,
    )

    positions = torch.arange(max_model_len, device=device)[None, :]
    valid_mask = positions < context_lens.reshape(-1, 1)
    torch.testing.assert_close(
        tiled_logits.masked_fill(~valid_mask, 0),
        default_logits.masked_fill(~valid_mask, 0),
    )

    default_indices = torch.empty(
        (batch_size * next_n, topk_tokens), dtype=torch.int32, device=device
    )
    tiled_indices = torch.empty_like(default_indices)
    torch.ops._C.top_k_per_row_decode(
        default_logits,
        next_n,
        context_lens,
        default_indices,
        default_logits.shape[0],
        default_logits.stride(0),
        default_logits.stride(1),
        topk_tokens,
    )
    torch.ops._C.top_k_per_row_decode(
        tiled_logits,
        next_n,
        context_lens,
        tiled_indices,
        tiled_logits.shape[0],
        tiled_logits.stride(0),
        tiled_logits.stride(1),
        topk_tokens,
    )
    torch.testing.assert_close(tiled_indices, default_indices)
