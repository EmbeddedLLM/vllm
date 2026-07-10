# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ast
import inspect
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ATOM_MODEL = REPO_ROOT.parent / "ATOM" / "atom" / "models" / "deepseek_v4.py"


ATOM_V4_KERNEL_IMPORTS = {
    "CompressPlan",
    "csa_translate_pack",
    "fused_compress_attn",
    "inverse_rope_inplace",
    "qk_norm_rope_maybe_quant",
    "scale_indexer_weights",
    "sparse_attn_v4_paged_decode",
    "sparse_attn_v4_paged_prefill",
    "swa_write",
    "update_compressor_states",
}


VLLM_ATOM_OP_SURFACE = {
    "CompressPlan": "vllm/models/deepseek_v4/amd/v4_kernels/compress_plan.py",
    "csa_translate_pack": (
        "vllm/models/deepseek_v4/amd/v4_kernels/csa_translate_pack.py"
    ),
    "fused_compress_attn": ("vllm/models/deepseek_v4/amd/v4_kernels/fused_compress.py"),
    "inverse_rope_inplace": ("vllm/models/deepseek_v4/amd/v4_kernels/inverse_rope.py"),
    "qk_norm_rope_maybe_quant": (
        "vllm/models/deepseek_v4/amd/v4_kernels/qk_norm_rope_maybe_quant.py"
    ),
    "scale_indexer_weights": ("vllm/models/deepseek_v4/common/ops/fused_indexer_q.py"),
    "sparse_attn_v4_paged_decode": (
        "vllm/models/deepseek_v4/amd/v4_kernels/paged_decode.py"
    ),
    "sparse_attn_v4_paged_prefill": (
        "vllm/models/deepseek_v4/amd/v4_kernels/paged_prefill.py"
    ),
    "swa_write": "vllm/models/deepseek_v4/amd/v4_kernels/state_writes.py",
    "update_compressor_states": (
        "vllm/models/deepseek_v4/amd/v4_kernels/state_writes.py"
    ),
}


ATOM_AITER_OP_STRINGS = {
    "cp_gather_indexer_k_quant_cache",
    "deepgemm_fp8_paged_mqa_logits",
    "fp8_mqa_logits",
    "fused_clamp_act_mul",
    "get_hip_quant",
    "mhc_fused_post_pre",
    "mhc_post",
    "mhc_pre",
    "maybe_dual_stream_forward",
    "rope_rotate_activation",
    "top_k_per_row_decode",
    "top_k_per_row_prefill",
}


def _atom_tree() -> ast.Module:
    if not ATOM_MODEL.exists():
        pytest.skip(f"ATOM model file is unavailable: {ATOM_MODEL}")
    return ast.parse(ATOM_MODEL.read_text(), filename=str(ATOM_MODEL))


def _imported_names_from(module_name: str) -> set[str]:
    tree = _atom_tree()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module_name:
            names.update(alias.name for alias in node.names)
    return names


def _vllm_file_contains_symbol(rel_path: str, symbol: str) -> bool:
    source = (REPO_ROOT / rel_path).read_text()
    return (
        f"def {symbol}" in source
        or f"class {symbol}" in source
        or f'"{symbol}"' in source
        or f"'{symbol}'" in source
    )


def test_atom_v4_kernel_import_surface_is_explicitly_mapped_in_vllm():
    atom_imports = _imported_names_from("atom.model_ops.v4_kernels")

    assert atom_imports >= ATOM_V4_KERNEL_IMPORTS
    assert set(VLLM_ATOM_OP_SURFACE) == ATOM_V4_KERNEL_IMPORTS

    missing = [
        f"{symbol} -> {rel_path}"
        for symbol, rel_path in VLLM_ATOM_OP_SURFACE.items()
        if not _vllm_file_contains_symbol(rel_path, symbol)
    ]
    assert missing == []


def test_atom_aiter_op_surface_is_classified_in_notes():
    if not ATOM_MODEL.exists():
        pytest.skip(f"ATOM model file is unavailable: {ATOM_MODEL}")
    atom_source = ATOM_MODEL.read_text()
    notes = (REPO_ROOT / "docs" / "deepseek_v4_atom_op_surface_audit.md").read_text()

    atom_hits = {name for name in ATOM_AITER_OP_STRINGS if name in atom_source}
    assert atom_hits >= ATOM_AITER_OP_STRINGS

    missing_from_notes = [
        name for name in sorted(ATOM_AITER_OP_STRINGS) if name not in notes
    ]
    assert missing_from_notes == []


def test_atom_attention_forward_order_is_documented():
    notes = (REPO_ROOT / "docs" / "deepseek_v4_atom_op_surface_audit.md").read_text()
    ordered_steps = [
        "maybe_compressors_async",
        "qk_norm_rope_maybe_quant",
        "swa_write before decode",
        "indexer_score_topk",
        "csa_translate_pack",
        "sparse_attn_v4_paged_decode",
        "sparse_attn_v4_paged_prefill",
        "swa_write after prefill",
        "inverse_rope_inplace",
    ]

    cursor = -1
    for step in ordered_steps:
        next_pos = notes.find(step, cursor + 1)
        assert next_pos > cursor, step
        cursor = next_pos


def test_component_verdict_matrix_keeps_full_scope_answer_explicit():
    notes = (REPO_ROOT / "docs" / "deepseek_v4_atom_op_surface_audit.md").read_text()

    required_phrases = [
        "## Component Verdict Matrix",
        "It does not yet have every native ATOM kernel benefit",
        "vLLM scheduler and V2 model runner",
        "ROCm ModelState request rings",
        "vLLM-owned packed KV spec/allocation",
        "Native packed sparse attention ABI",
        "Native packed compressor ABI",
        "Full ATOM indexer dispatcher",
        "aiter `mhc_fused_post_pre`",
        "ATOM auxiliary stream and MoE overlap",
        "MHC is model-equivalence work, not the current attention/compressor ABI blocker",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in notes]
    assert missing == []


def test_native_abi_integration_target_is_actionable():
    notes = (REPO_ROOT / "docs" / "deepseek_v4_atom_op_surface_audit.md").read_text()

    required_phrases = [
        "## Native ABI Integration Target",
        "### Target A: Native split-packed ABI",
        "SWA prefix: BF16/model-dtype",
        "Compressed tail: `uint8 [num_blocks, k_per_block, 584]`",
        "448 FP8 NoPE bytes, 64 BF16 RoPE values, and 8 embedded",
        "atom_split_kv_swa",
        "atom_split_kv_compressed",
        "atom_split_kv_scales=None",
        'atom_split_kv_layout="fp8_ds_mla"',
        "packed `fp8_ds_mla` compressor dispatch no longer falls through",
        "packed `fp8_ds_mla` decode and prefill no longer require the Triton split-KV",
        "### Target B: ROCm-only homogeneous native ABI",
        "`atom_unified_kv` must be present for packed deployment",
        "sparse_attn_v4_paged_decode",
        "sparse_attn_v4_paged_prefill",
        "CUDA/NVIDIA MLA cache specs and bindings are unchanged",
    ]

    missing = [phrase for phrase in required_phrases if phrase not in notes]
    assert missing == []


def test_installed_aiter_has_no_packed_fp8_ds_mla_attention_or_compressor_abi():
    flydsl_compress = pytest.importorskip(
        "aiter.ops.flydsl.kernels.fused_compress_attn"
    )
    flydsl_hca = pytest.importorskip("aiter.ops.flydsl.kernels.fused_compress_attn_hca")
    opus_prefill = pytest.importorskip("aiter.ops.pa_sparse_prefill_opus")
    pa_mqa_logits = pytest.importorskip("aiter.ops.triton.attention.pa_mqa_logits")

    inspected_objects = [
        flydsl_compress.flydsl_fused_compress_attn,
        flydsl_hca.flydsl_hca_compress_attn,
        pa_mqa_logits.deepgemm_fp8_paged_mqa_logits,
        pa_mqa_logits.deepgemm_fp8_paged_mqa_logits_ragged_k,
    ]
    forbidden_terms = {
        "packed_fp8_ds_mla",
        "compressed_kv_layout",
        "swa_pages",
        "split_kv",
        "584",
        "576",
    }
    offenders: list[str] = []

    for obj in inspected_objects:
        signature = str(inspect.signature(obj)).lower()
        for term in forbidden_terms:
            if term in signature:
                offenders.append(f"{obj.__name__}:{term}:{signature}")

    opus_source = Path(opus_prefill.__file__).read_text().lower()
    assert "unified_kv dtype mismatch" in opus_source
    assert "kv dtype mismatch" in opus_source
    for term in forbidden_terms:
        if term in opus_source:
            offenders.append(f"pa_sparse_prefill_opus:{term}")

    assert offenders == []
