# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for sparse-indexer config defaults and IndexCache overrides."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import GlmMoeDsaConfig, PreTrainedConfig

import vllm.utils.deep_gemm as deep_gemm
from vllm.config import AttentionConfig, ModelConfig, VllmConfig
from vllm.model_executor.models.config import MODELS_CONFIG_MAP
from vllm.model_executor.models.utils import compute_skip_topk_layers, pp_topk_buffers


@pytest.mark.parametrize(
    "arch,sm100_default",
    [
        ("DeepseekV32ForCausalLM", "bfloat16"),
        ("DeepseekV4ForCausalLM", "bfloat16"),
        ("DeepseekV41ForCausalLM", "bfloat16"),
        ("GlmMoeDsaForCausalLM", "float32"),
        ("Glm5NextForCausalLM", "float32"),
    ],
)
@pytest.mark.parametrize("sm100", [True, False])
@pytest.mark.parametrize("requested", ["auto", "float32", "bfloat16"])
def test_indexer_logits_dtype_follows_model_default_unless_set(
    monkeypatch, arch: str, sm100_default: str, sm100: bool, requested: str
) -> None:
    monkeypatch.setattr(deep_gemm, "mqa_logits_use_mx_scales", lambda: sm100)
    attention_config = AttentionConfig(
        indexer_logits_dtype=requested, indexer_sparse_logits=False
    )
    vllm_config = SimpleNamespace(
        attention_config=attention_config,
        # A non-auto KV dtype keeps the hooks off their platform-specific paths.
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(index_topk=2048),
            hf_text_config=SimpleNamespace(index_topk=2048),
            hf_overrides={},
        ),
        parallel_config=SimpleNamespace(set_dcp_defaults=lambda **_: None),
        speculative_config=None,
    )

    MODELS_CONFIG_MAP[arch].verify_and_update_config(vllm_config)

    # The indexer layers resolve a remaining "auto" to float32.
    default = sm100_default if sm100 else "float32"
    expected = default if requested == "auto" else requested
    assert attention_config.resolve_indexer_logits_dtype("float32") == expected


def _glm_dsa_config(tmp_path: Path, hf_overrides: dict) -> PreTrainedConfig:
    """12-layer GlmMoeDsa config shipping a freq-4 indexer_types list, run
    through the real ModelConfig/VllmConfig chain so hf_overrides and the
    verify_and_update_config hook apply as they do at engine startup."""
    GlmMoeDsaConfig(
        num_hidden_layers=12, index_topk_freq=4, index_skip_topk_offset=3
    ).save_pretrained(tmp_path)
    model_config = ModelConfig(
        str(tmp_path), hf_overrides=hf_overrides, skip_tokenizer_init=True
    )
    VllmConfig(model_config=model_config)
    return model_config.hf_config


@pytest.mark.parametrize(
    ("hf_overrides", "expected_skip"),
    [
        ({"index_topk_freq": 8}, {3, 4, 5, 6, 7, 8, 9, 11}),
        # freq=4, offset=0: full layers are 3, 7, 11.
        ({"index_skip_topk_offset": 0}, {0, 1, 2, 4, 5, 6, 8, 9, 10}),
        ({"index_topk_pattern": "FS" * 6}, {1, 3, 5, 7, 9, 11}),
    ],
    ids=["freq", "offset", "pattern"],
)
def test_glm_dsa_schedule_override_replaces_shipped_indexer_types(
    tmp_path: Path, hf_overrides: dict, expected_skip: set[int]
) -> None:
    hf_config = _glm_dsa_config(tmp_path, hf_overrides)

    assert compute_skip_topk_layers(hf_config) == expected_skip


def test_glm_dsa_without_overrides_keeps_shipped_indexer_types(tmp_path: Path) -> None:
    hf_config = _glm_dsa_config(tmp_path, {})

    shipped = GlmMoeDsaConfig(
        num_hidden_layers=12, index_topk_freq=4, index_skip_topk_offset=3
    ).indexer_types
    assert hf_config.indexer_types == shipped
    assert compute_skip_topk_layers(hf_config) == {3, 4, 5, 7, 8, 9, 11}


def test_glm_dsa_explicit_indexer_types_survives_freq_override(tmp_path: Path) -> None:
    explicit = ["full"] * 6 + ["shared"] * 6
    hf_config = _glm_dsa_config(
        tmp_path, {"index_topk_freq": 8, "indexer_types": explicit}
    )

    assert hf_config.indexer_types == explicit
    assert compute_skip_topk_layers(hf_config) == set(range(6, 12))


def test_glm_dsa_freq_override_keeps_pp_topk_buffers_consistent(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "vllm.distributed.parallel_state.get_pp_group",
        lambda: SimpleNamespace(world_size=2),
    )
    buffer = torch.empty(4, 2048, dtype=torch.int32)

    overridden = _glm_dsa_config(tmp_path / "overridden", {"index_topk_freq": 8})
    assert pp_topk_buffers(overridden, buffer, range(12)) == {"topk_indices": buffer}

    shipped = _glm_dsa_config(tmp_path / "shipped", {})
    assert pp_topk_buffers(shipped, buffer, range(12)) == {}
