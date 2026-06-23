# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.v1.kv_cache_interface import (
    DeepseekV4AtomMLAAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.utils import (
    AttentionGroup,
    KVBlockZeroer,
    _representative_worker_spec,
    bind_kv_cache,
)


class _PostBindLayer:
    def __init__(self) -> None:
        self.kv_cache: torch.Tensor | None = None
        self.post_bind_calls: list[torch.Tensor] = []

    def post_bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        assert self.kv_cache is kv_cache
        self.post_bind_calls.append(kv_cache)


def test_bind_kv_cache(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    ctx = {
        "layers.0.self_attn": Attention(32, 128, 0.1, prefix="layers.0.self_attn"),
        "layers.1.self_attn": Attention(32, 128, 0.1, prefix="layers.1.self_attn"),
        "layers.2.self_attn": Attention(32, 128, 0.1, prefix="layers.2.self_attn"),
        "layers.3.self_attn": Attention(32, 128, 0.1, prefix="layers.3.self_attn"),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.zeros((1,)),
        "layers.2.self_attn": torch.zeros((1,)),
        "layers.3.self_attn": torch.zeros((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)
    assert ctx["layers.0.self_attn"].kv_cache is kv_cache["layers.0.self_attn"]
    assert ctx["layers.1.self_attn"].kv_cache is kv_cache["layers.1.self_attn"]
    assert ctx["layers.2.self_attn"].kv_cache is kv_cache["layers.2.self_attn"]
    assert ctx["layers.3.self_attn"].kv_cache is kv_cache["layers.3.self_attn"]

    assert runner_kv_caches[0] is kv_cache["layers.0.self_attn"]
    assert runner_kv_caches[1] is kv_cache["layers.1.self_attn"]
    assert runner_kv_caches[2] is kv_cache["layers.2.self_attn"]
    assert runner_kv_caches[3] is kv_cache["layers.3.self_attn"]


def test_bind_kv_cache_calls_post_bind_hook(default_vllm_config):
    ctx = {
        "layers.0.self_attn": _PostBindLayer(),
        "layers.1.self_attn": _PostBindLayer(),
    }
    kv_cache = {
        "layers.0.self_attn": torch.zeros((1,)),
        "layers.1.self_attn": torch.ones((1,)),
    }
    runner_kv_caches: list[torch.Tensor] = []

    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["layers.0.self_attn"].post_bind_calls == [kv_cache["layers.0.self_attn"]]
    assert ctx["layers.1.self_attn"].post_bind_calls == [kv_cache["layers.1.self_attn"]]
    assert runner_kv_caches == [
        kv_cache["layers.0.self_attn"],
        kv_cache["layers.1.self_attn"],
    ]


def test_bind_kv_cache_non_attention(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    # example from Jamba PP=2
    ctx = {
        "model.layers.20.attn": Attention(32, 128, 0.1, prefix="model.layers.20.attn"),
        "model.layers.28.attn": Attention(32, 128, 0.1, prefix="model.layers.28.attn"),
    }
    kv_cache = {
        "model.layers.20.attn": torch.zeros((1,)),
        "model.layers.28.attn": torch.zeros((1,)),
    }

    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.20.attn"].kv_cache is kv_cache["model.layers.20.attn"]
    assert ctx["model.layers.28.attn"].kv_cache is kv_cache["model.layers.28.attn"]

    assert runner_kv_caches[0] is kv_cache["model.layers.20.attn"]
    assert runner_kv_caches[1] is kv_cache["model.layers.28.attn"]


def test_bind_kv_cache_draft_model(default_vllm_config):
    from vllm.model_executor.layers.attention import Attention

    layer_names = [
        "model.layers.0.attn",
        "model.layers.1.attn",
        "draft_model.layers.0.attn",
        "draft_model.layers.1.attn",
    ]
    ctx = {
        layer_name: Attention(32, 128, 0.1, prefix=layer_name)
        for layer_name in layer_names
    }
    kv_cache = {layer_name: torch.zeros((1,)) for layer_name in layer_names}
    runner_kv_caches: list[torch.Tensor] = []
    bind_kv_cache(kv_cache, ctx, runner_kv_caches)

    assert ctx["model.layers.0.attn"].kv_cache is kv_cache["model.layers.0.attn"]
    assert ctx["model.layers.1.attn"].kv_cache is kv_cache["model.layers.1.attn"]
    assert (
        ctx["draft_model.layers.0.attn"].kv_cache
        is kv_cache["draft_model.layers.0.attn"]
    )
    assert (
        ctx["draft_model.layers.1.attn"].kv_cache
        is kv_cache["draft_model.layers.1.attn"]
    )

    # caches are ordered by layer_index, interleaving target and draft model
    assert runner_kv_caches[0] is kv_cache["model.layers.0.attn"]
    assert runner_kv_caches[1] is kv_cache["draft_model.layers.0.attn"]
    assert runner_kv_caches[2] is kv_cache["model.layers.1.attn"]
    assert runner_kv_caches[3] is kv_cache["draft_model.layers.1.attn"]


def test_representative_worker_spec_prefers_atom_mla():
    regular = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
    )
    atom = DeepseekV4AtomMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        atom_swa_prefix_bytes=100,
        atom_swa_pages=5,
    )
    group_spec = UniformTypeKVCacheSpecs(
        block_size=regular.block_size,
        kv_cache_specs={"regular": regular, "atom": atom},
    )

    representative = _representative_worker_spec(group_spec)

    assert representative is atom
    assert representative.atom_swa_prefix_bytes == atom.atom_swa_prefix_bytes
    assert representative.atom_swa_pages == atom.atom_swa_pages


def _deepseek_v4_attn_stub(
    *,
    compress_ratio: int = 4,
    head_dim: int = 512,
    kv_cache_dtype: str = "bf16",
) -> SimpleNamespace:
    return SimpleNamespace(
        compress_ratio=compress_ratio,
        head_dim=head_dim,
        kv_cache_dtype=kv_cache_dtype,
        kv_cache_torch_dtype=torch.bfloat16,
        window_size=128,
    )


def test_deepseek_v4_kv_cache_spec_stays_regular_mla_off_rocm(
    default_vllm_config, monkeypatch
):
    from vllm.models.deepseek_v4 import attention as attention_mod
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    default_vllm_config.cache_config.block_size = 128
    monkeypatch.setattr(attention_mod.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(attention_mod, "_ATOM_ROCM_DSV4_ENABLED", False)
    monkeypatch.setattr(attention_mod, "_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED", True)
    monkeypatch.setattr(attention_mod, "_ATOM_MIXED_KV_ENABLED", True)
    attn = _deepseek_v4_attn_stub()

    spec = DeepseekV4Attention.get_kv_cache_spec(attn, default_vllm_config)

    assert isinstance(spec, MLAAttentionSpec)
    assert not isinstance(spec, DeepseekV4AtomMLAAttentionSpec)
    assert not hasattr(attn, "atom_vllm_unified_kv_prefix_bytes")


def test_deepseek_v4_kv_cache_spec_uses_atom_mla_only_for_rocm_unified(
    default_vllm_config, monkeypatch
):
    from vllm.models.deepseek_v4 import attention as attention_mod
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    default_vllm_config.cache_config.block_size = 128
    default_vllm_config.scheduler_config.max_num_seqs = 2
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    monkeypatch.setattr(attention_mod.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(attention_mod, "_ATOM_ROCM_DSV4_ENABLED", True)
    monkeypatch.setattr(attention_mod, "_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED", True)
    monkeypatch.setattr(attention_mod, "_ATOM_MIXED_KV_ENABLED", True)
    attn = _deepseek_v4_attn_stub()

    spec = DeepseekV4Attention.get_kv_cache_spec(attn, default_vllm_config)

    expected_swa_pages = default_vllm_config.scheduler_config.max_num_seqs * 128
    expected_prefix_bytes = expected_swa_pages * attn.head_dim * torch.bfloat16.itemsize
    assert isinstance(spec, DeepseekV4AtomMLAAttentionSpec)
    assert spec.cache_dtype_str == "fp8_ds_mla"
    assert spec.atom_compressed_layout == "fp8_ds_mla"
    assert spec.atom_swa_pages == expected_swa_pages
    assert spec.atom_swa_prefix_bytes == expected_prefix_bytes
    assert attn.atom_vllm_unified_kv_prefix_bytes == expected_prefix_bytes


def test_deepseek_v4_vllm_owned_atom_kv_enforces_block_size(
    default_vllm_config, monkeypatch
):
    from vllm.models.deepseek_v4 import attention as attention_mod
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    default_vllm_config.cache_config.block_size = 256
    default_vllm_config.model_config = SimpleNamespace(dtype=torch.bfloat16)
    monkeypatch.setattr(attention_mod.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(attention_mod, "_ATOM_ROCM_DSV4_ENABLED", False)
    monkeypatch.setattr(attention_mod, "_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED", True)
    attn = _deepseek_v4_attn_stub()

    with pytest.raises(ValueError, match="--block-size 128"):
        DeepseekV4Attention.get_kv_cache_spec(attn, default_vllm_config)


def test_deepseek_v4_model_state_cls_stays_default_off_rocm(monkeypatch):
    from vllm.models.deepseek_v4.amd import model as model_mod
    from vllm.models.deepseek_v4.amd.model import DeepseekV4ForCausalLM
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    monkeypatch.setattr(model_mod.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(model_mod, "_ROCM_DSV4_ATOM_STATE_ENABLED", True)

    assert DeepseekV4ForCausalLM.get_model_state_cls() is DefaultModelState


def test_deepseek_v4_model_state_cls_stays_default_without_atom_state(
    monkeypatch,
):
    from vllm.models.deepseek_v4.amd import model as model_mod
    from vllm.models.deepseek_v4.amd.model import DeepseekV4ForCausalLM
    from vllm.v1.worker.gpu.model_states.default import DefaultModelState

    monkeypatch.setattr(model_mod.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(model_mod, "_ROCM_DSV4_ATOM_STATE_ENABLED", False)

    assert DeepseekV4ForCausalLM.get_model_state_cls() is DefaultModelState


def test_deepseek_v4_model_state_cls_uses_atom_state_only_for_rocm_atom(
    monkeypatch,
):
    from vllm.models.deepseek_v4.amd import model as model_mod
    from vllm.models.deepseek_v4.amd.model import DeepseekV4ForCausalLM
    from vllm.models.deepseek_v4.amd.model_state import DeepseekV4RocmAtomModelState

    monkeypatch.setattr(model_mod.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(model_mod, "_ROCM_DSV4_ATOM_STATE_ENABLED", True)

    assert DeepseekV4ForCausalLM.get_model_state_cls() is DeepseekV4RocmAtomModelState


def test_deepseek_v4_split_only_kv_decode_auto_selects_split_path(monkeypatch):
    from vllm.models.deepseek_v4.amd import rocm as rocm_mod

    unified = torch.zeros((1, 8), dtype=torch.bfloat16)
    split_swa = torch.zeros((1, 1, 8), dtype=torch.bfloat16)
    split_tail = torch.zeros((1, 4, 584), dtype=torch.uint8)

    monkeypatch.setattr(rocm_mod, "_ATOM_DECODE_KV_SPLITS", 0)
    monkeypatch.setattr(rocm_mod, "_ATOM_SPLIT_KV_DECODE", False)

    assert rocm_mod._should_use_atom_split_kv_decode(None, split_swa, split_tail)
    assert not rocm_mod._should_use_atom_split_kv_decode(unified, split_swa, split_tail)

    monkeypatch.setattr(rocm_mod, "_ATOM_SPLIT_KV_DECODE", True)

    assert not rocm_mod._should_use_atom_split_kv_decode(unified, split_swa, split_tail)

    monkeypatch.setattr(rocm_mod, "_ATOM_DECODE_KV_SPLITS", 1)

    assert rocm_mod._should_use_atom_split_kv_decode(unified, split_swa, split_tail)

    monkeypatch.setattr(rocm_mod, "_ATOM_DECODE_KV_SPLITS", 2)

    assert rocm_mod._should_use_atom_split_kv_decode(None, split_swa, split_tail)


def test_deepseek_v4_atom_kv_views_prefer_metadata_bundle():
    from vllm.models.deepseek_v4.amd.model_state import (
        DeepseekV4RocmAtomUnifiedKVBuffers,
    )
    from vllm.models.deepseek_v4.amd.rocm import _resolve_atom_kv_views

    stale_tail = torch.zeros((1, 4, 8), dtype=torch.bfloat16)
    packed_tail = torch.zeros((1, 4, 584), dtype=torch.uint8)
    split_swa = torch.zeros((1, 1, 8), dtype=torch.bfloat16)
    scale = torch.ones((4, 1), dtype=torch.float32)
    attn = SimpleNamespace(
        _atom_layer_id=3,
        atom_split_kv_swa=split_swa,
        atom_split_kv_compressed=stale_tail,
        atom_split_kv_scales=scale,
        atom_split_kv_layout="dense",
    )
    buffers = DeepseekV4RocmAtomUnifiedKVBuffers(
        unified_kv=(),
        unified_kv_by_layer={},
        compressed_kv_cache={3: packed_tail},
        compressed_kv_scales={3: None},
        compressed_kv_layout={3: "fp8_ds_mla"},
        active_layer_ids=(3,),
        num_blocks=1,
        swa_pages=1,
        k1_csa=4,
        k2_hca=1,
    )
    atom_state = SimpleNamespace(
        unified_kv_buffers=buffers,
        swa_pages=1,
        win_with_spec=1,
    )

    views = _resolve_atom_kv_views(attn, atom_state)

    assert views.split_swa_kv is split_swa
    assert views.split_compressed_kv is packed_tail
    assert views.split_kv_scales is None
    assert views.split_kv_layout == "fp8_ds_mla"


def test_deepseek_v4_atom_kv_views_use_layer_map_for_homogeneous_kv():
    from vllm.models.deepseek_v4.amd.model_state import (
        DeepseekV4RocmAtomUnifiedKVBuffers,
    )
    from vllm.models.deepseek_v4.amd.rocm import _resolve_atom_kv_views

    dense_unified = torch.zeros((5, 8), dtype=torch.bfloat16)
    dense_tail = dense_unified[1:].view(1, 4, 8)
    attn = SimpleNamespace(_atom_layer_id=3, max_num_reqs=1, head_dim=8)
    buffers = DeepseekV4RocmAtomUnifiedKVBuffers(
        unified_kv=(dense_unified,),
        unified_kv_by_layer={3: dense_unified},
        compressed_kv_cache={3: dense_tail},
        compressed_kv_scales={3: None},
        compressed_kv_layout={3: "dense"},
        active_layer_ids=(0, 3),
        num_blocks=1,
        swa_pages=1,
        k1_csa=4,
        k2_hca=1,
    )
    atom_state = SimpleNamespace(
        unified_kv_buffers=buffers,
        swa_pages=1,
        win_with_spec=1,
    )

    views = _resolve_atom_kv_views(attn, atom_state)

    assert views.unified_kv is dense_unified
    assert views.split_swa_kv.shape == (1, 1, 8)
    assert views.split_compressed_kv is dense_tail
    assert views.split_kv_layout == "dense"


class _AtomMLATestBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        *,
        cache_dtype_str: str | None = None,
    ) -> tuple[int, int, int]:
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)


class _AtomPackedMLATestBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        *,
        cache_dtype_str: str | None = None,
    ) -> tuple[int, int, int]:
        assert num_kv_heads == 1
        if cache_dtype_str == "fp8_ds_mla":
            return (num_blocks, block_size, 584)
        return (num_blocks, block_size, head_size)


class _ZeroerTestBackend(_AtomPackedMLATestBackend):
    @staticmethod
    def get_kv_cache_block_dim(
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        *,
        cache_dtype_str: str | None = None,
    ) -> int:
        return 0


def test_kv_block_zeroer_includes_atom_attention_specs():
    atom = DeepseekV4AtomMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        atom_swa_prefix_bytes=1024,
        atom_swa_pages=1,
        atom_compressed_kv_dtype=torch.uint8,
        atom_compressed_layout="fp8_ds_mla",
    )
    kv_cache = torch.zeros((2, atom.storage_block_size, 584), dtype=torch.uint8)
    group = AttentionGroup(
        backend=_ZeroerTestBackend,  # type: ignore[arg-type]
        layer_names=["atom"],
        kv_cache_spec=atom,
        kv_cache_group_id=0,
    )
    layer = SimpleNamespace(kv_cache=kv_cache)

    zeroer = KVBlockZeroer(
        device=torch.device("cpu"),
        pin_memory=False,
        attn_groups_iter=[group],
        kernel_block_sizes=[atom.storage_block_size],
        cache_dtype="auto",
        static_forward_context={"atom": layer},
    )

    assert len(zeroer._metas) == 1
    _, page_size_el, _, n_segs = zeroer._metas[0]
    assert page_size_el == atom.page_size_bytes // 4
    assert n_segs == 1


def test_kv_block_zeroer_groups_nonuniform_atom_page_sizes():
    regular = MLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
    )
    atom = DeepseekV4AtomMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        atom_swa_prefix_bytes=1024,
        atom_swa_pages=1,
        atom_compressed_kv_dtype=torch.uint8,
        atom_compressed_layout="fp8_ds_mla",
    )
    groups = [
        AttentionGroup(
            backend=_ZeroerTestBackend,  # type: ignore[arg-type]
            layer_names=["regular"],
            kv_cache_spec=regular,
            kv_cache_group_id=0,
        ),
        AttentionGroup(
            backend=_ZeroerTestBackend,  # type: ignore[arg-type]
            layer_names=["atom"],
            kv_cache_spec=atom,
            kv_cache_group_id=1,
        ),
    ]
    context = {
        "regular": SimpleNamespace(
            kv_cache=torch.zeros((2, 4, 8), dtype=torch.bfloat16)
        ),
        "atom": SimpleNamespace(kv_cache=torch.zeros((2, 4, 584), dtype=torch.uint8)),
    }

    zeroer = KVBlockZeroer(
        device=torch.device("cpu"),
        pin_memory=False,
        attn_groups_iter=groups,
        kernel_block_sizes=[regular.block_size, atom.storage_block_size],
        cache_dtype="auto",
        static_forward_context=context,
    )

    page_sizes = sorted(meta[1] for meta in zeroer._metas)
    assert page_sizes == sorted(
        [
            regular.page_size_bytes // 4,
            atom.page_size_bytes // 4,
        ]
    )


def test_reshape_kv_cache_strides_atom_mixed_tail_scales():
    atom = DeepseekV4AtomMLAAttentionSpec(
        block_size=4,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.uint8,
        atom_swa_prefix_bytes=100,
        atom_swa_pages=5,
        atom_compressed_kv_dtype=torch.uint8,
        atom_compressed_scale_dtype=torch.float32,
        atom_compressed_scale_bytes_per_page=16,
    )
    raw = torch.zeros(
        atom.atom_swa_prefix_bytes + 2 * atom.page_size_bytes,
        dtype=torch.int8,
    )
    raw[atom.atom_swa_prefix_bytes + atom.page_size_bytes] = 7
    group = AttentionGroup(
        backend=_AtomMLATestBackend,  # type: ignore[arg-type]
        layer_names=["atom"],
        kv_cache_spec=atom,
        kv_cache_group_id=0,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=2,
        kv_cache_tensors=[
            KVCacheTensor(
                size=raw.numel(),
                shared_by=["atom"],
                fixed_prefix_size=atom.atom_swa_prefix_bytes,
            )
        ],
        kv_cache_groups=[],
    )

    kv_caches = _reshape_kv_cache(
        [group],
        {"atom": raw},
        cache_dtype="auto",
        kernel_block_sizes=[atom.block_size],
        shared_kv_cache_layers={},
    )

    kv_cache = kv_caches["atom"]
    assert kv_cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert kv_cache.storage_offset() == atom.atom_swa_prefix_bytes
    assert kv_cache.shape == (2, 4, 8)
    assert kv_cache.stride(0) == atom.page_size_bytes
    assert kv_cache.stride(1) == 8 + atom.atom_compressed_scale_bytes_per_page
    assert int(kv_cache[1, 0, 0]) == 7
    assert kv_cache_config.kv_cache_tensors[0].fixed_prefix_size == (
        atom.atom_swa_prefix_bytes
    )


def test_reshape_kv_cache_atom_packed_fp8_tail_keeps_584_byte_slots():
    atom = DeepseekV4AtomMLAAttentionSpec(
        block_size=8,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        compress_ratio=4,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        atom_swa_prefix_bytes=128,
        atom_swa_pages=1,
        atom_compressed_kv_dtype=torch.uint8,
        atom_compressed_layout="fp8_ds_mla",
    )
    raw = torch.zeros(
        atom.atom_swa_prefix_bytes + 3 * atom.page_size_bytes,
        dtype=torch.int8,
    )
    first_tail_byte = atom.atom_swa_prefix_bytes + atom.page_size_bytes
    raw[first_tail_byte] = 11
    group = AttentionGroup(
        backend=_AtomPackedMLATestBackend,  # type: ignore[arg-type]
        layer_names=["atom"],
        kv_cache_spec=atom,
        kv_cache_group_id=0,
    )

    kv_caches = _reshape_kv_cache(
        [group],
        {"atom": raw},
        cache_dtype="auto",
        kernel_block_sizes=[atom.storage_block_size],
        shared_kv_cache_layers={},
    )

    kv_cache = kv_caches["atom"]
    assert kv_cache.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert kv_cache.storage_offset() == atom.atom_swa_prefix_bytes
    assert atom.real_page_size_bytes == atom.storage_block_size * 584
    assert kv_cache.shape == (3, atom.storage_block_size, 584)
    assert kv_cache.dtype == torch.uint8
    assert kv_cache.stride(0) == atom.page_size_bytes
    assert int(kv_cache[1, 0, 0]) == 11


def test_deepseek_v4_post_bind_stays_noop_off_rocm(monkeypatch):
    from vllm.models.deepseek_v4 import attention as attention_mod
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    monkeypatch.setattr(attention_mod.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(attention_mod, "_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED", True)
    attn = SimpleNamespace(
        compress_ratio=4,
        atom_vllm_unified_kv_prefix_bytes=1024,
        atom_vllm_unified_kv_swa_pages=1,
        atom_vllm_unified_kv_swa_dtype=torch.bfloat16,
        atom_vllm_compressed_layout="fp8_ds_mla",
        kv_cache_torch_dtype=torch.bfloat16,
        head_dim=512,
        max_num_reqs=1,
        prefix="layers.0.attn",
        compressor=None,
    )

    DeepseekV4Attention.post_bind_kv_cache(attn, torch.zeros(1))

    assert not hasattr(attn, "atom_swa_kv")
    assert not hasattr(attn, "atom_split_kv_compressed")
    assert not hasattr(attn, "atom_unified_kv")


def test_deepseek_v4_post_bind_stays_noop_for_swa_layer(monkeypatch):
    from vllm.models.deepseek_v4 import attention as attention_mod
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    monkeypatch.setattr(attention_mod.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(attention_mod, "_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED", True)
    attn = SimpleNamespace(
        compress_ratio=1,
        atom_vllm_unified_kv_prefix_bytes=1024,
        atom_vllm_unified_kv_swa_pages=1,
        atom_vllm_unified_kv_swa_dtype=torch.bfloat16,
        atom_vllm_compressed_layout="fp8_ds_mla",
        kv_cache_torch_dtype=torch.bfloat16,
        head_dim=512,
        max_num_reqs=1,
        prefix="layers.0.attn",
        compressor=None,
    )

    DeepseekV4Attention.post_bind_kv_cache(attn, torch.zeros(1))

    assert not hasattr(attn, "atom_swa_kv")
    assert not hasattr(attn, "atom_split_kv_compressed")
    assert not hasattr(attn, "atom_unified_kv")


def test_deepseek_v4_post_bind_exposes_mixed_atom_split_views(monkeypatch):
    from vllm.models.deepseek_v4 import attention as attention_mod
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    monkeypatch.setattr(attention_mod.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(attention_mod, "_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED", True)

    head_dim = 64
    swa_pages = 1
    prefix_bytes = swa_pages * head_dim * torch.bfloat16.itemsize
    num_blocks = 2
    k_per_block = 4
    scale_bytes = torch.float32.itemsize
    row_bytes = head_dim * torch.float8_e4m3fnuz.itemsize
    tail_pages = num_blocks * k_per_block
    raw = torch.zeros(
        prefix_bytes + tail_pages * (row_bytes + scale_bytes), dtype=torch.int8
    )
    raw.view(torch.float32)[(prefix_bytes + row_bytes) // torch.float32.itemsize] = 3.5
    tail = torch.as_strided(
        raw[prefix_bytes:].view(torch.float8_e4m3fnuz),
        size=(num_blocks, k_per_block, head_dim),
        stride=(k_per_block * (row_bytes + scale_bytes), row_bytes + scale_bytes, 1),
    )
    compressor = SimpleNamespace()
    attn = SimpleNamespace(
        compress_ratio=4,
        atom_vllm_unified_kv_prefix_bytes=prefix_bytes,
        atom_vllm_unified_kv_swa_pages=swa_pages,
        atom_vllm_unified_kv_swa_dtype=torch.bfloat16,
        atom_vllm_compressed_scale_bytes_per_page=scale_bytes,
        kv_cache_torch_dtype=torch.bfloat16,
        head_dim=head_dim,
        max_num_reqs=1,
        prefix="layers.0.attn",
        compressor=compressor,
    )

    DeepseekV4Attention.post_bind_kv_cache(attn, tail)

    assert attn.atom_swa_kv.shape == (1, 1, head_dim)
    assert attn.atom_swa_kv.dtype is torch.bfloat16
    assert attn.atom_split_kv_compressed is tail
    assert attn.atom_split_kv_layout == "dense"
    assert attn.atom_split_kv_scales.shape == (tail_pages, 1)
    assert attn.atom_split_kv_scales.stride() == (
        (row_bytes + scale_bytes) // torch.float32.itemsize,
        1,
    )
    assert float(attn.atom_split_kv_scales[0, 0]) == 3.5
    assert compressor.atom_kv_cache is tail
    assert compressor.atom_kv_scales is attn.atom_split_kv_scales
    assert compressor.atom_kv_layout == "dense"
    assert not hasattr(attn, "atom_unified_kv")


def test_deepseek_v4_post_bind_rejects_unknown_atom_layout(monkeypatch):
    from vllm.models.deepseek_v4 import attention as attention_mod
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    monkeypatch.setattr(attention_mod.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(attention_mod, "_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED", True)
    attn = SimpleNamespace(
        compress_ratio=4,
        atom_vllm_unified_kv_prefix_bytes=128,
        atom_vllm_unified_kv_swa_pages=1,
        atom_vllm_unified_kv_swa_dtype=torch.bfloat16,
        atom_vllm_compressed_layout="unexpected",
        atom_vllm_compressed_scale_bytes_per_page=0,
        kv_cache_torch_dtype=torch.bfloat16,
        head_dim=64,
        max_num_reqs=1,
        prefix="layers.0.attn",
        compressor=None,
    )

    with pytest.raises(RuntimeError, match="unsupported compressed layout"):
        DeepseekV4Attention.post_bind_kv_cache(attn, torch.zeros((1, 4, 64)))


def test_deepseek_v4_post_bind_exposes_packed_atom_split_view(monkeypatch):
    from vllm.models.deepseek_v4 import attention as attention_mod
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    monkeypatch.setattr(attention_mod.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(attention_mod, "_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED", True)

    head_dim = 512
    swa_pages = 1
    prefix_bytes = swa_pages * head_dim * torch.bfloat16.itemsize
    num_blocks = 2
    k_per_block = 4
    tail_pages = num_blocks * k_per_block
    raw = torch.zeros(prefix_bytes + tail_pages * 584, dtype=torch.uint8)
    raw[prefix_bytes] = 13
    tail = raw[prefix_bytes:].view(num_blocks, k_per_block, 584)
    attn = SimpleNamespace(
        compress_ratio=4,
        atom_vllm_unified_kv_prefix_bytes=prefix_bytes,
        atom_vllm_unified_kv_swa_pages=swa_pages,
        atom_vllm_unified_kv_swa_dtype=torch.bfloat16,
        atom_vllm_compressed_scale_bytes_per_page=0,
        atom_vllm_compressed_layout="fp8_ds_mla",
        kv_cache_torch_dtype=torch.bfloat16,
        head_dim=head_dim,
        max_num_reqs=1,
        prefix="layers.0.attn",
        compressor=None,
    )

    DeepseekV4Attention.post_bind_kv_cache(attn, tail)

    assert attn.atom_swa_kv.shape == (1, 1, head_dim)
    assert attn.atom_split_kv_compressed is tail
    assert attn.atom_split_kv_layout == "fp8_ds_mla"
    assert attn.atom_split_kv_scales is None
    assert not hasattr(attn, "atom_unified_kv")


def test_deepseek_v4_model_state_binds_packed_vllm_owned_kv():
    from vllm.models.deepseek_v4.amd.model_state import (
        DeepseekV4RocmAtomModelState,
        DeepseekV4RocmAtomUnifiedKVBuffers,
    )

    head_dim = 512
    swa_pages = 1
    prefix_bytes = swa_pages * head_dim * torch.bfloat16.itemsize
    num_blocks = 2
    k_per_block = 4
    raw = torch.zeros(prefix_bytes + num_blocks * k_per_block * 584, dtype=torch.uint8)
    raw.view(torch.bfloat16)[0] = 7
    raw[prefix_bytes] = 13
    tail = raw[prefix_bytes:].view(num_blocks, k_per_block, 584)
    compressor = SimpleNamespace()
    attn = SimpleNamespace(
        compress_ratio=4,
        kv_cache=tail,
        atom_vllm_unified_kv_prefix_bytes=prefix_bytes,
        atom_vllm_unified_kv_swa_pages=swa_pages,
        atom_vllm_unified_kv_swa_dtype=torch.bfloat16,
        atom_vllm_compressed_layout="fp8_ds_mla",
        prefix="layers.0.attn",
        compressor=compressor,
    )
    state = DeepseekV4RocmAtomModelState.__new__(DeepseekV4RocmAtomModelState)
    state._enable_atom_unified_kv_from_vllm = True
    state._atom_unified_kv_from_vllm_bound = False
    state._atom_unified_kv_buffers = None
    state._atom_state_buffers = None
    state.max_num_reqs = 1
    state.win_with_spec = 1
    state.swa_pages = swa_pages
    state.head_dim = head_dim
    state.dtype = torch.bfloat16
    state.k1_csa = k_per_block
    state.k2_hca = 1
    state._iter_active_attn_modules = lambda: [(0, attn)]
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    assert state._try_bind_atom_unified_kv_from_vllm(kv_cache_config)

    buffers = state._atom_unified_kv_buffers
    assert isinstance(buffers, DeepseekV4RocmAtomUnifiedKVBuffers)
    assert buffers.active_layer_ids == (0,)
    assert buffers.unified_kv == ()
    assert buffers.unified_kv_by_layer == {}
    assert buffers.compressed_kv_cache[0] is tail
    assert buffers.compressed_kv_scales[0] is None
    assert buffers.compressed_kv_layout[0] == "fp8_ds_mla"
    assert state._atom_unified_kv_from_vllm_bound
    assert attn.atom_swa_kv.shape == (1, 1, head_dim)
    assert attn.atom_swa_kv.dtype is torch.bfloat16
    assert float(attn.atom_swa_kv[0, 0, 0]) == 7
    assert attn.atom_split_kv_compressed is tail
    assert attn.atom_split_kv_layout == "fp8_ds_mla"
    assert attn.atom_split_kv_scales is None
    assert attn.atom_compressed_kv_cache is tail
    assert compressor.atom_kv_cache is tail
    assert compressor.atom_kv_scales is None
    assert compressor.atom_kv_layout == "fp8_ds_mla"
    assert not hasattr(attn, "atom_unified_kv")

    state._reset_atom_request_slot(0)

    assert float(attn.atom_swa_kv[0, 0, 0]) == 0
    assert int(raw[prefix_bytes]) == 13


def test_deepseek_v4_model_state_binds_sidecar_vllm_owned_kv(monkeypatch):
    from vllm.models.deepseek_v4 import attention as attention_mod
    from vllm.models.deepseek_v4.amd.model_state import (
        DeepseekV4RocmAtomModelState,
        DeepseekV4RocmAtomUnifiedKVBuffers,
    )
    from vllm.models.deepseek_v4.attention import DeepseekV4Attention

    monkeypatch.setattr(attention_mod.current_platform, "is_rocm", lambda: True)
    monkeypatch.setattr(attention_mod, "_ATOM_UNIFIED_KV_FROM_VLLM_ENABLED", True)

    head_dim = 64
    swa_pages = 1
    prefix_bytes = swa_pages * head_dim * torch.bfloat16.itemsize
    num_blocks = 2
    k_per_block = 4
    scale_bytes = torch.float32.itemsize
    row_bytes = head_dim * torch.float8_e4m3fnuz.itemsize
    tail_pages = num_blocks * k_per_block
    raw = torch.zeros(
        prefix_bytes + tail_pages * (row_bytes + scale_bytes),
        dtype=torch.int8,
    )
    raw.view(torch.bfloat16)[0] = 7
    raw.view(torch.float32)[(prefix_bytes + row_bytes) // torch.float32.itemsize] = 3.5
    tail = torch.as_strided(
        raw[prefix_bytes:].view(torch.float8_e4m3fnuz),
        size=(num_blocks, k_per_block, head_dim),
        stride=(k_per_block * (row_bytes + scale_bytes), row_bytes + scale_bytes, 1),
    )
    compressor = SimpleNamespace()
    attn = SimpleNamespace(
        compress_ratio=4,
        kv_cache=tail,
        atom_vllm_unified_kv_prefix_bytes=prefix_bytes,
        atom_vllm_unified_kv_swa_pages=swa_pages,
        atom_vllm_unified_kv_swa_dtype=torch.bfloat16,
        atom_vllm_compressed_scale_bytes_per_page=scale_bytes,
        kv_cache_torch_dtype=torch.bfloat16,
        head_dim=head_dim,
        max_num_reqs=1,
        prefix="layers.0.attn",
        compressor=compressor,
    )
    DeepseekV4Attention.post_bind_kv_cache(attn, tail)

    state = DeepseekV4RocmAtomModelState.__new__(DeepseekV4RocmAtomModelState)
    state._enable_atom_unified_kv_from_vllm = True
    state._atom_unified_kv_from_vllm_bound = False
    state._atom_unified_kv_buffers = None
    state.max_num_reqs = 1
    state.win_with_spec = 1
    state.swa_pages = swa_pages
    state.head_dim = head_dim
    state.dtype = torch.bfloat16
    state.k1_csa = k_per_block
    state.k2_hca = 1
    state._iter_active_attn_modules = lambda: [(0, attn)]
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    assert state._try_bind_atom_unified_kv_from_vllm(kv_cache_config)

    buffers = state._atom_unified_kv_buffers
    assert isinstance(buffers, DeepseekV4RocmAtomUnifiedKVBuffers)
    assert buffers.active_layer_ids == (0,)
    assert buffers.unified_kv == ()
    assert buffers.unified_kv_by_layer == {}
    assert buffers.compressed_kv_cache[0] is tail
    assert buffers.compressed_kv_scales[0] is attn.atom_split_kv_scales
    assert buffers.compressed_kv_layout[0] == "dense"
    assert attn.atom_compressed_kv_cache is tail
    assert attn.atom_split_kv_layout == "dense"
    assert float(attn.atom_swa_kv[0, 0, 0]) == 7
    assert float(buffers.compressed_kv_scales[0][0, 0]) == 3.5
    assert compressor.atom_kv_cache is tail
    assert compressor.atom_kv_scales is attn.atom_split_kv_scales
    assert compressor.atom_kv_layout == "dense"


def test_deepseek_v4_model_state_refuses_vllm_owned_kv_fallback():
    from vllm.models.deepseek_v4.amd.model_state import (
        DeepseekV4RocmAtomModelState,
    )

    state = DeepseekV4RocmAtomModelState.__new__(DeepseekV4RocmAtomModelState)
    state._enable_atom_unified_kv = True
    state._enable_atom_unified_kv_from_vllm = True
    state._atom_unified_kv_buffers = None
    state._try_bind_atom_unified_kv_from_vllm = lambda _config: False
    kv_cache_config = KVCacheConfig(
        num_blocks=1,
        kv_cache_tensors=[],
        kv_cache_groups=[],
    )

    with pytest.raises(RuntimeError, match="Refusing to fall back"):
        state._maybe_allocate_atom_unified_kv(kv_cache_config)


def test_deepseek_v4_legacy_metadata_carries_unified_kv_layout_bundle():
    from vllm.models.deepseek_v4.amd.model_state import (
        DeepseekV4RocmAtomModelState,
        DeepseekV4RocmAtomUnifiedKVBuffers,
    )

    tail = torch.zeros((1, 4, 584), dtype=torch.uint8)
    buffers = DeepseekV4RocmAtomUnifiedKVBuffers(
        unified_kv=(),
        unified_kv_by_layer={},
        compressed_kv_cache={0: tail},
        compressed_kv_scales={0: None},
        compressed_kv_layout={0: "fp8_ds_mla"},
        active_layer_ids=(0,),
        num_blocks=1,
        swa_pages=1,
        k1_csa=4,
        k2_hca=1,
    )
    state = DeepseekV4RocmAtomModelState.__new__(DeepseekV4RocmAtomModelState)
    state.win_with_spec = 1
    state.swa_pages = 1
    state._atom_state_buffers = None
    state._atom_unified_kv_buffers = buffers
    state._atom_decode_buffers = None
    state._atom_prefill_buffers = None
    state._maybe_allocate_atom_unified_kv = lambda _config: None
    state._build_compress_plans_from_arrays = lambda *_args: None
    state._prepare_atom_decode_metadata = lambda _metadata: None

    state._state_slot_mapping = torch.empty(1, dtype=torch.int32)
    state._state_slot_mapping_cpu_tensor = torch.empty(1, dtype=torch.int32)
    state._state_slot_mapping_cpu = state._state_slot_mapping_cpu_tensor.numpy()
    state._batch_id_per_token = torch.empty(1, dtype=torch.int32)
    state._batch_id_per_token_cpu_tensor = torch.empty(1, dtype=torch.int32)
    state._batch_id_per_token_cpu = state._batch_id_per_token_cpu_tensor.numpy()
    state._chunk_start_per_seq = torch.empty(1, dtype=torch.int32)
    state._chunk_start_per_seq_cpu_tensor = torch.empty(1, dtype=torch.int32)
    state._chunk_start_per_seq_cpu = state._chunk_start_per_seq_cpu_tensor.numpy()
    state._positions_cpu = np.empty(1, dtype=np.int32)
    state._n_committed_csa_per_seq = torch.empty(1, dtype=torch.int32)
    state._n_committed_csa_per_seq_cpu_tensor = torch.empty(1, dtype=torch.int32)
    state._n_committed_csa_per_seq_cpu = (
        state._n_committed_csa_per_seq_cpu_tensor.numpy()
    )
    state._n_committed_hca_per_seq = torch.empty(1, dtype=torch.int32)
    state._n_committed_hca_per_seq_cpu_tensor = torch.empty(1, dtype=torch.int32)
    state._n_committed_hca_per_seq_cpu = (
        state._n_committed_hca_per_seq_cpu_tensor.numpy()
    )

    metadata = state.build_legacy_runner_metadata(
        num_actual_reqs=1,
        num_reqs=1,
        num_actual_tokens=1,
        num_tokens=1,
        positions=torch.zeros(1, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
        req_indices_cpu=np.zeros(1, dtype=np.int32),
        num_scheduled_tokens_cpu=np.ones(1, dtype=np.int32),
        num_computed_tokens_cpu=np.zeros(1, dtype=np.int32),
        kv_cache_config=KVCacheConfig(
            num_blocks=1,
            kv_cache_tensors=[],
            kv_cache_groups=[],
        ),
    )

    assert metadata.unified_kv_buffers is buffers
    assert metadata.unified_kv_buffers.compressed_kv_cache[0] is tail
    assert metadata.unified_kv_buffers.compressed_kv_scales[0] is None
    assert metadata.unified_kv_buffers.compressed_kv_layout[0] == "fp8_ds_mla"


def test_deepseek_v4_bind_split_only_unified_kv_bundle():
    from vllm.models.deepseek_v4.amd.model_state import (
        DeepseekV4RocmAtomModelState,
        DeepseekV4RocmAtomUnifiedKVBuffers,
    )

    swa = torch.zeros((1, 1, 8), dtype=torch.bfloat16)
    tail = torch.zeros((1, 4, 584), dtype=torch.uint8)
    compressor = SimpleNamespace()
    attn = SimpleNamespace(atom_swa_kv=swa, compressor=compressor)
    buffers = DeepseekV4RocmAtomUnifiedKVBuffers(
        unified_kv=(),
        unified_kv_by_layer={},
        compressed_kv_cache={0: tail},
        compressed_kv_scales={0: None},
        compressed_kv_layout={0: "fp8_ds_mla"},
        active_layer_ids=(0,),
        num_blocks=1,
        swa_pages=1,
        k1_csa=4,
        k2_hca=1,
    )
    state = DeepseekV4RocmAtomModelState.__new__(DeepseekV4RocmAtomModelState)
    state.max_num_reqs = 1
    state.win_with_spec = 1
    state.head_dim = 8
    state._iter_active_attn_modules = lambda: [(0, attn)]

    state._bind_atom_unified_kv_buffers(buffers)

    assert not hasattr(attn, "atom_unified_kv")
    assert attn.atom_split_kv_swa is swa
    assert attn.atom_split_kv_compressed is tail
    assert attn.atom_split_kv_scales is None
    assert attn.atom_split_kv_layout == "fp8_ds_mla"
    assert compressor.atom_kv_cache is tail
    assert compressor.atom_kv_scales is None
    assert compressor.atom_kv_layout == "fp8_ds_mla"


def test_deepseek_v4_bind_mixed_split_and_homogeneous_bundle():
    from vllm.models.deepseek_v4.amd.model_state import (
        DeepseekV4RocmAtomModelState,
        DeepseekV4RocmAtomUnifiedKVBuffers,
    )

    split_swa = torch.zeros((1, 1, 8), dtype=torch.bfloat16)
    split_tail = torch.zeros((1, 4, 584), dtype=torch.uint8)
    dense_unified = torch.zeros((5, 8), dtype=torch.bfloat16)
    dense_unified[0, 0] = 17
    dense_tail = dense_unified[1:].view(1, 4, 8)
    split_compressor = SimpleNamespace()
    dense_compressor = SimpleNamespace()
    split_attn = SimpleNamespace(atom_swa_kv=split_swa, compressor=split_compressor)
    dense_attn = SimpleNamespace(compressor=dense_compressor)
    buffers = DeepseekV4RocmAtomUnifiedKVBuffers(
        unified_kv=(dense_unified,),
        unified_kv_by_layer={1: dense_unified},
        compressed_kv_cache={0: split_tail, 1: dense_tail},
        compressed_kv_scales={0: None, 1: None},
        compressed_kv_layout={0: "fp8_ds_mla", 1: "dense"},
        active_layer_ids=(0, 1),
        num_blocks=1,
        swa_pages=1,
        k1_csa=4,
        k2_hca=1,
    )
    state = DeepseekV4RocmAtomModelState.__new__(DeepseekV4RocmAtomModelState)
    state.max_num_reqs = 1
    state.win_with_spec = 1
    state.head_dim = 8
    state._iter_active_attn_modules = lambda: [(0, split_attn), (1, dense_attn)]

    state._bind_atom_unified_kv_buffers(buffers)

    assert not hasattr(split_attn, "atom_unified_kv")
    assert split_attn.atom_split_kv_layout == "fp8_ds_mla"
    assert split_attn.atom_split_kv_compressed is split_tail
    assert split_compressor.atom_kv_layout == "fp8_ds_mla"
    assert dense_attn.atom_unified_kv is dense_unified
    assert dense_attn.atom_swa_kv.shape == (1, 1, 8)
    assert float(dense_attn.atom_swa_kv[0, 0, 0]) == 17
    assert dense_attn.atom_split_kv_layout == "dense"
    assert dense_attn.atom_split_kv_compressed is dense_tail
    assert dense_compressor.atom_kv_cache is dense_tail


def test_deepseek_v4_model_state_side_allocates_when_not_vllm_owned():
    from vllm.models.deepseek_v4.amd.model_state import (
        DeepseekV4RocmAtomModelState,
        DeepseekV4RocmAtomUnifiedKVBuffers,
    )

    head_dim = 8
    num_blocks = 2
    k_per_block = 4
    compressor = SimpleNamespace()
    attn = SimpleNamespace(compress_ratio=4, compressor=compressor)
    state = DeepseekV4RocmAtomModelState.__new__(DeepseekV4RocmAtomModelState)
    state._enable_atom_unified_kv = True
    state._enable_atom_unified_kv_from_vllm = False
    state._atom_unified_kv_buffers = None
    state._try_bind_atom_unified_kv_from_vllm = lambda _config: False
    state._iter_active_attn_modules = lambda: [(0, attn)]
    state.device = torch.device("cpu")
    state.max_num_reqs = 1
    state.win_with_spec = 2
    state.swa_pages = 2
    state.head_dim = head_dim
    state.dtype = torch.bfloat16
    state.k1_csa = k_per_block
    state.k2_hca = 1
    spec = DeepseekV4AtomMLAAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=head_dim,
        dtype=torch.bfloat16,
        compress_ratio=4,
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["atom"], spec)],
    )

    state._maybe_allocate_atom_unified_kv(kv_cache_config)

    buffers = state._atom_unified_kv_buffers
    assert isinstance(buffers, DeepseekV4RocmAtomUnifiedKVBuffers)
    assert buffers.active_layer_ids == (0,)
    assert buffers.num_blocks == num_blocks
    assert buffers.swa_pages == state.swa_pages
    assert buffers.compressed_kv_cache[0].shape == (
        num_blocks,
        k_per_block,
        head_dim,
    )
    assert buffers.compressed_kv_scales[0] is None
    assert buffers.compressed_kv_layout[0] == "dense"
    assert buffers.unified_kv_by_layer[0] is attn.atom_unified_kv
    assert attn.atom_unified_kv.shape == (
        state.swa_pages + num_blocks * k_per_block,
        head_dim,
    )
    assert attn.atom_swa_kv.shape == (1, 2, head_dim)
    assert attn.atom_split_kv_compressed is buffers.compressed_kv_cache[0]
    assert attn.atom_split_kv_layout == "dense"
    assert compressor.atom_kv_cache is buffers.compressed_kv_cache[0]
    assert compressor.atom_kv_scales is None
    assert compressor.atom_kv_layout == "dense"
