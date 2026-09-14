# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Allocation-to-object tests with CPU buffers, not a native transport test.

The mapper must move only one group's requested block, preserving logical KV
across physical layouts and allocation capacities. Byte-level round trips and
guard regions cheaply catch incorrect strides, aliases and cache identities.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.unit.test_umbp_store import NativeStore
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.layout import (
    CacheTopology,
    UMBPLayout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import UMBPStore
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheLayout,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
    compute_layout_strides,
    create_kv_cache_views,
)


def attention_spec(**kwargs):
    return FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=2, dtype=torch.float16, **kwargs
    )


def allocation(groups, *, layout=KVCacheLayout.LBHNC, capacity=3, kernel_size=None):
    placements = []
    layer_specs = {}
    for group in groups:
        for name in group.layer_names:
            layer_specs[name] = (
                group.kv_cache_spec.kv_cache_specs[name]
                if isinstance(group.kv_cache_spec, UniformTypeKVCacheSpecs)
                else group.kv_cache_spec
            )
        batches: list[list[str]] = []
        for name in group.layer_names:
            for batch in batches:
                if layer_specs[batch[0]] == layer_specs[name]:
                    batch.append(name)
                    break
            else:
                batches.append([name])
        offset = 0
        for names in batches:
            spec = layer_specs[names[0]]
            strides = compute_layout_strides(spec, capacity, len(names), layout)
            placements.append(
                KVCacheTensor(0, names, strides[0], strides[1], offset=offset)
            )
            offset += capacity * len(names) * spec.page_size_bytes
    size = max(
        p.offset + capacity * len(p.layers) * layer_specs[p.layers[0]].page_size_bytes
        for p in placements
    )
    for placement in placements:
        placement.size = size
    raw = torch.full((size + 64,), -99, dtype=torch.int8)
    config = KVCacheConfig(capacity, placements, groups, kv_cache_layout=layout.name)
    caches: dict[str, torch.Tensor] = {}
    for placement in placements:
        name = placement.layers[0]
        spec = layer_specs[name]
        if spec.has_layer_views:
            views = create_kv_cache_views(
                raw, spec, capacity, layout, placement, kernel_size
            )
            caches.update(zip(placement.layers, views, strict=True))
        else:
            caches.update((name, raw) for name in placement.layers)
    return config, raw, caches


def byte_tensor(obj):
    """Logical expected bytes, independent of pointer slice generation."""
    return bytes(obj.contiguous().view(torch.uint8).flatten().tolist())


@pytest.mark.parametrize("source_layout", list(KVCacheLayout))
@pytest.mark.parametrize("target_layout", list(KVCacheLayout))
def test_roundtrip_preserves_logical_kv_and_neighbor_blocks(
    source_layout, target_layout
):
    groups = [KVCacheGroupSpec(["layer.1", "layer.0"], attention_spec())]
    source_config, _, source = allocation(groups, layout=source_layout)
    target_config, target_raw, target = allocation(
        groups, layout=target_layout, capacity=5
    )
    for index, tensor in enumerate(source.values()):
        tensor.copy_(torch.arange(tensor.numel()).reshape(tensor.shape) + 100 * index)
    before = target_raw.clone()
    producer = UMBPLayout(source_config, source, CacheTopology(tp_size=8))
    consumer = UMBPLayout(target_config, target, CacheTopology(tp_size=8))
    assert producer.identity == consumer.identity
    native = NativeStore()
    store = UMBPStore(native, SimpleNamespace(CPU="cpu", GPU="gpu"))
    try:
        for region in (*producer.regions, *consumer.regions):
            store.register_region(region)
        assert len(producer.regions) == len(consumer.regions) == 1
        src = producer.block_object("block", group_id=0, block_id=1)
        dst = consumer.block_object("block", group_id=0, block_id=3)
        assert store.store((src,)).result(5) == (True,)
        assert native.data["block"] == b"".join(
            byte_tensor(source[name][1]) for name in sorted(source)
        )
        assert store.load((dst,)).result(5) == (True,)
        changed = torch.zeros_like(target_raw, dtype=torch.bool)
        for part in dst.slices:
            changed[part.offset : part.offset + part.size] = True
        assert torch.equal(target_raw[~changed], before[~changed])
        for name in source:
            torch.testing.assert_close(target[name][3], source[name][1])
    finally:
        store.close()


@pytest.mark.parametrize("layout", [KVCacheLayout.LBHNC, KVCacheLayout.LBNHC])
def test_split_kernel_blocks_have_the_same_object_as_unsplit_blocks(layout):
    groups = [KVCacheGroupSpec(["layer"], attention_spec())]
    full_config, _, full = allocation(groups, layout=layout)
    split_config, _, split = allocation(groups, layout=layout, kernel_size=2)
    full["layer"].copy_(
        torch.arange(full["layer"].numel()).reshape(full["layer"].shape)
    )
    # Set the same logical H/N/C values, despite the kernel's smaller N axis.
    for block in range(3):
        for half in range(2):
            split["layer"][block * 2 + half].copy_(
                full["layer"][block, :, half * 2 : (half + 1) * 2]
            )
    native = NativeStore()
    store = UMBPStore(native, SimpleNamespace(CPU="cpu", GPU="gpu"))
    layouts = [
        UMBPLayout(c, t, CacheTopology())
        for c, t in [(full_config, full), (split_config, split)]
    ]
    try:
        assert layouts[0].identity == layouts[1].identity
        for index, mapped in enumerate(layouts):
            store.register_region(mapped.regions[0])
            assert store.store(
                (mapped.block_object(str(index), group_id=0, block_id=1),)
            ).result(5) == (True,)
        assert native.data["0"] == native.data["1"]
    finally:
        store.close()


def test_packed_layers_and_aliased_hybrid_groups_preserve_padding():
    spec = attention_spec(page_size_padded=128)
    groups = [
        KVCacheGroupSpec(["a", "b"], spec),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=4,
                shapes=((2, 3), (3, 2)),
                dtypes=(torch.float32, torch.float16),
                page_size_padded=128,
            ),
        ),
    ]
    config, raw, _ = allocation(groups)
    config.kv_cache_layout = "BLHNC"
    config.kv_cache_tensors = [
        KVCacheTensor(768, ["a", "b"], 128, 256),
        KVCacheTensor(768, ["mamba"], 128, 256),
    ]
    caches: dict[str, torch.Tensor] = {}
    for group, placement in zip(groups, config.kv_cache_tensors, strict=True):
        caches.update(
            zip(
                placement.layers,
                create_kv_cache_views(
                    raw,
                    group.kv_cache_spec,
                    3,
                    KVCacheLayout.BLHNC,
                    placement,
                ),
                strict=True,
            )
        )
    mapped = UMBPLayout(config, caches, CacheTopology())
    assert len(mapped.regions) == 1
    attn = mapped.block_object("attention", group_id=0, block_id=1)
    mamba = mapped.block_object("state", group_id=1, block_id=2)
    assert attn.size == 2 * spec.unpadded_page_size_bytes
    assert mamba.size == groups[1].kv_cache_spec.state_content_size_bytes
    assert [(s.offset, s.size) for s in attn.slices] == [(256, 64), (384, 64)]
    assert [(s.offset, s.size) for s in mamba.slices] == [(512, 36)]


def test_uniform_group_uses_each_layers_actual_spec_and_semantic_identity():
    specs = {
        "attention": attention_spec(),
        "mla": MLAAttentionSpec(
            block_size=4,
            num_kv_heads=1,
            head_size=4,
            dtype=torch.float16,
        ),
    }
    wrapper = UniformTypeKVCacheSpecs(block_size=4, kv_cache_specs=specs)
    config, _, caches = allocation([KVCacheGroupSpec(list(specs), wrapper)])
    mapped = UMBPLayout(config, caches, CacheTopology())
    obj = mapped.block_object("hybrid", group_id=0, block_id=1)
    assert obj.size == sum(s.real_page_size_bytes for s in specs.values())
    altered = replace(
        config,
        kv_cache_groups=[
            KVCacheGroupSpec(
                list(specs),
                replace(
                    wrapper,
                    kv_cache_specs={
                        **specs,
                        "attention": SlidingWindowSpec(
                            block_size=4,
                            num_kv_heads=2,
                            head_size=2,
                            dtype=torch.float16,
                            sliding_window=128,
                        ),
                    },
                ),
            )
        ],
    )
    # Window semantics change identity even if every byte/stride is unchanged.
    assert UMBPLayout(altered, caches, CacheTopology()).identity != mapped.identity
    assert (
        UMBPLayout(config, caches, CacheTopology(tp_size=2)).identity != mapped.identity
    )


@pytest.mark.parametrize(
    "fault", ["missing", "offset", "stride", "capacity", "unresolved"]
)
def test_invalid_allocation_is_rejected_before_native_registration(fault):
    config, raw, caches = allocation([KVCacheGroupSpec(["layer"], attention_spec())])
    if fault == "missing":
        caches = {}
    elif fault == "offset":
        caches["layer"] = caches["layer"].as_strided(
            caches["layer"].shape, caches["layer"].stride(), storage_offset=1
        )
    elif fault == "stride":
        caches["layer"] = caches["layer"].transpose(1, 2)
    elif fault == "capacity":
        config.kv_cache_tensors[0].size = raw.numel() + 1
    else:
        config.kv_cache_layout = None
    with pytest.raises(ValueError):
        UMBPLayout(config, caches, CacheTopology())


@pytest.mark.parametrize(
    "group_id,block_id", [(0, -1), (0, 3), (0, True), (-1, 0), (True, 0), (1, 0)]
)
def test_invalid_or_disabled_group_and_block_ids_are_rejected(group_id, block_id):
    config, _, caches = allocation(
        [
            KVCacheGroupSpec(["layer"], attention_spec()),
            KVCacheGroupSpec(["replica"], attention_spec(), enable_kv_transfer=False),
        ]
    )
    mapped = UMBPLayout(config, caches, CacheTopology())
    with pytest.raises(ValueError):
        mapped.block_object("bad", group_id=group_id, block_id=block_id)
