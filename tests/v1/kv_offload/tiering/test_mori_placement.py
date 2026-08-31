# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from typing import Any, cast

import msgspec

from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_GPU,
    MEDIUM_STORAGE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.tiering.mori.placement import (
    MoriPhysicalPlacementResolver,
    MoriPlacementClient,
    encode_umbp_event_key,
    enrich_umbp_logical_placement,
)


class FakeTierType:
    HBM = "HBM"
    DRAM = "DRAM"
    SSD = "SSD"


class FakeLocation:
    def __init__(self, node_id, tier, size=4096, peer_address="peer:1"):
        self.node_id = node_id
        self.tier = tier
        self.size = size
        self.peer_address = peer_address


class FakeMasterClient:
    def __init__(self):
        self.reported = []
        self.revoked = []
        self.cleared = []
        self.matches = []
        self.locations = {}
        self.location_queries = []

    def report_external_kv_blocks(self, node_id, keys, tier):
        self.reported.append((node_id, keys, tier))

    def revoke_external_kv_blocks(self, node_id, keys, tier):
        self.revoked.append((node_id, keys, tier))

    def revoke_all_external_kv_blocks_at_tier(self, node_id, tier):
        self.cleared.append((node_id, tier))

    def match_external_kv(self, keys, count_as_hit=False):
        self.matches.append((keys, count_as_hit))
        return ["match"]

    def get_external_kv_hit_counts(self, keys):
        return keys

    def batch_lookup_locations(self, keys):
        self.location_queries.append(keys)
        return [self.locations.get(key, []) for key in keys]


def _install_fake_mori(monkeypatch):
    module = types.ModuleType("mori.cpp")
    cast(Any, module).UMBPTierType = FakeTierType
    monkeypatch.setitem(sys.modules, "mori.cpp", module)


def _stored(block_hash=b"hash", medium=MEDIUM_GPU, group_idx=2):
    return BlockStored(
        block_hashes=[block_hash],
        parent_block_hash=None,
        token_ids=[],
        block_size=16,
        lora_id=None,
        medium=medium,
        lora_name=None,
        group_idx=group_idx,
    )


def test_placement_client_reports_and_revokes_by_medium(monkeypatch):
    _install_fake_mori(monkeypatch)
    master = FakeMasterClient()
    placement = MoriPlacementClient("master:15558", "node-0", "vllm:test:", master)
    gpu = _stored()
    cpu = _stored(medium=MEDIUM_CPU)

    placement.process_batch(KVEventBatch(0.0, [gpu, cpu]))
    placement.process_event(BlockRemoved([b"hash"], medium=MEDIUM_GPU, group_idx=2))

    key = encode_umbp_event_key(b"hash", 2, "vllm:test:")
    assert master.reported == [
        ("node-0", [key], FakeTierType.HBM),
        ("node-0", [key], FakeTierType.DRAM),
    ]
    assert master.revoked == [("node-0", [key], FakeTierType.HBM)]


def test_placement_client_reference_counts_duplicate_events(monkeypatch):
    _install_fake_mori(monkeypatch)
    master = FakeMasterClient()
    placement = MoriPlacementClient("master:15558", "node-0", "prefix:", master)
    stored = _stored()
    removed = BlockRemoved([b"hash"], medium=MEDIUM_GPU, group_idx=2)

    placement.process_event(stored)
    placement.process_event(stored)
    placement.process_event(removed)
    assert not master.revoked
    placement.process_event(removed)
    assert len(master.revoked) == 1


def test_placement_client_clear_and_query(monkeypatch):
    _install_fake_mori(monkeypatch)
    master = FakeMasterClient()
    placement = MoriPlacementClient("master:15558", "node-0", "prefix:", master)

    placement.process_event(AllBlocksCleared())
    assert master.cleared == [
        ("node-0", FakeTierType.HBM),
        ("node-0", FakeTierType.DRAM),
    ]
    assert placement.match([b"hash"], group_idx=3, count_as_hit=True) == ["match"]
    expected = encode_umbp_event_key(b"hash", 3, "prefix:")
    assert master.matches == [([expected], True)]
    assert placement.get_hit_counts([b"hash"], group_idx=3) == [expected]


def test_legacy_integer_hashes_have_unambiguous_namespace():
    key = encode_umbp_event_key(42, 1, "prefix:")
    assert key == "prefix:legacy-int:000000000000002a:00000001"


def test_byte_event_key_matches_mori_offload_key():
    offload_key = make_offload_key(b"hash", 2)
    assert encode_umbp_event_key(b"hash", 2, "prefix:") == (
        f"prefix:{bytes(offload_key).hex()}"
    )


def test_logical_umbp_placement_survives_kv_event_wire_round_trip():
    batch = KVEventBatch(
        1.0,
        [
            _stored(medium=MEDIUM_STORAGE),
            BlockRemoved([b"hash"], medium=MEDIUM_STORAGE, group_idx=2),
            _stored(medium=MEDIUM_CPU),
        ],
    )
    encoded = msgspec.msgpack.encode(enrich_umbp_logical_placement(batch))
    decoded = msgspec.msgpack.decode(encoded, type=KVEventBatch)

    assert [event.storage_tier for event in decoded.events] == ["UMBP", "UMBP", None]
    assert all(event.source_node is None for event in decoded.events)
    assert all(event.locality is None for event in decoded.events)


def test_physical_placement_uses_chunk_key_and_prefers_local_best_tier(monkeypatch):
    _install_fake_mori(monkeypatch)
    master = FakeMasterClient()
    placement = MoriPlacementClient("master:15558", "node-0", "prefix:", master)
    chunk = _stored(block_hash=b"last", medium=MEDIUM_STORAGE, group_idx=2)
    chunk = msgspec.structs.replace(chunk, block_hashes=[b"first", b"last"])
    key = encode_umbp_event_key(b"last", 2, "prefix:")
    master.locations[key] = [
        FakeLocation("node-1", FakeTierType.DRAM),
        FakeLocation("node-0", FakeTierType.SSD),
        FakeLocation("node-0", FakeTierType.DRAM),
    ]
    resolver = MoriPhysicalPlacementResolver(
        placement,
        "node-0",
        lookup_timeout_s=0,
        bandwidth_bps={("LOCAL", "DRAM"): 12.5e9},
    )

    enriched = resolver.enrich_batch(KVEventBatch(0.0, [chunk]))

    assert master.location_queries == [[key]]
    event = enriched.events[0]
    assert event.storage_tier == "DRAM"
    assert event.locality == "LOCAL"
    assert event.source_node == "node-0"
    assert event.estimated_bandwidth_bps == 12.5e9


def test_physical_placement_cache_splits_removals_by_source(monkeypatch):
    _install_fake_mori(monkeypatch)
    master = FakeMasterClient()
    placement = MoriPlacementClient("master:15558", "reader", "prefix:", master)
    first = _stored(block_hash=b"a", medium=MEDIUM_STORAGE, group_idx=3)
    second = _stored(block_hash=b"b", medium=MEDIUM_STORAGE, group_idx=3)
    master.locations[encode_umbp_event_key(b"a", 3, "prefix:")] = [
        FakeLocation("node-a", FakeTierType.DRAM)
    ]
    master.locations[encode_umbp_event_key(b"b", 3, "prefix:")] = [
        FakeLocation("node-b", FakeTierType.SSD)
    ]
    resolver = MoriPhysicalPlacementResolver(placement, "reader", lookup_timeout_s=0)
    resolver.enrich_batch(KVEventBatch(0.0, [first, second]))

    removed = BlockRemoved([b"a", b"b"], medium=MEDIUM_STORAGE, group_idx=3)
    events = resolver.enrich_batch(KVEventBatch(1.0, [removed])).events

    assert len(events) == 2
    assert [
        (event.block_hashes, event.storage_tier, event.source_node) for event in events
    ] == [
        ([b"a"], "DRAM", "node-a"),
        ([b"b"], "SSD", "node-b"),
    ]
    assert all(event.locality == "REMOTE" for event in events)


def test_physical_placement_falls_back_to_logical_umbp(monkeypatch):
    _install_fake_mori(monkeypatch)
    master = FakeMasterClient()
    placement = MoriPlacementClient("master:15558", "node-0", "prefix:", master)
    resolver = MoriPhysicalPlacementResolver(placement, "node-0", lookup_timeout_s=0)

    event = resolver.enrich_batch(
        KVEventBatch(0.0, [_stored(medium=MEDIUM_STORAGE)])
    ).events[0]

    assert event.storage_tier == "UMBP"
    assert event.locality is None
    assert event.source_node is None


def test_physical_placement_skips_lossy_integer_hashes(monkeypatch):
    _install_fake_mori(monkeypatch)
    master = FakeMasterClient()
    placement = MoriPlacementClient("master:15558", "node-0", "prefix:", master)
    resolver = MoriPhysicalPlacementResolver(placement, "node-0", lookup_timeout_s=0)

    event = resolver.enrich_batch(
        KVEventBatch(0.0, [_stored(block_hash=42, medium=MEDIUM_STORAGE)])
    ).events[0]

    assert master.location_queries == []
    assert event.storage_tier == "UMBP"
    assert event.locality is None
    assert event.source_node is None
