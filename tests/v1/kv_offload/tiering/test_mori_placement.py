# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from typing import Any, cast

import msgspec
import pytest

from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_GPU,
    MEDIUM_STORAGE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.v1.kv_offload.tiering.mori.placement import (
    MoriPlacementClient,
    MoriPlacementReconciler,
    encode_umbp_event_key,
)


class FakeTierType:
    HBM = "HBM"
    DRAM = "DRAM"


class FakeMasterClient:
    def __init__(self):
        self.reported = []
        self.revoked = []
        self.cleared = []
        self.matches = []

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


class FakePlacement:
    def __init__(self, node_id, tier, size=1024):
        self.node_id = node_id
        self.tier = tier
        self.size = size
        self.peer_address = f"{node_id}:1234"


class FakeInspectClient:
    def __init__(self):
        self.results = []
        self.requests = []

    def batch_inspect(self, keys):
        self.requests.append(keys)
        return self.results[: len(keys)]


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


def test_reconciler_refreshes_and_migrates_authoritative_placement():
    client = FakeInspectClient()
    reconciler = MoriPlacementReconciler(
        client,
        node_id="node-a",
        key_prefix="prefix:",
        bandwidth_bps={"LOCAL:DRAM": 4e9, "REMOTE:SSD": 1e9},
    )
    stored = _stored(medium=MEDIUM_STORAGE)
    reconciler.observe_batch(KVEventBatch(0.0, [stored]))

    client.results = [FakePlacement("node-a", "DRAM")]
    first = reconciler.reconcile()
    assert len(first) == 1
    assert isinstance(first[0], BlockStored)
    assert first[0].storage_tier == "DRAM"
    assert first[0].locality == "LOCAL"
    assert first[0].estimated_bandwidth_bps == 4e9

    refreshed = reconciler.reconcile()
    assert len(refreshed) == 1
    assert isinstance(refreshed[0], BlockStored)

    client.results = [FakePlacement("node-b", "SSD")]
    migrated = reconciler.reconcile()
    assert [type(event) for event in migrated] == [BlockRemoved, BlockStored]
    assert migrated[0].storage_tier == "DRAM"
    assert migrated[1].storage_tier == "SSD"
    assert migrated[1].source_node == "node-b"
    assert migrated[1].locality == "REMOTE"
    assert migrated[1].estimated_bandwidth_bps == 1e9


def test_reconciler_removes_disappeared_and_untracked_placements():
    client = FakeInspectClient()
    reconciler = MoriPlacementReconciler(client, "node-a", "prefix:")
    stored = _stored(medium=MEDIUM_STORAGE)
    reconciler.observe_batch(KVEventBatch(0.0, [stored]))
    client.results = [FakePlacement("node-a", "DRAM")]
    reconciler.reconcile()

    client.results = [None]
    missing = reconciler.reconcile()
    assert len(missing) == 1
    assert isinstance(missing[0], BlockRemoved)

    client.results = [FakePlacement("node-a", "DRAM")]
    reconciler.reconcile()
    reconciler.observe_batch(
        KVEventBatch(
            0.0,
            [BlockRemoved([b"hash"], medium=MEDIUM_STORAGE, group_idx=2)],
        )
    )
    client.results = []
    removed = reconciler.reconcile()
    assert len(removed) == 1
    assert isinstance(removed[0], BlockRemoved)


def test_reconciler_bounds_master_request_batches():
    client = FakeInspectClient()
    reconciler = MoriPlacementReconciler(client, "node-a", "prefix:", max_batch_size=2)
    events = [
        _stored(block_hash=f"hash-{index}".encode(), medium=MEDIUM_STORAGE)
        for index in range(5)
    ]
    reconciler.observe_batch(KVEventBatch(0.0, events))
    client.results = [None, None]

    reconciler.reconcile()
    assert [len(request) for request in client.requests] == [2, 2, 1]


def test_reconciler_rejects_invalid_batch_size():
    with pytest.raises(ValueError, match="max_batch_size"):
        MoriPlacementReconciler(
            FakeInspectClient(), "node-a", "prefix:", max_batch_size=0
        )


def test_reconciled_placement_survives_kv_event_wire_round_trip():
    client = FakeInspectClient()
    reconciler = MoriPlacementReconciler(client, "node-a", "prefix:")
    reconciler.observe_batch(KVEventBatch(0.0, [_stored(medium=MEDIUM_STORAGE)]))
    client.results = [FakePlacement("node-b", "SSD")]

    encoded = msgspec.msgpack.encode(KVEventBatch(1.0, reconciler.reconcile()))
    decoded = msgspec.msgpack.decode(encoded, type=KVEventBatch)

    assert len(decoded.events) == 1
    event = decoded.events[0]
    assert isinstance(event, BlockStored)
    assert event.medium == MEDIUM_STORAGE
    assert event.storage_tier == "SSD"
    assert event.source_node == "node-b"
    assert event.locality == "REMOTE"


def test_reconciler_rejects_non_positive_bandwidth():
    with pytest.raises(ValueError, match="bandwidth_bps"):
        MoriPlacementReconciler(
            FakeInspectClient(),
            "node-a",
            "prefix:",
            bandwidth_bps={"LOCAL:DRAM": 0},
        )
