# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from typing import Any, cast

from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVEventBatch,
)
from vllm.v1.kv_offload.tiering.mori.placement import (
    MoriPlacementClient,
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
