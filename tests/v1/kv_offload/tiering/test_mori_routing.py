# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.kv_offload.tiering.mori.placement import encode_umbp_event_key
from vllm.v1.kv_offload.tiering.mori.routing import (
    MoriCacheAwareRouter,
    MoriReplica,
    hash_text_prefix,
)


class FakePlacement:
    def __init__(self, matches):
        self.matches = matches
        self.queries = []
        self._key_prefix = "prefix:"

    def encode_keys(self, hashes, group_idx):
        return [encode_umbp_event_key(h, group_idx, self._key_prefix) for h in hashes]

    def match(self, hashes, group_idx=0, count_as_hit=False):
        self.queries.append((list(hashes), group_idx, count_as_hit))
        return self.matches


def test_hash_text_prefix_matches_chained_vllm_hashes():
    hashes = hash_text_prefix(list(range(10)), block_size=4)
    assert len(hashes) == 2
    assert hashes == hash_text_prefix(list(range(8)), block_size=4)
    assert hashes != hash_text_prefix(list(range(1, 9)), block_size=4)


def test_router_prefers_longest_consecutive_hbm_prefix():
    hashes = hash_text_prefix(list(range(12)), block_size=4)
    placement = FakePlacement([])
    keys = placement.encode_keys(hashes, 0)
    placement.matches = [
        SimpleNamespace(node_id="a", hashes_by_tier={"HBM": keys[:2]}),
        SimpleNamespace(node_id="b", hashes_by_tier={"HBM": keys}),
    ]
    router = MoriCacheAwareRouter(
        placement, [MoriReplica("a", "http://a"), MoriReplica("b", "http://b")]
    )

    assert router.select(hashes).node_id == "b"
    assert placement.queries[-1][2] is True


def test_router_stops_at_first_prefix_gap_and_uses_tier_cost():
    hashes = hash_text_prefix(list(range(12)), block_size=4)
    placement = FakePlacement([])
    keys = placement.encode_keys(hashes, 0)
    placement.matches = [
        SimpleNamespace(node_id="a", hashes_by_tier={"HBM": [keys[0], keys[2]]}),
        SimpleNamespace(node_id="b", hashes_by_tier={"DRAM": keys[:2]}),
    ]
    router = MoriCacheAwareRouter(
        placement, [MoriReplica("a", "http://a"), MoriReplica("b", "http://b")]
    )

    assert router.select(hashes).node_id == "b"


def test_router_load_balances_when_there_are_no_cache_hits():
    placement = FakePlacement([])
    replicas = [MoriReplica("a", "http://a"), MoriReplica("b", "http://b")]
    router = MoriCacheAwareRouter(placement, replicas)

    first = router.select([])
    second = router.select([])
    assert {first.node_id, second.node_id} == {"a", "b"}
    router.finish(first)
    router.finish(second)
