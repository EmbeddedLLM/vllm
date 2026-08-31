# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMBP control-plane adapter for vLLM KV cache events."""

import time
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import msgspec

from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_GPU,
    MEDIUM_STORAGE,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import ExternalBlockHash

logger = init_logger(__name__)


@dataclass(frozen=True)
class UMBPPhysicalPlacement:
    """Physical UMBP source selected for one vLLM storage event."""

    storage_tier: str
    source_node: str
    locality: str
    estimated_bandwidth_bps: float | None = None


def encode_umbp_event_key(
    block_hash: ExternalBlockHash,
    group_idx: int | None,
    key_prefix: str,
) -> str:
    """Encode a wire-format vLLM block hash as a UMBP placement key."""
    group_bytes = (group_idx or 0).to_bytes(4, "big", signed=False)
    if isinstance(block_hash, bytes):
        return f"{key_prefix}{(block_hash + group_bytes).hex()}"
    return f"{key_prefix}legacy-int:{block_hash:016x}:{group_bytes.hex()}"


class MoriPlacementClient:
    """Forward vLLM GPU/CPU KV events to UMBP's placement directory.

    The client is deliberately independent of the UMBP byte-store path. It
    can run in a router or sidecar process and consume event batches published
    by one vLLM engine. The corresponding UMBP data client must already have
    registered ``node_id`` with the master.
    """

    def __init__(
        self,
        master_address: str,
        node_id: str,
        key_prefix: str,
        client: Any | None = None,
    ) -> None:
        if not node_id:
            raise ValueError("node_id must not be empty")
        if client is None:
            try:
                from mori.cpp import UMBPMasterClient
            except ImportError as exc:
                raise ImportError(
                    "UMBP placement forwarding requires amd-mori built with "
                    "BUILD_UMBP=ON"
                ) from exc
            client = UMBPMasterClient(master_address)
        self._client = client
        self._node_id = node_id
        self._key_prefix = key_prefix
        self._refcounts: dict[str, Counter[str]] = {
            MEDIUM_GPU: Counter(),
            MEDIUM_CPU: Counter(),
        }

        from mori.cpp import UMBPTierType

        self._tier_types = {
            MEDIUM_GPU: UMBPTierType.HBM,
            MEDIUM_CPU: UMBPTierType.DRAM,
        }
        self._physical_tier_names = {
            UMBPTierType.HBM: "HBM",
            UMBPTierType.DRAM: "DRAM",
            UMBPTierType.SSD: "SSD",
        }

    def encode_keys(
        self,
        block_hashes: Iterable[ExternalBlockHash],
        group_idx: int | None,
    ) -> list[str]:
        return [
            encode_umbp_event_key(block_hash, group_idx, self._key_prefix)
            for block_hash in block_hashes
        ]

    def process_event(self, event: KVCacheEvent) -> None:
        if isinstance(event, AllBlocksCleared):
            for clear_medium, tier in self._tier_types.items():
                self._client.revoke_all_external_kv_blocks_at_tier(self._node_id, tier)
                self._refcounts[clear_medium].clear()
            return

        if not isinstance(event, (BlockStored, BlockRemoved)):
            return
        medium = event.medium
        if medium not in self._tier_types:
            return
        keys = self.encode_keys(event.block_hashes, event.group_idx)
        counts = self._refcounts[medium]
        tier = self._tier_types[medium]

        if isinstance(event, BlockStored):
            new_keys = [key for key in keys if counts[key] == 0]
            counts.update(keys)
            if new_keys:
                self._client.report_external_kv_blocks(self._node_id, new_keys, tier)
            return

        removed_keys = []
        for key in keys:
            if counts[key] <= 1:
                if counts[key]:
                    del counts[key]
                    removed_keys.append(key)
            else:
                counts[key] -= 1
        if removed_keys:
            self._client.revoke_external_kv_blocks(self._node_id, removed_keys, tier)

    def process_batch(self, batch: KVEventBatch) -> None:
        for event in batch.events:
            self.process_event(event)

    def match(
        self,
        block_hashes: Iterable[ExternalBlockHash],
        group_idx: int | None = 0,
        count_as_hit: bool = False,
    ) -> list[Any]:
        """Return UMBP nodes holding any of the supplied KV block hashes."""
        keys = self.encode_keys(block_hashes, group_idx)
        return self._client.match_external_kv(keys, count_as_hit=count_as_hit)

    def get_hit_counts(
        self,
        block_hashes: Iterable[ExternalBlockHash],
        group_idx: int | None = 0,
    ) -> list[Any]:
        keys = self.encode_keys(block_hashes, group_idx)
        return self._client.get_external_kv_hit_counts(keys)

    def lookup_locations(
        self,
        block_hashes: Iterable[ExternalBlockHash],
        group_idx: int | None = 0,
    ) -> list[list[Any]]:
        """Return every live physical UMBP location for each supplied key."""
        keys = self.encode_keys(block_hashes, group_idx)
        return self._client.batch_lookup_locations(keys)

    def physical_tier_name(self, tier: Any) -> str | None:
        return self._physical_tier_names.get(tier)


class MoriPhysicalPlacementResolver:
    """Attach authoritative MoRI placement to vLLM storage events.

    MoRI publishes placement through heartbeats after the data operation has
    completed, so missing locations are retried for a bounded interval. Store
    decisions are cached for removal events, which can arrive after the master
    has already retired the physical location.
    """

    def __init__(
        self,
        placement: MoriPlacementClient,
        node_id: str,
        lookup_timeout_s: float = 2.0,
        lookup_interval_s: float = 0.02,
        bandwidth_bps: dict[tuple[str, str], float] | None = None,
    ) -> None:
        if lookup_timeout_s < 0:
            raise ValueError("lookup_timeout_s must not be negative")
        if lookup_interval_s <= 0:
            raise ValueError("lookup_interval_s must be positive")
        self._placement = placement
        self._node_id = node_id
        self._lookup_timeout_s = lookup_timeout_s
        self._lookup_interval_s = lookup_interval_s
        self._bandwidth_bps = bandwidth_bps or {}
        self._stored: dict[
            tuple[ExternalBlockHash, int], Counter[UMBPPhysicalPlacement]
        ] = {}

    @staticmethod
    def _cache_key(
        block_hash: ExternalBlockHash, group_idx: int | None
    ) -> tuple[ExternalBlockHash, int]:
        return block_hash, group_idx or 0

    def _select(self, locations: list[Any]) -> UMBPPhysicalPlacement | None:
        candidates: list[tuple[int, int, str, str, str]] = []
        tier_priority = {"HBM": 0, "DRAM": 1, "SSD": 2}
        for location in locations:
            tier = self._placement.physical_tier_name(location.tier)
            if tier is None:
                continue
            locality = "LOCAL" if location.node_id == self._node_id else "REMOTE"
            candidates.append(
                (
                    tier_priority[tier],
                    0 if locality == "LOCAL" else 1,
                    location.node_id,
                    location.peer_address,
                    tier,
                )
            )
        if not candidates:
            return None
        _, _, source_node, _, tier = min(candidates)
        locality = "LOCAL" if source_node == self._node_id else "REMOTE"
        bandwidth = self._bandwidth_bps.get((locality, tier))
        return UMBPPhysicalPlacement(
            storage_tier=tier,
            source_node=source_node,
            locality=locality,
            estimated_bandwidth_bps=bandwidth if bandwidth and bandwidth > 0 else None,
        )

    def _lookup_stores_by_group(
        self, events: list[BlockStored]
    ) -> list[UMBPPhysicalPlacement | None]:
        results: list[UMBPPhysicalPlacement | None] = [None] * len(events)
        by_group: dict[int, list[int]] = {}
        for index, event in enumerate(events):
            by_group.setdefault(event.group_idx or 0, []).append(index)
        for group_idx, indices in by_group.items():
            grouped = [events[index] for index in indices]
            pending = list(range(len(grouped)))
            deadline = time.monotonic() + self._lookup_timeout_s
            while pending:
                locations = self._placement.lookup_locations(
                    [grouped[index].block_hashes[-1] for index in pending], group_idx
                )
                if len(locations) != len(pending):
                    raise RuntimeError(
                        "MoRI BatchLookupLocations response length does not "
                        "match request"
                    )
                next_pending = []
                for index, candidates in zip(pending, locations):
                    selected = self._select(candidates)
                    if selected is None:
                        next_pending.append(index)
                    else:
                        results[indices[index]] = selected
                if not next_pending or time.monotonic() >= deadline:
                    break
                time.sleep(self._lookup_interval_s)
                pending = next_pending
        return results

    @staticmethod
    def _annotate(
        event: BlockStored | BlockRemoved,
        placement: UMBPPhysicalPlacement | None,
    ) -> BlockStored | BlockRemoved:
        if placement is None:
            return msgspec.structs.replace(
                event,
                storage_tier="UMBP",
                locality=None,
                source_node=None,
                estimated_bandwidth_bps=None,
            )
        return msgspec.structs.replace(
            event,
            storage_tier=placement.storage_tier,
            locality=placement.locality,
            source_node=placement.source_node,
            estimated_bandwidth_bps=placement.estimated_bandwidth_bps,
        )

    def enrich_batch(self, batch: KVEventBatch) -> KVEventBatch:
        stores = [
            event
            for event in batch.events
            if isinstance(event, BlockStored)
            and event.medium == MEDIUM_STORAGE
            and event.block_hashes
        ]
        store_placements = iter(self._lookup_stores_by_group(stores))
        events: list[KVCacheEvent] = []
        for event in batch.events:
            if isinstance(event, AllBlocksCleared):
                self._stored.clear()
                events.append(event)
                continue

            if isinstance(event, BlockStored) and event.medium == MEDIUM_STORAGE:
                placement = next(store_placements) if event.block_hashes else None
                if placement is not None:
                    for block_hash in event.block_hashes:
                        key = self._cache_key(block_hash, event.group_idx)
                        self._stored.setdefault(key, Counter())[placement] += 1
                events.append(self._annotate(event, placement))
                continue

            if isinstance(event, BlockRemoved) and event.medium == MEDIUM_STORAGE:
                by_placement: dict[
                    UMBPPhysicalPlacement | None, list[ExternalBlockHash]
                ] = {}
                for block_hash in event.block_hashes:
                    key = self._cache_key(block_hash, event.group_idx)
                    counts = self._stored.get(key)
                    selected = (
                        max(counts.items(), key=lambda item: item[1])[0]
                        if counts
                        else None
                    )
                    by_placement.setdefault(selected, []).append(block_hash)
                    if selected is not None and counts is not None:
                        counts[selected] -= 1
                        if counts[selected] == 0:
                            del counts[selected]
                        if not counts:
                            del self._stored[key]
                if not by_placement:
                    by_placement[None] = []
                for placement, block_hashes in by_placement.items():
                    split = msgspec.structs.replace(event, block_hashes=block_hashes)
                    events.append(self._annotate(split, placement))
                continue

            events.append(event)
        return msgspec.structs.replace(batch, events=events)


def enrich_umbp_logical_placement(batch: KVEventBatch) -> KVEventBatch:
    """Mark successful storage events as logical UMBP availability.

    Released MoRI does not expose physical key placement. The publisher,
    locality, and physical DRAM/SSD tier therefore remain unspecified.
    """
    events = [
        msgspec.structs.replace(event, storage_tier="UMBP")
        if isinstance(event, (BlockStored, BlockRemoved))
        and event.medium == MEDIUM_STORAGE
        else event
        for event in batch.events
    ]
    return msgspec.structs.replace(batch, events=events)
