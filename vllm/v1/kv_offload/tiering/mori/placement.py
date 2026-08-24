# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMBP control-plane adapter for vLLM KV cache events."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

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
class MoriPhysicalPlacement:
    storage_tier: str
    source_node: str
    locality: str
    estimated_bandwidth_bps: float | None


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


class MoriPlacementReconciler:
    """Convert authoritative MoRI placements into refreshable KV events."""

    def __init__(
        self,
        master_client: Any,
        node_id: str,
        key_prefix: str,
        bandwidth_bps: dict[str, float] | None = None,
        max_batch_size: int = 1024,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be greater than zero")
        if any(value <= 0 for value in (bandwidth_bps or {}).values()):
            raise ValueError("bandwidth_bps values must be greater than zero")
        self._client = master_client
        self._node_id = node_id
        self._key_prefix = key_prefix
        self._bandwidth_bps = {
            key.upper(): value for key, value in (bandwidth_bps or {}).items()
        }
        self._max_batch_size = max_batch_size
        self._tracked: dict[str, tuple[ExternalBlockHash, int | None]] = {}
        self._metadata: dict[str, tuple[ExternalBlockHash, int | None]] = {}
        self._published: dict[str, MoriPhysicalPlacement] = {}

    def _encode_key(self, block_hash: ExternalBlockHash, group_idx: int | None) -> str:
        return encode_umbp_event_key(block_hash, group_idx, self._key_prefix)

    def observe_batch(self, batch: KVEventBatch) -> None:
        for event in batch.events:
            if isinstance(event, AllBlocksCleared):
                self._tracked.clear()
            elif isinstance(event, BlockStored) and event.medium == MEDIUM_STORAGE:
                for block_hash in event.block_hashes:
                    self._tracked[self._encode_key(block_hash, event.group_idx)] = (
                        block_hash,
                        event.group_idx,
                    )
                    self._metadata[self._encode_key(block_hash, event.group_idx)] = (
                        block_hash,
                        event.group_idx,
                    )
            elif isinstance(event, BlockRemoved) and event.medium == MEDIUM_STORAGE:
                for block_hash in event.block_hashes:
                    self._tracked.pop(
                        self._encode_key(block_hash, event.group_idx), None
                    )

    @staticmethod
    def _tier_name(tier: Any) -> str:
        return str(getattr(tier, "name", tier)).upper()

    def _placement(self, result: Any) -> MoriPhysicalPlacement:
        tier = self._tier_name(result.tier)
        locality = "LOCAL" if result.node_id == self._node_id else "REMOTE"
        bandwidth = self._bandwidth_bps.get(f"{locality}:{tier}")
        return MoriPhysicalPlacement(tier, result.node_id, locality, bandwidth)

    @staticmethod
    def _event(
        metadata: tuple[ExternalBlockHash, int | None],
        placement: MoriPhysicalPlacement,
        removed: bool,
    ) -> BlockStored | BlockRemoved:
        block_hash, group_idx = metadata
        common = {
            "block_hashes": [block_hash],
            "medium": MEDIUM_STORAGE,
            "group_idx": group_idx,
            "locality": placement.locality,
            "storage_tier": placement.storage_tier,
            "source_node": placement.source_node,
            "estimated_bandwidth_bps": placement.estimated_bandwidth_bps,
        }
        if removed:
            return BlockRemoved(**common)
        return BlockStored(
            parent_block_hash=None,
            token_ids=[],
            block_size=0,
            lora_id=None,
            lora_name=None,
            **common,
        )

    def reconcile(self) -> list[BlockStored | BlockRemoved]:
        keys = list(self._tracked)
        results = []
        for start in range(0, len(keys), self._max_batch_size):
            results.extend(
                self._client.batch_inspect(keys[start : start + self._max_batch_size])
            )
        if len(results) != len(keys):
            raise RuntimeError(
                "MoRI BatchInspect returned a result count different from its request"
            )
        current = {
            key: self._placement(result)
            for key, result in zip(keys, results)
            if result is not None
        }
        events: list[BlockStored | BlockRemoved] = []
        for key, old in self._published.items():
            if current.get(key) != old:
                events.append(self._event(self._metadata[key], old, removed=True))
        for key, placement in current.items():
            events.append(self._event(self._tracked[key], placement, removed=False))
        self._published = current
        for key in list(self._metadata):
            if key not in self._tracked and key not in self._published:
                del self._metadata[key]
        return events
