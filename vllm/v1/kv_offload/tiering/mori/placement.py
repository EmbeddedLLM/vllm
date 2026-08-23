# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""UMBP control-plane adapter for vLLM KV cache events."""

from collections import Counter
from collections.abc import Iterable
from typing import Any

from vllm.distributed.kv_events import (
    MEDIUM_CPU,
    MEDIUM_GPU,
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import ExternalBlockHash

logger = init_logger(__name__)


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
