# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cache-aware replica selection using UMBP placement metadata."""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    hash_block_tokens,
    init_none_hash,
)


@dataclass(frozen=True)
class MoriReplica:
    node_id: str
    url: str
    dp_rank: int | None = None


def hash_text_prefix(
    token_ids: Sequence[int],
    block_size: int,
    hash_algorithm: str = "sha256",
    lora_name: str | None = None,
    cache_salt: str | None = None,
) -> list[BlockHash]:
    """Build the same chained full-block hashes used by vLLM prefix caching."""
    if block_size <= 0:
        raise ValueError("block_size must be greater than zero")
    hash_fn = get_hash_fn_by_name(hash_algorithm)
    init_none_hash(hash_fn)
    hashes = []
    parent = None
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        extra = []
        if lora_name:
            extra.append(lora_name)
        if start == 0 and cache_salt:
            extra.append(cache_salt)
        parent = hash_block_tokens(
            hash_fn,
            parent,
            token_ids[start : start + block_size],
            tuple(extra) if extra else None,
        )
        hashes.append(parent)
    return hashes


class MoriCacheAwareRouter:
    """Choose the replica with the longest and fastest cached prefix."""

    DEFAULT_TIER_WEIGHTS = {"HBM": 1.0, "DRAM": 0.7, "SSD": 0.2}

    def __init__(
        self,
        placement_client: Any,
        replicas: Sequence[MoriReplica],
        tier_weights: dict[str, float] | None = None,
    ) -> None:
        if not replicas:
            raise ValueError("at least one replica is required")
        node_ids = [replica.node_id for replica in replicas]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("replica node_id values must be unique")
        self._placement = placement_client
        self._replicas = tuple(replicas)
        self._by_node = {replica.node_id: replica for replica in replicas}
        self._tier_weights = tier_weights or self.DEFAULT_TIER_WEIGHTS
        self._inflight: defaultdict[str, int] = defaultdict(int)
        self._tie_break = 0

    @staticmethod
    def _tier_name(tier: Any) -> str:
        name = getattr(tier, "name", None)
        return str(name if name is not None else tier).upper()

    def _scores(
        self,
        block_hashes: Sequence[BlockHash],
        group_indices: Iterable[int],
    ) -> dict[str, tuple[int, float]]:
        scores = {replica.node_id: (0, 0.0) for replica in self._replicas}
        if not block_hashes:
            return scores
        for group_idx in group_indices:
            matches = self._placement.match(
                block_hashes, group_idx=group_idx, count_as_hit=False
            )
            for match in matches:
                if match.node_id not in scores:
                    continue
                per_hash_weight: dict[str, float] = {}
                for tier, keys in match.hashes_by_tier.items():
                    weight = self._tier_weights.get(self._tier_name(tier), 0.0)
                    for key in keys:
                        per_hash_weight[key] = max(
                            per_hash_weight.get(key, 0.0), weight
                        )
                encoded = self._placement.encode_keys(block_hashes, group_idx)
                prefix_len = 0
                weighted_score = 0.0
                for key in encoded:
                    if key not in per_hash_weight:
                        break
                    prefix_len += 1
                    weighted_score += per_hash_weight[key]
                old_len, old_weight = scores[match.node_id]
                scores[match.node_id] = (
                    old_len + prefix_len,
                    old_weight + weighted_score,
                )
        return scores

    def select(
        self,
        block_hashes: Sequence[BlockHash],
        group_indices: Iterable[int] = (0,),
    ) -> MoriReplica:
        group_indices = tuple(group_indices)
        scores = self._scores(block_hashes, group_indices)
        start = self._tie_break % len(self._replicas)
        ordered = self._replicas[start:] + self._replicas[:start]
        self._tie_break += 1
        selected = max(
            ordered,
            key=lambda replica: (
                scores[replica.node_id][0],
                scores[replica.node_id][1],
                -self._inflight[replica.node_id],
            ),
        )
        self._inflight[selected.node_id] += 1
        if scores[selected.node_id][0] > 0:
            for group_idx in group_indices:
                self._placement.match(
                    block_hashes,
                    group_idx=group_idx,
                    count_as_hit=True,
                )
        return selected

    def finish(self, replica: MoriReplica) -> None:
        self._inflight[replica.node_id] = max(0, self._inflight[replica.node_id] - 1)
