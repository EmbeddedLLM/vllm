# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side lifecycle for prefetching offloaded KV into CPU memory."""

import time
from collections.abc import Callable, Collection, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from vllm.v1.core.kv_cache_utils import BlockHash, hash_block_tokens
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadingManager,
    OffloadKey,
    ReqContext,
    ScheduleEndContext,
)


@dataclass(frozen=True)
class PrefetchMMFeature:
    identifier: str
    offset: int
    length: int


def hash_token_prefix(
    token_ids: Sequence[int],
    block_size: int,
    hash_fn: Callable[[Any], bytes],
    cache_salt: str | None = None,
    lora_name: str | None = None,
    mm_features: Sequence[PrefetchMMFeature] = (),
) -> list[BlockHash]:
    """Hash full token blocks using vLLM's configured chained hash."""
    hashes: list[BlockHash] = []
    parent = None
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        end = start + block_size
        extra_keys: list[Any] = []
        if lora_name:
            extra_keys.append(lora_name)
        for feature in mm_features:
            if end > feature.offset and start < feature.offset + feature.length:
                extra_keys.append((feature.identifier, feature.offset - start))
        if start == 0 and cache_salt:
            extra_keys.append(cache_salt)
        parent = hash_block_tokens(
            hash_fn,
            parent,
            token_ids[start : start + block_size],
            tuple(extra_keys) if extra_keys else None,
        )
        hashes.append(parent)
    return hashes


class PrefetchStatus(Enum):
    PENDING = "pending"
    READY = "ready"
    PARTIAL = "partial"
    MISS = "miss"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PrefetchResult:
    prefetch_id: str
    status: PrefetchStatus
    total_blocks: int
    ready_blocks: int


@dataclass
class _PrefetchState:
    keys: tuple[OffloadKey, ...]
    req_context: ReqContext
    ready: set[OffloadKey] = field(default_factory=set)
    missed: set[OffloadKey] = field(default_factory=set)
    status: PrefetchStatus = PrefetchStatus.PENDING
    updated_at: float = 0.0


class KVOffloadPrefetchCoordinator:
    """Drive existing offload promotions before normal request admission.

    The coordinator runs on the scheduler thread. Calling ``poll`` advances
    asynchronous tier lookups and promotions without allocating GPU blocks.
    """

    def __init__(
        self,
        manager: OffloadingManager,
        *,
        max_pending_requests: int = 64,
        max_pending_blocks: int = 4096,
        pending_ttl_seconds: float = 30.0,
        terminal_ttl_seconds: float = 60.0,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_pending_requests <= 0 or max_pending_blocks <= 0:
            raise ValueError("prefetch limits must be greater than zero")
        if pending_ttl_seconds <= 0 or terminal_ttl_seconds <= 0:
            raise ValueError("prefetch TTLs must be greater than zero")
        self._manager = manager
        self._states: dict[str, _PrefetchState] = {}
        self._max_pending_requests = max_pending_requests
        self._max_pending_blocks = max_pending_blocks
        self._pending_ttl_seconds = pending_ttl_seconds
        self._terminal_ttl_seconds = terminal_ttl_seconds
        self._time = time_fn

    def start(
        self,
        prefetch_id: str,
        keys: Collection[OffloadKey],
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> PrefetchResult:
        self.expire()
        if not prefetch_id:
            raise ValueError("prefetch_id must not be empty")
        normalized_keys = tuple(dict.fromkeys(keys))
        if not normalized_keys:
            raise ValueError("keys must not be empty")
        existing = self._states.get(prefetch_id)
        if existing is not None:
            if existing.keys != normalized_keys:
                raise ValueError("prefetch_id already refers to different keys")
            return self._result(prefetch_id, existing)

        pending_states = [
            state
            for state in self._states.values()
            if state.status is PrefetchStatus.PENDING
        ]
        if len(pending_states) >= self._max_pending_requests:
            raise RuntimeError("prefetch request capacity is exhausted")
        pending_blocks = sum(len(state.keys) for state in pending_states)
        if pending_blocks + len(normalized_keys) > self._max_pending_blocks:
            raise RuntimeError("prefetch block capacity is exhausted")

        req_context = ReqContext(
            req_id=f"kv-prefetch:{prefetch_id}",
            kv_transfer_params=kv_transfer_params,
        )
        self._manager.on_new_request(req_context)
        state = _PrefetchState(
            keys=normalized_keys,
            req_context=req_context,
            updated_at=self._time(),
        )
        self._states[prefetch_id] = state
        return self._advance(prefetch_id, state, is_new=True)

    def poll(self, prefetch_id: str) -> PrefetchResult:
        self.expire()
        state = self._states.get(prefetch_id)
        if state is None:
            raise KeyError(prefetch_id)
        if state.status is not PrefetchStatus.PENDING:
            return self._result(prefetch_id, state)
        return self._advance(prefetch_id, state, is_new=False)

    def cancel(self, prefetch_id: str) -> PrefetchResult:
        self.expire()
        state = self._states.get(prefetch_id)
        if state is None:
            raise KeyError(prefetch_id)
        if state.status is PrefetchStatus.PENDING:
            state.status = PrefetchStatus.CANCELLED
            state.updated_at = self._time()
            self._manager.on_request_finished(state.req_context)
        return self._result(prefetch_id, state)

    def forget(self, prefetch_id: str) -> None:
        self.expire()
        state = self._states[prefetch_id]
        if state.status is PrefetchStatus.PENDING:
            raise RuntimeError("cannot forget a pending prefetch")
        del self._states[prefetch_id]

    def expire(self) -> None:
        """Cancel stale work and discard expired terminal records."""
        now = self._time()
        expired = []
        for prefetch_id, state in self._states.items():
            age = now - state.updated_at
            if state.status is PrefetchStatus.PENDING:
                if age < self._pending_ttl_seconds:
                    continue
                state.status = PrefetchStatus.CANCELLED
                self._manager.on_request_finished(state.req_context)
            elif age < self._terminal_ttl_seconds:
                continue
            expired.append(prefetch_id)
        for prefetch_id in expired:
            del self._states[prefetch_id]

    def _advance(
        self, prefetch_id: str, state: _PrefetchState, *, is_new: bool
    ) -> PrefetchResult:
        pending = False
        state.updated_at = self._time()
        for key in state.keys:
            if key in state.ready or key in state.missed:
                continue
            result = self._manager.lookup(key, state.req_context)
            if result is LookupResult.HIT:
                state.ready.add(key)
            elif result is LookupResult.MISS:
                state.missed.add(key)
            else:
                pending = True

        self._manager.on_schedule_end(
            ScheduleEndContext(
                new_req_ids=(state.req_context.req_id,) if is_new else (),
                preempted_req_ids=(),
            )
        )
        if not pending:
            if not state.missed:
                state.status = PrefetchStatus.READY
            elif state.ready:
                state.status = PrefetchStatus.PARTIAL
            else:
                state.status = PrefetchStatus.MISS
            self._manager.on_request_finished(state.req_context)
        return self._result(prefetch_id, state)

    @staticmethod
    def _result(prefetch_id: str, state: _PrefetchState) -> PrefetchResult:
        return PrefetchResult(
            prefetch_id=prefetch_id,
            status=state.status,
            total_blocks=len(state.keys),
            ready_blocks=len(state.ready),
        )
