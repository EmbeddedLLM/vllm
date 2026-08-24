# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict
from contextlib import suppress

import pytest

from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import hash_block_tokens, init_none_hash
from vllm.v1.kv_offload.base import LookupResult, OffloadKey
from vllm.v1.kv_offload.prefetch import (
    KVOffloadPrefetchCoordinator,
    PrefetchMMFeature,
    PrefetchStatus,
    hash_token_prefix,
)


class MockOffloadingManager:
    def __init__(self, results):
        self.results = {key: iter(values) for key, values in results.items()}
        self.last = {}
        self.new_requests = []
        self.finished_requests = []
        self.schedule_ends = []
        self.lookup_counts = defaultdict(int)

    def on_new_request(self, context):
        self.new_requests.append(context)

    def lookup(self, key, context):
        self.lookup_counts[key] += 1
        with suppress(StopIteration):
            self.last[key] = next(self.results[key])
        return self.last[key]

    def on_schedule_end(self, context):
        self.schedule_ends.append(context)

    def on_request_finished(self, context):
        self.finished_requests.append(context)


def _key(value: bytes) -> OffloadKey:
    return OffloadKey(value)


def test_hash_token_prefix_matches_vllm_extra_key_order():
    hash_fn = get_hash_fn_by_name("sha256")
    init_none_hash(hash_fn)
    tokens = [1, 2, 3, 4]
    features = [PrefetchMMFeature("image", offset=1, length=2)]

    hashes = hash_token_prefix(
        tokens,
        2,
        hash_fn,
        cache_salt="tenant",
        lora_name="adapter-a",
        mm_features=features,
    )

    first = hash_block_tokens(
        hash_fn,
        None,
        [1, 2],
        ("adapter-a", ("image", 1), "tenant"),
    )
    second = hash_block_tokens(
        hash_fn,
        first,
        [3, 4],
        ("adapter-a", ("image", -1)),
    )
    assert hashes == [first, second]


def test_prefetch_promotes_pending_blocks_and_is_idempotent():
    key = _key(b"key")
    manager = MockOffloadingManager({key: [LookupResult.HIT_PENDING, LookupResult.HIT]})
    coordinator = KVOffloadPrefetchCoordinator(manager)

    first = coordinator.start("request-1", [key, key])
    duplicate = coordinator.start("request-1", [key])
    ready = coordinator.poll("request-1")

    assert first.status is PrefetchStatus.PENDING
    assert duplicate == first
    assert ready.status is PrefetchStatus.READY
    assert ready.total_blocks == ready.ready_blocks == 1
    assert len(manager.new_requests) == 1
    assert len(manager.finished_requests) == 1
    assert manager.schedule_ends[0].new_req_ids == ("kv-prefetch:request-1",)


def test_prefetch_reports_partial_result():
    hit, miss = _key(b"hit"), _key(b"miss")
    manager = MockOffloadingManager(
        {hit: [LookupResult.HIT], miss: [LookupResult.MISS]}
    )
    coordinator = KVOffloadPrefetchCoordinator(manager)

    result = coordinator.start("request-1", [hit, miss])

    assert result.status is PrefetchStatus.PARTIAL
    assert result.ready_blocks == 1
    assert len(manager.finished_requests) == 1


def test_prefetch_cancel_finishes_context_and_blocks_forget_while_pending():
    key = _key(b"key")
    manager = MockOffloadingManager({key: [LookupResult.RETRY]})
    coordinator = KVOffloadPrefetchCoordinator(manager)
    coordinator.start("request-1", [key])

    with pytest.raises(RuntimeError, match="pending"):
        coordinator.forget("request-1")

    cancelled = coordinator.cancel("request-1")
    coordinator.forget("request-1")

    assert cancelled.status is PrefetchStatus.CANCELLED
    assert len(manager.finished_requests) == 1


def test_prefetch_rejects_id_reuse_with_different_keys():
    first, second = _key(b"first"), _key(b"second")
    manager = MockOffloadingManager({first: [LookupResult.RETRY]})
    coordinator = KVOffloadPrefetchCoordinator(manager)
    coordinator.start("request-1", [first])

    with pytest.raises(ValueError, match="different keys"):
        coordinator.start("request-1", [second])


def test_prefetch_admission_bounds_pending_requests_and_blocks():
    first, second = _key(b"first"), _key(b"second")
    manager = MockOffloadingManager(
        {first: [LookupResult.RETRY], second: [LookupResult.RETRY]}
    )
    coordinator = KVOffloadPrefetchCoordinator(
        manager, max_pending_requests=1, max_pending_blocks=1
    )
    coordinator.start("request-1", [first])

    with pytest.raises(RuntimeError, match="request capacity"):
        coordinator.start("request-2", [second])

    coordinator.cancel("request-1")
    assert coordinator.start("request-2", [second]).status is PrefetchStatus.PENDING

    block_limited = KVOffloadPrefetchCoordinator(
        manager, max_pending_requests=2, max_pending_blocks=1
    )
    block_limited.start("block-1", [first])
    with pytest.raises(RuntimeError, match="block capacity"):
        block_limited.start("block-2", [second])


def test_prefetch_expires_stale_pending_and_terminal_records():
    now = 0.0

    def clock():
        return now

    pending, ready = _key(b"pending"), _key(b"ready")
    manager = MockOffloadingManager(
        {pending: [LookupResult.RETRY], ready: [LookupResult.HIT]}
    )
    coordinator = KVOffloadPrefetchCoordinator(
        manager,
        pending_ttl_seconds=2,
        terminal_ttl_seconds=1,
        time_fn=clock,
    )
    coordinator.start("pending", [pending])
    coordinator.start("ready", [ready])

    now = 1.5
    coordinator.expire()
    with pytest.raises(KeyError):
        coordinator.poll("ready")
    assert coordinator.poll("pending").status is PrefetchStatus.PENDING

    now = 4.0
    coordinator.expire()
    with pytest.raises(KeyError):
        coordinator.poll("pending")
    assert len(manager.finished_requests) == 2
