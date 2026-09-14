# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fenced transfer execution for the UMBP KVConnector worker.

This is the transfer component, not a registered KVConnector. Scheduler hooks
must retain cache-manager blocks until the all-rank completion barrier passes.
Calls into this component are serialized by the model-runner thread.
"""

import ctypes
import time
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.key import UMBPKeySpace
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.layout import UMBPLayout
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    BlockKey,
    ControlJob,
    RankCompletion,
    TransferId,
    TransferJob,
    UMBPWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import (
    BufferSlice,
    MemoryRegion,
    StoreBusyError,
    TransferObject,
    UMBPStore,
)


class ComputeFence(Protocol):
    """An event recorded after prior GPU use of the job's cache blocks."""

    def query(self) -> bool: ...

    def synchronize(self) -> None: ...


@dataclass
class _Pending:
    job: TransferJob
    objects: tuple[TransferObject, ...]
    ranges: tuple[tuple[int, int], ...]
    fence: ComputeFence
    future: Future[tuple[bool, ...]] | None = None
    cancelled: bool = False
    ready_future: Future[tuple[bool, ...]] | None = None
    ready: bool = False
    deadline: float = 0.0
    next_probe: float = 0.0


def _ranges(objects: tuple[TransferObject, ...]) -> tuple[tuple[int, int], ...]:
    ranges = sorted(
        (part.region.ptr + part.offset, part.region.ptr + part.offset + part.size)
        for obj in objects
        for part in obj.slices
    )
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def _overlap(
    first: tuple[tuple[int, int], ...], second: tuple[tuple[int, int], ...]
) -> bool:
    i = j = 0
    while i < len(first) and j < len(second):
        start_a, end_a = first[i]
        start_b, end_b = second[j]
        if start_a < end_b and start_b < end_a:
            return True
        if end_a <= start_b:
            i += 1
        else:
            j += 1
    return False


class UMBPTransferWorker:
    """Own registered allocations, deferred fences and native futures.

    Admission returns False without starting I/O if the worker is full or a
    destination overlaps an active transfer. Native queue pressure is retried
    by poll() within the already-bounded admitted job set. No timeout releases
    allocations still in native use. Closing drains native work, never clear().
    """

    def __init__(
        self,
        store: UMBPStore,
        layout: UMBPLayout,
        keyspace: UMBPKeySpace,
        *,
        epoch: str,
        rank: int,
        max_pending: int = 8,
    ) -> None:
        if keyspace.layout != layout.identity:
            raise ValueError("Cache key identity must match the registered layout")
        TransferId(epoch, 0)
        if type(rank) is not int or not 0 <= rank < layout.num_shards:
            raise ValueError("Worker rank must lie within the cache topology")
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("Pending transfer limit must be positive")
        self._store = store
        self._layout: UMBPLayout | None = layout
        self._keyspace = keyspace
        self._epoch = epoch
        self._rank = rank
        self._max_pending = max_pending
        self._highest_sequence = -1
        self._pending: dict[TransferId, _Pending] = {}
        self._closed = False
        self._ready_region: MemoryRegion | None = None
        try:
            for region in layout.regions:
                store.register_region(region)
        except Exception:
            store.close()
            raise

    @property
    def epoch(self) -> str:
        return self._epoch

    @property
    def rank(self) -> int:
        return self._rank

    def lookup(self, blocks: tuple[BlockKey, ...]) -> Future[tuple[bool, ...]]:
        """Probe this rank's own pool; scheduler combines results from all ranks."""
        if self._closed:
            raise RuntimeError("UMBP transfer worker is closed")
        assert self._layout is not None
        if not isinstance(blocks, tuple) or any(
            type(block) is not BlockKey
            or block.group_id not in self._layout.prefix_cacheable_group_ids
            for block in blocks
        ):
            raise ValueError("Lookup requires immutable prefix-cacheable keys")
        return self._store.lookup(
            tuple(
                self._keyspace.block_key(
                    block.block_hash, group=block.group_id, shard=self._rank
                )
                for block in blocks
            )
        )

    def control(self, job: ControlJob) -> Future[tuple[bool, ...]]:
        """Use the same bounded store for immutable one-byte ready markers."""
        if self._closed or job.id.epoch != self._epoch:
            raise ValueError("Control job belongs to a closed or different worker")
        if job.handle.namespace != self._keyspace.prefix:
            raise ValueError("Readiness namespace disagrees with the worker layout")
        if job.operation == "release" or time.time() * 1000 >= job.handle.expires_at_ms:
            result: Future[tuple[bool, ...]] = Future()
            result.set_result((job.operation == "release",))
            return result
        if self._ready_region is None:
            owner = ctypes.create_string_buffer(4096)
            owner[0] = b"\x01"
            region = MemoryRegion(ctypes.addressof(owner), 4096, owner)
            self._store.register_region(region)
            self._ready_region = region
        region = self._ready_region
        return self._store.store(
            (
                TransferObject(
                    job.handle.ready_key(self._rank),
                    1,
                    (BufferSlice(region, 0, 1, 0),),
                ),
            )
        )

    def _ready(self, pending: _Pending) -> bool:
        handle = pending.job.handoff
        if handle is None or pending.ready:
            return True
        if pending.ready_future is None:
            if time.monotonic() < pending.next_probe:
                return False
            # Local marker visibility can precede heartbeat-delivered routes
            # for KV held by another peer. Gate on the missing objects too.
            pending.ready_future = self._store.lookup(
                (handle.ready_key(self._rank),)
                + tuple(obj.key for obj in pending.objects)
            )
        if pending.ready_future.done():
            pending.ready = pending.ready_future.result() == (True,) * (
                len(pending.objects) + 1
            )
            pending.ready_future = None
            pending.next_probe = time.monotonic() + 0.01
        return pending.ready

    def submit(self, job: TransferJob, fence: ComputeFence) -> bool:
        """Admit an immutable job; caller has already pinned its GPU blocks."""
        if self._closed:
            raise RuntimeError("UMBP transfer worker is closed")
        if job.id.epoch != self._epoch:
            raise ValueError("Transfer belongs to a different engine generation")
        pending = self._pending.get(job.id)
        if pending is not None:
            if pending.job != job:
                raise ValueError("Transfer ID was reused with different contents")
            return True
        if job.id.sequence <= self._highest_sequence:
            raise ValueError("Stale transfer cannot be admitted again")
        if len(self._pending) >= self._max_pending:
            return False
        assert self._layout is not None
        if job.handoff is not None and job.handoff.namespace != self._keyspace.prefix:
            raise ValueError("Handoff namespace disagrees with the worker layout")
        if any(
            block.group_id not in self._layout.prefix_cacheable_group_ids
            for block in job.blocks
        ):
            raise ValueError("Non-prefix KV state requires a request-scoped handoff")
        objects = tuple(
            self._layout.block_object(
                self._keyspace.block_key(
                    block.block_hash, group=block.group_id, shard=self._rank
                ),
                group_id=block.group_id,
                block_id=block.block_id,
            )
            for block in job.blocks
        )
        ranges = _ranges(objects)
        for other in self._pending.values():
            if (job.operation == "load" or other.job.operation == "load") and _overlap(
                ranges, other.ranges
            ):
                return False
        self._pending[job.id] = _Pending(
            job,
            objects,
            ranges,
            fence,
            deadline=time.monotonic() + job.readiness_timeout,
        )
        self._highest_sequence = job.id.sequence
        return True

    def cancel(self, job_id: TransferId) -> None:
        """Suppress publication; running native I/O must still finish."""
        if pending := self._pending.get(job_id):
            pending.cancelled = True
            if pending.future is not None:
                pending.future.cancel()
            if pending.ready_future is not None:
                pending.ready_future.cancel()

    def poll(self) -> UMBPWorkerMetadata:
        """Return each local completion once, after its final native access."""
        completed: dict[TransferId, dict[int, RankCompletion]] = {}
        for job_id, pending in list(self._pending.items()):
            failed = (False,) * len(pending.objects)
            result = None
            if pending.future is None:
                # Fence failures are fatal, not a cache miss: they do not prove
                # that prior GPU work stopped touching the retained allocation.
                if not pending.fence.query():
                    continue
                expired = pending.job.handoff is not None and (
                    time.monotonic() >= pending.deadline
                    or time.time() * 1000 >= pending.job.handoff.expires_at_ms
                )
                if pending.cancelled or expired:
                    # An existence probe owns no KV addresses. The store keeps
                    # its native call alive/bounded and drains it at shutdown;
                    # this load can fail safely before any KV read was issued.
                    if pending.ready_future is not None:
                        pending.ready_future.cancel()
                    result = failed
                else:
                    try:
                        if not self._ready(pending):
                            continue
                        operation = (
                            self._store.load
                            if pending.job.operation == "load"
                            else self._store.store
                        )
                        pending.future = operation(pending.objects)
                    except StoreBusyError:
                        continue
                    except Exception:
                        result = failed
            if pending.future is not None and pending.future.done():
                try:
                    result = pending.future.result()
                except Exception:
                    result = failed
            if result is not None:
                completed[job_id] = {
                    self._rank: RankCompletion(result, cancelled=pending.cancelled)
                }
                del self._pending[job_id]
        return UMBPWorkerMetadata(completed)

    def close(self) -> None:
        self._closed = True
        for job_id in self._pending:
            self.cancel(job_id)
        for pending in self._pending.values():
            if pending.future is None:
                pending.fence.synchronize()
        self._store.close()
        self._pending.clear()
        self._layout = None
        self._ready_region = None
