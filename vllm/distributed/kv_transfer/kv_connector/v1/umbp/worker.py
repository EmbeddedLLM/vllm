# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fenced transfer execution for the UMBP KVConnector worker.

This is the transfer component, not a registered KVConnector. Scheduler hooks
must retain cache-manager blocks until the all-rank completion barrier passes.
Calls into this component are serialized by the model-runner thread.
"""

from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.key import UMBPKeySpace
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.layout import UMBPLayout
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    RankCompletion,
    TransferId,
    TransferJob,
    UMBPWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import (
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
        try:
            for region in layout.regions:
                store.register_region(region)
        except Exception:
            store.close()
            raise

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
        self._pending[job.id] = _Pending(job, objects, ranges, fence)
        self._highest_sequence = job.id.sequence
        return True

    def cancel(self, job_id: TransferId) -> None:
        """Suppress publication; running native I/O must still finish."""
        if pending := self._pending.get(job_id):
            pending.cancelled = True
            if pending.future is not None:
                pending.future.cancel()

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
                if pending.cancelled:
                    result = failed
                else:
                    try:
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
