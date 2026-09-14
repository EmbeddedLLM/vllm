# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded scheduler ownership for UMBP lookups and transfers.

This is the job lifecycle, not the request-to-KV planner. Stores must refer to
actual vLLM cache records; loads must use exclusively owned, unpublished blocks.
The planner must choose valid group boundaries and use a fresh Request object
for every request incarnation. No timeout substitutes for native completion.
"""

from dataclasses import dataclass
from typing import Literal

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    BlockKey,
    BlockTransfer,
    CompletionBarrier,
    ControlJob,
    JobOutcome,
    LookupJob,
    TransferId,
    TransferJob,
    UMBPConnectorMetadata,
    UMBPWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.protocol import HandoffHandle
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request


@dataclass
class _PendingJob:
    request: Request
    barrier: CompletionBarrier
    pinned: tuple[KVCacheBlock, ...] = ()
    dispatched: bool = False
    cancelled: bool = False
    finalization_dispatched: bool = False
    retired_ranks: frozenset[int] = frozenset()


class UMBPTransferScheduler:
    """Keep exact physical blocks alive across request free and preemption.

    Completed jobs remain bounded until every worker acknowledges scheduler
    finalization. Receives are published in that later step, never merely
    because one rank has finished. All public calls run on the scheduler thread.
    """

    def __init__(
        self,
        manager: KVCacheManager,
        *,
        epoch: str,
        ranks: frozenset[int],
        max_pending: int = 8,
    ) -> None:
        TransferId(epoch, 0)
        if not ranks or any(type(rank) is not int or rank < 0 for rank in ranks):
            raise ValueError("Scheduler requires explicit worker ranks")
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("Pending job limit must be positive")
        if any(
            tensor.host_resident for tensor in manager.kv_cache_config.kv_cache_tensors
        ):
            raise ValueError("Host-resident KV needs pool-qualified block identities")
        self._manager = manager
        self._pool = manager.block_pool
        self._groups = frozenset(manager.kv_cache_config.prefix_cacheable_group_ids)
        self._epoch = epoch
        self._ranks = frozenset(ranks)
        self._max_pending = max_pending
        self._sequence = 0
        self._pending: dict[TransferId, _PendingJob] = {}
        self._retiring: dict[TransferId, _PendingJob] = {}
        self._finalized: list[JobOutcome] = []
        self._cancelled: set[TransferId] = set()

    def _has_capacity(self) -> bool:
        return len(self._pending) + len(self._retiring) < self._max_pending

    def can_admit(self) -> bool:
        """Check capacity on the scheduler thread before allocating a receive."""
        return self._has_capacity()

    @property
    def epoch(self) -> str:
        return self._epoch

    def control(
        self,
        request: Request,
        handle: HandoffHandle,
        operation: Literal["publish", "release"],
    ) -> ControlJob | None:
        job = ControlJob(
            TransferId(self._epoch, self._sequence),
            request.request_id,
            handle,
            operation,
        )
        if not self._has_capacity():
            return None
        self._pending[job.id] = _PendingJob(
            request, CompletionBarrier(job, self._ranks)
        )
        self._sequence += 1
        return job

    def lookup(self, request: Request, keys: tuple[BlockKey, ...]) -> LookupJob | None:
        job = LookupJob(
            TransferId(self._epoch, self._sequence), request.request_id, keys
        )
        if any(key.group_id not in self._groups for key in keys):
            raise ValueError("Lookup requires prefix-cacheable groups")
        if not self._has_capacity():
            return None
        self._pending[job.id] = _PendingJob(
            request, CompletionBarrier(job, self._ranks)
        )
        self._sequence += 1
        return job

    def transfer(
        self,
        request: Request,
        operation: Literal["load", "store"],
        blocks: tuple[BlockTransfer, ...],
        *,
        handoff: HandoffHandle | None = None,
        readiness_timeout: float = 30.0,
    ) -> TransferJob | None:
        job = TransferJob(
            TransferId(self._epoch, self._sequence),
            request.request_id,
            operation,
            blocks,
            handoff,
            readiness_timeout,
        )
        if not self._has_capacity():
            return None
        if operation == "load" and any(
            isinstance(pending.barrier.job, TransferJob)
            and pending.barrier.job.operation == "load"
            and pending.barrier.job.request_id == request.request_id
            for pending in (*self._pending.values(), *self._retiring.values())
        ):
            # finished_recving identifies a request, not an individual job.
            return None
        if len({(block.group_id, block.block_hash) for block in blocks}) != len(blocks):
            raise ValueError("A transfer cannot contain duplicate object keys")
        if len({block.block_id for block in blocks}) != len(blocks):
            raise ValueError("A transfer cannot alias physical blocks across objects")
        pinned = []
        owned = self._manager.get_blocks(request.request_id).blocks
        for item in blocks:
            if item.group_id not in self._groups:
                raise ValueError("Non-prefix state needs a request-scoped handoff")
            if item.block_id >= len(self._pool.blocks):
                raise ValueError("Transfer block is outside the GPU pool")
            block = self._pool.blocks[item.block_id]
            if block.is_null or (operation == "load" and block.ref_cnt <= 0):
                raise ValueError("Transfer requires an allocated non-null block")
            if operation == "store":
                key = make_block_hash_with_group_id(
                    BlockHash(item.block_hash), item.group_id
                )
                if not self._pool.cached_block_hash_to_block.contain(
                    key, item.block_id
                ):
                    raise ValueError("Store source does not match a vLLM cache record")
                # A synchronous scheduler can release a CoW destination to
                # the cache before offering it here. Its exact record remains
                # valid; touch below removes it from the free queue before any
                # further allocation. Never apply this rule to load targets.
            elif not any(
                candidate is block for candidate in owned[item.group_id]
            ) or not self._pool.is_block_writable(block):
                raise ValueError(
                    "Load destination must be exclusively owned and unpublished"
                )
            pinned.append(block)
        ids = {block.block_id for block in pinned}
        for pending in self._pending.values():
            other = pending.barrier.job
            if (
                isinstance(other, TransferJob)
                and (operation == "load" or other.operation == "load")
                and ids.intersection(block.block_id for block in pending.pinned)
            ):
                return None
        self._pool.touch(pinned)
        self._pending[job.id] = _PendingJob(
            request, CompletionBarrier(job, self._ranks), tuple(pinned)
        )
        self._sequence += 1
        return job

    def cancel_request(self, request: Request) -> None:
        """Suppress publication without releasing dispatched native ownership."""
        for job_id, pending in list(self._pending.items()):
            if pending.request is not request or pending.cancelled:
                continue
            pending.cancelled = True
            # Even an undispatched receive must retire through worker feedback
            # so an aborted WAITING_FOR_REMOTE_KVS request can leave that state.
            self._cancelled.add(job_id)

    def build_connector_meta(self) -> UMBPConnectorMetadata:
        jobs = []
        for pending in self._pending.values():
            if not pending.dispatched:
                pending.dispatched = True
                jobs.append(pending.barrier.job)
        metadata = UMBPConnectorMetadata(
            self._epoch,
            tuple(jobs),
            tuple(sorted(self._cancelled)),
            tuple(self._finalized),
        )
        self._cancelled.clear()
        for outcome in self._finalized:
            self._retiring[outcome.job.id].finalization_dispatched = True
        self._finalized.clear()
        return metadata

    def update_connector_output(
        self, output: KVConnectorOutput
    ) -> tuple[JobOutcome, ...]:
        metadata = output.kv_connector_worker_meta
        if metadata is None:
            return ()
        if not isinstance(metadata, UMBPWorkerMetadata):
            raise TypeError("Wrong connector worker metadata")
        # Validate the entire feedback batch before releasing any references.
        updated = []
        for job_id in metadata.completions:
            pending = self._pending.get(job_id)
            if pending is None:
                continue  # Already retired or from a prior engine generation.
            if not pending.dispatched:
                raise ValueError("Worker acknowledged an undispatched job")
            barrier = CompletionBarrier(pending.barrier.job, self._ranks)
            barrier.update(pending.barrier.snapshot())
            barrier.update(metadata)
            updated.append((job_id, pending, barrier))
        retirements = []
        for job_id, ranks in metadata.retired.items():
            pending = self._retiring.get(job_id)
            if pending is None:
                if job_id in self._pending:
                    raise ValueError("Job retired before native completion")
                continue
            if not pending.finalization_dispatched or not ranks <= self._ranks:
                raise ValueError("Retirement disagrees with scheduler finalization")
            retirements.append((job_id, pending, pending.retired_ranks | ranks))
        completed = []
        for job_id, pending, barrier in updated:
            pending.barrier = barrier
            if not barrier.done:
                continue
            successes = barrier.successes
            if pending.cancelled:
                successes = (False,) * len(successes)
            outcome = JobOutcome(barrier.job, successes)
            # Load IDs also identify failures in a future worker snapshot.
            # Retain them until that snapshot is consumed, not merely until
            # DMA stops, so an old error cannot target a newly reused block.
            if (
                not isinstance(barrier.job, TransferJob)
                or barrier.job.operation != "load"
            ):
                self._pool.free_blocks(reversed(pending.pinned))
                pending.pinned = ()
            del self._pending[job_id]
            self._retiring[job_id] = pending
            self._cancelled.discard(job_id)
            self._finalized.append(outcome)
            completed.append(outcome)
        for job_id, pending, ranks in retirements:
            pending.retired_ranks = ranks
            if ranks == self._ranks:
                self._pool.free_blocks(reversed(pending.pinned))
                del self._retiring[job_id]
        return tuple(completed)

    def has_pending_push_work(self) -> bool:
        return bool(self._pending or self._retiring)

    def has_pending_block_frees(self) -> bool:
        return any(
            pending.pinned
            for pending in (*self._pending.values(), *self._retiring.values())
        )
