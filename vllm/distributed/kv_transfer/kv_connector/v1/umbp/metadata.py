# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation-scoped transfer jobs and idempotent per-rank acknowledgements."""

from dataclasses import dataclass, field
from typing import Literal

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.protocol import HandoffHandle


@dataclass(frozen=True, order=True)
class TransferId:
    """Sequence is strictly increasing within one scheduler engine generation."""

    epoch: str
    sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.epoch, str) or not self.epoch:
            raise ValueError("A transfer needs an engine generation")
        if type(self.sequence) is not int or self.sequence < 0:
            raise ValueError("Transfer sequence must be a nonnegative integer")


@dataclass(frozen=True)
class BlockKey:
    """A vLLM-owned block hash and cache group, without a destination address."""

    block_hash: bytes
    group_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.block_hash, bytes) or not self.block_hash:
            raise ValueError("A transfer needs a nonempty vLLM block hash")
        if type(self.group_id) is not int or self.group_id < 0:
            raise ValueError("Group ID must be a nonnegative integer")


@dataclass(frozen=True)
class BlockTransfer(BlockKey):
    """A vLLM-owned block hash and the allocated group/block it maps to."""

    block_id: int

    def __post_init__(self) -> None:
        super().__post_init__()
        if type(self.block_id) is not int or self.block_id < 0:
            raise ValueError("Block ID must be a nonnegative integer")


@dataclass(frozen=True)
class LookupJob:
    """Probe each worker's own native pool and compatible shard keys."""

    id: TransferId
    request_id: str
    blocks: tuple[BlockKey, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.id, TransferId):
            raise ValueError("A lookup needs a generation-scoped ID")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("A lookup needs a request ID")
        if (
            not isinstance(self.blocks, tuple)
            or not self.blocks
            or any(type(block) is not BlockKey for block in self.blocks)
        ):
            raise ValueError("A lookup needs a nonempty immutable key tuple")

    @property
    def num_objects(self) -> int:
        return len(self.blocks)


@dataclass(frozen=True)
class TransferJob:
    """One atomic scheduler ownership unit, potentially spanning cache groups."""

    id: TransferId
    request_id: str
    operation: Literal["load", "store"]
    blocks: tuple[BlockTransfer, ...]
    handoff: HandoffHandle | None = None
    readiness_timeout: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.id, TransferId):
            raise ValueError("A transfer needs a generation-scoped ID")
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("A transfer needs a request ID")
        if self.operation not in ("load", "store"):
            raise ValueError("Invalid transfer operation")
        if (
            not isinstance(self.blocks, tuple)
            or not self.blocks
            or any(not isinstance(block, BlockTransfer) for block in self.blocks)
        ):
            raise ValueError("A transfer needs a nonempty immutable block tuple")
        if self.handoff is not None:
            if self.operation != "load" or not isinstance(self.handoff, HandoffHandle):
                raise ValueError("Only a load can wait on a P/D handoff")
            if (
                type(self.readiness_timeout) not in (int, float)
                or not 0 < self.readiness_timeout <= 3600
            ):
                raise ValueError("Readiness timeout must be in (0, 3600] seconds")

    @property
    def num_objects(self) -> int:
        return len(self.blocks)


@dataclass(frozen=True)
class ControlJob:
    """Publish per-rank readiness, or release a failed producer without it."""

    id: TransferId
    request_id: str
    handle: HandoffHandle
    operation: Literal["publish", "release"]

    def __post_init__(self) -> None:
        if not isinstance(self.id, TransferId) or not isinstance(
            self.handle, HandoffHandle
        ):
            raise ValueError(
                "Control jobs require typed generation and handoff identities"
            )
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("Control jobs require a request ID")
        if self.operation not in ("publish", "release"):
            raise ValueError("Unknown UMBP control operation")

    @property
    def num_objects(self) -> int:
        return 1


@dataclass(frozen=True)
class RankCompletion:
    """Per-object outcomes; cancellation prevents publication even after I/O."""

    successes: tuple[bool, ...]
    cancelled: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.successes, tuple)
            or not self.successes
            or any(type(value) is not bool for value in self.successes)
            or type(self.cancelled) is not bool
        ):
            raise ValueError("Completion outcomes must be an immutable boolean tuple")

    @property
    def succeeded(self) -> bool:
        return not self.cancelled and bool(self.successes) and all(self.successes)


@dataclass
class UMBPWorkerMetadata(KVConnectorWorkerMetadata):
    """A rank may acknowledge a job once, even if feedback is delivered twice."""

    completions: dict[TransferId, dict[int, RankCompletion]] = field(
        default_factory=dict
    )
    retired: dict[TransferId, frozenset[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for job, ranks in self.completions.items():
            if not isinstance(job, TransferId):
                raise ValueError("UMBP feedback needs generation-scoped job IDs")
            if any(
                type(rank) is not int
                or rank < 0
                or not isinstance(result, RankCompletion)
                for rank, result in ranks.items()
            ):
                raise ValueError("UMBP feedback needs explicit rank outcomes")
        for job, retired_ranks in self.retired.items():
            if (
                not isinstance(job, TransferId)
                or not isinstance(retired_ranks, frozenset)
                or any(type(rank) is not int or rank < 0 for rank in retired_ranks)
            ):
                raise ValueError("Retirement feedback needs immutable explicit ranks")

    def aggregate(self, other: KVConnectorWorkerMetadata) -> "UMBPWorkerMetadata":
        if not isinstance(other, UMBPWorkerMetadata):
            raise TypeError("Cannot aggregate another connector's metadata")
        merged = {job: dict(ranks) for job, ranks in self.completions.items()}
        for job, ranks in other.completions.items():
            destination = merged.setdefault(job, {})
            for rank, completion in ranks.items():
                if rank in destination and destination[rank] != completion:
                    raise ValueError(
                        "Conflicting UMBP completion for the same rank/job"
                    )
                destination[rank] = completion
        retired = dict(self.retired)
        for job, retired_ranks in other.retired.items():
            retired[job] = retired.get(job, frozenset()) | retired_ranks
        return UMBPWorkerMetadata(merged, retired)


class CompletionBarrier:
    """Retain scheduler ownership until every required rank finishes the job.

    A failed rank does not permit early release: another rank may still be
    accessing the retained blocks. Receipts may arrive in different steps.
    """

    def __init__(
        self, job: TransferJob | LookupJob | ControlJob, ranks: frozenset[int]
    ) -> None:
        if (
            not isinstance(ranks, frozenset)
            or not ranks
            or any(type(rank) is not int or rank < 0 for rank in ranks)
        ):
            raise ValueError("A completion barrier needs explicit worker ranks")
        self.job = job
        self.ranks = ranks
        self._received: dict[int, RankCompletion] = {}

    def update(self, metadata: UMBPWorkerMetadata) -> None:
        received = dict(self._received)
        for rank, result in metadata.completions.get(self.job.id, {}).items():
            if (
                type(rank) is not int
                or rank not in self.ranks
                or len(result.successes) != self.job.num_objects
            ):
                raise ValueError("UMBP completion disagrees with the submitted job")
            if rank in received and received[rank] != result:
                raise ValueError("Conflicting UMBP completion for the same rank/job")
            received[rank] = result
        self._received = received

    @property
    def done(self) -> bool:
        return self._received.keys() == self.ranks

    def snapshot(self) -> UMBPWorkerMetadata:
        return UMBPWorkerMetadata({self.job.id: dict(self._received)})

    @property
    def succeeded(self) -> bool:
        return self.done and all(result.succeeded for result in self._received.values())

    @property
    def successes(self) -> tuple[bool, ...]:
        if not self.done:
            raise RuntimeError("Cannot publish results before every rank completes")
        return tuple(
            all(
                not result.cancelled and result.successes[i]
                for result in self._received.values()
            )
            for i in range(self.job.num_objects)
        )


@dataclass(frozen=True)
class JobOutcome:
    """Scheduler-authorized finalization, after every rank's last native access."""

    job: TransferJob | LookupJob | ControlJob
    successes: tuple[bool, ...]

    def __post_init__(self) -> None:
        RankCompletion(self.successes)
        if not isinstance(self.job, TransferJob | LookupJob | ControlJob) or (
            len(self.successes) != self.job.num_objects
        ):
            raise ValueError("Outcome must describe every object in its job")


@dataclass(frozen=True)
class UMBPConnectorMetadata(KVConnectorMetadata):
    """One ordered scheduler step; finalizations precede new job admission."""

    epoch: str
    jobs: tuple[TransferJob | LookupJob | ControlJob, ...] = ()
    cancelled: tuple[TransferId, ...] = ()
    finalized: tuple[JobOutcome, ...] = ()
    worker_generations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        TransferId(self.epoch, 0)
        if not isinstance(self.worker_generations, tuple) or any(
            not isinstance(value, str) or not value for value in self.worker_generations
        ):
            raise ValueError("Worker generations must be immutable nonempty strings")
        for values, types in (
            (self.jobs, (TransferJob, LookupJob, ControlJob)),
            (self.cancelled, (TransferId,)),
            (self.finalized, (JobOutcome,)),
        ):
            if not isinstance(values, tuple) or any(
                not isinstance(value, types) for value in values
            ):
                raise ValueError("Connector metadata must contain immutable job tuples")
        ids = tuple(job.id for job in self.jobs)
        if any(a.sequence >= b.sequence for a, b in zip(ids, ids[1:])):
            raise ValueError("Jobs must have strictly increasing sequences")
        all_ids = (
            *ids,
            *self.cancelled,
            *(outcome.job.id for outcome in self.finalized),
        )
        if any(job_id.epoch != self.epoch for job_id in all_ids):
            raise ValueError("Connector metadata mixes engine generations")
        final_ids = tuple(outcome.job.id for outcome in self.finalized)
        if len(set(final_ids)) != len(final_ids) or set(ids).intersection(final_ids):
            raise ValueError("A job cannot be finalized twice or before dispatch")
