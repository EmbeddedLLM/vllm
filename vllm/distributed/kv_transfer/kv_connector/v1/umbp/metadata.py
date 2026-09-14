# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Generation-scoped transfer jobs and idempotent per-rank acknowledgements."""

from dataclasses import dataclass, field
from typing import Literal

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorWorkerMetadata


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
class BlockTransfer:
    """A vLLM-owned block hash and the allocated group/block it maps to."""

    block_hash: bytes
    group_id: int
    block_id: int

    def __post_init__(self) -> None:
        if not isinstance(self.block_hash, bytes) or not self.block_hash:
            raise ValueError("A transfer needs a nonempty vLLM block hash")
        if any(type(n) is not int or n < 0 for n in (self.group_id, self.block_id)):
            raise ValueError("Group and block IDs must be nonnegative integers")


@dataclass(frozen=True)
class TransferJob:
    """One atomic scheduler ownership unit, potentially spanning cache groups."""

    id: TransferId
    request_id: str
    operation: Literal["load", "store"]
    blocks: tuple[BlockTransfer, ...]

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
        return UMBPWorkerMetadata(merged)


class CompletionBarrier:
    """Retain scheduler ownership until every required rank finishes the job.

    A failed rank does not permit early release: another rank may still be
    accessing the retained blocks. Receipts may arrive in different steps.
    """

    def __init__(self, job: TransferJob, ranks: frozenset[int]) -> None:
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
                or len(result.successes) != len(self.job.blocks)
            ):
                raise ValueError("UMBP completion disagrees with the submitted job")
            if rank in received and received[rank] != result:
                raise ValueError("Conflicting UMBP completion for the same rank/job")
            received[rank] = result
        self._received = received

    @property
    def done(self) -> bool:
        return self._received.keys() == self.ranks

    @property
    def succeeded(self) -> bool:
        return self.done and all(result.succeeded for result in self._received.values())
