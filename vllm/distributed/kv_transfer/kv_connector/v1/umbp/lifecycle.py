# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side step integration with globally authorized receive publication.

The model runner supplies a compute event recorded after this step's GPU work
and cache zeroing. Loads are asynchronous: they never feed the current forward.
Local completion reports only a job receipt. A later scheduler finalization,
after every rank completes, permits finished_recving/error-block publication.
"""

from concurrent.futures import Future
from dataclasses import dataclass

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorTransferResults
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    LookupJob,
    RankCompletion,
    TransferId,
    TransferJob,
    UMBPConnectorMetadata,
    UMBPWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import StoreBusyError
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import (
    ComputeFence,
    UMBPTransferWorker,
)


@dataclass
class _TrackedJob:
    job: TransferJob | LookupJob
    fence: ComputeFence
    future: Future[tuple[bool, ...]] | None = None
    completion: RankCompletion | None = None
    cancelled: bool = False
    rejected: bool = False
    reported: bool = False


class UMBPWorkerLifecycle:
    """Use the existing model-runner transfer-results/metadata snapshot APIs.

    Owns the transfer worker. Retains bounded local terminal receipts until
    scheduler finalization, so a replay cannot restart retired native I/O.
    This class does not choose request boundaries or advertise a P/D handoff.
    """

    def __init__(self, worker: UMBPTransferWorker, *, max_pending: int = 8) -> None:
        if type(max_pending) is not int or max_pending <= 0:
            raise ValueError("Pending job limit must be positive")
        self._worker = worker
        self._max_pending = max_pending
        self._highest_sequence = -1
        self._jobs: dict[TransferId, _TrackedJob] = {}
        self._finished_recving: set[str] = set()
        self._invalid_blocks: set[int] = set()
        self._retired: dict[TransferId, frozenset[int]] = {}
        self._closed = False

    def start_step(self, metadata: UMBPConnectorMetadata, fence: ComputeFence) -> None:
        if self._closed:
            raise RuntimeError("UMBP worker lifecycle is closed")
        if metadata.epoch != self._worker.epoch:
            raise ValueError("Metadata belongs to a different engine generation")
        for outcome in metadata.finalized:
            tracked = self._jobs.get(outcome.job.id)
            if tracked is None:
                if outcome.job.id.sequence > self._highest_sequence:
                    raise ValueError("Finalization refers to an unknown future job")
                continue
            result = tracked.completion
            if tracked.job != outcome.job or result is None or not tracked.reported:
                raise ValueError(
                    "Finalization arrived before local completion feedback"
                )
            if any(
                success and (result.cancelled or not local)
                for success, local in zip(
                    outcome.successes, result.successes, strict=True
                )
            ):
                raise ValueError("Finalization contradicts the local rank's failure")
            if isinstance(tracked.job, TransferJob) and tracked.job.operation == "load":
                self._finished_recving.add(tracked.job.request_id)
                self._invalid_blocks.update(
                    block.block_id
                    for block, success in zip(
                        tracked.job.blocks, outcome.successes, strict=True
                    )
                    if not success
                )
            del self._jobs[outcome.job.id]
            self._retired[outcome.job.id] = frozenset({self._worker.rank})

        new_jobs = [job for job in metadata.jobs if job.id not in self._jobs]
        if len(self._jobs) + len(new_jobs) > self._max_pending:
            raise ValueError("Scheduler exceeded the agreed outstanding-job limit")
        for job in metadata.jobs:
            if tracked := self._jobs.get(job.id):
                if tracked.job != job:
                    raise ValueError("Job ID reused with different contents")
                continue
            if job.id.sequence <= self._highest_sequence:
                raise ValueError("A retired job cannot restart native I/O")
            tracked = _TrackedJob(job, fence)
            self._jobs[job.id] = tracked
            self._highest_sequence = job.id.sequence
            if isinstance(job, TransferJob):
                try:
                    tracked.rejected = not self._worker.submit(job, fence)
                except (ValueError, RuntimeError):
                    tracked.rejected = True
        for job_id in metadata.cancelled:
            if tracked := self._jobs.get(job_id):
                tracked.cancelled = True
                if isinstance(tracked.job, TransferJob):
                    self._worker.cancel(job_id)
                elif tracked.future is not None:
                    tracked.future.cancel()
        self._poll()

    def _poll(self) -> None:
        receipts = self._worker.poll()
        for job_id, ranks in receipts.completions.items():
            tracked = self._jobs[job_id]
            tracked.completion = ranks[self._worker.rank]
        for tracked in self._jobs.values():
            if tracked.completion is not None:
                continue
            failed = (False,) * len(tracked.job.blocks)
            if tracked.rejected:
                if tracked.fence.query():
                    tracked.completion = RankCompletion(failed, tracked.cancelled)
            elif isinstance(tracked.job, LookupJob):
                if tracked.future is None:
                    if tracked.cancelled:
                        tracked.completion = RankCompletion(failed, True)
                        continue
                    try:
                        tracked.future = self._worker.lookup(tracked.job.blocks)
                    except StoreBusyError:
                        continue
                    except Exception:
                        tracked.completion = RankCompletion(failed)
                if tracked.future is not None and tracked.future.done():
                    try:
                        successes = tracked.future.result()
                    except Exception:
                        successes = failed
                    tracked.completion = RankCompletion(successes, tracked.cancelled)

    def get_transfer_results(
        self, finished_req_ids: set[str]
    ) -> KVConnectorTransferResults:
        # Store jobs own their block references independently of request free;
        # no sending request ID is ever emitted by this shared lifecycle.
        self._poll()
        result = KVConnectorTransferResults(finished_recving=self._finished_recving)
        self._finished_recving = set()
        return result

    def get_block_ids_with_load_errors(self) -> set[int]:
        result = self._invalid_blocks
        self._invalid_blocks = set()
        return result

    def build_connector_worker_meta(self) -> UMBPWorkerMetadata:
        self._poll()
        receipts = {}
        for job_id, tracked in self._jobs.items():
            if tracked.completion is not None and not tracked.reported:
                receipts[job_id] = {self._worker.rank: tracked.completion}
                tracked.reported = True
        retired = self._retired
        self._retired = {}
        return UMBPWorkerMetadata(receipts, retired)

    def close(self) -> None:
        self._closed = True
        for tracked in self._jobs.values():
            if tracked.future is not None:
                tracked.future.cancel()
            if isinstance(tracked.job, TransferJob) and tracked.completion is None:
                tracked.fence.synchronize()
        self._worker.close()
        self._jobs.clear()
        self._finished_recving.clear()
        self._invalid_blocks.clear()
        self._retired.clear()
