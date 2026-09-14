# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pool-mediated P/D: finish-time export, global readiness, gated receive."""

import time
import uuid
from dataclasses import dataclass
from math import lcm
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    BlockTransfer,
    JobOutcome,
    TransferId,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.protocol import HandoffHandle
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPTransferScheduler,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import resolve_block_hashes
from vllm.v1.request import Request, RequestStatus


@dataclass
class _Export:
    request: Request
    handle: HandoffHandle
    blocks: tuple[BlockTransfer, ...]
    store_job: TransferId | None = None
    stored: bool | None = None
    control_job: TransferId | None = None


@dataclass
class _Receive:
    request: Request
    handle: HandoffHandle
    local: int
    target: int
    load_job: TransferId | None = None


class UMBPHandoffPlanner:
    """Use ordinary request-finish holds and the existing all-rank job ledger.

    Prefix objects can be evicted after readiness. Neither a handle nor a ready
    marker leases those objects. Receives still report failed GETs to vLLM.
    """

    def __init__(
        self,
        manager: KVCacheManager,
        transfers: UMBPTransferScheduler,
        *,
        namespace: str,
        producer: bool,
        consumer: bool,
        timeout: float,
    ) -> None:
        self.manager, self.transfers = manager, transfers
        self.namespace = namespace
        self.producer, self.consumer = producer, consumer
        self.timeout = timeout
        self.sizes = tuple(
            group.kv_cache_spec.block_size
            for group in manager.kv_cache_config.kv_cache_groups
        )
        self.alignment = lcm(*self.sizes)
        self.exports: dict[str, _Export] = {}
        self.receives: dict[str, _Receive] = {}

    def is_sender(self, request: Request) -> bool:
        return (
            self.producer
            and (request.kv_transfer_params or {}).get("do_remote_decode") is True
        )

    def is_receiver(self, request: Request) -> bool:
        return (
            self.consumer
            and (request.kv_transfer_params or {}).get("do_remote_prefill") is True
        )

    def _blocks(
        self, request: Request, blocks: KVCacheBlocks, start: int, end: int
    ) -> tuple[BlockTransfer, ...]:
        return tuple(
            BlockTransfer(
                bytes(hashes[index]), group, blocks.blocks[group][index].block_id
            )
            for group, size in enumerate(self.sizes)
            for hashes in (
                resolve_block_hashes(
                    request.block_hashes, self.manager.block_pool.hash_block_size, size
                ),
            )
            for index in range(start // size, end // size)
        )

    def get_num_new_matched_tokens(
        self, request: Request, local: int
    ) -> tuple[int | None, bool]:
        previous = self.receives.get(request.request_id)
        if previous is not None and previous.load_job is not None:
            return 0, False  # Failed receive recovery must not start a retry loop.
        if not self.manager.prefix_cache_lookup_enabled(request):
            return 0, False
        try:
            handle = HandoffHandle.from_dict(
                (request.kv_transfer_params or {}).get("umbp_handoff")
            )
        except ValueError:
            return 0, False  # Invalid hints do not authorize external KV reuse.
        hash_size = self.manager.block_pool.hash_block_size
        if (
            handle.namespace != self.namespace
            or handle.expires_at_ms <= time.time() * 1000
            or handle.token_boundary % self.alignment
            or handle.token_boundary > len(request.block_hashes) * hash_size
            or request.block_hashes[handle.token_boundary // hash_size - 1].hex()
            != handle.boundary_hash
        ):
            return 0, False
        target = (
            min(handle.token_boundary, request.num_tokens - 1)
            // self.alignment
            * self.alignment
        )
        if target <= local:
            return 0, False
        if local % self.alignment:
            raise ValueError("P/D local prefix must align across dense groups")
        if not self.transfers.can_admit():
            return None, False
        self.receives[request.request_id] = _Receive(request, handle, local, target)
        # As with pull P/D connectors, allocation is for a pending receive,
        # not a claim of servable KV. Workers wait on readiness before GET.
        return target - local, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, count: int
    ) -> None:
        receive = self.receives[request.request_id]
        if (
            receive.request is not request
            or receive.load_job is not None
            or count != receive.target - receive.local
        ):
            raise ValueError("P/D allocation disagrees with the admitted handoff")
        job = self.transfers.transfer(
            request,
            "load",
            self._blocks(request, blocks, receive.local, receive.target),
            handoff=receive.handle,
            readiness_timeout=self.timeout,
        )
        if job is None:
            raise RuntimeError("P/D receive lost its scheduler admission slot")
        receive.load_job = job.id

    def request_finished(self, request: Request) -> tuple[bool, dict[str, Any] | None]:
        self.receives.pop(request.request_id, None)
        if (
            not self.is_sender(request)
            or request.status != RequestStatus.FINISHED_LENGTH_CAPPED
        ):
            return False, None
        computed = min(
            request.num_prompt_tokens,
            request.num_computed_tokens - request.num_in_flight_tokens,
        )
        boundary = computed // self.alignment * self.alignment
        if boundary <= 0:
            return False, None
        handle = HandoffHandle(
            self.namespace,
            self.transfers.epoch,
            uuid.uuid4().hex,
            boundary,
            request.block_hashes[
                boundary // self.manager.block_pool.hash_block_size - 1
            ].hex(),
            int((time.time() + self.timeout) * 1000),
        )
        self.exports[request.request_id] = _Export(
            request,
            handle,
            self._blocks(
                request, self.manager.get_blocks(request.request_id), 0, boundary
            ),
        )
        # Return the handle immediately. True preserves the request allocation
        # until workers return finished_sending after the global control outcome.
        return True, {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "umbp_handoff": handle.to_dict(),
        }

    def build_jobs(self) -> None:
        for export in self.exports.values():
            if export.control_job is not None:
                continue
            expired = time.time() * 1000 >= export.handle.expires_at_ms
            if export.store_job is None and export.stored is None:
                if expired:
                    export.stored = False
                elif job := self.transfers.transfer(
                    export.request, "store", export.blocks
                ):
                    export.store_job = job.id
            if export.stored is not None:
                control = self.transfers.control(
                    export.request,
                    export.handle,
                    "publish" if export.stored and not expired else "release",
                )
                if control is not None:
                    export.control_job = control.id

    def update_outcome(self, outcome: JobOutcome) -> None:
        export = self.exports.get(outcome.job.request_id)
        if export is not None and outcome.job.id == export.store_job:
            # This outcome exists only after every rank's native work ends.
            export.stored = all(outcome.successes)

    def owns_receive(self, job: TransferJob) -> bool:
        receive = self.receives.get(job.request_id)
        return receive is not None and receive.load_job == job.id

    def finished_sending(self, request_id: str) -> None:
        self.exports.pop(request_id, None)

    def preempted(self, request_id: str) -> None:
        self.receives.pop(request_id, None)

    def has_pending(self) -> bool:
        return bool(self.exports)
