# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request planning for immutable full-attention/MLA group blocks.

This planner is the ordinary offload path, not the P/D ready-record protocol.
Hybrid checkpoint and non-prefix state must not silently use this dense-prefix
algorithm. The connector rejects those configurations until their planner lands.
"""

import time
from dataclasses import dataclass
from math import lcm

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    BlockKey,
    BlockTransfer,
    JobOutcome,
    LookupJob,
    TransferId,
    UMBPConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPTransferScheduler,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    make_block_hash_with_group_id,
    resolve_block_hashes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus


@dataclass
class _Lookup:
    request: Request
    local: int
    maximum: int
    preemptions: int
    deadline: float
    job: LookupJob | None = None
    hit: int | None = None
    allocated: bool = False
    load_job: TransferId | None = None


class UMBPRequestPlanner:
    """Keep request selection separate from the all-rank ownership protocol."""

    def __init__(
        self,
        manager: KVCacheManager,
        transfers: UMBPTransferScheduler,
        *,
        lookup_timeout: float = 5.0,
    ) -> None:
        self.manager = manager
        self.transfers = transfers
        self._sizes = tuple(
            group.kv_cache_spec.block_size
            for group in manager.kv_cache_config.kv_cache_groups
        )
        self._alignment = lcm(*self._sizes)
        self._timeout = lookup_timeout
        self._requests: dict[str, Request] = {}
        self._lookups: dict[str, _Lookup] = {}
        self._saved: dict[str, set[BlockKey]] = {}
        self._save_cursor: dict[str, list[int]] = {}
        self._store_owners: dict[TransferId, Request] = {}

    def on_new_request(self, request: Request) -> None:
        previous = self._requests.get(request.request_id)
        if previous is not None and previous is not request:
            raise ValueError("Request ID is still owned by another incarnation")
        self._requests[request.request_id] = request

    def _hashes(self, request: Request, group: int):
        return resolve_block_hashes(
            request.block_hashes,
            self.manager.block_pool.hash_block_size,
            self._sizes[group],
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        self.on_new_request(request)
        if not self.manager.prefix_cache_lookup_enabled(request):
            return 0, False
        # Follow the ordinary prefix-cache contract: leave logits to a forward.
        maximum = (request.num_tokens - 1) // self._alignment * self._alignment
        if maximum <= num_computed_tokens:
            return 0, False
        if num_computed_tokens % self._alignment:
            raise ValueError("Dense external lookup requires an aligned local prefix")
        lookup = self._lookups.get(request.request_id)
        if lookup is not None and (
            lookup.local != num_computed_tokens
            or lookup.maximum != maximum
            or lookup.preemptions != request.num_preemptions
        ):
            self.transfers.cancel_request(request)
            lookup = None
        if lookup is None:
            lookup = _Lookup(
                request,
                num_computed_tokens,
                maximum,
                request.num_preemptions,
                time.monotonic() + self._timeout,
            )
            self._lookups[request.request_id] = lookup
        if lookup.allocated:
            return 0, False  # A failed receive must recompute, not retry forever.
        expired = time.monotonic() >= lookup.deadline
        if expired:
            self.transfers.cancel_request(request)
            lookup.hit = lookup.local
        if lookup.hit is not None:
            count = max(0, lookup.hit - num_computed_tokens)
            if not self.transfers.can_admit() and (
                count or (lookup.job is not None and not expired)
            ):
                return None, False  # The lookup must retire before slot reuse.
            return count, count > 0
        if lookup.job is None:
            keys = tuple(
                BlockKey(bytes(hashes[index]), group)
                for group, size in enumerate(self._sizes)
                for hashes in (self._hashes(request, group),)
                for index in range(num_computed_tokens // size, maximum // size)
            )
            lookup.job = self.transfers.lookup(request, keys)
        return None, False

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ) -> None:
        self.on_new_request(request)
        if not num_external_tokens:
            return
        lookup = self._lookups[request.request_id]
        if (
            lookup.allocated
            or lookup.hit is None
            or lookup.hit - lookup.local != num_external_tokens
        ):
            raise ValueError("External allocation disagrees with the lookup result")
        items = tuple(
            BlockTransfer(
                bytes(hashes[index]), group, blocks.blocks[group][index].block_id
            )
            for group, size in enumerate(self._sizes)
            for hashes in (self._hashes(request, group),)
            for index in range(lookup.local // size, lookup.hit // size)
        )
        job = self.transfers.transfer(request, "load", items)
        if job is None:
            # The scheduler calls allocation immediately after matched-token
            # admission on the same thread. Losing that slot is a contract bug.
            raise RuntimeError("UMBP receive admission changed during allocation")
        lookup.allocated = True
        lookup.load_job = job.id

    def _lookup_done(self, outcome: JobOutcome) -> None:
        lookup = self._lookups.get(outcome.job.request_id)
        if lookup is None or lookup.job != outcome.job or lookup.hit is not None:
            return
        hit = lookup.maximum
        assert isinstance(outcome.job, LookupJob)
        offsets = [lookup.local // size for size in self._sizes]
        for key, success in zip(outcome.job.blocks, outcome.successes, strict=True):
            if not success:
                hit = min(hit, offsets[key.group_id] * self._sizes[key.group_id])
            offsets[key.group_id] += 1
        lookup.hit = hit // self._alignment * self._alignment

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        for outcome in self.transfers.update_connector_output(output):
            if isinstance(outcome.job, LookupJob):
                self._lookup_done(outcome)
            elif outcome.job.operation == "load":
                lookup = self._lookups.get(outcome.job.request_id)
                if lookup is not None and lookup.load_job == outcome.job.id:
                    # Do not immediately PUT objects just restored from this
                    # pool. Failed objects remain eligible after recompute.
                    loaded = self._saved.setdefault(outcome.job.request_id, set())
                    loaded.update(
                        BlockKey(block.block_hash, block.group_id)
                        for block, success in zip(
                            outcome.job.blocks, outcome.successes, strict=True
                        )
                        if success
                    )
            elif outcome.job.operation == "store":
                owner = self._store_owners.pop(outcome.job.id, None)
                if owner is not self._requests.get(outcome.job.request_id):
                    continue
                saved = self._saved.get(outcome.job.request_id)
                if saved is not None:
                    for block, success in zip(
                        outcome.job.blocks, outcome.successes, strict=True
                    ):
                        if not success:
                            saved.discard(BlockKey(block.block_hash, block.group_id))
                            self._save_cursor.pop(outcome.job.request_id, None)

    def build_connector_meta(self, output: SchedulerOutput) -> UMBPConnectorMetadata:
        for request_id in output.preempted_req_ids or ():
            if request := self._requests.get(request_id):
                self.transfers.cancel_request(request)
            self._lookups.pop(request_id, None)
            self._saved.pop(request_id, None)
            self._save_cursor.pop(request_id, None)
        for request_id, count in output.num_scheduled_tokens.items():
            request = self._requests[request_id]
            # Scheduler counters advance after this hook. Speculative decoding
            # is rejected at startup until committed-token save planning lands.
            end = min(request.num_computed_tokens + count, request.num_tokens)
            saved = self._saved.setdefault(request_id, set())
            cursors = self._save_cursor.setdefault(request_id, [0] * len(self._sizes))
            groups = self.manager.get_blocks(request_id).blocks
            items = []
            for group, size in enumerate(self._sizes):
                hashes = self._hashes(request, group)
                for index in range(cursors[group], min(end // size, len(hashes))):
                    block = groups[group][index]
                    key = BlockKey(bytes(hashes[index]), group)
                    if key in saved:
                        if cursors[group] == index:
                            cursors[group] += 1
                        continue
                    if block.is_null:
                        continue
                    record = make_block_hash_with_group_id(hashes[index], group)
                    if self.manager.block_pool.cached_block_hash_to_block.contain(
                        record, block.block_id
                    ):
                        items.append(
                            BlockTransfer(key.block_hash, group, block.block_id)
                        )
            if items and (
                job := self.transfers.transfer(request, "store", tuple(items))
            ):
                self._store_owners[job.id] = request
                saved.update(BlockKey(item.block_hash, item.group_id) for item in items)
        return self.transfers.build_connector_meta()

    def request_finished(self, request: Request) -> None:
        # Finished stores retain their own references; aborts suppress them.
        if request.status in (
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_ERROR,
        ):
            self.transfers.cancel_request(request)
        self._requests.pop(request.request_id, None)
        self._lookups.pop(request.request_id, None)
        self._saved.pop(request.request_id, None)
        self._save_cursor.pop(request.request_id, None)
