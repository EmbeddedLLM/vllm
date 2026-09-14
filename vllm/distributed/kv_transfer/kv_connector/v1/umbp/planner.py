# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request planning for dense KV and validated recurrent checkpoints.

Ordinary offload and optional P/D handoff share the transfer ownership ledger.
Recurrent saves use the scheduler's exact boundary offers, never mutable table
positions. Hybrid P/D requires separate mandatory-export planning.
"""

import time
from dataclasses import dataclass
from math import lcm
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    BlockKey,
    BlockTransfer,
    ControlJob,
    JobOutcome,
    LookupJob,
    TransferId,
    UMBPConnectorMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.pd import UMBPHandoffPlanner
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPTransferScheduler,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    make_block_hash_with_group_id,
    resolve_block_hashes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import MambaSpec
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
        handoff: UMBPHandoffPlanner | None = None,
    ) -> None:
        self.manager = manager
        self.transfers = transfers
        self.handoff = handoff
        self._sizes = tuple(
            group.kv_cache_spec.block_size
            for group in manager.kv_cache_config.kv_cache_groups
        )
        self._checkpoint_groups = frozenset(
            index
            for index, group in enumerate(manager.kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
        )
        self._hash_units = tuple(
            manager.block_pool.hash_block_size
            if group in self._checkpoint_groups
            else size
            for group, size in enumerate(self._sizes)
        )
        self._alignment = lcm(*self._hash_units)
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
            self._hash_units[group],
        )

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        self.on_new_request(request)
        if self.handoff is not None and self.handoff.is_receiver(request):
            return self.handoff.get_num_new_matched_tokens(request, num_computed_tokens)
        if not self.manager.prefix_cache_lookup_enabled(request):
            return 0, False
        # Follow the ordinary prefix-cache contract: leave logits to a forward.
        maximum = (request.num_tokens - 1) // self._alignment * self._alignment
        if maximum <= num_computed_tokens:
            return 0, False
        if any(
            num_computed_tokens % size
            for group, size in enumerate(self._sizes)
            if group not in self._checkpoint_groups
        ):
            # A fine-grained local hit may end inside an attention block. Do
            # not overwrite its shared prefix with a whole-object receive.
            return 0, False
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
                for group, size in enumerate(self._hash_units)
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
        if self.handoff is not None and self.handoff.is_receiver(request):
            self.handoff.update_state_after_alloc(request, blocks, num_external_tokens)
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
                bytes(hashes[index]),
                group,
                blocks.blocks[group][
                    (lookup.hit - 1) // self._sizes[group]
                    if group in self._checkpoint_groups
                    else index
                ].block_id,
            )
            for group, size in enumerate(self._hash_units)
            for hashes in (self._hashes(request, group),)
            for index in (
                (lookup.hit // size - 1,)
                if group in self._checkpoint_groups
                else range(lookup.local // size, lookup.hit // size)
            )
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
        offsets = [lookup.local // size for size in self._hash_units]
        checkpoints: dict[int, set[int]] = {
            group: set() for group in self._checkpoint_groups
        }
        for key, success in zip(outcome.job.blocks, outcome.successes, strict=True):
            if key.group_id in checkpoints:
                if success:
                    checkpoints[key.group_id].add(
                        (offsets[key.group_id] + 1) * self._hash_units[key.group_id]
                    )
            elif not success:
                hit = min(hit, offsets[key.group_id] * self._sizes[key.group_id])
            offsets[key.group_id] += 1
        hit = hit // self._alignment * self._alignment
        while hit > lookup.local and any(
            hit not in boundaries for boundaries in checkpoints.values()
        ):
            hit -= self._alignment
        lookup.hit = max(lookup.local, hit)

    def _checkpoint_items(
        self, request: Request, offers: list[tuple[int, int, int]], end: int
    ) -> tuple[BlockTransfer, ...]:
        saved = self._saved.get(request.request_id, set())
        items: dict[BlockKey, BlockTransfer] = {}
        for group, block_id, boundary in offers:
            if (
                group not in self._checkpoint_groups
                or boundary <= 0
                or boundary > end
                or boundary % self._hash_units[group]
            ):
                continue
            hashes = self._hashes(request, group)
            index = boundary // self._hash_units[group] - 1
            if index >= len(hashes):
                continue
            key = BlockKey(bytes(hashes[index]), group)
            if key not in saved:
                items.setdefault(key, BlockTransfer(key.block_hash, group, block_id))
        return tuple(items.values())

    def register_finished_partial_tail(
        self, request: Request, offers: list[tuple[int, int, int]]
    ) -> None:
        if request.status in (
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_ERROR,
        ):
            return
        items = self._checkpoint_items(
            request, offers, request.num_computed_tokens - request.num_in_flight_tokens
        )
        if items and (job := self.transfers.transfer(request, "store", items)):
            self._store_owners[job.id] = request
            self._saved.setdefault(request.request_id, set()).update(
                BlockKey(item.block_hash, item.group_id) for item in items
            )

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        for outcome in self.transfers.update_connector_output(output):
            if self.handoff is not None:
                self.handoff.update_outcome(outcome)
            if isinstance(outcome.job, ControlJob):
                continue
            if isinstance(outcome.job, LookupJob):
                self._lookup_done(outcome)
            elif outcome.job.operation == "load":
                lookup = self._lookups.get(outcome.job.request_id)
                if (lookup is not None and lookup.load_job == outcome.job.id) or (
                    self.handoff is not None and self.handoff.owns_receive(outcome.job)
                ):
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
        for request_id in output.finished_sending or ():
            if self.handoff is not None:
                self.handoff.finished_sending(request_id)
            self._forget(request_id)

    def build_connector_meta(self, output: SchedulerOutput) -> UMBPConnectorMetadata:
        for request_id in output.preempted_req_ids or ():
            if request := self._requests.get(request_id):
                self.transfers.cancel_request(request)
            self._lookups.pop(request_id, None)
            self._saved.pop(request_id, None)
            self._save_cursor.pop(request_id, None)
            if self.handoff is not None:
                self.handoff.preempted(request_id)
        if self.handoff is not None:
            self.handoff.build_jobs()
        block_state = output.kv_connector_block_state
        offers = block_state.boundary_state_offloads if block_state else {}
        for request_id in dict.fromkeys((*output.num_scheduled_tokens, *offers)):
            request = self._requests.get(request_id)
            if request is None:
                continue
            count = output.num_scheduled_tokens.get(request_id, 0)
            if self.handoff is not None and self.handoff.is_sender(request):
                continue  # Mandatory export is owned by request_finished.
            # Scheduler counters advance after this hook. Speculative decoding
            # is rejected at startup until committed-token save planning lands.
            end = min(request.num_computed_tokens + count, request.num_tokens)
            saved = self._saved.setdefault(request_id, set())
            cursors = self._save_cursor.setdefault(request_id, [0] * len(self._sizes))
            groups = self.manager.get_blocks(request_id).blocks
            items = []
            for group, size in enumerate(self._sizes):
                if group in self._checkpoint_groups:
                    continue
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
            # Offers are valid in this scheduling pass. Pin now on admission;
            # best-effort backpressure drops the offer, not an unowned block ID
            # deferred to a later pass where its bytes could have changed.
            items.extend(
                self._checkpoint_items(request, offers.get(request_id, []), end)
            )
            if items and (
                job := self.transfers.transfer(request, "store", tuple(items))
            ):
                self._store_owners[job.id] = request
                saved.update(BlockKey(item.block_hash, item.group_id) for item in items)
        return self.transfers.build_connector_meta()

    def request_finished(self, request: Request) -> tuple[bool, dict[str, Any] | None]:
        # Finished stores retain their own references; aborts suppress them.
        if request.status in (
            RequestStatus.FINISHED_ABORTED,
            RequestStatus.FINISHED_ERROR,
        ):
            self.transfers.cancel_request(request)
        result = (
            self.handoff.request_finished(request) if self.handoff else (False, None)
        )
        if not result[0]:
            self._forget(request.request_id)
        return result

    def _forget(self, request_id: str) -> None:
        self._requests.pop(request_id, None)
        self._lookups.pop(request_id, None)
        self._saved.pop(request_id, None)
        self._save_cursor.pop(request_id, None)
