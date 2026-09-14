# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Transfer lifetime tests; real CPU buffers and deterministic native barriers.

Jobs must not read before compute fences or release destinations before native
completion. Public submit/poll/cancel/close behavior and per-rank receipts catch
these failures without a GPU. These are not composed vLLM scheduler/P-D tests.
"""

import gc
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.unit.test_umbp_layout import allocation, attention_spec
from tests.v1.kv_connector.unit.test_umbp_store import NativeStore
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.key import UMBPKeySpace
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.layout import (
    CacheTopology,
    UMBPLayout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    BlockTransfer,
    CompletionBarrier,
    RankCompletion,
    TransferId,
    TransferJob,
    UMBPWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import UMBPStore
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import UMBPTransferWorker
from vllm.v1.kv_cache_interface import KVCacheGroupSpec


class Fence:
    def __init__(self, ready=True):
        self.event = threading.Event()
        if ready:
            self.event.set()
        self.error = False

    def query(self):
        if self.error:
            raise RuntimeError("compute event failed")
        return self.event.is_set()

    def synchronize(self):
        assert self.event.wait(5), "compute fence not released"


class BlockingNativeStore(NativeStore):
    def batch_get_ranges_into_ptr(self, *args):
        self.entered.set()
        assert self.release.wait(5), "native read barrier not released"
        return super().batch_get_ranges_into_ptr(*args)


def job(sequence=0, operation="store", *, block_id=1, epoch="engine", blocks=None):
    return TransferJob(
        TransferId(epoch, sequence),
        "request",
        operation,
        blocks or (BlockTransfer(b"prefix", 0, block_id),),
    )


def await_completion(worker, transfer):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        receipt = worker.poll()
        if transfer.id in receipt.completions:
            return receipt
        threading.Event().wait(0.001)
    pytest.fail("transfer did not complete")


@pytest.fixture
def make_worker():
    resources = []

    def create(*, native=None, epoch="engine", rank=0, max_pending=8, store_pending=8):
        native = native or NativeStore()
        config, raw, caches = allocation(
            [
                KVCacheGroupSpec(["attention"], attention_spec()),
                KVCacheGroupSpec(["other-group"], attention_spec()),
            ]
        )
        layout = UMBPLayout(config, caches, CacheTopology(tp_size=2))
        keys = UMBPKeySpace("test", "model", "revision", layout.identity, "sha256")
        store = UMBPStore(
            native,
            SimpleNamespace(CPU="cpu", GPU="gpu"),
            workers=1,
            max_pending=store_pending,
        )
        worker = UMBPTransferWorker(
            store, layout, keys, epoch=epoch, rank=rank, max_pending=max_pending
        )
        resources.append((native, worker))
        return worker, native, store, raw, caches

    yield create
    for native, worker in resources:
        native.release.set()
        worker.close()


def test_separate_workers_restore_only_requested_missing_blocks(make_worker):
    producer, source_native, _, _, source = make_worker(epoch="prefill")
    consumer, target_native, _, target_raw, target = make_worker(epoch="decode")
    target_native.data = source_native.data
    source["attention"][1].fill_(7)
    before = target_raw.clone()
    send = job(epoch="prefill")
    receive = job(operation="load", block_id=2, epoch="decode")
    assert producer.submit(send, Fence())
    assert await_completion(producer, send).completions[send.id][0].succeeded
    assert consumer.submit(receive, Fence())
    assert await_completion(consumer, receive).completions[receive.id][0].succeeded
    torch.testing.assert_close(target["attention"][2], source["attention"][1])
    # The two groups alias the allocation; only physical block 2 is written.
    assert torch.equal(target_raw[:128], before[:128])
    assert torch.equal(target_raw[192:], before[192:])


def test_cancel_before_compute_completion_waits_for_fence_without_native_io(
    make_worker,
):
    worker, native, _, _, _ = make_worker()
    fence = Fence(False)
    transfer = job()
    try:
        assert worker.submit(transfer, fence)
        assert worker.poll().completions == {}
        worker.cancel(transfer.id)
        assert worker.poll().completions == {}
        assert native.data == {}
    finally:
        fence.event.set()
    receipt = await_completion(worker, transfer).completions[transfer.id][0]
    assert receipt.cancelled and not receipt.succeeded
    assert native.data == {}


def test_running_cancel_retains_destination_until_native_read_finishes(make_worker):
    native = BlockingNativeStore()
    worker, _, _, _, caches = make_worker(native=native)
    caches["attention"][1].fill_(9)
    stored = job()
    assert worker.submit(stored, Fence())
    await_completion(worker, stored)
    caches["attention"][1].zero_()
    native.release.clear()
    read = job(1, "load")
    assert worker.submit(read, Fence())
    worker.poll()
    assert native.entered.wait(5)
    worker.cancel(read.id)
    assert worker.poll().completions == {}
    # Even another group aliases these bytes, so group ID is not a fence.
    conflict = job(2, blocks=(BlockTransfer(b"other", 1, 1),))
    assert not worker.submit(conflict, Fence())
    native.release.set()
    result = await_completion(worker, read).completions[read.id][0]
    assert result.cancelled and not result.succeeded
    assert result.successes == (True,)  # Native I/O did run despite cancellation.
    assert bool(torch.all(caches["attention"][1] == 9))
    assert worker.submit(conflict, Fence())
    await_completion(worker, conflict)


def test_native_queue_pressure_retries_without_early_acknowledgement(make_worker):
    worker, native, store, _, _ = make_worker(store_pending=1)
    native.release.clear()
    occupied = store.lookup(("blocked",))
    assert native.entered.wait(5)
    transfer = job()
    assert worker.submit(transfer, Fence())
    assert worker.poll().completions == {}
    assert native.data == {}
    native.release.set()
    occupied.result(5)
    assert await_completion(worker, transfer).completions[transfer.id][0].succeeded


def test_admission_is_bounded_and_generation_replay_cannot_repeat_io(make_worker):
    worker, native, _, _, _ = make_worker(max_pending=1)
    first = job()
    assert worker.submit(first, Fence())
    assert worker.submit(first, Fence())  # Idempotent while pending.
    with pytest.raises(ValueError, match="different contents"):
        worker.submit(replace(first, request_id="different"), Fence())
    assert not worker.submit(job(1), Fence())
    await_completion(worker, first)
    assert worker.poll().completions == {}
    with pytest.raises(ValueError, match="Stale"):
        worker.submit(first, Fence())
    with pytest.raises(ValueError, match="generation"):
        worker.submit(job(1, epoch="old-engine"), Fence())
    # Reusing a request ID is safe: ownership and feedback use the new job ID.
    next_job = job(1)
    assert worker.submit(next_job, Fence())
    assert await_completion(worker, next_job).completions.keys() == {next_job.id}
    assert len(native.data) == 1


def test_fence_error_is_fatal_and_keeps_the_allocation_registered(make_worker):
    worker, native, _, _, _ = make_worker()
    fence = Fence()
    fence.error = True
    transfer = job()
    assert worker.submit(transfer, fence)
    with pytest.raises(RuntimeError, match="compute event failed"):
        worker.poll()
    assert native.registered and not native.data
    fence.error = False
    assert await_completion(worker, transfer).completions[transfer.id][0].succeeded


def test_shutdown_drains_running_native_reads_before_deregistering(make_worker):
    native = BlockingNativeStore()
    worker, _, _, _, _ = make_worker(native=native)
    native.release.clear()
    transfer = job(operation="load")
    assert worker.submit(transfer, Fence())
    worker.poll()
    assert native.entered.wait(5)
    started = threading.Event()

    def close():
        started.set()
        worker.close()

    with ThreadPoolExecutor(1) as executor:
        closing = executor.submit(close)
        try:
            assert started.wait(5)
            assert native.registered and not closing.done()
        finally:
            native.release.set()
        closing.result(5)
    assert not native.registered
    with pytest.raises(RuntimeError, match="closed"):
        worker.submit(job(1), Fence())


def test_all_rank_all_group_barrier_waits_after_partial_failure(make_worker):
    first, _, _, _, _ = make_worker(rank=0)
    second, _, _, _, _ = make_worker(rank=1)
    transfer = job(
        operation="load",
        blocks=(
            BlockTransfer(b"a", 0, 1),
            BlockTransfer(b"b", 1, 2),
        ),
    )
    barrier = CompletionBarrier(transfer, frozenset({0, 1}))
    assert first.submit(transfer, Fence())
    first_meta = await_completion(first, transfer)
    barrier.update(first_meta)
    barrier.update(first_meta)  # Replay is not an extra worker.
    assert not barrier.done and not barrier.succeeded
    assert second.submit(transfer, Fence())
    second_meta = await_completion(second, transfer)
    barrier.update(second_meta)
    assert barrier.done and not barrier.succeeded
    merged = first_meta.aggregate(second_meta).aggregate(first_meta)
    assert set(merged.completions[transfer.id]) == {0, 1}


def test_barrier_requires_every_object_and_ignores_prior_generations():
    transfer = job(blocks=(BlockTransfer(b"a", 0, 1), BlockTransfer(b"b", 1, 2)))
    barrier = CompletionBarrier(transfer, frozenset({0, 1}))
    old = TransferId("old-engine", transfer.id.sequence)
    barrier.update(UMBPWorkerMetadata({old: {0: RankCompletion((True, True))}}))
    assert not barrier.done
    first = UMBPWorkerMetadata({transfer.id: {0: RankCompletion((True, True))}})
    barrier.update(first)
    for invalid in [
        UMBPWorkerMetadata({transfer.id: {1: RankCompletion((True,))}}),
        UMBPWorkerMetadata({transfer.id: {2: RankCompletion((True, True))}}),
        UMBPWorkerMetadata({transfer.id: {0: RankCompletion((True, False))}}),
    ]:
        with pytest.raises(ValueError):
            barrier.update(invalid)
        assert not barrier.done
    conflict = UMBPWorkerMetadata({transfer.id: {0: RankCompletion((True, False))}})
    with pytest.raises(ValueError, match="Conflicting"):
        first.aggregate(conflict)
    barrier.update(UMBPWorkerMetadata({transfer.id: {1: RankCompletion((True, True))}}))
    assert barrier.done and barrier.succeeded


def test_queued_cancel_does_not_wait_for_an_unrelated_running_transfer(make_worker):
    native = BlockingNativeStore()
    worker, _, _, _, _ = make_worker(native=native)
    native.release.clear()
    read = job(operation="load")
    queued = job(1, block_id=2)
    assert worker.submit(read, Fence())
    worker.poll()
    assert native.entered.wait(5)
    assert worker.submit(queued, Fence())
    worker.poll()
    worker.cancel(queued.id)
    receipt = await_completion(worker, queued)
    assert read.id not in receipt.completions
    assert receipt.completions[queued.id][0].cancelled
    assert not native.data
    native.release.set()
    await_completion(worker, read)


def test_native_failure_after_partial_write_fails_every_destination(make_worker):
    class FailedRead(NativeStore):
        def batch_get_ranges_into_ptr(self, *args):
            super().batch_get_ranges_into_ptr(*args)
            raise RuntimeError("native error after changing destination bytes")

    worker, _, _, _, caches = make_worker(native=FailedRead())
    caches["attention"][1].fill_(5)
    stored = job()
    assert worker.submit(stored, Fence())
    await_completion(worker, stored)
    caches["attention"][1].zero_()
    read = job(
        1,
        "load",
        blocks=(
            BlockTransfer(b"prefix", 0, 1),
            BlockTransfer(b"missing", 1, 2),
        ),
    )
    assert worker.submit(read, Fence())
    result = await_completion(worker, read).completions[read.id][0]
    assert result.successes == (False, False)
    assert not result.succeeded
    assert bool(torch.all(caches["attention"][1] == 5))


def test_shutdown_waits_for_deferred_compute_fence_before_releasing_memory(make_worker):
    worker, native, _, _, _ = make_worker()
    fence = Fence(False)
    assert worker.submit(job(), fence)
    started = threading.Event()

    def close():
        started.set()
        worker.close()

    with ThreadPoolExecutor(1) as executor:
        closing = executor.submit(close)
        try:
            assert started.wait(5)
            assert not closing.done() and native.registered
        finally:
            fence.event.set()
        closing.result(5)
    assert not native.registered and not native.data


def test_vllm_output_aggregator_preserves_generation_scoped_rank_receipts():
    from tests.v1.kv_connector.unit.test_output_aggregator import DummyModelRunnerOutput
    from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator

    transfer = job()
    outputs = [DummyModelRunnerOutput(), DummyModelRunnerOutput()]
    for rank, output in enumerate(outputs):
        output.kv_connector_output.kv_connector_worker_meta = UMBPWorkerMetadata(
            {transfer.id: {rank: RankCompletion((True,))}}
        )
    output = KVOutputAggregator(expected_finished_count=2).aggregate(outputs)
    metadata = output.kv_connector_output.kv_connector_worker_meta
    barrier = CompletionBarrier(transfer, frozenset({0, 1}))
    barrier.update(metadata)
    assert barrier.succeeded
    # Job completion is not yet a request handoff: the scheduler must decide it.
    assert not output.kv_connector_output.finished_sending
    assert not output.kv_connector_output.finished_recving


def test_shutdown_releases_the_workers_allocation_owners_after_deregistration():
    config, raw, caches = allocation([KVCacheGroupSpec(["layer"], attention_spec())])
    layout = UMBPLayout(config, caches, CacheTopology())
    owners = [weakref.ref(region.owner) for region in layout.regions]
    keys = UMBPKeySpace("test", "model", "revision", layout.identity, "sha256")
    native = NativeStore()
    store = UMBPStore(native, SimpleNamespace(CPU="cpu", GPU="gpu"))
    worker = UMBPTransferWorker(store, layout, keys, epoch="engine", rank=0)
    del raw, caches, layout
    try:
        gc.collect()
        assert all(owner() is not None for owner in owners)
    finally:
        worker.close()
    gc.collect()
    assert not native.registered
    assert all(owner() is None for owner in owners)
