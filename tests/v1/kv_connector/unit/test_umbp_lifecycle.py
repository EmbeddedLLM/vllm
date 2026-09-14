# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real cache-manager ownership plus CPU workers; no model/native MoRI claims."""

import threading
import time
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from tests.v1.kv_connector.unit.test_kv_connector_lifecycle import (
    _make_empty_scheduler_output,
)
from tests.v1.kv_connector.unit.test_umbp_layout import allocation, attention_spec
from tests.v1.kv_connector.unit.test_umbp_store import NativeStore
from tests.v1.kv_connector.unit.test_umbp_worker import BlockingNativeStore, Fence
from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.key import UMBPKeySpace
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.layout import (
    CacheTopology,
    UMBPLayout,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.lifecycle import (
    UMBPWorkerLifecycle,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.metadata import (
    BlockKey,
    BlockTransfer,
    JobOutcome,
    RankCompletion,
    TransferId,
    UMBPConnectorMetadata,
    UMBPWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPTransferScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import UMBPStore
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import UMBPTransferWorker
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector
from vllm.v1.worker.kv_connector_model_runner_mixin import KVConnectorModelRunnerMixin


def collect(worker, job):
    deadline = time.monotonic() + 5
    metadata = UMBPWorkerMetadata()
    while time.monotonic() < deadline:
        metadata = metadata.aggregate(worker.build_connector_worker_meta())
        if job.id in metadata.completions:
            return metadata
        threading.Event().wait(0.001)
    pytest.fail("Worker did not emit its completion")


def feedback(scheduler, metadata):
    return scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=metadata)
    )


@pytest.fixture
def setup():
    resources = []

    def create(*, limit=8, worker_limit=8, store_limit=8, natives=None):
        groups = [KVCacheGroupSpec([name], attention_spec()) for name in ("a", "b")]
        cfg, _, _ = allocation(groups, capacity=8)
        manager = KVCacheManager(cfg, 64, 4, 4)
        scheduler = UMBPTransferScheduler(
            manager, epoch="engine", ranks=frozenset({0, 1}), max_pending=limit
        )
        workers = []
        natives = natives or [NativeStore(), NativeStore()]
        fences: list[Fence] = []
        for rank, native in enumerate(natives):
            cfg, _, caches = allocation(groups, capacity=8)
            layout = UMBPLayout(cfg, caches, CacheTopology(tp_size=2))
            keys = UMBPKeySpace("test", "model", "revision", layout.identity, "sha256")
            store = UMBPStore(
                native, SimpleNamespace(CPU="cpu", GPU="gpu"), max_pending=store_limit
            )
            worker = UMBPTransferWorker(
                store, layout, keys, epoch="engine", rank=rank, max_pending=worker_limit
            )
            workers.append(UMBPWorkerLifecycle(worker, max_pending=limit))
            resources.append((native, workers[-1], fences))

        def request(name="request", *, load=False, allocate=True, block_hash=b"prefix"):
            req = Request(
                name,
                [1, 2, 3, 4],
                SamplingParams(max_tokens=1),
                None,
                block_hasher=lambda _: [BlockHash(block_hash)],
            )
            if not allocate:
                return req, ()
            assert manager.allocate_slots(req, 4, delay_cache_blocks=load) is not None
            blocks = manager.get_block_ids(name)
            return req, tuple(
                BlockTransfer(block_hash, group, ids[0])
                for group, ids in enumerate(blocks)
            )

        def dispatch(metadata=None, *, ready=True):
            metadata = metadata or scheduler.build_connector_meta()
            for worker in workers:
                fence = Fence(ready)
                fences.append(fence)
                worker.start_step(metadata, fence)
            return metadata

        return SimpleNamespace(
            manager=manager,
            scheduler=scheduler,
            workers=workers,
            natives=natives,
            request=request,
            dispatch=dispatch,
            fences=fences,
        )

    yield create
    for native, worker, fences in resources:
        native.release.set()
        for fence in fences:
            fence.event.set()
        worker.close()


def test_store_pins_survive_request_free_until_every_rank_finishes(setup):
    s = setup()
    req, blocks = s.request()
    job = s.scheduler.transfer(req, "store", blocks)
    s.dispatch(ready=False)
    s.manager.free(req)
    pool = s.manager.block_pool
    ids = {block.block_id for block in blocks}
    other = pool.get_new_blocks(pool.get_num_free_blocks())
    assert ids.isdisjoint(block.block_id for block in other)
    assert all(pool.blocks[index].ref_cnt == 1 for index in ids)
    s.fences[0].event.set()
    first = collect(s.workers[0], job)
    assert feedback(s.scheduler, first) == ()
    assert feedback(s.scheduler, first) == ()
    assert s.scheduler.has_pending_block_frees() and pool.get_num_free_blocks() == 0
    s.fences[1].event.set()
    outcomes = feedback(s.scheduler, collect(s.workers[1], job))
    assert outcomes[0].successes == (True, True)
    assert pool.get_num_free_blocks() == 2
    assert s.scheduler.has_pending_push_work()  # Finalization still needs a step.
    s.dispatch()
    for worker in s.workers:
        assert not worker.get_transfer_results({req.request_id}).finished_sending
        feedback(s.scheduler, worker.build_connector_worker_meta())
    assert not s.scheduler.has_pending_push_work()
    pool.free_blocks(other)


def test_receive_failure_is_published_only_after_all_ranks_complete(setup):
    blocking = BlockingNativeStore()
    blocking.release.clear()
    s = setup(natives=[NativeStore(), blocking])
    req, blocks = s.request(load=True)
    job = s.scheduler.transfer(req, "load", blocks)
    s.dispatch()
    assert blocking.entered.wait(5)
    assert feedback(s.scheduler, collect(s.workers[0], job)) == ()
    assert not s.workers[0].get_transfer_results(set()).finished_recving
    assert not s.workers[0].get_block_ids_with_load_errors()
    assert all(s.manager.block_pool.blocks[b.block_id].ref_cnt == 2 for b in blocks)
    blocking.release.set()
    assert feedback(s.scheduler, collect(s.workers[1], job))[0].successes == (
        False,
        False,
    )
    final = s.dispatch()
    assert len(final.finalized) == 1
    for worker in s.workers:
        assert worker.get_transfer_results(set()).finished_recving == {req.request_id}
        assert worker.get_block_ids_with_load_errors() == {b.block_id for b in blocks}
        assert not worker.get_transfer_results(set()).finished_recving
        assert not worker.get_block_ids_with_load_errors()
        feedback(s.scheduler, worker.build_connector_worker_meta())
    s.manager.free(req)


@pytest.mark.parametrize("before_dispatch", [False, True])
def test_cancelled_load_drains_without_publishing_or_reusing_its_blocks(
    setup, before_dispatch
):
    s = setup()
    req, blocks = s.request(load=True)
    job = s.scheduler.transfer(req, "load", blocks)
    if not before_dispatch:
        s.dispatch(ready=False)
    s.scheduler.cancel_request(req)
    s.manager.free(req)
    s.dispatch(ready=False)
    for worker in s.workers:
        assert not worker.build_connector_worker_meta().completions
    assert all(s.manager.block_pool.blocks[b.block_id].ref_cnt == 1 for b in blocks)
    for fence in s.fences:
        fence.event.set()
    for worker in s.workers:
        feedback(s.scheduler, collect(worker, job))
    s.dispatch()
    for worker in s.workers:
        worker.get_transfer_results(set())
        worker.get_block_ids_with_load_errors()
        feedback(s.scheduler, worker.build_connector_worker_meta())
    assert all(s.manager.block_pool.blocks[b.block_id].ref_cnt == 0 for b in blocks)
    assert all(not native.data for native in s.natives)


def test_store_requires_actual_cache_identity_and_load_cannot_overwrite_local_hit(
    setup,
):
    s = setup()
    req, blocks = s.request()
    with pytest.raises(ValueError, match="cache record"):
        s.scheduler.transfer(req, "store", (replace(blocks[0], block_hash=b"wrong"),))
    with pytest.raises(ValueError, match="exclusively owned"):
        s.scheduler.transfer(req, "load", blocks)
    with pytest.raises(ValueError, match="non-null"):
        s.scheduler.transfer(req, "load", (replace(blocks[0], block_id=0),))
    other, destinations = s.request("other", load=True)
    with pytest.raises(ValueError, match="exclusively owned"):
        s.scheduler.transfer(req, "load", destinations)
    assert all(
        s.manager.block_pool.blocks[b.block_id].ref_cnt == 1
        for b in (*blocks, *destinations)
    )
    s.manager.free(req)
    s.manager.free(other)


def test_lookup_checks_each_private_worker_pool_and_combines_objects_separately(setup):
    s = setup()
    req, blocks = s.request()
    store = s.scheduler.transfer(req, "store", blocks)
    s.dispatch()
    for worker in s.workers:
        feedback(s.scheduler, collect(worker, store))
    s.dispatch()
    # Leave one group's key only on rank 0: a rank-0 lookup is not a full hit.
    missing_key = next(key for key in s.natives[1].data if ":g1:" in key)
    del s.natives[1].data[missing_key]
    lookup = s.scheduler.lookup(
        req, tuple(BlockKey(b.block_hash, b.group_id) for b in blocks)
    )
    s.dispatch()
    assert feedback(s.scheduler, collect(s.workers[0], lookup)) == ()
    outcomes = feedback(s.scheduler, collect(s.workers[1], lookup))
    assert outcomes[0].successes == (True, False)
    s.dispatch()
    assert all(not w.get_transfer_results(set()).finished_recving for w in s.workers)
    s.manager.free(req)


def test_job_pressure_is_bounded_and_rejected_rank_still_waits_for_compute(setup):
    s = setup(limit=2, worker_limit=1)
    req, blocks = s.request()
    first = s.scheduler.transfer(req, "store", (blocks[0],))
    second = s.scheduler.transfer(req, "store", (blocks[1],))
    assert s.scheduler.lookup(req, (BlockKey(b"key", 0),)) is None
    s.dispatch(ready=False)
    for worker in s.workers:
        assert not worker.build_connector_worker_meta().completions
    for fence in s.fences:
        fence.event.set()
    results = UMBPWorkerMetadata()
    deadline = time.monotonic() + 5
    while len(results.completions) != 2 or any(
        len(ranks) != 2 for ranks in results.completions.values()
    ):
        assert time.monotonic() < deadline
        for worker in s.workers:
            results = results.aggregate(worker.build_connector_worker_meta())
        threading.Event().wait(0.001)
    outcomes = {outcome.job.id: outcome for outcome in feedback(s.scheduler, results)}
    assert outcomes[first.id].successes == (True,)
    assert outcomes[second.id].successes == (False,)
    assert s.scheduler.lookup(req, (BlockKey(b"key", 0),)) is None
    s.dispatch()
    for worker in s.workers:
        feedback(s.scheduler, worker.build_connector_worker_meta())
    assert s.scheduler.lookup(req, (BlockKey(b"key", 0),)) is not None
    s.manager.free(req)


def test_reused_request_id_cannot_cancel_new_incarnation_and_stale_receipts_cannot_free(
    setup,
):
    s = setup()
    old, blocks = s.request()
    old_job = s.scheduler.transfer(old, "store", blocks)
    s.manager.free(old)
    new, targets = s.request(load=True)
    new_job = s.scheduler.transfer(new, "load", targets)
    s.scheduler.cancel_request(old)
    metadata = s.dispatch()
    assert metadata.cancelled == (old_job.id,)
    stale = UMBPWorkerMetadata(
        {
            TransferId("old-engine", new_job.id.sequence): {
                0: RankCompletion((True, True)),
                1: RankCompletion((True, True)),
            }
        }
    )
    assert feedback(s.scheduler, stale) == ()
    assert all(s.manager.block_pool.blocks[b.block_id].ref_cnt == 2 for b in targets)
    s.manager.free(new)


def test_invalid_feedback_batch_cannot_partially_release_unrelated_jobs(setup):
    s = setup()
    req, blocks = s.request()
    first = s.scheduler.transfer(req, "store", (blocks[0],))
    second = s.scheduler.transfer(req, "store", (blocks[1],))
    s.scheduler.build_connector_meta()
    bad = UMBPWorkerMetadata(
        {
            first.id: {0: RankCompletion((True,)), 1: RankCompletion((True,))},
            second.id: {2: RankCompletion((True,))},
        }
    )
    with pytest.raises(ValueError, match="disagrees"):
        feedback(s.scheduler, bad)
    assert all(s.manager.block_pool.blocks[b.block_id].ref_cnt == 2 for b in blocks)
    s.manager.free(req)


def test_finalization_cannot_precede_feedback_or_reverse_a_local_failure(setup):
    s = setup()
    req, blocks = s.request(load=True)
    job = s.scheduler.transfer(req, "load", blocks)
    s.dispatch(ready=False)
    final = UMBPConnectorMetadata("engine", finalized=(JobOutcome(job, (True, True)),))
    with pytest.raises(ValueError, match="before local completion"):
        s.workers[0].start_step(final, Fence())
    for fence in s.fences:
        fence.event.set()
    collect(s.workers[0], job)
    with pytest.raises(ValueError, match="contradicts"):
        s.workers[0].start_step(final, Fence())
    s.manager.free(req)


@pytest.mark.parametrize("runner", ["v1", "v2"])
def test_real_model_runner_collector_reports_finalized_receive_and_errors_together(
    setup,
    runner,
):
    s = setup()
    req, blocks = s.request(load=True)
    job = s.scheduler.transfer(req, "load", blocks)
    s.dispatch()
    for worker in s.workers:
        feedback(s.scheduler, collect(worker, job))
    metadata = s.scheduler.build_connector_meta()
    worker = s.workers[0]
    connector = MagicMock(spec=KVConnectorBase_V1)
    connector.start_load_kv.side_effect = lambda _: worker.start_step(metadata, Fence())
    connector.get_transfer_results.side_effect = worker.get_transfer_results
    connector.get_block_ids_with_load_errors.side_effect = (
        worker.get_block_ids_with_load_errors
    )
    connector.build_connector_worker_meta.side_effect = (
        worker.build_connector_worker_meta
    )
    connector.get_kv_connector_stats.return_value = None
    connector.get_kv_connector_kv_cache_events.return_value = None
    output = _make_empty_scheduler_output()
    output.kv_connector_metadata = metadata
    if runner == "v1":
        module = "vllm.v1.worker.kv_connector_model_runner_mixin"
        with (
            patch(f"{module}.get_kv_transfer_group", return_value=connector),
            patch(f"{module}.get_forward_context", return_value=None),
            KVConnectorModelRunnerMixin._get_kv_connector_output(output) as result,
        ):
            pass
    else:
        module = "vllm.v1.worker.gpu.kv_connector"
        with (
            patch(f"{module}.get_kv_transfer_group", return_value=connector),
            patch(f"{module}.get_forward_context", return_value=None),
            patch(f"{module}.is_forward_context_available", return_value=True),
        ):
            adapter = ActiveKVConnector(MagicMock(), {})
            result = adapter.no_forward(output).kv_connector_output
            assert result is not None
    assert result.finished_recving == {req.request_id}
    assert result.invalid_block_ids == {b.block_id for b in blocks}
    assert not result.failed_recving  # Unique GPU IDs preserve local-prefix fallback.
    assert result.kv_connector_worker_meta.retired == {job.id: frozenset({0})}
    connector.clear_connector_metadata.assert_called_once()
    s.manager.free(req)


def test_failed_load_ids_cannot_be_reused_until_all_error_snapshots_are_consumed(setup):
    s = setup()
    req, blocks = s.request(load=True)
    job = s.scheduler.transfer(req, "load", blocks)
    s.dispatch()
    for worker in s.workers:
        feedback(s.scheduler, collect(worker, job))
    s.manager.free(req)
    pool = s.manager.block_pool
    ids = {block.block_id for block in blocks}
    other = pool.get_new_blocks(pool.get_num_free_blocks())
    assert ids.isdisjoint(block.block_id for block in other)
    assert all(pool.blocks[index].ref_cnt == 1 for index in ids)
    with pytest.raises(ValueError, match="Retirement disagrees"):
        feedback(s.scheduler, UMBPWorkerMetadata(retired={job.id: frozenset({0, 1})}))
    s.dispatch()
    for rank, worker in enumerate(s.workers):
        assert worker.get_transfer_results(set()).finished_recving == {req.request_id}
        assert worker.get_block_ids_with_load_errors() == ids
        metadata = worker.build_connector_worker_meta()
        feedback(s.scheduler, metadata)
        feedback(s.scheduler, metadata)  # Replay cannot count as another rank.
        assert pool.get_num_free_blocks() == (2 if rank == 1 else 0)
    assert not s.scheduler.has_pending_push_work()
    assert ids == {block.block_id for block in pool.get_new_blocks(2)}
    pool.free_blocks([*other, *(pool.blocks[index] for index in ids)])


def test_only_one_outstanding_receive_per_request_can_publish_a_completion(setup):
    s = setup()
    req, blocks = s.request(load=True)
    job = s.scheduler.transfer(req, "load", (blocks[0],))
    assert s.scheduler.transfer(req, "load", (blocks[1],)) is None
    s.dispatch()
    for worker in s.workers:
        feedback(s.scheduler, collect(worker, job))
    assert s.scheduler.transfer(req, "load", (blocks[1],)) is None
    s.dispatch()
    for worker in s.workers:
        worker.get_transfer_results(set())
        worker.get_block_ids_with_load_errors()
        feedback(s.scheduler, worker.build_connector_worker_meta())
    assert s.scheduler.transfer(req, "load", (blocks[1],)) is not None
    s.manager.free(req)


def test_cancelled_lookup_drains_while_native_pressure_retries_the_next_job(setup):
    class CountingNative(NativeStore):
        def __init__(self):
            super().__init__()
            self.lookups = []
            self.release.clear()

        def batch_exists(self, keys):
            self.lookups.append(keys)
            return super().batch_exists(keys)

    natives = [CountingNative(), CountingNative()]
    s = setup(limit=2, store_limit=1, natives=natives)
    first_req, _ = s.request("first", allocate=False)
    second_req, _ = s.request("second", allocate=False)
    first = s.scheduler.lookup(first_req, (BlockKey(b"first", 0),))
    second = s.scheduler.lookup(second_req, (BlockKey(b"second", 0),))
    s.dispatch()
    assert all(native.entered.wait(5) for native in natives)
    s.scheduler.cancel_request(first_req)
    s.dispatch()
    for worker in s.workers:
        assert not worker.build_connector_worker_meta().completions
    assert all(len(native.lookups) == 1 for native in natives)
    for native in natives:
        native.release.set()
    metadata = UMBPWorkerMetadata()
    for worker in s.workers:
        metadata = metadata.aggregate(collect(worker, second))
    assert all(result.cancelled for result in metadata.completions[first.id].values())
    assert all(
        not result.cancelled for result in metadata.completions[second.id].values()
    )
    assert all(len(native.lookups) == 2 for native in natives)
    assert len(feedback(s.scheduler, metadata)) == 2
    s.dispatch()
    for worker in s.workers:
        feedback(s.scheduler, worker.build_connector_worker_meta())
    assert not s.scheduler.has_pending_push_work()
