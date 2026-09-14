# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real cache-manager ownership plus CPU workers; no model/native MoRI claims."""

import threading
import time
from dataclasses import replace
from math import lcm
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tests.v1.kv_connector.unit.test_kv_connector_lifecycle import (
    _make_empty_scheduler_output,
)
from tests.v1.kv_connector.unit.test_umbp_layout import allocation, attention_spec
from tests.v1.kv_connector.unit.test_umbp_store import NativeStore
from tests.v1.kv_connector.unit.test_umbp_worker import BlockingNativeStore, Fence
from tests.v1.kv_connector.unit.utils import (
    create_request,
    create_scheduler,
    create_vllm_config,
)
from vllm import SamplingParams
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.config import UMBPStoreConfig
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector import UMBPConnector
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
    ControlJob,
    JobOutcome,
    LookupJob,
    RankCompletion,
    TransferId,
    TransferJob,
    UMBPConnectorMetadata,
    UMBPWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.pd import UMBPHandoffPlanner
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.protocol import HandoffHandle
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.scheduler import (
    UMBPTransferScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import UMBPStore
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.worker import UMBPTransferWorker
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import BlockHash, get_request_block_hasher
from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVQuantMode
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
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

    def create(*, limit=8, worker_limit=8, store_limit=8, natives=None, epoch="engine"):
        groups = [KVCacheGroupSpec([name], attention_spec()) for name in ("a", "b")]
        cfg, _, _ = allocation(groups, capacity=8)
        manager = KVCacheManager(cfg, 64, 4, 4)
        scheduler = UMBPTransferScheduler(
            manager, epoch=epoch, ranks=frozenset({0, 1}), max_pending=limit
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
                store, layout, keys, epoch=epoch, rank=rank, max_pending=worker_limit
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
            namespace=keys.prefix,
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


@pytest.fixture
def engine(monkeypatch):
    """Real factory, Scheduler, cache manager and worker connector; fake GPU/IO."""
    resources: list[tuple[NativeStore, KVConnectorBase_V1]] = []

    class Event(Fence):
        def __init__(self, **kwargs):
            super().__init__()

        def record(self):
            pass

    monkeypatch.setattr(torch, "Event", Event)
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector.get_tp_group",
        lambda: SimpleNamespace(rank_in_group=0),
    )

    def create(
        *,
        limit=8,
        budget=16,
        timeout=5.0,
        groups=None,
        revision="unit-revision",
        dtype="float16",
        kv_role="kv_both",
        enable_pd=False,
        handoff_timeout=30.0,
        load_failure_policy="recompute",
    ):
        groups = groups or [KVCacheGroupSpec(["a"], attention_spec())]
        cfg, raw, caches = allocation(groups, capacity=32)
        config = create_vllm_config(
            dtype=dtype,
            block_size=lcm(*(group.kv_cache_spec.block_size for group in groups)),
            max_model_len=64,
            max_num_seqs=4,
            max_num_batched_tokens=budget,
            kv_role=kv_role,
            kv_load_failure_policy=load_failure_policy,
            kv_connector="UMBPConnector",
            kv_connector_module_path="vllm.distributed.kv_transfer.kv_connector.v1.umbp.connector",
            kv_connector_extra_config={
                "deployment": "unit",
                "model": "test-model",
                "revision": revision,
                "lookup_timeout": timeout,
                "handoff_timeout": handoff_timeout,
                "enable_pd": enable_pd or kv_role != "kv_both",
                "storage": {
                    "page_size_bytes": 4096,
                    "dram_capacity_bytes": 8192,
                    "max_pending": limit,
                },
            },
        )
        if enable_pd or kv_role != "kv_both":
            extra = config.kv_transfer_config.kv_connector_extra_config
            extra["storage"]["master_address"] = "unit-master:1234"
            extra["nodes"] = [
                {
                    "node_id": f"unit-{len(resources)}",
                    "node_address": "127.0.0.1",
                    "io_engine_host": "127.0.0.1",
                    "peer_service_port": 12000 + len(resources),
                }
            ]
        # Avoid async-scheduler policy defaults in a serialized lifecycle test.
        config.scheduler_config.async_scheduling = False
        native = NativeStore()
        store = UMBPStore(
            native, SimpleNamespace(CPU="cpu", GPU="gpu"), max_pending=limit
        )
        monkeypatch.setattr(
            UMBPStoreConfig, "open_store", lambda *args, **kwargs: store
        )
        worker = KVConnectorFactory.create_connector(
            config, KVConnectorRole.WORKER, cfg
        )
        worker.register_kv_caches(caches)
        resources.append((native, worker))
        scheduler = create_scheduler(
            config, num_blocks=32, kv_cache_config=cfg, hash_block_size=4
        )
        scheduler.connector.set_xfer_handshake_metadata_pp_aware(
            {(0, 0): worker.get_handshake_metadata()}
        )
        jobs = []
        computed: dict[str, int] = {}
        outputs = []
        engine_outputs = []

        def step():
            scheduled = scheduler.schedule()
            jobs.extend(scheduled.kv_connector_metadata.jobs)
            req_ids = list(scheduled.num_scheduled_tokens)
            samples = []
            for req_id, count in scheduled.num_scheduled_tokens.items():
                request = scheduler.requests[req_id]
                computed[req_id] = computed.get(req_id, 0) + count
                end = request.num_computed_tokens
                block_ids = scheduler.kv_cache_manager.get_block_ids(req_id)
                for group, spec_group in enumerate(groups):
                    size = spec_group.kv_cache_spec.block_size
                    for index in range((end - count) // size, (end + size - 1) // size):
                        for name in spec_group.layer_names:
                            caches[name][block_ids[group][index]].fill_(index + 7)
                samples.append([1000] if end >= request.num_tokens else [])
            with (
                patch(
                    "vllm.v1.worker.kv_connector_model_runner_mixin.get_kv_transfer_group",
                    return_value=worker,
                ),
                patch(
                    "vllm.v1.worker.kv_connector_model_runner_mixin.get_forward_context",
                    return_value=None,
                ),
                KVConnectorModelRunnerMixin._get_kv_connector_output(
                    scheduled
                ) as output,
            ):
                pass
            outputs.append(output)
            result = ModelRunnerOutput(
                req_ids=req_ids,
                req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
                sampled_token_ids=samples,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                kv_connector_output=output,
            )
            returned = scheduler.update_from_output(scheduled, result)
            for batch in returned.values():
                engine_outputs.extend(batch.outputs)
            return scheduled

        def until(condition):
            deadline = time.monotonic() + 5
            while not condition():
                assert time.monotonic() < deadline, "Scheduler did not make progress"
                step()
                threading.Event().wait(0.001)

        def drain():
            until(lambda: not scheduler.has_requests())

        def request(request_id, tokens=13):
            return create_request(
                request_id=request_id,
                num_tokens=tokens,
                common_prefix_len=tokens,
                max_tokens=1,
                block_size=4,
            )

        return SimpleNamespace(
            scheduler=scheduler,
            worker=worker,
            native=native,
            store=store,
            cfg=cfg,
            config=config,
            caches=caches,
            raw=raw,
            jobs=jobs,
            computed=computed,
            outputs=outputs,
            engine_outputs=engine_outputs,
            step=step,
            until=until,
            drain=drain,
            request=request,
        )

    yield create
    for native, worker in resources:
        native.release.set()
        worker.shutdown()
        assert not native.registered


@pytest.mark.parametrize("enable_pd", [False, True])
@pytest.mark.parametrize("local_prefix", [0, 4])
def test_full_scheduler_offload_restores_only_the_missing_prefix(
    engine, local_prefix, enable_pd
):
    e = engine(limit=1, enable_pd=enable_pd)
    first = e.request(100)
    e.scheduler.add_request(first)
    e.drain()
    assert e.computed[first.request_id] == 13
    assert len(e.native.data) == 3
    pool = e.scheduler.kv_cache_manager.block_pool
    assert pool.reset_prefix_cache()
    e.raw.fill_(-99)
    if local_prefix:
        seed = e.request(101, tokens=local_prefix)
        e.scheduler.kv_cache_manager.allocate_slots(seed, local_prefix)
        seed_block = e.scheduler.kv_cache_manager.get_block_ids(seed.request_id)[0][0]
        e.caches["a"][seed_block].fill_(7)
        e.scheduler.kv_cache_manager.free(seed)
    e.jobs.clear()
    second = e.request(102)
    e.scheduler.add_request(second)
    e.until(lambda: second.request_id in e.scheduler.finished_recving_kv_req_ids)
    blocks = e.scheduler.kv_cache_manager.get_block_ids(second.request_id)[0]
    for index in range(3):
        assert torch.all(e.caches["a"][blocks[index]] == index + 7)
    e.drain()
    loads = [
        job
        for job in e.jobs
        if isinstance(job, TransferJob) and job.operation == "load"
    ]
    assert len(loads) == 1 and len(loads[0].blocks) == (12 - local_prefix) // 4
    assert e.computed[second.request_id] == 1  # The final-logit token is executed.
    assert pool.get_num_free_blocks() == 31
    assert not e.scheduler.connector.has_pending_block_frees()


def test_full_scheduler_failed_receive_recomputes_without_retry_loop(
    engine, monkeypatch
):
    e = engine()
    first = e.request(110)
    e.scheduler.add_request(first)
    e.drain()
    assert e.scheduler.kv_cache_manager.block_pool.reset_prefix_cache()
    monkeypatch.setattr(
        e.native, "batch_get_ranges_into_ptr", lambda keys, *args: [False] * len(keys)
    )
    e.jobs.clear()
    second = e.request(111)
    e.scheduler.add_request(second)
    e.drain()
    assert e.computed[second.request_id] == 13
    loads = [
        job
        for job in e.jobs
        if isinstance(job, TransferJob) and job.operation == "load"
    ]
    assert len(loads) == 1
    failures = [output for output in e.outputs if output.invalid_block_ids]
    assert len(failures) == 1
    assert second.request_id in failures[0].finished_recving
    assert not e.scheduler.connector.has_pending_push_work()
    assert e.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 31


def test_connector_startup_rejects_mismatched_layout_generation_and_pd_role(engine):
    e = engine()
    connector = UMBPConnector(e.config, KVConnectorRole.SCHEDULER, e.cfg)
    connector.bind_kv_cache_manager(e.scheduler.kv_cache_manager)
    handshake = e.worker.get_handshake_metadata()
    with pytest.raises(ValueError, match="handshake"):
        connector.set_xfer_handshake_metadata({1: handshake})
    connector.set_xfer_handshake_metadata({0: handshake})
    with pytest.raises(RuntimeError, match="generation"):
        connector.set_xfer_handshake_metadata({0: handshake})
    e.config.kv_transfer_config.kv_role = "kv_producer"
    with pytest.raises(ValueError, match="enable_pd"):
        UMBPConnector(e.config, KVConnectorRole.SCHEDULER, e.cfg)


def test_connector_handshake_requires_all_shards_to_agree_on_cache_identity(engine):
    e = engine()
    e.config.parallel_config.tensor_parallel_size = 2
    connector = UMBPConnector(e.config, KVConnectorRole.SCHEDULER, e.cfg)
    connector.bind_kv_cache_manager(e.scheduler.kv_cache_manager)
    first = e.worker.get_handshake_metadata()
    second = replace(
        first, rank=1, generation="another-worker", namespace="incompatible"
    )
    with pytest.raises(ValueError, match="identities"):
        connector.set_xfer_handshake_metadata({0: first, 1: second})
    second = replace(second, namespace=first.namespace)
    connector.set_xfer_handshake_metadata({0: first, 1: second})


def test_new_worker_rejects_old_engine_metadata_before_binding_its_first_epoch(engine):
    old, new = engine(), engine()
    metadata = old.scheduler.schedule().kv_connector_metadata
    new.worker.bind_connector_metadata(metadata)
    with pytest.raises(ValueError, match="fresh generation"):
        new.worker.start_load_kv(None)
    assert not new.native.entered.is_set()
    new.worker.clear_connector_metadata()
    new.step()


@pytest.mark.parametrize(
    "spec,message",
    [
        (replace(attention_spec(), non_causal=True), "Non-causal"),
        (
            replace(attention_spec(), kv_quant_mode=KVQuantMode.FP8_PER_TENSOR),
            "runtime-scale",
        ),
    ],
)
def test_connector_rejects_unsafe_actual_specs_even_with_auto_cache_dtype(
    engine, spec, message
):
    e = engine()
    cfg = replace(e.cfg, kv_cache_groups=[KVCacheGroupSpec(["a"], spec)])
    with pytest.raises(ValueError, match=message):
        UMBPConnector(e.config, KVConnectorRole.SCHEDULER, cfg)


def test_full_scheduler_chunked_prefill_stores_each_completed_block_once(engine):
    e = engine(budget=4)
    request = e.request(120, tokens=17)
    e.scheduler.add_request(request)
    e.drain()
    stores = [
        job
        for job in e.jobs
        if isinstance(job, TransferJob) and job.operation == "store"
    ]
    keys = [
        BlockKey(block.block_hash, block.group_id)
        for job in stores
        for block in job.blocks
    ]
    assert len(keys) == len(set(keys)) == 4
    assert e.computed[request.request_id] == 17
    assert len(e.native.data) == 4


def test_full_scheduler_unequal_group_blocks_share_a_valid_hit_boundary(engine):
    groups = [
        KVCacheGroupSpec(["a"], attention_spec()),
        KVCacheGroupSpec(["b"], replace(attention_spec(), block_size=8)),
    ]
    e = engine(budget=32, groups=groups)
    first = e.request(130, tokens=17)
    e.scheduler.add_request(first)
    e.drain()
    assert len(e.native.data) == 6
    assert e.scheduler.kv_cache_manager.block_pool.reset_prefix_cache()
    e.jobs.clear()
    second = e.request(131, tokens=17)
    e.scheduler.add_request(second)
    e.drain()
    loads = [
        job
        for job in e.jobs
        if isinstance(job, TransferJob) and job.operation == "load"
    ]
    assert len(loads) == 1
    assert [item.group_id for item in loads[0].blocks] == [0, 0, 0, 0, 1, 1]
    assert e.computed[second.request_id] == 1


def test_full_scheduler_lookup_timeout_recomputes_while_native_lookup_drains(
    engine, monkeypatch
):
    e = engine(limit=1)
    tick = [0.0]
    monkeypatch.setattr(
        "vllm.distributed.kv_transfer.kv_connector.v1.umbp.planner.time",
        SimpleNamespace(monotonic=lambda: tick[0]),
    )
    e.native.release.clear()
    request = e.request(140)
    e.scheduler.add_request(request)
    e.step()
    assert e.native.entered.wait(5)
    tick[0] = 10.0
    e.step()
    assert e.computed[request.request_id] == 13
    assert e.scheduler.connector.has_pending_push_work()
    e.native.release.set()
    e.drain()
    assert not any(
        isinstance(job, TransferJob) and job.operation == "load" for job in e.jobs
    )


def test_full_scheduler_abort_keeps_receive_owned_until_native_completion(
    engine, monkeypatch
):
    e = engine()
    first = e.request(150)
    e.scheduler.add_request(first)
    e.drain()
    assert e.scheduler.kv_cache_manager.block_pool.reset_prefix_cache()
    entered, release = threading.Event(), threading.Event()
    native_get = e.native.batch_get_ranges_into_ptr

    def blocked_get(*args):
        entered.set()
        assert release.wait(5)
        return native_get(*args)

    monkeypatch.setattr(e.native, "batch_get_ranges_into_ptr", blocked_get)
    request = e.request(151)
    e.scheduler.add_request(request)
    try:
        e.until(lambda: request.status == RequestStatus.WAITING_FOR_REMOTE_KVS)
        assert entered.wait(5)
        pool = e.scheduler.kv_cache_manager.block_pool
        ids = e.scheduler.kv_cache_manager.get_block_ids(request.request_id)[0]
        e.scheduler.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
        e.step()
        assert all(pool.blocks[index].ref_cnt > 0 for index in ids)
        assert request.request_id in e.scheduler.requests
    finally:
        release.set()
    e.drain()
    assert request.request_id not in e.scheduler.requests
    assert request.request_id not in e.computed
    assert pool.get_num_free_blocks() == 31


def test_full_scheduler_lookup_hole_limits_restore_to_contiguous_prefix(engine):
    e = engine()
    first = e.request(160)
    e.scheduler.add_request(first)
    e.drain()
    middle_hash = first.block_hashes[1].hex()
    for key in tuple(e.native.data):
        if key.endswith(middle_hash):
            del e.native.data[key]
    assert len(e.native.data) == 2
    assert e.scheduler.kv_cache_manager.block_pool.reset_prefix_cache()
    e.jobs.clear()
    second = e.request(161)
    e.scheduler.add_request(second)
    e.drain()
    loads = [
        job
        for job in e.jobs
        if isinstance(job, TransferJob) and job.operation == "load"
    ]
    assert len(loads) == 1 and len(loads[0].blocks) == 1
    assert e.computed[second.request_id] == 9


def test_full_scheduler_read_opt_out_performs_no_external_lookup(engine):
    e = engine()
    request = e.request(170)
    request.skip_reading_prefix_cache = True
    e.scheduler.add_request(request)
    e.drain()
    assert not any(isinstance(job, LookupJob) for job in e.jobs)
    assert e.computed[request.request_id] == 13


@pytest.mark.parametrize(
    "revision,dtype,expected_compute",
    [
        ("unit-revision", "float16", 1),
        ("different-weights", "float16", 13),
        ("unit-revision", "bfloat16", 13),
    ],
)
def test_pool_reuse_between_engines_requires_matching_model_identity(
    engine,
    revision,
    dtype,
    expected_compute,
):
    """Ordinary serving replicas share storage; this is NOT P/D handoff."""
    first = engine()
    request = first.request(180)
    first.scheduler.add_request(request)
    first.drain()
    second = engine(revision=revision, dtype=dtype)
    second.native.data = first.native.data
    request = second.request(181)
    second.scheduler.add_request(request)
    second.drain()
    assert (
        first.worker.get_handshake_metadata().generation
        != second.worker.get_handshake_metadata().generation
    )
    assert second.computed[request.request_id] == expected_compute
    if expected_compute == 1:
        assert not any(
            isinstance(job, TransferJob) and job.operation == "store"
            for job in second.jobs
        )


def decode_request(request_id, prefill_request, params):
    request = Request(
        request_id,
        list(prefill_request.all_token_ids),
        SamplingParams(max_tokens=1),
        None,
        block_hasher=get_request_block_hasher(4, sha256),
    )
    request.kv_transfer_params = params
    return request


def complete_prefill(prefill, *, tokens=16):
    request = prefill.request(220, tokens=tokens)
    request.kv_transfer_params = {"do_remote_decode": True}
    prefill.scheduler.add_request(request)
    prefill.drain()
    params = next(
        output.kv_transfer_params
        for output in prefill.engine_outputs
        if output.kv_transfer_params
    )
    return request, params, HandoffHandle.from_dict(params["umbp_handoff"])


@pytest.mark.parametrize("failed_rank", [None, 1])
def test_pd_readiness_requires_successful_export_from_every_rank(
    setup, monkeypatch, failed_rank
):
    s = setup(epoch="e" * 64)
    if failed_rank is not None:
        monkeypatch.setattr(
            s.natives[failed_rank],
            "batch_put_ranges_from_ptr",
            lambda keys, *args: [False] * len(keys),
        )
    planner = UMBPHandoffPlanner(
        s.manager,
        s.scheduler,
        namespace=s.namespace,
        producer=True,
        consumer=False,
        timeout=30.0,
    )
    request, blocks = s.request()
    request.num_computed_tokens = 4
    request.status = RequestStatus.FINISHED_LENGTH_CAPPED
    request.kv_transfer_params = {"do_remote_decode": True}
    delayed, params = planner.request_finished(request)
    assert delayed and params is not None
    handle = HandoffHandle.from_dict(params["umbp_handoff"])
    planner.build_jobs()
    export = s.dispatch(ready=False).jobs[0]
    s.fences[0].event.set()
    assert feedback(s.scheduler, collect(s.workers[0], export)) == ()
    planner.build_jobs()
    assert not s.scheduler.build_connector_meta().jobs
    assert all(
        handle.ready_key(rank) not in native.data
        for rank, native in enumerate(s.natives)
    )
    assert all(s.manager.block_pool.blocks[b.block_id].ref_cnt == 2 for b in blocks)

    s.fences[1].event.set()
    outcomes = feedback(s.scheduler, collect(s.workers[1], export))
    assert len(outcomes) == 1
    planner.update_outcome(outcomes[0])
    planner.build_jobs()
    control = s.dispatch().jobs[0]
    assert isinstance(control, ControlJob)
    assert control.operation == ("publish" if failed_rank is None else "release")
    for worker in s.workers:
        feedback(s.scheduler, collect(worker, control))
        assert not worker.get_transfer_results({request.request_id}).finished_sending
    assert all(s.manager.block_pool.blocks[b.block_id].ref_cnt == 1 for b in blocks)
    s.dispatch()
    for rank, worker in enumerate(s.workers):
        assert worker.get_transfer_results({request.request_id}).finished_sending == {
            request.request_id
        }
        feedback(s.scheduler, worker.build_connector_worker_meta())
        assert (handle.ready_key(rank) in s.natives[rank].data) == (failed_rank is None)
    planner.finished_sending(request.request_id)
    s.manager.free(request)
    assert not s.scheduler.has_pending_push_work() and not planner.has_pending()
    assert s.manager.block_pool.get_num_free_blocks() == 7


@pytest.mark.parametrize("tokens", [13, 16])
@pytest.mark.parametrize("local_prefix", [0, 4])
def test_pd_handle_precedes_readiness_and_decode_loads_only_missing_prefix(
    engine, monkeypatch, local_prefix, tokens
):
    prefill = engine(kv_role="kv_producer", limit=1)
    decode = engine(kv_role="kv_consumer", limit=1)
    decode.native.data = prefill.native.data
    first = prefill.request(200, tokens=tokens)
    first.kv_transfer_params = {"do_remote_decode": True}
    prefill.scheduler.add_request(first)
    prefill.until(
        lambda: any(output.kv_transfer_params for output in prefill.engine_outputs)
    )
    params = next(
        output.kv_transfer_params
        for output in prefill.engine_outputs
        if output.kv_transfer_params
    )
    handle = HandoffHandle.from_dict(params["umbp_handoff"])
    boundary = tokens // 4 * 4
    assert handle.token_boundary == boundary
    assert handle.ready_key(0) not in prefill.native.data
    assert first.request_id in prefill.scheduler.requests
    assert prefill.scheduler.requires_kv_delivery
    assert not decode.scheduler.requires_kv_delivery
    if local_prefix:
        seed = decode.request(201, tokens=local_prefix)
        decode.scheduler.kv_cache_manager.allocate_slots(seed, local_prefix)
        block = decode.scheduler.kv_cache_manager.get_block_ids(seed.request_id)[0][0]
        decode.caches["a"][block].fill_(7)
        decode.scheduler.kv_cache_manager.free(seed)
    get = MagicMock(wraps=decode.native.batch_get_ranges_into_ptr)
    monkeypatch.setattr(decode.native, "batch_get_ranges_into_ptr", get)
    second = decode_request("decode-200", first, params)
    decode.scheduler.add_request(second)
    decode.step()
    assert second.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    get.assert_not_called()
    assert second.request_id not in decode.computed
    prefill.drain()
    assert handle.ready_key(0) in prefill.native.data
    assert prefill.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 31
    decode.until(
        lambda: second.request_id in decode.scheduler.finished_recving_kv_req_ids
    )
    blocks = decode.scheduler.kv_cache_manager.get_block_ids(second.request_id)[0]
    assert all(
        torch.all(decode.caches["a"][block] == index + 7)
        for index, block in enumerate(blocks[: boundary // 4])
    )
    decode.drain()
    loads = [
        job
        for job in decode.jobs
        if isinstance(job, TransferJob) and job.operation == "load"
    ]
    assert len(loads) == 1 and len(loads[0].blocks) == (boundary - local_prefix) // 4
    assert decode.computed[second.request_id] == tokens + 1 - boundary
    assert get.call_count == 1
    assert len(get.call_args.args[0]) == (boundary - local_prefix) // 4
    assert all(not output.invalid_block_ids for output in decode.outputs)


def test_pd_marker_cannot_authorize_get_before_missing_objects_are_visible(
    engine, monkeypatch
):
    """A local marker can precede a peer's heartbeat-delivered KV routes."""
    prefill = engine(kv_role="kv_producer")
    decode = engine(kv_role="kv_consumer")
    decode.native.data = prefill.native.data
    first, params, handle = complete_prefill(prefill)
    visible = threading.Event()
    exists = decode.native.batch_exists

    def delayed_visibility(keys):
        results = exists(keys)
        return [
            found and (key == handle.ready_key(0) or visible.is_set())
            for key, found in zip(keys, results, strict=True)
        ]

    probe = MagicMock(side_effect=delayed_visibility)
    get = MagicMock(wraps=decode.native.batch_get_ranges_into_ptr)
    monkeypatch.setattr(decode.native, "batch_exists", probe)
    monkeypatch.setattr(decode.native, "batch_get_ranges_into_ptr", get)
    second = decode_request("delayed-visibility", first, params)
    decode.scheduler.add_request(second)
    decode.until(lambda: probe.call_count >= 2 or get.call_count > 0)
    get.assert_not_called()
    assert second.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    assert second.request_id not in decode.computed
    visible.set()
    decode.drain()
    get.assert_called_once()
    assert probe.call_args.args[0] == [handle.ready_key(0), *get.call_args.args[0]]
    assert decode.computed[second.request_id] == 1
    assert not any(output.invalid_block_ids for output in decode.outputs)
    assert decode.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 31


def test_pd_failed_export_never_publishes_readiness_and_decode_recomputes(
    engine, monkeypatch
):
    prefill = engine(kv_role="kv_producer")
    decode = engine(kv_role="kv_consumer", handoff_timeout=0.02)
    decode.native.data = prefill.native.data
    first = prefill.request(210, tokens=16)
    first.kv_transfer_params = {"do_remote_decode": True}
    monkeypatch.setattr(
        prefill.native,
        "batch_put_ranges_from_ptr",
        lambda keys, *args: [False] * len(keys),
    )
    prefill.scheduler.add_request(first)
    prefill.drain()
    params = next(
        output.kv_transfer_params
        for output in prefill.engine_outputs
        if output.kv_transfer_params
    )
    handle = HandoffHandle.from_dict(params["umbp_handoff"])
    assert handle.ready_key(0) not in prefill.native.data
    assert not any(
        isinstance(job, ControlJob) and job.operation == "publish"
        for job in prefill.jobs
    )
    second = decode_request("decode-210", first, params)
    decode.scheduler.add_request(second)
    decode.drain()
    assert decode.computed[second.request_id] == 17
    failures = [output for output in decode.outputs if output.invalid_block_ids]
    assert len(failures) == 1 and second.request_id in failures[0].finished_recving
    assert decode.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 31


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("namespace", "vllm-umbp:v1:" + "f" * 64),
        ("boundary_hash", "00" * 32),
        ("expires_at_ms", 1),
        ("version", 2),
    ],
)
def test_pd_invalid_handoff_never_authorizes_a_receive(
    engine, monkeypatch, field, value
):
    prefill = engine(kv_role="kv_producer")
    decode = engine(kv_role="kv_consumer")
    decode.native.data = prefill.native.data
    first, params, _ = complete_prefill(prefill)
    params["umbp_handoff"][field] = value
    get = MagicMock(wraps=decode.native.batch_get_ranges_into_ptr)
    monkeypatch.setattr(decode.native, "batch_get_ranges_into_ptr", get)
    second = decode_request("invalid-handoff", first, params)
    decode.scheduler.add_request(second)
    decode.drain()
    get.assert_not_called()
    assert decode.computed[second.request_id] == 17
    assert not any(output.finished_recving for output in decode.outputs)
    assert decode.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 31


@pytest.mark.parametrize("policy", ["recompute", "fail"])
def test_pd_eviction_after_readiness_obeys_receive_failure_policy(
    engine, monkeypatch, policy
):
    prefill = engine(kv_role="kv_producer")
    decode = engine(kv_role="kv_consumer", load_failure_policy=policy)
    decode.native.data = prefill.native.data
    first, params, handle = complete_prefill(prefill)
    assert handle.ready_key(0) in prefill.native.data
    victim = next(key for key in prefill.native.data if ":pd-ready:" not in key)
    get = decode.native.batch_get_ranges_into_ptr

    def evict_after_probe(*args):
        del prefill.native.data[victim]
        return get(*args)

    monkeypatch.setattr(decode.native, "batch_get_ranges_into_ptr", evict_after_probe)
    second = decode_request("evicted-handoff", first, params)
    decode.scheduler.add_request(second)
    decode.drain()
    failures = [output for output in decode.outputs if output.invalid_block_ids]
    assert len(failures) == 1 and second.request_id in failures[0].finished_recving
    if policy == "recompute":
        assert decode.computed[second.request_id] == 17
        assert second.status == RequestStatus.FINISHED_LENGTH_CAPPED
    else:
        assert second.request_id not in decode.computed
        assert second.status == RequestStatus.FINISHED_ERROR
    assert decode.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 31


@pytest.mark.parametrize("abort", [False, True])
def test_pd_blocked_readiness_probe_can_retire_without_issuing_kv_io(
    engine, monkeypatch, abort
):
    prefill = engine(kv_role="kv_producer")
    decode = engine(kv_role="kv_consumer", handoff_timeout=0.1, limit=1)
    decode.native.data = prefill.native.data
    first, params, _ = complete_prefill(prefill)
    decode.native.release.clear()
    get = MagicMock(wraps=decode.native.batch_get_ranges_into_ptr)
    monkeypatch.setattr(decode.native, "batch_get_ranges_into_ptr", get)
    second = decode_request("blocked-readiness", first, params)
    decode.scheduler.add_request(second)
    decode.step()
    assert decode.native.entered.wait(5)
    assert second.status == RequestStatus.WAITING_FOR_REMOTE_KVS
    if abort:
        decode.scheduler.finish_requests(
            [second.request_id], RequestStatus.FINISHED_ABORTED
        )
    decode.until(lambda: any(output.finished_recving for output in decode.outputs))
    # The still-running EXISTS call carries no KV addresses, so unlike a GET it
    # need not retain the failed/aborted receive's allocations until it returns.
    assert not decode.native.release.is_set()
    get.assert_not_called()
    decode.native.release.set()
    decode.drain()
    if abort:
        assert second.status == RequestStatus.FINISHED_ABORTED
        assert second.request_id not in decode.computed
    else:
        assert decode.computed[second.request_id] == 17
    get.assert_not_called()
    assert decode.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 31


def test_pd_unequal_dense_groups_restore_a_jointly_aligned_boundary(engine):
    groups = [
        KVCacheGroupSpec([name], replace(attention_spec(), block_size=size))
        for name, size in (("a", 4), ("b", 8))
    ]
    prefill = engine(kv_role="kv_producer", groups=groups)
    decode = engine(kv_role="kv_consumer", groups=groups)
    decode.native.data = prefill.native.data
    first, params, handle = complete_prefill(prefill, tokens=17)
    assert handle.token_boundary == 16
    second = decode_request("unequal-groups", first, params)
    decode.scheduler.add_request(second)
    decode.until(
        lambda: second.request_id in decode.scheduler.finished_recving_kv_req_ids
    )
    blocks = decode.scheduler.kv_cache_manager.get_block_ids(second.request_id)
    for group, spec in enumerate(groups):
        for index, block in enumerate(
            blocks[group][: 16 // spec.kv_cache_spec.block_size]
        ):
            assert torch.all(decode.caches[spec.layer_names[0]][block] == index + 7)
    decode.drain()
    loads = [
        job
        for job in decode.jobs
        if isinstance(job, TransferJob) and job.operation == "load"
    ]
    assert len(loads) == 1 and len(loads[0].blocks) == 6
    assert decode.computed[second.request_id] == 2
    assert decode.scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == 31
