# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import time
import types
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from vllm.v1.kv_offload.base import Medium, OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.base import TransferJob
from vllm.v1.kv_offload.tiering.mori.manager import MoriSecondaryTierManager


class FakeConfig:
    def __init__(self):
        self.dram = SimpleNamespace()
        self.ssd = SimpleNamespace()
        self.eviction = SimpleNamespace()
        self.copy_pipeline = SimpleNamespace()
        self.distributed = None

    @classmethod
    def from_environment(cls):
        config = cls()
        config.dram.high_watermark = 0.8
        config.copy_pipeline.worker_threads = 2
        return config


class FakeDistributedConfig:
    def __init__(self):
        self.master_config = SimpleNamespace()
        self.io_engine = SimpleNamespace()


class FakeClient:
    instances: list["FakeClient"] = []
    put_results: list[bool] | None = None

    def __init__(self, config):
        self.config = config
        self.data = {}
        self.registration = None
        self.deregistered = None
        self.flush_count = 0
        self.__class__.instances.append(self)

    def register_memory(self, ptr, size):
        self.registration = (ptr, size)
        return True

    def deregister_memory(self, ptr):
        self.deregistered = ptr

    def batch_exists(self, keys):
        return [key in self.data for key in keys]

    def batch_put_from_ptr(self, keys, ptrs, sizes):
        import ctypes

        for key, ptr, size in zip(keys, ptrs, sizes):
            self.data[key] = ctypes.string_at(ptr, size)
        return [True] * len(keys) if self.put_results is None else self.put_results

    def batch_get_into_ptr(self, keys, ptrs, sizes):
        import ctypes

        results = []
        for key, ptr, size in zip(keys, ptrs, sizes):
            value = self.data.get(key)
            results.append(value is not None)
            if value is not None:
                ctypes.memmove(ptr, value, size)
        return results

    def flush(self):
        self.flush_count += 1
        return True


def _install_fake_mori(monkeypatch):
    module = types.ModuleType("mori.umbp")
    dynamic_module = cast(Any, module)
    dynamic_module.UMBPClient = FakeClient
    dynamic_module.UMBPConfig = FakeConfig
    dynamic_module.UMBPDistributedConfig = FakeDistributedConfig
    monkeypatch.setitem(sys.modules, "mori.umbp", module)
    FakeClient.instances.clear()
    FakeClient.put_results = None


def _make_spec(*, enable_kv_cache_events=False):
    config = SimpleNamespace(
        engine_id="engine-0",
        model=SimpleNamespace(name="model", dtype="float16"),
        groups=(SimpleNamespace(tokens_per_block=16, layer_names=("layer.0",)),),
        cache=SimpleNamespace(blocks_per_chunk=1),
        parallel=SimpleNamespace(is_parallelism_agnostic=True),
    )
    return SimpleNamespace(
        config=config,
        kv_bytes_per_chunk=16,
        kv_events_config=SimpleNamespace(enable_kv_cache_events=enable_kv_cache_events),
    )


def _wait_for_result(tier):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        results = list(tier.get_finished_jobs())
        if results:
            return results[0]
        time.sleep(0.001)
    raise AssertionError("MoRI transfer did not finish")


def test_mori_tier_round_trip_uses_registered_primary_buffer(monkeypatch):
    _install_fake_mori(monkeypatch)
    backing = bytearray(range(64))
    view = memoryview(backing).cast("B", shape=(4, 16))
    tier = MoriSecondaryTierManager(
        _make_spec(), view, "mori", dram_capacity_bytes=1024, io_threads=1
    )
    key = OffloadKey(b"hash" + (0).to_bytes(4, "big"))

    tier.submit_store(
        TransferJob(1, [key], np.array([1]), False, ReqContext("request"))
    )
    assert _wait_for_result(tier).success
    backing[16:32] = b"\0" * 16
    tier.submit_load(TransferJob(2, [key], np.array([1]), True, ReqContext("request")))
    assert _wait_for_result(tier).success
    assert backing[16:32] == bytes(range(16, 32))

    client = FakeClient.instances[0]
    assert client.registration == (tier._base_addr, 64)
    tier.shutdown()
    assert client.deregistered == tier._base_addr


def test_mori_tier_builds_distributed_config(monkeypatch):
    _install_fake_mori(monkeypatch)
    view = memoryview(bytearray(32)).cast("B", shape=(2, 16))
    tier = MoriSecondaryTierManager(
        _make_spec(),
        view,
        "mori",
        dram_capacity_bytes=1024,
        master_address="master:15558",
        node_address="10.0.0.2",
        io_engine_port=16000,
        peer_service_port=17000,
    )

    distributed = FakeClient.instances[0].config.distributed
    assert distributed.master_config.master_address == "master:15558"
    assert distributed.master_config.node_id == "engine-0"
    assert distributed.master_config.node_address == "10.0.0.2"
    assert distributed.io_engine.host == "10.0.0.2"
    assert distributed.io_engine.port == 16000
    assert distributed.peer_service_port == 17000
    tier.shutdown()


def test_mori_tier_applies_tuning_over_environment(monkeypatch):
    _install_fake_mori(monkeypatch)
    view = memoryview(bytearray(32)).cast("B", shape=(2, 16))
    tier = MoriSecondaryTierManager(
        _make_spec(),
        view,
        "mori",
        dram_capacity_bytes=1024,
        dram_high_watermark=0.9,
        dram_low_watermark=0.7,
        dram_use_hugepages=True,
        dram_hugepage_size=2 * 1024 * 1024,
        dram_numa_node=1,
        dram_prefault=True,
        ssd_enabled=True,
        ssd_capacity_bytes=4096,
        ssd_backend="spdk",
        ssd_segment_size_bytes=1024,
        ssd_high_watermark=0.85,
        ssd_low_watermark=0.65,
        copy_pipeline_worker_threads=4,
        copy_pipeline_queue_depth=256,
        copy_pipeline_batch_max_ops=32,
        master_address="master:15558",
        cache_remote_fetches=False,
        cache_remote_admission=False,
        dram_page_size=2 * 1024 * 1024,
    )

    config = FakeClient.instances[0].config
    assert config.dram.high_watermark == 0.9
    assert config.dram.low_watermark == 0.7
    assert config.dram.use_hugepages
    assert config.dram.hugepage_size == 2 * 1024 * 1024
    assert config.dram.numa_node == 1
    assert config.dram.prefault
    assert config.ssd.ssd_backend == "spdk"
    assert config.ssd.segment_size_bytes == 1024
    assert config.ssd.high_watermark == 0.85
    assert config.ssd.low_watermark == 0.65
    assert config.copy_pipeline.worker_threads == 4
    assert config.copy_pipeline.queue_depth == 256
    assert config.copy_pipeline.batch_max_ops == 32
    assert not config.distributed.cache_remote_fetches
    assert not config.distributed.cache_remote_admission
    assert config.distributed.dram_page_size == 2 * 1024 * 1024
    tier.shutdown()


def test_mori_tier_emits_event_only_after_successful_store(monkeypatch):
    _install_fake_mori(monkeypatch)
    view = memoryview(bytearray(32)).cast("B", shape=(2, 16))
    tier = MoriSecondaryTierManager(
        _make_spec(enable_kv_cache_events=True),
        view,
        "mori",
        dram_capacity_bytes=1024,
        enable_kv_events=True,
        io_threads=1,
    )
    key = OffloadKey(b"hash" + (0).to_bytes(4, "big"))
    other_key = OffloadKey(b"other" + (0).to_bytes(4, "big"))

    tier.submit_store(
        TransferJob(1, [key], np.array([0]), False, ReqContext("request"))
    )
    assert _wait_for_result(tier).success
    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key]
    assert events[0].medium is Medium.STORAGE
    assert not events[0].removed
    assert list(tier.take_events()) == []

    FakeClient.put_results = [True, False]
    tier.submit_store(
        TransferJob(
            2,
            [key, other_key],
            np.array([0, 1]),
            False,
            ReqContext("request"),
        )
    )
    assert not _wait_for_result(tier).success
    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key]

    FakeClient.put_results = [True]
    tier.submit_store(
        TransferJob(
            3,
            [key, other_key],
            np.array([0, 1]),
            False,
            ReqContext("request"),
        )
    )
    assert not _wait_for_result(tier).success
    events = list(tier.take_events())
    assert len(events) == 1
    assert events[0].keys == [key]

    FakeClient.put_results = [False, True]
    tier.submit_store(
        TransferJob(
            4,
            [key, other_key],
            np.array([0, 1]),
            False,
            ReqContext("request"),
        )
    )
    assert not _wait_for_result(tier).success
    assert list(tier.take_events()) == []
    assert FakeClient.instances[0].flush_count == 3
    tier.shutdown()


def test_mori_tier_events_require_global_enable(monkeypatch):
    _install_fake_mori(monkeypatch)
    tier = MoriSecondaryTierManager(
        _make_spec(enable_kv_cache_events=False),
        memoryview(bytearray(16)).cast("B", shape=(1, 16)),
        "mori",
        dram_capacity_bytes=1024,
        enable_kv_events=True,
        io_threads=1,
    )
    key = OffloadKey(b"hash" + (0).to_bytes(4, "big"))

    tier.submit_store(
        TransferJob(1, [key], np.array([0]), False, ReqContext("request"))
    )
    assert _wait_for_result(tier).success
    assert tier.events is None
    assert list(tier.take_events()) == []
    tier.shutdown()
