# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Policy translation and private-path lifetime, without native MoRI or SSD I/O."""

import json
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.v1.kv_connector.unit.test_umbp_store import NativeStore, object_for, region
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.config import (
    UMBPNodeConfig,
    UMBPStoreConfig,
)


@pytest.fixture
def native_api(monkeypatch):
    monkeypatch.delenv("UMBP_WORKLOAD_TRACE_PATH", raising=False)

    class NativeConfig:
        __slots__ = ("dram", "ssd", "distributed")

        def __init__(self):
            self.dram = SimpleNamespace(capacity_bytes=4 << 30)
            self.ssd = SimpleNamespace(enabled=True)

    class DistributedConfig:
        __slots__ = (
            "master_config",
            "io_engine",
            "peer_service_port",
            "dram_page_size",
            "ranged_scratch_size",
            "cache_remote_fetches",
            "ranged_locality_prefetch",
            "local_first",
            "backend_policy_path",
        )

        def __init__(self):
            self.master_config = SimpleNamespace()
            self.io_engine = SimpleNamespace()

    created = []

    def client(config):
        native = NativeStore()
        policy_path = Path(config.distributed.backend_policy_path)
        policy = json.loads(policy_path.read_text())
        paths = [policy_path.parent]
        if "ssd" in policy["backends"]:
            paths.extend(Path(p) for p in policy["backends"]["ssd"]["path"].split(","))
        record = SimpleNamespace(
            config=config,
            policy=policy,
            paths=paths,
            native=weakref.ref(native),
            destroyed_with_paths=False,
        )
        weakref.finalize(
            native,
            lambda: setattr(
                record, "destroyed_with_paths", all(path.is_dir() for path in paths)
            ),
        )
        created.append(record)
        return native

    api = SimpleNamespace(
        UMBPConfig=NativeConfig,
        UMBPDistributedConfig=DistributedConfig,
        UMBPClient=client,
        MemoryLocationType=SimpleNamespace(CPU="cpu", GPU="gpu"),
    )
    monkeypatch.setitem(sys.modules, "mori", SimpleNamespace(cpp=api))
    monkeypatch.setitem(sys.modules, "mori.cpp", api)
    return api, created


def config(**overrides):
    return UMBPStoreConfig.from_dict(
        {
            "page_size_bytes": 4096,
            "dram_capacity_bytes": 32768,
            **overrides,
        }
    )


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"backend": "local"}, "Unknown"),
        ({"page_size_bytes": True}, "integer"),
        ({"page_size_bytes": 4097}, "multiple"),
        ({"dram_capacity_bytes": 0}, "At least one"),
        ({"dram_capacity_bytes": 4095}, "one page"),
        ({"dram_capacity_bytes": 1 << 64}, "integer"),
        ({"ssd_capacity_bytes": 4096}, "together"),
        ({"ssd_roots": ["/mnt/ssd"]}, "together"),
        ({"ssd_roots": " /mnt/ssd"}, "immutable"),
        ({"ssd_roots": ["relative"]}, "absolute"),
        ({"ssd_roots": ["/mnt/../ssd"]}, "absolute"),
        ({"ssd_roots": ["/mnt/a,/mnt/b"]}, "absolute"),
        ({"ssd_roots": ["/mnt/a", "/mnt/b"], "ssd_capacity_bytes": 4096}, "Each SSD"),
        ({"ranged_scratch_bytes": 0}, "integer"),
        ({"max_pending": -1}, "integer"),
        ({"ssd_staging_slots": 0}, "integer"),
        ({"master_address": "host"}, "host:port"),
        ({"master_address": "user:secret@host:1234"}, "credentials"),
        ({"master_address": "host:1234/path"}, "host:port"),
    ],
)
def test_invalid_configuration_fails_before_native_start(native_api, overrides, match):
    with pytest.raises(ValueError, match=match):
        config(**overrides)
    assert native_api[1] == []


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"max_object_bytes": 4097}, "page must fit"),
        ({"max_object_bytes": 0}, "integer"),
        ({"max_object_bytes": 2048, "scratch": 1024}, "scratch arena"),
    ],
)
def test_layout_sizes_are_checked_before_start(native_api, kwargs, match):
    kwargs = dict(kwargs)
    cfg = config(ranged_scratch_bytes=kwargs.pop("scratch", 4096))
    with pytest.raises(ValueError, match=match):
        cfg.open_store(UMBPNodeConfig("worker-0"), **kwargs)
    assert native_api[1] == []


def test_dram_only_disables_implicit_ssd_and_uses_explicit_page_policy(native_api):
    cfg = config(dram_capacity_bytes=4096, ranged_scratch_bytes=4096)
    store = cfg.open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    record = native_api[1][0]
    try:
        assert not record.config.ssd.enabled
        # The native embedded factory uses this unused legacy field for page
        # fitting; actual allocation must still use the policy's 4096 bytes.
        assert record.config.dram.capacity_bytes >= 8 * 4096
        assert record.policy["backends"] == {
            "dram": {"type": "dram", "capacity": "4096B"}
        }
        dist = record.config.distributed
        assert dist.dram_page_size == dist.ranged_scratch_size == 4096
        assert not dist.cache_remote_fetches and not dist.ranged_locality_prefetch
        assert not dist.io_engine.host and not dist.peer_service_port
        assert not dist.master_config.auto_heartbeat
        memory = region(b"test")
        store.register_region(memory)
        assert store.store((object_for("key", memory),)).result(timeout=5) == (True,)
    finally:
        store.close()
    assert record.native() is None
    assert record.destroyed_with_paths
    assert all(not path.exists() for path in record.paths)


@pytest.mark.parametrize("dram_bytes", [0, 32768])
def test_ssd_roots_are_private_and_only_tiered_when_both_media_enabled(
    native_api, tmp_path, dram_bytes
):
    roots = [tmp_path / f"ssd{i}" for i in range(2)]
    for root in roots:
        root.mkdir()
        (root / "existing").write_text("preserve")
    cfg = config(
        dram_capacity_bytes=dram_bytes,
        ssd_capacity_bytes=1 << 30,
        ssd_roots=[str(root) for root in roots],
        ssd_staging_slots=3,
    )
    store = cfg.open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    record = native_api[1][0]
    try:
        ssd = record.policy["backends"]["ssd"]
        assert ssd["capacity"] == "1073741824B" and ssd["staging_slots"] == 3
        assert [path.parent for path in record.paths[1:]] == roots
        assert all(path.is_dir() for path in record.paths)
        # Native SSD Resolve rejects a batch larger than its staging arena.
        native = record.native()
        original_get = native.batch_get_ranges_into_ptr
        batches = []

        def bounded_get(keys, *args, _get=original_get):
            batches.append(len(keys))
            assert len(keys) <= ssd["staging_slots"]
            return _get(keys, *args)

        native.batch_get_ranges_into_ptr = bounded_get
        buffers = tuple(region(b"XXXX") for _ in range(7))
        for memory in buffers:
            store.register_region(memory)
        objects = tuple(object_for(str(i), mem) for i, mem in enumerate(buffers))
        assert store.load(objects).result(timeout=5) == (False,) * 7
        assert batches == [3, 3, 1]
        del native.batch_get_ranges_into_ptr
        del native, original_get, bounded_get
        if dram_bytes:
            assert record.policy["tiers"][0]["offload_to"] == ["disk"]
            assert record.policy["tiers"][0]["offload_trigger"] == "on_evict"
            assert record.policy["tiers"][1]["promote_trigger"] == "on_read"
        else:
            assert record.policy["tiers"] == [{"name": "disk", "backends": {"ssd": 1}}]
        # Two ranks/processes cannot write the same segment log.
        other = cfg.open_store(UMBPNodeConfig("worker-1"), max_object_bytes=4096)
        try:
            assert set(record.paths).isdisjoint(native_api[1][1].paths)
        finally:
            other.close()
    finally:
        store.close()
    assert all(list(root.iterdir()) == [root / "existing"] for root in roots)
    assert all((root / "existing").read_text() == "preserve" for root in roots)


def test_shared_master_requires_complete_worker_endpoints(native_api):
    cfg = config(master_address="master:50051")
    with pytest.raises(ValueError, match="RDMA host and peer port"):
        cfg.open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    node = UMBPNodeConfig("worker-0", "node-a", "rdma-a", 51001, 52001)
    with pytest.raises(ValueError, match="require a shared master"):
        replace(cfg, master_address="").open_store(node, max_object_bytes=4096)
    assert native_api[1] == []
    store = cfg.open_store(node, max_object_bytes=4096)
    try:
        dist = native_api[1][0].config.distributed
        assert dist.master_config.master_address == "master:50051"
        assert dist.master_config.node_id == "worker-0"
        assert dist.master_config.node_address == "node-a"
        assert dist.master_config.auto_heartbeat
        assert (dist.io_engine.host, dist.io_engine.port) == ("rdma-a", 52001)
        assert dist.peer_service_port == 51001
    finally:
        store.close()


def test_duplicate_ssd_aliases_and_missing_roots_cannot_create_storage(
    native_api, tmp_path
):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    for roots in ([str(tmp_path), str(alias)], [str(tmp_path / "missing")]):
        cfg = config(ssd_roots=roots, ssd_capacity_bytes=1 << 30)
        with pytest.raises((ValueError, FileNotFoundError)):
            cfg.open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    assert list(tmp_path.iterdir()) == [alias]
    assert native_api[1] == []


@pytest.mark.parametrize("failure", ["constructor", "unsupported"])
def test_native_startup_failure_removes_only_its_private_paths(
    native_api, monkeypatch, tmp_path, failure
):
    api, records = native_api
    create = api.UMBPClient

    def fail(config):
        native = create(config)
        if failure == "constructor":
            raise RuntimeError("native startup failed")
        native.supports_ranged_io = lambda: False
        return native

    monkeypatch.setattr(api, "UMBPClient", fail)
    cfg = config(ssd_roots=[str(tmp_path)], ssd_capacity_bytes=1 << 30)
    with pytest.raises(RuntimeError):
        cfg.open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    assert list(tmp_path.iterdir()) == []
    assert all(not path.exists() for record in records for path in record.paths)
    if failure == "unsupported":
        assert all(record.destroyed_with_paths for record in records)


def test_failed_deregistration_retains_ssd_paths_until_retry(native_api, tmp_path):
    cfg = config(ssd_roots=[str(tmp_path)], ssd_capacity_bytes=1 << 30)
    store = cfg.open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    record = native_api[1][0]
    native = record.native()
    store.register_region(region(b"test"))
    try:
        native.fail_deregister = True
        with pytest.raises(RuntimeError, match="deregistration failed"):
            store.close()
        assert all(path.is_dir() for path in record.paths)
    finally:
        native.fail_deregister = False
        store.close()
    assert all(not path.exists() for path in record.paths)


def test_storage_paths_survive_native_io_until_close_drains(native_api, tmp_path):
    cfg = config(ssd_roots=[str(tmp_path)], ssd_capacity_bytes=1 << 30)
    store = cfg.open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    record = native_api[1][0]
    native = record.native()
    native.release.clear()
    started = threading.Event()

    def close():
        started.set()
        store.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            lookup = store.lookup(("key",))
            assert native.entered.wait(5)
            closing = executor.submit(close)
            assert started.wait(5)
            assert not closing.done()
            assert all(path.is_dir() for path in record.paths)
        finally:
            native.release.set()
            del native
            store.close()
        closing.result(timeout=5)
        assert lookup.result(timeout=5) == (False,)
    assert record.destroyed_with_paths
    assert all(not path.exists() for path in record.paths)


def test_cleanup_failure_can_be_retried_after_native_teardown(
    native_api, monkeypatch, tmp_path
):
    import shutil

    cfg = config(ssd_roots=[str(tmp_path)], ssd_capacity_bytes=1 << 30)
    store = cfg.open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    record = native_api[1][0]
    rmtree = shutil.rmtree

    def fail(path):
        raise PermissionError("cleanup denied")

    try:
        monkeypatch.setattr(shutil, "rmtree", fail)
        with pytest.raises(PermissionError, match="cleanup denied"):
            store.close()
        assert record.native() is None and record.destroyed_with_paths
        assert all(path.is_dir() for path in record.paths)
    finally:
        monkeypatch.setattr(shutil, "rmtree", rmtree)
        store.close()
    assert all(not path.exists() for path in record.paths)


def test_alias_cannot_hide_a_comma_separated_storage_path(native_api, tmp_path):
    target = tmp_path / "with,comma"
    target.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=True)
    cfg = config(ssd_roots=[str(alias)], ssd_capacity_bytes=1 << 30)
    with pytest.raises(ValueError, match="absolute paths"):
        cfg.open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    assert list(target.iterdir()) == []
    assert native_api[1] == []


def test_ambient_trace_path_cannot_create_unowned_files(
    native_api, monkeypatch, tmp_path
):
    trace = tmp_path / "trace.json"
    monkeypatch.setenv("UMBP_WORKLOAD_TRACE_PATH", str(trace))
    with pytest.raises(ValueError, match="Unset UMBP_WORKLOAD_TRACE_PATH"):
        config().open_store(UMBPNodeConfig("worker-0"), max_object_bytes=4096)
    assert not trace.exists() and native_api[1] == []
