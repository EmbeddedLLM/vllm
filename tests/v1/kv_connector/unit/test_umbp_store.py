# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; the fake models the pinned MoRI ranged API, not RDMA."""

import ctypes
import gc
import os
import tempfile
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.umbp import store as store_module
from vllm.distributed.kv_transfer.kv_connector.v1.umbp.store import (
    BufferSlice,
    MemoryRegion,
    StoreBusyError,
    TransferObject,
    UMBPStore,
)


class NativeStore:
    def __init__(self):
        self.data = {}
        self.registered = {}
        self.deregistered = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.response = None
        self.fail_deregister = False

    def supports_ranged_io(self):
        return True

    def register_memory(self, ptr, size, loc, device):
        self.registered[ptr] = (size, loc, device)
        return True

    def deregister_memory(self, ptr):
        if self.fail_deregister:
            raise RuntimeError("deregistration failed")
        del self.registered[ptr]
        self.deregistered.append(ptr)

    def batch_exists(self, keys):
        self.entered.set()
        assert self.release.wait(5), "test I/O barrier was not released"
        if isinstance(self.response, Exception):
            raise self.response
        if self.response is not None:
            return self.response
        return [key in self.data for key in keys]

    def batch_put_ranges_from_ptr(self, keys, object_sizes, ptrs, sizes, dst_offsets):
        for key, size, addresses, lengths, offsets in zip(
            keys, object_sizes, ptrs, sizes, dst_offsets, strict=True
        ):
            value = bytearray(size)
            for ptr, length, offset in zip(addresses, lengths, offsets, strict=True):
                value[offset : offset + length] = ctypes.string_at(ptr, length)
            self.data[key] = bytes(value)
        return [True] * len(keys)

    def batch_get_ranges_into_ptr(self, keys, ptrs, sizes, src_offsets):
        results = []
        for key, addresses, lengths, offsets in zip(
            keys, ptrs, sizes, src_offsets, strict=True
        ):
            value = self.data.get(key)
            if value is not None:
                for ptr, length, offset in zip(
                    addresses, lengths, offsets, strict=True
                ):
                    ctypes.memmove(ptr, value[offset : offset + length], length)
            results.append(value is not None)
        return results


def region(data, *, device=None):
    owner = ctypes.create_string_buffer(data, len(data))
    return MemoryRegion(ctypes.addressof(owner), len(data), owner, device)


def object_for(key, memory):
    return TransferObject(key, memory.size, (BufferSlice(memory, 0, memory.size, 0),))


@pytest.mark.skipif(
    not os.getenv("UMBP_NATIVE_TEST_ROOT"),
    reason="Requires explicit native MoRI/GPU/ext4 runner and task-owned root",
)
@pytest.mark.parametrize("medium", ["dram", "ssd"])
@pytest.mark.parametrize("source_device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("target_device", ["cpu", "cuda:0"])
def test_native_ranged_store_preserves_bytes_and_releases_private_paths(
    medium, source_device, target_device
):
    """Real native CPU/GPU I/O; an SSD pass does not establish direct GDS."""
    import torch

    from vllm.distributed.kv_transfer.kv_connector.v1.umbp.config import (
        UMBPNodeConfig,
        UMBPStoreConfig,
    )

    root = Path(os.environ["UMBP_NATIVE_TEST_ROOT"])
    assert root.is_absolute() and root.is_dir()
    source_cpu = (torch.arange(8192) % 251).to(torch.uint8)
    source = source_cpu.to(source_device)
    target = torch.full((9216,), 239, dtype=torch.uint8, device=target_device)
    torch.accelerator.synchronize()
    expected = torch.cat((source_cpu[4096:], source_cpu[:4096]))
    regions = tuple(
        MemoryRegion(
            value.data_ptr(),
            value.numel(),
            value,
            0 if value.device.type == "cuda" else None,
        )
        for value in (source, target)
    )
    with tempfile.TemporaryDirectory(prefix="native-ranged-", dir=root) as case:
        case_root = Path(case)
        config = UMBPStoreConfig(
            page_size_bytes=8192,
            dram_capacity_bytes=32 << 20 if medium == "dram" else 0,
            ssd_capacity_bytes=64 << 20 if medium == "ssd" else 0,
            ssd_roots=(case,) if medium == "ssd" else (),
            ranged_scratch_bytes=64 << 10,
        )
        store = config.open_store(
            UMBPNodeConfig("native-ranged"), max_object_bytes=8192
        )
        try:
            for memory in regions:
                store.register_region(memory)
            send = TransferObject(
                "native-ranged",
                8192,
                (
                    BufferSlice(regions[0], 0, 4096, 4096),
                    BufferSlice(regions[0], 4096, 4096, 0),
                ),
            )
            assert store.store((send,)).result(timeout=30) == (True,)
            assert store.lookup((send.key, "absent")).result(timeout=30) == (
                True,
                False,
            )
            receive = TransferObject(
                send.key,
                8192,
                (
                    BufferSlice(regions[1], 512, 4096, 0),
                    BufferSlice(regions[1], 4608, 4096, 4096),
                ),
            )
            assert store.load((receive,)).result(timeout=30) == (True,)
            actual = target.cpu()
            assert torch.equal(actual[512:8704], expected)
            assert torch.all(actual[:512] == 239) and torch.all(actual[8704:] == 239)
            target.fill_(239)
            torch.accelerator.synchronize()
            partial = TransferObject(
                send.key,
                8192,
                (
                    BufferSlice(regions[1], 128, 256, 37),
                    BufferSlice(regions[1], 5000, 257, 4103),
                ),
            )
            assert store.load((partial,)).result(timeout=30) == (True,)
            partial_expected = torch.full((9216,), 239, dtype=torch.uint8)
            partial_expected[128:384] = expected[37:293]
            partial_expected[5000:5257] = expected[4103:4360]
            assert torch.equal(target.cpu(), partial_expected)
            missing = TransferObject("absent", 8192, receive.slices)
            assert store.load((missing,)).result(timeout=30) == (False,)
        finally:
            store.close()
        assert not list(case_root.iterdir()), "Native SSD paths survived store.close"


@pytest.fixture
def native():
    return NativeStore()


@pytest.fixture
def store(native):
    store = UMBPStore(native, SimpleNamespace(CPU="cpu", GPU="gpu"), max_pending=2)
    yield store
    native.release.set()
    native.fail_deregister = False
    store.close()


def test_ranged_roundtrip_assembles_layers_and_loads_only_missing_slice(store, native):
    first, second, destination = region(b"abcd"), region(b"efgh"), region(b"XXXXXXXX")
    for memory in (first, second, destination):
        store.register_region(memory)
    stored = TransferObject(
        "prefix",
        8,
        (
            BufferSlice(second, 0, 4, 4),
            BufferSlice(first, 0, 4, 0),
        ),
    )
    assert store.store((stored,)).result(timeout=5) == (True,)
    assert native.data["prefix"] == b"abcdefgh"
    assert store.lookup(("prefix", "missing")).result(timeout=5) == (True, False)
    partial = TransferObject("prefix", 8, (BufferSlice(destination, 2, 3, 3),))
    assert store.load((partial,)).result(timeout=5) == (True,)
    assert destination.owner.raw == b"XXdefXXX"


def test_failed_object_does_not_inherit_another_objects_success(store, native):
    first, second = region(b"XXXX"), region(b"YYYY")
    store.register_region(first)
    store.register_region(second)
    native.data["present"] = b"good"
    objects = (object_for("present", first), object_for("missing", second))
    assert store.load(objects).result(timeout=5) == (True, False)
    assert first.owner.raw == b"good" and second.owner.raw == b"YYYY"


def test_bounded_native_batches_preserve_result_order_and_missing_objects(
    native, monkeypatch
):
    """A long restore must fit the native arena without dropping later objects."""
    calls = []
    warnings = []
    monkeypatch.setattr(
        store_module.logger, "warning", lambda fmt, *args: warnings.append(fmt % args)
    )
    put = native.batch_put_ranges_from_ptr
    get = native.batch_get_ranges_into_ptr

    def bounded(operation, method):
        def invoke(keys, *args):
            assert len(keys) <= 2, "Native staging arena exceeded"
            calls.append((operation, tuple(keys)))
            return method(keys, *args)

        return invoke

    native.batch_put_ranges_from_ptr = bounded("store", put)
    native.batch_get_ranges_into_ptr = bounded("load", get)
    store = UMBPStore(
        native, SimpleNamespace(CPU="cpu", GPU="gpu"), max_batch_objects=2
    )
    buffers = tuple(region(bytes([i]) * 4) for i in range(5))
    objects = tuple(object_for(str(i), memory) for i, memory in enumerate(buffers))
    try:
        for memory in buffers:
            store.register_region(memory)
        assert store.store(objects).result(timeout=5) == (True,) * 5
        del native.data["2"]
        for memory in buffers:
            ctypes.memset(memory.ptr, 255, memory.size)
        assert store.load(objects).result(timeout=5) == (True, True, False, True, True)
        assert [memory.owner.raw for memory in buffers] == [
            bytes([255 if i == 2 else i]) * 4 for i in range(5)
        ]
        assert calls == [
            (operation, keys)
            for operation in ("store", "load")
            for keys in (("0", "1"), ("2", "3"), ("4",))
        ]
        assert len(warnings) == 1
        assert "load failed for 1/2 objects in chunk starting at 2" in warnings[0]
        assert "failed indices (first 16): [2]" in warnings[0]
    finally:
        store.close()


def test_store_failure_diagnostics_are_bounded_and_omit_cache_keys(native, monkeypatch):
    """A failed native PUT keeps its per-object results without leaking keys."""
    warnings = []
    monkeypatch.setattr(
        store_module.logger, "warning", lambda fmt, *args: warnings.append(fmt % args)
    )
    native.batch_put_ranges_from_ptr = lambda keys, *args: [False] * len(keys)
    store = UMBPStore(
        native, SimpleNamespace(CPU="cpu", GPU="gpu"), max_batch_objects=32
    )
    memory = region(b"private-value")
    store.register_region(memory)
    objects = tuple(object_for(f"private-key-{i}", memory) for i in range(32))
    try:
        assert store.store(objects).result(timeout=5) == (False,) * 32
        assert len(warnings) == 1
        assert "store failed for 32/32 objects" in warnings[0]
        assert str(list(range(16))) in warnings[0]
        assert "private" not in warnings[0]
    finally:
        store.close()


@pytest.fixture
def retry_clock(monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(
        store_module, "time", SimpleNamespace(monotonic=lambda: clock.now)
    )

    def attach(store):
        def wait(delay):
            clock.now += delay
            return store._stop_retries.is_set()

        monkeypatch.setattr(store._stop_retries, "wait", wait)

    return attach


def test_put_retries_only_failed_objects_and_preserves_chunk_order(native, retry_clock):
    """A transient failure cannot discard successes or resubmit their buffers."""
    calls = []
    put = native.batch_put_ranges_from_ptr

    def transient(keys, *args):
        calls.append(tuple(keys))
        outcome = put(keys, *args)
        if len(calls) == 1:
            outcome[1] = False
        return outcome

    native.batch_put_ranges_from_ptr = transient
    store = UMBPStore(
        native,
        SimpleNamespace(CPU="cpu", GPU="gpu"),
        max_batch_objects=2,
        store_retry_timeout_s=1,
    )
    retry_clock(store)
    memory = region(b"data")
    store.register_region(memory)
    try:
        objects = tuple(object_for(str(i), memory) for i in range(3))
        assert store.store(objects).result(timeout=5) == (True,) * 3
        assert calls == [("0", "1"), ("1",), ("2",)]
        assert native.data == {str(i): b"data" for i in range(3)}
    finally:
        store.close()


def test_put_retry_budget_is_shared_by_all_chunks(native, retry_clock):
    calls = []

    def failed(keys, *args):
        calls.append(tuple(keys))
        return [False] * len(keys)

    native.batch_put_ranges_from_ptr = failed
    store = UMBPStore(
        native,
        SimpleNamespace(CPU="cpu", GPU="gpu"),
        max_batch_objects=1,
        store_retry_timeout_s=0.025,
    )
    retry_clock(store)
    memory = region(b"data")
    store.register_region(memory)
    try:
        objects = tuple(object_for(str(i), memory) for i in range(3))
        assert store.store(objects).result(timeout=5) == (False,) * 3
        assert calls == [("0",), ("0",), ("1",), ("2",)]
    finally:
        store.close()


@pytest.mark.parametrize("response", [[], RuntimeError("private backend error")])
def test_put_retry_never_hides_native_exceptions_or_malformed_results(
    native, response, retry_clock
):
    calls = []

    def failed(keys, *args):
        calls.append(tuple(keys))
        if len(calls) == 1:
            return [False]
        if isinstance(response, Exception):
            raise response
        return response

    native.batch_put_ranges_from_ptr = failed
    store = UMBPStore(
        native,
        SimpleNamespace(CPU="cpu", GPU="gpu"),
        store_retry_timeout_s=1,
    )
    retry_clock(store)
    memory = region(b"data")
    store.register_region(memory)
    try:
        with pytest.raises(RuntimeError):
            store.store((object_for("key", memory),)).result(timeout=5)
        assert calls == [("key",), ("key",)]
    finally:
        store.close()


def test_put_retry_budget_does_not_retry_failed_gets(native, retry_clock):
    calls = []

    def missing(keys, *args):
        calls.append(tuple(keys))
        return [False]

    native.batch_get_ranges_into_ptr = missing
    store = UMBPStore(
        native,
        SimpleNamespace(CPU="cpu", GPU="gpu"),
        store_retry_timeout_s=1,
    )
    retry_clock(store)
    memory = region(b"data")
    store.register_region(memory)
    try:
        assert store.load((object_for("absent", memory),)).result(timeout=5) == (False,)
        assert calls == [("absent",)]
    finally:
        store.close()


def test_close_stops_retry_wait_before_deregistering_source(native, monkeypatch):
    calls = []
    waiting = threading.Event()
    store = UMBPStore(
        native,
        SimpleNamespace(CPU="cpu", GPU="gpu"),
        store_retry_timeout_s=60,
    )
    wait = store._stop_retries.wait

    def blocked(delay):
        waiting.set()
        return wait(5)

    def failed(keys, *args):
        calls.append(tuple(keys))
        return [False]

    native.batch_put_ranges_from_ptr = failed
    monkeypatch.setattr(store._stop_retries, "wait", blocked)
    memory = region(b"data")
    store.register_region(memory)
    try:
        future = store.store((object_for("key", memory),))
        assert waiting.wait(5)
        assert not future.done() and not native.deregistered
        assert not future.cancel(), "Running retries still own their source buffers"
        with ThreadPoolExecutor() as executor:
            executor.submit(store.close).result(timeout=2)
        assert future.result() == (False,)
        assert calls == [("key",)] and native.deregistered == [memory.ptr]
    finally:
        store.close()


@pytest.mark.parametrize(
    "timeout", [True, -1, 61, float("nan"), float("inf"), 1 << 1024]
)
def test_invalid_store_retry_budget_is_rejected(native, timeout):
    with pytest.raises(ValueError, match="store_retry_timeout_s"):
        UMBPStore(native, SimpleNamespace(), store_retry_timeout_s=timeout)


def test_chunked_load_keeps_allocation_owners_until_last_native_call(native):
    store = UMBPStore(
        native, SimpleNamespace(CPU="cpu", GPU="gpu"), max_batch_objects=1
    )
    memories = tuple(region(b"XXXX") for _ in range(3))
    objects = tuple(object_for(str(i), memory) for i, memory in enumerate(memories))
    entered, release, closing_started = (threading.Event() for _ in range(3))
    get = native.batch_get_ranges_into_ptr

    def delayed(keys, *args):
        if keys == ["1"]:
            entered.set()
            assert release.wait(5)
        return get(keys, *args)

    native.batch_get_ranges_into_ptr = delayed
    native.data.update({str(i): b"good" for i in range(3)})
    for memory in memories:
        store.register_region(memory)
    future = store.load(objects)

    def close():
        closing_started.set()
        store.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            assert entered.wait(5)
            assert not future.done() and not future.cancel()
            assert memories[0].owner.raw == b"good"
            closing = executor.submit(close)
            assert closing_started.wait(5)
            assert not closing.done() and not native.deregistered
        finally:
            release.set()
            store.close()
        closing.result(timeout=5)
    assert future.result(timeout=5) == (True,) * 3
    assert all(memory.owner.raw == b"good" for memory in memories)


def test_malformed_later_chunk_fails_future_without_submitting_remaining(native):
    store = UMBPStore(
        native, SimpleNamespace(CPU="cpu", GPU="gpu"), max_batch_objects=1
    )
    memories = tuple(region(b"XXXX") for _ in range(3))
    calls = []
    get = native.batch_get_ranges_into_ptr

    def malformed(keys, *args):
        calls.extend(keys)
        return [] if keys == ["1"] else get(keys, *args)

    native.batch_get_ranges_into_ptr = malformed
    native.data["0"] = b"good"
    try:
        for memory in memories:
            store.register_region(memory)
        objects = tuple(object_for(str(i), memory) for i, memory in enumerate(memories))
        with pytest.raises(RuntimeError, match="malformed"):
            store.load(objects).result(timeout=5)
        assert calls == ["0", "1"]
        assert [memory.owner.raw for memory in memories] == [b"good", b"XXXX", b"XXXX"]
    finally:
        store.close()


@pytest.mark.parametrize("device,loc,ordinal", [(None, "cpu", -1), (3, "gpu", 3)])
def test_registration_preserves_memory_location(store, native, device, loc, ordinal):
    memory = region(b"bytes", device=device)
    store.register_region(memory)
    store.register_region(memory)
    assert native.registered == {memory.ptr: (memory.size, loc, ordinal)}


def test_overlapping_allocations_cannot_replace_a_live_owner(store):
    memory = region(b"abcdefgh")
    store.register_region(memory)
    alias = MemoryRegion(memory.ptr + 1, 2, memory.owner)
    with pytest.raises(ValueError, match="overlap"):
        store.register_region(alias)


def test_unregistered_buffers_never_reach_native_io(store, native):
    with pytest.raises(ValueError, match="unregistered"):
        store.store((object_for("key", region(b"test")),))
    assert native.data == {}


def test_failed_registration_does_not_admit_transfers(store, native):
    native.register_memory = lambda *args: False
    memory = region(b"abcd")
    with pytest.raises(RuntimeError, match="registration failed"):
        store.register_region(memory)
    with pytest.raises(ValueError, match="unregistered"):
        store.load((object_for("key", memory),))


def test_failed_read_can_modify_bytes_but_never_reports_success(store, native):
    memory = region(b"XXXX")
    store.register_region(memory)

    def partial_failure(keys, ptrs, sizes, offsets):
        ctypes.memmove(ptrs[0][0], b"bad!", 4)
        return [False]

    native.batch_get_ranges_into_ptr = partial_failure
    assert store.load((object_for("key", memory),)).result(timeout=5) == (False,)
    assert memory.owner.raw == b"bad!"


def test_shutdown_does_not_clear_another_engines_cached_keys(store, native):
    native.data["another-engine"] = b"protected"
    store.close()
    assert native.data == {"another-engine": b"protected"}


@pytest.mark.parametrize("ptr,size", [(0, 1), (1, 0), (True, 4), (2**64 - 1, 1)])
def test_invalid_memory_extents_cannot_be_registered(ptr, size):
    with pytest.raises(ValueError):
        MemoryRegion(ptr, size, object())


def test_invalid_slices_are_rejected_before_native_pointer_access():
    memory = region(b"abcd")
    with pytest.raises(ValueError, match="outside its memory"):
        BufferSlice(memory, 3, 2, 0)
    with pytest.raises(ValueError, match="outside its stored object"):
        TransferObject("key", 4, (BufferSlice(memory, 0, 4, 1),))


@pytest.mark.parametrize("slices", [((0, 2),), ((0, 3), (2, 2)), ((1, 3),)])
def test_incomplete_or_overlapping_put_is_rejected_before_publication(
    store, native, slices
):
    memory = region(b"abcd")
    store.register_region(memory)
    obj = TransferObject(
        "key", 4, tuple(BufferSlice(memory, 0, size, offset) for offset, size in slices)
    )
    with pytest.raises(ValueError, match="exactly tile"):
        store.store((obj,))
    assert native.data == {}


def test_duplicate_writes_and_overlapping_read_destinations_are_rejected(store):
    memory = region(b"abcd")
    store.register_region(memory)
    obj = object_for("key", memory)
    with pytest.raises(ValueError, match="same key"):
        store.store((obj, obj))
    with pytest.raises(ValueError, match="destinations"):
        store.load((obj, object_for("another", memory)))


@pytest.mark.parametrize("response", [[], [True, False], [1], [None], "yes"])
def test_malformed_native_results_cannot_become_hits(store, native, response):
    native.response = response
    with pytest.raises(RuntimeError, match="malformed"):
        store.lookup(("key",)).result(timeout=5)


def test_native_exceptions_remain_failed_futures(store, native):
    native.response = RuntimeError("peer unavailable")
    with pytest.raises(RuntimeError, match="peer unavailable"):
        store.lookup(("key",)).result(timeout=5)


def test_admission_is_bounded_and_queued_cancellation_releases_capacity(native):
    native.release.clear()
    store = UMBPStore(
        native, SimpleNamespace(CPU="cpu", GPU="gpu"), workers=1, max_pending=2
    )
    try:
        running = store.lookup(("first",))
        assert native.entered.wait(5)
        queued = store.lookup(("second",))
        with pytest.raises(StoreBusyError):
            store.lookup(("rejected",))
        assert not running.cancel(), "running native I/O must not appear cancelled"
        assert queued.cancel()
        replacement = store.lookup(("third",))
        native.release.set()
        assert running.result(timeout=5) == (False,)
        assert replacement.result(timeout=5) == (False,)
    finally:
        native.release.set()
        store.close()


def test_shutdown_drains_before_releasing_allocation_owners(store, native):
    memory = region(b"abcd")
    owner_ref = weakref.ref(memory.owner)
    store.register_region(memory)
    del memory
    gc.collect()
    assert owner_ref() is not None
    native.release.clear()
    future = store.lookup(("key",))
    assert native.entered.wait(5)
    close_started = threading.Event()

    def close():
        close_started.set()
        store.close()

    with ThreadPoolExecutor(max_workers=1) as executor:
        closing = executor.submit(close)
        try:
            assert close_started.wait(5)
            assert not closing.done() and not native.deregistered
            assert owner_ref() is not None
        finally:
            native.release.set()
        closing.result(timeout=5)
    assert future.result(timeout=5) == (False,)
    gc.collect()
    assert owner_ref() is None and not native.registered
    store.close()
    with pytest.raises(RuntimeError, match="closing or closed"):
        store.lookup(("key",))


def test_failed_deregistration_retains_ownership_for_retry(store, native):
    memory = region(b"abcd")
    owner_ref = weakref.ref(memory.owner)
    store.register_region(memory)
    del memory
    native.fail_deregister = True
    with pytest.raises(RuntimeError, match="deregistration failed"):
        store.close()
    gc.collect()
    assert owner_ref() is not None
    native.fail_deregister = False
    store.close()
    gc.collect()
    assert owner_ref() is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("workers", 0),
        ("workers", True),
        ("max_pending", -1),
        ("max_batch_objects", 0),
        ("max_batch_objects", True),
    ],
)
def test_invalid_admission_limits_are_rejected(native, field, value):
    with pytest.raises(ValueError, match="positive"):
        UMBPStore(native, SimpleNamespace(), **{field: value})


def test_unsupported_native_backend_is_not_silently_replaced(native):
    native.supports_ranged_io = lambda: False
    with pytest.raises(RuntimeError, match="does not support ranged"):
        UMBPStore(native, SimpleNamespace())
