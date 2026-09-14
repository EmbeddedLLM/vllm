# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; the fake models the pinned MoRI ranged API, not RDMA."""

import ctypes
import gc
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

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
    "field,value", [("workers", 0), ("workers", True), ("max_pending", -1)]
)
def test_invalid_admission_limits_are_rejected(native, field, value):
    with pytest.raises(ValueError, match="positive"):
        UMBPStore(native, SimpleNamespace(), **{field: value})


def test_unsupported_native_backend_is_not_silently_replaced(native):
    native.supports_ranged_io = lambda: False
    with pytest.raises(RuntimeError, match="does not support ranged"):
        UMBPStore(native, SimpleNamespace())
