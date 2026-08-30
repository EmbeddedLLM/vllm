# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate remote UMBP reads and logical availability events."""

import argparse
import contextlib
import ctypes
import hashlib
import socket
import subprocess
import time
from pathlib import Path

from mori.cpp import UMBPClient, UMBPConfig, UMBPDistributedConfig

from vllm.distributed.kv_events import MEDIUM_STORAGE, BlockStored, KVEventBatch
from vllm.v1.kv_offload.tiering.mori.placement import (
    encode_umbp_event_key,
    enrich_umbp_logical_placement,
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(predicate, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise TimeoutError("timed out waiting for UMBP placement")


def close_client(client: UMBPClient) -> None:
    close = getattr(client, "close", None)
    if close is not None:
        close()


def make_client(
    master_address: str,
    node_id: str,
    node_address: str,
    io_engine_port: int = 0,
    peer_service_port: int = 0,
) -> UMBPClient:
    config = UMBPConfig()
    config.dram.capacity_bytes = 8 * 1024 * 1024
    config.ssd.enabled = False
    distributed = UMBPDistributedConfig()
    distributed.master_config.master_address = master_address
    distributed.master_config.node_id = node_id
    distributed.master_config.node_address = node_address
    distributed.master_config.auto_heartbeat = True
    distributed.peer_service_port = peer_service_port or free_port()
    distributed.io_engine.host = node_address
    distributed.io_engine.port = io_engine_port
    distributed.cache_remote_fetches = False
    config.distributed = distributed
    return UMBPClient(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--role", choices=("local", "source", "reader"), default="local"
    )
    parser.add_argument("--master-bin", type=Path)
    parser.add_argument("--master-address")
    parser.add_argument("--node-id")
    parser.add_argument("--node-address", default="127.0.0.1")
    parser.add_argument("--io-engine-port", type=int, default=0)
    parser.add_argument("--peer-service-port", type=int, default=0)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument(
        "--reader-staging",
        action="store_true",
        help="read through Mori's registered staging buffer instead of zero-copy",
    )
    parser.add_argument("--key-prefix", default="vllm:validation:")
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def _payload(size: int) -> ctypes.Array:
    return (ctypes.c_ubyte * size)(*((index * 131 + 17) % 256 for index in range(size)))


def _assert_payload(restored: ctypes.Array, expected: ctypes.Array) -> None:
    actual_bytes = bytes(restored)
    expected_bytes = bytes(expected)
    if actual_bytes == expected_bytes:
        return
    first = next(
        index
        for index, (actual, wanted) in enumerate(zip(actual_bytes, expected_bytes))
        if actual != wanted
    )
    start = max(0, first - 8)
    end = min(len(actual_bytes), first + 24)
    mismatch_count = sum(
        actual != wanted for actual, wanted in zip(actual_bytes, expected_bytes)
    )
    raise AssertionError(
        "remote data mismatch: "
        f"first_offset={first} mismatched_bytes={mismatch_count}/{len(actual_bytes)} "
        f"actual_sha256={hashlib.sha256(actual_bytes).hexdigest()} "
        f"expected_sha256={hashlib.sha256(expected_bytes).hexdigest()} "
        f"actual[{start}:{end}]={actual_bytes[start:end].hex()} "
        f"expected[{start}:{end}]={expected_bytes[start:end].hex()}"
    )


def _keys(key_prefix: str) -> tuple[bytes, str, str]:
    block_hash = b"remote-block"
    key = encode_umbp_event_key(block_hash, None, key_prefix)
    return block_hash, key, f"{key}:reader-verified"


def _validate_logical_event(block_hash: bytes) -> None:
    event_batch = enrich_umbp_logical_placement(
        KVEventBatch(
            ts=time.time(),
            events=[
                BlockStored(
                    block_hashes=[block_hash],
                    parent_block_hash=None,
                    token_ids=[],
                    block_size=0,
                    lora_id=None,
                    lora_name=None,
                    medium=MEDIUM_STORAGE,
                )
            ],
        )
    )
    event = event_batch.events[0]
    assert event.storage_tier == "UMBP"
    assert event.source_node is None
    assert event.locality is None


def _run_source(args: argparse.Namespace) -> None:
    client = make_client(
        args.master_address,
        args.node_id or "replica-a",
        args.node_address,
        args.io_engine_port,
        args.peer_service_port,
    )
    try:
        block_hash, key, ack_key = _keys(args.key_prefix)
        source_data = _payload(args.size)
        source_ptr = ctypes.addressof(source_data)
        assert client.register_memory(source_ptr, args.size)
        assert client.put_from_ptr(key, source_ptr, args.size)
        client.flush()
        wait_until(lambda: client.exists(key), args.timeout)
        source_check = (ctypes.c_ubyte * args.size)()
        source_check_ptr = ctypes.addressof(source_check)
        ctypes.memset(source_check_ptr, 0xA5, args.size)
        assert client.register_memory(source_check_ptr, args.size)
        assert client.get_into_ptr(key, source_check_ptr, args.size)
        _assert_payload(source_check, source_data)
        _validate_logical_event(block_hash)
        print(f"SELF-CHECK: source restored {args.size} byte-correct bytes", flush=True)
        print(f"READY: source owns {args.size} bytes at key={key}", flush=True)
        wait_until(lambda: client.exists(ack_key), args.timeout)
        print("PASS: reader acknowledged byte-correct restore", flush=True)
    finally:
        close_client(client)


def _run_reader(args: argparse.Namespace) -> None:
    client = make_client(
        args.master_address,
        args.node_id or "replica-b",
        args.node_address,
        args.io_engine_port,
        args.peer_service_port,
    )
    try:
        _, key, ack_key = _keys(args.key_prefix)
        wait_until(lambda: client.exists(key), args.timeout)
        expected = _payload(args.size)
        restored = (ctypes.c_ubyte * args.size)()
        restored_ptr = ctypes.addressof(restored)
        ctypes.memset(restored_ptr, 0xA5, args.size)
        if not args.reader_staging:
            assert client.register_memory(restored_ptr, args.size)
        path = "staging" if args.reader_staging else "zero-copy"
        print(f"READ: requesting {args.size} bytes through path={path}", flush=True)
        assert client.get_into_ptr(key, restored_ptr, args.size)
        _assert_payload(restored, expected)
        ack = (ctypes.c_ubyte * 1)(1)
        assert client.register_memory(ctypes.addressof(ack), 1)
        assert client.put_from_ptr(ack_key, ctypes.addressof(ack), 1)
        client.flush()
        print(
            f"PASS: restored {args.size} byte-correct bytes; "
            f"availability=UMBP path={path}"
        )
        # Keep the acknowledgement resident long enough for the source's
        # polling loop to observe it before this client unregisters.
        time.sleep(2.0)
    finally:
        close_client(client)


def _run_local(args: argparse.Namespace) -> None:
    if args.master_bin is None or not args.master_bin.is_file():
        raise FileNotFoundError("--master-bin is required for role=local")

    master_port = free_port()
    master_address = f"127.0.0.1:{master_port}"
    master = subprocess.Popen(
        [str(args.master_bin), master_address, "0"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    source = destination = None
    try:
        wait_until(
            lambda: master.poll() is None and _port_is_open("127.0.0.1", master_port),
            args.timeout,
        )
        source = make_client(master_address, "replica-a", "127.0.0.1")
        block_hash, key, _ = _keys(args.key_prefix)

        source_data = _payload(args.size)
        restored_b = (ctypes.c_ubyte * args.size)()
        source_ptr = ctypes.addressof(source_data)
        restored_b_ptr = ctypes.addressof(restored_b)
        assert source.register_memory(source_ptr, args.size)
        assert source.put_from_ptr(key, source_ptr, args.size)
        source.flush()
        wait_until(lambda: source.exists(key), args.timeout)

        destination = make_client(master_address, "replica-b", "127.0.0.1")
        ctypes.memset(restored_b_ptr, 0xA5, args.size)
        if not args.reader_staging:
            assert destination.register_memory(restored_b_ptr, args.size)
        assert destination.get_into_ptr(key, restored_b_ptr, args.size)
        _assert_payload(restored_b, source_data)

        _validate_logical_event(block_hash)
        path = "staging" if args.reader_staging else "zero-copy"
        print(
            f"PASS: replica-b restored {args.size} correct bytes from "
            f"replica-a; availability=UMBP path={path}"
        )
    finally:
        for client in (destination, source):
            if client is not None:
                with contextlib.suppress(Exception):
                    close_client(client)
        master.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            master.wait(timeout=5)
        if master.poll() is None:
            master.kill()
            master.wait()


def main() -> None:
    args = parse_args()
    if args.size <= 0 or args.timeout <= 0:
        raise ValueError("--size and --timeout must be greater than zero")
    if args.role == "local":
        _run_local(args)
        return
    if not args.master_address:
        raise ValueError("--master-address is required for source and reader roles")
    if args.role == "source":
        _run_source(args)
    else:
        _run_reader(args)


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    main()
