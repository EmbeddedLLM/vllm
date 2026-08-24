# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate same-host remote UMBP reads and placement reconciliation."""

import argparse
import contextlib
import ctypes
import socket
import subprocess
import time
from pathlib import Path

from mori.cpp import UMBPClient, UMBPConfig, UMBPDistributedConfig, UMBPMasterClient

from vllm.distributed.kv_events import MEDIUM_STORAGE, BlockStored, KVEventBatch
from vllm.v1.kv_offload.tiering.mori.placement import (
    MoriPlacementReconciler,
    encode_umbp_event_key,
)


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_until(predicate, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.1)
    raise TimeoutError("timed out waiting for UMBP placement")


def make_client(master_address: str, node_id: str) -> UMBPClient:
    config = UMBPConfig()
    config.dram.capacity_bytes = 8 * 1024 * 1024
    config.ssd.enabled = False
    distributed = UMBPDistributedConfig()
    distributed.master_config.master_address = master_address
    distributed.master_config.node_id = node_id
    distributed.master_config.node_address = "127.0.0.1"
    distributed.master_config.auto_heartbeat = True
    distributed.peer_service_port = free_port()
    distributed.io_engine.host = "127.0.0.1"
    distributed.cache_remote_fetches = False
    config.distributed = distributed
    return UMBPClient(config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-bin", type=Path, required=True)
    parser.add_argument("--size", type=int, default=4096)
    parser.add_argument("--key-prefix", default="vllm:validation:")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.size <= 0:
        raise ValueError("--size must be greater than zero")
    if not args.master_bin.is_file():
        raise FileNotFoundError(args.master_bin)

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
            lambda: master.poll() is None and _port_is_open("127.0.0.1", master_port)
        )
        source = make_client(master_address, "replica-a")
        destination = make_client(master_address, "replica-b")
        query = UMBPMasterClient(master_address)
        block_hash = b"remote-block"
        key = encode_umbp_event_key(block_hash, None, args.key_prefix)

        source_data = (ctypes.c_ubyte * args.size)(
            *((index * 131 + 17) % 256 for index in range(args.size))
        )
        restored_a = (ctypes.c_ubyte * args.size)()
        restored_b = (ctypes.c_ubyte * args.size)()
        source_ptr = ctypes.addressof(source_data)
        restored_a_ptr = ctypes.addressof(restored_a)
        restored_b_ptr = ctypes.addressof(restored_b)
        assert source.register_memory(source_ptr, args.size)
        assert source.register_memory(restored_a_ptr, args.size)
        assert destination.register_memory(restored_b_ptr, args.size)
        assert source.put_from_ptr(key, source_ptr, args.size)
        source.flush()

        wait_until(lambda: query.batch_inspect([key])[0] is not None)
        placement = query.batch_inspect([key])[0]
        if placement.node_id == "replica-a":
            reader = destination
            reader_id = "replica-b"
            restored_data = restored_b
            restored_ptr = restored_b_ptr
        elif placement.node_id == "replica-b":
            reader = source
            reader_id = "replica-a"
            restored_data = restored_a
            restored_ptr = restored_a_ptr
        else:
            raise AssertionError(f"unexpected placement node: {placement.node_id}")
        assert reader.get_into_ptr(key, restored_ptr, args.size)
        assert bytes(restored_data) == bytes(source_data), "remote data mismatch"

        reconciler = MoriPlacementReconciler(
            query,
            node_id=reader_id,
            key_prefix=args.key_prefix,
        )
        reconciler.observe_batch(
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
        events = reconciler.reconcile()
        assert len(events) == 1
        assert events[0].source_node == placement.node_id
        assert events[0].locality == "REMOTE"
        assert events[0].storage_tier == "DRAM"
        print(
            f"PASS: {reader_id} restored {args.size} correct bytes from "
            f"{placement.node_id}; placement=REMOTE:DRAM"
        )
    finally:
        for client in (destination, source):
            if client is not None:
                with contextlib.suppress(Exception):
                    client.close()
        master.terminate()
        with contextlib.suppress(subprocess.TimeoutExpired):
            master.wait(timeout=5)
        if master.poll() is None:
            master.kill()
            master.wait()


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.2):
            return True
    except OSError:
        return False


if __name__ == "__main__":
    main()
