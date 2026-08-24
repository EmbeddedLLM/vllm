# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proxy vLLM KV events and add authoritative UMBP placement refreshes."""

import argparse
import json
import logging
import time

import msgspec
import zmq

from vllm.distributed.kv_events import KVEventBatch, ZmqEventPublisher
from vllm.v1.kv_offload.tiering.mori.placement import (
    MoriPlacementClient,
    MoriPlacementReconciler,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--output-endpoint", default="tcp://*:5558")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--key-prefix", required=True)
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--max-batch-size", type=int, default=1024)
    parser.add_argument(
        "--bandwidth-bps",
        type=json.loads,
        default={},
        help='JSON map such as {"LOCAL:DRAM":4e9,"REMOTE:SSD":1e9}',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_interval <= 0:
        raise ValueError("--poll-interval must be greater than zero")
    if args.endpoint == args.output_endpoint:
        raise ValueError("input and output endpoints must differ")
    from mori.cpp import UMBPMasterClient

    master_client = UMBPMasterClient(args.master_address)
    placement = MoriPlacementClient(
        master_address=args.master_address,
        node_id=args.node_id,
        key_prefix=args.key_prefix,
        client=master_client,
    )
    reconciler = MoriPlacementReconciler(
        master_client=master_client,
        node_id=args.node_id,
        key_prefix=args.key_prefix,
        bandwidth_bps=args.bandwidth_bps,
        max_batch_size=args.max_batch_size,
    )
    decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(args.endpoint)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    publisher = ZmqEventPublisher(
        data_parallel_rank=0,
        endpoint=args.output_endpoint,
        topic=args.topic,
    )
    poller = zmq.Poller()
    poller.register(subscriber, zmq.POLLIN)
    next_reconcile = time.monotonic()
    try:
        while True:
            timeout_ms = max(0, int((next_reconcile - time.monotonic()) * 1000))
            if subscriber in dict(poller.poll(timeout_ms)):
                _, _, payload = subscriber.recv_multipart()
                batch = decoder.decode(payload)
                placement.process_batch(batch)
                reconciler.observe_batch(batch)
                publisher.publish(batch)
            if time.monotonic() >= next_reconcile:
                try:
                    events = reconciler.reconcile()
                    if events:
                        publisher.publish(KVEventBatch(ts=time.time(), events=events))
                except Exception:
                    logger.exception("UMBP placement reconciliation failed")
                next_reconcile = time.monotonic() + args.poll_interval
    except KeyboardInterrupt:
        pass
    finally:
        publisher.shutdown()
        subscriber.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
