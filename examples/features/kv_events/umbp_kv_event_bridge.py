# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Proxy vLLM KV events and label logical UMBP availability."""

import argparse

import msgspec
import zmq

from vllm.distributed.kv_events import KVEventBatch, ZmqEventPublisher
from vllm.v1.kv_offload.tiering.mori.placement import (
    MoriPhysicalPlacementResolver,
    MoriPlacementClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--output-endpoint", default="tcp://*:5558")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--key-prefix", required=True)
    parser.add_argument("--placement-lookup-timeout", type=float, default=2.0)
    parser.add_argument("--placement-lookup-interval", type=float, default=0.02)
    for locality in ("local", "remote"):
        for tier in ("hbm", "dram", "ssd"):
            parser.add_argument(
                f"--{locality}-{tier}-bandwidth-bps",
                type=float,
                help=f"Estimated {locality.upper()} {tier.upper()} read bandwidth",
            )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    bandwidth_bps = {
        (locality.upper(), tier.upper()): value
        for locality in ("local", "remote")
        for tier in ("hbm", "dram", "ssd")
        if (value := getattr(args, f"{locality}_{tier}_bandwidth_bps")) is not None
    }
    physical_placement = MoriPhysicalPlacementResolver(
        placement=placement,
        node_id=args.node_id,
        lookup_timeout_s=args.placement_lookup_timeout,
        lookup_interval_s=args.placement_lookup_interval,
        bandwidth_bps=bandwidth_bps,
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
    try:
        while True:
            _, _, payload = subscriber.recv_multipart()
            batch = decoder.decode(payload)
            placement.process_batch(batch)
            publisher.publish(physical_placement.enrich_batch(batch))
    except KeyboardInterrupt:
        pass
    finally:
        publisher.shutdown()
        subscriber.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
