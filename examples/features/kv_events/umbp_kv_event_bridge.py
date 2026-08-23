# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Forward vLLM KV-cache events to the UMBP placement directory."""

import argparse

import msgspec
import zmq

from vllm.distributed.kv_events import KVEventBatch
from vllm.v1.kv_offload.tiering.mori.placement import MoriPlacementClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://127.0.0.1:5557")
    parser.add_argument("--topic", default="kv-events")
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--node-id", required=True)
    parser.add_argument("--key-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    placement = MoriPlacementClient(
        master_address=args.master_address,
        node_id=args.node_id,
        key_prefix=args.key_prefix,
    )
    decoder = msgspec.msgpack.Decoder(type=KVEventBatch)
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)
    subscriber.connect(args.endpoint)
    subscriber.setsockopt_string(zmq.SUBSCRIBE, args.topic)
    try:
        while True:
            _, _, payload = subscriber.recv_multipart()
            placement.process_batch(decoder.decode(payload))
    except KeyboardInterrupt:
        pass
    finally:
        subscriber.close(linger=0)
        context.term()


if __name__ == "__main__":
    main()
