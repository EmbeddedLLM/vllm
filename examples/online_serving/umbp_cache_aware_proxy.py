# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference UMBP cache-aware proxy for text completion traffic."""

import argparse
import asyncio
import json
from collections.abc import Sequence

import aiohttp
from quart import Quart, Response, make_response, request
from transformers import AutoTokenizer

from vllm.v1.kv_offload.tiering.mori.placement import MoriPlacementClient
from vllm.v1.kv_offload.tiering.mori.routing import (
    MoriCacheAwareRouter,
    MoriReplica,
    hash_text_prefix,
)

app = Quart(__name__)
router: MoriCacheAwareRouter
tokenizer = None
hash_block_size = 16
hash_algorithm = "sha256"
group_indices = (0,)
base_model_name: str | None = None


def _token_ids(payload: dict, api: str) -> Sequence[int] | None:
    if api == "completions":
        prompt = payload.get("prompt")
        if isinstance(prompt, list) and all(isinstance(token, int) for token in prompt):
            return prompt
        if isinstance(prompt, str):
            return tokenizer.encode(prompt)
        return None
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return None
    if any(not isinstance(message.get("content"), str) for message in messages):
        return None
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=payload.get("add_generation_prompt", True),
    )


async def _select_replica(payload: dict, api: str) -> MoriReplica:
    token_ids = _token_ids(payload, api)
    if token_ids is None:
        return router.select([])
    requested_model = payload.get("model")
    lora_name = (
        requested_model
        if base_model_name and requested_model and requested_model != base_model_name
        else None
    )
    hashes = hash_text_prefix(
        token_ids,
        block_size=hash_block_size,
        hash_algorithm=hash_algorithm,
        lora_name=lora_name,
        cache_salt=payload.get("cache_salt"),
    )
    return await asyncio.to_thread(router.select, hashes, group_indices)


async def _forward(api: str):
    payload = await request.get_json()
    replica = await _select_replica(payload, api)
    headers = {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in {"content-length", "host"}
    }
    if replica.dp_rank is not None:
        headers["X-data-parallel-rank"] = str(replica.dp_rank)
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    try:
        response = await session.post(
            f"{replica.url.rstrip('/')}/v1/{api}", json=payload, headers=headers
        )
    except Exception:
        await session.close()
        router.finish(replica)
        raise

    response_headers = {
        key: value
        for key, value in response.headers.items()
        if key.lower() not in {"content-length", "transfer-encoding", "connection"}
    }
    if payload.get("stream", False):

        async def stream():
            try:
                async for chunk in response.content.iter_any():
                    yield chunk
            finally:
                response.release()
                await session.close()
                router.finish(replica)

        return Response(stream(), status=response.status, headers=response_headers)
    try:
        body = await response.read()
        return await make_response(body, response.status, response_headers)
    finally:
        response.release()
        await session.close()
        router.finish(replica)


@app.post("/v1/completions")
async def completions():
    return await _forward("completions")


@app.post("/v1/chat/completions")
async def chat_completions():
    return await _forward("chat/completions")


@app.get("/health")
async def health():
    return "ok", 200


def _parse_replicas(value: str) -> list[MoriReplica]:
    entries = json.loads(value)
    return [MoriReplica(**entry) for entry in entries]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-model-name")
    parser.add_argument("--master-address", required=True)
    parser.add_argument("--key-prefix", required=True)
    parser.add_argument("--replicas", required=True, type=_parse_replicas)
    parser.add_argument("--hash-block-size", type=int, default=16)
    parser.add_argument("--hash-algorithm", default="sha256")
    parser.add_argument("--group-indices", default="0")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args()

    global router, tokenizer, hash_block_size, hash_algorithm, group_indices
    global base_model_name
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    hash_block_size = args.hash_block_size
    hash_algorithm = args.hash_algorithm
    group_indices = tuple(int(value) for value in args.group_indices.split(","))
    base_model_name = args.base_model_name
    placement = MoriPlacementClient(args.master_address, "router", args.key_prefix)
    router = MoriCacheAwareRouter(placement, args.replicas)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
