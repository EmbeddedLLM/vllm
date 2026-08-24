# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.serve.kv_cache.api_router import attach_router


class MockEngineClient:
    def __init__(self) -> None:
        self.started: tuple[Any, ...] | None = None

    async def start_kv_prefetch(
        self,
        prefetch_id,
        prompts,
        cache_salt=None,
        lora_name=None,
        multimodal_features=None,
        target_tier="cpu",
    ):
        self.started = (
            prefetch_id,
            prompts,
            cache_salt,
            lora_name,
            multimodal_features,
            target_tier,
        )
        return {
            "prefetch_id": prefetch_id,
            "status": "pending",
            "total_blocks": 2,
            "ready_blocks": 0,
        }

    async def poll_kv_prefetch(self, prefetch_id):
        if prefetch_id == "missing":
            raise KeyError(prefetch_id)
        return {"prefetch_id": prefetch_id, "status": "ready"}

    async def cancel_kv_prefetch(self, prefetch_id):
        return {"prefetch_id": prefetch_id, "status": "cancelled"}


def _client():
    app = FastAPI()
    engine = MockEngineClient()
    app.state.engine_client = engine
    attach_router(app)
    return TestClient(app), engine


def test_prefetch_lifecycle_contract():
    client, engine = _client()
    response = client.post(
        "/v1/kv_cache/prefetch",
        json={
            "version": "v1",
            "prefetch_id": "request-1",
            "model": "Qwen/Qwen3-0.6B",
            "prompts": [[1, 2], [3, 4]],
            "cache_salt": "tenant",
            "lora_name": "adapter-a",
            "multimodal_features": [
                [{"hash": "image-hash", "offset": 1, "length": 1}],
                [],
            ],
            "target_tier": "cpu",
        },
    )

    assert response.status_code == 202
    assert response.json()["status"] == "pending"
    assert engine.started == (
        "request-1",
        [[1, 2], [3, 4]],
        "tenant",
        "adapter-a",
        [[{"hash": "image-hash", "offset": 1, "length": 1}], []],
        "cpu",
    )
    assert client.get("/v1/kv_cache/prefetch/request-1").json()["status"] == "ready"
    assert (
        client.delete("/v1/kv_cache/prefetch/request-1").json()["status"] == "cancelled"
    )


def test_prefetch_contract_validation_and_missing_id():
    client, _ = _client()

    invalid = client.post(
        "/v1/kv_cache/prefetch",
        json={"prefetch_id": "request-1", "model": "model", "prompts": []},
    )
    assert invalid.status_code == 422
    assert client.get("/v1/kv_cache/prefetch/missing").status_code == 404
