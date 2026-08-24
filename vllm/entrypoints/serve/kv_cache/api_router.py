# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Literal

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from pydantic import BaseModel, Field

from vllm.engine.protocol import EngineClient

router = APIRouter(prefix="/v1/kv_cache", tags=["KV cache"])


class KVPrefetchMMFeature(BaseModel):
    hash: str = Field(min_length=1)
    offset: int = Field(ge=0)
    length: int = Field(gt=0)


class KVPrefetchRequest(BaseModel):
    version: Literal["v1"] = "v1"
    prefetch_id: str = Field(min_length=1)
    model: str = Field(min_length=1)
    prompts: list[list[int]] = Field(min_length=1)
    cache_salt: str | None = None
    lora_name: str | None = None
    multimodal_features: list[list[KVPrefetchMMFeature]] | None = None
    target_tier: Literal["cpu", "gpu"] = "cpu"


def engine_client(request: Request) -> EngineClient:
    return request.app.state.engine_client


@router.post("/prefetch", status_code=status.HTTP_202_ACCEPTED)
async def start_prefetch(body: KVPrefetchRequest, raw_request: Request):
    try:
        return await engine_client(raw_request).start_kv_prefetch(
            body.prefetch_id,
            body.prompts,
            body.cache_salt,
            body.lora_name,
            (
                [
                    [feature.model_dump() for feature in prompt]
                    for prompt in body.multimodal_features
                ]
                if body.multimodal_features is not None
                else None
            ),
            body.target_tier,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/prefetch/{prefetch_id}")
async def poll_prefetch(prefetch_id: str, raw_request: Request):
    try:
        return await engine_client(raw_request).poll_kv_prefetch(prefetch_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="unknown prefetch_id") from error


@router.delete("/prefetch/{prefetch_id}")
async def cancel_prefetch(prefetch_id: str, raw_request: Request):
    try:
        return await engine_client(raw_request).cancel_kv_prefetch(prefetch_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail="unknown prefetch_id") from error


def attach_router(app: FastAPI) -> None:
    app.include_router(router)
