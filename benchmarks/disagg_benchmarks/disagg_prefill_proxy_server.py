# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from collections.abc import AsyncGenerator

import aiohttp
from quart import Quart, abort, make_response, request

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)
logger = logging.getLogger("vllm-proxy")


async def forward_request(
    url: str, data: dict, is_stream: bool
) -> AsyncGenerator[bytes, None]:
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        try:
            async with session.post(url, json=data, headers=headers) as response:
                if response.status != 200:
                    logger.error(
                        "Backend error: %d %s", response.status, await response.text()
                    )
                    abort(500, description="Backend service error")

                if is_stream:
                    async for chunk in response.content.iter_any():
                        yield chunk
                else:
                    content = await response.read()
                    yield content
        except aiohttp.ClientError as e:
            logger.error("Connection error: %s", str(e))
            abort(503, description="Service unavailable")


@app.route("/v1/completions", methods=["POST"])
async def handle_request():
    try:
        original_data = await request.get_json()
        is_stream = original_data.get("stream", False)

        # change max_tokens = 1 to let it only do prefill
        prefill_data = original_data.copy()
        prefill_data["max_tokens"] = 1
        prefill_data["stream"] = False  # Force non-stream for prefill
        prefill_data["stream_options"] = None

        # execute prefill and verify success
        prefill_buffer = b""
        async for chunk in forward_request(
            "http://localhost:8100/v1/completions", prefill_data, is_stream=False
        ):
            prefill_buffer += chunk

        try:
            prefill_result = json.loads(prefill_buffer.decode())
            if not prefill_result.get("choices"):
                logger.error("Prefill failed to generate first token")
                abort(500, description="Prefill phase failed")
        except json.JSONDecodeError:
            logger.error("Invalid prefill response")
            abort(502, description="Invalid backend response")

        async def response_generator():
            try:
                async for chunk in forward_request(
                    "http://localhost:8200/v1/completions",
                    original_data,
                    is_stream=is_stream,
                ):
                    if is_stream:
                        yield chunk
                    else:
                        yield chunk
            except Exception as e:
                logger.error("Decode phase error: %s", {str(e)})
                yield json.dumps(
                    {"error": "Decode phase failed", "message": str(e)}
                ).encode()

        # return decode
        if is_stream:
            response = await make_response(response_generator())
            response.timeout = None
            response.headers["Content-Type"] = "text/event-stream"
            return response
        else:
            buffer = b""
            async for chunk in response_generator():
                buffer += chunk
            try:
                # Validate JSON for non-stream responses
                json.loads(buffer.decode())
                return buffer.decode(), 200, {"Content-Type": "application/json"}
            except json.JSONDecodeError:
                abort(502, description="Invalid JSON response from backend")

    except Exception as e:
        logger.error("Request handling failed: %s", {str(e)})
        return {"error": str(e)}, 500


if __name__ == "__main__":
    app.run(port=8000)
