# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate pre-admission DRAM/SSD-to-HBM preload through a live vLLM server."""

import argparse
import json
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


@dataclass
class ValidationResult:
    model: str
    prompt_tokens: int
    prefetch_status: str
    prefetch_time_ms: float
    post_prefetch_ttft_ms: float
    post_prefetch_latency_ms: float
    cached_prompt_tokens: int | None
    reference_output: str
    restored_output: str
    reference_token_logprob: float
    restored_token_logprob: float
    token_logprob_abs_diff: float
    output_correct: bool
    gpu_to_cpu_bytes_before_prefetch: float
    cpu_to_gpu_bytes: float
    cpu_to_gpu_time_s: float
    cpu_to_gpu_gib_per_s: float
    tier_read_bytes: float
    tier_read_time_s: float
    tier_read_gib_per_s: float
    amd_runtime_detected: bool
    passed: bool


@dataclass
class CompletionResult:
    ttft_s: float
    latency_s: float
    cached_tokens: int | None
    output: str
    token_logprob: float


def _metric_total(metrics: str, name: str, required_label: str = "") -> float:
    total = 0.0
    for line in metrics.splitlines():
        if not line.startswith(name) or required_label not in line:
            continue
        total += float(line.rsplit(maxsplit=1)[1])
    return total


def _metrics(base_url: str) -> dict[str, float]:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    body = response.text
    return {
        "cpu_to_gpu_bytes": _metric_total(
            body,
            "vllm:kv_offload_total_bytes_total",
            'transfer_type="CPU_to_GPU"',
        ),
        "cpu_to_gpu_time": _metric_total(
            body,
            "vllm:kv_offload_total_time_total",
            'transfer_type="CPU_to_GPU"',
        ),
        "gpu_to_cpu_bytes": _metric_total(
            body,
            "vllm:kv_offload_total_bytes_total",
            'transfer_type="GPU_to_CPU"',
        ),
        "tier_read_bytes": _metric_total(
            body, "vllm:kv_offload_tiering_read_bytes_total"
        ),
        "tier_read_time": _metric_total(
            body, "vllm:kv_offload_tiering_read_time_total"
        ),
        "tier_write_bytes": _metric_total(
            body, "vllm:kv_offload_tiering_write_bytes_total"
        ),
    }


def _completion(
    base_url: str,
    model: str,
    prompt: list[int],
    cache_salt: str | None,
    *,
    stream: bool,
) -> CompletionResult:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": 1,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if cache_salt:
        payload["cache_salt"] = cache_salt

    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/completions",
        json=payload,
        stream=stream,
        timeout=180,
    )
    response.raise_for_status()
    if not stream:
        elapsed = time.perf_counter() - started
        body = response.json()
        usage = body.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        choice = body["choices"][0]
        token_logprobs = choice["logprobs"]["token_logprobs"]
        return CompletionResult(
            ttft_s=elapsed,
            latency_s=elapsed,
            cached_tokens=details.get("cached_tokens"),
            output=choice["text"],
            token_logprob=float(token_logprobs[0]),
        )

    first_token_at = None
    cached_tokens = None
    output_parts = []
    token_logprob = None
    for line in response.iter_lines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        for choice in chunk.get("choices") or ():
            if first_token_at is None:
                first_token_at = time.perf_counter()
            output_parts.append(choice.get("text", ""))
            logprobs = choice.get("logprobs") or {}
            values = logprobs.get("token_logprobs") or []
            if values and token_logprob is None:
                token_logprob = float(values[0])
        usage = chunk.get("usage") or {}
        details = usage.get("prompt_tokens_details") or {}
        if details.get("cached_tokens") is not None:
            cached_tokens = int(details["cached_tokens"])
    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream ended without a completion chunk")
    if token_logprob is None:
        raise RuntimeError("stream ended without the requested token log probability")
    return CompletionResult(
        ttft_s=first_token_at - started,
        latency_s=finished - started,
        cached_tokens=cached_tokens,
        output="".join(output_parts),
        token_logprob=token_logprob,
    )


def _make_prompts(model: str, length: int, count: int) -> list[list[int]]:
    tokenizer = AutoTokenizer.from_pretrained(model)
    special_ids = set(tokenizer.all_special_ids)
    usable_ids = [
        token_id
        for token_id in range(tokenizer.vocab_size)
        if token_id not in special_ids
    ]
    if len(usable_ids) < length * count:
        raise ValueError("tokenizer vocabulary is too small for unique prompts")
    return [usable_ids[i * length : (i + 1) * length] for i in range(count)]


def _prefetch(
    base_url: str,
    model: str,
    prompt: list[int],
    cache_salt: str | None,
    timeout: float,
) -> tuple[dict[str, Any], float]:
    prefetch_id = f"hbm-local-{uuid.uuid4().hex}"
    payload: dict[str, Any] = {
        "version": "v1",
        "prefetch_id": prefetch_id,
        "model": model,
        "prompts": [prompt],
        "target_tier": "gpu",
    }
    if cache_salt:
        payload["cache_salt"] = cache_salt

    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/kv_cache/prefetch", json=payload, timeout=30
    )
    response.raise_for_status()
    result = response.json()
    deadline = time.monotonic() + timeout
    while result["status"] == "pending":
        if time.monotonic() >= deadline:
            requests.delete(
                f"{base_url}/v1/kv_cache/prefetch/{prefetch_id}", timeout=30
            )
            raise TimeoutError(f"HBM prefetch did not finish within {timeout}s")
        time.sleep(0.05)
        response = requests.get(
            f"{base_url}/v1/kv_cache/prefetch/{prefetch_id}", timeout=30
        )
        response.raise_for_status()
        result = response.json()
    return result, time.perf_counter() - started


def _detect_amd_runtime() -> bool:
    for command in (["rocminfo"], ["rocm-smi", "--showproductname"]):
        try:
            result = subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return True
    return False


def _delta(after: dict[str, float], before: dict[str, float], key: str) -> float:
    return max(after[key] - before[key], 0.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--eviction-prompts", type=int, default=4)
    parser.add_argument("--poll-timeout", type=float, default=60.0)
    parser.add_argument("--logprob-atol", type=float, default=1e-4)
    parser.add_argument("--cache-salt")
    parser.add_argument("--require-tier-read", action="store_true")
    parser.add_argument("--require-amd", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.prompt_tokens < args.block_size:
        parser.error("--prompt-tokens must contain at least one cache block")
    if args.eviction_prompts < 1:
        parser.error("--eviction-prompts must be positive")
    if args.logprob_atol < 0:
        parser.error("--logprob-atol must be non-negative")
    prompt_tokens = args.prompt_tokens // args.block_size * args.block_size
    cache_salt = args.cache_salt or f"hbm-validation-{uuid.uuid4().hex}"

    health = requests.get(f"{args.base_url}/health", timeout=10)
    health.raise_for_status()
    amd_runtime = _detect_amd_runtime()
    if args.require_amd and not amd_runtime:
        raise RuntimeError("ROCm tools cannot see an AMD runtime in this container")

    prompts = _make_prompts(args.model, prompt_tokens, args.eviction_prompts + 1)
    target, pressure = prompts[0], prompts[1:]

    print("1/4 populating and offloading the target prefix", flush=True)
    metrics_initial = _metrics(args.base_url)
    baseline = _completion(args.base_url, args.model, target, cache_salt, stream=False)
    reference = _completion(args.base_url, args.model, target, cache_salt, stream=True)
    for prompt in pressure:
        _completion(args.base_url, args.model, prompt, None, stream=False)
    metrics_before = _metrics(args.base_url)
    setup_offload_bytes = _delta(metrics_before, metrics_initial, "gpu_to_cpu_bytes")
    setup_tier_bytes = _delta(metrics_before, metrics_initial, "tier_write_bytes")
    if setup_offload_bytes <= 0 and setup_tier_bytes <= 0:
        print(
            "warning: no tier-write increase was observed during cache pressure; "
            "increase --eviction-prompts or reduce GPU cache blocks",
            flush=True,
        )

    print("2/4 requesting direct pre-admission HBM preload", flush=True)
    prefetch, prefetch_time = _prefetch(
        args.base_url,
        args.model,
        target,
        cache_salt,
        args.poll_timeout,
    )
    metrics_after_prefetch = _metrics(args.base_url)

    print("3/4 submitting the real request after preload", flush=True)
    restored = _completion(args.base_url, args.model, target, cache_salt, stream=True)

    cpu_bytes = _delta(metrics_after_prefetch, metrics_before, "cpu_to_gpu_bytes")
    cpu_time = _delta(metrics_after_prefetch, metrics_before, "cpu_to_gpu_time")
    tier_bytes = _delta(metrics_after_prefetch, metrics_before, "tier_read_bytes")
    tier_time = _delta(metrics_after_prefetch, metrics_before, "tier_read_time")
    cached_ok = restored.cached_tokens is None or (
        restored.cached_tokens >= prompt_tokens - args.block_size
    )
    logprob_diff = abs(reference.token_logprob - restored.token_logprob)
    output_correct = (
        baseline.output == reference.output == restored.output
        and logprob_diff <= args.logprob_atol
    )
    passed = (
        prefetch["status"] == "ready" and cpu_bytes > 0 and cached_ok and output_correct
    )
    if args.require_tier_read:
        passed &= tier_bytes > 0

    result = ValidationResult(
        model=args.model,
        prompt_tokens=prompt_tokens,
        prefetch_status=prefetch["status"],
        prefetch_time_ms=prefetch_time * 1000,
        post_prefetch_ttft_ms=restored.ttft_s * 1000,
        post_prefetch_latency_ms=restored.latency_s * 1000,
        cached_prompt_tokens=restored.cached_tokens,
        reference_output=reference.output,
        restored_output=restored.output,
        reference_token_logprob=reference.token_logprob,
        restored_token_logprob=restored.token_logprob,
        token_logprob_abs_diff=logprob_diff,
        output_correct=output_correct,
        gpu_to_cpu_bytes_before_prefetch=setup_offload_bytes,
        cpu_to_gpu_bytes=cpu_bytes,
        cpu_to_gpu_time_s=cpu_time,
        cpu_to_gpu_gib_per_s=(cpu_bytes / cpu_time / (1 << 30) if cpu_time else 0.0),
        tier_read_bytes=tier_bytes,
        tier_read_time_s=tier_time,
        tier_read_gib_per_s=(tier_bytes / tier_time / (1 << 30) if tier_time else 0.0),
        amd_runtime_detected=amd_runtime,
        passed=passed,
    )
    output = json.dumps(asdict(result), indent=2)
    print("4/4 validation result")
    print(output)
    if args.output:
        args.output.write_text(f"{output}\n", encoding="utf-8")
    if not passed:
        raise RuntimeError(
            "HBM preload validation failed; inspect transfer metrics, cached "
            "tokens, generated output, and token-logprob difference"
        )


if __name__ == "__main__":
    main()
