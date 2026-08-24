# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure forced-eviction KV restoration latency through a vLLM server."""

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass

import requests
from transformers import AutoTokenizer


@dataclass
class Result:
    mode: str
    trials: int
    prompt_tokens: int
    ttft_ms_mean: float
    ttft_ms_p50: float
    ttft_ms_p90: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p90: float
    cpu_to_gpu_bytes: float
    cpu_to_gpu_time_s: float
    tier_read_bytes: float
    tier_read_time_s: float
    tier_write_bytes: float
    cpu_to_gpu_gib_per_s: float
    tier_read_gib_per_s: float
    restorations_proven: int
    min_cpu_to_gpu_bytes_per_trial: float
    min_tier_read_bytes_per_trial: float
    lookup_sync_time_s: float
    lookup_async_time_s: float
    request_queue_time_s: float
    server_e2e_time_s: float


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile)
    return ordered[index]


def _metric_total(metrics: str, name: str, required_label: str = "") -> float:
    total = 0.0
    for line in metrics.splitlines():
        if not line or line.startswith("#"):
            continue
        metric_name = line.split(maxsplit=1)[0].split("{", 1)[0]
        if metric_name != name or required_label not in line:
            continue
        total += float(line.rsplit(maxsplit=1)[1])
    return total


def _metrics(base_url: str) -> dict[str, float]:
    response = requests.get(f"{base_url}/metrics", timeout=30)
    response.raise_for_status()
    body = response.text
    return {
        "cpu_to_gpu": _metric_total(
            body,
            "vllm:kv_offload_total_bytes_total",
            'transfer_type="CPU_to_GPU"',
        ),
        "cpu_to_gpu_time": _metric_total(
            body,
            "vllm:kv_offload_total_time_total",
            'transfer_type="CPU_to_GPU"',
        ),
        "tier_read": _metric_total(body, "vllm:kv_offload_tiering_read_bytes_total"),
        "tier_read_time": _metric_total(
            body, "vllm:kv_offload_tiering_read_time_total"
        ),
        "tier_write": _metric_total(body, "vllm:kv_offload_tiering_write_bytes_total"),
        "lookup_sync_time": _metric_total(
            body, "vllm:kv_offload_tiering_lookup_sync_delay_seconds_sum"
        ),
        "lookup_async_time": _metric_total(
            body, "vllm:kv_offload_tiering_lookup_async_delay_seconds_sum"
        ),
        "request_queue_time": _metric_total(
            body, "vllm:request_queue_time_seconds_sum"
        ),
        "server_e2e_time": _metric_total(body, "vllm:e2e_request_latency_seconds_sum"),
    }


def _completion(
    base_url: str,
    model: str,
    prompt: list[int],
    *,
    stream: bool,
) -> tuple[float, float]:
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "stream": stream,
        },
        stream=stream,
        timeout=120,
    )
    response.raise_for_status()
    if not stream:
        elapsed = time.perf_counter() - started
        return elapsed, elapsed

    first_token_at = None
    for line in response.iter_lines():
        if not line.startswith(b"data: ") or line == b"data: [DONE]":
            continue
        chunk = json.loads(line[6:])
        if chunk.get("choices") and first_token_at is None:
            first_token_at = time.perf_counter()
    finished = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("stream ended without a completion chunk")
    return first_token_at - started, finished - started


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
    return [usable_ids[index * length : (index + 1) * length] for index in range(count)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--mode", choices=("recompute", "cpu", "fs", "umbp"), required=True
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--eviction-prompts", type=int, default=3)
    args = parser.parse_args()
    if args.trials < 1 or args.eviction_prompts < 1:
        parser.error("trials and eviction-prompts must be positive")

    prompts = _make_prompts(
        args.model,
        args.prompt_tokens,
        1 + args.trials * args.eviction_prompts,
    )
    target = prompts[0]
    _completion(args.base_url, args.model, target, stream=False)
    metrics_before = _metrics(args.base_url)
    ttfts = []
    latencies = []
    trial_cpu_to_gpu_bytes = []
    trial_tier_read_bytes = []

    next_prompt = 1
    for trial in range(args.trials):
        _completion(args.base_url, args.model, target, stream=False)
        for _ in range(args.eviction_prompts):
            _completion(
                args.base_url,
                args.model,
                prompts[next_prompt],
                stream=False,
            )
            next_prompt += 1
        trial_metrics_before = _metrics(args.base_url)
        ttft, latency = _completion(args.base_url, args.model, target, stream=True)
        trial_metrics_after = _metrics(args.base_url)
        cpu_restore_bytes = (
            trial_metrics_after["cpu_to_gpu"] - trial_metrics_before["cpu_to_gpu"]
        )
        tier_restore_bytes = (
            trial_metrics_after["tier_read"] - trial_metrics_before["tier_read"]
        )
        trial_cpu_to_gpu_bytes.append(cpu_restore_bytes)
        trial_tier_read_bytes.append(tier_restore_bytes)
        if args.mode != "recompute" and cpu_restore_bytes <= 0:
            raise RuntimeError(
                f"trial {trial + 1} did not perform a CPU-to-GPU restoration; "
                "increase eviction pressure"
            )
        if args.mode in ("fs", "umbp") and tier_restore_bytes <= 0:
            raise RuntimeError(
                f"trial {trial + 1} did not perform a {args.mode} tier read; "
                "reduce faster-tier capacity or increase eviction pressure"
            )
        ttfts.append(ttft * 1000)
        latencies.append(latency * 1000)
        print(
            f"trial={trial + 1} ttft_ms={ttfts[-1]:.3f} latency_ms={latencies[-1]:.3f}",
            flush=True,
        )

    metrics_after = _metrics(args.base_url)
    cpu_to_gpu_bytes = metrics_after["cpu_to_gpu"] - metrics_before["cpu_to_gpu"]
    cpu_to_gpu_time = (
        metrics_after["cpu_to_gpu_time"] - metrics_before["cpu_to_gpu_time"]
    )
    tier_read_bytes = metrics_after["tier_read"] - metrics_before["tier_read"]
    tier_read_time = metrics_after["tier_read_time"] - metrics_before["tier_read_time"]
    result = Result(
        mode=args.mode,
        trials=args.trials,
        prompt_tokens=args.prompt_tokens,
        ttft_ms_mean=statistics.mean(ttfts),
        ttft_ms_p50=_percentile(ttfts, 0.5),
        ttft_ms_p90=_percentile(ttfts, 0.9),
        latency_ms_mean=statistics.mean(latencies),
        latency_ms_p50=_percentile(latencies, 0.5),
        latency_ms_p90=_percentile(latencies, 0.9),
        cpu_to_gpu_bytes=cpu_to_gpu_bytes,
        cpu_to_gpu_time_s=cpu_to_gpu_time,
        tier_read_bytes=tier_read_bytes,
        tier_read_time_s=tier_read_time,
        tier_write_bytes=metrics_after["tier_write"] - metrics_before["tier_write"],
        cpu_to_gpu_gib_per_s=(
            cpu_to_gpu_bytes / cpu_to_gpu_time / (1 << 30) if cpu_to_gpu_time else 0.0
        ),
        tier_read_gib_per_s=(
            tier_read_bytes / tier_read_time / (1 << 30) if tier_read_time else 0.0
        ),
        restorations_proven=sum(value > 0 for value in trial_cpu_to_gpu_bytes),
        min_cpu_to_gpu_bytes_per_trial=min(trial_cpu_to_gpu_bytes),
        min_tier_read_bytes_per_trial=min(trial_tier_read_bytes),
        lookup_sync_time_s=(
            metrics_after["lookup_sync_time"] - metrics_before["lookup_sync_time"]
        ),
        lookup_async_time_s=(
            metrics_after["lookup_async_time"] - metrics_before["lookup_async_time"]
        ),
        request_queue_time_s=(
            metrics_after["request_queue_time"] - metrics_before["request_queue_time"]
        ),
        server_e2e_time_s=(
            metrics_after["server_e2e_time"] - metrics_before["server_e2e_time"]
        ),
    )
    if args.mode != "recompute" and result.cpu_to_gpu_bytes <= 0:
        raise RuntimeError("benchmark did not force any CPU-to-GPU restoration")
    if args.mode in ("fs", "umbp") and result.tier_read_bytes <= 0:
        raise RuntimeError(f"benchmark did not force any {args.mode} restoration")
    print(json.dumps(asdict(result), indent=2))


if __name__ == "__main__":
    main()
