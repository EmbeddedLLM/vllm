# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from benchmarks.kv_offload.benchmark_umbp_vs_cpu import _metric_total


def test_metric_total_matches_exact_prometheus_sample_name():
    metrics = """
vllm:request_queue_time_seconds_bucket{le="0.1"} 9
vllm:request_queue_time_seconds_sum 1.25
vllm:request_queue_time_seconds_count 3
vllm:request_queue_time_seconds_created 100
"""

    assert _metric_total(metrics, "vllm:request_queue_time_seconds_sum") == 1.25


def test_metric_total_filters_and_sums_matching_labels():
    metrics = """
vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU",engine="0"} 4
vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU",engine="0"} 8
vllm:kv_offload_total_bytes_total{transfer_type="CPU_to_GPU",engine="1"} 16
"""

    assert (
        _metric_total(
            metrics,
            "vllm:kv_offload_total_bytes_total",
            'transfer_type="CPU_to_GPU"',
        )
        == 20
    )
