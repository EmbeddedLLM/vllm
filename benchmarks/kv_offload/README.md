# KV offload restoration benchmark

## Validate direct HBM preload

See the complete [HBM preload validation runbook](../../docs/features/hbm_prefetch_validation.md)
for server configurations, correctness criteria, UMBP SSD validation, expected
output, and troubleshooting.

With an AMD vLLM server running with prefix caching, the offloading connector,
metrics, and `--enable-prompt-tokens-details`, run:

```bash
.venv/bin/python benchmarks/kv_offload/validate_hbm_prefetch.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B \
  --prompt-tokens 128 \
  --eviction-prompts 4 \
  --require-amd \
  --output /tmp/hbm-prefetch-validation.json
```

The script populates a target prefix, pressures HBM with unique prefixes,
requests `target_tier: "gpu"`, and then sends the real request. A pass requires
a `ready` preload, a positive `CPU_to_GPU` byte delta during preload, and—when
the server exposes it—a near-full cached-token count on the real request. It
also requires the generated token to match both the seed and normal HBM-hit
reference, and the restored token log probability to match the HBM reference
within `--logprob-atol`.

Add `--require-tier-read` to prove an SSD/secondary-tier read also occurred in
the preload interval. If no tier write or CPU-to-GPU transfer is observed,
increase `--eviction-prompts` or start the server with fewer GPU cache blocks.

In a container without exposed AMD devices, omit `--require-amd` to exercise
the HTTP workflow against a remote server; local ROCm detection will be reported
separately and does not by itself prove where that remote server ran.

`benchmark_umbp_vs_cpu.py` compares cache-hit restoration after forced GPU
eviction. It uses fixed token-ID prompts, emits one output token, and reports
streaming time to first token (TTFT), total request latency, transfer bytes,
transfer time, and effective bandwidth. Metrics are bracketed around every
timed request: non-recompute modes fail immediately if any trial lacks a
CPU-to-GPU transfer, and filesystem/UMBP modes also require a secondary-tier
read in every trial. Aggregate bytes cannot hide an unproven trial.

Run each mode from a fresh server. Reusing a server contaminates later trials
because the deterministic eviction prompts remain cached.

For a recomputation baseline, omit `--kv-transfer-config` from the server and
run the benchmark with `--mode recompute`.

## CPU tier

```bash
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m vllm.entrypoints.cli.main serve \
  Qwen/Qwen3-0.6B --port 8000 --enforce-eager \
  --enable-prefix-caching --max-model-len 224 \
  --num-gpu-blocks-override 16 --gpu-memory-utilization 0.2 \
  --kv-transfer-config '{
    "kv_connector":"OffloadingConnector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "spec_name":"CPUOffloadingSpec",
      "cpu_bytes_to_use":268435456,
      "offload_prompt_only":false
    }
  }'
```

```bash
.venv/bin/python benchmarks/kv_offload/benchmark_umbp_vs_cpu.py \
  --base-url http://127.0.0.1:8000 --mode cpu --trials 10
```

## Standalone UMBP DRAM tier

```bash
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m vllm.entrypoints.cli.main serve \
  Qwen/Qwen3-0.6B --port 8000 --enforce-eager \
  --enable-prefix-caching --max-model-len 224 \
  --num-gpu-blocks-override 16 --gpu-memory-utilization 0.2 \
  --kv-transfer-config '{
    "kv_connector":"OffloadingConnector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "spec_name":"TieringOffloadingSpec",
      "cpu_bytes_to_use":16777216,
      "offload_prompt_only":false,
      "secondary_tiers":[{
        "type":"mori",
        "dram_capacity_bytes":1073741824,
        "io_threads":2,
        "key_prefix":"vllm:perf:"
      }]
    }
  }'
```

```bash
.venv/bin/python benchmarks/kv_offload/benchmark_umbp_vs_cpu.py \
  --base-url http://127.0.0.1:8000 --mode umbp --trials 10
```

The CPU tier is the latency baseline because it restores directly to GPU.
UMBP first reads into the CPU primary tier and then performs the same CPU-to-GPU
copy, so a local UMBP hit is not expected to beat a CPU hit. UMBP should also be
evaluated for capacity-weighted hit rate, aggregate concurrency, SSD spill, and
cross-replica reuse; those benefits can avoid recomputation even when an
individual hit has higher latency.

## Longer prefixes

To test whether transfer overhead is amortized by avoided prefill work, repeat
all three modes with a larger cache and prompt. For example, use
`--max-model-len 768 --num-gpu-blocks-override 64`, increase the CPU baseline
to 1 GiB, increase UMBP to 2 GiB with a 64 MiB CPU primary tier, and run:

```bash
.venv/bin/python benchmarks/kv_offload/benchmark_umbp_vs_cpu.py \
  --base-url http://127.0.0.1:8000 --mode MODE \
  --prompt-tokens 512 --trials 10
```

Keep the model, GPU, block count, prompt sequence, and number of trials equal
between modes. Report both TTFT and transfer bandwidth. For capacity or remote
reuse claims, add a separate concurrent workload and report request throughput,
cache hit rate, and recomputed tokens; the single-request benchmark does not
measure those system-level benefits.
