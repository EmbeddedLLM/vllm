# Validating pre-admission HBM preload

This runbook validates that pre-admission KV preload performs a physical
host-to-device transfer, publishes correct KV data in vLLM's HBM prefix cache,
and prevents prompt recomputation for the following request.

## What the validator checks

`benchmarks/kv_offload/validate_hbm_prefetch.py` fails unless all applicable
checks pass:

1. The seed request completes and its KV is displaced from HBM by unique
   pressure prompts.
2. `POST /v1/kv_cache/prefetch` with `target_tier: "gpu"` reaches `ready`.
3. The `CPU_to_GPU` transfer-byte counter increases during preload. This rules
   out an HBM-only cache hit.
4. The real request reports at least all but one cache block as cached when
   `--enable-prompt-tokens-details` is enabled.
5. The deterministic output token agrees across the seed computation, a normal
   in-HBM cache-hit reference request, and the request after preload.
6. The selected token's log probability after preload matches the normal HBM
   cache-hit reference within `--logprob-atol` (default `1e-4`). This catches KV
   corruption that happens to leave cache bookkeeping intact. The comparison is
   intentionally cache-hit to cache-hit: on some backends, recomputing the full
   prompt and recomputing only the final token can produce different logprob
   rounding even when they select the same output token.
7. With `--require-tier-read`, the secondary-tier read-byte counter must also
   increase during preload. Use this for UMBP DRAM or SSD validation.

The output and cache-hit-reference log-probability comparison is the correctness
check. Transfer counters and cached-token counts alone only prove data movement
and reuse, not that the restored KV contents are correct.

## Prerequisites

Run commands from the repository root. Use the repository virtual environment,
not system Python.

```bash
cd /app/umbp/vllmumbp
.venv/bin/python -c \
  'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
rocminfo | grep -m1 -A3 'Agent'
```

The model must be available locally or downloadable from Hugging Face. MoRI
validation additionally requires an `amd-mori` build with `BUILD_UMBP=ON`.
The validator generates a unique cache salt by default, so repeated runs do not
reuse a previous run's target. Pass `--cache-salt` only when intentionally
reproducing an identical cache identity.

## DRAM-to-HBM validation

Terminal 1—start a deliberately small GPU cache so the pressure prompts force
eviction:

```bash
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m vllm.entrypoints.cli.main serve \
  Qwen/Qwen3-0.6B \
  --port 8000 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --max-model-len 224 \
  --num-gpu-blocks-override 16 \
  --gpu-memory-utilization 0.2 \
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

Terminal 2—run the validator:

```bash
.venv/bin/python benchmarks/kv_offload/validate_hbm_prefetch.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B \
  --prompt-tokens 128 \
  --block-size 16 \
  --eviction-prompts 4 \
  --require-amd \
  --output /tmp/hbm-prefetch-dram.json
```

A successful result has:

```json
{
  "prefetch_status": "ready",
  "cached_prompt_tokens": 128,
  "output_correct": true,
  "cpu_to_gpu_bytes": 14680064.0,
  "tier_read_bytes": 0.0,
  "amd_runtime_detected": true,
  "passed": true
}
```

Exact timings and byte counts depend on model layout and hardware. A zero
secondary-tier read is expected for `CPUOffloadingSpec`.

## UMBP DRAM/SSD-to-HBM validation

Create a dedicated SSD directory:

```bash
mkdir -p /tmp/umbp-hbm-validation-ssd
```

Start the server with a small CPU primary and small UMBP DRAM capacity so the
pressure workload can spill into SSD:

```bash
HIP_VISIBLE_DEVICES=0 .venv/bin/python -m vllm.entrypoints.cli.main serve \
  Qwen/Qwen3-0.6B \
  --port 8000 \
  --enforce-eager \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --max-model-len 224 \
  --num-gpu-blocks-override 16 \
  --gpu-memory-utilization 0.2 \
  --kv-transfer-config '{
    "kv_connector":"OffloadingConnector",
    "kv_role":"kv_both",
    "kv_connector_extra_config":{
      "spec_name":"TieringOffloadingSpec",
      "cpu_bytes_to_use":16777216,
      "offload_prompt_only":false,
      "secondary_tiers":[{
        "type":"mori",
        "dram_capacity_bytes":16777216,
        "ssd_enabled":true,
        "ssd_storage_dir":"/tmp/umbp-hbm-validation-ssd",
        "ssd_capacity_bytes":1073741824,
        "dram_high_watermark":0.5,
        "dram_low_watermark":0.25,
        "io_threads":2,
        "key_prefix":"vllm:hbm-validation:"
      }]
    }
  }'
```

Run with stronger eviction pressure and require a secondary-tier read:

```bash
.venv/bin/python benchmarks/kv_offload/validate_hbm_prefetch.py \
  --base-url http://127.0.0.1:8000 \
  --model Qwen/Qwen3-0.6B \
  --prompt-tokens 128 \
  --block-size 16 \
  --eviction-prompts 8 \
  --require-amd \
  --require-tier-read \
  --output /tmp/hbm-prefetch-umbp-ssd.json
```

For this run, `passed: true` requires both `tier_read_bytes > 0` and
`cpu_to_gpu_bytes > 0`, plus the output/token-logprob correctness checks.

## Interpreting failures

- `prefetch_status` is `miss`: the target was not retained off-device. Increase
  CPU/UMBP/SSD capacity or wait for pending stores to complete.
- `cpu_to_gpu_bytes` is zero: the target was probably still in HBM. Increase
  `--eviction-prompts` or reduce `--num-gpu-blocks-override`.
- `tier_read_bytes` is zero with `--require-tier-read`: the target stayed in the
  primary CPU tier or UMBP DRAM. Reduce those capacities or increase pressure.
- `cached_prompt_tokens` is low: preload did not publish the expected prefix or
  the block size does not match the server configuration.
- `output_correct` is false: treat this as a correctness failure even when all
  transfer counters are positive. Preserve the JSON result and server logs.
- Only the log probability differs slightly: first repeat with the same server
  and eager mode. If the difference is expected numerical noise, pass a
  justified larger `--logprob-atol`; do not disable the output-token check.

Stop the server with `Ctrl-C` after collecting the JSON result.
