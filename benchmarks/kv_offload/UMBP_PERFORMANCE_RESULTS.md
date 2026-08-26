# UMBP KV offloading performance evaluation

Date: 2026-08-23

## Conclusion

The current UMBP implementation was not faster than either comparison tier in
these single-request, forced-eviction tests:

- Against pinned CPU DRAM, UMBP DRAM increased mean TTFT by 54.8% for a
  128-token prefix and 26.4% for a 512-token prefix.
- Against vLLM's filesystem secondary tier, UMBP SSD increased mean TTFT from
  24.48 ms to 150.86 ms for a 128-token prefix: 6.16x the latency.
- The 512-token UMBP SSD attempts also exposed admission instability. Only one
  of five restores completed with a 64 MiB UMBP DRAM admission tier, and only
  two of five completed with 128 MiB. Those runs are invalid as latency
  comparisons, but are retained below as failure data.

These results measure local restoration latency. They do not measure UMBP's
potential capacity, aggregate-concurrency, or cross-replica reuse benefits.

## What was compared

The names of the tiers are important:

| Mode | Data path |
| --- | --- |
| Recompute | GPU prefill, with no offload connector |
| CPU DRAM | Pinned CPU primary tier to GPU |
| UMBP DRAM | UMBP DRAM to CPU primary tier to GPU |
| Existing filesystem tier | Filesystem storage to CPU primary tier to GPU |
| UMBP SSD | UMBP SSD through UMBP DRAM and CPU primary tier to GPU |

`CPUOffloadingSpec` is DRAM-only. The existing SSD-capable baseline is
`TieringOffloadingSpec` with a `type: "fs"` secondary tier.

## Environment

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3-0.6B` |
| Repository commit | `13bb64481f18822874463ace9e8f1b06f94a669b` |
| Installed vLLM metadata | `0.1.dev20314+ga3561ef8e` |
| PyTorch | `2.12.0+git6bbd260` |
| ROCm | `7.2.53211` |
| MoRI | `amd-mori 1.2.2` from `/app/umbp/moriv112` |
| GPU selected | GPU 0, AMD `gfx950`; product name unavailable from libdrm |
| Storage path | `/app/umbp` on `/dev/vda1`, ext4 |
| Storage device report | 128 GiB virtual disk, `ROTA=1` |

Eight unformatted 3.5 TB NVMe devices were visible but were not mounted or
modified. Consequently, the storage results compare the two software paths on
the same ext4 volume; they are not native-NVMe performance claims. The
filesystem tier emitted no buffered-I/O fallback warning, indicating its
`O_DIRECT` probe succeeded.

## Method

The benchmark uses exact token-ID prompts, one output token, streaming TTFT,
and three unique eviction prompts before every measured target request. Every
mode starts in a fresh server process. Metrics are sampled after initial cache
population and after the final trial.

Common server settings:

```text
--enforce-eager
--enable-prefix-caching
--gpu-memory-utilization 0.2
HIP_VISIBLE_DEVICES=0
```

For 128 tokens, the server used `--max-model-len 224` and 16 GPU blocks. The
CPU primary tier and UMBP DRAM admission tier were each 16 MiB for the SSD
comparison. For 512 tokens, the server used `--max-model-len 768` and 64 GPU
blocks; the CPU primary tier was 64 MiB.

The filesystem and UMBP managers both used two configured I/O threads. UMBP
SSD capacity was 1 GiB at 128 tokens and 2 GiB at 512 tokens. Temporary storage
directories were unique for every clean-server run.

The driver is `benchmark_umbp_vs_cpu.py`. Full launch examples and scaling
guidance are in this directory's `README.md`.

## Valid summary results

### Client-observed latency

| Prefix | Mode | Trials | Mean TTFT (ms) | p50 (ms) | p90 (ms) | Mean total (ms) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 128 | Recompute | 10 | 11.321 | 11.207 | 11.647 | 11.333 |
| 128 | CPU DRAM | 10 | 17.364 | 17.282 | 17.633 | 17.377 |
| 128 | UMBP DRAM | 10 | 26.885 | 26.845 | 27.443 | 26.898 |
| 128 | Filesystem | 10 | 24.483 | 24.096 | 25.068 | 24.498 |
| 128 | UMBP SSD | 10 | 150.865 | 150.825 | 155.810 | 150.877 |
| 512 | Recompute | 10 | 12.064 | 12.006 | 12.509 | 12.078 |
| 512 | CPU DRAM | 10 | 35.302 | 35.283 | 35.800 | 35.314 |
| 512 | UMBP DRAM | 10 | 44.610 | 44.554 | 44.705 | 44.628 |
| 512 | Filesystem | 5 | 63.776 | 64.114 | 65.278 | 63.791 |

### Transfer counters

Counters cover all requests between metric snapshots, including eviction
traffic, rather than only the timed target requests.

| Prefix | Mode | CPU-to-GPU bytes | CPU-to-GPU time (s) | CPU-to-GPU GiB/s | Tier-read bytes | Tier-read time (s) | Tier-read GiB/s | Tier-write bytes |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | CPU DRAM | 165,150,720 | 0.035913 | 4.283 | 0 | 0 | 0 | 0 |
| 128 | UMBP DRAM | 165,150,720 | 0.036097 | 4.261 | 205,520,896 | 0.061751 | 3.100 | 440,401,920 |
| 128 | Filesystem | 165,150,720 | 0.036200 | 4.249 | 196,345,856 | 0.015876 | 11.518 | 308,281,344 |
| 128 | UMBP SSD | 165,150,720 | 0.037641 | 4.086 | 1,394,606,080 | 1.387017 | 0.936 | 161,480,704 |
| 512 | CPU DRAM | 605,552,640 | 0.133009 | 4.240 | 0 | 0 | 0 | 0 |
| 512 | UMBP DRAM | 605,552,640 | 0.131195 | 4.299 | 1,321,205,760 | 0.135551 | 9.078 | 1,820,327,936 |
| 512 | Filesystem | 302,776,320 | 0.069237 | 4.073 | 1,027,604,480 | 0.086490 | 11.065 | 645,922,816 |

The nearly equal CPU-to-GPU bandwidth confirms that UMBP's DRAM penalty is
before the common GPU DMA stage. For SSD, the tier-read counter reports
11.52 GiB/s for the existing filesystem path and 0.94 GiB/s for UMBP. These
are effective software-path rates on this virtual storage stack, not physical
device bandwidth measurements.

## Raw per-trial TTFT data

### 128-token filesystem tier

```text
25.435, 24.934, 23.839, 24.583, 24.731,
24.096, 24.035, 24.048, 24.063, 25.068 ms
```

### 128-token UMBP SSD

```text
139.092, 148.221, 150.825, 149.550, 151.659,
150.627, 151.071, 154.891, 155.810, 156.903 ms
```

### 512-token filesystem tier

```text
64.809, 60.789, 65.278, 63.892, 64.114 ms
```

Raw per-trial values for the earlier recompute, CPU DRAM, and UMBP DRAM runs
were not persisted to a file; their complete aggregate output is recorded in
the tables above.

## Invalid 512-token UMBP SSD attempts

These attempts are diagnostic data, not benchmark results. The transfer
counter proves fewer UMBP-to-GPU restores occurred than measured trials.

| UMBP DRAM admission | TTFT trials (ms) | CPU-to-GPU bytes | Expected interpretation |
| ---: | --- | ---: | --- |
| 64 MiB | 602.919, 17.002, 16.640, 16.836, 17.586 | 64,225,280 | One restore; four recomputations or GPU hits |
| 128 MiB | 545.327, 17.057, 17.110, 468.690, 18.207 | 122,945,536 | Two restores; three recomputations or GPU hits |

The 64 MiB attempt reported 2,284,584,960 tier-read bytes over 0.651594 s and
117,440,512 tier-write bytes. The 128 MiB attempt reported 3,244,294,144
tier-read bytes over 1.015082 s and 234,881,024 tier-write bytes. The mismatch
between large tier-read counters and completed CPU-to-GPU restorations warrants
profiling UMBP admission, lookup, and SSD promotion before using long-prefix
results.

## Interpretation

For a local hit, both secondary tiers must stage through CPU before the same
GPU DMA. UMBP adds its own placement, lookup, admission, and promotion work.
At 128 tokens that work added 126.38 ms mean TTFT relative to the existing
filesystem tier.

The tiny Qwen model recomputed 128- and 512-token prefixes faster than either
storage path. A larger model or longer prefix is required to find the
restore-versus-recompute crossover. UMBP may still improve system throughput
when its distributed capacity or remote reuse converts misses into hits, but
that claim requires a concurrent two-replica experiment and cannot be inferred
from this local latency benchmark.

## Next evaluation matrix

1. Provision one of the visible NVMe devices in an approved, non-destructive
   test environment and repeat filesystem versus UMBP SSD with cold reads.
2. Add per-medium UMBP counters so DRAM hits and SSD hits are independently
   verified; require one completed CPU-to-GPU restore per trial.
3. Sweep 128, 512, 2,048, 8,192, and 16,384-token prefixes on a production-size
   model.
4. Sweep concurrency 1, 8, 32, and 128 and report request throughput, p50/p90/
   p99 TTFT, hit rate, recomputed tokens, CPU utilization, and device IOPS.
5. Run two replicas and two nodes to quantify cross-replica hits and remote
   reads against the filesystem tier's local-miss behavior.

## AMD GPU correctness checkpoint (2026-08-26)

The current container exposes eight gfx950 GPUs with 309,220,868,096 bytes of
VRAM each. The host has two NUMA nodes, 18,432 free 2 MiB hugepages, eight NVMe
devices, and both Ionic and mlx5 RDMA devices. These facts establish available
test capacity; they are not performance results.

The opt-in Qwen/Qwen3-0.6B UMBP integration tests passed on TP=1 and TP=2. Each
test generated a prefix, forced it out of the 16-block GPU cache with six other
prompts, restored the prefix through UMBP, checked exact generated-output
equality, and required tier-read plus CPU-to-GPU byte counters to increase.

```bash
RUN_UMBP_INTEGRATION_TEST=1 HIP_VISIBLE_DEVICES=0 \
  .venv/bin/python -m pytest \
  tests/v1/kv_offload/tiering/test_mori_gpu_integration.py::\
test_mori_restores_evicted_gpu_kv_from_umbp -v -s

RUN_UMBP_TP2_INTEGRATION_TEST=1 HIP_VISIBLE_DEVICES=0,1 \
  .venv/bin/python -m pytest \
  tests/v1/kv_offload/tiering/test_mori_gpu_integration.py::\
test_mori_restores_evicted_tp2_kv_from_umbp -v -s
```

The TP=2 run reported 77,070,336 UMBP write bytes and 11,010,048 UMBP read
bytes. The CPU-to-GPU load counter also reported 11,010,048 bytes. The UMBP
read timer accumulated 0.001362 seconds and the CPU-to-GPU load timer
accumulated 0.004674 seconds. This is a single correctness workload with JIT,
startup, and test orchestration effects. It must not be used as a steady-state
bandwidth or latency result.
