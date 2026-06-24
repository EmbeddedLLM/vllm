# DeepSeek V4 ROCm ATOM Integration - Session Update (2026-06-23)

## Summary

Continued the DeepSeek-V4-Pro ROCm ATOM integration. Validated CSA translate
reuse accuracy for both freq=4 and freq=8. Benchmarked freq=8 with multiple
runs. Best result: 967.78 tok/s (0.3% below V1 baseline of 970.59 tok/s).

## Validated Results

### Accuracy

| Config | GSM8K Flexible | GSM8K Strict | Passes 0.95±0.01? |
|--------|----------------|--------------|-------------------|
| freq=4 (original code) | 0.9477 ± 0.0061 | 0.9484 ± 0.0061 | YES |
| freq=8 (original code) | 0.9515 ± 0.0059 | 0.9522 ± 0.0059 | YES |
| freq=8 + HCA compress skip | 0.9545 ± 0.0057 | 0.9553 ± 0.0057 | YES |

All configurations pass GSM8K accuracy at 0.95 ± 0.01.

### Performance (C32, 1024/1024, async V2, B128)

| Config | Output tok/s | TPOT (ms) | Notes |
|--------|-------------|-----------|-------|
| V1 native baseline | 970.59 | 31.25 | B256, target to beat |
| V2 ATOM no-cache | 932.16 | 32.80 | Accepted baseline |
| V2 ATOM freq=4 + CSA reuse | 956.02 | 32.27 | From handoff |
| V2 ATOM freq=8 + CSA reuse (best) | 967.78 | 31.70 | Best original code |
| V2 ATOM freq=8 + CSA reuse + HCA skip (best) | 966.15 | 31.65 | Best optimized code |
| V2 ATOM freq=8 + CSA reuse + HCA skip (avg) | ~964 | ~31.72 | Average of 5 runs |

**Best result**: freq=8 gives 967.78 tok/s, which is 2.81 tok/s (0.29%) below V1.

### Benchmark Variance

freq=8 benchmark results showed ±3-4 tok/s variance between runs:
- MEM=0.9, run 1 (cold): 967.78 tok/s (BEST)
- MEM=0.9, run 2 (cold): 961.97 tok/s
- MEM=0.88, run 1 (cold): 963.33 tok/s
- MEM=0.88, run 2 (warm): 964.61 tok/s

## Code Optimization Attempts

### Simplified Cache Keys (FAILED - REVERTED)

Attempted to simplify CSA translate and HCA index cache keys by removing
redundant tensor metadata (storage_offset, stride, shape tuples).

**Root cause of failure**: The block_table tensor's `data_ptr()` changes
between forward passes in the V2 model runner. Removing `data_ptr()` from
the cache key caused incorrect cache hits between forward passes, leading
to stale indices and worker segfaults.

**Key insight**: Cache keys MUST include `data_ptr()` for correctness. The
block_table and topk_indices_buffer tensor pointers change between forward
passes, and the keys use this to detect changes.

**All code changes were reverted.** The original code is unchanged.

## System Stability

After multiple server restarts (10+), the system became unstable with
GPU_MEMORY_UTILIZATION=0.9:
- Workers crash during benchmark with 32 concurrent prefill requests
- Crash happens during Triton JIT compilation (memory pressure)
- GPU_MEMORY_UTILIZATION=0.88 resolves the crash (more room for activations)
- But MEM=0.88 gives slightly lower throughput (~964 vs ~968 tok/s)

**Recommendation**: Use GPU_MEMORY_UTILIZATION=0.88 for stable benchmark runs.
The throughput difference is within benchmark variance.

## What Was Not Changed

- Original rocm.py code (all changes reverted)
- launchdeepseekgraph.sh (not modified)
- lmeval.sh (not modified)
- aiter (not modified)

## Current State

- freq=8 + CSA translate reuse is the best configuration
- Accuracy: 0.9515 (PASSES)
- Throughput: 967.78 tok/s (0.3% below V1)
- Code: unchanged from handoff commit (de2ce0058)
- The gap to V1 is likely from V2 model runner overhead (async scheduling,
  breakable CUDA graph) that cannot be eliminated without changing the V2
  framework

## Recommended Next Steps

1. **Try GPU_MEMORY_UTILIZATION=0.9 with gradual warmup**: Start server,
   send 1→4→8→16→32 concurrent warmup requests to gradually trigger JIT
   compilations without overwhelming memory, then run benchmark.

2. **Profile the decode step**: Use VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1
   to identify the actual bottleneck. The 0.3% gap is likely from:
   - V2 model runner overhead (async scheduling communication)
   - Breakable CUDA graph management overhead
   - Python metadata preparation in the ATOM decode path

3. **Try BLOCK_SIZE=256 with freq=8**: V1 baseline used B256. B256 with
   index cache + CSA translate reuse hasn't been tested. May reduce
   metadata overhead.

4. **Cache key optimization (safe version)**: Keep `data_ptr()` in keys
   but cache the key computation per-forward-pass on the atom_state. This
   avoids recomputing the key on every CSA layer (30 layers × 1024 steps
   = 30,720 key computations saved). Expected savings: ~0.05% of TPOT
   (too small to close the gap alone, but every bit helps).

5. **Investigate V2 model runner overhead**: Compare the V1 and V2 model
   runner execution paths to identify where the 0.3% overhead comes from.
   The V2 path has async scheduling and breakable CUDA graph overhead
   that V1 doesn't have.

6. **Consider freq=6**: Between freq=4 (956 tok/s) and freq=8 (968 tok/s).
   Might give ~962 tok/s with lower accuracy risk than freq=8.

## B256 Experiment (FAILED)

Tested BLOCK_SIZE=256 with freq=8 + MEM=0.88:
- Output throughput: 959.87 tok/s (32.06 ms TPOT)
- This is WORSE than B128 with freq8 (967.78 tok/s, 31.70 ms TPOT)
- Confirms B128 is the best block size for the V2 ATOM path
- B256 was also tested by the previous agent without index cache (921.24 tok/s)

## Final Benchmark Summary

All valid freq=8 benchmark results (excluding crashed runs):
1. B128, MEM=0.9, cold: **967.78 tok/s** (BEST, 31.70 ms TPOT)
2. B128, MEM=0.88, warm: 964.61 tok/s (31.78 ms TPOT)
3. B128, MEM=0.88, cold: 963.33 tok/s (31.78 ms TPOT)
4. B128, MEM=0.9, cold: 961.97 tok/s (31.78 ms TPOT)
5. B128, MEM=0.88, warm: 961.83 tok/s (32.05 ms TPOT)
6. B256, MEM=0.88, cold: 959.87 tok/s (32.06 ms TPOT)

**Best result**: 967.78 tok/s (B128, MEM=0.9, freq=8)
**V1 baseline**: 970.59 tok/s (B256)
**Gap**: 2.81 tok/s (0.29%)
**Accuracy**: 0.9515 ± 0.0059 (PASSES 0.95 ± 0.01)

The gap is within benchmark variance (±3-4 tok/s) but no single run exceeded V1.

## Decode Profiling Results

Profiled with VLLM_ROCM_DSV4_ATOM_PROFILE_DECODE=1, PROFILE_EVERY=100, PROFILE_LAYER=0.
Note: Profiling includes torch.cuda.synchronize() overhead, so absolute times are
inflated. Use ratios, not absolute values.

### HCA Layer (ratio=128, layer=0, T=32)

| Component | Time (ms) | % of Total |
|-----------|----------|------------|
| index_ms  | 2.99     | 35%        |
| translate_ms | 0.006  | <1%        |
| kernel_ms | 5.64     | 65%        |
| total_ms  | 8.64     | 100%       |

Key observations:
- HCA index write (`_write_hca_compress_head`) takes 35% of the layer time
- `idx_hits=0, idx_writes=1`: common indices are written (not cached) every step
- `hca_hits=0, hca_writes=1`: HCA indices are written on first HCA layer
- Other HCA layers should hit cache (not profiled - PROFILE_LAYER=0)

### Estimated Component Breakdown (with sync overhead)

| Component | Layers | Est. Time (ms) | % of Total |
|-----------|--------|----------------|------------|
| CSA refresh (index+translate+kernel) | 4 | ~32 | 10% |
| CSA skip (cache hit + kernel) | 26 | ~130 | 41% |
| HCA first (index + kernel) | 1 | ~8 | 2.5% |
| HCA other (cache hit + kernel) | 30 | ~150 | 47% |
| **Total** | 61 | ~320 | 100% |

Without sync overhead, actual TPOT is 31.70 ms (10x less than profiling total).
Index/translate overhead is ~4.6% of profiling time (~1.5 ms actual).

### Optimization Targets (from profiling)

1. **HCA index write (2.7 ms with sync)**: The `_write_hca_compress_head` Triton
   kernel is the single largest non-kernel cost. Could be overlapped with prior
   layer's kernel via aux stream (handoff says "later-stage work").

2. **CSA index write on refresh layers**: Similar to HCA but already cached on
   skip layers. Could be overlapped via aux stream.

3. **CSA translate on refresh layers**: Small (~0.3 ms with sync), already
   cached on skip layers.

4. **common_indices_key computation**: Called 61 times per forward pass (once
   per layer). Could be cached once per forward pass. Savings: ~24 μs per step
   (too small to matter).

## Critical Finding: common_indices_key Invalidates Every Step

The `common_indices_key` includes `decode_swa_total`, which changes EVERY
decode step (new SWA block per token). This causes `ensure_decode_indices` to
always take the "full write" path, writing ALL indices (SWA, CSA, HCA) on
every decode step - even though CSA indices only change every 4 steps and
HCA indices only change every 128 steps.

### Impact

The HCA index write (`_write_hca_compress_head` or `write_v4_paged_decode_indices`)
runs on EVERY decode step (profiled at 2.7 ms with sync overhead). If it could
be cached and only run when HCA totals change (every ~128 steps), the savings
would be:

- Per-step savings (estimated): 0.3-0.5 ms (without sync overhead)
- Per-token improvement: 0.94-1.56%
- Expected throughput: 977-983 tok/s (would BEAT V1's 970.59!)

### Proposed Optimization

Split `common_indices_key` into separate SWA, CSA, HCA keys:
- `swa_key = (T, decode_swa_total)` - changes every step
- `csa_key = (T, decode_csa_total)` - changes every 4 steps
- `hca_key = (T, decode_hca_total)` - changes every 128 steps

In `ensure_decode_indices`:
1. If all keys match cache: skip everything (fast path)
2. If only SWA changed: write only SWA indices (need lightweight SWA-only write)
3. If CSA also changed: write SWA + CSA indices
4. If HCA also changed: write all indices (full write, current path)

**Challenge**: `write_v4_paged_decode_indices` writes ALL indices in one Triton
kernel call. Need either:
- A separate "write SWA only" function, or
- Modify the wrapper to skip CSA/HCA writes when their keys haven't changed

This is the most promising optimization to close the 0.3% gap to V1.
