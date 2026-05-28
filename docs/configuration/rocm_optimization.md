# AMD ROCm Optimization

This guide covers performance tuning and optimization flags for AMD GPUs (MI200, MI300, MI350 series) running vLLM with ROCm. Most optimizations are provided through [AITER](https://github.com/ROCm/aiter) (AMD Innovative Tensor Engine for ROCm), a library of high-performance kernels.

!!! tip
    For installation instructions, see the [ROCm installation guide](../getting_started/installation/gpu.rocm.inc.md).
    For MI300x system-level tuning, refer to AMD's [MI300x tuning guide](https://rocm.docs.amd.com/en/latest/how-to/tuning-guides/mi300x/index.html).

## AITER Overview

AITER provides optimized kernels for common operations (GEMM, attention, MoE, normalization, etc.) on AMD GPUs. Most AITER optimizations are **disabled by default** and controlled via environment variables.

### Enabling AITER

Set `VLLM_ROCM_USE_AITER=1` to enable the AITER kernel library. Individual AITER features can then be toggled independently via their own environment variables.

```bash
# Enable AITER with default sub-feature settings
VLLM_ROCM_USE_AITER=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

```python
import os
os.environ["VLLM_ROCM_USE_AITER"] = "1"

from vllm import LLM
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct")
```

!!! note
    When `VLLM_ROCM_USE_AITER=0` (the default), all AITER sub-feature flags are ignored regardless of their individual settings.

## Environment Variables

### Master Switch

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ROCM_USE_AITER` | `0` | Enable the AITER kernel library. Must be set to `1` for any AITER sub-feature to take effect. |

### Attention

These flags control the attention kernel used on ROCm. For an overview of attention backend selection, see [Attention Backend Selection](./optimization.md#attention-backend-selection).

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ROCM_USE_AITER_PAGED_ATTN` | `0` | Use AITER paged attention kernels. |
| `VLLM_ROCM_USE_AITER_MLA` | `1` | Use AITER Multi-head Latent Attention (MLA) kernels for DeepSeek-style models. |
| `VLLM_ROCM_USE_AITER_MHA` | `1` | Use AITER Multi-head Attention (MHA) kernels. |
| `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION` | `0` | Use AITER Triton unified attention for V1 attention. |
| `VLLM_ROCM_USE_AITER_FP8BMM` | `1` | Use AITER Triton FP8 batched matrix multiply kernel. |
| `VLLM_ROCM_USE_AITER_FP4BMM` | `1` | Use AITER Triton FP4 batched matrix multiply kernel. |
| `VLLM_ROCM_FP8_MFMA_PAGE_ATTN` | `0` | Use FP8 MFMA (Matrix Fused Multiply-Add) in paged attention. |
| `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT` | `0` | Use shuffled KV cache layout for improved memory access patterns. |

### GEMM (General Matrix Multiply)

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ROCM_USE_AITER_LINEAR` | `1` | Use AITER tuned GEMMs for linear layers, including `scaled_mm` (per-tensor / rowwise) and unquantized GEMMs. |
| `VLLM_ROCM_USE_AITER_TRITON_GEMM` | `1` | Use AITER Triton kernels for GEMM operations. |
| `VLLM_ROCM_USE_AITER_FP4_ASM_GEMM` | `0` | Use AITER FP4 assembly GEMM kernels. |
| `VLLM_ROCM_USE_SKINNY_GEMM` | `1` | Use skinny (shape-optimized) GEMM kernels. |

### Mixture of Experts (MoE)

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ROCM_USE_AITER_MOE` | `1` | Use AITER fused MoE kernels. |
| `VLLM_ROCM_AITER_MOE_DISPATCH_POLICY` | `0` | MoE sorting dispatch policy for AITER fused MoE kernels. `0` = auto (single-pass for small batches, multi-pass for large), `1` = always single-pass (one kernel launch, no workspace, preferred for low-concurrency decode), `2` = always multi-pass (can be faster for MoE-heavy models). |
| `VLLM_ROCM_MOE_PADDING` | `1` | Pad MoE kernel weights for aligned access. |
| `VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS` | `0` | Fuse shared expert computation with the MoE combine step. |

### Normalization

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ROCM_USE_AITER_RMSNORM` | `1` | Use AITER RMSNorm kernels. |

### Positional Encoding

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ROCM_USE_AITER_TRITON_ROPE` | `0` | Use AITER Triton Rotary Position Embedding (RoPE) kernels. |

### Quantization

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ROCM_FP8_PADDING` | `1` | Pad FP8 weights to 256 bytes for aligned memory access. |

### Memory

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ROCM_SLEEP_MEM_CHUNK_SIZE` | `256` | Chunk size (in MB) for sleeping memory allocations. Controls how memory is divided when using sleep mode. |

### Communication

These flags tune the QuickReduce custom all-reduce kernel for MI300-series cards.

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION` | `NONE` | Quantization level for QuickReduce. Choices: `FP`, `INT8`, `INT6`, `INT4`, `NONE`. Recommended for large models to improve all-reduce performance. |
| `VLLM_ROCM_QUICK_REDUCE_CAST_BF16_TO_FP16` | `1` | Cast BF16 inputs to FP16 before QuickReduce. Due to the lack of BF16 ASM instructions, BF16 kernels are slower than FP16 on AMD GPUs. |
| `VLLM_ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB` | _(auto)_ | Maximum data size (in MB) for QuickReduce. Data exceeding this size falls back to RCCL. Default uses a built-in threshold table. |
| `VLLM_ROCM_QUICK_REDUCE_MIN_SIZE_BYTES_MB` | _(auto)_ | Minimum data size (in MB) required to use QuickReduce. Default uses a built-in threshold table. |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION_MIN_SIZE_KB` | _(auto)_ | Minimum tensor size (in KB) required to use the configured QuickReduce quantization codec. Smaller tensors use FP QuickReduce. Does not affect QuickReduce eligibility. |

## Recommended Configurations

### Default (Balanced)

The default settings provide a good balance of performance and compatibility. With `VLLM_ROCM_USE_AITER=1`, the following sub-features are enabled by default:

- AITER linear GEMMs, Triton GEMMs, skinny GEMMs
- AITER MHA, MLA attention kernels
- AITER fused MoE, RMSNorm
- FP8 and MoE weight padding

```bash
VLLM_ROCM_USE_AITER=1 vllm serve meta-llama/Llama-3.1-8B-Instruct
```

### Maximum Performance

Enable additional experimental optimizations for maximum throughput. Test thoroughly for your workload.

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_PAGED_ATTN=1 \
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
VLLM_ROCM_USE_AITER_TRITON_ROPE=1 \
VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1 \
vllm serve meta-llama/Llama-3.1-8B-Instruct
```

### DeepSeek-style MLA Models

For models using Multi-head Latent Attention (e.g., DeepSeek-V3), AITER MLA kernels are enabled by default. To also enable the unified attention path:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1 \
vllm serve deepseek-ai/DeepSeek-V3
```

### MoE-Heavy Models

For Mixture of Experts models, AITER MoE kernels are enabled by default. For MoE-heavy workloads, consider enabling fused shared experts and tuning the dispatch policy:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=1 \
VLLM_ROCM_AITER_MOE_DISPATCH_POLICY=2 \
vllm serve Qwen/Qwen3-MoE
```

### QuickReduce for Large Models

For large models where all-reduce is a bottleneck, enable QuickReduce with quantization:

```bash
VLLM_ROCM_USE_AITER=1 \
VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT8 \
vllm serve meta-llama/Llama-3.3-70B-Instruct --tensor-parallel-size 4
```

## Attention Backend Selection

On ROCm, vLLM selects from the following attention backends in priority order:

| Backend | Description |
| --- | --- |
| `ROCM_AITER_MLA` | AITER MLA for DeepSeek-style models (when `VLLM_ROCM_USE_AITER_MLA=1`) |
| `ROCM_AITER_FA` | AITER Flash Attention (when `VLLM_ROCM_USE_AITER_MHA=1`) |
| `ROCM_AITER_UNIFIED_ATTN` | AITER unified attention (when `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1`) |
| `ROCM_ATTN` | Default ROCm attention backend |

You can override backend selection explicitly:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct --attention-backend ROCM_AITER_FA
```

For the full feature support matrix of each backend, see [Attention Backend Feature Support](../design/attention_backends.md).
