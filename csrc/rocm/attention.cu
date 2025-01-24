/*
 * Copyright (c) 2024, The vLLM team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <hip/hip_bf16.h>
#include "cuda_compat.h"

#include <algorithm>
#include "../attention/dtype_fp8.cuh"
#include "../quantization/fp8/amd/quant_utils.cuh"

#if defined(__HIPCC__) && (defined(__gfx90a__) || defined(__gfx940__) || \
                           defined(__gfx941__) || defined(__gfx942__))
  #define __HIP__MI300_MI250__
#endif

#if defined(NDEBUG)
  #undef NDEBUG
  #include <assert.h>
  #define UNREACHABLE_CODE assert(false);
  #define NDEBUG
#else
  #define UNREACHABLE_CODE assert(false);
#endif

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define DIVIDE_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

#if defined(__HIP__MI300_MI250__)  // TODO: Add NAVI support

  #define GCN_MFMA_INSTR1 __builtin_amdgcn_mfma_f32_16x16x4f32
  #define GCN_MFMA_INSTR __builtin_amdgcn_mfma_f32_4x4x4f16

using floatx4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using float16x4 =
    __attribute__((__vector_size__(4 * sizeof(_Float16)))) _Float16;
typedef float16x4 _Half4;
using float16x2 =
    __attribute__((__vector_size__(2 * sizeof(_Float16)))) _Float16;
typedef float16x2 _Half2;
typedef struct _Half8 {
  _Half4 xy[2];
} _Half8;

using bit16_t = uint16_t;
using bit16x4 = __attribute__((__vector_size__(4 * sizeof(uint16_t)))) uint16_t;
typedef bit16x4 _B16x4;
typedef struct _B16x8 {
  _B16x4 xy[2];
} _B16x8;

using _B8x8 = uint2;
using _B8x4 = int32_t;  // used in builtins
using bit8_t = uint8_t;

typedef struct _B8x16 {
  _B8x8 xy[2];
} _B8x16;

////// Non temporal loads ///////
template <typename T>
__device__ __forceinline__ T loadnt(T* addr) {
  return __builtin_nontemporal_load(addr);
}

__device__ __forceinline__ _B16x8 load_ntmprl_16Byte(const _B16x8* addr) {
  auto addr_alias = reinterpret_cast<const float*>(addr);
  auto dat0 = loadnt(addr_alias);
  auto dat1 = loadnt(addr_alias + 1);
  auto dat2 = loadnt(addr_alias + 2);
  auto dat3 = loadnt(addr_alias + 3);
  auto res = make_float4(dat0, dat1, dat2, dat3);
  return *reinterpret_cast<_B16x8*>(&res);
}
///////////////////////////////////

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma4x4x4_instr(const _B16x4& inpA,
                                                       const _B16x4& inpB,
                                                       const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4f16(inpA, inpB, inpC, absz, cbid,
                                              blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_4x4x4bf16_1k(inpA, inpB, inpC, absz, cbid,
                                                  blgp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T, int absz, int cbid, int blgp>
__device__ __forceinline__ floatx4 gcn_mfma16x16x16_instr(const _B16x4& inpA,
                                                          const _B16x4& inpB,
                                                          const floatx4& inpC) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(inpA, inpB, inpC, absz, cbid,
                                                 blgp);
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(inpA, inpB, inpC, absz,
                                                     cbid, blgp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float(const T& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ float to_float_b16(const bit16_t& inp) {
  union tmpcvt {
    bit16_t u;
    _Float16 f;
    __hip_bfloat16 b;
  } t16;
  t16.u = inp;
  if constexpr (std::is_same<T, _Float16>::value) {
    return (float)t16.f;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __bfloat162float(t16.b);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ T from_float(const float& inp) {
  if constexpr (std::is_same<T, _Float16>::value) {
    return (_Float16)inp;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    return __float2bfloat16(inp);
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4(const floatx4& inp) {
  union tmpcvt {
    uint16_t u;
    _Float16 f;
    __hip_bfloat16 b;
  } t16;
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      __half2 h2[2];
      _B16x4 b16x4;
    } u;
    u.h2[0] = __float22half2_rn(make_float2(inp[0], inp[1]));
    u.h2[1] = __float22half2_rn(make_float2(inp[2], inp[3]));
    return u.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
        uint32_t u32;
        float f32;
      } u;
      u.f32 = inp[i];
      u.u32 += 0x7fff + ((u.u32 >> 16) & 1);  // BF16 RNE with no nan/inf check
      ret[i] = uint16_t(u.u32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x4 addx4(const _B16x4& inp1,
                                        const _B16x4& inp2) {
  union tmpcvt {
    uint16_t u;
    _Float16 f;
    __hip_bfloat16 b;
  } t1, t2, res;
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      _B16x4 b16x4;
      __half2 h2[2];
    } u1, u2, s;
    u1.b16x4 = inp1;
    u2.b16x4 = inp2;
    s.h2[0] = u1.h2[0] + u2.h2[0];
    s.h2[1] = u1.h2[1] + u2.h2[1];
    return s.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
        float f32;
        uint32_t i32;
      } u1, u2, s;
      u1.i32 = uint32_t(inp1[i]) << 16;
      u2.i32 = uint32_t(inp2[i]) << 16;
      s.f32 = u1.f32 + u2.f32;
      ret[i] = uint16_t(s.i32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T, vllm::Fp8KVCacheDataType KV_DTYPE>
__device__ __forceinline__ _B16x8 scaled_convert_b8x8(const _B8x8 input,
                                                      const float scale) {
  union alignas(16) {
    uint4 u4;
    _B16x8 u16x8;
    vllm::bf16_8_t b16x8;
  } tmp;
  if constexpr (std::is_same<T, _Float16>::value) {
    tmp.u4 = vllm::fp8::scaled_convert<uint4, _B8x8, KV_DTYPE>(input, scale);
    return tmp.u16x8;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    tmp.b16x8 = vllm::fp8::scaled_convert<vllm::bf16_8_t, _B8x8, KV_DTYPE>(
        input, scale);
    return tmp.u16x8;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x8
scaled_convert_b8x8_custom(const _B8x8 input, const float scale) {
  union {
    floatx4 f32x4[2];
    vllm::Float8_ f32x8;
  } tmpf8;
  tmpf8.f32x8 = vllm::fp8::vec_conversion<vllm::Float8_, uint2>(
      *reinterpret_cast<const uint2*>(&input));

  tmpf8.f32x4[0] *= scale;
  tmpf8.f32x4[1] *= scale;

  _B16x8 ret;
  ret.xy[0] = from_floatx4<T>(tmpf8.f32x4[0]);
  ret.xy[1] = from_floatx4<T>(tmpf8.f32x4[1]);
  return ret;
}

__device__ __forceinline__ floatx4 to_float_fp8x4(const _B8x4& inp) {
  #if defined(__gfx90a__)
  float4 f32x4 = vllm::fp8::vec_conversion<float4, uint32_t>(
      *reinterpret_cast<const uint32_t*>(&inp));
  return *reinterpret_cast<floatx4*>(&f32x4);
  #else  // MI3xx+ optimized builtins
  const auto f0 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, false);
  const auto f1 = __builtin_amdgcn_cvt_pk_f32_fp8(inp, true);
  floatx4 ret;
  ret[0] = f0[0];
  ret[1] = f0[1];
  ret[2] = f1[0];
  ret[3] = f1[1];
  return ret;
  #endif
}

template <typename T>
__device__ __forceinline__ _B16x4 from_floatx4_rtz(const floatx4& inp) {
  _B16x4 ret;
  if constexpr (std::is_same<T, _Float16>::value) {
    union h2cvt {
      _Half2 h2[2];
      _B16x4 b16x4;
    } u;
    u.h2[0] = __builtin_amdgcn_cvt_pkrtz(inp[0], inp[1]);
    u.h2[1] = __builtin_amdgcn_cvt_pkrtz(inp[2], inp[3]);
    return u.b16x4;
  } else if constexpr (std::is_same<T, __hip_bfloat16>::value) {
    for (int i = 0; i < 4; i++) {
      union fcvt {
        uint32_t i32;
        float f32;
      } u;
      u.f32 = inp[i];
      ret[i] = uint16_t(u.i32 >> 16);
    }
    return ret;
  } else {
    static_assert(false, "unsupported 16b dtype");
  }
}

template <typename T>
__device__ __forceinline__ _B16x8 convert_b8x8_custom(const _B8x8 input) {
  union {
    _B8x8 b8x8;
    _B8x4 b8x4[2];
  } tmp;
  tmp.b8x8 = input;
  _B16x8 ret;
  for (int i = 0; i < 2; i++) {
    ret.xy[i] = from_floatx4_rtz<T>(to_float_fp8x4(tmp.b8x4[i]));
  }
  return ret;
}

///////////////////////////////////////
// grid (num_seqs, num_partitions,num_kv_heads)
// block (256)
template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED, int GQA_RATIO>
__global__
__launch_bounds__(NUM_THREADS, 5) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    scalar_t* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale_ptr, const float* v_scale_ptr) {
  constexpr int NWARPS = NUM_THREADS / WARP_SIZE;
  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;
  const int lane4id = laneid % 4;

  const int seq_idx = blockIdx.x;
  const int partition_idx = blockIdx.y;
  const int partition_size = blockDim.x;
  const int max_num_partitions = gridDim.y;

  const int context_len = context_lens[seq_idx];
  const int partition_start_token_idx = partition_idx * partition_size;
  // exit if partition is out of context for seq
  if (partition_start_token_idx >= context_len) {
    return;
  }
  constexpr int QHLOOP =
      DIVIDE_ROUND_UP(GQA_RATIO, 4);  // each 4 lanes fetch 4 different qheads
  constexpr int GQA_RATIO4 = 4 * QHLOOP;
  __shared__ float shared_qk_max[NWARPS][GQA_RATIO4 + 1];
  __shared__ float shared_exp_sum[NWARPS][GQA_RATIO4 + 1];
  _B16x8 Qlocal[QHLOOP];
  constexpr int x = 16 / sizeof(scalar_t);
  constexpr int KHELOOP = HEAD_SIZE / x;
  _B16x8 Klocal[KHELOOP];
  _B8x8 Klocalb8[KHELOOP];
  // v head_size dimension is distributed across warp
  constexpr int VHELOOP = HEAD_SIZE / WARP_SIZE;
  constexpr int VTLOOP = 8;  // 16 separate 4xtokens across warp -> 16/2
                             // 8xtokens
  constexpr int VBLOCKS = 8 * VTLOOP / BLOCK_SIZE;
  int vphysical_blocks[VBLOCKS];
  _B16x8 Vlocal[VHELOOP][VTLOOP];
  _B8x8 Vlocalb8[VHELOOP][VTLOOP];
  floatx4 dout[QHLOOP];
  float qk_max[QHLOOP];

  __shared__ _B16x4 vout_shared[QHLOOP][VHELOOP][WARP_SIZE][NWARPS + 1];

  for (int h = 0; h < QHLOOP; h++) {
    dout[h] = {0};
    qk_max[h] = -FLT_MAX;
  }

  const int wg_start_head_idx = blockIdx.z * GQA_RATIO;
  const int wg_start_kv_head_idx = blockIdx.z;

  const int warp_start_token_idx =
      partition_start_token_idx + warpid * WARP_SIZE;

  if (warp_start_token_idx >= context_len) {  // warp out of context
  #pragma unroll
    for (int h = 0; h < GQA_RATIO4; h++) {
      shared_qk_max[warpid][h] = -FLT_MAX;
      shared_exp_sum[warpid][h] = 0.0f;
    }
  } else {  // warp within context

    const int num_context_blocks = DIVIDE_ROUND_UP(context_len, BLOCK_SIZE);
    const int last_ctx_block = num_context_blocks - 1;

    const int* block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    const int local_token_idx = threadIdx.x;
    const int global_token_idx = partition_start_token_idx + local_token_idx;

    // fetch block number for k
    const int block_idx = (global_token_idx < context_len)
                              ? global_token_idx / BLOCK_SIZE
                              : last_ctx_block;
    // int32 physical_block_number leads to overflow when multiplied with
    // kv_block_stride
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // fetch vphysical block numbers up front
    const int warp_start_block_idx = warp_start_token_idx / BLOCK_SIZE;
    for (int b = 0; b < VBLOCKS; b++) {
      const int vblock_idx = warp_start_block_idx + b;
      const int vblock_idx_ctx =
          (vblock_idx <= last_ctx_block) ? vblock_idx : last_ctx_block;
      vphysical_blocks[b] = block_table[vblock_idx_ctx];
    }

    // each 4 lanes fetch 8 helems, so warp fetches 8*16 = 128 helems
    const scalar_t* q_ptr =
        q + seq_idx * q_stride + wg_start_head_idx * HEAD_SIZE;
    const _B16x8* q_ptrh8 = reinterpret_cast<const _B16x8*>(q_ptr);
    const int qhead_elemh8 = laneid / 4;

    for (int h = 0; h < QHLOOP - 1; h++) {
      const int qhead_idx = h * 4 + lane4id;
      Qlocal[h] = q_ptrh8[qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    }
    const int final_qhead_idx = 4 * (QHLOOP - 1) + lane4id;
    if (final_qhead_idx < GQA_RATIO) {
      Qlocal[QHLOOP - 1] =
          q_ptrh8[final_qhead_idx * HEAD_SIZE / 8 + qhead_elemh8];
    } else {
      Qlocal[QHLOOP - 1].xy[0] = {0};
      Qlocal[QHLOOP - 1].xy[1] = {0};
    }

    const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride +
                           wg_start_kv_head_idx * kv_head_stride;

    const int physical_block_offset =
        local_token_idx % BLOCK_SIZE;  // since x=half8, physical_block_offset
                                       // is already cast as _H8
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const _B16x8* k_ptrh8 = reinterpret_cast<const _B16x8*>(k_ptr);
      for (int d = 0; d < KHELOOP; d++) {
        Klocal[d] = k_ptrh8[d * BLOCK_SIZE + physical_block_offset];
      }
    } else {
      // vllm defines X as 16 Bytes of elements of cache_t
      constexpr int X = 16 / sizeof(cache_t);
      const cache_t* k_ptr2 = k_ptr + physical_block_offset * X;
      for (int d = 0; d < KHELOOP; d++) {
        const int head_elem = d * 8;
        const int offset1 = head_elem / X;
        const int offset2 = head_elem % X;
        const cache_t* k_ptr3 = k_ptr2 + offset1 * BLOCK_SIZE * X + offset2;
        Klocalb8[d] = *reinterpret_cast<const _B8x8*>(k_ptr3);
      }
    }

    float alibi_slope[QHLOOP];
    if constexpr (ALIBI_ENABLED) {
      for (int h = 0; h < QHLOOP; h++) {
        const int qhead_idx = h * 4 + lane4id;
        alibi_slope[h] = (qhead_idx < GQA_RATIO)
                             ? alibi_slopes[wg_start_head_idx + qhead_idx]
                             : 0.f;
      }
    }

    const cache_t* v_ptr = v_cache + wg_start_kv_head_idx * kv_head_stride;
    // fetch vcache in kv cache auto case
    if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto) {
      const _B16x8* v_ptrh8 = reinterpret_cast<const _B16x8*>(v_ptr);
      // iterate over each v block
      for (int b = 0; b < VBLOCKS; b++) {
        // int32 physical_block_number leads to overflow when multiplied with
        // kv_block_stride
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const _B16x8* v_ptrh8b =
            v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
        // iterate over each head elem (within head_size)
        for (int h = 0; h < VHELOOP; h++) {
          const int head_size_elem = h * WARP_SIZE + laneid;
          const _B16x8* v_ptrh8be = v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
          // iterate over all velems within block
          for (int d = 0; d < BLOCK_SIZE / 8; d++) {
            Vlocal[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
          }
        }
      }
    }  // if constexpr (KV_DTYPE == vllm::Fp8KVCacheDataType::kAuto)
    // fetch vcache in fp8 case
    else {  // if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto)
      const _B8x8* v_ptrh8 = reinterpret_cast<const _B8x8*>(v_ptr);
      // iterate over each v block
      for (int b = 0; b < VBLOCKS; b++) {
        // int32 physical_block_number leads to overflow when multiplied with
        // kv_block_stride
        const int64_t vphysical_block_number =
            static_cast<int64_t>(vphysical_blocks[b]);
        const _B8x8* v_ptrh8b =
            v_ptrh8 + (vphysical_block_number * kv_block_stride) / 8;
        // iterate over each head elem (within head_size)
        for (int h = 0; h < VHELOOP; h++) {
          const int head_size_elem = h * WARP_SIZE + laneid;
          const _B8x8* v_ptrh8be = v_ptrh8b + head_size_elem * BLOCK_SIZE / 8;
          // iterate over all velems within block
          for (int d = 0; d < BLOCK_SIZE / 8; d++) {
            // Vlocalb8[h][b * BLOCK_SIZE / 8 + d] = v_ptrh8be[d];
            const _B8x8 Vlocalb8 = v_ptrh8be[d];
            Vlocal[h][b * BLOCK_SIZE / 8 + d] =
                scaled_convert_b8x8<scalar_t, KV_DTYPE>(Vlocalb8, *v_scale_ptr);
          }
        }
      }
    }

  #define QK_mfma(x)                                             \
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) { \
      Klocal[x] = convert_b8x8_custom<scalar_t>(Klocalb8[x]);    \
    }                                                            \
    for (int h = 0; h < QHLOOP; h++) {                           \
      dout[h] = gcn_mfma4x4x4_instr<scalar_t, 4, x, 0>(          \
          Qlocal[h].xy[0], Klocal[x].xy[0], dout[h]);            \
      dout[h] = gcn_mfma4x4x4_instr<scalar_t, 4, x, 0>(          \
          Qlocal[h].xy[1], Klocal[x].xy[1], dout[h]);            \
    }
    // QK mfma
    QK_mfma(0);
    QK_mfma(1);
    QK_mfma(2);
    QK_mfma(3);
    QK_mfma(4);
    QK_mfma(5);
    QK_mfma(6);
    QK_mfma(7);
    // below only needed for head size 128
    if constexpr (KHELOOP > 8) {
      QK_mfma(8);
      QK_mfma(9);
      QK_mfma(10);
      QK_mfma(11);
      QK_mfma(12);
      QK_mfma(13);
      QK_mfma(14);
      QK_mfma(15);
    }
  #undef QK_mfma

    float scale2 = scale;
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
  #pragma unroll
      for (int d = 0; d < KHELOOP; d++) {
        Klocal[d] =
            scaled_convert_b8x8<scalar_t, KV_DTYPE>(Klocalb8[d], *k_scale_ptr);
      }
    }

    for (int h = 0; h < QHLOOP; h++) {
      dout[h] *= scale2;
    }

    // transpose dout so that 4 token ids are in each lane, and 4 heads are
    // across 4 lanes
    for (int h = 0; h < QHLOOP; h++) {
      floatx4 tmp = {0};
      for (int i = 0; i < 4; i++) {
        const float B = (lane4id == i) ? 1.0f : 0.0f;
        tmp = __builtin_amdgcn_mfma_f32_4x4x1f32(dout[h][i], B, tmp, 0, 0, 0);
      }
      dout[h] = tmp;
    }

    const int lane4_token_idx = 4 * (global_token_idx >> 2);

    if constexpr (ALIBI_ENABLED) {
      const int alibi_offset = lane4_token_idx - context_len + 1;
      for (int h = 0; h < QHLOOP; h++) {
        for (int i = 0; i < 4; i++) {
          dout[h][i] += alibi_slope[h] * (alibi_offset + i);
        }
      }
    }

    const int bpermute_mask = 4 * (16 * ((laneid >> 2) % 4) + lane4id);

    for (int h = 0; h < QHLOOP; h++) {
      qk_max[h] = -FLT_MAX;
      for (int i = 0; i < 4; i++) {
        qk_max[h] = (lane4_token_idx + i < context_len)
                        ? fmaxf(qk_max[h], dout[h][i])
                        : qk_max[h];
      }

      // for (int mask = WARP_SIZE / 2; mask >= 4; mask /= 2) {
      //   qk_max[h] = fmaxf(qk_max[h], __shfl_xor(qk_max[h], mask));
      // }
      // faster version of above code with dpp
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));

      auto tmp = __builtin_amdgcn_ds_bpermute(
          bpermute_mask, *reinterpret_cast<int*>(&qk_max[h]));
      qk_max[h] = *reinterpret_cast<float*>(&tmp);
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));
      asm("v_nop\n v_nop\n v_max_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(qk_max[h])
          : "v"(qk_max[h]), "v"(qk_max[h]));
    }

    float exp_sum[QHLOOP];
    for (int h = 0; h < QHLOOP; h++) {
      exp_sum[h] = 0.0f;
      for (int i = 0; i < 4; i++) {
        dout[h][i] = (lane4_token_idx + i < context_len)
                         ? __expf(dout[h][i] - qk_max[h])
                         : 0.0f;
        exp_sum[h] += dout[h][i];
      }
      // for (int mask = WARP_SIZE / 2; mask >= 4; mask /= 2) {
      //   exp_sum[h] += __shfl_xor(exp_sum[h], mask);
      // }
      // faster version of above code with dpp
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));

      auto tmp = __builtin_amdgcn_ds_bpermute(
          bpermute_mask, *reinterpret_cast<int*>(&exp_sum[h]));
      exp_sum[h] = *reinterpret_cast<float*>(&tmp);
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:4"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));
      asm("v_nop\n v_nop\n v_add_f32_dpp %0, %1, %2 row_ror:8"
          : "=v"(exp_sum[h])
          : "v"(exp_sum[h]), "v"(exp_sum[h]));
    }

    if (laneid < 4) {
      for (int h = 0; h < QHLOOP; h++) {
        const int head_idx = 4 * h + lane4id;
        shared_qk_max[warpid][head_idx] = qk_max[h];
        shared_exp_sum[warpid][head_idx] = exp_sum[h];
      }
    }
  }  // warp within context

  __syncthreads();

  const int num_heads = gridDim.z * GQA_RATIO;
  float* max_logits_ptr =
      max_logits + seq_idx * num_heads * max_num_partitions + partition_idx;
  float* exp_sums_ptr =
      exp_sums + seq_idx * num_heads * max_num_partitions + partition_idx;
  for (int h = 0; h < QHLOOP; h++) {
    float global_qk_max = -FLT_MAX;
    float warp_qk_max[NWARPS];
    const int head_idx = 4 * h + lane4id;
    for (int w = 0; w < NWARPS; w++) {
      warp_qk_max[w] = shared_qk_max[w][head_idx];
      global_qk_max = fmaxf(global_qk_max, warp_qk_max[w]);
    }
    float global_exp_sum = 0.0f;
    for (int w = 0; w < NWARPS; w++) {
      global_exp_sum +=
          shared_exp_sum[w][head_idx] * __expf(warp_qk_max[w] - global_qk_max);
    }
    if (head_idx < GQA_RATIO) {
      max_logits_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_qk_max;
      exp_sums_ptr[(wg_start_head_idx + head_idx) * max_num_partitions] =
          global_exp_sum;
    }
    const float global_inv_sum_scale = __fdividef(1.f, global_exp_sum + 1e-6f) *
                                       __expf(qk_max[h] - global_qk_max);
    dout[h] *= global_inv_sum_scale;
  }
  // logits[h] -> every 4 lanes hold 4 heads, each lane holds 4 tokens, there
  // are 4x16 tokens across warp
  _B16x4 logits[QHLOOP];
  for (int h = 0; h < QHLOOP; h++) {
    logits[h] = from_floatx4_rtz<scalar_t>(dout[h]);
  }

  if (warp_start_token_idx >= context_len) {  // warp out of context
    for (int qh = 0; qh < QHLOOP; qh++) {
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout_shared[qh][vh][laneid][warpid] = {0};
      }
    }
  } else {  // warp in context
  #define SV_mfma(x)                                                  \
    if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {      \
      Vlocal[vh][x] = convert_b8x8_custom<scalar_t>(Vlocalb8[vh][x]); \
    }                                                                 \
    for (int qh = 0; qh < QHLOOP; qh++) {                             \
      acc[qh] = gcn_mfma4x4x4_instr<scalar_t, 4, 2 * x, 0>(           \
          logits[qh], Vlocal[vh][x].xy[0], acc[qh]);                  \
      acc[qh] = gcn_mfma4x4x4_instr<scalar_t, 4, 2 * x + 1, 0>(       \
          logits[qh], Vlocal[vh][x].xy[1], acc[qh]);                  \
    }

    for (int vh = 0; vh < VHELOOP; vh++) {
      floatx4 acc[QHLOOP];
      for (int qh = 0; qh < QHLOOP; qh++) {
        acc[qh] = {0};
      }
      // iterate over tokens
      SV_mfma(0);
      SV_mfma(1);
      SV_mfma(2);
      SV_mfma(3);
      SV_mfma(4);
      SV_mfma(5);
      SV_mfma(6);
      SV_mfma(7);

      for (int qh = 0; qh < QHLOOP; qh++) {
        if constexpr (KV_DTYPE != vllm::Fp8KVCacheDataType::kAuto) {
          acc[qh] *= v_scale;
        }
        vout_shared[qh][vh][laneid][warpid] = from_floatx4<scalar_t>(acc[qh]);
      }
    }

  #undef SV_mfma
  }  // warp in context

  __syncthreads();

  if (warpid == 0) {
    _B16x4 vout[QHLOOP][VHELOOP];
    // iterate across heads
    for (int qh = 0; qh < QHLOOP; qh++) {
      // iterate over each v head elem (within head_size)
      for (int vh = 0; vh < VHELOOP; vh++) {
        vout[qh][vh] = {0};
        for (int w = 0; w < NWARPS; w++) {
          vout[qh][vh] =
              addx4<scalar_t>(vout[qh][vh], vout_shared[qh][vh][laneid][w]);
        }
      }
    }

    scalar_t* out_ptr = out +
                        seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
                        partition_idx * HEAD_SIZE;
    const int out_num_partitions = max_num_partitions;
    bit16_t* out_ptr_b16 = reinterpret_cast<bit16_t*>(out_ptr);
    for (int qh = 0; qh < QHLOOP; qh++) {
      for (int vh = 0; vh < VHELOOP; vh++) {
        const int head_size_elem = vh * WARP_SIZE + laneid;
        for (int i = 0; i < 4; i++) {
          const int head_idx = 4 * qh + i;
          if (head_idx < GQA_RATIO) {
            out_ptr_b16[(wg_start_head_idx + head_idx) * out_num_partitions *
                            HEAD_SIZE +
                        head_size_elem] = vout[qh][vh][i];
          }
        }
      }
    }
  }  // warpid == 0
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_partitions) {
  const int num_heads = gridDim.x;
  const int head_idx = blockIdx.x;
  const int seq_idx = blockIdx.y;
  const int context_len = context_lens[seq_idx];
  const int num_partitions = DIVIDE_ROUND_UP(context_len, PARTITION_SIZE);
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  const int warpid = threadIdx.x / WARP_SIZE;
  const int laneid = threadIdx.x % WARP_SIZE;

  __shared__ float shared_global_exp_sum;
  // max num partitions supported is warp_size * NPAR_LOOPS
  __shared__ float shared_exp_sums[NPAR_LOOPS * WARP_SIZE];

  if (warpid == 0) {
    const float* max_logits_ptr = max_logits +
                                  seq_idx * num_heads * max_num_partitions +
                                  head_idx * max_num_partitions;

    // valid partition is the last valid partition in case threadid > num
    // partitions
    int valid_partition[NPAR_LOOPS];
    float reg_max_logit[NPAR_LOOPS];
    const int last_valid_partition = num_partitions - 1;

  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      valid_partition[i] =
          (partition_no < num_partitions) ? partition_no : last_valid_partition;
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      reg_max_logit[i] = max_logits_ptr[valid_partition[i]];
    }
    float max_logit = reg_max_logit[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      max_logit = fmaxf(max_logit, reg_max_logit[i]);
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      max_logit = fmaxf(max_logit, __shfl_xor(max_logit, mask));
    }

    const float* exp_sums_ptr = exp_sums +
                                seq_idx * num_heads * max_num_partitions +
                                head_idx * max_num_partitions;

    float rescaled_exp_sum[NPAR_LOOPS];
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      rescaled_exp_sum[i] = exp_sums_ptr[valid_partition[i]];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      rescaled_exp_sum[i] *= (partition_no < num_partitions)
                                 ? expf(reg_max_logit[i] - max_logit)
                                 : 0.0f;
    }
    float global_exp_sum = rescaled_exp_sum[0];
  #pragma unroll
    for (int i = 1; i < NPAR_LOOPS; i++) {
      global_exp_sum += rescaled_exp_sum[i];
    }
  #pragma unroll
    for (int i = 0; i < NPAR_LOOPS; i++) {
      const int partition_no = i * WARP_SIZE + threadIdx.x;
      shared_exp_sums[partition_no] = rescaled_exp_sum[i];
    }

  #pragma unroll
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2) {
      global_exp_sum += __shfl_xor(global_exp_sum, mask);
    }
    if (threadIdx.x == 0) {
      shared_global_exp_sum = global_exp_sum;
    }
  }  // warpid == 0
  const scalar_t* tmp_out_ptr =
      tmp_out + seq_idx * num_heads * max_num_partitions * HEAD_SIZE +
      head_idx * max_num_partitions * HEAD_SIZE + threadIdx.x;
  constexpr int MAX_NPAR = 64;
  scalar_t tmps[MAX_NPAR];
  const float dzero = 0.0f;
  #pragma unroll
  for (int j = 0; j < MAX_NPAR; j++) {
    tmps[j] = from_float<scalar_t>(dzero);
  }
  const int last_partition_offset = (num_partitions - 1) * HEAD_SIZE;
  const int num_partition_offset = (num_partitions)*HEAD_SIZE;
  int idx = 0;

  constexpr int JCHUNK = 16;

  #pragma unroll
  for (int j = 0; j < JCHUNK * HEAD_SIZE; j += HEAD_SIZE) {
    // lastj is last valid partition
    const int lastj_offset =
        (j < num_partition_offset) ? j : last_partition_offset;
    tmps[idx] = tmp_out_ptr[lastj_offset];
    idx++;
  }
  __syncthreads();

  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK * HEAD_SIZE; j < 2 * JCHUNK * HEAD_SIZE;
         j += HEAD_SIZE) {
      const int lastj_offset =
          (j < num_partition_offset) ? j : last_partition_offset;
      tmps[idx] = tmp_out_ptr[lastj_offset];
      idx++;
    }

    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK * HEAD_SIZE; j < MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }
    }
  }  // num_partitions > JCHUNK

  // Aggregate tmp_out to out.
  float acc = 0.0f;
  #pragma unroll
  for (int j = 0; j < JCHUNK; j++) {
    acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
  }
  if (num_partitions > JCHUNK) {
  #pragma unroll
    for (int j = JCHUNK; j < 2 * JCHUNK; j++) {
      acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
    }
    if (num_partitions > 2 * JCHUNK) {
  #pragma unroll
      for (int j = 2 * JCHUNK; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j];
      }
    }
  }

  for (int p = 1; p < NPAR_LOOPS; p++) {
    if (num_partitions > p * MAX_NPAR) {
      idx = 0;
  #pragma unroll
      for (int j = p * MAX_NPAR * HEAD_SIZE; j < (p + 1) * MAX_NPAR * HEAD_SIZE;
           j += HEAD_SIZE) {
        // lastj is last valid partition
        const int lastj_offset =
            (j < num_partition_offset) ? j : last_partition_offset;
        tmps[idx] = tmp_out_ptr[lastj_offset];
        idx++;
      }

  #pragma unroll
      for (int j = 0; j < MAX_NPAR; j++) {
        acc += to_float<scalar_t>(tmps[j]) * shared_exp_sums[j + p * MAX_NPAR];
      }
    }
  }

  const float inv_global_exp_sum =
      __fdividef(1.0f, shared_global_exp_sum + 1e-6f);
  acc *= inv_global_exp_sum;

  OUTT* out_ptr = out + seq_idx * num_heads * HEAD_SIZE + head_idx * HEAD_SIZE;
  if constexpr (std::is_same<OUTT, bit8_t>::value) {
    out_ptr[threadIdx.x] = hip_fp8(acc).data;
  } else {
    out_ptr[threadIdx.x] = from_float<scalar_t>(acc);
  }
}

#else  // !defined(__HIP__MI300_MI250__) TODO: Add NAVI support

template <typename scalar_t, typename cache_t,
          vllm::Fp8KVCacheDataType KV_DTYPE, typename OUTT, int BLOCK_SIZE,
          int HEAD_SIZE, int NUM_THREADS, bool ALIBI_ENABLED,
          int GQA_RATIO>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_QKV_mfma16_kernel(
    const scalar_t* __restrict__ q,       // [num_seqs, num_heads, head_size]
    const cache_t* __restrict__ k_cache,  // [num_blocks, num_kv_heads,
                                          // head_size/x, block_size, x]
    const cache_t* __restrict__ v_cache,  // [num_blocks, num_kv_heads,
                                          // head_size, block_size]
    const int num_kv_heads, const float scale,
    const int* __restrict__ block_tables,  // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_blocks_per_seq,
    const float* __restrict__ alibi_slopes,  // [num_heads]
    const int q_stride, const int kv_block_stride, const int kv_head_stride,
    float* __restrict__ exp_sums,  // [num_seqs, num_heads, max_num_partitions]
    float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                     // max_num_partitions]
    scalar_t* __restrict__ out,  // [num_seqs, num_heads, max_num_partitions,
                                 // head_size]
    scalar_t* __restrict__ final_out,  // [num_seqs, num_heads, head_size]
    int max_ctx_blocks, const float* k_scale, const float* v_scale) {
  UNREACHABLE_CODE
}

// Grid: (num_heads, num_seqs).
template <typename scalar_t, typename OUTT, int HEAD_SIZE, int NUM_THREADS,
          int PARTITION_SIZE, int NPAR_LOOPS>
__global__
__launch_bounds__(NUM_THREADS) void paged_attention_ll4mi_reduce_kernel(
    OUTT* __restrict__ out,                // [num_seqs, num_heads, head_size]
    const float* __restrict__ exp_sums,    // [num_seqs, num_heads,
                                           // max_num_partitions]
    const float* __restrict__ max_logits,  // [num_seqs, num_heads,
                                           // max_num_partitions]
    const scalar_t* __restrict__ tmp_out,  // [num_seqs, num_heads,
                                           // max_num_partitions, head_size]
    const int* __restrict__ context_lens,  // [num_seqs]
    const int max_num_partitions){UNREACHABLE_CODE}

#endif  // defined(__HIP__MI300_MI250__) TODO: Add NAVI support

#define LAUNCH_CUSTOM_ATTENTION_MFMA16(GQA_RATIO)                             \
  paged_attention_ll4mi_QKV_mfma16_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE, \
                                          HEAD_SIZE, NTHR, ALIBI_ENABLED,     \
                                          GQA_RATIO>                          \
      <<<grid, block, 0, stream>>>(                                           \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,     \
          block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,         \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,        \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr, max_ctx_blocks, \
          k_scale_ptr, v_scale_ptr);

#define LAUNCH_CUSTOM_ATTENTION_MFMA4(GQA_RATIO)                              \
  paged_attention_ll4mi_QKV_mfma4_kernel<T, KVT, KV_DTYPE, OUTT, BLOCK_SIZE,  \
                                         HEAD_SIZE, NTHR, ALIBI_ENABLED,      \
                                         GQA_RATIO>                           \
      <<<grid, block, 0, stream>>>(                                           \
          query_ptr, key_cache_ptr, value_cache_ptr, num_kv_heads, scale,     \
          block_tables_ptr, context_lens_ptr, max_num_blocks_per_seq,         \
          alibi_slopes_ptr, q_stride, kv_block_stride, kv_head_stride,        \
          exp_sums_ptr, max_logits_ptr, tmp_out_ptr, out_ptr, max_ctx_blocks, \
          k_scale, v_scale);

#define LAUNCH_CUSTOM_REDUCTION(NPAR_LOOPS)                          \
  paged_attention_ll4mi_reduce_kernel<T, OUTT, HEAD_SIZE, HEAD_SIZE, \
                                      PARTITION_SIZE, NPAR_LOOPS>    \
      <<<reduce_grid, reduce_block, 0, stream>>>(                    \
          out_ptr, exp_sums_ptr, max_logits_ptr, tmp_out_ptr,        \
          context_lens_ptr, max_num_partitions);

template <typename T, typename KVT, vllm::Fp8KVCacheDataType KV_DTYPE,
          int BLOCK_SIZE, int HEAD_SIZE, typename OUTT, int PARTITION_SIZE_OLD,
          bool ALIBI_ENABLED>
void paged_attention_custom_launcher(
    torch::Tensor& out, torch::Tensor& exp_sums, torch::Tensor& max_logits,
    torch::Tensor& tmp_out, torch::Tensor& query, torch::Tensor& key_cache,
    torch::Tensor& value_cache, const int num_kv_heads, float scale,
    torch::Tensor& block_tables, torch::Tensor& context_lens,
    int max_context_len, const std::optional<torch::Tensor>& alibi_slopes,
    torch::Tensor& k_scale, torch::Tensor& v_scale) {
  int num_seqs = query.size(0);
  int num_heads = query.size(1);
  int head_size = query.size(2);
  int max_num_blocks_per_seq = block_tables.size(1);
  int q_stride = query.stride(0);
  int kv_block_stride = key_cache.stride(0);
  int kv_head_stride = key_cache.stride(1);

  // NOTE: alibi_slopes is optional.
  const float* alibi_slopes_ptr =
      alibi_slopes
          ? reinterpret_cast<const float*>(alibi_slopes.value().data_ptr())
          : nullptr;

  float* exp_sums_ptr = reinterpret_cast<float*>(exp_sums.data_ptr());
  float* max_logits_ptr = reinterpret_cast<float*>(max_logits.data_ptr());
  T* tmp_out_ptr = reinterpret_cast<T*>(tmp_out.data_ptr());
  T* query_ptr = reinterpret_cast<T*>(query.data_ptr());
  KVT* key_cache_ptr = reinterpret_cast<KVT*>(key_cache.data_ptr());
  KVT* value_cache_ptr = reinterpret_cast<KVT*>(value_cache.data_ptr());
  int* block_tables_ptr = block_tables.data_ptr<int>();
  int* context_lens_ptr = context_lens.data_ptr<int>();
  const float* k_scale_ptr = reinterpret_cast<const float*>(k_scale.data_ptr());
  const float* v_scale_ptr = reinterpret_cast<const float*>(v_scale.data_ptr());

  OUTT* out_ptr = reinterpret_cast<OUTT*>(out.data_ptr());

  const int max_ctx_blocks = DIVIDE_ROUND_UP(max_context_len, BLOCK_SIZE);

  // partition size is fixed at 256 since both mfma4 and mfma16 kernels support
  // it mfma4 kernel also supports partition size 512
  constexpr int PARTITION_SIZE = 256;
  const int max_num_partitions = DIVIDE_ROUND_UP(max_context_len, PARTITION_SIZE);
  const int gqa_ratio = num_heads / num_kv_heads;
  assert(num_heads % num_kv_heads == 0);
  assert(head_size == HEAD_SIZE);

  constexpr int NTHR = 256;
  dim3 grid(num_seqs, max_num_partitions, num_kv_heads);
  dim3 block(NTHR);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(query));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  // mfma4 kernel is faster than mfma16 for gqa_ratio <= 4
  switch (gqa_ratio) {
    case 1:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(1);
      break;
    case 2:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(2);
      break;
    case 3:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(3);
      break;
    case 4:
      LAUNCH_CUSTOM_ATTENTION_MFMA4(4);
      break;
    case 5:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(5);
      break;
    case 6:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(6);
      break;
    case 7:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(7);
      break;
    case 8:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(8);
      break;
    case 9:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(9);
      break;
    case 10:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(10);
      break;
    case 11:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(11);
      break;
    case 12:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(12);
      break;
    case 13:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(13);
      break;
    case 14:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(14);
      break;
    case 15:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(15);
      break;
    case 16:
      LAUNCH_CUSTOM_ATTENTION_MFMA16(16);
      break;
    default:
      TORCH_CHECK(false, "Unsupported gqa ratio: ", gqa_ratio);
      break;
  }

  dim3 reduce_grid(num_heads, num_seqs);
  dim3 reduce_block(head_size);
  const int npar_loops = DIVIDE_ROUND_UP(max_num_partitions, WARP_SIZE);
  // reduction kernel supports upto 8 NPAR *64*256 = 128K context length
  switch (npar_loops) {
    case 1:
      LAUNCH_CUSTOM_REDUCTION(1);
      break;
    case 2:
      LAUNCH_CUSTOM_REDUCTION(2);
      break;
    case 3:
      LAUNCH_CUSTOM_REDUCTION(3);
      break;
    case 4:
      LAUNCH_CUSTOM_REDUCTION(4);
      break;
    case 5:
      LAUNCH_CUSTOM_REDUCTION(5);
      break;
    case 6:
      LAUNCH_CUSTOM_REDUCTION(6);
      break;
    case 7:
      LAUNCH_CUSTOM_REDUCTION(7);
      break;
    case 8:
      LAUNCH_CUSTOM_REDUCTION(8);
      break;
    default:
      TORCH_CHECK(false, "Unsupported npar_loops: ", npar_loops);
      break;
  }
}

#define CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, PSIZE, ALIBI_ENABLED) \
  paged_attention_custom_launcher<T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, T,               \
                                  PSIZE, ALIBI_ENABLED>(                                  \
      out, exp_sums, max_logits, tmp_out, query, key_cache, value_cache,                  \
      num_kv_heads, scale, block_tables, context_lens, max_context_len,                   \
      alibi_slopes, k_scale, v_scale);

#define CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, PSIZE) \
  if (alibi_slopes) {                                                            \
    CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, PSIZE, true);    \
  } else {                                                                       \
    CALL_CUSTOM_LAUNCHER(T, KVT, KV_DTYPE, BLK_SIZE, HEAD_SIZE, PSIZE, false);   \
  }

#define CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, HEAD_SIZE)           \
  switch (block_size) {                                                 \
    case 16:                                                            \
      CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, 16, HEAD_SIZE, 256); \
      break;                                                            \
    case 32:                                                            \
      CALL_CUSTOM_LAUNCHER_ALIBI(T, KVT, KV_DTYPE, 32, HEAD_SIZE, 256); \
      break;                                                            \
    default:                                                            \
      TORCH_CHECK(false, "Unsupported block size: ", block_size);       \
      break;                                                            \
  }

#define CALL_CUSTOM_LAUNCHER_BLK_HEAD(T, KVT, KV_DTYPE)         \
  switch (head_size) {                                          \
    case 64:                                                    \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 64);           \
      break;                                                    \
    case 128:                                                   \
      CALL_CUSTOM_LAUNCHER_BLK(T, KVT, KV_DTYPE, 128);          \
      break;                                                    \
    default:                                                    \
      TORCH_CHECK(false, "Unsupported head size: ", head_size); \
      break;                                                    \
  }
void paged_attention(
    torch::Tensor& out,         // [num_seqs, num_heads, head_size]
    torch::Tensor& exp_sums,    // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor& max_logits,  // [num_seqs, num_heads, max_num_partitions]
    torch::Tensor&
        tmp_out,  // [num_seqs, num_heads, max_num_partitions, head_size]
    torch::Tensor& query,  // [num_seqs, num_heads, head_size]
    torch::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    int64_t num_kv_heads, double scale,
    torch::Tensor& block_tables,  // [num_seqs, max_num_blocks_per_seq]
    torch::Tensor& context_lens,  // [num_seqs]
    int64_t block_size, int64_t max_context_len,
    const std::optional<torch::Tensor>& alibi_slopes,
    const std::string& kv_cache_dtype, torch::Tensor& k_scale,
    torch::Tensor& v_scale) {
  const int head_size = query.size(2);
  if (kv_cache_dtype == "auto") {
    if (query.dtype() == at::ScalarType::Half) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, _Float16,
                                    vllm::Fp8KVCacheDataType::kAuto);
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, __hip_bfloat16,
                                    vllm::Fp8KVCacheDataType::kAuto);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (query.dtype() == at::ScalarType::Half) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(_Float16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (query.dtype() == at::ScalarType::BFloat16) {
      CALL_CUSTOM_LAUNCHER_BLK_HEAD(__hip_bfloat16, uint8_t,
                                    vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else {
      TORCH_CHECK(false, "Unsupported data type: ", query.dtype());
    }
  } else {
    TORCH_CHECK(false, "Unsupported KV cache dtype: ", kv_cache_dtype);
  }
}

#undef WARP_SIZE
#undef MAX
#undef MIN
#undef DIVIDE_ROUND_UP
