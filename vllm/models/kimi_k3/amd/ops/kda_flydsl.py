# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Dedicated one-workgroup gfx950 convolution + KDA + gated RMSNorm kernel.

This specialization owns all 128 V rows in one workgroup. RMSNorm is entirely
local: raw FP32 outputs stay in LDS until the workgroup computes the RMS scale.

Do not enable ``from __future__ import annotations`` in this file.  FlyDSL
needs the concrete annotations on ``SharedStorage`` when it calculates LDS
storage.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl._mlir.dialects import arith as _mlir_arith
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from aiter.ops.flydsl.kernels import dpp_utils
from vllm.utils.torch_utils import direct_register_custom_op


K_DIM = 128
V_DIM = 128

QKV_PARTS = 3
CONV_WIDTH = 4
CONV_HISTORY = CONV_WIDTH - 1

WAVE_SIZE = 64
STATE_VEC_ELEMS = 4
CACHED_CACHE_MODIFIER = 0
NONTEMPORAL_CACHE_MODIFIER = 2  # Emits the gfx950 nt cache-policy hint.


def depthwise_conv4_silu(
    current,
    history_0,
    history_1,
    history_2,
    weight_0,
    weight_1,
    weight_2,
    weight_3,
    fm_fast,
):
    """Compute one channel of Kimi's width-4 causal convolution.

    The caller owns all memory operations. It supplies the three cached inputs
    [x[t-3], x[t-2], x[t-1]], the current input x[t], and the four weights for
    one depthwise channel. Keeping this helper purely arithmetic lets Q, K,
    and V use the same implementation regardless of how their packed
    convolution cache is laid out.

    Returns the SiLU output and the shifted three-element history.
    """
    accumulator = history_0 * weight_0
    accumulator = fmath.fma(history_1, weight_1, accumulator, fastmath=fm_fast)
    accumulator = fmath.fma(history_2, weight_2, accumulator, fastmath=fm_fast)
    accumulator = fmath.fma(current, weight_3, accumulator, fastmath=fm_fast)

    sigmoid = rocdl.rcp(
        T.f32,
        1.0 + fmath.exp(-accumulator, fastmath=fm_fast),
    )
    return accumulator * sigmoid, history_1, history_2, current


@fx.struct
class SharedStorage:
    q: fx.Array[fx.Float32, K_DIM, 16]
    k: fx.Array[fx.Float32, K_DIM, 16]
    decay: fx.Array[fx.Float32, K_DIM, 16]
    v: fx.Array[fx.Float32, V_DIM, 16]

    # Sixteen slots hold one partial per 8-lane group across 128 Q/K channels.
    q_red_slot: fx.Array[fx.Float32, 16, 16]
    k_red_slot: fx.Array[fx.Float32, 16, 16]
    qk_red_slot: fx.Array[fx.Float32, 16, 16]
    output: fx.Array[fx.Float32, V_DIM, 16]


@flyc.kernel
def _kda_conv_rms_single_kernel(
    mixed_qkv: fx.Tensor,
    conv_weight: fx.Tensor,
    conv_state: fx.Tensor,
    raw_g: fx.Tensor,
    raw_beta: fx.Tensor,
    output_gate: fx.Tensor,
    norm_weight: fx.Tensor,
    A_log: fx.Tensor,
    dt_bias: fx.Tensor,
    state_indices: fx.Tensor,
    state: fx.Tensor,
    out: fx.Tensor,
    conv_state_slot_stride: fx.Int32,
    conv_state_channel_stride: fx.Int32,
    conv_state_history_stride: fx.Int32,
    state_slot_stride: fx.Int32,
    heads: fx.Int32,
    mixed_qkv_token_stride: fx.Int32,
    raw_g_token_stride: fx.Int32,
    raw_beta_token_stride: fx.Int32,
    output_gate_token_stride: fx.Int32,
    query_scale: fx.Float32,
    norm_eps: fx.Float32,
    lower_bound: fx.Float32,
    use_lower_bound: fx.Constexpr,
    threads_per_row: fx.Constexpr,
    num_waves: fx.Constexpr,
    early_prefetch: fx.Constexpr,
    state_store_cache_modifier: fx.Constexpr,
    state_prefetch_split_tiles: fx.Constexpr,
):
    """Run fused convolution, KDA state update, and gated RMSNorm for one head.

    Each workgroup owns one ``(token, head)`` pair and all 128 output channels.
    It applies the width-4 causal depthwise convolution to Q, K, and V, advances
    their convolution histories, normalizes Q and K, and computes the learned
    per-channel decay. Lane groups cooperatively load and update rows of the
    128-by-128 recurrent state in FP32. The resulting attention values remain
    in LDS while the workgroup computes their RMS, applies the normalization
    weights and sigmoid output gate, and writes BF16 output. ``early_prefetch``
    controls whether state loads are issued before or after Q/K normalization.

    A non-positive state slot denotes padding: the workgroup writes zero output and
    leaves recurrent and convolution state untouched.
    """

    tid = gpu.thread_idx.x
    token_head = gpu.block_idx.x
    token = token_head // heads
    head = token_head % heads
    lane = tid % WAVE_SIZE

    block_threads = WAVE_SIZE * num_waves
    state_elems_per_thread = K_DIM // threads_per_row
    rows_per_tile = block_threads // threads_per_row
    num_v_tiles = V_DIM // rows_per_tile
    state_loads_per_thread = state_elems_per_thread // STATE_VEC_ELEMS
    state_chunk_stride = STATE_VEC_ELEMS * threads_per_row
    qk_reduce_groups = K_DIM // threads_per_row

    qk_group = tid // threads_per_row
    row_in_tile = qk_group
    k_group = lane % threads_per_row
    k_start = k_group * STATE_VEC_ELEMS
    output_channel = tid - (block_threads - V_DIM)

    fm_fast = arith.FastMathFlags.fast
    zero = fx.Float32(0.0)

    raw_g_rsrc = buffer_ops.create_buffer_resource(raw_g)
    raw_beta_rsrc = buffer_ops.create_buffer_resource(raw_beta)
    a_log_rsrc = buffer_ops.create_buffer_resource(A_log)
    mixed_qkv_rsrc = buffer_ops.create_buffer_resource(mixed_qkv)
    conv_weight_rsrc = buffer_ops.create_buffer_resource(conv_weight)
    conv_state_rsrc = buffer_ops.create_buffer_resource(conv_state)
    output_gate_rsrc = buffer_ops.create_buffer_resource(output_gate)
    norm_weight_rsrc = buffer_ops.create_buffer_resource(norm_weight)
    dt_bias_rsrc = buffer_ops.create_buffer_resource(dt_bias)
    slot_rsrc = buffer_ops.create_buffer_resource(state_indices)
    state_rsrc = buffer_ops.create_buffer_resource(state)
    out_rsrc = buffer_ops.create_buffer_resource(out)

    slot_raw = buffer_ops.buffer_load(
        slot_rsrc, token, vec_width=1, is_scalar=True
    )
    slot = fx.Int32(slot_raw)

    if slot <= 0:
        if tid < V_DIM:
            buffer_ops.buffer_store(
                fx.BFloat16(0.0),
                out_rsrc,
                token_head * V_DIM + tid,
            )
        return

    shared = fx.SharedAllocator().allocate(SharedStorage).peek()
    q_lds = shared.q.view(fx.make_layout(K_DIM, 1))
    k_lds = shared.k.view(fx.make_layout(K_DIM, 1))
    decay_lds = shared.decay.view(fx.make_layout(K_DIM, 1))
    v_lds = shared.v.view(fx.make_layout(V_DIM, 1))
    q_red_slot = shared.q_red_slot.view(fx.make_layout(qk_reduce_groups, 1))
    k_red_slot = shared.k_red_slot.view(fx.make_layout(qk_reduce_groups, 1))
    qk_red_slot = shared.qk_red_slot.view(fx.make_layout(qk_reduce_groups, 1))
    output_lds = shared.output.view(fx.make_layout(V_DIM, 1))

    def row_reduce_add(value, width):
        """Reduce an aligned 16-lane row using directly-fusible DPP adds."""
        result = value
        dpp_controls = [0xB1, 0x4E, 0x141, 0x140]
        for step in range_constexpr(int(math.log2(width))):
            src_i32 = arith.unwrap(result).bitcast(T.i32)
            peer_i32 = dpp_utils.update_dpp_i32(
                0,
                src_i32,
                dpp_controls[step],
                0xF,
                0xF,
                True,
            )
            peer = fx.Float32(_mlir_arith.BitcastOp(T.f32, peer_i32).result)
            result = result.addf(peer, fastmath=fm_fast)
        return result

    def block_reduce_add3(
        value_q, value_k, value_qk, q_scratch, k_scratch, qk_scratch
    ):
        """Reduce Q/K partials across all active lane groups."""
        if tid < K_DIM:
            group_sum_q = row_reduce_add(value_q, threads_per_row)
            group_sum_k = row_reduce_add(value_k, threads_per_row)
            group_sum_qk = row_reduce_add(value_qk, threads_per_row)
            if k_group == 0:
                fx.memref_store(group_sum_q, q_scratch, qk_group)
                fx.memref_store(group_sum_k, k_scratch, qk_group)
                fx.memref_store(group_sum_qk, qk_scratch, qk_group)

        gpu.barrier()

        if tid < qk_reduce_groups:
            block_sum_q = row_reduce_add(
                fx.memref_load(q_scratch, tid),
                qk_reduce_groups,
            )
            block_sum_k = row_reduce_add(
                fx.memref_load(k_scratch, tid),
                qk_reduce_groups,
            )
            block_sum_qk = row_reduce_add(
                fx.memref_load(qk_scratch, tid),
                qk_reduce_groups,
            )
            if tid == 0:
                fx.memref_store(block_sum_q, q_scratch, 0)
                fx.memref_store(block_sum_k, k_scratch, 0)
                fx.memref_store(block_sum_qk, qk_scratch, 0)

        gpu.barrier()

        return (
            fx.memref_load(q_scratch, 0),
            fx.memref_load(k_scratch, 0),
            fx.memref_load(qk_scratch, 0),
        )


    def load_state_chunks(state_resource, state_elem_base):
        chunks = [None] * state_loads_per_thread
        for load_idx in range_constexpr(state_loads_per_thread):
            first = load_idx * state_chunk_stride
            chunks[load_idx] = Vec(
                buffer_ops.buffer_load(
                    state_resource,
                    state_elem_base + first,
                    vec_width=STATE_VEC_ELEMS,
                    dtype=T.f32,
                    cache_modifier=NONTEMPORAL_CACHE_MODIFIER,
                )
            )
        return chunks

    def conv_input_index(part, channel):
        return (
            token * mixed_qkv_token_stride
            + (part * heads + head) * K_DIM
            + channel
        )

    def conv_weight_index(part, tap, channel):
        packed_channel = (part * heads + head) * K_DIM + channel
        return packed_channel * CONV_WIDTH + tap

    def conv_state_index(history, part, channel):
        packed_channel = (part * heads + head) * K_DIM + channel
        return (
            slot * conv_state_slot_stride
            + packed_channel * conv_state_channel_stride
            + history * conv_state_history_stride
        )

    def load_conv_channel(
        part,
        channel,
        mixed_qkv_resource,
        conv_weight_resource,
        conv_state_resource,
    ):
        current_raw = buffer_ops.buffer_load(
            mixed_qkv_resource,
            conv_input_index(part, channel),
            vec_width=1,
            dtype=T.bf16,
        )
        history_raw = [
            buffer_ops.buffer_load(
                conv_state_resource,
                conv_state_index(history, part, channel),
                vec_width=1,
                dtype=T.bf16,
            )
            for history in range_constexpr(CONV_HISTORY)
        ]
        weights = [
            buffer_ops.buffer_load(
                conv_weight_resource,
                conv_weight_index(part, tap, channel),
                vec_width=1,
                dtype=T.f32,
            )
            for tap in range_constexpr(CONV_WIDTH)
        ]
        current = fx.Float32(current_raw)
        history = [
            fx.Float32(history_raw[idx])
            for idx in range_constexpr(CONV_HISTORY)
        ]
        return depthwise_conv4_silu(
            current,
            history[0],
            history[1],
            history[2],
            weights[0],
            weights[1],
            weights[2],
            weights[3],
            fm_fast,
        )

    def store_conv_history(
        part, channel, next_0, next_1, next_2, conv_state_resource
    ):
        buffer_ops.buffer_store(
            next_0.to(fx.BFloat16),
            conv_state_resource,
            conv_state_index(0, part, channel),
        )
        buffer_ops.buffer_store(
            next_1.to(fx.BFloat16),
            conv_state_resource,
            conv_state_index(1, part, channel),
        )
        buffer_ops.buffer_store(
            next_2.to(fx.BFloat16),
            conv_state_resource,
            conv_state_index(2, part, channel),
        )

    def compute_qk_convolution_and_gate(
        mixed_qkv_resource,
        conv_weight_resource,
        conv_state_resource,
        raw_g_resource,
        dt_bias_resource,
    ):
        (
            q_value,
            q_next_0,
            q_next_1,
            q_next_2,
        ) = load_conv_channel(
            0,
            tid,
            mixed_qkv_resource,
            conv_weight_resource,
            conv_state_resource,
        )
        (
            k_value,
            k_next_0,
            k_next_1,
            k_next_2,
        ) = load_conv_channel(
            1,
            tid,
            mixed_qkv_resource,
            conv_weight_resource,
            conv_state_resource,
        )
        store_conv_history(
            0,
            tid,
            q_next_0,
            q_next_1,
            q_next_2,
            conv_state_resource,
        )
        store_conv_history(
            1,
            tid,
            k_next_0,
            k_next_1,
            k_next_2,
            conv_state_resource,
        )

        gate_elem = buffer_ops.buffer_load(
            raw_g_resource,
            token * raw_g_token_stride + head * K_DIM + tid,
            vec_width=1,
            dtype=T.bf16,
        )
        bias_elem = buffer_ops.buffer_load(
            dt_bias_resource,
            head * K_DIM + tid,
            vec_width=1,
            dtype=T.f32,
        )
        return q_value, k_value, fx.Float32(gate_elem) + bias_elem

    def prepare_v_and_output_gate(
        mixed_qkv_resource,
        conv_weight_resource,
        conv_state_resource,
        output_gate_resource,
        norm_weight_resource,
        v_shared,
    ):
        output_gate_raw = buffer_ops.buffer_load(
            output_gate_resource,
            token * output_gate_token_stride
            + head * V_DIM
            + output_channel,
            vec_width=1,
            dtype=T.bf16,
        )
        gate_sigmoid = rocdl.rcp(
            T.f32,
            1.0 + fmath.exp(-fx.Float32(output_gate_raw), fastmath=fm_fast),
        )
        weight = buffer_ops.buffer_load(
            norm_weight_resource, output_channel, vec_width=1, dtype=T.f32
        )
        (
            v_value,
            v_next_0,
            v_next_1,
            v_next_2,
        ) = load_conv_channel(
            2,
            output_channel,
            mixed_qkv_resource,
            conv_weight_resource,
            conv_state_resource,
        )
        fx.memref_store(v_value, v_shared, output_channel)
        store_conv_history(
            2,
            output_channel,
            v_next_0,
            v_next_1,
            v_next_2,
            conv_state_resource,
        )
        return gate_sigmoid, weight

    def compute_decay(
        gate, a_log, decay_lower_bound, use_decay_lower_bound
    ):
        a = fmath.exp(a_log, fastmath=fm_fast)
        if const_expr(use_decay_lower_bound):
            sigmoid_arg = a * gate
            log_decay = decay_lower_bound / (
                1.0 + fmath.exp(-sigmoid_arg, fastmath=fm_fast)
            )
        else:
            softplus = gate
            if gate <= 20.0:
                softplus = fmath.log1p(
                    fmath.exp(gate, fastmath=fm_fast), fastmath=fm_fast
                )
            log_decay = -a * softplus

        exp2_arg = log_decay * fx.Float32(1.4426950408889634)
        return rocdl.exp2(T.f32, arith.unwrap(exp2_arg))

    def prefetch_state_tile(state_resource, prefetch_tile):
        return load_state_chunks(
            state_resource,
            slot * state_slot_stride
            + (head * V_DIM + prefetch_tile * rows_per_tile + row_in_tile)
            * K_DIM
            + k_start
        )

    def prefetch_state_tiles(state_resource):
        return [
            prefetch_state_tile(state_resource, prefetch_tile)
            for prefetch_tile in range_constexpr(num_v_tiles)
        ]

    def lds_vec(lds_view, first):
        return Vec.from_elements(
            [
                fx.memref_load(
                    lds_view,
                    k_start + first + vec_elem,
                )
                for vec_elem in range_constexpr(STATE_VEC_ELEMS)
            ],
            fx.Float32,
        )

    def load_lds_fragments(lds_view):
        return [
            lds_vec(lds_view, chunk * state_chunk_stride)
            for chunk in range_constexpr(state_loads_per_thread)
        ]

    def update_state_row(
        state_chunks,
        v_elem,
        state_elem_base,
        normalized_qk,
        sigmoid_beta,
        q_fragments,
        k_fragments,
        decay_fragments,
        state_resource,
    ):
        partial_k = Vec.from_elements(
            [zero for _ in range_constexpr(STATE_VEC_ELEMS)], fx.Float32
        )
        partial_q = Vec.from_elements(
            [zero for _ in range_constexpr(STATE_VEC_ELEMS)], fx.Float32
        )
        for chunk in range_constexpr(state_loads_per_thread):
            decayed_vec = state_chunks[chunk] * decay_fragments[chunk]
            state_chunks[chunk] = decayed_vec
            partial_k = fmath.fma(
                decayed_vec, k_fragments[chunk], partial_k, fastmath=fm_fast
            )
            partial_q = fmath.fma(
                decayed_vec, q_fragments[chunk], partial_q, fastmath=fm_fast
            )

        lane_dot_k = (partial_k[0] + partial_k[1]) + (
            partial_k[2] + partial_k[3]
        )
        lane_dot_q = (partial_q[0] + partial_q[1]) + (
            partial_q[2] + partial_q[3]
        )
        prediction = row_reduce_add(lane_dot_k, threads_per_row)
        history_dot_q = row_reduce_add(lane_dot_q, threads_per_row)
        delta = sigmoid_beta * (v_elem - prediction)
        delta_vec = Vec.from_elements([delta], fx.Float32).broadcast_to(
            STATE_VEC_ELEMS
        )

        for chunk in range_constexpr(state_loads_per_thread):
            state_chunks[chunk] = fmath.fma(
                k_fragments[chunk],
                delta_vec,
                state_chunks[chunk],
                fastmath=fm_fast,
            )

        for chunk in range_constexpr(state_loads_per_thread):
            buffer_ops.buffer_store(
                state_chunks[chunk],
                state_resource,
                state_elem_base + chunk * state_chunk_stride,
                cache_modifier=state_store_cache_modifier,
            )

        return fmath.fma(
            delta, normalized_qk, history_dot_q, fastmath=fm_fast
        )

    def store_gated_rms_output(
        output_gate_sigmoid,
        output_norm_weight,
        norm_epsilon,
        sumsq_shared,
        output_shared,
        output_resource,
    ):
        local_sumsq = fx.memref_load(sumsq_shared, 0)
        inv_rms = fmath.rsqrt(
            local_sumsq / fx.Float32(V_DIM) + norm_epsilon, fastmath=fm_fast
        )
        normalized = (
            fx.memref_load(output_shared, output_channel)
            * inv_rms
            * output_norm_weight
            * output_gate_sigmoid
        )
        buffer_ops.buffer_store(
            normalized.to(fx.BFloat16),
            output_resource,
            token_head * V_DIM + output_channel,
        )

    q_f32 = zero
    k_f32 = zero
    gate_f32 = zero
    output_gate_sigmoid = zero
    output_norm_weight = zero
    a_log_uniform_raw = buffer_ops.buffer_load(
        a_log_rsrc, head, vec_width=1, is_scalar=True
    )
    a_log_uniform = _mlir_arith.BitcastOp(T.f32, a_log_uniform_raw).result

    # Early prefetch can hide state-load latency behind convolution and Q/K work,
    # but it also keeps every prefetched tile live in VGPRs for longer,
    # increasing register pressure. I found out that staging the prefetch after the 
    # q/k normalization can reduce VGPR usage and improve latency for lower batch sizes.
    # So i just exposed it as a tunable parameter and emperically obtained the best choice
    # for each batch size.
    if const_expr(state_prefetch_split_tiles > 0):
        early_state_tiles = [
            prefetch_state_tile(state_rsrc, tile)
            for tile in range_constexpr(state_prefetch_split_tiles)
        ]
    elif const_expr(early_prefetch):
        prefetched_state_tiles = prefetch_state_tiles(state_rsrc)
    beta_raw = buffer_ops.buffer_load(
        raw_beta_rsrc,
        token * raw_beta_token_stride + head,
        vec_width=1,
        dtype=T.bf16,
    )

    if tid < K_DIM:
        q_f32, k_f32, gate_f32 = compute_qk_convolution_and_gate(
            mixed_qkv_rsrc,
            conv_weight_rsrc,
            conv_state_rsrc,
            raw_g_rsrc,
            dt_bias_rsrc,
        )

    if tid >= block_threads - V_DIM:
        output_gate_sigmoid, output_norm_weight = prepare_v_and_output_gate(
            mixed_qkv_rsrc,
            conv_weight_rsrc,
            conv_state_rsrc,
            output_gate_rsrc,
            norm_weight_rsrc,
            v_lds,
        )

    q_sumsq, k_sumsq, qk_sum = block_reduce_add3(
        q_f32 * q_f32,
        k_f32 * k_f32,
        q_f32 * k_f32,
        q_red_slot,
        k_red_slot,
        qk_red_slot,
    )

    q_inv_norm = fmath.rsqrt(q_sumsq + 1.0e-6, fastmath=fm_fast)
    k_inv_norm = fmath.rsqrt(k_sumsq + 1.0e-6, fastmath=fm_fast)
    normalized_qk = qk_sum * q_inv_norm * query_scale * k_inv_norm

    if const_expr(state_prefetch_split_tiles > 0):
        late_state_tiles = [
            prefetch_state_tile(state_rsrc, state_prefetch_split_tiles + tile)
            for tile in range_constexpr(num_v_tiles - state_prefetch_split_tiles)
        ]
    elif const_expr(not early_prefetch):
        prefetched_state_tiles = prefetch_state_tiles(state_rsrc)

    if tid < K_DIM:
        q_normalized = q_f32 * q_inv_norm * query_scale
        k_normalized = k_f32 * k_inv_norm

        decay = compute_decay(
            gate_f32,
            a_log_uniform,
            lower_bound,
            use_lower_bound,
        )

        fx.memref_store(q_normalized, q_lds, tid)
        fx.memref_store(k_normalized, k_lds, tid)
        fx.memref_store(decay, decay_lds, tid)

    sigmoid_beta = rocdl.rcp(
        T.f32,
        1.0 + fmath.exp(-fx.Float32(beta_raw), fastmath=fm_fast),
    )

    gpu.barrier()


    q_fragments = load_lds_fragments(q_lds)
    k_fragments = load_lds_fragments(k_lds)
    decay_fragments = load_lds_fragments(decay_lds)

    local_output_sumsq = zero

    for tile in range_constexpr(num_v_tiles):
        v_row = tile * rows_per_tile + row_in_tile
        state_elem_base = (
            slot * state_slot_stride
            + (head * V_DIM + v_row) * K_DIM
            + k_start
        )
        if const_expr(state_prefetch_split_tiles > 0):
            if const_expr(tile < state_prefetch_split_tiles):
                state_chunks = early_state_tiles[tile]
            else:
                state_chunks = late_state_tiles[tile - state_prefetch_split_tiles]
        else:
            state_chunks = prefetched_state_tiles[tile]
        v_elem = fx.memref_load(v_lds, v_row)

        raw_output = update_state_row(
            state_chunks,
            v_elem,
            state_elem_base,
            normalized_qk,
            sigmoid_beta,
            q_fragments,
            k_fragments,
            decay_fragments,
            state_rsrc,
        )

        if k_group == 0:
            local_output_sumsq = local_output_sumsq + raw_output * raw_output
            fx.memref_store(raw_output, output_lds, v_row)

    if k_group == 0:
        fx.memref_store(local_output_sumsq, q_lds, row_in_tile)
    gpu.barrier()
    if tid < threads_per_row:
        leader_sums = zero
        for rms_group in range_constexpr(rows_per_tile // threads_per_row):
            leader_sums = leader_sums + fx.memref_load(
                q_lds,
                k_group + rms_group * threads_per_row,
            )
        output_sumsq = row_reduce_add(leader_sums, threads_per_row)
        if k_group == 0:
            fx.memref_store(output_sumsq, q_lds, 0)
    gpu.barrier()
    if tid >= block_threads - V_DIM:
        store_gated_rms_output(
            output_gate_sigmoid,
            output_norm_weight,
            norm_eps,
            q_lds,
            output_lds,
            out_rsrc,
        )


@flyc.jit
def _launch_kda_conv_rms_single(
    mixed_qkv: fx.Tensor,
    conv_weight: fx.Tensor,
    conv_state: fx.Tensor,
    raw_g: fx.Tensor,
    raw_beta: fx.Tensor,
    output_gate: fx.Tensor,
    norm_weight: fx.Tensor,
    A_log: fx.Tensor,
    dt_bias: fx.Tensor,
    state_indices: fx.Tensor,
    state: fx.Tensor,
    out: fx.Tensor,
    conv_state_slot_stride: fx.Int32,
    conv_state_channel_stride: fx.Int32,
    conv_state_history_stride: fx.Int32,
    state_slot_stride: fx.Int32,
    heads: fx.Int32,
    token_heads: fx.Int32,
    mixed_qkv_token_stride: fx.Int32,
    raw_g_token_stride: fx.Int32,
    raw_beta_token_stride: fx.Int32,
    output_gate_token_stride: fx.Int32,
    query_scale: fx.Float32,
    norm_eps: fx.Float32,
    lower_bound: fx.Float32,
    use_lower_bound: fx.Constexpr,
    threads_per_row: fx.Constexpr,
    num_waves: fx.Constexpr,
    early_prefetch: fx.Constexpr,
    state_store_cache_modifier: fx.Constexpr,
    state_prefetch_split_tiles: fx.Constexpr,
    stream: fx.Stream = fx.Stream(None),
):
    _kda_conv_rms_single_kernel(
        mixed_qkv,
        conv_weight,
        conv_state,
        raw_g,
        raw_beta,
        output_gate,
        norm_weight,
        A_log,
        dt_bias,
        state_indices,
        state,
        out,
        conv_state_slot_stride,
        conv_state_channel_stride,
        conv_state_history_stride,
        state_slot_stride,
        heads,
        mixed_qkv_token_stride,
        raw_g_token_stride,
        raw_beta_token_stride,
        output_gate_token_stride,
        query_scale,
        norm_eps,
        lower_bound,
        use_lower_bound,
        threads_per_row,
        num_waves,
        early_prefetch,
        state_store_cache_modifier,
        state_prefetch_split_tiles,
    ).launch(
        grid=(token_heads, 1, 1),
        block=(WAVE_SIZE * num_waves, 1, 1),
        stream=stream,
    )

_COMPILED_KERNELS: dict[tuple, object] = {}


def _get_launch_config(num_tokens: int) -> tuple[int, int, int, bool, int, int]:
    # Empirically tuned with cold state on AMD Instinct MI355X (gfx950).
    state_store_cache_modifier = NONTEMPORAL_CACHE_MODIFIER
    state_prefetch_split_tiles = 0
    if num_tokens <= 8:
        threads_per_row, num_waves = 8, 4
        waves_per_eu, early_prefetch = 0, False
        if num_tokens > 4:
            state_store_cache_modifier = CACHED_CACHE_MODIFIER
    elif num_tokens <= 32:
        threads_per_row, num_waves = 8, 4
        waves_per_eu, early_prefetch = 0, True
    elif num_tokens <= 64:
        threads_per_row, num_waves = 8, 2
        waves_per_eu, early_prefetch = 4, True
    elif num_tokens <= 128:
        threads_per_row, num_waves = 8, 4
        waves_per_eu, early_prefetch = 0, False
    else:
        threads_per_row, num_waves = 16, 4
        waves_per_eu, early_prefetch = 2, True
    return (
        threads_per_row, num_waves, waves_per_eu, early_prefetch,
        state_store_cache_modifier, state_prefetch_split_tiles,
    )


def is_supported(
    mixed_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    output_gate: torch.Tensor,
    state: torch.Tensor,
    head_dim: int,
    conv_width: int,
    has_speculative_decode: bool,
    num_decodes: int,
) -> bool:
    """Return whether the fused kernel supports these args.
    """
    return (
        not has_speculative_decode
        and num_decodes > 0
        and head_dim == K_DIM
        and conv_width == CONV_WIDTH
        and mixed_qkv.dtype == torch.bfloat16
        and conv_state.dtype == torch.bfloat16
        and state.dtype == torch.float32
        and mixed_qkv.stride(-1) == 1
        and raw_g.stride(-1) == 1
        and raw_g.stride(-2) == K_DIM
        and raw_beta.stride(-1) == 1
        and output_gate.stride(-1) == 1
        and output_gate.stride(-2) == V_DIM
        and state.stride(-1) == 1
        and state.stride(-2) == K_DIM
        and state.stride(-3) == V_DIM * K_DIM
    )


def _kda_flydsl_decode_impl(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    query_scale: float,
    norm_eps: float,
    lower_bound: float,
    use_lower_bound: bool,
) -> None:
    num_tokens = mixed_qkv.shape[0]
    heads = a_log.numel()
    token_heads = num_tokens * heads
    (
        threads_per_row,
        num_waves,
        waves_per_eu,
        early_prefetch,
        state_store_cache_modifier,
        state_prefetch_split_tiles,
    ) = _get_launch_config(num_tokens)
    stream = torch.cuda.current_stream(mixed_qkv.device)
    launch_args = (
        mixed_qkv,
        conv_weight,
        conv_state,
        raw_g,
        raw_beta,
        output_gate,
        norm_weight,
        a_log,
        dt_bias,
        state_indices,
        state,
        out,
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        state.stride(0),
        heads,
        token_heads,
        mixed_qkv.stride(0),
        raw_g.stride(1),
        raw_beta.stride(1),
        output_gate.stride(0),
        query_scale,
        norm_eps,
        lower_bound,
        use_lower_bound,
        threads_per_row,
        num_waves,
        early_prefetch,
        state_store_cache_modifier,
        state_prefetch_split_tiles,
        stream,
    )
    key = (
        mixed_qkv.device.index,
        tuple(mixed_qkv.shape),
        tuple(conv_state.shape),
        tuple(state.shape),
        tuple(conv_state.stride()),
        tuple(state.stride()),
        threads_per_row,
        num_waves,
        waves_per_eu,
        early_prefetch,
        state_store_cache_modifier,
        state_prefetch_split_tiles,
        use_lower_bound,
    )
    compiled = _COMPILED_KERNELS.get(key)
    if compiled is None:
        _launch_kda_conv_rms_single.compile_hints = {
            "waves_per_eu": waves_per_eu
        }
        compiled = flyc.compile(_launch_kda_conv_rms_single, *launch_args)
        _COMPILED_KERNELS[key] = compiled
    else:
        compiled(*launch_args)


def _kda_flydsl_decode_fake(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    query_scale: float,
    norm_eps: float,
    lower_bound: float,
    use_lower_bound: bool,
) -> None:
    return


direct_register_custom_op(
    op_name="kda_flydsl_decode",
    op_func=_kda_flydsl_decode_impl,
    mutates_args=["conv_state", "state", "out"],
    fake_impl=_kda_flydsl_decode_fake,
)


def kda_flydsl_decode(
    mixed_qkv: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    state_indices: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    query_scale: float,
    norm_eps: float,
    lower_bound: float,
    use_lower_bound: bool,
) -> None:
    torch.ops.vllm.kda_flydsl_decode(
        mixed_qkv,
        conv_weight,
        conv_state,
        raw_g,
        raw_beta,
        output_gate,
        norm_weight,
        a_log,
        dt_bias,
        state_indices,
        state,
        out,
        query_scale,
        norm_eps,
        lower_bound,
        use_lower_bound,
    )
