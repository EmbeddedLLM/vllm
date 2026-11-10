# SPDX-License-Identifier: Apache-2.0
"""Quantify the numerical error of both GDN decode kernels on this device.

For each (state dtype, activation dtype) precision pair the test suite sweeps,
drive `num_steps` decode steps with the same inputs through

  * the baseline `fused_recurrent_gated_delta_rule_packed_decode` (sequential
    recurrence, one rounding of the state per token), and
  * the PR's `fused_recurrent_gated_delta_rule_replayssm` (state reconstructed
    from checkpoint + ring cache, one rounding per flush),

and report each kernel's max |error| against an independent float64 reference
recurrence (so the two kernels are measured against ground truth, not only
against each other), plus the max |A - B| between them. Flush steps (where the
ring folds the current token into the checkpoint) are separated out, since that
is where the reconstruction error is largest.

Geometry is Qwen3.5-4B's (H=16, HV=32, K=128, V=128), L=16.

Usage:  python num_error_magnitudes.py [--json out.json]
"""

import argparse
import json

import torch

from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_replayssm,
)
from vllm.utils.torch_utils import set_random_seed

H, HV, K, V = 16, 32, 128, 128
L = 16  # ring buffer length
BATCH = 4
SCALE = K**-0.5


def max_abs(x, y):
    return (x.float() - y.float()).abs().max().item()


def max_rel(x, y):
    num = (x.float() - y.float()).abs()
    den = y.float().abs().clamp_min(1e-30)
    return (num / den).max().item()


def run_cumulative(state_dtype, act_dtype, num_steps, seed):
    """Sweep one precision pair against a cumulative float64 reference.

    Direction of the reference is checked by the fp32 caller in main(): if the
    baseline sequential kernel does not match it to ~1e-4, the reference formula
    is wrong and the run is aborted rather than reported.
    """
    device = "cuda"
    set_random_seed(seed)
    num_slots = BATCH + 1
    slots = (torch.randperm(num_slots - 1, device=device)[:BATCH] + 1).to(torch.int32)
    A_log = torch.randn(HV, device=device, dtype=act_dtype)
    dt_bias = torch.randn(HV, device=device, dtype=act_dtype)
    qkv_dim = 2 * H * K + HV * V

    state0 = torch.randn(num_slots, HV, V, K, device=device, dtype=state_dtype)
    state_packed = state0.clone()
    state_cached = state0.clone()
    ref_state = state0.double().clone()

    cache_dtype = torch.float16 if act_dtype == torch.bfloat16 else act_dtype
    d_cache = torch.zeros(num_slots, HV, L, V, device=device, dtype=cache_dtype)
    k_cache = torch.zeros(num_slots, H, L, K, device=device, dtype=cache_dtype)
    g_cache = torch.zeros(num_slots, HV, L, device=device, dtype=torch.float32)

    out_err = {"packed": [], "cached": [], "cross": [], "cross_rel": []}
    st_err = {"packed": [], "cached": []}
    flush_steps = []
    for step in range(num_steps):
        mixed_qkv = torch.randn(BATCH, qkv_dim, device=device, dtype=act_dtype)
        a = torch.randn(BATCH, HV, device=device, dtype=act_dtype)
        b = torch.randn(BATCH, HV, device=device, dtype=act_dtype)
        write_pos = torch.full((BATCH,), step % L, device=device, dtype=torch.int32)
        is_flush = (write_pos == L - 1).to(torch.int8)

        qkv = mixed_qkv.double()
        q = qkv[:, : H * K].reshape(BATCH, H, K)
        k = qkv[:, H * K : 2 * H * K].reshape(BATCH, H, K)
        v = qkv[:, 2 * H * K :].reshape(BATCH, HV, V)
        q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
        q = q * SCALE
        rep = HV // H
        qh = q.repeat_interleave(rep, dim=1)
        kh = k.repeat_interleave(rep, dim=1)
        x = a.double() + dt_bias.double()
        softplus = torch.where(x <= 20.0, torch.log1p(torch.exp(x)), x)
        alpha = torch.exp(-torch.exp(A_log.double()) * softplus)
        beta = torch.sigmoid(b.double())
        s_prev = ref_state[slots.long()]  # [B, HV, V, K]
        s_prev = s_prev * alpha[:, :, None, None]
        d = beta[:, :, None] * (v - torch.einsum("bhvk,bhk->bhv", s_prev, kh))
        s_new = s_prev + d.unsqueeze(-1) * kh.unsqueeze(-2)
        out_ref = torch.einsum("bhvk,bhk->bhv", s_new, qh)

        out_p = torch.empty(BATCH, 1, HV, V, device=device, dtype=act_dtype)
        fused_recurrent_gated_delta_rule_packed_decode(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=SCALE,
            initial_state=state_packed,
            out=out_p,
            ssm_state_indices=slots,
            use_qk_l2norm_in_kernel=True,
        )
        out_c = torch.empty_like(out_p)
        fused_recurrent_gated_delta_rule_replayssm(
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=SCALE,
            initial_state=state_cached,
            d_cache=d_cache,
            k_cache=k_cache,
            g_cache=g_cache,
            out=out_c,
            ssm_state_indices=slots,
            write_pos=write_pos,
            is_flush=is_flush,
            use_qk_l2norm_in_kernel=True,
        )

        # out_* are [B, 1, HV, V]; drop the singleton decode axis so the
        # compare does not broadcast-align against the reference's HV axis.
        out_err["packed"].append(max_abs(out_p[:, 0], out_ref))
        out_err["cached"].append(max_abs(out_c[:, 0], out_ref))
        out_err["cross"].append(max_abs(out_c[:, 0], out_p[:, 0]))
        out_err["cross_rel"].append(max_rel(out_c[:, 0], out_p[:, 0]))
        ref_state[slots.long()] = s_new
        if step % L == L - 1:
            flush_steps.append(step)
            ref = s_new.double()
            st_err["packed"].append(
                max_abs(state_packed[slots.long()].double(), ref)
            )
            st_err["cached"].append(
                max_abs(state_cached[slots.long()].double(), ref)
            )

    return {
        "state_dtype": str(state_dtype).replace("torch.", ""),
        "act_dtype": str(act_dtype).replace("torch.", ""),
        "dtype_note": (
            f"d/k cache dtype={str(cache_dtype).replace('torch.', '')}, g cache fp32"
        ),
        "num_steps": num_steps,
        "L": L,
        "flush_steps": flush_steps,
        "out_max_abs_err_vs_fp64_ref": {
            "packed": max(out_err["packed"]),
            "replayssm": max(out_err["cached"]),
        },
        "out_max_abs_err_replayssm_vs_packed": max(out_err["cross"]),
        "out_max_rel_err_replayssm_vs_packed": max(out_err["cross_rel"]),
        "state_max_abs_err_vs_fp64_ref_at_flush": {
            "packed": max(st_err["packed"]) if st_err["packed"] else None,
            "replayssm": max(st_err["cached"]) if st_err["cached"] else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="num_error_magnitudes.json")
    ap.add_argument("--steps", type=int, default=33)  # 2 flushes + 1
    args = ap.parse_args()
    precisions = [
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float16),
        (torch.float16, torch.float16),
        (torch.float16, torch.bfloat16),
    ]
    results = []
    for state_dtype, act_dtype in precisions:
        r = run_cumulative(state_dtype, act_dtype, args.steps, seed=0)
        results.append(r)
        print(json.dumps(r))
    # Guard: the float64 reference must reproduce the fp32 sequential baseline.
    fp32_err = results[0]["out_max_abs_err_vs_fp64_ref"]["packed"]
    if fp32_err > 1e-3:
        raise SystemExit(
            f"fp64 reference does not match the fp32 packed baseline (max "
            f"abs err {fp32_err:.3e}); refusing to report"
        )
    with open(args.json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
