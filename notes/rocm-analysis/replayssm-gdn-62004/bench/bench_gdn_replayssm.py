# SPDX-License-Identifier: Apache-2.0
"""Kernel-level latency: GDN ReplaySSM cached decode vs the baseline packed decode.

Both kernels are launched from inside a torch CUDA graph (HIP graph on ROCm) and
timed by replaying the graph, because an eager Triton launch from Python is
CPU-launch-bound on this box and would measure the host, not the GPU.

Geometry is Qwen3.5-4B's GDN layer: H=16 key heads, HV=32 value heads,
K=V=128, ring buffer L=16. Two cache configurations are measured:

  * "bf16-model": bf16 activations -> fp16 d_cache/k_cache, fp32 g_cache
    (what `append_gated_delta_net_replayssm_ring` allocates for a bf16 model),
    so the tl.dot in the ReplaySSM kernel is fp16 x fp16 -> MFMA on gfx950;
  * "fp32-cache": fp32 activations + fp32 d/k caches, i.e. an fp32 tl.dot.

The per-row `write_pos`/`is_flush` pattern is the serving one: rows sit at
staggered ring positions and every L-th row is at the last slot with
`is_flush=1` (so flush work is amortized at its steady-state rate of 1/L).

Usage:
  python bench_gdn_replayssm.py                 # batch sweep, both caches
  python bench_gdn_replayssm.py --sweep-config  # + coordinate-wise launch cfg
"""

import argparse
import json
import statistics
import time

import torch

from vllm.model_executor.layers.mamba.ops import replayssm_config
from vllm.third_party.flash_linear_attention.ops import (
    fused_recurrent_gated_delta_rule_packed_decode,
    fused_recurrent_gated_delta_rule_replayssm,
)

H, HV, K, V, L = 16, 32, 128, 128, 16
QKV_DIM = 2 * H * K + HV * V
SHIPPED_NON_BLACKWELL_GDN_CONFIG = (64, 1, 3, 2)  # block_v, num_warps, num_stages, nk

CACHE_CASES = {
    # name: (act_dtype, cache_dtype)
    "bf16-model": (torch.bfloat16, torch.float16),
    "fp32-cache": (torch.float32, torch.float32),
}


def make_state(bs, cache_dtype, act_dtype, seed=0):
    """Realistic decode-step inputs; per-row ring cursor staggered + 1 flush / L."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    num_slots = bs + 1  # slot 0 is the null block
    slots = (torch.randperm(num_slots - 1, device="cuda", generator=g)[:bs] + 1).to(
        torch.int32
    )
    write_pos = torch.arange(bs, device="cuda", dtype=torch.int32) % (L - 1)
    flush_rows = torch.arange(bs, device="cuda") % L == 0
    write_pos[flush_rows] = L - 1
    is_flush = flush_rows.to(torch.int8)

    return dict(
        mixed_qkv=torch.randn(
            bs, QKV_DIM, device="cuda", dtype=act_dtype, generator=g
        ),
        a=torch.randn(bs, HV, device="cuda", dtype=act_dtype, generator=g),
        b=torch.randn(bs, HV, device="cuda", dtype=act_dtype, generator=g),
        A_log=torch.randn(HV, device="cuda", dtype=torch.float32, generator=g),
        dt_bias=torch.randn(HV, device="cuda", dtype=torch.float32, generator=g),
        initial_state=torch.randn(
            num_slots, HV, V, K, device="cuda", dtype=torch.float32, generator=g
        ),
        d_cache=torch.zeros(num_slots, HV, L, V, device="cuda", dtype=cache_dtype),
        k_cache=torch.zeros(num_slots, H, L, K, device="cuda", dtype=cache_dtype),
        g_cache=torch.zeros(num_slots, HV, L, device="cuda", dtype=torch.float32),
        out=torch.empty(bs, 1, HV, V, device="cuda", dtype=act_dtype),
        ssm_state_indices=slots,
        write_pos=write_pos,
        is_flush=is_flush,
        scale=K**-0.5,
    )


def call_packed(t, cfg=None):
    fused_recurrent_gated_delta_rule_packed_decode(
        mixed_qkv=t["mixed_qkv"],
        a=t["a"],
        b=t["b"],
        A_log=t["A_log"],
        dt_bias=t["dt_bias"],
        scale=t["scale"],
        initial_state=t["initial_state"],
        out=t["out"],
        ssm_state_indices=t["ssm_state_indices"],
        use_qk_l2norm_in_kernel=True,
    )


def call_replayssm(t, cfg=None):
    fused_recurrent_gated_delta_rule_replayssm(
        mixed_qkv=t["mixed_qkv"],
        a=t["a"],
        b=t["b"],
        A_log=t["A_log"],
        dt_bias=t["dt_bias"],
        scale=t["scale"],
        initial_state=t["initial_state"],
        d_cache=t["d_cache"],
        k_cache=t["k_cache"],
        g_cache=t["g_cache"],
        out=t["out"],
        ssm_state_indices=t["ssm_state_indices"],
        write_pos=t["write_pos"],
        is_flush=t["is_flush"],
        use_qk_l2norm_in_kernel=True,
        **({} if cfg is None else dict(zip(("block_v", "num_warps", "num_stages", "nk"), cfg))),
    )


IMPLS = {"packed_decode": call_packed, "replayssm": call_replayssm}


def capture(call, warmup=3):
    """Capture one launch (Triton JIT included) into a repluggable CUDA graph."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    torch.cuda.synchronize()
    return graph


def time_graph(graph, iters, trials=3):
    """ms per graph replay: min over trials of (100 replays / 100)."""
    per_trial = []
    for _ in range(trials):
        for _ in range(20):  # warmup replays
            graph.replay()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        cpu0 = time.perf_counter()
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.cuda.synchronize()
        cpu_dt = time.perf_counter() - cpu0
        per_trial.append(start.elapsed_time(end) / iters)
        if cpu_dt * 1e3 < start.elapsed_time(end):
            raise RuntimeError(
                f"launch loop ({cpu_dt * 1e3:.2f} ms) outran the GPU "
                f"({start.elapsed_time(end):.2f} ms): timing is CPU-bound"
            )
    return {
        "ms_per_step_min": min(per_trial),
        "ms_per_step_median": statistics.median(per_trial),
        "trials_ms": per_trial,
    }


def run_case(impl, bs, cache_name, iters, cfg=None):
    act_dtype, cache_dtype = CACHE_CASES[cache_name]
    t = make_state(bs, cache_dtype, act_dtype)
    call = lambda: IMPLS[impl](t, cfg)  # noqa: E731
    graph = capture(call)
    res = time_graph(graph, iters)
    # A long replay run must not have produced inf/nan (would change the math
    # cadence on some GPUs); check the final buffers are finite.
    finite = bool(torch.isfinite(t["out"]).all() and torch.isfinite(t["initial_state"]).all())
    res.update(
        impl=impl,
        batch=bs,
        cache=cache_name,
        act_dtype=str(act_dtype).replace("torch.", ""),
        cache_dtype=str(cache_dtype).replace("torch.", ""),
        state_dtype="float32",
        config=list(cfg) if cfg else None,
        finite_after_replay=finite,
    )
    return res


def _measure_config(cfg, sizes, cache_name, iters, rows):
    replayssm_config._overrides["gdn_decode"] = cfg
    ms = {}
    for bs in sizes:
        try:
            r = run_case("replayssm", bs, cache_name, iters)
        except Exception as exc:  # invalid tile / compile failure
            print(f"  cfg={cfg} bs={bs}: FAILED {type(exc).__name__}: {exc}")
            rows.append(
                {
                    "impl": "replayssm",
                    "batch": bs,
                    "cache": cache_name,
                    "config": list(cfg),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            ms[bs] = float("inf")
            continue
        # The launch config came from the module override, not the call site;
        # record which one was active so the row is self-describing.
        r["config"] = list(cfg)
        rows.append(r)
        ms[bs] = r["ms_per_step_min"]
        print(f"  cache={cache_name} bs={bs:4d} cfg={cfg} -> {ms[bs]:.4f} ms")
    return ms


def sweep(sizes, cache_name, iters, rounds=2):
    """Coordinate descent on the launch config, starting at the shipped default.

    Round 1 walks block_v {32,64,128}, num_warps {1,2,4}, num_stages {1,2,3},
    nk {1,2,4} one axis at a time; round 2 re-walks every axis (block_v also
    probing 16) from the round-1 winner. A config's score is the summed ms/step
    over `sizes`, so one config must win at both batch sizes.
    """
    axes = [(32, 64, 128), (1, 2, 4), (1, 2, 3), (1, 2, 4)]
    cur = list(SHIPPED_NON_BLACKWELL_GDN_CONFIG)
    rows = []
    scores = {}  # cfg -> summed ms; re-used so the walk can compare a cfg it has
    # already measured in an earlier axis (it stays a candidate).
    for rnd in range(rounds):
        for axis, values in enumerate(axes):
            if rnd > 0 and axis == 0:
                values = (16, *values)
            best = (float("inf"), cur[axis])
            for v in values:
                cfg = list(cur)
                cfg[axis] = v
                cfg = tuple(cfg)
                if cfg in scores:
                    score = scores[cfg]
                else:
                    ms = _measure_config(cfg, sizes, cache_name, iters, rows)
                    score = scores[cfg] = sum(ms.values())
                if score < best[0]:
                    best = (score, v)
            cur[axis] = best[1]
            print(f"  round {rnd} axis {axis} -> {best[1]} (cur={tuple(cur)})")
    del replayssm_config._overrides["gdn_decode"]
    return rows, tuple(cur)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 128, 256, 512])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--sweep-config", action="store_true")
    ap.add_argument("--sweep-only", action="store_true")
    ap.add_argument("--sweep-rounds", type=int, default=2)
    ap.add_argument("--sweep-batches", type=int, nargs="+", default=[256, 512])
    ap.add_argument(
        "--config",
        type=int,
        nargs=4,
        default=None,
        metavar=("BLOCK_V", "NUM_WARPS", "NUM_STAGES", "NK"),
        help="ReplaySSM launch config for the whole batch table (default: shipped)",
    )
    ap.add_argument("--json", default="bench_gdn_replayssm.json")
    ap.add_argument("--csv", default="bench_gdn_replayssm.csv")
    args = ap.parse_args()

    print(torch.cuda.get_device_name(0), torch.version.hip)
    if args.config is not None:
        replayssm_config._overrides["gdn_decode"] = tuple(args.config)
        print(f"replayssm launch config override: {tuple(args.config)}")
    rows = []
    for cache_name in [] if args.sweep_only else CACHE_CASES:
        for bs in args.batches:
            per = {}
            for impl in IMPLS:
                r = run_case(impl, bs, cache_name, args.iters)
                per[impl] = r
                rows.append(r)
                print(
                    f"cache={cache_name:11s} bs={bs:4d} {impl:14s} "
                    f"{r['ms_per_step_min']:.4f} ms/step"
                )
            speedup = (
                per["packed_decode"]["ms_per_step_min"]
                / per["replayssm"]["ms_per_step_min"]
            )
            print(f"    -> speedup {speedup:.3f}x")
            rows.append(
                {
                    "impl": "speedup",
                    "batch": bs,
                    "cache": cache_name,
                    "speedup": speedup,
                    "packed_ms": per["packed_decode"]["ms_per_step_min"],
                    "replayssm_ms": per["replayssm"]["ms_per_step_min"],
                }
            )

    sweep_best = {}
    if args.sweep_config:
        for cache_name in CACHE_CASES:
            print(f"sweep: cache={cache_name}")
            sweep_rows, best = sweep(
                args.sweep_batches, cache_name, args.iters, args.sweep_rounds
            )
            rows.extend(sweep_rows)
            # Re-measure the winner and the shipped default back to back.
            ms_best = _measure_config(best, args.sweep_batches, cache_name, args.iters, rows)
            ms_ship = _measure_config(
                SHIPPED_NON_BLACKWELL_GDN_CONFIG,
                args.sweep_batches,
                cache_name,
                args.iters,
                rows,
            )
            sweep_best[cache_name] = {
                "best_config": list(best),
                "best_ms": ms_best,
                "shipped_config": list(SHIPPED_NON_BLACKWELL_GDN_CONFIG),
                "shipped_ms": ms_ship,
            }
            print(f"  BEST cache={cache_name}: {best} {ms_best} vs shipped {ms_ship}")
        rows.append({"impl": "sweep_summary", "sweep_best": sweep_best})

    with open(args.json, "w") as f:
        json.dump(rows, f, indent=2)
    with open(args.csv, "w") as f:
        f.write("impl,batch,cache,config,ms_per_step_min,speedup\n")
        for r in rows:
            if r["impl"] == "sweep_summary":
                continue
            f.write(
                f"{r['impl']},{r.get('batch', '')},{r.get('cache', '')},"
                f"{r.get('config') or ''},{r.get('ms_per_step_min', '')},"
                f"{r.get('speedup', '')}\n"
            )
    print(f"wrote {args.json} and {args.csv}")


if __name__ == "__main__":
    main()
