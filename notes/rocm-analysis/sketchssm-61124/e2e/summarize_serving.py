#!/usr/bin/env python3
"""Build the SERVING_ROCM.md tables from the saved `vllm bench serve` JSON files.

Every row is one benchmark run: the client JSON records the measured values, and
the seed / server-log evidence is carried here because the JSON does not store
`--seed` or the server-side prefix-cache hit rate.
"""

import json
from pathlib import Path

EW = Path(__file__).resolve().parent

# (mode, concurrency, rep) -> (json file, seed, server-side prefix cache hit rate
# printed by the server during that run, server log)
RUNS = {
    ("baseline", 256, 1): (
        "serving_bench_baseline_c256_cold1.json", 1234, "0.4%", "server_baseline_cold1.log"),
    ("baseline", 256, 2): (
        "serving_bench_baseline_c256_cold2.json", 2024, "0.0%", "server_baseline_cold2.log"),
    ("sketchssm", 256, 1): (
        "serving_bench_sketchssm_c256.json", 1234, "0.0%", "server_sketchssm_serve3.log"),
    ("sketchssm", 256, 2): (
        "serving_bench_sketchssm_c256_cold.json", 2024, "0.0%", "server_sketchssm_serve3.log"),
    ("baseline", 32, 1): (
        "serving_bench_baseline_c32_cold3.json", 1234, "0.0-0.4%", "server_baseline_cold3.log"),
    ("baseline", 32, 2): (
        "serving_bench_baseline_c32_cold4.json", 2024, "0.0-0.4%", "server_baseline_cold4.log"),
    ("sketchssm", 32, 1): (
        "serving_bench_sketchssm_c32.json", 1234, "0.0%", "server_sketchssm_serve3.log"),
    ("sketchssm", 32, 2): (
        "serving_bench_sketchssm_c32_cold.json", 2024, "0.0%", "server_sketchssm_serve3.log"),
}

# Contaminated comparison kept as evidence, not part of the headline tables:
# these re-send prompts an earlier run already cached.
WARM = {
    ("baseline", 256, "warm cache"): "serving_bench_baseline_c256_r2.json",
    ("baseline", 32, "warm cache"): "serving_bench_baseline_c32_r2.json",
    ("sketchssm", 256, "warm cache"): "serving_bench_sketchssm_c256_r2.json",
    ("sketchssm", 32, "warm cache"): "serving_bench_sketchssm_c32_r2.json",
}

METRICS = [
    ("request_throughput", "req/s", 2, "throughput"),
    ("output_throughput", "out tok/s", 1, "throughput"),
    ("total_token_throughput", "total tok/s", 1, "throughput"),
    ("p50_ttft_ms", "TTFT p50 (ms)", 1, "latency"),
    ("p99_ttft_ms", "TTFT p99 (ms)", 1, "latency"),
    ("p50_tpot_ms", "TPOT p50 (ms)", 2, "latency"),
    ("p99_tpot_ms", "TPOT p99 (ms)", 2, "latency"),
    ("p50_itl_ms", "ITL p50 (ms)", 2, "latency"),
    ("p99_itl_ms", "ITL p99 (ms)", 2, "latency"),
]


def load(name):
    with open(EW / name) as f:
        d = json.load(f)
    assert d["num_prompts"] == 256, (name, d["num_prompts"])
    assert d["failed"] == 0, (name, d["failed"])
    assert str(d["request_rate"]) == "inf", (name, d["request_rate"])
    assert d["total_output_tokens"] == 65536, (name, d["total_output_tokens"])
    return d


def main():
    data = {}
    for key, (fname, seed, hits, srvlog) in RUNS.items():
        data[key] = (load(fname), fname, seed, hits, srvlog)

    for conc in (256, 32):
        print(f"\n### concurrency {conc} (cold prefix cache)\n")
        hdr = ("| metric | baseline | sketchssm | sketchssm vs baseline "
               "(>1 = sketchssm better) |")
        print(hdr)
        print("|---|---|---|---|")
        for field, label, nd, direction in METRICS:
            b = [data[("baseline", conc, r)][0][field] for r in (1, 2)]
            s = [data[("sketchssm", conc, r)][0][field] for r in (1, 2)]
            bm, sm = sum(b) / 2, sum(s) / 2
            # throughput: higher is better; latency: lower is better
            ratio = sm / bm if direction == "throughput" else bm / sm
            print(f"| {label} | {bm:.{nd}f} | {sm:.{nd}f} | {ratio:.2f}x |")
        print("\nper-run values and provenance:\n")
        print("| mode | rep | seed | server-side prefix cache hit rate | req/s | out tok/s "
              "| TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | json |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for mode in ("baseline", "sketchssm"):
            for rep in (1, 2):
                d, fname, seed, hits, srvlog = data[(mode, conc, rep)]
                print(f"| {mode} | {rep} | {seed} | {hits} | {d['request_throughput']:.2f} "
                      f"| {d['output_throughput']:.1f} | {d['p50_ttft_ms']:.1f} "
                      f"| {d['p99_ttft_ms']:.1f} | {d['p50_tpot_ms']:.2f} "
                      f"| {d['p99_tpot_ms']:.2f} | {d['p50_itl_ms']:.2f} "
                      f"| {d['p99_itl_ms']:.2f} | {fname} |")

    print("\n### warm-cache runs (identical prompts re-sent; NOT apples-to-apples)\n")
    print("| mode | concurrency | req/s | out tok/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 | json |")
    print("|---|---|---|---|---|---|---|---|---|")
    for (mode, conc, tag), fname in WARM.items():
        d = load(fname)
        print(f"| {mode} | {conc} | {d['request_throughput']:.2f} | {d['output_throughput']:.1f} "
              f"| {d['p50_ttft_ms']:.1f} | {d['p99_ttft_ms']:.1f} | {d['p50_tpot_ms']:.2f} "
              f"| {d['p99_tpot_ms']:.2f} | {fname} |")

    print("\nruns and their timestamps:")
    for key, (d, fname, seed, hits, srvlog) in data.items():
        print(f"  {key}: {d['date']}  {fname}  (log {srvlog})")


if __name__ == "__main__":
    main()
