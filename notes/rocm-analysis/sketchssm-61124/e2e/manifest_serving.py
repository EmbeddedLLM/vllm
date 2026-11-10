#!/usr/bin/env python3
"""Write MANIFEST.md: provenance for every serving_bench_*.json in this directory.

The seed, cache state and engine log are not stored in the client JSON, so they
are recorded here. Without this, the suffix-free files look canonical while some
of them are the discarded warm-cache runs.
"""

import json
from pathlib import Path

EW = Path(__file__).resolve().parent

# file -> (seed, prefix cache state observed server-side, server log, role)
PROVENANCE = {
    "serving_bench_baseline_c256_cold1.json": (
        1234, "cold, 0.4%", "server_baseline_cold1.log", "HEADLINE c=256 baseline rep 1"),
    "serving_bench_baseline_c256_cold2.json": (
        2024, "cold, 0.0%", "server_baseline_cold2.log", "HEADLINE c=256 baseline rep 2"),
    "serving_bench_sketchssm_c256.json": (
        1234, "0.0%", "server_sketchssm_serve3.log", "HEADLINE c=256 sketchssm rep 1"),
    "serving_bench_sketchssm_c256_cold.json": (
        2024, "0.0%", "server_sketchssm_serve3.log", "HEADLINE c=256 sketchssm rep 2"),
    "serving_bench_baseline_c32_cold3.json": (
        1234, "cold, 0.0-0.4%", "server_baseline_cold3.log", "HEADLINE c=32 baseline rep 1"),
    "serving_bench_baseline_c32_cold4.json": (
        2024, "cold, 0.0-0.4%", "server_baseline_cold4.log", "HEADLINE c=32 baseline rep 2"),
    "serving_bench_sketchssm_c32.json": (
        1234, "0.0%", "server_sketchssm_serve3.log", "HEADLINE c=32 sketchssm rep 1"),
    "serving_bench_sketchssm_c32_cold.json": (
        2024, "0.0%", "server_sketchssm_serve3.log", "HEADLINE c=32 sketchssm rep 2"),
    "serving_bench_baseline_c256_greedy.json": (
        1234, "cold", "server_baseline_cold5.log", "cross-check: --temperature=0"),
    "serving_bench_sketchssm_c256_greedy.json": (
        1234, "0.0%", "server_sketchssm_serve3.log", "cross-check: --temperature=0"),
    "serving_bench_baseline_c256.json": (
        1234, "cold, 0.4%", "server_baseline_serve3.log", "first attempt, superseded by cold1"),
    "serving_bench_baseline_c256_coldA.json": (
        777, "near-cold", "server_baseline_serve3.log", "corroborating, superseded"),
    "serving_bench_sketchssm_c256_coldA.json": (
        777, "0.0%", "server_sketchssm_serve3.log", "corroborating"),
    "serving_bench_baseline_c32.json": (
        1234, "warm, 19.4%", "server_baseline_serve3.log",
        "DO NOT USE: warm cache bias"),
    "serving_bench_baseline_c32_r2.json": (
        1234, "warm, 55.3%", "server_baseline_serve3.log",
        "DO NOT USE: warm cache bias, kept as evidence"),
    "serving_bench_baseline_c256_r2.json": (
        1234, "warm, 63.7%", "server_baseline_serve3.log",
        "DO NOT USE: warm cache bias, kept as evidence"),
    "serving_bench_baseline_c32_coldA.json": (
        4321, "decaying from warm", "server_baseline_serve3.log",
        "DO NOT USE: cache state unclear, superseded by cold3"),
    "serving_bench_sketchssm_c32_coldA.json": (
        4321, "0.0%", "server_sketchssm_serve3.log", "corroborating"),
    "serving_bench_sketchssm_c256_r2.json": (
        1234, "0.0%", "server_sketchssm_serve3.log",
        "repeat of identical prompts: shows sketchssm gets no cache benefit"),
    "serving_bench_sketchssm_c32_r2.json": (
        1234, "0.0%", "server_sketchssm_serve3.log",
        "repeat of identical prompts: shows sketchssm gets no cache benefit"),
}


def main():
    rows = []
    for fname, (seed, cache, srvlog, role) in PROVENANCE.items():
        path = EW / fname
        if not path.exists():
            print(f"missing {fname}")
            continue
        d = json.loads(path.read_text())
        rows.append((d["date"], fname, d["max_concurrency"], seed, cache, srvlog, role,
                     d["request_throughput"], d["p50_ttft_ms"], d["p50_tpot_ms"]))
    rows.sort()

    out = [
        "# Manifest of serving benchmark runs",
        "",
        "Provenance for every `serving_bench_*.json` in this directory. The seed and the",
        "server-side prefix-cache state are not stored in the client JSON; they are",
        "transcribed here from the commands and the server logs. `SERVING_ROCM.md` uses only",
        "the rows marked HEADLINE.",
        "",
        "| date | json | c | seed | prefix cache state | server log | role | req/s | TTFT p50 | TPOT p50 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for (date, fname, conc, seed, cache, srvlog, role, rps, ttft, tpot) in rows:
        out.append(f"| {date} | `{fname}` | {conc} | {seed} | {cache} | `{srvlog}` | {role} "
                   f"| {rps:.2f} | {ttft:.1f} | {tpot:.2f} |")
    (EW / "MANIFEST.md").write_text("\n".join(out) + "\n")
    print(f"wrote MANIFEST.md with {len(rows)} runs")


if __name__ == "__main__":
    main()
