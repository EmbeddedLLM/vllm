#!/usr/bin/env python3
"""Build SERVING_LOWCONC_ROCM.md tables from the low-concurrency run artifacts.

Each run produced two files:
  serving_bench_<mode>_c<conc>_<rep>.json         --save-result client JSON
  serving_bench_<mode>_c<conc>_<rep>.cache.json   run-scoped prefix-cache evidence
                                                  (/metrics counter deltas around the
                                                  run + server-log lines in that window)

The server-log lines are re-read from the stored (server_log, line range) pair and
kept only when their timestamp falls inside the run window, because stdout to a file
is block-buffered: lines written during an earlier run can land in the file while a
later run is in flight. The /metrics delta is the authoritative per-run cache number.
"""

import json
import re
from pathlib import Path

EW = Path(__file__).resolve().parent
CONCS = (1, 2, 4, 8)
REPS = ("rep1", "rep2")

# (mode, conc, rep) -> seed used for that run
SEEDS = {
    ("baseline", 1, "rep1"): 41001, ("sketchssm", 1, "rep1"): 41002,
    ("baseline", 2, "rep1"): 41003, ("sketchssm", 2, "rep1"): 41004,
    ("baseline", 4, "rep1"): 41005, ("sketchssm", 4, "rep1"): 41006,
    ("baseline", 8, "rep1"): 41007, ("sketchssm", 8, "rep1"): 41008,
    ("baseline", 1, "rep2"): 42001, ("sketchssm", 1, "rep2"): 42002,
    ("baseline", 2, "rep2"): 42003, ("sketchssm", 2, "rep2"): 42004,
    ("baseline", 4, "rep2"): 42005, ("sketchssm", 4, "rep2"): 42006,
    ("baseline", 8, "rep2"): 42007, ("sketchssm", 8, "rep2"): 42008,
}

TIMESTAMP = re.compile(r"\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
OUT_LEN = 256


def stem(mode, conc, rep):
    return f"serving_bench_{mode}_c{conc}_{rep}"


def load(mode, conc, rep):
    bench = json.loads((EW / f"{stem(mode, conc, rep)}.json").read_text())
    cache = json.loads((EW / f"{stem(mode, conc, rep)}.cache.json").read_text())
    assert bench["num_prompts"] == 64, bench["num_prompts"]
    assert bench["failed"] == 0, bench["failed"]
    assert bench["completed"] == 64, bench["completed"]
    assert bench["max_concurrency"] == conc, bench["max_concurrency"]
    assert cache["seed"] == SEEDS[(mode, conc, rep)], (cache["seed"], mode, conc, rep)
    assert cache["metrics_delta"]["queries"] == 64 * 1024, cache["metrics_delta"]
    return bench, cache, log_evidence(cache)


_STATS_CACHE: dict[str, list[tuple[str, str]]] = {}


def stat_lines(server_log):
    """Every engine-stat line of a server log as (MM-DD HH:MM:SS, hit rate)."""
    if server_log not in _STATS_CACHE:
        out = []
        with open(server_log, errors="replace") as f:
            for line in f:
                if "Prefix cache hit rate" not in line:
                    continue
                m = TIMESTAMP.search(line)
                if m:
                    rate = line.rsplit("Prefix cache hit rate:", 1)[-1].strip()
                    out.append((m.group(0), rate))
        _STATS_CACHE[server_log] = out
    return _STATS_CACHE[server_log]


def log_evidence(cache):
    """Engine-stat lines bracketing the run: the last one printed at or before the
    run started and the first one printed at/after it ended.

    The printed rate is a cumulative average since engine start, computed over a
    10 s interval, so a single line cannot isolate one run. It is reported because
    the task asks for it; the run-scoped number is the /metrics delta.
    """
    lo, hi = cache["run_start"], cache["run_end"]  # YYYY-MM-DDTHH:MM:SS
    lo_t = f"{lo[5:7]}-{lo[8:10]} {lo[11:19]}"
    hi_t = f"{hi[5:7]}-{hi[8:10]} {hi[11:19]}"
    stats = stat_lines(cache["server_log"])
    before = [s for s in stats if s[0] <= lo_t]
    after = [s for s in stats if s[0] >= hi_t]
    return (before[-1] if before else None), (after[0] if after else None)


def num(x, nd=2):
    return f"{x:.{nd}f}"


def prefill_share(d):
    """Derived: fraction of one request's wall time spent before first token.

    vLLM reports TPOT = (e2e - TTFT) / (output_len - 1), so one request occupies
    TTFT + (output_len - 1) * TPOT. Uses the JSON *mean* values.
    """
    per_req = d["mean_ttft_ms"] + (OUT_LEN - 1) * d["mean_tpot_ms"]
    return d["mean_ttft_ms"] / per_req, per_req


def main():
    data = {(m, c, r): load(m, c, r) for c in CONCS for r in REPS for m in
            ("baseline", "sketchssm")}

    print("## Per-run values\n")
    print("| mode | conc | rep | seed | duration (s) | req/s | out tok/s | TTFT mean | "
          "TTFT p50 | TTFT p99 | TPOT mean | TPOT p50 | TPOT p99 | ITL p50 | ITL p99 | "
          "run prefix-cache hit rate |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in CONCS:
        for r in REPS:
            for m in ("baseline", "sketchssm"):
                b, cache, _ = data[(m, c, r)]
                hit = cache["run_prefix_cache_hit_rate_pct"]
                print(f"| {m} | {c} | {r} | {SEEDS[(m, c, r)]} | {num(b['duration'])} | "
                      f"{num(b['request_throughput'])} | {num(b['output_throughput'], 1)} | "
                      f"{num(b['mean_ttft_ms'], 1)} | {num(b['p50_ttft_ms'], 1)} | "
                      f"{num(b['p99_ttft_ms'], 1)} | {num(b['mean_tpot_ms'])} | "
                      f"{num(b['p50_tpot_ms'])} | {num(b['p99_tpot_ms'])} | "
                      f"{num(b['p50_itl_ms'])} | {num(b['p99_itl_ms'])} | "
                      f"{num(hit, 3)}% |")

    print("\n## Baseline vs SketchSSM per concurrency (2 reps, arithmetic mean)\n")
    for c in CONCS:
        print(f"\n### concurrency {c}\n")
        print("| metric | baseline | sketchssm | sketchssm vs baseline "
              "(>1 = sketchssm better) |")
        print("|---|---|---|---|")
        rows = [
            ("request throughput (req/s)", "request_throughput", 2, True),
            ("output token throughput (tok/s)", "output_throughput", 1, True),
            ("TTFT mean (ms)", "mean_ttft_ms", 1, False),
            ("TTFT p50 (ms)", "p50_ttft_ms", 1, False),
            ("TTFT p99 (ms)", "p99_ttft_ms", 1, False),
            ("TPOT mean (ms)", "mean_tpot_ms", 2, False),
            ("TPOT p50 (ms)", "p50_tpot_ms", 2, False),
            ("TPOT p99 (ms)", "p99_tpot_ms", 2, False),
            ("ITL p50 (ms)", "p50_itl_ms", 2, False),
            ("ITL p99 (ms)", "p99_itl_ms", 2, False),
            ("benchmark duration (s)", "duration", 2, False),
        ]
        for label, field, nd, higher_better in rows:
            b = sum(data[("baseline", c, r)][0][field] for r in REPS) / 2
            s = sum(data[("sketchssm", c, r)][0][field] for r in REPS) / 2
            ratio = (s / b) if higher_better else (b / s)
            print(f"| {label} | {num(b, nd)} | {num(s, nd)} | {num(ratio)}x |")
        for m in ("baseline", "sketchssm"):
            share = [prefill_share(data[(m, c, r)][0]) for r in REPS]
            per_req = sum(p for _, p in share) / 2
            frac = sum(f for f, _ in share) / 2
            durs = [data[(m, c, r)][0]["duration"] for r in REPS]
            print(f"\n{m}: derived per-request wall = mean_TTFT + 255*mean_TPOT = "
                  f"{num(per_req)} ms -> prefill share {num(100 * frac, 1)}%; "
                  f"64 requests / concurrency {c} predicts {num(64 * per_req / c / 1000)} s, "
                  f"measured {num(sum(durs) / 2)} s")

    print("\n## Prefix-cache evidence per run\n")
    print("| mode | conc | rep | seed | /metrics queries | /metrics hits | run hit rate | "
          "log line before run | log line after run |")
    print("|---|---|---|---|---|---|---|---|---|")
    for c in CONCS:
        for r in REPS:
            for m in ("baseline", "sketchssm"):
                _, cache, (before, after) = data[(m, c, r)]
                d = cache["metrics_delta"]
                print(f"| {m} | {c} | {r} | {cache['seed']} | {d['queries']} | {d['hits']} | "
                      f"{num(cache['run_prefix_cache_hit_rate_pct'], 3)}% | "
                      f"{before[0] if before else '-'}: {before[1] if before else '-'} | "
                      f"{after[0] if after else '-'}: {after[1] if after else '-'} |")

    print("\n## Warm-cache control (same seed re-sent; NOT part of the tables)\n")
    wb = json.loads((EW / "serving_bench_baseline_c8_warmprobe.json").read_text())
    wc = json.loads((EW / "serving_bench_baseline_c8_warmprobe.cache.json").read_text())
    cold = data[("baseline", 8, "rep1")][0]
    print(f"baseline c=8, seed 41007 re-run right after the cold run of the same seed: "
          f"run prefix-cache hit rate {num(wc['run_prefix_cache_hit_rate_pct'], 3)}% "
          f"({wc['metrics_delta']['hits']}/{wc['metrics_delta']['queries']} tokens), "
          f"TTFT p50 {num(wb['p50_ttft_ms'], 1)} ms vs {num(cold['p50_ttft_ms'], 1)} ms cold, "
          f"req/s {num(wb['request_throughput'])} vs {num(cold['request_throughput'])}")


if __name__ == "__main__":
    main()
