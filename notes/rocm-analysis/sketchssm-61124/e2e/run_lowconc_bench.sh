#!/bin/bash
# One low-concurrency serving measurement against an already-running server.
# Usage: run_lowconc_bench.sh <baseline|sketchssm> <concurrency> <rep> <seed> [num-prompts]
#
# Records:
#   serving_bench_<mode>_c<conc>_<rep>.json           (--save-result)
#   serving_bench_<mode>_c<conc>_<rep>.console.log    (verbatim client output)
#   serving_bench_<mode>_c<conc>_<rep>.cache.json     (per-run prefix-cache evidence:
#       /metrics counter deltas around the run + the server-log "Prefix cache hit rate"
#       lines emitted during the run + the file offsets they came from)
# A fresh --seed per run keeps the random dataset unique, so neither server can serve
# prompts an earlier run already cached.
set -u
WT=/app/local/yh/sketchssm-e2e-wt
EW=/app/local/yh/issue-workdirs/pr61124-sketchssm-rocm/e2e
MODE=$1; CONC=$2; REP=$3; SEED=$4; NP=${5:-64}
if [ "$MODE" = "baseline" ]; then
  PORT=8056; SRVLOG=$EW/server_baseline_cold5.log
else
  PORT=8057; SRVLOG=$EW/server_sketchssm_serve3.log
fi
TAG=c${CONC}_${REP}

metrics() {  # emit the counters we need as k=v lines
  curl -s "http://127.0.0.1:$PORT/metrics" | awk '
    /^vllm:prefix_cache_queries_total/ {q=$2}
    /^vllm:prefix_cache_hits_total/    {h=$2}
    /^vllm:prompt_tokens_total/        {p=$2}
    /^vllm:generation_tokens_total/    {g=$2}
    /^vllm:request_success_total.*finished_reason="length"/ {s+=$2}
    END {printf "queries=%d hits=%d prompt_tokens=%d generation_tokens=%d success_length=%d\n", q, h, p, g, s}'
}

BEFORE_LINE=$(wc -l < "$SRVLOG")
BEFORE_MET=$(metrics)
TS_START=$(date +%FT%T)

cd "$WT" || exit 1
PYTHONPATH=$WT timeout 1800 python3 -m vllm.entrypoints.cli.main bench serve \
  --backend openai --endpoint /v1/completions --base-url http://127.0.0.1:$PORT \
  --model nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 --served-model-name nemotron --trust-remote-code \
  --dataset-name random --random-input-len 1024 --random-output-len 256 \
  --ignore-eos --num-prompts "$NP" --max-concurrency "$CONC" \
  --percentile-metrics ttft,tpot,itl --metric-percentiles 50,99 --seed "$SEED" \
  --save-result --result-dir "$EW" --result-filename "serving_bench_${MODE}_${TAG}.json" \
  2>&1 | tee "$EW/serving_bench_${MODE}_${TAG}.console.log" \
  | grep -E "Successful|Failed|Benchmark duration|Request throughput|Output token through|Peak output|Total token through|Mean TTFT|P50 TTFT|P99 TTFT|Mean TPOT|P50 TPOT|P99 TPOT|Mean ITL|P50 ITL|P99 ITL"

AFTER_MET=$(metrics)
TS_END=$(date +%FT%T)
sleep 2  # let the 10 s engine logger flush a post-run line
AFTER_LINE=$(wc -l < "$SRVLOG")

python3 - "$EW/serving_bench_${MODE}_${TAG}.cache.json" "$MODE" "$CONC" "$REP" "$SEED" "$NP" \
  "$PORT" "$BEFORE_MET" "$AFTER_MET" "$SRVLOG" "$BEFORE_LINE" "$AFTER_LINE" \
  "$TS_START" "$TS_END" <<'PY'
import json, sys
out, mode, conc, rep, seed, np, port, before, after, srvlog, l0, l1, ts0, ts1 = sys.argv[1:]
def parse(s):
    return {k: int(v) for k, v in (kv.split("=") for kv in s.split())}
b, a = parse(before), parse(after)
d = {k: a[k] - b[k] for k in a}
q, h = d["queries"], d["hits"]
lines = []
TIMESTAMP = __import__("re").compile(r"\d{2}-\d{2} \d{2}:\d{2}:\d{2}")
with open(srvlog, errors="replace") as f:
    for i, line in enumerate(f, 1):
        if int(l0) < i <= int(l1) and "Prefix cache hit rate" in line:
            m = TIMESTAMP.search(line)
            lines.append({"log_line": i, "text": line.split("] ", 2)[-1].strip(),
                          "timestamp": m.group(0) if m else None})
json.dump({"mode": mode, "concurrency": int(conc), "rep": rep, "seed": int(seed),
           "num_prompts": int(np), "port": int(port),
           "run_start": ts0, "run_end": ts1,
           "metrics_before": b, "metrics_after": a, "metrics_delta": d,
           "run_prefix_cache_hit_rate_pct": (100.0 * h / q) if q else None,
           "run_prefix_cache_hits": h, "run_prefix_cache_queries": q,
           "server_log": srvlog, "server_log_lines": [int(l0), int(l1)],
           "server_log_hit_rate_lines": lines}, open(out, "w"), indent=1)
print(f"\n--- run-scoped prefix cache (from /metrics counter deltas, {mode} {rep} seed={seed}) ---")
print(f"queries={q} hits={h} -> run hit rate {(100.0*h/q) if q else 0.0:.3f}%")
print(f"cumulative before: {100.0*b['hits']/max(b['queries'],1):.3f}% "
      f"({b['hits']}/{b['queries']})  after: {100.0*a['hits']/max(a['queries'],1):.3f}% "
      f"({a['hits']}/{a['queries']})")
for line in lines:
    print("server log:", line["timestamp"], "|", line["text"])
PY
