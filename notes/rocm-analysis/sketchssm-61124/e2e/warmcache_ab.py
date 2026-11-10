#!/usr/bin/env python3
"""Does the SketchSSM server serve from the prefix cache at all?

For one already-running server:
  A. send the SAME >=1024-token prompt 4x consecutively (warm-cache control)
  B. run the seed-777 random bench twice in a row (run 2 is warm by construction)
Reporting vllm:prefix_cache_{queries,hits}_total deltas around every step.

Usage: warmcache_ab.py <baseline|sketchssm>
"""
import json
import subprocess
import sys
import urllib.request

EW = "/app/local/yh/issue-workdirs/pr61124-sketchssm-rocm/e2e"
WT = "/app/local/yh/sketchssm-e2e-wt"
MODE = sys.argv[1]
PORT = {"baseline": 8056, "sketchssm": 8057}[MODE]
SRVLOG = {"baseline": f"{EW}/server_baseline_cold5.log",
          "sketchssm": f"{EW}/server_sketchssm_serve3.log"}[MODE]

PARA = (
    "The prefix cache stores the key and value tensors of every processed prompt "
    "block so that a later request sharing that prefix can skip the prefill. In a "
    "hybrid model the recurrent state of the mamba layers is not a plain tensor, so "
    "the block boundaries have to respect the mamba page size, and the cache has to "
    "restore the recurrent checkpoint alongside the attention pages. Whether this "
    "mechanism is exercised at all is exactly what the counters below answer. "
)
PROMPT = (PARA * 30).strip()  # ~2k tokens


def counters():
    txt = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=30).read().decode()
    out = {}
    for line in txt.splitlines():
        if line.startswith("#"):
            continue
        for name in ("vllm:prefix_cache_queries_total",
                     "vllm:prefix_cache_hits_total"):
            if line.startswith(name):
                out[name] = float(line.split()[1])
    return out


def complete():
    body = json.dumps({"model": "nemotron", "prompt": PROMPT, "temperature": 0.0,
                       "max_tokens": 64, "ignore_eos": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/completions", data=body,
                                headers={"Content-Type": "application/json"})
    t0 = __import__("time").time()
    resp = json.loads(urllib.request.urlopen(req, timeout=600).read())
    return {"elapsed_s": round(__import__("time").time() - t0, 3),
            "prompt_tokens": resp["usage"]["prompt_tokens"],
            "output_tokens": resp["usage"]["completion_tokens"]}


def bench(tag):
    cmd = ["python3", "-m", "vllm.entrypoints.cli.main", "bench", "serve",
           "--backend", "openai", "--endpoint", "/v1/completions",
           "--base-url", f"http://127.0.0.1:{PORT}",
           "--model", "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
           "--served-model-name", "nemotron", "--trust-remote-code",
           "--dataset-name", "random", "--random-input-len", "1024",
           "--random-output-len", "64", "--num-prompts", "32",
           "--max-concurrency", "8", "--ignore-eos", "--seed", "777",
           "--save-result", "--result-dir", EW,
           "--result-filename", f"serving_bench_{MODE}_warmcache_{tag}.json"]
    console = f"{EW}/serving_bench_{MODE}_warmcache_{tag}.console.log"
    with open(console, "w") as fh:
        subprocess.run(cmd, cwd=WT, stdout=fh, stderr=subprocess.STDOUT,
                       env={**__import__("os").environ, "PYTHONPATH": WT}, timeout=1800)
    for line in open(console, errors="replace"):
        if "Successful requests" in line or "Benchmark duration" in line:
            print("   ", line.strip())


def step(label, fn):
    b = counters()
    extra = fn()
    a = counters()
    dq = a["vllm:prefix_cache_queries_total"] - b["vllm:prefix_cache_queries_total"]
    dh = a["vllm:prefix_cache_hits_total"] - b["vllm:prefix_cache_hits_total"]
    rec = {"case": label, "queries_delta": int(dq), "hits_delta": int(dh),
           "hit_rate": (dh / dq) if dq else None, "extra": extra}
    print(f"  {label:24s} hits={rec['hits_delta']:8d} queries={rec['queries_delta']:8d} "
          f"rate={(100.0*dh/dq) if dq else float('nan'):7.3f}%  {extra}")
    return rec


def main():
    print(f"=== {MODE} (port {PORT}) ===")
    start = counters()
    print("  cumulative at start:",
          {k.replace('vllm:prefix_cache_', '').replace('_total', ''): int(v)
           for k, v in start.items()})
    rows = []

    def one():
        return complete()

    steps = [step(f"single-prompt repeat {i}/4", one) for i in range(1, 5)]
    rows += steps
    # Pooled over the 4 repeats as well as per-repeat.
    total = {"case": "single-prompt repeat x4 (pooled)",
             "queries_delta": sum(s["queries_delta"] for s in steps),
             "hits_delta": sum(s["hits_delta"] for s in steps)}
    total["hit_rate"] = (total["hits_delta"] / total["queries_delta"]
                         if total["queries_delta"] else None)
    print(f"  {'POOLED x4':24s} hits={total['hits_delta']:8d} "
          f"queries={total['queries_delta']:8d} "
          f"rate={(100.0*total['hit_rate']) if total['hit_rate'] is not None else float('nan'):7.3f}%")

    rows.append(step("seed777 run 1", lambda: bench("run1")))
    rows.append(step("seed777 run 2", lambda: bench("run2")))
    rows.append(total)
    end = counters()

    out = {"mode": MODE, "port": PORT, "server_log": SRVLOG,
           "prompt_chars": len(PROMPT), "rows": rows,
           "cumulative_start": start, "cumulative_end": end}
    path = f"{EW}/warmcache_ab_{MODE}.json"
    json.dump(out, open(path, "w"), indent=1)
    print("  wrote", path)


if __name__ == "__main__":
    main()
