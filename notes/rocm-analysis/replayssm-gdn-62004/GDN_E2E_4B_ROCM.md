# Qwen3.5-4B end-to-end: GDN ReplaySSM (PR #62004 + MRV2 spike) on ROCm

First real-model e2e number for the PR, on the tree that already decodes
(worktree `/app/local/yh/replayssm-gdn-wt`, branch `replayssm-gdn-rocm` = main
`54cb743bbf` with PR #62004 + the two MRV2 spike fixes in
`mrv2_spike_mrv2only.patch`). Predecessor: `MRV2_SPIKE.md` (Qwen3.5-0.8B,
~1.00x, config-only claim: no win to be had at 0.8B).

**Verdict: at 4B on MI355X (gfx950) ReplaySSM is a real win, 1.09x / 1.14x /
1.14x at bs 64 / 256 / 512** (output tok/s, 2 reps per cell, run-to-run spread
<=0.5%). It is a decode-side win: TPOT p50 drops 10-14% at every batch size. The
gfx950-tuned launch config `(16,1,1,2)` adds nothing measurable at model level
over the shipped `(64,1,3,2)` (<=0.6%, inside noise), which matches the 0.8B
finding and the kernel-level data.

## Setup

| item | value |
| --- | --- |
| model | `Qwen/Qwen3.5-4B`, snapshot `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| weights | downloaded fresh with `HF_HUB_CACHE=/models hf download Qwen/Qwen3.5-4B` (2 safetensors shards, 5.33 GB + 3.99 GB). The earlier snapshot on this box was config-only (29 KB); it is now complete. |
| arch | `Qwen3_5ForConditionalGeneration`, 32 layers (24 GDN, 8 full attn), 16 k-heads / 32 v-heads x 128, conv kernel 4, bf16 |
| GPU | **GPU 4 only** (`CUDA_VISIBLE_DEVICES=4`) for all three configs, one server at a time. This sweep never used GPU 5 (it was free at start and picked up by an unrelated job after the sweep); GPU 6/7 held unrelated servers throughout and were not touched, and GPU 1 carries long-running foreign processes per `rocm-smi --showpids`. |
| stack | MI355X gfx950, ROCm 7.15, torch 2.13+rocm, tree run as `PYTHONPATH=/app/local/yh/replayssm-gdn-wt` (no rebuild) |
| runner | MRV2 (`v1` LLM engine, `cudagraph_mode FULL_AND_PIECEWISE`, capture sizes to 512) |

### Server flags (common to all three configs)

```
cd /app/local/yh/replayssm-gdn-wt
PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=4 python3 -m vllm.entrypoints.cli.main serve \
  /models/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --trust-remote-code --served-model-name qwen35 \
  --max-model-len 4096 --mamba-ssm-cache-dtype float32 --mamba-cache-mode none \
  --gpu-memory-utilization 0.80 --no-enable-prefix-caching --port <8131|8132|8133>
# replayssm configs add: --use-replayssm --replayssm-buffer-len 16
```

Wrapper: `bench/gdn4b_server.sh <base|rs|rstuned> <port> <log>`.
KV cache: 13,027 blocks / 4,850,781 tokens (max concurrency for 4096 tokens =
1184x), so none of the cells below are KV-cache limited.

`--mamba-cache-mode none` is a **deprecation no-op on current main**
(`--mamba-cache-mode is deprecated ... It has no effect: Mamba prefix caching
follows --enable-prefix-caching`, logged by the server). It is passed only to
match the spike's command line. Its consequence for the PR: the unsolved
align-mode requirement is moot on main, and with `--no-enable-prefix-caching`
the mamba cache mode is `none` regardless, which is the mode the spike measured.
Prefix caching is explicitly **off** in every cell (it interacts with the
unsolved align path, per `MRV2_SPIKE.md`).

### Bench command

```
PYTHONPATH=$PWD python3 -m vllm.entrypoints.cli.main bench serve \
  --backend openai --base-url http://127.0.0.1:<port> --endpoint /v1/completions \
  --model /models/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a \
  --served-model-name qwen35 --tokenizer <same path> \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --num-prompts 2048 --max-concurrency <64|256|512> --seed <fresh per run> \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
  --save-result --result-dir bench --result-filename gdn4b_<cfg>_bs<N>_r<M>.json
```

Driver: `bench/gdn4b_bench.sh <tag> <port>` (one discarded warmup run at bs 64,
then 2 reps x [64, 256, 512]; 2048 prompts, so every cell is many waves and
`completed == 2048, failed == 0`).

## Results (mean of 2 reps; per-rep values in `bench/gdn4b_<cfg>_bs<N>_r<M>.json`)

### bs = 64

| config | out tok/s | req/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | tok/s vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline (no `--use-replayssm`) | 7820 | 61.1 | 226 | 334 | 6.46 | 7.50 | 1.000 |
| ReplaySSM, shipped `(64,1,3,2)` | 8504 | 66.4 | 229 | 335 | 5.78 | 6.86 | **1.087** |
| ReplaySSM, gfx950-tuned `(16,1,1,2)` | 8553 | 66.8 | 224 | 341 | 5.75 | 6.82 | **1.094** |

### bs = 256

| config | out tok/s | req/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | tok/s vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 11295 | 88.2 | 427 | 1319 | 19.27 | 21.51 | 1.000 |
| ReplaySSM shipped | 12815 | 100.1 | 422 | 1306 | 16.63 | 18.78 | **1.135** |
| ReplaySSM tuned | 12823 | 100.2 | 491 | 1284 | 16.11 | 18.73 | **1.135** |

### bs = 512

| config | out tok/s | req/s | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms) | TPOT p99 (ms) | tok/s vs baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 12328 | 96.3 | 488 | 2612 | 37.24 | 39.71 | 1.000 |
| ReplaySSM shipped | 14105 | 110.2 | 459 | 2595 | 32.22 | 34.31 | **1.144** |
| ReplaySSM tuned | 14168 | 110.7 | 495 | 2555 | 31.97 | 34.43 | **1.149** |

Per-rep raw values (out tok/s) and run-to-run spread, for honesty:

| cell | baseline | shipped | tuned |
| --- | --- | --- | --- |
| bs 64 | 7831 / 7809 (0.3%) | 8503 / 8505 (0.0%) | 8554 / 8551 (0.0%) |
| bs 256 | 11286 / 11304 (0.2%) | 12823 / 12807 (0.1%) | 12804 / 12842 (0.3%) |
| bs 512 | 12298 / 12359 (0.5%) | 14103 / 14108 (0.0%) | 14171 / 14164 (0.1%) |

All 9 cells measured; nothing estimated. Notes on individual cells:

* TTFT p50 at bs 256 tuned is the only visibly noisy number in the set
  (554 then 429 ms; baseline 427/426, shipped 422/421). TTFT here includes
  queueing (2048 prompts at a capped concurrency), so it tracks throughput
  rather than prefill latency; the TTFT p99 column is queue-dominated by
  construction (2.6 s at bs 512). Do not read prefill speed out of these.
* Throughput reps agree to <=0.5% within a config, so a 1.09-1.14x gap between
  configs is far outside the noise floor.
* shipped vs tuned differ by 0.6% (bs 64), 0.1% (bs 256), 0.4% (bs 512) - i.e.
  no measurable model-level difference, even though the tuned config is
  1.45-1.55x faster at kernel level for bs >= 128 (see `MRV2_SPIKE.md` and
  `logs/bench_gdn_*.json`). At model level the GDN recurrence is not the
  Amdahl-dominant term at these batch sizes; the win over baseline is already
  what ReplaySSM removes (state read/write traffic), not the launch config.

## Correctness sanity (config (b), shipped)

8 prompts x 64 greedy tokens, `temperature=0`, `logprobs=0`, same server
config as the bench, `bench/gdn4b_correctness.py`; diff with
`bench/gdn4b_compare.py`.

| run | token-identical vs baseline | max abs dlogp | mean abs dlogp |
| --- | --- | --- | --- |
| baseline vs baseline (2nd call, same server) | **8 / 8** | 0.000 | 0.000 |
| shipped ReplaySSM vs baseline | **6 / 8** | 1.744 | 0.0969 |
| tuned ReplaySSM vs baseline | 7 / 8 | 1.885 | 0.0508 |

Per-prompt (shipped vs baseline): the per-token logprob delta on positions that
agree is 0.02-0.10 nats (same magnitude as the 0.8B spike, i.e. fp16-ring
reconstruction noise; the model is bf16, the ring is fp16). Both divergences are
greedy near-tie flips ("Once upon a time, in a small village," at token 34;
"In the year 2050, ..." at token 12) after which the texts legitimately fork, so
each prompt's max dlogp is a downstream artifact of its flip, not an extra
error. The baseline-vs-baseline control is bit-exact, so the 6/8 is a real
ReplaySSM effect and **proves ReplaySSM was actually on the decode path** (see
the next section) rather than a silent no-op. 6/8 here vs 7/8 on 0.8B is prompt
sampling, not a new class of error.

## What is apples-to-apples and what is not

* **baseline vs shipped ReplaySSM: apples-to-apples.** Same tree, same server
  binary, same GPU (4), same flags except `--use-replayssm --replayssm-buffer-len
  16`; separate server starts, warmup run before each measurement set.
* **shipped vs tuned: a launch-config change only.** The only difference is
  `_gdn_decode`'s non-Blackwell return in
  `vllm/model_executor/layers/mamba/ops/replayssm_config.py`, edited in place to
  `return 16, 1, 1, 2` for this measurement and **reverted afterwards** -
  `git diff` on that file is empty and the patch does not contain it (verified
  `get_replayssm_config("gdn_decode", L=16) == (16,1,1,2)` while the server was
  up, and `(64,1,3,2)` after the revert).
* **All three cells are on one GPU, one config at a time**, so no cross-GPU
  clock/thermal confound. Servers were fully stopped between configs (GPU 4 VRAM
  back to ~0.3 GB before each start).
* **Not apples-to-apples with `MRV2_SPIKE.md`'s throughput table**: that table
  ran 128 in / 128 out with prefix caching at its default (on), this one 512 in /
  128 out with prefix caching off, and a different model. Only the shapes of the
  conclusions are comparable, not the numbers.
* **ReplaySSM is genuinely engaged on this box**: `GDN_AITER_TRITON_AVAILABLE`
  is `False` here (`rocm_aiter_ops.are_gdn_triton_kernels_available()` -> False),
  so GDN decode runs the generic `qwen_gdn_attention_core` path that consumes the
  ReplaySSM ring (`kv_cache[2:5]`), not `_forward_core_decode_aiter`. That
  also means the AITER-decode hole (`MRV2_SPIKE.md` item 5) does not affect these
  numbers - and conversely these numbers say nothing about it.
* **Not measured / out of scope**: `--mamba-cache-mode align` (moot on main, see
  above), prefix caching on, bs >= 1024, spec decode, TP > 1, and any model
  larger than 4B. The 4B is still not state-bandwidth-bound at bs 64 (the 1.09x
  there is smaller than the 1.14x at 512); a larger model or bs 1024+ should be
  expected to push the ratio up, but that is an expectation, not a measurement.

## Files

Servers: `logs/gdn4b_server_{base,rs,rstuned}.log`.
Bench logs: `logs/gdn4b_bench_<cfg>_bs<N>_r<M>.log` (+ `..._warmup_<cfg>.log`).
Raw bench json: `bench/gdn4b_<cfg>_bs<N>_r<M>.json`, distilled in
`bench/gdn4b_summary.json`.
Correctness: `base_correctness{,_run2}.json`, `rs_correctness.json`,
`rstuned_correctness.json`, diffs in `rs_vs_base_compare.json`,
`rstuned_vs_base_compare.json`.
Scripts: `bench/gdn4b_server.sh`, `bench/gdn4b_bench.sh`,
`bench/gdn4b_correctness.py`, `bench/gdn4b_compare.py`.
Download log: `logs/gdn4b_download.log`.
