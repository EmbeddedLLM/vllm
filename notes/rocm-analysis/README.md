SPDX-License-Identifier: Apache-2.0

# ROCm analysis record: SketchSSM (#61124) and GDN ReplaySSM (#62004)

Internal record of two upstream PR evaluations on AMD MI355X (gfx950). This
branch carries only analysis artifacts and patches - no vLLM code changes, and
nothing here is proposed upstream as-is.

Base: this branch is based on `main` at `21ef2e5b0a`. The experiments, patches
and numbers below were produced against `main` at `73864f8360` (2026-11-08,
26 commits earlier); patches apply to that base.

Environment: MI355X (gfx950) x8, ROCm 7.15.26333, torch 2.13.0+rocm10.0.0,
Triton 3.8.0, CUDA graphs via vLLM MRV2. Other tenants held GPUs 0-5 for part of
the session; every measurement was pinned to one exclusive GPU and every ratio
is computed against a same-session baseline.

## sketchssm-61124/

[SketchSSM] Add SketchSSM decode for Mamba-2 with Triton kernels
(https://github.com/vllm-project/vllm/pull/61124)

Verdict: does not run on AMD as-is (two one-line blockers), but the kernels are
portable and correct once past them.

| one Mamba-2 layer, CUDA-graph captured, mean rank 8 | batch 128 / 256 / 512 |
| --- | --- |
| vs stock `selective_state_update` (aiter off) | 1.94x / 3.38x / 4.70x |
| tuned (`bf16x3` dots, flush `num_warps=1`) | 2.44x / 4.08x / 5.10x |
| offline e2e, Nemotron-3-Nano-4B, batch 256 | 2.27x (17.19 -> 7.57 ms/step) |
| serving, c=256 (HTTP) | 1.51x req/s, 1.70x TPOT |
| serving, c=1 / 2 / 4 / 8 | 0.83x / 0.86x / 0.90x / 0.95x (crossover near c=32) |

Files: `REVIEW_AND_ROCM.md` (full review + findings), `ROCM_ASK.md` and
`PR_COMMENT_DRAFT.md` (unsent reply drafts), `rocm_port.patch` (the minimal port:
platform gate + dot precision), kernel-test and benchmark harnesses
(`smoke_module_rocm.py`, `bench_graph_step.py`, `sweep_*.py`, `flush_*.py`,
`bench_replayssm_vs_sketch.py`), `probe_triton.py` (HIP dot-precision probe), and
`e2e/` (two-server comparison harness, `SERVING_ROCM.md`, `SERVING_LOWCONC_ROCM.md`,
`MANIFEST.md`).

## replayssm-gdn-62004/

[Kernel] ReplaySSM: cache SSM inputs for faster Gated DeltaNet standard decode
(https://github.com/vllm-project/vllm/pull/62004)

Verdict: works on ROCm once ported to MRV2 (25 lines, 2 files) and is worth
1.09x / 1.14x / 1.14x end to end on Qwen3.5-4B at bs 64 / 256 / 512.

| item | result |
| --- | --- |
| kernel tests on gfx950 | 80/80, 12/12, 82/82 - nothing skipped |
| numerics vs float64 (fp16 ring) | 8.6e-4 output, 5.2e-4 state at flush |
| shipped non-Blackwell config | 1.05x kernel, 1.09-1.14x e2e |
| `input_precision` seam | fp32 rings: tf32 on NVIDIA, IEEE fp32 on gfx950 |
| AITER + `--use-replayssm` | silent no-op; one-line gate restores 1.137x |
| routing replay through the AITER kernel | 1.002x - not worth it |

Files: `TRACE.md` (portability trace), `REPLAYSSM_GDN_ROCM.md` (numeric + kernel
latency), `GDN_E2E_4B_ROCM.md` (e2e), `AITER_PATH_ROCM.md`,
`AITER_EXCLUDE_GDN_ROCM.md`, `AITER_ROUTE_REPLAY_ROCM.md` (AITER interaction),
`MRV2_SPIKE.md` + `mrv2_spike.patch` (the MRV2 port), `REPLY_DRAFT.md` (unsent
reply), `bench/` (server/bench/correctness harnesses and the three AITER
experiment patches), `pr62004.diff`.

## Not included

Raw per-run logs and bench JSON dumps (about 40 MB) stay on the workstation;
each writeup names its own evidence files and the harnesses here regenerate
them. Nothing in either directory was posted upstream.
