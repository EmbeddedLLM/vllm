# SPDX-License-Identifier: Apache-2.0
"""Milestone 3 probe: can GDN ReplaySSM start on this tree (main + PR merge)?

Runs a tiny greedy decode twice with Qwen3.5-0.8B (real weights, GDN hybrid,
16 key / 16 value heads x 128) through the offline LLM API: baseline, then
`use_replayssm=True`. Prints both outputs, or the exact exception each mode
raises. `--mode baseline|replayssm` restricts to one mode.

Usage: python e2e_replayssm_gdn.py --mode replayssm
"""

import argparse
import glob
import os
import traceback

os.environ.setdefault("VLLM_WORKER_MULTICROP_DISABLE", "1")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = glob.glob("/models/models--Qwen--Qwen3.5-0.8B/snapshots/*/")[0]
PROMPTS = ["The capital of France is", "Once upon a time, in a small village,"]


def run(mode: str):
    kwargs = dict(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=2048,
        enable_prefix_caching=False,
        mamba_backend="triton",
        mamba_ssm_cache_dtype="float32",
        language_model_only=True,
        additional_config={"gdn_prefill_backend": "triton"},
        disable_log_stats=True,
    )
    if mode == "replayssm":
        kwargs.update(use_replayssm=True, replayssm_buffer_len=16)
    llm = LLM(**kwargs)
    outs = llm.generate(PROMPTS, SamplingParams(temperature=0.0, max_tokens=16))
    for o in outs:
        print(f"[{mode}] {o.prompt!r} -> {o.outputs[0].text!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "replayssm", "both"], default="both")
    args = ap.parse_args()
    modes = ["baseline", "replayssm"] if args.mode == "both" else [args.mode]
    for mode in modes:
        try:
            run(mode)
        except Exception as exc:
            print(f"[{mode}] FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            raise SystemExit(1)


if __name__ == "__main__":
    main()
