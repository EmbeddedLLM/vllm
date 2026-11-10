# SPDX-License-Identifier: Apache-2.0
"""MRV2 spike: sample 8 greedy 64-token completions from a running server.

Writes JSON with token ids, text and per-token logprobs so a ReplaySSM run can
be diffed against the baseline run of the same server config.
Usage: python mrv2_spike_correctness.py --port 8123 --out rs.json
"""

import argparse
import json

import requests

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "def fibonacci(n):",
    "Q: What is 17 * 23? A:",
    "The three laws of thermodynamics are:",
    "In the year 2050, the most surprising technology was",
    "Translate 'the cat sat on the mat' into French:",
    "A short poem about the sea at night:",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/v1/completions"
    rows = []
    for prompt in PROMPTS:
        resp = requests.post(
            url,
            json={
                "model": "qwen3.5-0.8b",
                "prompt": prompt,
                "max_tokens": 64,
                "temperature": 0.0,
                "logprobs": 0,
                "echo": False,
                "ignore_eos": False,
            },
            timeout=600,
        )
        resp.raise_for_status()
        body = resp.json()
        choice = body["choices"][0]
        logprobs = choice["logprobs"]
        rows.append(
            {
                "prompt": prompt,
                "text": choice["text"],
                "token_ids": logprobs["tokens"],
                "token_logprobs": logprobs["token_logprobs"],
            }
        )
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
