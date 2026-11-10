# SPDX-License-Identifier: Apache-2.0
"""One short greedy generation, to make the SPIKE_TRACE prints fire.

Used to record which GDN core function a server config enters: paired with
SPIKE_TRACE=1 and --enforce-eager on the server, the server log shows either
``_forward_core_decode_aiter`` or the ReplaySSM branch of ``_forward_core``.
Usage: python gdn4b_trace.py --port 8151 --tokens 32
"""

import argparse

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", default="qwen35")
    ap.add_argument("--tokens", type=int, default=32)
    args = ap.parse_args()

    resp = requests.post(
        f"http://127.0.0.1:{args.port}/v1/completions",
        json={
            "model": args.model,
            "prompt": "The capital of France is",
            "max_tokens": args.tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
        timeout=600,
    )
    resp.raise_for_status()
    print("trace generation ok:", resp.json()["usage"])


if __name__ == "__main__":
    main()
