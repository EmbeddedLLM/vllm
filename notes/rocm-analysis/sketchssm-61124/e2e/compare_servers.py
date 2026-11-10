#!/usr/bin/env python3
"""Compare greedy outputs and logprobs of the baseline and SketchSSM servers.

Sends the same prompts to both OpenAI-compatible servers concurrently (batch of
4), and reports per-token logprob deltas plus exact text mismatches.
"""

import argparse
import asyncio
import json

import aiohttp

PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n",
    "Q: What is 17 * 23?\nA:",
    "Write a haiku about the ocean.\n",
    "The three laws of thermodynamics are:",
    "In Python, a generator is",
    "Translate 'good morning' into Japanese:",
]


async def complete(session, url, prompt, max_tokens, num_logprobs):
    payload = {
        "model": "nemotron",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "logprobs": num_logprobs,
        "ignore_eos": False,
    }
    async with session.post(url, json=payload) as r:
        r.raise_for_status()
        return await r.json()


async def batch(url_a, url_b, prompts, max_tokens, num_logprobs):
    async with aiohttp.ClientSession() as session:
        names = [f"{url_a.split(':')[-1]}/sketch-{i}" for i in range(len(prompts))]
        results = await asyncio.gather(
            *(
                complete(session, url, p, max_tokens, num_logprobs)
                for url in (url_a, url_b)
                for p in prompts
            )
        )
        # Names are not supported by this endpoint; keep results keyed by tuple.
        _ = names
    n = len(prompts)
    return results[:n], results[n:]


def summarize(name, prompt, out, other, max_tokens):
    choice = out["choices"][0]
    other_choice = other["choices"][0]
    text = choice["text"]
    other_text = other_choice["text"]
    lp = (choice.get("logprobs") or {}).get("token_logprobs") or []
    other_lp = (other_choice.get("logprobs") or {}).get("token_logprobs") or []
    n = min(len(lp), len(other_lp))
    deltas = [
        abs((a if a is not None else 0.0) - (b if b is not None else 0.0))
        for a, b in zip(lp[:n], other_lp[:n])
    ]
    first_div = next((i for i, d in enumerate(deltas) if d > 1e-3), None)
    return {
        "prompt": prompt,
        "name": name,
        "baseline_text": text,
        "sketchssm_text": other_text,
        "text_match": text == other_text,
        "tokens_compared": n,
        "max_logprob_delta": max(deltas) if deltas else None,
        "mean_logprob_delta": sum(deltas) / len(deltas) if deltas else None,
        "first_token_logprob_delta_gt_1e-3": first_div,
        "baseline_finish": choice.get("finish_reason"),
        "sketchssm_finish": other_choice.get("finish_reason"),
        "max_tokens": max_tokens,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-url", default="http://127.0.0.1:8056/v1/completions")
    p.add_argument("--sketchssm-url", default="http://127.0.0.1:8057/v1/completions")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--num-logprobs", type=int, default=5)
    p.add_argument("--out", default="compare.json")
    args = p.parse_args()

    base, sketch = asyncio.run(
        batch(
            args.baseline_url,
            args.sketchssm_url,
            PROMPTS,
            args.max_tokens,
            args.num_logprobs,
        )
    )
    rows = [
        summarize("prompt%d" % i, prompt, b, s, args.max_tokens)
        for i, (prompt, b, s) in enumerate(zip(PROMPTS, base, sketch))
    ]
    text_matches = sum(r["text_match"] for r in rows)
    print(f"text matches: {text_matches}/{len(rows)}")
    for r in rows:
        print(
            f"[{r['name']}] match={r['text_match']} tokens={r['tokens_compared']} "
            f"maxdlp={r['max_logprob_delta']:.4g} "
            f"first_div={r['first_token_logprob_delta_gt_1e-3']}"
        )
        if not r["text_match"]:
            print(f"   baseline : {r['baseline_text']!r}")
            print(f"   sketchssm: {r['sketchssm_text']!r}")
    with open(args.out, "w") as f:
        json.dump({"max_tokens": args.max_tokens, "rows": rows}, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
