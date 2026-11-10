# SPDX-License-Identifier: Apache-2.0
"""Diff two 8x64 greedy runs (baseline vs ReplaySSM).

Reports token-identical counts and the logprob delta on the positions where the
two runs emit the same token (so a greedy flip is not counted as a "delta").
Usage: python gdn4b_compare.py base.json rs.json
"""

import json
import sys


def main() -> None:
    base = json.load(open(sys.argv[1]))
    rs = json.load(open(sys.argv[2]))
    identical = 0
    rows = []
    for b, r in zip(base, rs):
        bt, rt = b["token_ids"], r["token_ids"]
        same = bt == rt
        identical += same
        deltas = [
            abs(x - y)
            for x, y in zip(b["token_logprobs"], r["token_logprobs"])
            if x is not None and y is not None
        ]
        # first position where the two runs diverge, if any
        first_diff = next(
            (i for i, (x, y) in enumerate(zip(bt, rt)) if x != y), None
        )
        rows.append(
            {
                "prompt": b["prompt"],
                "identical": same,
                "first_diff_token": first_diff,
                "max_abs_dlogp": max(deltas) if deltas else 0.0,
                "mean_abs_dlogp": sum(deltas) / len(deltas) if deltas else 0.0,
                "n_tokens_compared": len(deltas),
            }
        )
    all_deltas = [
        r["max_abs_dlogp"] for r in rows
    ]  # per-prompt max, not used for the aggregate mean
    n = sum(r["n_tokens_compared"] for r in rows)
    weighted_mean = (
        sum(r["mean_abs_dlogp"] * r["n_tokens_compared"] for r in rows) / n
        if n
        else 0.0
    )
    out = {
        "token_identical": identical,
        "total_prompts": len(rows),
        "max_abs_dlogp": max(all_deltas) if all_deltas else 0.0,
        "mean_abs_dlogp": weighted_mean,
        "per_prompt": rows,
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
