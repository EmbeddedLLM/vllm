# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Microbenchmark ROCm DeepSeek-V4 ATOM paged decode kernels.

This is intentionally narrow: it exercises the vendored ATOM sparse paged
decode wrappers at deployment-like C32 decode shapes so kernel parameters can
be screened before full server benchmarks.
"""

from __future__ import annotations

import argparse
import itertools

import torch
import triton

from vllm.models.deepseek_v4.amd.v4_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_aiter_direct,
    _sparse_attn_v4_paged_decode_triton,
    sparse_attn_v4_paged_decode_kv_splits,
    sparse_attn_v4_paged_decode_split_kv,
)


def _make_ragged_indices(
    *,
    t: int,
    kv_len: int,
    total_pages: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    kv_indptr = torch.arange(
        0,
        (t + 1) * kv_len,
        kv_len,
        device=device,
        dtype=torch.int32,
    )
    base = torch.arange(kv_len, device=device, dtype=torch.int32)
    offsets = torch.arange(t, device=device, dtype=torch.int32)[:, None] * 17
    indices = (base[None, :] + offsets) % total_pages
    return indices.contiguous().view(-1), kv_indptr


def _bench_one(fn, *, warmup: int, rep: int) -> float:
    result = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if isinstance(result, tuple):
        return float(result[0])
    return float(result)


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is required")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    dtype = torch.bfloat16

    q = torch.randn((args.tokens, args.heads, args.dim), device=device, dtype=dtype)
    attn_sink = torch.randn((args.heads,), device=device, dtype=torch.float32)
    out = torch.empty_like(q)

    print(
        "shape "
        f"T={args.tokens} H={args.heads} D={args.dim} "
        f"kv_lens={args.kv_lens} block_ks={args.block_ks} "
        f"kv_splits={args.kv_splits}"
    )
    heuristic_splits, source = sparse_attn_v4_paged_decode_kv_splits(
        args.tokens,
        args.heads,
    )
    print(f"heuristic kv_splits={heuristic_splits} source={source}")

    for kv_len in args.kv_lens:
        total_pages = max(args.total_pages, kv_len + args.tokens * 17)
        unified_kv = torch.randn((total_pages, args.dim), device=device, dtype=dtype)
        kv_indices, kv_indptr = _make_ragged_indices(
            t=args.tokens,
            kv_len=kv_len,
            total_pages=total_pages,
            device=device,
        )
        swa_pages = min(args.swa_pages, total_pages // 2)
        split_swa = unified_kv[:swa_pages].contiguous()
        split_tail = unified_kv[swa_pages:].contiguous()
        split_indices = kv_indices.clone()
        split_indices %= total_pages

        print(f"\nkv_len={kv_len} total_pages={total_pages} swa_pages={swa_pages}")
        for block_k, kv_splits in itertools.product(args.block_ks, args.kv_splits):
            if kv_splits <= 0:
                continue
            dense_ms = _bench_one(
                lambda: _sparse_attn_v4_paged_decode_triton(
                    q,
                    unified_kv,
                    kv_indices,
                    kv_indptr,
                    attn_sink,
                    args.softmax_scale,
                    out=out,
                    kv_splits=kv_splits,
                    block_k=block_k,
                ),
                warmup=args.warmup,
                rep=args.rep,
            )
            aiter_ms = _bench_one(
                lambda: _sparse_attn_v4_paged_decode_aiter_direct(
                    q,
                    unified_kv,
                    kv_indices,
                    kv_indptr,
                    attn_sink,
                    args.softmax_scale,
                    out=out,
                ),
                warmup=args.warmup,
                rep=args.rep,
            )
            split_ms = _bench_one(
                lambda: sparse_attn_v4_paged_decode_split_kv(
                    q,
                    split_swa,
                    split_tail,
                    split_indices,
                    kv_indptr,
                    attn_sink,
                    args.softmax_scale,
                    swa_pages=swa_pages,
                    out=out,
                    kv_splits=kv_splits,
                    block_k=block_k,
                ),
                warmup=args.warmup,
                rep=args.rep,
            )
            print(
                f"block_k={block_k:<2} kv_splits={kv_splits:<2} "
                f"dense_ms={dense_ms:.4f} aiter_ms={aiter_ms:.4f} "
                f"split_ms={split_ms:.4f}"
            )


def _csv_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--kv-lens", type=_csv_ints, default=[144, 512])
    parser.add_argument("--block-ks", type=_csv_ints, default=[16, 32, 64])
    parser.add_argument("--kv-splits", type=_csv_ints, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--total-pages", type=int, default=8192)
    parser.add_argument("--swa-pages", type=int, default=4096)
    parser.add_argument("--softmax-scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
