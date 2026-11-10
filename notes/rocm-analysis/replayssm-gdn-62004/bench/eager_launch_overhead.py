# SPDX-License-Identifier: Apache-2.0
"""Eager (graph-free) launch cost, to show why the latency harness captures graphs."""
import time, torch
from bench_gdn_replayssm import make_state, call_packed, call_replayssm
torch.cuda.init()
for bs in (1, 256):
    t = make_state(bs, torch.float16, torch.bfloat16)
    for name, fn in (("packed_decode", call_packed), ("replayssm", call_replayssm)):
        for _ in range(20):
            fn(t)
        torch.cuda.synchronize()
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            fn(t)
        cpu_us = (time.perf_counter() - t0) / n * 1e6
        torch.cuda.synchronize()
        print(f"bs={bs} {name}: eager per-call wall {cpu_us:.1f} us (CPU launch bound)")
