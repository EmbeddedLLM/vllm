# Draft PR comment for #61124 (copy-paste; not posted)

Options: post the ROCm blockers + the flush finding (most useful), or just ask
the question. Keep it to one round of feedback.

---

Draft A - ROCm support question with the two concrete blockers

```
Tried this on ROCm (MI355X, gfx950, ROCm 7.15, Triton 3.8) with the two Triton
kernel files plus the kernel test, and it is close to working there:

1. `_validate_sketchssm` rejects ROCm with `if not current_platform.is_cuda() or
   not current_platform.has_device_capability(80)`. On ROCm `is_cuda()` is False,
   while `has_device_capability(80)` is True because gfx950 reports (9, 5), so
   only the first clause blocks it and the "compute capability 8.0" requirement
   is not really a requirement of these kernels. Could this use
   `is_cuda_alike()` (ReplaySSM, the feature this builds on, has no platform
   gate at all)?

2. `SKETCH_DOT_PRECISION = "tf32x3"` does not compile on HIP: the allowed dot
   input precisions there are `ieee`/`bf16x3`/`bf16x6`. ReplaySSM already does
   `None if current_platform.is_rocm() else "tf32x3"`. With that, both kernel
   tests pass on gfx950 unchanged, and the dot precision turns out not to matter
   for speed at serving batch sizes (memory bound): bf16x3 is only 4-6% faster
   than ieee at batch 128-512.

Also `tests/kernels/mamba/test_sketchssm_mamba2.py` skips on non-CUDA
(`is_cuda()`), so ROCm would never be covered if the above is fixed.

One perf observation while profiling there, probably relevant on NVIDIA too: in
steady state every step has ~batch/W flush rows, and on gfx950
`_sketch_flush_kernel` costs ~0.09-0.17 ms/step for any row count >= 2 (1 row:
0.015 ms), i.e. it is paid every step and caps the per-layer speedup at ~5x,
while a step with no flush rows is 13x faster than stock decode. Two things
measured: `num_warps=4 -> 1` nearly halves it (167.6 -> 89.7 us at 2 flush rows,
190.7 -> 103.9 us at 32), and the launch grid
`(cdiv(padded_batch, rows_per_program), heads)` dispatches 12288 programs of
which only `n_flush x heads` do work (0.098 ms at padded batch 64 vs 0.175 ms at
512 for the same 2 flush rows). Was the flush grid/config tuned against the
steady-state flush count, or only against all-flush?
```

---

Draft B - shorter, if the maintainers ask "any AMD interest?"

```
Is ROCm in scope for this? The feature is pure Triton and ReplaySSM (its base)
has no platform gate, but `_validate_sketchssm` requires `is_cuda()` + CC >= 8.0
and `SKETCH_DOT_PRECISION = "tf32x3"` does not compile on HIP (allowed there:
ieee/bf16x3/bf16x6). With those two lines changed the kernel tests pass
unchanged on gfx950 and one Mamba-2 layer is 2.4x/4.1x/5.1x faster than stock
decode at batch 128/256/512 (graph-captured, shipped configs plus num_warps=1
flush). Happy to send the numbers/commands if useful.
```
