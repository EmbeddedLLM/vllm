# custom-ua — MiMo-V2.6 + ROCM_AITER_UNIFIED_ATTN + gfx942/gfx950 (LOCAL ONLY, not for upstreaming)

Base: vLLM `29468dde8b515031dc6d4d9d06bf0a2fa0442098` (the prebuilt-image commit).

## Commits on top
1. PR #58262 — [ROCm][MoE] Support MiMo-V2.6 MXFP4 on gfx942 (upstream PR head, 3-way clean)
2. PR #58142 — [Bugfix][Model] MiMo: keep fused fp8 qkv_proj pairing state across weight-loading calls (3-way clean)
3. local UA delta — `csrc/libtorch_stable/cache_kernels.cu` (per-side V-head geometry; LMCache+HMA
   asymmetric hdim 192/128) + `vllm/model_executor/models/mimo_v2.py` (MTP target-select fix; emits
   `head_size_v=128 flows via the KV spec` / `Using ROCM_AITER_UNIFIED_ATTN for attention`)
4. this docs commit

## aiter — stays a wheel-file patch, shipped in `patches/`
- Target: `amd_aiter==0.1.22.post1`, apply from site-packages root:
  `patch -p1 < custom-ua/patches/30-aiter-ua-kernel.patch && patch -p1 < custom-ua/patches/40-aiter-ua-launcher.patch`
- Upstream aiter PR #3698 (kernel file auto) + 3 hand-ported launcher hunks for the wheel's
  consolidated launcher; inputs MUST hash to pristine wheel:
  kernel `57b7d54a16397934f8468b5eaeac310b4a8d6040c35cbc58d98e66e38ed79bb0`
  launcher `32960c434db93034b5b8919228b7a6922e987679331757894a642f47a76f30e7`
- Results: kernel `47e24447dd6d655cd4a1153dc358174c3c538bb3737a14d701125ab7d69ceefd`
  launcher `9448e7f99eac58c8a60ae1549cff99d337bf66d5f0cf7edcd104673d25e65c99`

## Build
- git tree: `PYTORCH_ROCM_ARCH="gfx942;gfx950" CMAKE_BUILD_PARALLEL_LEVEL=$(nproc) python3 setup.py develop`
  (semicolon separator — cmake list; comma does NOT work)
- non-git source tarball flow (upstream image): export `VLLM_VERSION_OVERRIDE=<installed vllm version>`
  for setuptools-scm; `build_ext`, then copy back only
  `build/lib.*/vllm/_C_stable_libtorch.abi3.so` (owns csrc/libtorch_stable/cache_kernels.cu).
  Piping dist-packages `.py` files + cp-layouts = host bundle `ua-recreate/`.

## Serve (LMCache pairing REQUIRES `--block-size 16`)
env CUDA_VISIBLE_DEVICES=<gpu> VLLM_ROCM_USE_AITER=1 VLLM_SERVER_DEV_MODE=1 \
    env -u VLLM_ALLOW_CHUNKED_LOCAL_ATTN_WITH_HYBRID_KV_CACHE \
    vllm serve XiaomiMiMo/MiMo-V2.6-Flash-RL --port <port> --trust-remote-code --generation-config vllm \
      --tensor-parallel-size 1 --max-model-len 200000 --gpu-memory-utilization 0.9 \
      --attention-backend ROCM_AITER_UNIFIED_ATTN --block-size 16 \
      --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":<zmq>}}' \
      --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
      --tool-call-parser mimo --enable-auto-tool-choice --reasoning-parser mimo

## Boot gates
`Using ROCM_AITER_UNIFIED_ATTN for attention` · `kv cache group sizes [32, 16, 16, 16, 16, 16]` ·
`Using external LMCacheMPConnector` (if L1 attached) · `Graph capturing finished` · and NO
`Setting kv cache block size to 64 for ROCM_AITER_UNIFIED_ATTN backend` (that line = LMCache-broken
rc-#3 config) · store/retrieve/`,invalid configuration argument`, alloc, fault counters = 0.

## Validated reference
- gsm8k strict 0.9727 / flex 0.9719 on the assembled gfx942 build (reference band 0.96–0.98; a
  wrong-KV stack scores ~0.34) · MTP active (pos0 accept ≈0.95).
- LMCache cross-backend sharing: UA engine fully served DiffKV-written L1 (19/19 retained keys,
  external hits +31,488, gsm8k 0.9735) — needs the bs16 flag.
- Patch provenance / verification log: ua-recreate/{README.md,RESULT.md} on the build host;
  perf: aiter-unified-ab/{REPORT.md,vllm_bench/RESULTS.md}.
- `patches/full-series-since-29468.patch` = `git am -3`-ready serial of commits 1–3 (apply to any
  git tree at 29468dde if you cannot take this branch directly).
