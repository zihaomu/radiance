# Qwen3.8 Flash Next on RDNA3 (gfx1100)

libr3d runs the Qwen3.8 Flash Next IQR-V4 container (`qwen4exp_fp8`) on gfx1100 GPUs: Radeon
RX 7900 XTX / XT / GRE and Radeon PRO W7900 / W7800. Kernels are built for gfx1100 only. The
`r3dh_ple_host` plugin keeps the FP8 PLE table (47.7 GiB) in host memory and gathers the rows
each step needs.

## Build

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DRAD_GPU_TARGETS=gfx1100
cmake --build build
```

## Serve (one GPU)

```yaml
server:
  kernels: libr3d:r3dh_ple_host
models:
  flashnext:
    model: /models/qwen3.8-next-flash-iqr-v4.rad
    tp: 1
    num-speculative-tokens: 0
    max-num-seqs: 1
    max-model-len: 4096
    max-num-batched-tokens: 64
    kv-cache-dtype: i8
    placement: expert_tiered
    host-pool-mib: 16384
    prefix-cache-mib: 0
    mm-max-patches: 0
```

`placement: expert_tiered` keeps the experts that do not fit in VRAM in pinned host memory. On a
48 GiB card the weights take 45.1 GiB of VRAM plus 16.6 GiB of host pool. Cards with less VRAM
need a larger `host-pool-mib`; that setup has not been measured.

## Fast paths

Kernel variants that give bit-identical output to the reference kernels are on by default;
setting one of the variables in `libr3d/r3d_flags.h` to `0` selects the reference kernel.
Variants that reorder floating-point sums are off unless set to `1`:

| variable | effect |
|---|---|
| `RAD_LOGITS_WAVE` | LM head with one wave per vocabulary row and FMA partials |
| `RAD_HC_DOWN_SPLIT`, `RAD_HC_UPMIX_FUSED` | hyper-connection down projection reduced in parallel over K |
| `RAD_HC_NORM_INJECT_FUSED`, `RAD_HC_MATRIX_DOWN`, `RAD_HC_WMMA` | hyper-connection inject and down projection on WMMA |
| `RAD_ATTN_SPLIT` | split-KV attention |

## Measured (Radeon PRO W7900, TP1, 128 decoded tokens)

| configuration | tok/s |
|---|---|
| reference kernels (all variables in `r3d_flags.h` set to `0`) | 5.9 |
| default | 24.3 |
| default + `RAD_LOGITS_WAVE=1` | 31.2 |

## Limits

- One GPU (`tp: 1`) without speculative decoding. `num-speculative-tokens` above 0 is not
  supported yet.
- Two-GPU serving needs a gfx1100 `ar_hc_write` kernel, which 2.2.0 graphs declare; not included.
