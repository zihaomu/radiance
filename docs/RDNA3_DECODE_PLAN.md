# RDNA3 (gfx1100) TP1 decode: closing the kernel gap to Strata-rdna3

Dev branch only (`rdna3_w7900_qwen3.8_flash_dev`). Finished steps are ported by hand to the
Codeberg branch.

## Baseline (2026-11-20, W7900, no draft, greedy, kv int8)

| | Strata-rdna3 (UD-IQ4_XS) | Ours default (exact) |
|---|---:|---:|
| Decode | 45.59 tok/s (21.9 ms/step) | 24.2 tok/s (41.3 ms/step) |
| GPU kernel time/step | 17.1 ms | 38.4 ms |
| Idle/host time/step | ~4.1 ms | ~4.2 ms |

The whole gap is GPU kernel time. The top three categories, in ms/step:

| Category | Strata | Ours exact | Ours tolerance |
|---|---:|---:|---:|
| LM head | 0.64 | 9.95 | 0.79 (`RAD_LOGITS_WAVE`) |
| Hyper-connection read/write | 1.62 | 7.51 | 4.22 (`RAD_HC_DOWN_SPLIT` + `RAD_HC_UPMIX_FUSED`) |
| MoE routed experts | 3.40 | 8.78 | 8.71 |

Method: our time per kernel comes from `RADIANCE_AQL_STAMP` (dispatch timestamps, no per-op syncs).
Strata's comes from a `rocprofv3` kernel trace. Both are averaged over 128 decode steps.

## Rules for every step

- The default path must stay bit-identical to the strict oracle. A new fast path is an exact
  flag in `r3d_exact_flags[]` (on unless set to "0").
- A device test compares the new kernel to the current one bit for bit, on random data and on
  real shapes.
- TP1 checks: strict and default give identical messages, 6/6 tasks pass, and a stamp profile
  (`TP1_STAMP=1`) refreshes the table.
- GPU4 only, under its flock.

## Step 1: exact LM head at full bandwidth (target 9.95 -> ~1.0 ms)

The head is int8 codes with one bf16 scale per 1x128 block (~645 MB). The exact rule per output
is `acc = acc + x[k] * (code[k] * scale[k/128])`, summed over ascending k, rounding at every
step, with no fused multiply-add. The current `logits_joined_staged_strict` runs at ~65 GB/s:

- it loads one byte at a time;
- it recomputes the scale address with a dtype switch for every element;
- it stages floats in LDS.

New kernel:

- 16-byte coalesced code loads into a padded raw-byte LDS tile.
- Each lane owns one output and loads its scale once per group.
- The activation sits in LDS.
- Exactly the same rounding sequence as today.

## Step 2: exact MoE experts (target 8.78 -> ~3-3.5 ms)

An IQR weight is `bytesum((state16 * mult + inc) & 0x3f3f3f3f) - 126`. The 16-bit state is built
from the 4-bit codes of rows r..r-3. Inside one 128-wide group the int8 dot product is an
integer sum, so its order does not matter. Only the float chain over the 20 groups must keep its
order. That leaves room to restructure while staying bit-exact:

- Microbenchmark `gate_up_dot4` and `down_run` with rocprofv3 counters to find the limiter.
- Split K across waves: integer group sums go to LDS, then one lane runs the 20-group float
  chain in order.
- Load whole code tiles coalesced, instead of 19 scattered 2-byte loads per group.
- Make the decode cheaper with `v_perm` and byte-dot, and fold the -126 into the activation sum.

## Step 3: hyper-connection (exact target 7.51 -> ~4 ms; tolerance 4.22 -> ~2 ms)

Each down output is one strict chain of 10240 float adds, a latency floor of ~20 us per call.
Today it takes 44 us, because the sum is handed lane to lane and the loads are not overlapped.

- Exact: compute the products in parallel into LDS, then one lane runs the chain with reads
  issued ahead. Fuse `up_exact` with `mix_quant_exact`: the arithmetic per output is the same,
  and one block covers a 128-channel group across all 4 streams.
- Tolerance (opt-in): fuse norm, down and up into one kernel.

## Projection

| Step | Saves (ms/step) | Exact |
|---|---:|---|
| 1 LM head | ~9 | yes |
| 2 MoE experts | ~5 | yes |
| 3 HC | ~3.5 | yes |
| Default | 41.3 -> ~24-25 ms (~40 tok/s) | |

## Status

| Step | State | Result |
|---|---|---|
| 0 Baseline and profiling tools | done | `results/strata-compare/README.md` (workspace) |
| 1 LM head exact | done | `logits_joined_stream` (`RAD_LOGITS_STREAM`), bit-identical to strict/staged on the real head. Serving: 9.95 -> 0.89 ms/step |
| 2a MoE gate_up decode kernel | done | `gate_up_split` (`RAD_IQR_SPLIT`), bit-identical on real experts (3 input patterns, 1-512 tokens). Per call with VRAM experts: p50 91 -> 52 us |
| 2b MoE host-resident experts | open | Experts in the pinned host pool are read over PCIe inside the kernel: gate_up p90 189 us, p99 534 us. This tail keeps the mean at 111 us (was 114). All-host bench: old kernel 14 GB/s, new 9 GB/s. Next: a PCIe-friendly access pattern for host experts, or keep them off the GPU path |
| 2c MoE down (w4nl) | not started | `down_run` p50 51 us, mean 65 us |
| 3a HC exact down chain | done | `down_inject_chain` prefetches the next segment and relays the sum with `readlane`; bit-identical. Serving 44.6 -> 41.8 us/call (4.33 -> 4.05 ms/step). It is now near its floor: 10240 dependent fp32 adds per output. Device suite 72 pass |
| 3b HC exact up/mix/norm | done | `mix_quant_exact`: register butterfly + parallel max, 0.70 -> 0.35 ms/step. `strict_norm_parallel`: gains prefetched, 0.79 -> 0.57. All bit-identical; default 31.0 tok/s, identical to strict, 6/6 tasks. `up_exact` stays ~1.8 ms (18.5 us/call). Measured: ~8.6 us is loading the 3.4 MB of up weights; the rest is the 320-term serial chain at ~1.7 waves per SIMD. ALU e4m3 decode, bulk LDS reads, coalesced staging and prefetching up during the down chain were tried and do not help |
| 3c HC tolerance (opt-in) | done | New `up_split` (`RAD_HC_UP_SPLIT`): four lanes per up row, 1280 waves, then the fast mix/quant. Replaces the 20-block fused kernel: 2.22 -> 1.19 + 0.34 ms/step. HC tolerance total 4.22 -> 3.38 ms/step (Strata 1.62). `default-tol` (wave head + down split + up split): 35.6 tok/s, 6/6 tasks; up logits within 8e-6 of exact; text diverges from strict mid-reasoning, same final answer on arithmetic. Device suite 72 pass |
| E2E after 1+2a (+3a: 30.5 tok/s, same) | done | default 24.2 -> 30.7 tok/s; strict 5.91; strict/default messages identical; 6/6 tasks; device suite 72 pass |

# TP2 (two W7900D, GPU4 + GPU5) decode: closing the gap to Strata `--tensor-parallel 2`

Full comparison: `doc/Qwen3.8_Flash_Next_TP2_W7900_vs_Strata.md` (workspace).

## TP2 baseline (2026-11-22, no draft, greedy, kv int8, decode128)

| | Strata TP2 (UD-IQ4_XS) | Ours default (exact) | Ours + opt-in (tolerance) |
|---|---:|---:|---:|
| Decode | 65.75 tok/s (15.21 ms) | 42.43 tok/s (23.57 ms) | 51.90 tok/s (19.27 ms) |

Gap at default, card 0, ms/step:

| Category | Strata | Ours | Gap | Main kernels |
|---|---:|---:|---:|---|
| Hyper-connection | 1.58 | 6.75 | +5.17 | `down_inject_chain` 4.06, `up_exact` 1.77 |
| Dense outside experts | 2.40 | 5.25 | +2.85 | `iqr_wmma_ksplit` 1.96, `split_gemv` 1.70 (170 calls), `project_wave` 1.09 |
| MoE down + gather | 1.05 | 1.96 | +0.91 | `down_run` 1.55, `gather_kernel` 0.41 |
| Norms / quantize / other | 0.17 | 0.72 | +0.55 | `quantize` 0.58 (96 calls) |
| Comm + waits + launch gaps | 3.97 | 4.80 | +0.83 | 1470 dispatches/step |

## TP2 rules

As above, plus:
- TP2 checks: run `scripts/pr-tp1.sh` with `TP1_GPUS=4,5 TP1_CONFIG=run/pr-tp2-mtp0.yaml`. Default and strict must give identical messages on all 9 workloads, 6/6 tasks must pass, and a stamp profile (`TP1_STAMP=1`) refreshes the table (`results/tp2/categories_tp2.py`).
- The device suite stays at 72 pass / 13 skip.
- A step that is not measurably faster end to end is reverted.
- Both GPU locks (renderD132 and renderD133) are held for every run.

## TP2 steps

### T1: dense 5-bit trellis GEMV on packed codes (exact; target 1.96 -> ~0.9 ms)
`iqr_wmma_ksplit` unpacks every weight to int8 for WMMA, and the unpack is ALU-bound. That leaves it at about 3x the bandwidth floor (8192x2560 u5 is 13 MB, about 16 us at 800 GB/s; we take 41 us).
- Port `gate_up_split`'s scheme to u5:
  - build the trellis state with `perm`;
  - mask with `(state*mult+inc)&0x3f3f3f3f`;
  - `sudot4` against the byte-splatted activation;
  - fold the −126 term through the group input sum.
- The integer group sums must be identical; the float chain over the 20 groups stays in order.
- Device A/B against `iqr_wmma_fused` on the real `ssm_inz` / `attn_qg` weights.

### T2: fewer launches (exact; target −0.5 to −1.0 ms)
- T2a: `ar_gather_hc_write` as one kernel (gather into the push), not `moe_gather` + fused AR. Saves 48 launches.
- T2b: fold `quantize` (96 calls) into the kernel that produces its input, where that kernel ends with a whole row.
- T2c: router: fuse `r3d_router_topk` + `scatter_small` + `build_run_map` (144 launches) into one decode-only kernel.

### T3: MoE down `down_run` (exact; target 1.55 -> ~0.8 ms)
32 us per call for about 4 MB. Moving the offsets into LDS did not help (reverted), so profile first.
- Suspects:
  - per-block header and codebook validation (dependent loads);
  - one block per (row, 32 columns), i.e. 800 small blocks;
  - `hoisted` LDS staging.
- Candidate fix: validate the headers once per expert in `build_run_map`-style prep, and let a block serve all 10 rows' columns.

### T4: small int8 GEMVs `split_gemv` (exact; target 1.70 -> ~1.1 ms)
There are 170 calls at about 10 us, near the launch floor.
- Merge calls that share one input into one launch over concatenated outputs: attention K + V, and the shared expert's gate_up.
- The per-output arithmetic is unchanged.

### T5: HC tolerance as one kernel (opt-in; target 3.08 -> ~1.6 ms)
Under the opt-in flags HC is `strict_norm_parallel` + `down_inject_split` + `up_split` + `mix_quant`, four launches and about 32 us per call. Strata's `rdna_gr_fused` does it in about 16 us.
- Fuse them into one kernel with a device-wide arrival counter between down and up, as `hc_up_mix_tiled` does for its groups.
- Needs a decision on making tolerance HC the TP2 default (open, user).

## TP2 projection

| Step | Saves (ms/step) | Exact |
|---|---:|---|
| T1 dense u5 GEMV | ~1.0 | yes |
| T2 launches | ~0.5-1.0 | yes |
| T3 MoE down | ~0.7 | yes |
| T4 int8 GEMVs | ~0.5 | yes |
| T5 HC fused | ~1.5 | no (opt-in) |
| Default (T1-T4) | 23.6 -> ~20.5 ms (~49 tok/s) | |
| Opt-in (T1-T5) | 19.3 -> ~15.5 ms (~64 tok/s) | |

## TP2 status

| Step | State | Result |
|---|---|---|
| T0 TP2 bring-up | done | gfx1100 `ar_hc_write` / `ar_gather_hc_write` rows; byte-identical to the unfused pair (`RAD_TP2_AR_PAIR=1`); 37.11 tok/s |
| T0a attention rows at 12/1 heads | done | `attention_paged_fast` at TP2: 1.44 -> 0.30 ms; 39.15 tok/s |
| T0b `gate_up_split` on uneven slices | done | 2.87 -> 1.26 ms; 41.57 tok/s |
| T0c dense K-split (`RAD_IQR_KSPLIT`) | done | 2.51 -> 1.96 ms; 42.43 tok/s; device suite 72 pass |
| T1 dense u5 packed GEMV | done | `RAD_IQR_SLIDE`: sliding trellis state + `sudot4` + DPP transpose-reduce in `iqr_wmma_ksplit<10>`; 1.96 -> 1.38 ms; bit-identical |
| T2a/b/c launch reduction | T2a done; T2b/c deferred | `RAD_TP2_AR_GATHER_FUSED`: one-row gather inside the fused AR, `gather_kernel` 0.41 ms and 48 launches removed. T2b needs a graph-level fusion; T2c saves only ~0.1-0.2 ms |
| T3 MoE down | partial | LDS staging [16][128] -> [10][128]: 1.55 -> 1.39 ms. Offsets in LDS (slower) and 256-thread blocks (no gain) were reverted |
| T4 int8 GEMV | no change | K + V merge is only 12 launches; a weight-preload try was slower (1.68 -> 1.78 ms) and was reverted. Open: K-split with an ascending fold |
| T5 HC fused (opt-in) | open | `up_split` rows-per-block tuning gave no gain (reverted). A single kernel needs a grid barrier; waits on the tolerance-default decision |
| After T1-T5 | measured | default 44.4-44.5 tok/s (22.50 ms), opt-in 54.82 (18.24 ms); Strata 65.75 (15.21 ms). Default byte-identical to strict; device suite 72 pass |
