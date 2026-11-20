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
