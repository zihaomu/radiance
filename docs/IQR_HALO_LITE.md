# IQR HALO-Lite: First Staged Implementation

Date: 2026-11-18. Base commit: 8ebf9c7.

## Implementation

- Build an expert/run map once per call in caller-owned device scratch.
- Split each expert run into at most 16-row tasks; no binary search in the
  mapped GEMM. Unused map entries exit immediately; dropped tails write zero.
- Use a 128-thread workgroup covering 16 rows by 64 output columns.
- Resolve source rows once per task into LDS, outside the K loop.
- Stage each 128-wide activation group and decoded int8 weight group in LDS.
- Decode both nibbles of a packed byte together, sharing the four predecessor
  loads while preserving the original state/MAD mapping and 128-row wrapping.
- Retain int8 WMMA, group order 0 through 19, separate scale multiplies and
  addition, final row scale, and BF16 RNE. No K rotation or reassociation.
- Register scratch for regular and mixed IQR rows. Mixed calls with scratch
  reuse the base launcher after mixed configuration validation. Missing,
  undersized, or misaligned scratch keeps the original launch path.
- Scratch is eight bytes per map entry, bounded by
  min(experts, routed_rows) + ceil(routed_rows / 16) + 1 entries.
- This first version does not reuse a weight tile across multiple M tiles.

## Hardware Validation

W7900D gfx1100, physical GPUs 4 and 5, ROCm 10.1 FFmpeg development image.
GPU locks were held throughout each run; temporary servers were stopped.

Final device tests: 2566 checks across two cases passed. IQR coverage includes
40 fixtures, 720 launches, 1600 rejected invalid calls, and 1,963,540,480 exact
output values. Protected-expert tests compare another 50,304,000 values.
Cases cover layers 0/34/44/47, both ranks, 508/510/512 experts, tokens
1/3/63/64/2048, physical/logical scale views, pitched operands, guards,
short/17-row/fragmented/long/all-dropped runs, and insufficient scratch.

Optional RAD_IQR_BENCH=1 enables event timing: three warmups and twenty calls
per variant, including map construction. Synthetic routing with repeated
expert tables is a microbenchmark, not model throughput. The pooled reference
time divided by mapped time at 2048 tokens was 5.977 across 24 cases.

## Serving Smoke

TP2 exact wire, expert-tiered placement, same existing request/config files,
one warmup and two measured calls per workload. Answers were 4 and 391.

| Mode | Prompt Tokens | Mean Prefill tok/s | Mean Decode tok/s |
| --- | ---: | ---: | ---: |
| MTP0 | 63 | 119.43 | 7.38 |
| MTP0 | 585 | 123.03 | 7.13 |
| MTP3 | 63 | 93.72 | 9.45 |
| MTP3 | 585 | 96.74 | 11.52 |

MTP3 accepted 40 of 48 draft tokens on the 585-token workload. Historical
same-request results were MTP0 94.35/3.12 and MTP3 78.97/6.66 prefill/decode
tok/s. These are historical comparisons, not a contemporaneous alternating A/B.

Raw evidence is in the parent workspace under results/r3d-iqr/implementation:
logs/halo-lite-final-exact.log, logs/halo-lite-maxbench.log, and
halo-lite-serving/*.json. Rebuild targets libr3d and r3d_mtp_ple_device_test;
run r3d_mtp_ple_device_test r3d_moe_ with two locked gfx1100 GPUs and
RAD_R3D_REQUIRE_DEVICE=1. Set RAD_IQR_BENCH=1 only for optional timing.

## Completed Interleaved Serving A/B

2026-11-18, GPU4/5, TP2 exact wire, profiling disabled. Eight rounds in
base/mapped/mapped/base order per MTP mode completed: 56 measured records.
Each workload had a warmup. Short/585-token workloads have four samples per
mode/variant; decode512 has two. Arithmetic mean decode throughput:

| Mode | Workload | Base tok/s | Mapped tok/s | Gain |
| --- | --- | ---: | ---: | ---: |
| MTP0 | short | 3.201 | 7.380 | 130.6% |
| MTP0 | 585-token prompt | 3.155 | 7.143 | 126.4% |
| MTP0 | decode512 | 3.184 | 7.290 | 129.0% |
| MTP3 | short | 5.473 | 9.459 | 72.8% |
| MTP3 | 585-token prompt | 6.639 | 11.565 | 74.2% |
| MTP3 | decode512 | 4.770 | 8.126 | 70.3% |

585-token prefill: MTP0 95.498 -> 122.990; MTP3 78.824 -> 96.649 tok/s.
prefill2k is NOT cold 2K prefill: 2121 prompt tokens, 2048 cached, 73 timed.
Decode includes reasoning tokens. Answers 4/391 passed; all eight decode512
responses have identical text and completion token counts. MTP3 decode512
acceptance was 301/636 for both variants. This is not an accuracy evaluation.

Raw data: ../results/r3d-iqr/implementation/ab-20261118/summary.jsonl and
per-request JSON/server logs. YAMLs differ only in radiance-home per mode.
Captured libr3d.so SHA256:

- Base: c17717e0bf37b6e09100dc32d79431e139af0c0c41c9c5d9306fad3bb33b7372
- Mapped: 1354e5d3c350ee318a4c0a8e6c9434c56ce562b7e8894c84651b809198c2bc1c

These are snapshot comparisons, not a 16/64-row tile-size A/B. The initial
implementation above used 16 rows; current experimental source uses 64.
An adaptive policy still needs an isolated benchmark.

Final rank-0 profile table in ../results/r3d-iqr/implementation/logs/profile-mapped-mtp0.log,
over 128 decode calls: hc_read 38.74 ms/token (26.75% of summed event time),
logits_gemm 24.56 (16.96%), moe_gemm_q 18.76 (12.95%), gemm_nt_q 17.95 (12.40%).
Use only the last table per rank; multiple snapshots exist. Profiling adds
synchronization: these are priorities, not additive wall-clock estimates.
Next targets: hc_read and logits_gemm, then dense quantized GEMM and an
isolated MoE tile-policy experiment. TP1 MTP0/MTP3 attempts in
../results/r3d-iqr/bench-tp-mtp/matrix-20261118-142003/ failed at startup;
no valid TP1 speed is available from those attempts.
