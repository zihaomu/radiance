# Qwen3.8 Flash Next on W7900D: TP2 versus Strata (decode, no drafting)

Date: 2026-11-22. This is the TP2 version of the TP1 study in `results/strata-compare/README.md`.

## 1. Setup

| Item | Value |
|---|---|
| GPUs | GPU4 + GPU5 (W7900D gfx1100, GPU-BDF-1 and GPU-BDF-2, PCIe 4.0 x16, both on NUMA node 1, P2P available). Both OJ locks (renderD132/133) were held for every run. |
| Workload | One sequence. The prompt is the TCP congestion-control prompt (`results/strata-compare/prompt-tcp.ids`, 35 tokens for Strata, 77 with our chat template). Decode is 128 tokens, greedy. KV is int8 and the context is 4096. |
| Ours | Model `qwen3.8-next-flash-iqr-v4.rad`. Dev branch `rdna3_w7900_qwen3.8_flash_dev` (e80f169) plus the new gfx1100 `ar_hc_write` / `ar_gather_hc_write` rows (§5). Config `run/pr-tp2-mtp0.yaml`: `tp: 2`, `tp-wire: exact`, MTP 0, `expert_tiered`. |
| Strata | UD-IQ4_XS. `--serve --layer-split 24 --tensor-parallel 2`, with the production 2-GPU environment from `rdna3/configs/iq4-2gpu.json` (91 switches). All 24576 experts are resident in VRAM (12292 + 12284 per card, split by id). |
| Strata "no drafting" | Strata only enables `--tensor-parallel` when `--mtp` is loaded and `--spec >= 2`; without the MTP layer it falls back to a plain layer split. So the MTP layer is loaded (`--mtp .../mtp/rt`, built from the official checkpoint's MTP tensors fetched from ModelScope) and capped with `--spec 2 --mtp-max-t 1 --suffix-draft 0`. Every run reported **drafts accepted 0 of 0**. |
| Runners | Ours: `scripts/pr-tp1.sh` with `TP1_GPUS=4,5 TP1_CONFIG=run/pr-tp2-mtp0.yaml`. Strata: `scripts/strata-tp2.sh`. |

## 2. Throughput (uninstrumented, decode128)

| Engine / variant | TP1 (one W7900D) | **TP2 (two W7900D)** | TP2 / TP1 |
|---|---:|---:|---:|
| Strata, no drafting | 45.59 tok/s (21.93 ms) | **65.75 tok/s (15.21 ms)** | 1.44x |
| Strata, layer split only (TP off, for reference) | - | 50.4-51.0 tok/s | - |
| Ours, dev default (exact) | 31.0 tok/s (32.3 ms) | **37.11 tok/s (26.95 ms)** | 1.20x |
| Ours, + opt-in flags (`RAD_LOGITS_WAVE=1 RAD_HC_DOWN_SPLIT=1 RAD_HC_UP_SPLIT=1`) | 35.6 tok/s (28.1 ms) | **42.62 tok/s (23.46 ms)** | 1.20x |
| Ours, strict (every fast path off) | - | 8.43 tok/s | - |

Strata values are the median of three 128-token requests (65.7, 65.8 and 64.3 tok/s, the first being cold).

Correctness at TP2 (ours):
- Default and strict produce byte-identical messages on all 9 workloads and tasks, and every variant passes 6/6 tasks.
- The new fused `ar_hc_write` matches the unfused pair (`all_reduce` + `hc_write`, forced with `RAD_TP2_AR_PAIR=1`) byte for byte on all 9 workloads. The fused kernel runs at 37.11 tok/s and the pair at 36.88.

## 3. Where the time goes (ms per decode step, TP2)

| Category (ms per step) | Strata (W7900 x2, TP2, no drafting) | Ours, dev default | Ours, + opt-in flags | Gap (ours default − Strata) | Gap (ours opt-in − Strata) |
|---|---:|---:|---:|---:|---:|
| Full attention (+ QSA indexer/select) | 0.83 | 1.83 | 1.83 | +1.00 | +1.00 |
| Dense matmuls outside the experts | 2.40 | 5.80 | 5.62 | +3.40 | +3.22 |
| MoE gate_up (+ router) | 2.83 | 3.62 | 3.61 | +0.79 | +0.78 |
| MoE down + gather | 1.05 | 1.95 | 1.95 | +0.90 | +0.90 |
| GDN (conv + recurrent + quant) | 2.38 | 0.98 | 0.98 | -1.40 | -1.40 |
| Hyper-connection | 1.58 | 6.75 | 3.08 | +5.17 | +1.50 |
| LM head | 0.00 (runs on card 1: 0.65) | 0.48 | 0.40 | +0.48 | +0.40 |
| PLE | 0.00 (card 1: 0.08) | 0.05 | 0.05 | +0.05 | +0.05 |
| Norms / quantize / other | 0.17 | 0.72 | 0.70 | +0.56 | +0.53 |
| Compute kernels, sum | 11.24 | 22.19 | 18.22 | +10.95 | +6.98 |
| TP all-reduce + waits + gaps between kernels (launch, host sync) = total − compute | 3.97 | 4.76 | 5.25 | +0.79 | +1.27 |
| **Total = measured step time** | **15.21** | **26.95** | **23.46** | **+11.74** | **+8.25** |

How the table is built (`results/tp2/categories_tp2.py`, output in `results/tp2/table.md`):
- **Category rows** are kernel time per step on card 0.
  - Ours: `RADIANCE_AQL_STAMP`, averaged over the last 128 decode passes of decode160. Stamps only record card 0.
  - Strata: rocprofv3 kernel trace, (n160 − n32) / 128, on Agent 6 (CUDA0).
  - Card 0 is the busier compute card in both engines: Strata's card 0 also runs the shared expert and the CPU ring. Strata's LM head and sampler run on card 1 (Strata's card-1 column is in `table.md`).
- **The total** is the uninstrumented decode128 step (1000 / tok/s).
- **Why one combined communication row:** tracing inflates every wait-type kernel. Strata ran at 27.9 ms per step under rocprofv3, against 15.2 ms uninstrumented, and its `tp_exchange` spin read 3.70 ms on card 0 and 9.34 ms on card 1. So all-reduce, peer waits and launch gaps are reported as one row: total − compute.
  - Raw numbers under tracing: our fused all-reduce + residual write is 1.64 ms per step (96 calls). Strata's CPU-ring spin is 1.29 ms.
- **The categories are not one-to-one:**
  - Strata's GDN row is `mk_gdn_kernel`, a megakernel that also does work our engine counts as dense GDN projections.
  - Dense + GDN together: Strata 4.78 ms, ours 6.78 ms (**+2.0**).

## 4. Findings: where the breakthroughs are at TP2

1. **Hyper-connection: +5.17 ms (default), +1.50 ms (opt-in). This is still the largest item.**
   - The HC read and write run in full on both ranks; they are not split by TP. So HC stays at TP1 size: 6.75 ms at TP2, 7.5 ms at TP1.
   - Meanwhile everything that does split has halved, so HC is now 25% of our step.
   - Strata does HC in 1.58 ms (`rdna_gr_fused`, 96 calls).
   - Options:
     - Shard HC by channel across ranks. The mix matrices are about 1.27 GB per step; splitting the up-mix by channel halves the weight traffic, and the down-mix reduction can ride the all-reduce that already exists.
     - Make the opt-in split kernels the default, if a tolerance contract is acceptable.
2. **Dense matmuls outside the experts: +3.4 ms, or +2.0 ms when counted together with GDN.**
   - `iqr_wmma_fused` 2.51 ms (48 calls), `split_gemv` 1.70 ms (170 calls), `project_wave` (bf16 router/gate GEMVs) 1.10 ms (96 calls), `gemm_m16` + `split_reduce` 0.43 ms.
   - Strata fuses the GDN projections into its GDN megakernel.
3. **Attention: +1.0 ms. This is TP2-specific and cheap to fix.**
   - At TP1, attention resolves to the `model_attn_paged_gate_had_tp1` row (n_head 24 / kv 2, `attention_paged_fast`, 0.30 ms).
   - At TP2 a rank has n_head 12 / kv 1, which falls through to the generic `r3dh_attn_fused_paged_gate` (`attention_kernel`, 1.44 ms for 12 calls).
   - A TP2 geometry row for the fast kernel should bring this back to about 0.3 ms, saving about 1.1 ms per step.
4. **MoE: +0.79 ms on gate_up and +0.90 ms on down + gather.**
   - `gate_up_mapped` 2.87 ms and `down_run` 1.54 ms, against Strata's `native_gu_multi` 1.78 ms and `native_down_multi` 0.55 ms.
   - At TP2 every expert is resident, so these are pure kernel gaps (the PCIe host-pool tail from TP1 step 2b does not apply here).
   - Decode TP2 does not use the TP1 `gate_up_split` fast path (`gate_up_mapped` runs instead): to check whether its conditions (tokens==1, full-width) fail on the per-rank expert split.
5. **Communication + gaps: +0.8 ms (default).**
   - We issue 1542 dispatches per step against Strata's ~1300, and each layer's all-reduce makes both ranks wait for the slower one.
   - The fused `ar_hc_write` saved about 0.2 ms per step over the pair.
   - `ar_gather_hc_write` is currently `moe_gather` followed by the fused all-reduce (two launches); fusing the gather into the same kernel would save 48 more launches.
6. **Norms / quantize / other: +0.56 ms.** This is mostly `quantize` (96 calls, 0.61 ms), which could be folded into the producing kernels.

Estimated headroom from items 1 (opt-in HC), 3 (attention row) and 6: about 23.5 → 21.5 ms, roughly 46 tok/s. Reaching Strata's 15.2 ms also needs the dense/GDN and MoE kernels to close their gaps.

## 5. Code added to make TP2 run (dev worktree, not committed yet)

The v2.2 TP2 graph declares `ar_hc_write` and `ar_gather_hc_write`, and libr3d had no gfx1100 rows for them, so startup stopped with "op has no declared schema". Added:

| File | Change |
|---|---|
| `libr3d/r3dh_ar_hc_write_2rank.hip` (new) | `r3dh_ar_hc_write_2rank_kernel`. It uses the same peer-pull transport, handshake and bf16 rounding as the exact all-reduce, with `hc_write`'s arithmetic (`bf16(fmaf(inj[c], y, h))`) as the epilogue on the reduced word. |
| `libr3d/r3dh_ar_entry.hip` | `r3dh_rad_ar_hc_init` (an all-reduce instance sized M_max × n), `r3dh_rad_ar_hc_write`, and `r3dh_rad_ar_gather_hc_write` (`moe_gather` on the op's own first six operands, then the fused write). Also the A/B switch `RAD_TP2_AR_PAIR=1`, which runs the unfused pair. |
| `libr3d/r3d_qwen_registry.cpp` | Schemas for `ar_hc_write` and `ar_gather_hc_write`, plus the rows `qwen_ar_hc_write_tp2_exact` and `qwen_ar_gather_hc_write_tp2_exact` (2 ranks, exact wire, P2P). |
| `libr3d/r3dh_ar.h`, `libr3d/CMakeLists.txt` | Declarations, and the new source (built with `-ffp-contract=off`). |

Runner and config: `scripts/pr-tp1.sh` now takes `TP1_GPUS=4,5`, holds both GPU locks and sets `ROCR/HIP_VISIBLE_DEVICES`. Added `run/pr-tp2-mtp0.yaml`, `run/pr-tp2-profile.yaml` and `scripts/strata-tp2.sh`.

## 6. Raw data

| Data | Path |
|---|---|
| Ours: tasks + decode | `results/tp2/dev-{default,strict,default-tol,pair}/` |
| Ours: stamp profiles | `results/tp2/stamp-{default,default-tol}/stamps.csv` |
| Strata: decode runs | `results/tp2/strata/run2-mtp-t1/` (TP2, 0 drafts) and `results/tp2/strata/run1/` (layer split without MTP) |
| Strata: rocprof traces | `results/tp2/strata/rocprof-n32`, `results/tp2/strata/rocprof-n160` |
| Category table | `results/tp2/table.md`; regenerate with `cd results/tp2 && python3 categories_tp2.py 65.75 37.11 42.62` |

## 7. Breakthroughs applied (exact: every message byte-identical to strict)

The gap analysis in §4 pointed at three places where TP2 had fallen off a fast path that TP1 already had, plus one latency-bound kernel. All four fixes keep the exact contract: at TP2, all 9 workloads and tasks produce byte-identical messages to strict, and 6/6 tasks pass.

| Step | Change | decode128 tok/s | ms/step | Δ ms |
|---|---|---:|---:|---:|
| 0 | TP2 baseline (fused `ar_hc_write` only) | 37.11 | 26.95 | - |
| 1 | **Attention:** the fused TP1 rows `model_qk_prepare_store_tp1` / `model_attn_paged_gate_had_tp1` now also accept n_head 12 / n_head_kv 1. TP2 had fallen back to the generic `r3dh_attn_fused_*` composition (`attention_kernel`, 1.44 ms); it now runs `attention_paged_fast` (0.30 ms). | 39.15 | 25.54 | −1.41 |
| 2 | **MoE gate_up:** at TP2 the per-rank expert slices are uneven (one parity holds 512 rows, the other 768, against a 768-wide output). `width == output_width` failed, so decode ran `gate_up_mapped` (2.87 ms). `gate_up_split` now accepts narrower experts and writes zeros for a 64-column block past the expert's rows, as `gate_up_mapped` does. It now takes 1.26 ms. | 41.57 | 24.05 | −1.49 |
| 3 | **Dense IQR GEMV (`RAD_IQR_KSPLIT`, new exact flag):** decode's 48 u5 trellis GEMVs (36× 8192×2560 GDN `ssm_inz`, 12× 6144×2560 `attn_qg`) ran `iqr_wmma_fused` with 512 waves walking 20 K-groups serially (2.51 ms). `iqr_wmma_ksplit<4>` spreads the groups over 4 waves per 16-column tile. Each group's WMMA integer sum goes to LDS, and one lane per column runs the float chain in group order. It now takes 1.96 ms. | 42.43 | 23.57 | −0.48 |
| **Total** | | **42.43 (+14%)** | **23.57** | **−3.38** |

With the opt-in tolerance flags on top of these defaults (`RAD_LOGITS_WAVE=1 RAD_HC_DOWN_SPLIT=1 RAD_HC_UP_SPLIT=1`): **51.90 tok/s (19.27 ms)**, up from 42.62, with 6/6 tasks passing. Strata is at 65.75 (15.21 ms), so the remaining gap is 4.06 ms.

Regression check: the device suite still passes all 72 cases (13 skipped, 0 failures; log in `results/tp2/device-suite.log`).

Tried and dropped: staging the MoE run offsets in LDS inside `down_run`, which removes 9 dependent global loads from its binary search, made no measurable difference (41.99 tok/s), so it was reverted.

### Updated gap (TP2, ms per step, card 0)

| Category | Strata | Ours default, before | **Ours default, now** | Gap now |
|---|---:|---:|---:|---:|
| Full attention (+ QSA indexer/select) | 0.83 | 1.83 | **0.63** | −0.20 |
| Dense matmuls outside the experts | 2.40 | 5.80 | **5.25** | +2.85 |
| MoE gate_up (+ router) | 2.83 | 3.62 | **2.01** | −0.82 |
| MoE down + gather | 1.05 | 1.95 | 1.96 | +0.91 |
| GDN (conv + recurrent + quant) | 2.38 | 0.98 | 0.93 | −1.45 |
| Hyper-connection | 1.58 | 6.75 | 6.75 | +5.17 |
| LM head | 0.00 (card 1: 0.65) | 0.48 | 0.47 | +0.47 |
| PLE | 0.00 | 0.05 | 0.05 | +0.05 |
| Norms / quantize / other | 0.17 | 0.72 | 0.72 | +0.55 |
| Compute kernels, sum | 11.24 | 22.19 | **18.77** | +7.53 |
| TP all-reduce + waits + gaps between kernels = total − compute | 3.97 | 4.76 | 4.80 | +0.83 |
| **Total = measured step time** | **15.21** | **26.95** | **23.57** | **+8.36** |

Dense + GDN together, now: Strata 4.78 ms, ours 6.18 ms (+1.4). Profile: `results/tp2/stamp2-default/`.

### What is left, in order of size

1. **Hyper-connection, +5.2 ms.** This is now 2/3 of the remaining compute gap. `down_inject_chain` is 4.06 ms: 97 calls of a 10240-term strictly ordered float chain per output, so it is bounded by add latency, not by bandwidth. Splitting it across ranks does not shorten the chain. Under the exact contract this cannot get much faster. The opt-in split kernels (`RAD_HC_DOWN_SPLIT=1 RAD_HC_UP_SPLIT=1`) take HC to 3.08 ms (+1.5 vs Strata) and are the lever if a tolerance contract is acceptable.
2. **Dense, +2.85 ms:** `iqr_wmma_ksplit` 1.96 ms (still about 3x its bandwidth floor, because the trellis decode is ALU-bound) and `split_gemv` (int8, 170 calls, 1.70 ms, about 10 µs each, close to the per-launch floor). The next steps are a decode that runs `sudot4` directly on the trellis values, as `gate_up_split` does, and fusing the small int8 GEMVs.
3. **MoE down + gather, +0.9 ms:** `down_run` takes 32 µs per call for about 4 MB of weights. The binary-search latency was not the bottleneck (see the reverted attempt above); the per-row block layout (10 rows × 80 column blocks) needs profiling.
4. **Dispatch gaps, +0.8 ms:** 1470 dispatches per step. Candidates are folding `quantize` (96 calls) into its producers and fusing the gather into `ar_gather_hc_write`.

## 8. TP2 plan T1-T5: execution status

The plan is in `rdna3-dev/docs/RDNA3_DECODE_PLAN.md` (TP2 section).

Every step is checked the same way:
- TP2 default and strict give byte-identical messages on all 9 workloads, and 6/6 tasks pass;
- the device suite stays at 72 pass / 13 skip (or gains a new case);
- the step must be faster end to end, or it is reverted.

T5 is opt-in only (tolerance), so it stays behind its flag.

Starting point: default 42.43 tok/s (23.57 ms/step), opt-in 51.90 tok/s (19.27 ms/step).

| Step | State | Kernel change | Kernel ms/step (before -> after) | decode128 default tok/s | opt-in tok/s | Exact check |
|---|---|---|---|---:|---:|---|
| T1 dense u5 packed GEMV | **done** | `RAD_IQR_SLIDE` (new exact flag): in `iqr_wmma_ksplit`, each lane slides the trellis state down the rows (`s = (s<<5 | code) & 0xffff`) instead of rebuilding it from 4 rows per weight. The masked bytes go into `sudot4` against the byte-splatted input, and the -126 offset folds in as `-126*sum(x)`. A 16-column transpose-reduce over DPP / `permlanex16` follows; the float chain is unchanged. 10 waves per tile. Tried: byte loads + `__shfl_xor` (2.46 ms, slower); 4 / 5 / 20 waves (all slower than 10) | `iqr_wmma_ksplit` 1.96 -> **1.38** | 42.43 -> **43.3-44.0** (4 runs) | 51.90 -> **53.27** | 0/9 mismatches vs strict, 6/6 tasks, device suite 72 pass / 13 skip |
| T2 launch reduction | **done (T2a); T2b/T2c deferred** | **T2a** (`RAD_TP2_AR_GATHER_FUSED`, new exact flag): a one-row `ar_gather_hc_write` (decode, M = the padded batch of 64, h has 1 row) gathers inside the fused all-reduce's push. That is `gather_kernel`'s arithmetic for 8 columns per word, with the slot table and weights in LDS and the shared arm folded in. Steps with more rows keep `moe_gather` + fused AR. **T2b** (fold `quantize`, 96 launches): deferred. The two `quantize` launches per MoE layer are separate graph ops (routed and shared SwiGLU-quant), so this needs a graph-level fusion. **T2c** (router): deferred. `router_topk_scatter` is already one op of two launches (top-k, then the counting sort). Merging them into one kernel means rewriting both (the top-k routine uses block barriers in 32-thread blocks) to save about 48 small launches (~0.1-0.2 ms) | `gather_kernel` 0.41 -> **0** (48 launches removed; dispatches 1470 -> 1422); the gather-carrying AR is 1.09 ms vs 0.82 ms for the plain AR | 43.3-44.0 -> **43.8-44.2** | 53.27 -> 53.09 (noise) | 0/9 mismatches vs strict, 6/6 tasks, device suite 72 pass / 13 skip |
| T3 MoE down | **done (partial)** | `down_run`: the LDS code staging shrinks from `hoisted[16][128]` (16 KB) to `[10][128]` (10 groups cover inner <= 640; a larger inner takes the existing non-staged loads). More blocks then stay resident per WGP; same bytes and arithmetic. Tried and reverted: offsets in LDS for the binary search (1.55 -> 1.73 ms, slower); 256-thread / 64-column blocks (1.40 ms, no gain). All experts are VRAM-resident at TP2 (log: "the whole expert plane is resident"), so this is not the PCIe tail. The rest of the gap to Strata (`native_down_multi` 0.55 ms) needs a different layout (several experts' rows per block) | `down_run` 1.55 -> **1.39** | 43.8-44.2 -> **44.2-44.4** | (see final row) | 0/9 mismatches vs strict (2 runs), 6/6 tasks; device suite 72 pass / 13 skip |
| T4 int8 GEMV | **investigated, no change kept** | The plan's premise did not hold. Per call site, the 170 `split_gemv` launches are: GDN `ssm_out` 36 x 18.0 us (0.65 ms), shared-expert gate_up 48 x 9.1 us (0.44 ms), attention out 12 x 15.8 us (0.19 ms), shared-expert down 44 x 3.9 us, attention K / V 24 x ~5.5 us. Merging K + V saves 12 small launches only. Tried and reverted: issue all of a wave's weight groups before summing (`RAD_I8_PRELOAD`); `split_gemv` went 1.68 -> 1.78 ms, slower. The large calls run at ~440 GB/s and are bound by 160 blocks x 3 groups per wave. The next idea is a K-split across blocks with a second-pass fold, but the fold order must stay ascending per column | `split_gemv` 1.68 (unchanged) | - | - | no code kept |
| T5 HC (tolerance default) | **done (2026-11-22), see section 10** | One-row HC read as a single persistent kernel `hc_fused` (down, grid barrier, up, mix + quant), `RAD_HC_FUSED`, now a tolerance default. HC per layer 31 us (4 dispatches) -> 21.7 us (1 dispatch) | default 53.6 -> 59.5 tok/s | - | - | - |

### Result after T1-T5 (2026-11-22)

| | decode128 tok/s | ms/step | vs Strata (65.75 tok/s, 15.21 ms) |
|---|---:|---:|---:|
| Ours default (exact), before T1 | 42.43 | 23.57 | +8.36 ms |
| **Ours default (exact), after** | **44.37 / 44.50** (2 runs) | **22.50** | **+7.29 ms** |
| Ours + opt-in, before T1 | 51.90 | 19.27 | +4.06 ms |
| **Ours + opt-in, after** | **54.82** | **18.24** | **+3.03 ms** |

Exactness:
- Default is byte-identical to strict on all 9 workloads, and 6/6 tasks pass.
- New exact flags: `RAD_IQR_SLIDE` (T1), `RAD_TP2_AR_GATHER_FUSED` (T2a).
- T3 changes only the LDS staging size; it adds no flag.
- Device suite 72 pass / 13 skip (final log: `results/tp2/device-suite-final.log`).

Historical per-category table before fused HC and the guarded router (default profile `results/tp2/stamp-final`, opt-in profile `results/tp2/stamp-tol-t4`; `results/tp2/table-final.md`). See the [post-router table](#11-post-router-category-table) for the current dev build.

| Category (ms per step) | Strata | Ours default | Ours + opt-in | Gap default | Gap opt-in |
|---|---:|---:|---:|---:|---:|
| Full attention (+ QSA indexer/select) | 0.83 | 0.63 | 0.63 | -0.21 | -0.20 |
| Dense matmuls outside the experts | 2.40 | 4.64 | 4.43 | +2.24 | +2.03 |
| MoE gate_up (+ router) | 2.83 | 2.01 | 2.00 | -0.82 | -0.83 |
| MoE down + gather | 1.05 | 1.39 | 1.39 | +0.34 | +0.34 |
| GDN (conv + recurrent + quant) | 2.38 | 0.93 | 0.92 | -1.45 | -1.45 |
| Hyper-connection | 1.58 | 6.74 | 3.03 | +5.16 | +1.45 |
| LM head | 0.00 (card 1: 0.65) | 0.48 | 0.41 | +0.48 | +0.41 |
| PLE | 0.00 | 0.05 | 0.05 | +0.05 | +0.05 |
| Norms / quantize / other | 0.17 | 0.72 | 0.72 | +0.55 | +0.55 |
| Compute kernels, sum | 11.24 | 17.59 | 13.57 | +6.35 | +2.33 |
| TP all-reduce (incl. gather in T2a) + waits + gaps between kernels = total − compute | 3.97 | 4.91 | 4.67 | +0.94 | +0.70 |
| **Total = measured step time** | **15.21** | **22.50** | **18.24** | **+7.29** | **+3.03** |

What remains, in order:
1. **HC exact (+5.2 ms).** The 10240-term ordered chain is a latency floor. The only lever is the tolerance path (+1.45 ms), plus a fused HC kernel (T5 open).
2. **Dense (+2.2 ms):**
   - `iqr_wmma_ksplit` 1.38 ms;
   - `split_gemv` 1.68 ms (the `ssm_out` and attention-out calls run at about half bandwidth);
   - `project_wave` 1.01 ms.
3. **Comm + gaps (+0.9 ms):** 1422 dispatches per step; T2b and T2c are deferred.

## 9. Precision switch: tolerance by default, exact on request (2026-11-22)

Decision: the default numerics are now **tolerance**, which is what Strata ships (`STRATA_GR_RDNA=1` / `STRATA_GR_FUSED=1` reorder its HC sums, use `fmaf` and `__expf`, and move the norm after the dot product). One switch selects the mode, read by libr3d through `r3d_flags.h`:

| `RAD_PRECISION` | Reordering paths | Output |
|---|---|---|
| unset or `tolerance` (default) | `RAD_LOGITS_WAVE`, `RAD_HC_DOWN_SPLIT` and `RAD_HC_UP_SPLIT` are on; each can still be turned off with `=0`. `RAD_HC_NORM_INJECT_FUSED`, `RAD_HC_WMMA`, `RAD_HC_MATRIX_DOWN`, `RAD_HC_UPMIX_FUSED` and `RAD_ATTN_SPLIT` stay opt-in | differs from strict in the last bits (text can diverge mid-reasoning; tasks pass) |
| `exact` | **every** reordering path off, even if one is set to `=1` | byte-identical to strict |

Every exact fast path (`r3d_exact_flags[]`) stays on in both modes.

Implementation:
- `r3d_precision_exact()` and `r3d_tolerance_path(name)` in `libr3d/r3d_flags.h`.
- The flag reads in `r3d_vocab_bf16.hip` (LM head) and `r3dh_hc_strict.hip` (HC) go through `r3d_tolerance_path`.
- `RAD_ATTN_SPLIT` is ignored under `exact`.
- The device test starts the three default tolerance flags at "0", like the exact flags, so its strict comparisons are unchanged.
- Runner: a new `exact` variant (`RAD_PRECISION=exact`); `strict` also pins `RAD_PRECISION=exact`.

Validation at TP2 (GPU4 + GPU5, decode128):

| Mode | tok/s | Messages | Tasks |
|---|---:|---|---|
| default (tolerance) | 53.57 | identical to the earlier opt-in run on all 9 workloads; 5 of 9 differ from strict, as expected | 6/6 |
| `RAD_PRECISION=exact` | 44.65 | byte-identical to strict on all 9 | 6/6 |
| `exact` + `RAD_HC_DOWN_SPLIT=1 RAD_LOGITS_WAVE=1 RAD_HC_UP_SPLIT=1` | 44.50 | byte-identical to strict on all 9 (the flags are ignored) | 6/6 |

With tolerance as the default, T5 (one fused HC kernel per call, like Strata's `rdna_gr_fused`) is unblocked. It is done: section 10.

## 10. T5: fused HC read on the tolerance default (2026-11-22)

Status: **done**. `RAD_HC_FUSED` is in `r3d_tolerance_defaults`, so it is on by default and off under `RAD_PRECISION=exact` or `RAD_HC_FUSED=0`.

### What it is

For a one-row decode step, `strict_launch` in `libr3d/r3dh_hc_strict.hip` now issues one persistent kernel, `hc_fused`. It replaces the four dispatches of the previous default: `strict_norm_parallel`, `down_inject_split`, `up_split` and `mix_quant_exact`. The grid is 168 blocks of 256 threads, all resident on the device. The kernel runs in two phases separated by one grid barrier:

1. **Down (split-K).** Each block covers 16 rows (two per wave) of one half-stream slice (1280 inputs). It computes h * (gain + wadd) before any norm, so the norm moves after the dot product, as in Strata. Block group 0 also writes each slice's sum of squares. Codes and scales are loaded first, then the inputs are staged in LDS.
2. **Grid barrier.** It is self-resetting: an arrival counter plus a generation word, held in a per-stream device buffer that is zeroed once (the `ru_conv_ctr` pattern). `fused_barrier` asks for at most four blocks per unit, so the whole grid is resident.
3. **Up.** Each block folds the 8 slice partials with each stream's 1/rms, then applies SiLU to get the 320 activations (block 0 also writes the 4 inject gates). It then runs 64 up rows and the norm for those rows. A block's 64 rows are 16 channels of all four streams, so it holds every gate those channels need. The last of the eight blocks of a 128-channel group to finish (per-group counter) runs that group's mix + quant (`mix_quant_body`, shared with `mix_quant_exact`). This removes a second barrier and a 20-block tail.
4. **Arithmetic.** Each e4m3 code becomes bf16 with two bit operations per pair. Inputs and activations are staged as permuted bf16 pairs times 2^100, and the dot products run on `v_dot2_f32_bf16`. The up loads are issued at kernel start, before the down phase. The sigmoid is in float (`__expf`).

### Steps and measurements (standalone bench, GPU4, 48 rotating weight copies so the reads come from VRAM)

| Step | us per layer |
|---|---:|
| previous default: norm + down_inject_split + up_split + mix (3.029 ms / 97 calls from the trace; excludes gaps) | 31.2 |
| two kernels (down, up) + mix | 12.2 + 12.6 + 3.4 |
| one kernel, two grid barriers, float math | 24.8 |
| + e4m3 -> bf16 bit decode and `v_dot2_f32_bf16` | 22.1 |
| + mix folded into the up phase (per-group counter, no second barrier) | **21.6** |

For reference: a 3.4 MB pure read costs 7.6 us including launch, and an empty launch of this grid costs 3-4 us.

Tried and not kept:
- Folding once in the last block to reach the barrier, then publishing 650 B (22.2 us: the serial fold adds 2 us).
- Relaxed polling and a per-block wake line (no change: the barrier wake-up spreads over ~2-7 us whatever is polled).
- A bf16 high + low split of the down inputs (error 2.6e-3 -> 2.5e-3 for +0.5 us).
- Prefetching the down scales, and pinning the up loads before the barrier (no change).

Phase timeline inside the kernel (cold launch, us from the first block start):

| Point | us |
|---|---:|
| Last block starts | 3.5 |
| Down phase done (first / last block) | 10.6 / 14.3 |
| Barrier released (first / median / last block) | 15.2 / 17.9 / 22.3 |
| End | 25.8 |

Block startup and barrier exit span multiple microseconds. These timestamps alone do not identify the cause or measure VRAM saturation. The [sync experiments](Qwen3.8_Flash_Next_TP2_W7900_HC_Sync_Plan.md) separate load placement, counter contention and work distribution.

### Accuracy (double CPU reference of the strict read, signed random weights, one row)

| Output | rel L2 | max abs | Values that differ |
|---|---:|---:|---|
| normalized | 9.7e-6 | 9.8e-4 | 1 / 10240 |
| gates | 2.6e-3 | 5.9e-3 | 3574 / 10240 (mostly 1 bf16 ulp) |
| inject | 0 | 0 | 0 / 4 |
| mixed output | 2.8e-3 | 1.6e-2 | 1057 / 2560 |

The main source is rounding the down inputs and the activations to bf16, and the bf16 rounding of every output. Strata's fused HC also reorders and uses fast exp.

### End-to-end (TP2, GPU4 + GPU5, decode128, uninstrumented)

| Mode | tok/s | Messages | Tasks |
|---|---:|---|---|
| default before T5 | 53.6-54.8 | - | 6/6 |
| **default with `hc_fused`** | **59.50** | 2 of 9 differ from strict (tolerance) | 6/6 |
| `RAD_PRECISION=exact` | 44.64 | byte-identical to strict on all 9 | 6/6 |
| Strata | 65.75 | - | - |

The gap to Strata on the default path is now 65.75 / 59.50, from 15.21 ms to 16.81 ms per token (was 18.24 ms before T5).

Stamped default comparison (card 0; last 128 modal decode passes in each trace, before T5 `stamp-tol-t4` vs committed T5 `sync-c1-fd34af7`):
- Dispatches per pass: 1422 -> 1131. HC dispatches: 388 -> 97.
- HC kernel sum: 3.029 -> 2.027 ms/token; pass span: 20.403 -> 18.198 ms. These are not uninstrumented TPS measurements.
- GPU4 strict regression suite for T5: 1,649,361 checks over 72 cases (13 skipped), no failures. Strict comparisons disable tolerance defaults; fused-path testing is separate.
- 2026-11-23 serving recheck: 57.97 tok/s tolerance (identical to the earlier tolerance run on 9/9 messages); 44.25 tok/s forced exact (9/9 identical to strict); 6/6 tasks in both modes.

Files:
- `libr3d/r3dh_hc_strict.hip`: `FusedArgs`, `fused_grid_sync`, `fused_down`, `fused_up_load`, `fused_up`, `mix_quant_body`, `hc_fused`, `fused_barrier`, and the dispatch in `strict_launch`.
- `libr3d/r3d_flags.h`: `RAD_HC_FUSED` added to `r3d_tolerance_defaults`.
- Bench and accuracy check: `results/tp2/t5bench/` (`bench.hip`, `check.hip`, `run.sh`, `check.sh`).

## 11. Post-router category table

2026-11-23, after [router step 13](Qwen3.8_Flash_Next_TP2_W7900_HC_Sync_Plan.md#router-projection-follow-up): W7900 GPU4+5, TP2, decode128, no drafting. The two columns follow the same **precision modes** as section 8. Its old "default" was **exact** (now `RAD_PRECISION=exact`); its old "+ opt-in" was **tolerance** (now the default, including fused HC and the guarded TP2 router). Explicitly enabling those same defaults is not a third mode. Positive gaps mean ours takes longer than Strata.

| Category (ms per step) | Strata | Ours exact | Ours tolerance (default) | Gap exact | Gap tolerance (default) |
|---|---:|---:|---:|---:|---:|
| Full attention (+ QSA indexer/select) | 0.83 | 0.63 | 0.59 | -0.20 | -0.24 |
| Dense matmuls outside the experts | 2.40 | 4.65 | 4.27 | +2.25 | +1.87 |
| MoE gate_up (+ router) | 2.83 | 2.01 | 2.02 | -0.82 | -0.81 |
| MoE down + gather | 1.05 | 1.39 | 1.39 | +0.34 | +0.34 |
| GDN (conv + recurrent + quant) | 2.38 | 0.93 | 0.93 | -1.45 | -1.45 |
| Hyper-connection | 1.58 | 6.74 | 2.02 | +5.16 | +0.44 |
| LM head | 0.00 (card 1: 0.65) | 0.48 | 0.41 | +0.48 | +0.41 |
| PLE | 0.00 | 0.05 | 0.05 | +0.05 | +0.05 |
| Norms / quantize / other | 0.17 | 0.72 | 0.78 | +0.55 | +0.61 |
| Compute kernels, sum | 11.24 | 17.60 | 12.46 | +6.36 | +1.22 |
| TP all-reduce + waits + gaps between kernels (launch, host sync) = total − compute | 3.97 | 4.74 | 4.00 | +0.76 | +0.03 |
| **Total = measured step time** | **15.21** | **22.33** | **16.46** | **+7.13** | **+1.25** |

**What changed:** Section 8's tolerance gap of 3.03 ms/token is now **1.25 ms/token**. That is a comparison across milestones, not a router-only gain. Holding tolerance and fused HC fixed, the integrated router-off/on trial went from 60.00365 to 60.75314 tok/s (16.67 to 16.46 ms/token; 1.20% median paired throughput gain). Dense is now the largest positive compute gap (+1.87 ms); faster GDN (-1.45 ms) and gate_up/top-k (-0.81 ms) offset part of it.

**Accounting and sources:**
- Category rows are card-0 kernel-time sums, not wall time or critical-path attribution. Ours: last 128 passes of the modal dispatch count in the integrated exact trace（历史引用，未随包保存：`results/tp2/router-integrated-exact-stamp/stamps.csv`） and integrated default trace（历史引用，未随包保存：`results/tp2/router-integrated-stamp/stamps.csv`）. Strata: unchanged section 8 baseline, Agent 6, n160（历史引用，未随包保存：`results/tp2/strata/rocprof-n160/run_kernel_trace.csv`） minus n32（历史引用，未随包保存：`results/tp2/strata/rocprof-n32/run_kernel_trace.csv`）, divided by 128. Agent 7's LM-head value is a note, not added to card 0.
- The historical categories put the BF16 router **projection** (`project_wave` or `project_reduce`) in Dense, and expert **selection** (`router_topk`) in MoE gate_up. The new `hc_fused` belongs to Hyper-connection. The all-reduce with a gather epilogue stays in the communication residual, not in MoE down.
- Total is 1000 divided by **unstamped** decode throughput: Strata 65.75 tok/s, exact 44.77337 tok/s (one decode128 run（历史引用，未随包保存：`results/tp2/router-final-exact/summary.jsonl`）), tolerance 60.75314 tok/s (median of six router-on session medians, five measured requests each; first session（历史引用，未随包保存：`results/tp2/router-final-p1-B/tps.tsv`）). The residual is total minus the profiled compute sum; it is **not** isolated network time. Gaps and sums are calculated before rounding. The two integrated stamp runs and the uninstrumented runs used the same dev plugin SHA-256, `ad8633bf3734af1255f1bfcb3d20790517631af45b482085c817c17307fdb851`.

Reproduce with the category generator（历史引用，未随包保存：`results/tp2/categories_tp2.py`）, from `results/tp2`:

```sh
python3 categories_tp2.py --precision-modes 65.75 44.77336724319159 60.7531406098427 \
   router-integrated-exact-stamp/stamps.csv router-integrated-stamp/stamps.csv
```

## 12. Post-router bandwidth and next breakpoint

**Measured during decode, 2026-11-23.** Same integrated plugin as section 11, tolerance default with no experiment flags, GPU4+5, TP2, no drafting. After two warm-ups, five ordinary decode128 requests gave a median **60.84 tok/s** (60.78-60.86). A host sampler read the amdgpu `mem_busy_percent` and `gpu_busy_percent` attributes approximately every 100 ms. For each request it used the response's reported decode duration and discarded 300 ms at each end of the estimated decode interval. This left 15 samples per request per card, 75 per card. These are coarse activity samples, not independent per-kernel measurements.

| Physical card | Memory-controller busy: mean | Median (min-max) | GFX busy: median |
|---|---:|---:|---:|
| GPU4 / rank 0 (`83:00.0`) | 22.96% | 23% (22-24%) | 100% |
| GPU5 / rank 1 (`a3:00.0`) | 22.69% | 23% (22-23%) | 100% |

**Physical GDDR6 read/write GB/s: not established.** Neither multiplying 23% by rated GB/s nor adding the two cards' activity percentages measures bandwidth. Device-wide ROCm `FETCH_SIZE` and `WRITE_SIZE` use L2 external-interface transactions; the joint configuration exceeded hardware limits. With each metric alone the container reported missing `SYS_PERFMON`, and the cross-process calibration returned zero on the active card for a 4 GiB read + 4 GiB write copy. Those readings are invalid, not zero serving traffic. No conclusion about short-lived peak bursts, Strata bandwidth, or PCIe bandwidth follows from the activity samples. Sampling windows（历史引用，未随包保存：`results/tp2/post-router-bandwidth-telemetry/decode-window.tsv`）, request rates（历史引用，未随包保存：`results/tp2/post-router-bandwidth-telemetry/tps.tsv`）, read calibration（历史引用，未随包保存：`results/tp2/post-router-bandwidth-telemetry/calibration-FETCH_SIZE.log`）, write calibration（历史引用，未随包保存：`results/tp2/post-router-bandwidth-telemetry/calibration-WRITE_SIZE.log`）.

**Useful weight bandwidth, not hardware counters.** Card 0; useful bytes for one logical read of the listed weights divided by mean kernel duration over the last 128 modal passes of the integrated trace（历史引用，未随包保存：`results/tp2/router-integrated-stamp/stamps.csv`）. MB and GB are decimal. The percentage uses a nominal 864 GB/s per W7900 as a reference only: AMD SMI reported 384-bit GDDR6 and no rated bandwidth value for these W7900D boards.

| Kernel / call site | Calls per token | Weight MB/call | us/call | Useful GB/s | % of nominal |
|---|---:|---:|---:|---:|---:|
| Guarded BF16 router (`project_reduce`) | 48 | 2.621 | 6.703 | 391 | 45% |
| HC read (`hc_fused`), down + up matrices only | 97 | 6.820 | 20.868 | 327 | 38% |
| Packed u5 input projections (`iqr_wmma_ksplit`), weighted mean | 48 | 12.319 | 27.406 | 449 | 52% |
| Int8 GDN output (`split_gemv`, 2560 x 3072) | 36 | 8.110 | 17.231 | 471 | 54% |
| Int8 attention output (same matrix shape) | 12 | 8.110 | 15.776 | 514 | 60% |

Byte models: router `512*2560*2`; HC `320*10560 + 10240*336` (excludes optional inject weights, gains and scratch); u5 `N*2560*5/8 + 4*(N+20)` for 36 projections at N=8192 and 12 at N=6144; int8 `2560*3072 + 4*2560*24` including per-group weight scales. These rates do not include all reads/writes, duplicate tile loads, prefetch, cache effects or overlap; do not sum them to get a per-GPU or TP2 bus rate. Sustained decode does not show a continuously busy controller, but a particular GEMV can still be bandwidth-limited while it runs.

**Recommended next experiment: exact one-row top-10 selection + scatter in one workgroup.** Leave the 512-column projection and `build_run_map` unchanged. In 128 passes, all 6,144 top-k/scatter pairs were consecutive on the same card-0 queue. Per pair: `router_topk` 6.131 us + inter-dispatch gap 4.170 us + `scatter_small` 5.190 us = **15.491 us**. Across 48 pairs/token this is **0.744 ms/token of stamped span**, including **0.200 ms/token of gap**. It is not all removable or necessarily on the critical path. The existing `r3d_rad_router_topk_scatter` entry already accepts the inputs and outputs but dispatches twice; verify that the serving graph resolves this entry before changing it. A one-row implementation can use one wave for the ten selections and broadcast the selected IDs in LDS to the workgroup for counts, offsets and stable placement; unlike the rejected AR/HC fusion this needs no grid-wide or peer barrier.

**Decision gate:** benchmark the whole pair vs the proposed single dispatch, using the same launch API and real routing inputs. Preserve the lower-ID tie rule, BF16 weights, local expert filtering, protected-expert ordering, counts, offsets, sorted slots and padding byte-for-byte. Retain fallback for unsupported shapes. A trace-equivalent target of **12 us or less** implies about **0.168 ms/token** less pair span (~1% of 16.46 ms) if fully exposed; only counterbalanced, uninstrumented TP2 decode128 A/B can establish that gain. No new serving optimization was enabled by this measurement.

## 13. Matched u5 streaming control

**Question:** how much faster is the production packed-weight access pattern when it does not have to decode and multiply? GPU4 (`83:00.0`, gfx1100), one row, K=2560. The benchmark（历史引用，未随包保存：`results/tp2/t5bench/u5-stream.hip`） uses real packed codes, MAD coefficients and scales from 36 `ssm_inz` and 12 `attn_qg` layers of the deployed bundle and deterministic signed int8 activations. This is a standalone HIP GPU-timing test of rank-0 shapes, not a two-GPU serving run.

**Control and validation:** the production kernel is `iqr_wmma_ksplit<10,true>`. The control generator（历史引用，未随包保存：`results/tp2/t5bench/u5-stream-gen.pl`） retains its block-to-tile mapping, 10 waves x 32 lanes, 20 K groups, run-256 wraparound and overlapping 32-bit packed load addresses. For each layer both launches get the *same device pointers* for weights, scales, activation and output, not independent copies. The control mixes each loaded word into a checksum, combines lane pairs, passes group checksums through LDS and includes the scale bits in its 16-bit output. It is deliberately not a GEMV. An independent host checksum matched all 368,640 control outputs; CPU trellis/dot references matched 336 sampled production outputs including run boundaries. Line-mapped gfx1100 ISA（历史引用，未随包保存：`results/tp2/t5bench/u5-matched-bench/device.s`） has 19 `global_load_b32` at the u5 packed-word source site in each group body in both kernels. HIP reported 112/113 registers (production/control), 23,040 B LDS for both, zero private bytes and four resident blocks/unit for both.

**GPU event results:** two warm-up rounds, 16 measured rounds per cohort, alternating AB/BA order and rotating the order of the three cohorts. Each timing covers 48 complete rotations of its bank; the mixed bank follows layer order. The packed weight + scale working sets are about 473 MB, 118 MB and 591 MB respectively, versus 96 MB Infinity Cache. Times include checksum reduction/output and production epilogue respectively. Useful GB/s counts unique packed payload and weight scales as in section 12, not extra reads or measured VRAM traffic.

| N / rotation | Blocks x threads | Production median (us) | Checksum median (us) | Median paired reduction | Useful GB/s, prod / control |
|---|---:|---:|---:|---:|---:|
| 8192, 36 GDN input weights | 512 x 320 | 27.672 | 22.226 | 19.68% | 475 / 591 |
| 6144, 12 attention QG weights | 384 x 320 | 21.411 | 17.139 | 19.96% | 460 / 575 |
| 36:12 mixed, 48 real weights | as above | 26.336 | 21.300 | 19.11% | 468 / 578 |

The checksum won 16/16 rounds in each cohort. Mixed min-max: 26.295-28.060 us for production, 21.289-21.315 us for checksum. Compare these **within the same harness**; the earlier stamped serving baseline was 27.406 us/call. The median within-round gap is **5.034 us/call**, equivalent to about 0.242 ms over 48 calls, not an achieved or guaranteed token-time saving. Removing the real computation saves around a fifth, while the matched control retains about four fifths of the elapsed time. Those times are not a decomposition into ALU and DRAM stalls. No spill or occupancy difference explains this pair. Test exact decode/dependency and register-lifetime variants before changing the packed format.

Raw per-round samples（历史引用，未随包保存：`results/tp2/t5bench/u5-matched-bench/run.log`）, real plane manifest（历史引用，未随包保存：`results/tp2/t5bench/u5-matched-bench/planes.txt`）, build/source hashes（历史引用，未随包保存：`results/tp2/t5bench/u5-matched-bench/source.sha256`）. Reproduce from `results/tp2/t5bench` with `bash u5-stream-run.sh NEW_DIR bench`.

**Compute-side follow-up (benchmark only).** Kept the same 48 buffers, shape, 320-thread geometry and ascending float fold. The first candidate specializes the u5/K=2560 path, eliminating the generic fallback tile; the second additionally overlaps two independent eight-output trellis windows (the same 19 source words, with three shared warm-up positions between windows). Each matched all 368,640 production BF16 outputs on the test bank. Separate experiment: 18 measured rounds, all six variant orders balanced. Compare candidates to the production column in *this* run.

| Rotation | Production (us) | U5 only (us) | Two chains (us) |
|---|---:|---:|---:|
| 36 x N=8192 | 27.662 | 27.464 | 27.553 |
| 12 x N=6144 | 22.359 | 22.794 | 22.664 |
| 36:12 mixed | 26.316 | 26.287 | 26.321 |

Both candidates cut reported LDS from 23,040 to 1,280 B and private bytes remained zero. Reported registers fell from 112 to 84 and 91 respectively. Mixed paired time reductions were **+0.129%** for specialization and **-0.011%** for two chains; both regressed N=6144. **Retain the serving implementation.** Less LDS/register allocation or a shorter state chain alone did not recover the checksum gap. Next discriminator: retain the u5 state/MAD decode and checksum *each decoded coefficient* instead of doing the dot product. Compare it with the raw-word checksum and the full kernel on the same bank, inspecting ISA/resources again. This separates the decode stage from the downstream dot/transpose/fold more directly before considering a layout change. Exact probe source（历史引用，未随包保存：`results/tp2/t5bench/u5-decode.inc`）, checks and raw timings（历史引用，未随包保存：`results/tp2/t5bench/u5-matched-decode/run.log`）. Reproduce with `bash u5-stream-run.sh NEW_DIR decode`. No serving change, new token/s result, or physical-bus measurement from these experiments.

## 14. Decoded-coefficient control

**Correctness gate passed.** The intermediate control keeps the production rolling state and MAD/mask for each u5 coefficient, but does not multiply by the activation. Each masked byte is at most 63, so four masked words can be added without carries between bytes (4 x 63 = 252). A horizontal byte sum and the signed offset then checksum the actual signed coefficients, not the raw codewords. It uses the raw control's activation seed, paired lanes, LDS, scale checksum and output shape. The CPU oracle instead reconstructs the state by random access and indexes the deployed model's signed 65,536-entry table. All **368,640 output checksums** across 48 layers matched (943,718,400 coefficient contributions; not individual per-coefficient comparisons). Production / raw / decoded used 112 / 113 / 113 registers, 23,040 B LDS each, zero private bytes, and four resident blocks per unit. Untimed gate（历史引用，未随包保存：`results/tp2/t5bench/u5-coeff-check/run.log`）.

**Balanced three-way timing.** Same addresses and GPU4 geometry as section 13; three warm-up rounds, then 18 measured rounds per cohort, all six kernel orders equally represented. Independent repeat in a fresh process. Each cell is a median in us/call. Use paired differences from the same round, not differences against section 13's earlier runs.

| Rotation | Raw words | Decoded coefficients | Full GEMV | Paired decoded minus raw | Paired full minus decoded |
|---|---:|---:|---:|---:|---:|
| GDN input, N=8192 (36 weights) | 22.249 | 24.612 | 27.743 | +2.365 | +3.122 |
| Attention QG, N=6144 (12 weights) | 18.525 | 20.361 | 22.389 | +1.842 | +2.028 |
| 36:12 mixed in layer order | 21.312 | 23.540 | 26.401 | +2.226 | +2.860 |
| 36:12 mixed, independent repeat | 21.198 | 23.394 | 26.255 | +2.198 | +2.856 |

On the mixed bank, decoded was slower than raw in 18/18 rounds in each run; full was slower than decoded in 17/18 and 18/18. The paired full-minus-raw gap was 5.087 / 5.053 us. The intermediate checksum therefore lands roughly **44%** of the way across the gap, with roughly **56%** remaining to full GEMV. These are contrasts between whole kernels: the decoded control pays for its own checksum; full GEMV instead does activation dot products, a lane transpose/reduction, and the ordered scaled FP32 fold. Neither contrast is a calibrated cost of one instruction or a guaranteed speedup. At 48 calls/token the two contrasts correspond to about **0.107 and 0.137 ms/token of standalone kernel time**, not end-to-end savings.

First run（历史引用，未随包保存：`results/tp2/t5bench/u5-coeff-balanced/run.log`）, repeat（历史引用，未随包保存：`results/tp2/t5bench/u5-coeff-repeat/run.log`）, reference and timing harness（历史引用，未随包保存：`results/tp2/t5bench/u5-coeff.inc`）. Reproduce with `bash u5-stream-run.sh NEW_DIR coeff` in `results/tp2/t5bench`.

**Integer-dot control.** A fourth variant calls the unmodified production `iqr_slide_group<5>`, retaining the activation dots, column transpose and LDS group subtotals. Only the ordered scaled float epilogue is replaced with the controls' integer/scale-bit checksum. All 368,640 output checksums matched the independent CPU table lookup and int8 group dot. It used 112 registers, the same LDS and occupancy as full, and no spills. The live u5 ISA audit（历史引用，未随包保存：`results/tp2/t5bench/u5-coeff-fold/isa-counts.tsv`） found 20 group loads in all four versions (19 packed plus one activation). Raw had no coefficient MAD or dot4 instructions. Decoded had 64 `v_mad_u32_u24`, zero activation dots and 16 `v_sad_u8` checksum instructions. Dot and full both had 64 MADs and 64 `v_dot4_i32_iu8`. These are static counts for the inlined five-bit group body.

| Rotation | Raw words (us) | Decoded checksum (us) | Integer dot + checksum (us) | Full GEMV (us) |
|---|---:|---:|---:|---:|
| GDN input, N=8192 | 22.231 | 24.590 | 27.525 | 27.716 |
| Attention QG, N=6144 | 18.339 | 20.243 | 22.124 | 22.221 |
| 36:12 mixed | 21.301 | 23.528 | 26.222 | 26.384 |

Four warm-ups, 24 measured rounds, all 24 kernel orders once per cohort. Mixed median paired increments: **+2.221 us** (decoded minus raw), **+2.683 us** (integer dot/transpose minus decoded), and **+0.182 us** (full minus dot control). Each was positive in 24/24 rounds. The last contrast is about **0.009 ms/token** over 48 calls. It measures replacing the FP32 fold with a checksum epilogue, not deleting the epilogue or measuring isolated stall costs.

**Decision:** keep the ordered FP32 fold and current packed layout. Next test how the `sudot4` consumers and the 16-column transpose are scheduled, keeping the coefficients, integer subtotals and ascending float order unchanged. Screen with the dot control, then require bit-identical full outputs and a repeatable full-GEMV win on the deployment-weighted mix without a material shape regression. The 2.683 us contrast is not a promised saving. No serving change or new token/s measurement.

Four-way raw measurements（历史引用，未随包保存：`results/tp2/t5bench/u5-coeff-fold/run.log`）, ISA audit tool（历史引用，未随包保存：`results/tp2/t5bench/u5-coeff-isa.pl`）. Reproduce with `bash u5-stream-run.sh NEW_DIR stages` then `perl u5-coeff-isa.pl NEW_DIR` from `results/tp2/t5bench`.

## 15. Exact u5 dot and lane schedules

**Hypothesis and gate.** Shorter integer accumulator chains or an earlier start to the column transpose might improve the full GEMV. Compare the production `iqr_wmma_ksplit<10,true>` with three test-only variants: `split_dot` uses two independent two-dot chains instead of one four-dot chain per column; `early_exchange` starts exchanging adjacent column results as soon as a pair is available (lane masks 2, 4, 8, 16, then 1 instead of 16, 8, 4, 2, then 1); `both` combines them. No change to packed layout, group size, grid, LDS handoff or ascending float fold. Screen for >=1% median paired time reduction on the 36:12 full-GEMV mix, wins in at least 20/24 paired rounds, no >1% median paired regression on either shape, and confirmation in a fresh run. Dot-control time alone does not qualify.

**Exactness before timing:** on GPU4, each variant matched 22,118,400 int32 group subtotals, 1,105,920 BF16 full outputs and 1,105,920 dot-checksum outputs against production. The 48 real model matrices were tested with the original deterministic signed activation, alternating -128/127 and sparse signed impulses at changing offsets within K groups. Outputs were poisoned before each run; the original inputs were restored for timing. All four full kernels and dot controls report 112 registers, 23,040 B LDS, zero private bytes and four resident blocks per unit. Correctness log（历史引用，未随包保存：`results/tp2/t5bench/u5-sched-check/run.log`）.

**First run (median us per complete GEMV):** 4 warm-up rounds, 24 measured rounds. Every variant order appears once per cohort and phase. The dot-only control measures the same integer work plus the checksum epilogue, not a replacement for the full result.

| Rotation | Production | Split dot | Early exchange | Both |
|---|---:|---:|---:|---:|
| GDN input, N=8192 | 27.916 | 28.200 | 28.085 | 28.076 |
| Attention QG, N=6144 | 20.986 | 21.274 | 21.058 | 21.204 |
| 36:12 mixed | 26.567 | 26.802 | 26.699 | 26.621 |

Mixed median paired time changes (positive means faster) were -0.958%, -0.576% and -0.264% respectively; full-GEMV wins were 0/24, 1/24 and 2/24. Dot-control changes were -0.965%, -0.523% and -0.116%. No candidate qualified in the first run. Raw balanced timings（历史引用，未随包保存：`results/tp2/t5bench/u5-sched-balanced/run.log`）.

**Independent repeat (same order balancing; fresh process):** mixed full medians were **26.525 / 26.789 / 26.681 / 26.594 us** for production / split dot / early exchange / both. All three lost 24/24 paired rounds. Mixed paired time changes were -0.999%, -0.612%, -0.285%; at N=8192 they were -1.102%, -0.686%, -0.579%, and at N=6144 -1.027%, -0.209%, -0.314%. Dot-control mixed changes were also negative: -0.952%, -0.546%, -0.171%. Paired percentages are computed within rounds, not from the rounded table entries. Repeat log（历史引用，未随包保存：`results/tp2/t5bench/u5-sched-repeat/run.log`）.

**ISA check and decision:** the live u5 group in each full kernel retained 19 packed 32-bit loads plus one activation load, 64 coefficient MADs and 64 activation dots. Static dot instructions with a zero accumulator: 0 / 16 / 0 / 16. The first lane exchange occurred after dot 60 / 64 / 20 / 21 respectively, and the 16 exchange instructions changed from 8 `v_permlanex16_b32` + 8 DPP to 1 + 15 in the early variants. The compiler did change the schedule; it did not reduce the total number of exchanges or produce a speedup. **Keep production.** The section 14 stage contrast is not time recoverable simply by moving those instructions earlier. No serving-path change, TP2 throughput claim, or serving regression rerun for these rejected probes. For the next end-to-end experiment, prioritize the exact top-k/scatter pair from section 12 over another accumulator-order permutation.

First timing/ISA audit（历史引用，未随包保存：`results/tp2/t5bench/u5-sched-balanced/audit.tsv`）, repeat audit（历史引用，未随包保存：`results/tp2/t5bench/u5-sched-repeat/audit.tsv`）, candidate schedules（历史引用，未随包保存：`results/tp2/t5bench/u5-schedule-group.inc`）, benchmark and exactness checks（历史引用，未随包保存：`results/tp2/t5bench/u5-schedule-bench.inc`）. Reproduce from `results/tp2/t5bench` with `bash u5-stream-run.sh NEW_DIR schedule` followed by `perl u5-schedule-audit.pl NEW_DIR`.

## 16. Exact one-row top-k + scatter

**Complete (verified 2026-11-24).** Both final TP2 profile runs finished, and the GPU4 device suite（历史引用，未随包保存：`results/tp2/topk-final-device/device-suite.log`） completed on 2026-11-23: 1,649,574 checks over 74 cases (13 skipped), no failures, 490.64 s. The log includes `r3d_tp2_topk_scatter_exact` and `r3d_tp2_router_reduce_policy`. Both profiles used the integrated plugin SHA-256 `49f5f5f7ef6a26960bc0cf8ff8f43b733fd54bd62bd98cc5a4beffd35a31370e`.

### Final category comparison

Same accounting as section 11. Card-0 category rows are profiled kernel sums. The totals come from **unstamped** no-draft decode128 throughput: Strata 65.75 tok/s, exact 44.94253 tok/s（历史引用，未随包保存：`results/tp2/topk-final-tasks-exact/summary.jsonl`） (one run), and tolerance default 61.59650 tok/s (median of six fused-on session medians). Gaps are ours minus Strata, calculated before rounding. The communication row is a residual, not a direct measurement of network time.

| Category (ms per step) | Strata | Ours exact | Ours tolerance (default) | Gap exact | Gap tolerance (default) |
|---|---:|---:|---:|---:|---:|
| Full attention (+ QSA indexer/select) | 0.83 | 0.63 | 0.62 | -0.21 | -0.21 |
| Dense matmuls outside the experts | 2.40 | 4.67 | 4.32 | +2.27 | +1.92 |
| MoE gate_up (+ router) | 2.83 | 1.92 | 1.92 | -0.91 | -0.91 |
| MoE down + gather | 1.05 | 1.39 | 1.39 | +0.34 | +0.34 |
| GDN (conv + recurrent + quant) | 2.38 | 0.92 | 0.92 | -1.45 | -1.46 |
| Hyper-connection | 1.58 | 6.74 | 2.00 | +5.16 | +0.43 |
| LM head | 0.00 (card 1: 0.65) | 0.48 | 0.41 | +0.48 | +0.41 |
| PLE | 0.00 | 0.05 | 0.05 | +0.05 | +0.05 |
| Norms / quantize / other | 0.17 | 0.74 | 0.83 | +0.58 | +0.66 |
| Compute kernels, sum | 11.24 | 17.54 | 12.45 | +6.31 | +1.22 |
| TP all-reduce + waits + gaps between kernels (launch, host sync) = total − compute | 3.97 | 4.71 | 3.78 | +0.73 | -0.19 |
| **Total = measured step time** | **15.21** | **22.25** | **16.23** | **+7.04** | **+1.03** |

Last 128 modal passes in each final trace: default（历史引用，未随包保存：`results/tp2/topk-release-stamp-default/stamps.csv`） 1,083 dispatches/pass and 9.651 us per fused call; exact（历史引用，未随包保存：`results/tp2/topk-release-stamp-exact/stamps.csv`） 1,374 dispatches/pass and 9.640 us per fused call. Both had 48 fused calls and no standalone top-k/scatter pair per pass. The fused kernel is counted in MoE gate_up; the projection remains in Dense. The **1.2233%** gain is from the matched uninstrumented off/on trial below, not from subtracting separately profiled category tables.

### Experiment log

1. **Live graph path confirmed.** An audit-only stage（历史引用，未随包保存：`results/tp2/topk-path-02/server.log`） saw `r3d_rad_router_topk_scatter` on both TP2 ranks. Decode passes band `M=64` but live tensor sizes are 512 logits, 10 IDs, 10 weights, 10 sorted slots, 513 offsets, 512 counts. Each rank sees all 512 experts (`expert_base=0`), normalized top-10, including protected `407,496`. Prefill also reaches the entry, e.g. 63 live rows; do not use `M==1` or assume a 256-expert local shard. Screen three ordinary-launch variants: the old padded pair, the same pair with one selection row, and a single 32-thread block doing both. Serving gates: exact outputs, fused span <=12 us on the stamped path, then a >=1% median paired decode128 TPS gain and >=5/6 wins in counterbalanced off/on sessions of the same binary.
2. **Exactness gate passed.** One wave runs the production selection arithmetic and scatters ten IDs using per-bucket masks/counts and stable positions; duplicate retained IDs use exact counting. Both W7900 ranks matched the two reference launch variants on 9,216 captured logits rows each (original + held-out captures) and 96 synthetic cases including ties, signed zero, nonfinite inputs, retained slots, local filtering, protected order, partial buckets and padding. Compared all five live output buffers and checked guards; the padded reference had storage for all 64 padded selection rows, only the live row is part of the comparison. Synthetic scatter also checked an independent stable CPU layout. Correctness log（历史引用，未随包保存：`results/tp2/t5bench/topk-exact-01/run.log`）. No serving default changed at this gate.
3. **Matched HIP whole-pair screen passed.** Median us/call across 18 balanced rounds and 48 rotating inputs: GPU4 padded pair / one-row pair / one-wave fused = **13.771 / 13.718 / 8.630**; GPU5 = **13.848 / 13.822 / 8.702**. Each rank timed separately, same ordinary stream API, both launches included for each control. Fusion wins even against the one-row control; the padded row count alone does not explain it. Raw samples（历史引用，未随包保存：`results/tp2/t5bench/topk-pair-01/run.log`）.
4. **Serving trace gate passed.** Same staged plugin, 128 steady TP2 passes/card 0, only `RAD_TP2_TOPK_SCATTER` toggled. Off（历史引用，未随包保存：`results/tp2/topk-stamp-v1-0`）: 1131 dispatches/pass, 48 immediate same-queue pairs, 15.482 us per pair including gap. On（历史引用，未随包保存：`results/tp2/topk-stamp-v1-1`）: 1083 dispatches/pass, 48 fused calls at 9.658 us, zero standalone top-k/scatter calls in the selected passes. The difference is 0.280 ms/token of stamped pair span, not a TPS measurement. Serving A/B uses two warm-ups, five measured decode128 requests/session, six fresh paired sessions per mode in alternating order, router projection unchanged.
5. **Staged serving correctness and interim A/B:** tolerance（历史引用，未随包保存：`results/tp2/topk-staged-tasks-default/tasks.log`） and exact（历史引用，未随包保存：`results/tp2/topk-staged-tasks-exact/tasks.log`） both passed 6/6 tasks. Tolerance matched the pre-fusion integrated router run on 9/9 messages; exact matched strict on 9/9. Initial three off -> on session medians: 60.812 -> 61.779, 60.816 -> 61.736, 60.812 -> 61.793 tok/s. No promotion decision from these three alone.
6. **Staged A/B gate passed:** 6/6 wins, median paired gain **1.558%**, median of session medians **60.814 -> 61.757 tok/s**; individual gains 1.590%, 1.513%, 1.613%, 1.526%, 1.643%, 1.281%. Only then integrated as an exact default for a gfx1100-only build, TP2, contiguous buffers, one live row, 512 total/local experts, base 0, normalized top-10. The opt-out `RAD_TP2_TOPK_SCATTER=0` uses a right-sized one-row two-launch reference for this shape. All other shapes retain the existing dispatch. The protected-list rules are shared with standalone scatter. Integrated two-rank replay（历史引用，未随包保存：`results/tp2/t5bench/topk-integrated-exact/run.log`） passed on the same real inputs. End-to-end repeat against the stronger control follows.
7. **Integration check caught a failed gate:** the first integrated kernel used compile-time local width/base/norm instead of the staged kernel's runtime arguments. The trace（历史引用，未随包保存：`results/tp2/topk-integrated-stamp-1/stamps.csv`） verified 48 replacements, but 12.363 us/call exceeded the 12 us threshold (right-sized pair: 16.197 us). Do not count this build as a pass. Restoring the staged kernel form, without broadening the guard, for a fresh trace.
8. **Revised integration passed.** Restoring runtime kernel arguments with the same host guard gave 9.609 us/call (on（历史引用，未随包保存：`results/tp2/topk-final-stamp-1/stamps.csv`）) vs 16.035 us/pair (right-sized off（历史引用，未随包保存：`results/tp2/topk-final-stamp-0/stamps.csv`）), 48/48 replaced and 1131 -> 1083 dispatches/pass, 128 steady decode passes. Repeat two-rank replay（历史引用，未随包保存：`results/tp2/t5bench/topk-integrated-v2-exact/run.log`） passed. The new public-entry GPU test covers the padded M=64 bound with one physically allocated row, 2-row fallback, ties, protected stable order, both switch values and output guard bytes. It and the neighboring router-policy regression passed 213 checks over two cases（历史引用，未随包保存：`results/tp2/topk-integrated-v2-exact-focused.log`）.
9. **Final uninstrumented serving gate passed against the right-sized control.** 6/6 paired wins, off/on medians **60.82890 -> 61.59650 tok/s**, median paired gain **1.2233%**. Individual gains: 1.5316%, 1.1328%, 1.7876%, 1.2146%, 0.8934%, 1.2320%. Sixty measured decode128 requests over 12 fresh sessions; two warm-ups excluded per session. First off（历史引用，未随包保存：`results/tp2/topk-final-p1-off/tps.tsv`） / first on（历史引用，未随包保存：`results/tp2/topk-final-p1-on/tps.tsv`）; pairs 2-6 follow the same naming pattern. Default tasks（历史引用，未随包保存：`results/tp2/topk-final-tasks-default/tasks.log`） and exact tasks（历史引用，未随包保存：`results/tp2/topk-final-tasks-exact/tasks.log`） each passed 6/6. No fusion override: default matched the pre-fusion router build 9/9, exact matched strict 9/9. Keep the exact fast path for the validated shape; neither the router projection nor `build_run_map` was fused or otherwise changed by this step.
