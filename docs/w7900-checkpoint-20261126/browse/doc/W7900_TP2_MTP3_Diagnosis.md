# W7900D TP2: why MTP3 is slower than MTP0 (2026-11-24)

Same-prompt controls reproduce the slowdown on both the PR #563 build and the frozen dev build. TP2 proposes and accepts drafts; it is not silently running MTP0. TP1 has a different failure: the MTP3 graph cannot be declared on the tested build.

## Uninstrumented serving

Two W7900Ds (GPU4/5), TP2 exact wire, default tolerance arithmetic, Qwen3.8 Flash Next IQR-V4, one sequence, 4096 context, batch 64, i8 KV, no prefix cache. Same request bytes across modes: temperature 0, seed 20261124, thinking off, `ignore_eos`, 256 generated tokens. The official essay prompt had 55 input tokens; the `merge_intervals` code prompt had 71. Only `num-speculative-tokens` changed within the model config. Mode groups were measured separately, not randomized paired trials. The old MTP0 `decode128` headline is not the control below.

| Workload | MTP0 tok/s | MTP3 tok/s | Change | Drafts accepted/proposed per request |
|---|---:|---:|---:|---:|
| Essay | 61.60 | 31.87 | -48.3% | 164/279 (58.8%) |
| Code | 61.55 | 40.98 | -33.4% | 183/216 (84.7%) |

These are medians of 16 server-reported rates per cell, two independent starts with eight measured requests after stabilization. Client medians agreed within 0.02 tok/s. Evidence: MTP0 essay（历史引用，未随包保存：`results/diag-mtp0-w7900tp2-essay-A/summary.json`）, MTP3 essay（历史引用，未随包保存：`results/mtp3-readme-warm-essay-A/summary.json`）, MTP0 code（历史引用，未随包保存：`results/diag-mtp0-w7900tp2-code-A/summary.json`）, MTP3 code（历史引用，未随包保存：`results/mtp3-readme-warm-code-A/summary.json`）; each has a sibling `-B` run. Tested executable code `26c1a1d`, worktree `06145aa` (only documentation changed).

An independent run on the unchanged dev binary `0405f35` (eight warm measured requests per cell) gave [61.54](../results/diag-frozen0405f35-w7900tp2-0-essay/summary.json) vs [31.89](../results/diag-frozen0405f35-w7900tp2-3-essay/summary.json) for essay and [61.52](../results/diag-frozen0405f35-w7900tp2-0-code/summary.json) vs [40.96](../results/diag-frozen0405f35-w7900tp2-3-code/summary.json) for code. Within each mode, the frozen dev and integration builds also produced the same output hashes and draft counts. This establishes that the regression relative to MTP0 predates the rebase, not that every historical build behaved this way.

Outputs were stable within each workload/mode but the **default MTP0 and MTP3 output text hashes differ** on both workloads. These are fixed-input/fixed-length throughput comparisons, not equal-output runs. A separate exact-mode essay check did match across [MTP0](../../README.md#evidence)（包内：`results/w7900-mtp-diagnosis-20261124/exact-essay-0/trace-response.json`） and [MTP3](../../README.md#evidence)（包内：`results/w7900-mtp-diagnosis-20261124/exact-essay-3/trace-response.json`）; that one prompt is not a general equality guarantee.

## Where the time goes

Separate instrumented essay, same request and output hash as the corresponding uninstrumented run. The scheduler logged 255 one-row decode steps at MTP0 and 93 four-row verification steps at MTP3. The timers below are mean first-to-last dispatch spans on physical GPU4 (rank 0) for the recorded repeating passes; not end-to-end wall times or sums across GPUs. The time stamps cover replayed tapes, not every startup or tail step.

| Card-0 pass | Recorded passes | Mean span | Measured hotspot(s), per pass |
|---|---:|---:|---|
| MTP0: one-row trunk | 253 | 18.24 ms | LM head `logits_joined_wave` 0.41 ms; 97 `hc_fused` launches totaling 2.02 ms |
| MTP3: four-row trunk verification | 91 | 59.37 ms | LM head `logits_joined_staged_strict` 19.59 ms; 170 `wmma_gemm` totaling 10.21 ms; 48 `gate_up_mapped` totaling 5.68 ms; no `hc_fused` launch |
| MTP3: history plus first draft | 91 | 10.69 ms | Two MTP `project` launches total 3.43 ms; `w2_head_strict` 3.22 ms |
| MTP3: second and third drafts | 92 each | 8.45 / 8.43 ms | Each has `w2_head_strict` ~3.22 ms, two `project` launches totaling ~1.57 ms, and `joined_rerank` ~0.79 ms |

[MTP0 timestamps](../../README.md#evidence)（包内：`results/w7900-mtp-diagnosis-20261124/trace-0/stamps.csv`）, [MTP3 timestamps](../../README.md#evidence)（包内：`results/w7900-mtp-diagnosis-20261124/trace-3/stamps.csv`）, MTP3 scheduler steps（历史引用，未随包保存：`results/w7900-mtp-diagnosis-20261124/trace-3/server.log`）. The LM head dispatch（历史引用，未随包保存：`pr563-integration/libr3d/r3d_vocab_bf16.hip`） requires a single live row for the tolerance wave kernel; the HC dispatch（历史引用，未随包保存：`pr563-integration/libr3d/r3dh_hc_strict.hip`） also requires one row for `hc_fused`. The engine（历史引用，未随包保存：`pr563-integration/core/engine_draft.cpp`） performs draft round one during the history pass and runs rounds two and three serially. These are measured dispatch differences, not estimates of a future patch's gain.

For essay, after the first token, 255 generated tokens over 93 verification steps is 2.74 tokens/step. With MTP0 at 61.60 tok/s, break-even is about 44.5 ms per complete speculative cycle. The uninstrumented MTP3 rate implies about 86.0 ms per cycle. Acceptance helps but cannot make up this cost. The profile does not establish that PCIe, host scheduling, or VRAM bandwidth is the cause.

## Scope and next gate

- TP1 MTP3 startup log（历史引用，未随包保存：`results/mtp3-readme-20261124-ready-tp1/server.log`）: no W4NL `moe_gemm_q` at N=1280, no `gemm_nt_q_gated` at N=1280, and no `logit_rerank` for the full 248320-token vocabulary. No TP1 MTP3 throughput was measured.
- TP2: first benchmark the four-row joined LM head against the ordered reference, then the two-bit draft head and its projections. Next inspect four-row dense/MoE/HC. Verify token-level behavior and benchmark whole-cycle cost on both prompts; do not claim a speedup from acceptance rate or a microbenchmark alone. No kernel or public README changes were made for this diagnosis.