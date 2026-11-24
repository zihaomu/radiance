# Matched warm benchmark plan: Qwen3.8-Flash-Next on R9700

Status: protocol only; the matched MTP0 measurements have not yet been run. New MTP0 and code results must pass the warm-state and control gates below before appearing in the README. The existing MTP3 essay numbers retain their separately documented historical protocol until replaced by a qualified matched run.

## Question and scope

Measure TP1 MTP0 and MTP3 on the **same input**, output length, R9700, model, source, runtime and resource budget. The first case repeats the existing short essay prompt so its MTP3 result can be checked against the historical 154.47 tok/s measurement. A separate 32K case tests context sensitivity. A code case supplies the currently missing R9700 MTP3 code result. Each case is analyzed independently.

The historical MTP0 number came from a different 32K prompt and included requests before a runtime warm-up transition. Comparing it with the short essay MTP3 number is invalid. The new publication target is *steady-state single-stream decode*, not cold-start or prompt throughput.

## Frozen setup

- Use `qwen3.8-next-flash-iqr-v4.rad`, revision `679c943216e5a33da5f458b5da378586219e874b`, SHA-256 `25aff8c950f4b823bc8b77c8184ed48827cb47058ffce08ae2c83b12e021fea0`.
- Use the same Radiance executable for both modes, based on source `2a2b756371c26f2768b8d129a098df275ace2c73` and the locked ROCm 10.1 image. Record the executable and image hashes before testing. The only intended model-setting change within a case is `num-speculative-tokens: 0` versus `3`.
- Use one physically idle R9700 for all four service starts of a case. Determine its current physical ↔ HIP ↔ PCI ↔ KFD mapping at run time; do not assume historical GPU7/HIP4 still applies. Do not use physical GPU3. Reserve the card and check for foreign processes before and throughout the run.
- Hold `tp: 1`, `max-num-seqs: 4`, `max-model-len: 200000`, `placement: expert_tiered`, `host-pool-mib: 43000`, `kv-cache-dtype: i8`, `gpu-headroom-mib: 96`, `max-num-batched-tokens: 4096`, `checkpoint-interval: 4096` and sampling settings fixed. Disable prefix caching and require `cache_n=0`. Record *actual* model, expert and KV pool allocations because disabling MTP can change effective placement even with identical requested settings.
- Use one request at a time. Set `temperature: 0`, `max_tokens: 256` and `ignore_eos: true`; require exactly 256 output tokens. Record the full request JSON, tokenizer count, response, draft proposals/accepts, server timings and client SSE timestamps.

## Workloads

| Case | Input | API | Purpose |
|---|---|---|---|
| A: essay | `Write a detailed explanation of how the water cycle works for a high school student. Cover evaporation, condensation, precipitation, groundwater, and human impacts. Write at least 500 words in connected paragraphs.` | `/v1/chat/completions`, thinking off | Primary matched comparison; verify the rendered prompt is 55 tokens in both modes. |
| B: 32K archive | The frozen synthetic archive in local `artifacts/flash-next-mtp32k-20261116-075410/prompt.txt`, SHA-256 `e7e3bd61ea7a2997a3791bba19ff56ead844ed47353a548ec12f2de9c89e87a6` | `/v1/completions` | Separate long-context result; require exactly 32,768 input tokens. |
| C: code | `Write a Python 3 function merge_intervals(intervals) that merges overlapping closed integer intervals. Sort the input, return a list of [start, end] pairs, handle empty input and negative endpoints, and include type hints, a short explanation, and three assert-based tests. Provide complete code.` | `/v1/chat/completions`, thinking off | Separate code workload; record the actual token count before the run. |

Do not put case B in the essay column. Do not pool cases or compare their absolute rates as an MTP effect. If the code prompt ends naturally before 256 tokens, `ignore_eos` keeps the decode measurement length fixed; label this as a fixed-length code continuation benchmark.

## Warm-up and measurement sequence

Run four **independent service starts per case** in the order MTP0-A → MTP3-A → MTP3-B → MTP0-B. Use a new artifact directory for each case/start. Never carry a warmed expert cache from one mode to the other. The old single-warm-up controller is a source for request/ownership logging, not an acceptable publication protocol.

For each start:

1. Start the service, verify model identity and GPU ownership, then send at least **four** full requests with the exact case prompt. Keep every warm-up result in the artifact directory and exclude it from the published throughput statistic.
2. Continue warm-up until the last **three** full-request server decode rates have `max/min ≤ 1.05`, no expert-prefetch sleep/wake transition occurs across those requests, and no foreign GPU process appears. Check expert-residency and pool statistics for continuing drift, recording them rather than silently adjusting the gate. Allow at most **12** warm-up requests. If the criteria do not pass, mark the start `warm_state_unqualified` and publish no number for that mode/case.
3. Freeze the warm-up boundary before taking measurements. Send **eight** additional full requests without a profiler or GPU timing hook. Preserve all eight, including slow samples. If prefetch changes state or another process uses the card during measurement, reject the entire start and retain its raw data; do not remove individual requests or move the warm-up boundary after seeing their rates.
4. Stop only this experiment's service, release only its reservation and wait for its GPU allocation to return to the pre-run level before the next start.

The primary value is the pooled median of the 16 measured server `predicted_per_second` values across two qualified starts. Also report each start's median, minimum/maximum, draft acceptance, TTFT and the client decode rate. Require the two same-mode start medians to agree within **5%** (`max/min ≤ 1.05`) and the pooled client/server medians within **1%**. If either fails, show the separate starts as diagnostic data and leave the README cell empty. Preserve startup drift and all rejected cohorts.

## Interpretation and publication gate

- Verify source/model/image hashes, prompt token counts, output counts, zero prefix-cache hits, no foreign GPU work and no truncated/error responses. Compare output SHA within each mode; record any MTP0/MTP3 text difference and task quality separately. A speed ratio is a *deployment-mode comparison*, not a pure MTP algorithm gain, if text or effective pool geometry differs.
- Compare the new essay MTP3 result with the historical 154.47 tok/s only as an A/A reference. A difference beyond 5% triggers a source, GPU, routing, pool and runtime audit; do not replace the historical value by choosing the faster cohort.
- Publish only case A values side by side for TP1 MTP0/MTP3 essay. Publish case C in the code column with its exact prompt and protocol. Put case B in a separate 32K table. The README must state that all shown values passed the warm-state gate, with date, source, model, physical GPU, token counts and sample count.
- Run AQL/profile collection only after the no-hook speed cohort, as a separate diagnostic run if a large matched gap remains. Its timings cannot replace the no-hook throughput values without a measured observer-overhead check.

Expected artifact set per start: `identity.json`, `plan.json`, config, exact requests and outputs, `warmup/*.json`, `measured/*.json`, service log, resource/ownership samples, `quality.json` and `summary.json`. The final case report records all four starts, the warm-state decision, A/A drift, numerical checks and publication decision.
