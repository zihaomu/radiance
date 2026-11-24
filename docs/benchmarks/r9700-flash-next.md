# Qwen3.8-Flash-Next IQR-V4 on Radeon AI PRO R9700

These are historical measurements from this fork's R9700 host, archived here with the underlying result files. Throughput is the server's `predicted_per_second` for generated text, excluding prompt processing.

## Matched short-prompt decode, 2026-11-16

Model: `qwen3.8-next-flash-iqr-v4.rad`, [revision `679c9432`](https://huggingface.co/StillDeadcode/qwen3.8-next-flash-iqr-v4/tree/679c943216e5a33da5f458b5da378586219e874b); radiance `2a2b7563` (2.1.0), ROCm 10.1.0. The model was run with IQR-V4 experts, int8 KV and MTP=3.

Both tests sent the same 55-token, non-thinking English water-cycle prompt, requested 256 generated tokens, and received all 256. The prompt cache reported zero hits. After an initial suite had warmed the expert cache, one warm-up and eight measured single-stream requests were recorded per configuration.

| Configuration | Physical R9700 cards | Median decode | Median draft accepted / proposed | Raw results |
|---|---|---:|---:|---|
| TP1 | GPU7 | **154.47 tok/s** | 163 / 282 | [JSON](r9700-flash-next-tp1-decode-warm.json) |
| TP2, `tp-wire: wht6` | GPU6/7 | **236.69 tok/s** | 162 / 282 | [JSON](r9700-flash-next-tp2-decode-warm.json) |

The observed TP2/TP1 ratio is 1.53. TP1 allowed 4 sequences, a 43,000 MiB host pool and 4096 prefill tokens per batch; TP2 allowed 8 sequences, a 12,288 MiB host pool and 2048 prefill tokens per batch. Both configured `max-model-len: 200000`. `wht6` uses a lossy cross-card transfer path. The different budgets and request histories mean the ratio is a deployment comparison, not a controlled TP-only speedup or a claim about equal output quality.

## Separate long-context TP2 observation, 2026-11-24

On the experimental [issue #594 headroom branch](https://github.com/zihaomu/radiance/tree/fix/issue-594-card-headroom), two R9700s ran TP2 with MTP=3, int8 KV, `tp-wire: wht6`, an approximately 150K-token cached conversation and 512 generated tokens per turn. Across 20 consecutive turns, the median server-reported decode rate was **194.24 tok/s** (range **170.96–229.88 tok/s**). The [raw JSON](r9700-flash-next-tp2-150k-20turns.json) records the per-turn context, cache hits and rate. This later result has a different context, output length, GPU pair and source branch from the short-prompt pair above, so it is listed separately.
