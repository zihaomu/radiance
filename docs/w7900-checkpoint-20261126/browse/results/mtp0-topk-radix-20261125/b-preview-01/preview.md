# W7900D TP2 / MTP0：Top-K radix 无插桩速度预览

只测新实现 B，独立新服务一次请求；不是正式 P3 的 A/B 配对。冻结的 32,768-token 输入、256-token 输出，零缓存、无草稿，完整生成文本与旧实现参考的 SHA-256 相同。请求前后两卡均只有本服务进程。

| 服务端解码 | 客户端解码 | 服务端首 token | 服务端整请求 |
|---:|---:|---:|---:|
| 54.03 tok/s（18.51 ms/token） | 54.25 tok/s | 102.97 s | 107.69 s |

相对旧实现 16 请求的历史中位数 15.294 tok/s，描述性比值约 3.53×，不代表新一轮配对已验收。首轮正式 A 在第 3 条请求后两张卡都报告了额外 PID，整次启动排除；未从中挑样本。正式重启 `retry-01` 按原顺序从新的 A 启动开始，预览不计入。

[原始响应](../../../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/b-preview-01/preview.json`） · [原始 SSE](../../../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/b-preview-01/preview.sse`） · 被拒的 GPU 所有者快照（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/p3-main-1-A/n32768/measured-3-after.gpu.json`） · [正式验收进度](../serving-progress.md)