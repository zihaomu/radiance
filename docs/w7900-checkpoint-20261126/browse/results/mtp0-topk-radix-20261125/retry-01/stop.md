# P3 retry-01：因 GPU 独占门槛停止

2026-11-25 15:29:54 +08:00，第一对的 A 启动在第三条测量响应的 `owners(after)` 检查结束，退出码 1。整次 A 启动排除；不选前两条，不与独立 B 预览配对。本轮没有正式 B 启动或配对，未生成 P3 最终报告。

| 实物 | 结果 |
|---|---|
| 启动快照（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/retry-01/p3-main-1-A/gpu-ready.json`）、第三条请求前（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/retry-01/p3-main-1-A/n32768/measured-3-before.gpu.json`） | GPU4、5 上均只有服务 PID `3820152`。 |
| 第三条请求后（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/retry-01/p3-main-1-A/n32768/measured-3-after.gpu.json`） | 两卡均列出额外 PID `3831010`；显示名 `N/A`，所报 VRAM 和 GFX 用量为 0。单张快照既不能确认身份，也不能证明有或没有执行负载。 |
| 服务日志（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/retry-01/p3-main-1-A/server.log`） | 4 次预热、3 条测量响应。三个已返回的 SSE 与 JSON 均独立核验：32,768 输入、256 输出，零缓存、MTP0、全文 SHA 匹配冻结参考。 |

前一次独立尝试也因额外 PID 被排除，但 PID 不同；前次快照（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/p3-main-1-A/n32768/measured-3-after.gpu.json`）。两次均未建立合格 A/B 对。P0–P2 的结果不受本次中止影响。[54.03 tok/s](../b-preview-01/preview.md) 为另一次无插桩 B 单请求，不能充当 P3 正式 B。

下一步先确认 GPU4/5 有能覆盖整组 A/B 的独占时段，或独立查明监控进程的身份和来源。保持原独占检查、配对顺序和所有阈值；没有隔离依据前，不再重复消耗同一组 32K 请求。