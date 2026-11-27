# QSA Top-K 服务验证进度

起始状态（2026-11-25）：P0、P1 已通过，随后开始 P2。最新状态按时间追加在本页末尾；每个服务启动的原始结果在同名子目录内，预热和插桩响应不作为 P3 速度成绩。

- P2 2016：A/B 均为 0 次选块、12 次 attention。2064：A/B 各 12 次选块，rank-0 选块 4.358 → 0.208 ms/步；[48 对快照](../../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-pages-2064.txt`）的有效输入与选出的页及计数均相同。
- P2 8192：[A](../../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-8192-A/decode.json`） / [B](../../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-8192-B/decode.json`） 的 rank-0 选块 16.691 / 0.324 ms/步，与两类 GEMV 重叠分别 16.661 / 0.295 ms/步。[48 对快照](../../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-pages-8192.txt`）的有效输入和全部输出相同。以上均非未插桩成绩。
- P2 32768：[A](../../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-32768-A/decode.json`） / [B](../../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-32768-B/decode.json`） 选块 47.864 / 0.721 ms/步，tape 跨度 67.033 / 19.763 ms/步，输出哈希相同。记录本条时这档的独立页表采集仍在运行，不据此提前通过 P2 或 P3。
- 2026-11-25T14:49:45+08:00 P2 通过：四档共 8 份 A/B 轨迹，三档剪枝形状两卡共 144 对快照的有效分数、页表、513 个输出页及两个计数逐字节一致。[设备报告](p2-report.md)。

- 2026-11-25T14:51:28+08:00 开始 p3-main-1-A（无插桩）。

- 2026-11-25T15:05:58+08:00 停止：P3，p3-main-1-A，exit=1；保留日志及全部样本。

- 首轮拒绝原因已复核：第 3 条请求的后置 GPU 快照（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/p3-main-1-A/n32768/measured-3-after.gpu.json`）在两卡均有额外 PID。整次 A 启动不计入正式 A/B。独立 B 预览和正式重启分开存放，门槛不变。

- 2026-11-25T15:13:42+08:00 P2 通过：四档共 8 份 A/B 轨迹，三档剪枝形状两卡共 144 对快照的有效分数、页表、513 个输出页及两个计数逐字节一致。[设备报告](p2-report.md)。

- 2026-11-25T15:15:25+08:00 开始 retry-01/p3-main-1-A（无插桩）。

- 2026-11-25T15:29:54+08:00 停止：P3，retry-01/p3-main-1-A，exit=1；保留日志及全部样本。
