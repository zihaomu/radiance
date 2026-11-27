# AQL：提前提交 barrier，只延迟信号进入回收池

**问题。** 辅助队列早等是否拖慢生产者，还是主线程过早读取刚退休的信号和 waiter 队列读指针？只比较早提交与晚提交会同时改变两件事。这里用同一个 `ride_wait_submit` 提交同一个 agent-acquire / no-release barrier，并在原位置记录实际 waiter 的队列和包序号。下面的“入池”指进入 `SigPool::retiring`，不意味着它已经可被重新置 1。

| 臂 | 辅助 barrier 的提交位置 | 信号何时进入 retiring | 比较目的 |
|---|---|---|---|
| A `early-now` | 生产者的信号携带 dispatch 之前 | 提交并登记 waiter 后立即入队 | 原有机制基线 |
| B `early-held` | 与 A 同一位置 | 批末按顺序归还，期间 CPU 回收扫描找不到它 | B−A：保持 GPU 等待不变，改变退休时机 |
| C `late-held` | 生产者的两个 dispatch 已提交之后，辅助消费者之前；不等 GPU 完成 | 与 B 相同的批末 | C−B：保持延迟退休不变，改变辅助等待包何时可见 |

**实现。** 在实际单卡 expert-prefetch 的 router-input ride 路径，默认即时退休。诊断开关 `RADIANCE_XP_DEFER_RIDE_RETIRE=1` 只开启 B；信号所有权由每个预取上下文保留到下一次层号回绕的 pass 边界，余量在关闭时交还，容量受该 pass 层数约束。错误路径保持单一所有者。没有新增 CPU 等待、设备包、fence 或清空完成位的操作；仍须通过原有的完成值、队列消费位置、stamp 和 DMA 保护才可复用。C 是独立基准臂，不更改服务调度。见[预取入口](../../../../core/runtime/xprefetch.cpp)、[AQL 实现](../../../../core/device/aql.cpp)、[三臂用例](../../../../tests/aql_ride_retire_control.h)。

## 正式结果

W7900D gfx1100 / GPU4；使用真实预取辅助流，提交线程固定 CPU32。每批 48 个信号和等待，每个位置写入独立 16 KiB 数据，再做一次生产者工作、由辅助流复制；逐字节检查在计时区外。每个臂独立进程，8 批预热、24 批测量；普通构建六轮，三臂所有排列各一次。表内时间为各进程中位数的中位数，单位为 µs / 批。`含退休`从批开始计到批末交还完成，`整批`还包含两个流的最终 drain，不把延后发生的成本藏在计时区外。

| 臂 | CPU 提交 | CPU 含退休 | 整批 | 读取当批退休信号 / 其 waiter 读指针（次/批） |
|---|---:|---:|---:|---:|
| A 早提交、立即退休 | 163.51 | 163.69 | 657.78 | 11 / 11 |
| B 早提交、延迟退休 | 164.34 | 165.09 | 659.74 | 0 / 0 |
| C 晚提交、延迟退休 | 164.86 | 165.60 | 656.80 | 0 / 0 |

计数来自另一个编入 `RAD_AQL_RETIRE_DIAGNOSTICS` 的构建，三轮，每臂共 72 个计数批次；这些耗时不参与上表。计数附着在现有 load 上，不新增一次 signal/queue load。三组的**总**信号读取中位数均为 48 次，预热后所有计数批次均未新建信号。B 消除的是对当批刚退休对象的读取，不是消除整个池的扫描。即时组当批信号读取为非零的中位数为 0，不能把读取次数当作 CPU 一直在等待未完成信号。

- B−A：按同轮进程中位数配对，整批差值中位数 **+1.00 µs**，六轮范围 **−0.12 至 +8.73 µs**，B 仅 1/6 轮更快；CPU 含退休差值中位数 **+0.68 µs**。没有观察到延后入池带来稳定提速。
- C−B：整批差值中位数 **−2.95 µs**，六轮范围 **−8.15 至 +6.77 µs**，C 有 5/6 轮更快。方向提示早提交可能有小代价，但范围跨零，不能宣称已证明辅助等待是瓶颈。
- 没有硬件一致性流量计数；信号地址复用、host 锁操作时序也随退休发生变化。不能从这些差值直接推出 cache-line bouncing 的因果或模型 tok/s。服务端 bring-up 只在非 deterministic 的 **TP1** 启用此预取路径；TP2 不启用，不能用本结果解释 MTP3/TP2。

**正确性与取证。** 封住辅助队列，生产者完成后 waiter 仍不可复用，开门、消费后才可复用；双流交叉 join 和 DMA mark 回归也通过，合计 6,530 项检查。TP1 开/关两臂确认 prefetch 已启用：一条预热和一条测量请求，各生成 128 token，完整消息与用量一致；仅作接入冒烟。普通流的初跑不与正式结果混合。两次早期尝试在首个计时进程前因无法归属 GPU PID 中止，未纳入统计。正式驱动持 OJ 锁，计时外登记实际宿主 PID；只等本次已退出 PID 的驱动记账消失，不允许外部 PID，无前后空闲证据就拒收。

[正式原始记录](../../README.md#evidence)（包内：`results/ride-retire-control-20261126/pinned-xp-owned/lifetime.stderr`） · [可复算数据](../results/ride-retire-control-20261126/summary.json) · [汇总脚本](../../README.md#evidence)（包内：`results/ride-retire-control-20261126/summary.pl`） · [运行驱动](../../README.md#evidence)（包内：`results/ride-retire-control-20261126/run.sh`） · TP1 关闭组（历史引用，未随包保存：`results/ride-retire-control-20261126/tp1-0/measured.json`） · TP1 开启组（历史引用，未随包保存：`results/ride-retire-control-20261126/tp1-1/measured.json`）