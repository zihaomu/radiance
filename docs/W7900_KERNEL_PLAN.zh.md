# Qwen3.8 Flash Next 算子快速实现计划

- 日期：2026-11-18。
- 状态：首批实现及后续实验已完成；最新完成 HC 分阶段计时和带舍入证书的并行 norm，轻量对照保持逐 bit 一致。结果见第 10 至 12 节；所有新路径仍为 opt-in。
- 工作目录：`../libr3d-iqr-dev`。
- 目标硬件：两张 W7900D，gfx1100，TP2。
- 参考实现：`fork/halo_qwen3.8_flash_next@5d2b950`。

## 1. 目标与边界

尽快交付可运行、可单独启停的 fast 算子。每完成一个算子，只做小规模正确性检查、短 microbench 和服务冒烟，再继续下一项。

- 聚焦单流 decode、MTP0/MTP3，兼顾现有 prefill 路径不回退。
- 保留现有 strict 路径，每次只替换一个算子。
- 减少测试规模，不默认放松既有数值语义。
- 改变浮点归约顺序或 FMA 行为的实现，作为明确标记的非 bit-exact 候选，不直接覆盖 strict 默认路径；默认启用前单独确认数值策略。
- 不同时修改权重量化格式、checkpoint、TP wire 和调度器。
- 本轮不处理 TP1 启动、视觉、长上下文和高并发。

## 2. 当前基线

TP2、GPU4/5、无 profile 的交错 A/B 已完成 8 轮、56 条测量。优化版连续生成 512 token 的 decode 为：

| 模式 | 当前速度 |
| --- | ---: |
| MTP0 | 7.29 tok/s |
| MTP3 | 8.13 tok/s |

速度包含 reasoning token。原始数据在 `../results/r3d-iqr/implementation/ab-20261118/`。

最新 MTP0 rank-0 profile 的主要热点为：

| 算子 | 累计事件时间/token | 占累计事件时间 |
| --- | ---: | ---: |
| hc_read | 38.74 ms | 26.75% |
| logits_gemm | 24.56 ms | 16.96% |
| gemm_nt_q | 17.95 ms | 12.40% |
| IQR gate/up | 16.51 ms | 约 11.4% |

Profile 会引入同步，只用于确定优先级，不能把这些时间直接相加预测真实延迟。标为 prefill2k 的旧测试命中了 2048-token 前缀缓存，只重算 73 token，不作为冷 2K prefill 成绩。

## 3. 实现顺序

| 阶段 | 交付内容 | 排序理由 |
| --- | --- | --- |
| P1 | IQR 短 run dot4 GEMV | 延续已有实现，改动集中，优先快速落地 |
| P2 | logits decode 快路径 | 接口简单，热点占比较高，容易独立测量 |
| P3 | HC decode 快路径 | 收益空间最大，但中间数值边界更多 |
| P4 | 组合启用及必要的预取优化 | 单项收益确认后再整合 |

第一批交付 P1 和 P2，HC 紧接着推进。dense gemm_nt_q 后续按新 profile 决定，不在第一批同时展开。

## 4. P1：IQR 短 run dot4 GEMV

入口：`../libr3d-iqr-dev/libr3d/r3dh_moe_iqr.hip`，同时保持 mixed IQR 委托路径可用。

- 先复用当前 run-map，不重写调度和权重布局。
- expert run 不超过 4 行时使用 int8 dot4 GEMV；长 run 继续使用 mapped WMMA。
- 将少量 activation 和 scales 一次放入 LDS。
- 提前读取下一组压缩 codes 和 weight scale，减少解码等待。
- 保持每个 128-wide group 的精确 int32 和、原有浮点 fold、group 顺序及 BF16 RNE。
- 保留原路径作为独立对照；短 run 未满足条件时回退。

暂不做：去掉 map launch、16x128 权重 relayout、大范围 tile 搜索。当前 64-row mapped 已能复用权重覆盖多个 16-row tile，不再把扩大 prefill tile 当成 decode 的首要方向。

交付：小规模样例逐 bit 一致，短 run microbench 有收益，prefill 回退样例正常。没有收益则不默认启用，不在这一阶段无限调参。

## 5. P2：logits decode 快路径

入口：`../libr3d-iqr-dev/libr3d/r3d_vocab_bf16.hip` 及对应注册入口。

- 第一版只支持当前 BF16 head、K=2560 和实际 TP shard，聚焦 decode/小 M。
- 采用协作加载和 activation 复用，改善权重读取合并，替换低效的逐列串行访问结构。
- 不先做全 dtype、全尺寸通用 GEMM，也不改 head 量化格式。
- 若保持 bit-exact，则保持原 K 累加顺序，先优化线程和加载布局。
- 若采用并行点积归约，则明确作为非 bit-exact fast 候选，保留 strict 默认和独立开关，不将输出接近宣称为逐 bit 一致。

交付：当前 head 形状可运行；报告耗时和数值差异；无 NaN/Inf；完成短服务冒烟。非 bit-exact 候选通过冒烟不等于获得完整模型精度认证。

## 6. P3：HC decode 快路径

当前 strict 入口：`../libr3d-iqr-dev/libr3d/r3dh_hc_strict.hip`。现有 fast 计算参考在 `../libr3d-iqr-dev/libr3d/r3dh_hc_hc_read.hip`，开始前先核对可复用部分和实际注册路径。

- 限定 hc=4、lowrank=320、小 M，不先扩展为通用 HC 实现。
- 保持 norm 结果及既有 BF16 舍入边界，优先替换最重的 down/up 标量矩阵乘。
- 借鉴 HALO decode 分工，优先复用已验证的 gfx1100 矩阵指令封装。
- 第一版保留中间 buffer 和分开的 launch，先完成计算路径，不立即做大融合。
- 后续仅在测量有必要时合并 mix、quant，减少中间读写和 launch。
- 与 strict 累加不同的部分作为独立 fast 候选，不直接改变默认数值策略。

交付：小规模 down/up 和最终输出对照、单算子计时、服务冒烟。出现明显偏差时先定位最早不一致的中间结果，不扩大测试面。

## 7. 轻量验收

每阶段只跑下面三项，不要求完整 accuracy 或十亿级逐 bit 检查。

1. 小规模算子对照：decode 行数 1/4/16，加一个 prefill 回退样例；IQR 额外覆盖 mixed expert 和 dropped route。bit-exact 路径检查逐 bit，非 bit-exact 候选报告最大绝对误差、相对误差及 NaN/Inf，不混用验收口径。
2. 短 microbench：少量预热加约 20 次 GPU event 计时，与旧路径交错比较；无全量参数搜索。
3. 服务冒烟：2+2、17x23、一个固定长生成请求；MTP0/MTP3 各执行一遍。记录答案、finish reason、实际生成 token 数、decode 速度，检查异常提前停止和明显重复。

P4 组合完成后再做一次简短 A/B，不要求每阶段重跑 8 轮。固定请求、相同 GPU 和服务参数；无 profile 计时，避免将前缀缓存收益当成 prefill 加速。

所有 GPU 测试持有 OJ GPU 锁，开始前确认设备空闲；实验环境变量只对单条命令生效。保留现有基线库快照，避免后续构建覆盖对照。新增算子应能独立开关，便于单项 A/B 和回退。

## 8. P4 与完成标准

- 先验证 P1、P2 的单项收益，再组合；HC 完成后重新检查热点分布。
- 只继续优化已测到的瓶颈，不自动展开权重 relayout、跨层融合或复杂调度。
- 每项留下实现状态、启停方式、支持条件、回退条件、数值检查和一次简短性能结果。
- 无收益或数值不满足要求的候选保留实验状态，不默认启用。
- 最终报告当前 TP2 MTP0/MTP3 速度、相对固定基线的变化和仍未验证的范围，不承诺达到 HALO 异构硬件上的历史速度。

## 9. 暂不纳入本轮

完整模型 accuracy、十亿级逐 bit 校验、长上下文压力、视觉回归、高并发、TP1 启动修复、全形状自动调优、完整权重 relayout、跨层大融合，以及无关代码重构。

## 10. 实施记录：2026-11-18

### 10.1 当前交付状态

| 阶段 | 状态 | 已实现及边界 |
| --- | --- | --- |
| P1 | 第一版完成 | run 不超过 4 行时使用 dot4，复用 map；其余仍走 mapped WMMA；完整暂存短 run 的 activation/scales；保持原浮点 fold |
| P2 | 第一版完成 | BF16 和真实模型 joined planes head 均支持协作 LDS staging；K=2560、小 M；保留 code 乘 scale、K 累加顺序和独立乘加 |
| P3 | 实验实现 | 仅替换 down/up；BF16 WMMA、down 16-way split-K；保留 strict norm、激活函数、mix、inject、quant；非 bit-exact，默认关闭 |
| P4 | 轻量验收完成 | 同构建开关 A/B，TP2 MTP0/MTP3，三种请求；exact 组合通过消息和生成 token 数一致性检查 |

所有新路径默认关闭，没有修改既有服务的默认配置。首批交付时，P1 的下一组 codes/scales 显式预取、去除 map launch、权重 relayout 尚未实现；预取的后续实验见第 11 节。

### 10.2 开关与入口

| 开关 | 值 1 的行为 | 值 0 或未设置 |
| --- | --- | --- |
| RAD_IQR_DOT4 | 短 run dot4，mixed IQR 复用 | 原 mapped WMMA/无 scratch 回退 |
| RAD_LOGITS_STAGED | K=2560、M 不超过 16 的 staged logits，支持 BF16/合法 joined planes | 原 strict logits |
| RAD_HC_WMMA | 当前 HC 形状、M 不超过 16 的 WMMA/split-K 实验路径 | 原 strict HC |

推荐测试组合：`RAD_IQR_DOT4=1 RAD_LOGITS_STAGED=1 RAD_HC_WMMA=0`。容器内运行需通过 `docker run -e` 传入；不在常驻 shell 中 export 实验变量。

源码入口：`r3dh_moe_iqr.hip` 的 gate_up_dot4，`r3d_vocab_bf16.hip` 的 logits_staged_strict/logits_joined_staged_strict，以及 `r3dh_hc_strict.hip` 的 decode_wmma/decode_down_finish。文件均位于 `../libr3d-iqr-dev/libr3d/`。

新增可复现服务脚本：`../libr3d-iqr-dev/scripts/quick-operators.sh`。从工作区根目录执行：

```bash
bash libr3d-iqr-dev/scripts/quick-operators.sh results/r3d-iqr/implementation/quick-operators-new
```

默认只比较 base 和 exact。显式包含 HC 实验路径：

```bash
QUICK_OPERATOR_VARIANTS='base:0:0:0 exact:1:1:0 hc:1:1:1' \
  bash libr3d-iqr-dev/scripts/quick-operators.sh results/r3d-iqr/implementation/quick-operators-hc-new
```

variant 顺序为标签:IQR:logits:HC；可用 `QUICK_OPERATOR_MODES=mtp0` 限制模式。脚本持有 GPU4/5 锁，保存库 SHA256、原始 JSON 和服务日志，结束后停止临时容器；已有 summary 的输出目录会被拒绝。

### 10.3 轻量算子验证

- IQR：真实权重和 mixed expert，1/4/16/64 token、短 run/4-row run/长 run/dropped route，逐 bit 对照通过。quick 模式约 1.7 秒；无需运行 2048-token 大例。
- IQR 初始计时：稀疏短 run 相对当前 mapped WMMA 约 3.3 至 3.6 倍；长 run 回退基本持平。此为合成路由 microbench，不等于服务加速。
- logits：dense exact、joined matrix、拒绝及真实模型 TP2 head 检查通过；真实 head 比较 2,989,272 个值。保留损坏 header 不写输出的行为。
- BF16 head 实际 shard 尺寸 microbench：M=1 为 51.22 -> 2.88 ms，M=4 为 77.69 -> 8.80 ms，M=16 为 227.48 -> 21.93 ms，输出逐 bit 一致。真实模型使用 joined planes，不能将这组 BF16 加速比当成真实模型 head 的加速比。
- HC：1/4/16 行的中间 down、up logits、gates 和最终输出检查通过，无 NaN/Inf；最终输出最大绝对误差为 0/0.000244141/0.000976562，按参考峰值归一化误差最高约 0.134%。这是合成小样例，不是完整精度认证。
- HC 第一版未做 split-K 时约慢 4 倍，未启用；16-way split-K 后约 313 -> 290、316 -> 284、339 -> 282 us，仍仅作为实验候选。

### 10.4 最终服务冒烟

TP2、GPU4/5、相同构建、profile 关闭；每种配置一个短请求预热，再测短问答、585-token 算术请求和固定 128-token 生成。每项一个测量样本，不是多轮统计结论。

| 模式/工作负载 | base tok/s | exact：IQR+logits tok/s | 再加 HC 候选 tok/s |
| --- | ---: | ---: | ---: |
| MTP0 短问答 | 7.37 | 9.32 | 9.66 |
| MTP0 算术请求 | 7.14 | 8.92 | 9.25 |
| MTP0 decode128 | 7.37 | 9.30 | 9.65 |
| MTP3 短问答 | 9.36 | 12.11 | 12.33 |
| MTP3 算术请求 | 11.43 | 14.89 | 14.54 |
| MTP3 decode128 | 8.73 | 11.33 | 12.86 |

exact 组合的 decode128 相对本轮 base 约提升 26%/30%。它与 base 在两种 MTP 模式、三种请求上的完整 message、completion token 数和 finish reason 一致；算术答案仍为 4/391。

HC 候选虽然算术答案仍为 391，但生成长度从 57 变为 65，已改变生成轨迹。不能宣称它无损或直接默认启用；其速度也不能替代 exact 组合成绩。没有运行完整模型 accuracy、长上下文或高并发。

最终证据：`../results/r3d-iqr/implementation/quick-operators-20261118-final/`。算子日志在相邻 `logs/` 下：quick-p1p2-acceptance.log、quick-p2-exact.log、quick-p2-joined-exact.log、quick-p2-model-head.log、quick-p3-split-candidate.log、quick-final-joined.log。最后一项是追加地址溢出保护后的回归；服务计时发生在该防御性修改之前。

### 10.5 后续范围

- 先使用 exact 组合；所有新开关仍为 opt-in，未提交或推送。
- HC 保持实验状态，决定接受何种数值策略后再考虑默认启用。
- 下一轮先采新 profile，再决定 IQR 预取或其他热点，不自动展开大规模优化。

## 11. 后续实施记录：2026-11-18

### 11.1 新 profile 与目标迁移

使用首批 exact 组合，即 IQR dot4 和 logits staging 开启、HC WMMA 关闭；TP2、GPU4/5、MTP0、固定 decode128。只聚合最后一张 rank-0 表，两张卡的耗时不相加。

| 算子 | 累计事件时间/token | 占累计事件时间 |
| --- | ---: | ---: |
| hc_read | 38.84 ms | 33.31% |
| gemm_nt_q | 17.80 ms | 15.26% |
| IQR MoE gate/up | 7.37 ms | 约 6.32% |
| W4NL MoE down | 2.24 ms | 约 1.92% |
| moe_gather | 7.31 ms | 6.27% |
| logits_gemm | 5.09 ms | 4.37% |

对照先前 profile，IQR MoE gate/up 约 16.51 -> 7.37 ms/token，logits 约 24.56 -> 5.09 ms/token。Profile 会改变同步和运行状态，这些数值只解释热点迁移，不是严格配对的延迟加速比。证据：`../results/r3d-iqr/implementation/logs/followup-profile-exact.log`。

### 11.2 实验结果与保留策略

| 开关 | 实现 | 验证与结论 |
| --- | --- | --- |
| RAD_IQR_PREFETCH | MoE dot4 下一组 packed codes 和 weight scale 预取；仅在 RAD_IQR_DOT4=1 时生效 | quick 真实权重逐 bit 通过；多数短 run 慢约 2% 至 9%，少数路由收益不稳定，不推荐启用 |
| RAD_HC_EXACT_TILE | 仅 M=1 改 strict down/up tile；不改变 norm、K 顺序、group fold、激活函数或量化 | 中间 down/logits/gates 和最终输出逐 bit 一致；微测约 3% 至 5% 收益，服务增益很小，保留实验状态 |
| RAD_I8_DOT4 | affine int8 编码、M 不超过 2、K 不超过 16384；16 列/wave、activation/scales 暂存 | 真实权重 exact 测试通过；N=2560 微测单行约 1.2 至 1.4 倍、双行约 1.9 至 2.3 倍；服务结果基本持平，不作为整机优化结论 |
| RAD_IQR_DENSE_WAVE | dense IQR group-BF16 分支使用 wave 整数归约；row/column-f32 的 impl=2 分支使用固定 bit-width 解码和 packed dot4 | group-scale、row/column-scale、真实权重、输出视图及边界测试通过；最终短 A/B 仅小幅变化，尚需重复测量 |

上述开关默认均为 0。RAD_HC_WMMA 仍是非 bit-exact 路径，继续关闭；它与 RAD_HC_EXACT_TILE 是不同实验，不能混淆。

### 11.3 重要派发事实

- op 名 gemm_nt_q 和 profile 的 dtype=i8a8h 不能确定权重编码。affine int8、IQR row/column-f32、IQR group-BF16 是不同路径。
- 声明检查确认 N=8192、K=2560 的重型 dense 项选择 gemm_nt_q_iqr_decode。注册默认 impl=2，即 row/column-scale 的 wave32 group reduction；group-BF16 分支在同一入口单独派发。
- 因此只优化 affine int8，或只优化 IQR group-BF16，不能假设覆盖了 ssm_inz 热点。本轮最后补齐 row/column-scale wave 的 packed dot4 版本。
- `RAD_TRACE_DENSE_SELECTION=1` 配合 `r3d_i8_gated_normal_device_coverage` 可输出所选 kernel 和 geometry；这属于声明事实，不代表实际 GPU 时间。

### 11.4 轻量服务结果

所有对照为相同构建、独立开关、TP2 GPU4/5、profile 关闭，HC WMMA 保持关闭。每配置一个短请求预热，再执行三类请求，每项仅一个测量样本。

第一轮：上一版 exact、再加 affine int8 dot4、再加 HC exact tile 的 decode128：

| 模式 | 上一版 exact | 加 affine dot4 | 再加 HC exact tile |
| --- | ---: | ---: | ---: |
| MTP0 | 9.24 | 9.31 | 9.36 |
| MTP3 | 11.33 | 11.32 | 11.31 |

最终轮：补齐 row/column-scale dense IQR packed 分支，仅比较该开关：

| 模式/工作负载 | 上一版 exact tok/s | 加 dense IQR packed tok/s |
| --- | ---: | ---: |
| MTP0 短问答 | 9.28 | 9.32 |
| MTP0 算术请求 | 8.88 | 8.93 |
| MTP0 decode128 | 9.25 | 9.31 |
| MTP3 短问答 | 12.08 | 12.27 |
| MTP3 算术请求 | 14.86 | 15.09 |
| MTP3 decode128 | 11.31 | 11.48 |

最终 decode128 变化约 +0.65%/+1.47%，不能据单次样本宣称稳定提升。两轮新 exact 候选在两种 MTP、三类请求上的完整 message、completion token 数和 finish reason 均与各自同轮对照一致。算术答案仍为 4/391；未跑完整 accuracy。

### 11.5 复现及证据

服务脚本兼容原四字段 variant，扩展顺序为：标签:IQR-dot4:logits:HC-WMMA:IQR-prefetch:HC-exact-tile:affine-int8-dot4:dense-IQR-wave；省略的后续字段默认 0。

```bash
QUICK_OPERATOR_VARIANTS='previous:1:1:0 iqr_dense:1:1:0:0:0:0:1' \
  bash libr3d-iqr-dev/scripts/quick-operators.sh results/r3d-iqr/implementation/quick-dense-new
```

继续推荐首批 exact 组合：`RAD_IQR_DOT4=1 RAD_LOGITS_STAGED=1 RAD_HC_WMMA=0`，其他新开关保持 0。本轮没有足够证据修改默认配置。

证据目录（相对 `../results/r3d-iqr/implementation/`）：

- `quick-operators-20261118-followup/`：affine dot4、HC exact tile 服务对照，18 条记录。
- `quick-operators-20261118-dense-iqr/`：仅 group-scale wave 版本的服务对照，不混入最终 packed 结果。
- `quick-operators-20261118-dense-packed/`：最终 row/column-scale packed 版本，12 条记录。
- `logs/followup-iqr-prefetch0.log` 和 `logs/followup-iqr-prefetch1.log`：MoE IQR 预取的 quick 精度和计时。
- `logs/followup-hc-exact.log` 和 `logs/followup-hc-exact-narrow.log`：HC exact tile 中间结果及计时。
- `logs/followup-i8-real-exact.log` 和 `logs/followup-i8-bench.log`：affine dense dot4 实权重对照及微测。
- `logs/followup-iqr-dense-exact.log` 和 `logs/followup-iqr-packed-exact.log`：两种 dense IQR 编码的实权重/边界对照。
- `logs/followup-dense-selection.log`：模型声明的 kernel 选择。

本轮未提交或推送。临时服务已停止，GPU 锁已释放。下一步优先细分 HC 的 norm/down/up/inject/quant 时间，再选单个阶段做 exact 优化；不继续盲目搜索 tile，也不默认启用非 bit-exact HC。

## 12. HC norm 优化记录：2026-11-18

### 12.1 分阶段定位

新增仅供诊断的入口 `r3dh_hc_rad_hc_read_profile(args, stream, stage_us)`，输出六个连续阶段的 GPU event 时间，单位 us：norm、down、up、mix、inject、quant。正常服务入口不创建计时事件、不等待计时事件同步；诊断入口只支持 strict 矩阵路径，不支持 HC_WMMA/HC_EXACT_TILE。事件间隔含标记开销，不应等同无 profile 的服务延迟。

完整小样例的 M=1 分解如下，实际接入 inject 和两组量化输出：

| 阶段 | 原 strict us | 并行 norm us |
| --- | ---: | ---: |
| norm | 152.72 | 20.72 |
| down | 140.88 | 137.36 |
| up | 27.24 | 27.64 |
| mix | 10.88 | 10.04 |
| inject | 87.64 | 85.24 |
| 两组 quant | 22.60 | 20.16 |

最初只接最终输出的计时中，inject/quant 是空区间；不能把它们当作已执行算子的耗时。上表和最终微测均使用完整输出。

### 12.2 实现及数值边界

- 开关：`RAD_HC_NORM_PARALLEL=1`，默认关闭。
- 每个 2560-channel 子流一个 workgroup，256 线程分别累加 BF16 平方的 FP64 partial，再做 wave/block 归约。
- 保持 gain、wadd、FP32 均方、sqrt、除法、BF16 RNE 的既有顺序及边界；不更改 down/up、inject、mix、quant 的计算。
- 有限非零输入的有效 BF16 指数跨度不超过 12 时，平方在共同整数格点上至多占 40 bit，2560 项求和最多再需 12 bit，FP64 的 53-bit 尾数可精确表示所有非负 partial 和最终和，因此归约顺序不影响结果。
- 宽指数输入采用保守的 FP32 舍入证书；NaN/Inf 或证书不通过时，使用原顺序的 FP64 求和。第一次只用指数跨度条件的版本服务收益很小，未作为最终方案。

对有限 BF16，平方可在 FP64 中精确表示，且不发生 FP64 上下溢。非负求和按标准最近舍入的累计误差可用 gamma_n 界定；对 n=2560，串行和与并行和之间的相对误差上界远小于所选裕量：

$$
u=2^{-53},\qquad \gamma_{2560}=\frac{2560u}{1-2560u},\qquad
2\gamma_{2560}<6\times10^{-13}<2^{-36}.
$$

令并行 FP64 和为 S。将 `S*(1-2^-36)/2560` 和 `S*(1+2^-36)/2560` 分别舍入至 FP32，只有两端 bit 相同才接受并行结果。裕量也覆盖端点计算和 FP64 除法的舍入。两端不同则按旧顺序求和；使用的可观察边界是 `float(sum/2560)`，不要求宽指数情况下 FP64 的 sum 本身逐 bit 一致。

这条路径不是之前的非 bit-exact HC WMMA 候选；`RAD_HC_WMMA` 继续保持 0。

### 12.3 轻量验收

- 1/4/16 行，9 类输入，直接比较 normalized scratch；同时比较 down、up logits、gates、最终输出、inject、普通/旋转 int8 codes 及 scales，均逐 bit 一致。
- 覆盖指数跨度 12/13、subnormal、宽指数范围、正负零、NaN/Inf，以及 FP32 均方半 ULP 附近和略高于临界点的构造输入。
- 输出 buffer 的前后 guard 检查通过，测试保持非默认队列。
- 完整 HC 调用的 3 次预热、20 次 GPU event 微测：M=1 约 396.55 -> 260.72 us，M=4 396.24 -> 263.20 us，M=16 435.98 -> 303.11 us，即约 1.52/1.51/1.44 倍。该数据来自合成权重和输入，不作为完整模型精度认证。

### 12.4 服务及真实模型 profile

相同构建、TP2 GPU4/5、profile 关闭，仅切换 norm。每种配置一个短请求预热，再执行三类请求，每项一个测量样本：

| 模式/工作负载 | 上一版 exact tok/s | 加并行 norm tok/s |
| --- | ---: | ---: |
| MTP0 短问答 | 9.31 | 10.69 |
| MTP0 算术请求 | 8.85 | 10.17 |
| MTP0 decode128 | 9.27 | 10.66 |
| MTP3 短问答 | 12.09 | 12.95 |
| MTP3 算术请求 | 14.87 | 15.88 |
| MTP3 decode128 | 11.32 | 12.10 |

decode128 约提升 15.0%/6.9%。两种 MTP、三类请求的完整 message、completion token 数和 finish reason 与同轮对照一致；答案仍为 4/391，算术请求仍生成 57 token。未运行完整 accuracy、长上下文或高并发。

另用相同短请求采样真实模型 MTP0 profile：HC 累计事件时间从 38.84 降到 25.42 ms/token，占比从 33.31% 降到 24.90%；dense gemm_nt_q 仍约 17.78 ms/token，其余主要算子基本未变。这是热点验证，不将带同步的 profile 速度当作服务吞吐。

### 12.5 使用与证据

推荐测试组合更新为：`RAD_IQR_DOT4=1 RAD_LOGITS_STAGED=1 RAD_HC_NORM_PARALLEL=1 RAD_HC_WMMA=0`；IQR_PREFETCH、HC_EXACT_TILE、I8_DOT4、IQR_DENSE_WAVE 仍保持 0。没有更改任何服务默认配置。

服务脚本在第 11 节的字段顺序后追加 HC-norm-parallel，省略仍默认 0：

```bash
QUICK_OPERATOR_VARIANTS='previous:1:1:0 norm:1:1:0:0:0:0:0:1' \
  bash libr3d-iqr-dev/scripts/quick-operators.sh results/r3d-iqr/implementation/quick-hcnorm-new
```

测试可通过 `RAD_HC_NORM_BENCH=1 RAD_HC_STAGE_BENCH=1` 启用 `r3d_hc_quick_wmma_candidate`；该历史测试名同时承载 exact norm 模式，日志中会明确标记 `HC_NORM_PARALLEL_BENCH`。测试须在既有容器环境中持有 GPU 锁运行，不能将这些变量误作普通服务配置。

证据均在 `../results/r3d-iqr/implementation/`：

- `quick-operators-20261118-hcnorm/`：早期仅指数跨度条件的版本，不与最终成绩混合。
- `quick-operators-20261118-hcnorm-certified/`：最终服务 A/B，12 条记录及库 SHA256。
- `logs/hcnorm-final-check.log`：最终完整输出、舍入边界、guard 和微测结果。
- `logs/hcnorm-profile.log`：开启 norm 后的真实模型最终算子表；只取最后一张每 rank 表。

本轮未提交或推送。临时测试容器已清理，GPU 锁已释放。下一步以 HC down 和 inject 为候选，各自先做一个局部假设，不继续改 norm，也不默认开启近似矩阵乘。
