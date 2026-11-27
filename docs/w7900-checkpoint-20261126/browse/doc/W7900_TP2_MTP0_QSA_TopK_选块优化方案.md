# W7900D TP2 / MTP0：长上下文 QSA Top-K 选块优化方案

**P3 状态（15:29:54 +08:00）：** `retry-01` 的首个 A 因请求后两卡均出现额外 PID 而整组排除；未形成正式 A/B 配对，未启动第三次尝试。见[停止原因及原始快照](../results/mtp0-topk-radix-20261125/retry-01/stop.md)。

状态：**P0、P1 和 [P2 服务轨迹](../results/mtp0-topk-radix-20261125/p2-report.md)均已通过。** 2026-11-25 14:49（北京时间）进入 P3；无插桩服务 A/B 的各次启动、异常及最终结论见[服务验证记录](../results/mtp0-topk-radix-20261125/serving-progress.md)及第 7 节。范围限 Qwen3.8 Flash Next、gfx1100、TP2/MTP0 单流解码；默认未改。

## 1. 为什么先做这一项

[五档同配置报告](../results/mtp0-context-20261125-a/report.md)：2,016 → 2,064 token，墙钟 +4.476 ms/token，rank-0 选块 +4.348 ms/步；2,064 → 32,768，墙钟 +43.779，选块 +43.501，打分 +0.052，选中 KV 的 attention -0.002。32K 每步跨 12 层，选块共 47.849 ms，其中 47.820 ms 与两类 GEMV 区间重叠。**不把 47.849 或 43.501 ms 当成可兑现的墙钟收益。**

可证伪假设：瓶颈在[有效块多于 K 后交给线程 0 的堆和回扫](../../../../libr3d/r3dh_sparse_qsa_index.hip#L220)。仅替换这个选择过程，应能在页表完全相同的前提下降低 32K 不插桩解码时间。若隔离测试变快而同配置服务不快，则否定这次实现的端到端收益，不据此改默认值。

## 2. A/B 边界与不可变契约

**A** 保留当前 `select_kernel_parallel`；**B** 计划在 `r3dh_sparse_qsa_select` 加一个独立、默认关闭的开关 `RAD_QSA_SELECT_RADIX=1`（拟用名）。放在已有 shape 校验之后，不改注册优先级。首版只在原并行选块已启用、gfx1100 wave32、单行单序列、`topk=512`、ratio=4、且 `512 < min(score_columns, table_columns) <= 9216` 时考虑新内核。9216 是本原型的 LDS 容量上限（**单位是块，不是 token**），不是模型上限；32K 约 8192 个完整块，须用实际调用宽度确认命中。在 GPU 内据真实 `complete` / `live` 判断是否剪枝。未命中条件仍走原分派；原并行开关关闭时保留原标量入口。只在 MTP0 TP2 测试进程中显式打开，不能从单行 shape 推断请求一定是 TP2 或解码；等形状的 prefill 尾行也必须正确。

新开关只认显式字符串 `1`；不能直接用 [r3d_exact_path](../../../../libr3d/r3d_flags.h)，该 helper 对未设置的变量返回 true。原 `RAD_QSA_SELECT_PARALLEL` 按旧语义读取。A/B 都保留旧并行开关为开启；禁用旧并行路径的测试只用于回退正确性，不拿标量路径作为性能 A。使用现有设备注册/编译目标或缓存的设备属性做架构保护，不在每层 dispatch 同步查询 GPU，也不从主机读取 `count`、`positions` 或 `live`。

以[旧 `candidate` / `select_row`](../../../../libr3d/r3dh_sparse_qsa_index.hip#L125) 和[独立 CPU oracle](../../../../tests/r3d_qsa_index_aql.h#L45)为准：

- `complete = min(count[seq], score_columns, table_columns, (position+1)/4)`；保留原序列归属和 tensor stride。按**逻辑列**比较/稳定输出，最后才查表取物理页号；页号可乱序、可重复，不去重。
- 页号 `<0` 无效。signed 模式只接受有限分数；ReLU 模式先执行 `value > -1e30f`，再把非正值归零，因而拒绝 NaN、负无穷及截止值，**接受正无穷**。先调用相同的候选判定，不用通用 `isfinite` 替换两种语义。
- `live <= 512` 时所有有效完整页均按原列序输出，**等于 512 时也不能重新分组**。`live > 512` 时令 θ 是第 512 大的有效分数：先按列序输出全部 `score > θ`，再按列序从 `score == θ` 中取足 512。两组分别稳定，不是全局按列序，也不是按分数排序。例如 K=3、分数 `[8,5,9,5]`，输出列应为 `[0,2,1]`。
- 最后才按旧条件追加未满 4-token 的尾页：`complete < width && (position+1)%4 != 0`，且 `table[complete] >= 0`；它不参与阈值选择。记总页数为 `written`，填 `selected[written..512] = -1`，`nselected=written`，`used=written ? (written-1)*4 + position%4 + 1 : 0`。不得触碰输出 pitch 的 padding、输入或其他行。

## 3. 第一候选：单工作组精确 radix-select

每行 **1 个工作组、256 个线程、8 个 wave32**。仍是一个 select dispatch；无新全局 scratch、无跨工作组同步、不把 8192 个页号全部排序。第一版限定宽度，优先验证收益，不把多工作组流水线或 score/select 融合混入同一个 A/B。

1. **并行读入一次。** 按 `column = tile*256 + tid` 连续读取。在设备上计算 `complete`；无需剪枝时复用现有无剪枝逻辑。对可能剪枝的行，按原规则过滤分数，将有效 F32 转为可按无符号整数比较的 key 存入 LDS：两种零统一为 `+0`，负数位模式取反，非负数位模式翻转符号位。整数 key `0` 留作无效哨兵（最小合法有限负数映射后仍大于 0）；不对无效/NaN 做排名。如果有符号极小数在目标编译模式下的比较与位序不一致，差分必须拦下，先回退，不放宽误差。若发现 `live<=512`，走稳定全收分支，不求 θ。
2. **并行求阈值，不做全排序。** 维护 1-based 剩余名次 `rank=512` 和已确定的高位前缀。从最高 nibble 向最低 nibble 做 8 轮 16 桶计数；每轮只计有效且匹配该前缀的 key。每个 wave 对连续 32 列形成有效掩码和 4 个位掩码，lane 0..15 分别 `popcount` 自己的桶，在寄存器跨 tile 累加。写 `hist[8][16]` 后按桶合并；从高桶向低桶累计，选中包含 `rank` 的桶，减去所有更高桶的计数。只允许最多 16 桶的串行决策，不再由一条 lane 扫几千列；所有 lane 都参与 ballot 和屏障，尾部越界列只置无效。8 轮后得到精确阈值 key，且严格更高的总数 `G=512-rank`。
3. **稳定写回。** 再按 tile 从低列到高列遍历 LDS；分别对 `>θ` 与 `==θ` 做 wave ballot、lane 前缀计数、wave 间前缀和，再叠加之前 tile 的总数。严格更高的候选写 `[0,G)`；同分候选仅当组内全局序号 `<512-G` 时写 `[G,512)`。输出是原表对应的页号，不是 key 或列下标；随后统一处理尾页、`-1`、`nselected/used`。禁止用原子抢输出槽位，避免同分时页序随调度变化。

**资源与备选。** 9216 个 32-bit key 占 36 KiB LDS，分桶表另 512 B，前缀计数少量；编译后以实际 LDS/VGPR、spill、支持的工作组大小为准。单序列只有一个工作组，这是首版的局限。若 8K 块微测仍慢，先量出阈值阶段和稳定写回各占多少；只有确认单工作组并行度不足，才另案做“多组分片统计 + 全行阈值 + 稳定回写”，单独评估 scratch、额外 launch 和依赖。拒绝近似阈值、半精度量化和只比页集合的方案。

## 4. 验证顺序与停止条件

**已做的是 CPU 设计验证，而非 GPU 验收。** 在工作区根运行 `perl scripts/mtp0-topk-design-check.pl`：[可复跑的检查](../../README.md#evidence)（包内：`scripts/mtp0-topk-design-check.pl`）已通过 **1,440 组**，对每组逐字节比较 513 页槽、`nselected`、`used`。参考侧用数值排序取前 512，再按契约还原输出顺序；候选侧按 8 轮整数 radix 求阈值。覆盖符号模式、特殊值、重复页、位于不同 tile 的并列及有效/无效/缺失尾页。两侧在同一个给定 `complete` 下比对；它不验证设备如何从张量求这个边界，也不验证 wave 指令、屏障、编译器浮点模式或速度，这些属于 P0/P1。

| 阶段 | 怎么做 | 通过条件及处理 |
|---|---|---|
| P0：先锁正确性 | 扩展[现有 AQL 差分](../../../../tests/r3d_qsa_index_aql.h)：现有最大 score 宽 514，不代表 8192 活跃块。用 CPU oracle、libref、旧 GPU A、新 GPU B 四方比较；强制检查 B 确实命中新符号，而非静默回退。 | 比较全部 513 个页槽、`nselected/used`、padding/guards、输入不变；所有输出字节零差异。在支持范围以外同样检查旧路径被保留。任一差异即停。 |
| P1：隔离设备计时 | 同一输入和 stream 的 A/B、B/A，预热后多次取中位数；测约 512、2K、8K **块**。覆盖大量并列和无效块，并用独立诊断进程采的真实层输入回放，保存幅度和分布；不把合成数据说成真实模型。 | 提议筛选线：8192 个有效块时 select 中位数至少快 2×，且剪枝边界无明显倒退，再进入服务验收。失败先分解资源/局部耗时；微测通过不等于项目接受。 |
| P2：设备轨迹 | 单独服务复跑 2016、2064、8192、32768 **token** 的 32 输出诊断，对比调用符号、次数、选块时间、相邻 GEMV 重叠及整步跨度。 | 2016 仍无独立选块，长档应实际进入 B；打分和 attention 分派不变。同一输入 A/B 的页表与文本一致。仅作归因，不以插桩 tok/s 验收。 |
| P3：不插桩服务 A/B | 按下述预注册口径在 GPU4/5 跑同一候选二进制的开关对照。 | 以 32K 的服务端配对结果判定；仅 kernel 下降、服务没过线，结论就是未达标。保持默认关闭。 |

P0 必测：无效后实际 `live=0/1/511/512/513`，以及 2048/8192/8193 个候选和 9216 容量边缘；所有分数相等、阈值并列跨 wave/工作组内 tile、递增/递减、打乱/重复页号、负页号、随机确定种子；`±0`、相邻浮点数、次正规数、NaN、`±Inf`、`-1e30f` 及相邻值；两种 signed 模式、`position%4=0..3`、无效尾页、表比 scores 宽/窄、带 padding 的 pitch、重复启动与两条 stream。多行/多序列、其他 K、宽度 9217、旧并行开关关闭均验证 fallback。原有非法形状的拒绝不能因新开关被绕过。以同一输入同时检查 exact 与默认数值模式；不要求两种模式彼此输出相同，而要求各自 A/B 一致。

## 5. 不插桩服务验收（2026-11-25 实施前已锁定）

- **隔离变量。** 在新目录从已归档候选构建，不覆盖旧二进制；同一新二进制 A=新开关关、B=开，其余八个历史开关及环境固定。保存补丁、插件/可执行文件、模型、镜像、配置、输入 SHA。GPU4/5 独占锁及每请求前后 PID 复核。TP2 exact wire，MTP0，单序列，`max-model-len: 33792`、分块 512、int8 KV、主机池 16384 MiB、`no-prefix-cache: true`；无 AQL 时间戳、profiler、诊断转储。temperature 0、固定 seed；A/B 须使用相同请求字节。
- **主指标。** 使用[同一冻结 32K 请求](../../README.md#evidence)（包内：`results/mtp3-dev-p0p4/archive32k-20261125-b/request.json`），固定输出 **256**。原服务 MTP0 的 [16 样本中位数 15.294 tok/s](../../README.md#evidence)（包内：`results/mtp3-dev-p0p4/archive32k-20261125-b/summary.json`）仅是漂移参考，不代替同批 A；此前 15.22 tok/s 属于输出 **32** 的诊断。新二进制关闭开关先与冻结参考复核文本；若新 A 与旧 MTP0 基线偏离超过 3%，先排查环境，不把偏慢基线当收益。
- **配对与判定。** 主档预排四对独立启动 `AB / BA / BA / AB`；每次同题至少 4 次完整预热，末 3 次速率 max/min ≤1.05（最多 12 次），再测 8 次，预热不入样本。每臂共 32 条正式请求；每对用各自服务端 tok/s 中位数之比 `B/A`，报告全部四对、配对倍率中位数、ms/token、客户端对照、TTFT 和整请求耗时，不混合各长度。提议验收线：32K 配对中位数 ≥1.10×，且至少 3/4 对更快；TTFT 的配对中位数 `B/A` 不超过 1.03。速度和耗时的比值方向不同。
- **短档与边界。** 使用五档固定请求（历史引用，未随包保存：`results/mtp0-context-20261125-a-prepared/ready`）的 **32 输出**口径；尤其 2016+32 不跨 2051，不可把它悄悄改成 256 输出。每档在干净 A/B 和 B/A 服务对中分别预热并取 8 次，报每档配对结果；1024、2016、2064、8192 的 `B/A` 中位数均 ≥0.99。短档 A/B 若走相同旧路径却明显变慢，先查计时环境。32K/32 输出只用于与五档历史对照，不代替主档。
- **资格与回退。** 要求同档各次 `usage.prompt_tokens == timings.prompt_n == 目标长度`、cache=0、固定输出长度、draft=0、A/B 文本 SHA 一致；每启动客户端/服务端吞吐中位数偏差 ≤1%，无外部 GPU 进程、prefetch 状态切换或热身不稳。失败样本保留，标明原因，不删除慢样本。默认不变，不发布；即使通过也只证明所测 TP2/MTP0 配置，是否提升为默认另行决定。

## 6. 实施记录

- **P0 开始。** 新增可观察分派的设备测试；冻结构建 `build-dev-mtp3-20261124` 只读，新实现独立构建。第 4、5 节的性能门槛和请求顺序不随结果调整。P1、P2、P3 尚未执行。
- **P0 首轮。** 冻结指纹与预期红灯（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/p0-red.log`）已核对；实际 HIP 单元编译成功。新夹具把一维向量也设成了非单位 stride，被 A 即拒 `RAD_E_SHAPE`，原紧凑 QSA 回归仍通过；失败记录（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/p0-device.log`）保留。仅修复夹具一维布局，矩阵 pitch/guards 不变，未放宽生产校验。
- **P0 设备门槛通过，P1 开始。** 实机日志（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/p0-device-v2.log`）：GPU4/gfx1100，38,470 项检查/2 个用例，零失败。新用例 264 组，实际 dispatch 208 次 radix、56 次回退，四方差分含 25,010,000 个设备缓冲字节，双 stream、默认/精确模式；旧 QSA 回归亦通过。候选编译在 `build-dev-topk-20261125`，未改冻结二进制。尚未测得服务端加速。
- **P1 合成筛选通过，真实回放待验。** 设备 event 微测（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/p1-synthetic.log`）：8192 块随机/同分/带无效页分别 105.16× / 57.32× / 83.87×；随机 4882.22 → 46.43 µs，513 块 343.28 → 16.90 µs。512 块仍是同一路径（约 6.8–7.1 µs）。四个样本/臂，各 20 次 launch；所有计时输出均校验。仅为隔离筛选，不当作服务结果。
- **P1 真实回放通过。** A 组捕获（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/p1-capture-A.log`）使用独立诊断用架构插件；48 份回放（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/p1-real.log`）覆盖 2 个 rank × 12 层 × 2 个位置 (32768/32769)，实际 8192 个完整块，列宽 8194/8448。四方逐字节一致并等于实机捕获的 513 个页槽、`nselected` 和 `used`。每份微测的 A/B 时间再取跨快照中位数为 4048.42 / 47.16 µs，配对倍率范围 68.13–106.04×，中位数 84.50×。准备冻结 P2/P3 插件时再加严格的单目标 gfx1100 构建守卫并重复 P0/回放；未改变分桶算法，未取得不插桩服务数据。

## 7. 最终二进制的服务验证

**独立追加实验已完成：32K × MTP0/MTP3 × 旧/新选块。** [四格实测表](../results/32k-topk-mtp-depth-20261125/report.md)使用同一二进制、相同 32,768-token 档案，零前缀缓存、固定输出 256 token；每格一次预热后测两次。旧→新服务端解码：MTP0 **15.30 → 55.36 tok/s（3.618×）**，MTP3 **24.84 → 26.81 tok/s（+7.92%）**；本次新 MTP0 的解码速度为新 MTP3 的 **2.07×**。MTP3 整请求只从 157.40 降至 156.70 秒，首 token 仍约 147 秒。MTP3 两臂均接受 159/288 个草稿，同一深度 A/B 输出 SHA 相同。MTP3 客户端读数分别高于服务端 1.08% / 1.07%，超过原 1% 诊断线。本表为小样本实测，不并入正式 P3，不将单行内核的收益外推到四行验证。

**为什么新实现下 MTP3 落后于 MTP0。** [MTP0 单独插桩轨迹](../results/mtp0-topk-radix-20261125/p2-report.md)显示，32K 的 12 次单行选块合计从 47.864 降到 0.721 ms/步；无插桩服务的每输出 token 时间从 65.35 降到 18.06 ms。MTP3 是三次串行草稿加一次主模型验证：符合条件的单行可以走 radix，但带满 3 个草稿的四行验证仍走旧选块。由[四格服务计数](../results/32k-topk-mtp-depth-20261125/summary.json)折算，每 3 个提案为一组，共 96 组；剔除首 token 后，255 个输出间隔平均每组 2.656 个。MTP3 的等效整组解码时间从 106.94 降至 99.09 ms；要以本次接受率追平新 MTP0，每组最多约 47.98 ms。这是整段解码摊算，不是验证或选块内核的逐轮计时。

两种新模式的整请求相差 49.72 秒，其中首 token 前相差 44.81 秒，后续解码相差 4.91 秒。多行验证是解码侧下一步要测的对象，不是首 token 差距的解释；要缩短这条 32K 请求的总等待，应单独分析预填充。

**四行验证已单独打点。** [分阶段设备实测](../results/32k-mtp3-verify-20261125/report.md)：同一 32K/256 请求，MTP3 93 个完整回放中，四行验证平均 87.23 ms，历史加首草稿 8.19 ms，后二次草稿 2.17 / 2.15 ms。验证 12 次旧堆选块的区间并集是 52.47 ms，而 MTP0 252 次单行回放每步的 12 次 radix 选块只有 0.72 ms。52.45 ms 与其他 kernel 重叠，不等于可直接省下的墙钟时间。此前 99.09 ms 仍是无插桩的整轮折算值。

最终 gfx1100 代码对象元数据（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/isa/metadata.log`）：wave32，最大工作组 256，LDS 37,460 B，29 VGPR / 38 SGPR，private segment 与两类 spill 均为 0。[真实层分布](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/real-score-distributions.tsv`）：48 份的有效分数总体从 0 到 15.10297，本组每份阈值相等数均为 1；并列正确性另由 P0 及合成压力检验覆盖。主档请求字节完全不变；32 输出短档保持原非流式请求作为正式速度，在同一服务另做 8 次仅改变流式传输参数的 client/server 对照，不能混入正式速率。

P2 阶段读数：2016 的 A/B 都不调选块，2064 的[A](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-2064-A/decode.json`）/[B](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-2064-B/decode.json`）共 12 次/步，4.358 → 0.208 ms，48 对页表相同；8192 的[A](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-8192-A/decode.json`）/[B](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-8192-B/decode.json`）为 16.691 → 0.324 ms/步。这些均带 AQL 时间戳，不能充当正式不插桩成绩。

**P2 32K 干净轨迹完成。** [A](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-32768-A/decode.json`） / [B](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2-32768-B/decode.json`）均为 12 次选块/单行步，rank-0 选块合计 **47.864 / 0.721 ms/步**；打分 0.125 / 0.125、attention 1.414 / 1.416，tape 首末跨度 67.033 / 19.763 ms/步。均用零缓存 32,768-token 输入、32-token 输出，全文 SHA 相同。这不是 256 输出的无插桩 A/B；不能以带戳 tape 跨度替代 P3。

**复现边界。** 在单独工作树 `0405f35` 上先应用前序冻结补丁（历史引用，未随包保存：`results/mtp3-dev-p0p4/candidate.patch`），再应用本次增量补丁（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/topk-tracked.patch`）（已在临时索引逐文件回比）；不要重建冻结基线。实测镜像 `radiance-r3d-dev:rocm10.1-ffmpeg`、gfx1100、Ninja Release、配置记录（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/configure.log`）。用标准 `r3d_mtp_ple_device_test` 运行 `r3d_qsa_radix_dispatch_contract`，另以 `RAD_QSA_RADIX_BENCH=1` 运行 `r3d_qsa_radix_select_benchmark`；可传 `RAD_QSA_RADIX_REPLAY=<实测快照>` 核对真实层。服务脚本保留组顺序和 GPU 锁：[P2](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p2.sh`）、[P3](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p3.sh`）。本次目录保存证据，不要向其中重跑；复测须另建目录并修改驱动输出根。正式资格汇总器[离线自测](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/test-p3-qualification.pl`）的虚构速率只在自动清理的临时目录出现，不能代替真实服务样本。

- **P0/P1 已在最终候选重验，P2 开始。** 架构限制改为单目标 gfx1100 才能选择 B，非 wave32 在设备端回退。ROCm Clang 未提供尝试使用的 wave 宏（失败编译（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/final-build.log`））；改用目标 builtin，不放宽守卫。[最终 P0/P1](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p0-p1-final.log`）共 42,183 项、3 个用例；[真实回放重验](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/p1-real-final.log`）48/48。[候选指纹](../../README.md#evidence)（包内：`results/mtp0-topk-radix-20261125/candidate.sha256`）和本次增量补丁（历史引用，未随包保存：`results/mtp0-topk-radix-20261125/topk-tracked.patch`）已冻结。P2 用干净架构采时间戳、诊断架构查页表，P3 不带上述仪器。
- **2026-11-25T14:49:45+08:00** P2 已通过。见[P2 设备报告](../results/mtp0-topk-radix-20261125/p2-report.md)；2016 无 select，2064/8192/32768 为 12 次/步；三档共 144 对实机页表一致。P3 开始，不以插桩吞吐作成绩。

- **2026-11-25T15:05:58+08:00** 阶段停止：P3，p3-main-1-A；见[记录](../results/mtp0-topk-radix-20261125/serving-progress.md)。未宣称验收通过。

- **2026-11-25T15:13:42+08:00** P2 已通过。见[P2 设备报告](../results/mtp0-topk-radix-20261125/p2-report.md)；2016 无 select，2064/8192/32768 为 12 次/步；三档共 144 对实机页表一致。P3 开始，不以插桩吞吐作成绩。

- **2026-11-25T15:29:54+08:00** 阶段停止：P3，retry-01/p3-main-1-A；见[记录](../results/mtp0-topk-radix-20261125/serving-progress.md)。未宣称验收通过。
