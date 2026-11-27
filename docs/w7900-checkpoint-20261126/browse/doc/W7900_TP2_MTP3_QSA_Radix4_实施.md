# W7900D TP2：四行 QSA radix 实施与验证

**独立复测（2026-11-26）。** [当前优化版 TP2 三档对照](../results/tp2-radix4-retest-20261126/report.md)：换用最新无计数器服务端，保持相同 radix4 插件、双卡和优化开关，不启用辅助信号延迟回收。固定输出 256 token，32,768 输入时服务端解码 MTP0 / MTP3 为 **55.27 / 62.50 tok/s**，首 token 为 **102.41 / 147.34 秒**，整请求为 **107.02 / 151.42 秒**。55-token 说明文为 61.73 / 66.00 tok/s，71-token 代码为 61.62 / 83.43 tok/s。每格一次预热后测两次；这不是回收策略的 A/B，亦非多轮配对验收。

目标是在不改变已有单行实验路径的语义和默认值的情况下，让单序列四行 MTP 验证使用逐行精确 Top-512。本文记录候选实验，不替代[原单行基线](W7900_TP2_MTP0_MTP3_QSA_TopK_加速原理.md)。

**结果：本轮 32K 解码目标达到。** [完整实测表](../results/mtp3-radix4-20261126/report.md)与[可复算 JSON](../results/mtp3-radix4-20261126/summary.json)：四行开启后的 MTP3 为 62.51 tok/s，较同构建关闭时提升 133.18%，较同次新 MTP0 快 13.23%。[冻结旧插件复测](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/speed-0-F/summary.json`）的 MTP0 长/短档为 55.26 / 61.73 tok/s，新候选相对变化 -0.09% / -0.17%；未观察到超过 1% 的回退。另有[原有 QSA 集成回归](../results/mtp3-radix4-20261126/gpu-integration.log)通过 27,215 项检查，包括 144 个 fixture 的打分加选块流水。尚未覆盖其他模型、更多提示词或正式多轮配对，不据此推广默认值。

| 阶段 | 状态与证据 |
|---|---|
| 内核 | 已实现。单行仍采用原符号、1 个块，编译期行号为 0；四行采用独立符号、4 个块，每块 256 线程。位置、分数、页输出和两个计数逐行寻址；块表、块数在单序列内共享。`RAD_QSA_SELECT_RADIX4=1` 仅在原 radix 同时启用且满足原架构、宽度、K 等条件时生效；未设置或非 `1` 不启用。 |
| 正确性 | [gfx1100 设备测试](../results/mtp3-radix4-20261126/gpu-contract.log)已通过：38,483 项检查，380 个 fixture；两个精度、两条 stream；逐字节比较独立 CPU 预期、参考库及 GPU，可见位置跨界、并列、不同赢家、尾页、页间填充和回退均纳入检查。 |
| 无插桩服务 | 同一服务二进制 A/B 已完成；两臂都启用单行 radix，只改四行开关。每个 32K/256 格子独立启动，1 次预热、2 次计量；MTP0 加测 55-token 短档，1 次预热、4 次计量。 |
| 设备归因 | 已完成。[关闭轨迹](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/trace-3-A/passes.json`）与[开启轨迹](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/trace-3-B/passes.json`）各有 93 次完整验证；逐轮 12 次预期选择器由[独立 CSV 检查](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/verify-trace.pl`）确认。打点不计入服务吞吐。 |

[复现实验入口](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/serve.sh`）。保留原二进制和历史数据，不自动提升默认开关。小样本不等于正式多轮性能验收。

**使用候选。** 在启动服务的容器上同时传 `-e RAD_QSA_SELECT_RADIX=1 -e RAD_QSA_SELECT_RADIX4=1`；关掉第二个（或显式设为 `0`）即可回到原单行 radix + 四行旧堆，不必重编译。不要改变原有 MTP 八开关、TP2 配置或关闭稀疏选块的并行选项来做比较。限定范围为本次 gfx1100、单序列、四行、K=512、宽度 513–9,216；其他形状或不支持的设备沿用已有路径。

**MTP3 同构建 A/B 已完成。** [A 四行关闭](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/speed-3-A/summary.json`）为 26.808 / 26.809 tok/s，[B 四行开启](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/speed-3-B/summary.json`）为 62.519 / 62.506 tok/s。两格中位数 **26.81 → 62.51 tok/s（2.33 倍）**，相同文本、相同草稿计数 159/288。首 token 147.23 → 147.24 秒，整请求 156.74 → 151.32 秒。客户端高于服务端 1.07% / 1.11%，超过此前 1% 诊断线。原服务端和架构二进制不变，[候选指纹](../results/mtp3-radix4-20261126/serving.sha256)只替换 GPU 插件；F 为原插件重新运行的 MTP0 回归对照。

**MTP0 同插件对照已完成。** 32K [A](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/speed-0-A/summary.json`） / [B](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/speed-0-B/summary.json`） 为 55.28 / 55.21 tok/s（-0.14%），短档为 61.80 / 61.62 tok/s（-0.29%）。这条 32K 输入的新 MTP3 **解码**比新 MTP0 快 **13.23%**。但生成固定 256 token 的**整请求**仍是 MTP0 快：106.99 秒对 MTP3 的 151.32 秒；首 token 分别为 102.37 / 147.24 秒。此次改动不解决预填充。

**同构建的设备阶段变化（rank 0）。** 主模型 12 次选块合计 **52.460 → 0.747 ms**，四行验证 **87.13 → 36.60 ms**；历史处理加首草稿的选块也从旧堆切到四行 radix，该段 **8.19 → 4.00 ms**。第二、第三草稿仍走原单行 radix，每次选块约 0.055 ms。验证到最后草稿的整轮设备跨度 **101.24 → 46.52 ms**，含阶段间隔；由无插桩服务吞吐折算的每轮成本则是 **99.08 → 42.49 ms**，不能混为同一次计时。最大收益对应验证阶段，不应把全部收益只记在 12 次选块上。

静态核对：单行新旧主体各 707 条指令；14 处不同编码均为可交换整数加法、与、或的输入操作数换序。单行和四行各为 wave32、29 VGPR、每块 37,460 字节 LDS、0 私有分配。这不是吞吐回归证明。仅四行增量补丁（历史引用，未随包保存：`results/mtp3-radix4-20261126/radix4-only.patch`）从已归档的单行版本起仅改内核和测试两个文件；单行及四行局部补丁（历史引用，未随包保存：`results/mtp3-radix4-20261126/qsa-with-single.patch`）从 `0405f357` 起，两者各自在独立 Git 索引回放后与本次源码逐字节匹配；不包含其他八个 MTP 优化的全部实现。

复现构建：使用原 `radiance-r3d-dev:rocm10.1-ffmpeg` 镜像，CMake Release/Ninja，`RAD_GPU_TARGETS=gfx1100`、`RAD_BUILD_TESTS=ON`、`CMAKE_CXX_FLAGS=-include x86intrin.h -g`、C/C++ 与 HIP 编译器取 `/opt/rocm/llvm/bin/clang{,++}`、`CMAKE_HIP_COMPILER_ROCM_ROOT=/opt/rocm/core-10.1`。构建 `r3d_mtp_ple_device_test` 目标，在 gfx1100 真卡上设置 `RADIANCE_HOME` 指向新构建的 `radiance_home`、`RAD_R3D_REQUIRE_DEVICE=1`，运行 `r3d_mtp_ple_device_test r3d_qsa_radix_dispatch_contract`。已归档的[原始会话复核工具](../../README.md#evidence)（包内：`results/mtp3-radix4-20261126/report.pl`）从工作区根目录运行，不会再占 GPU。