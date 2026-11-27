# W7900D 开发分支恢复点（2026-11-26）

本目录随 `zihaomu/radiance` 的 `rdna3_w7900_qwen3.8_flash_dev` 分支保存。它记录的是开发候选，不是默认配置、模型发布或上游 PR。此前已经推送的基线为 `0405f3578d16af56cef642df5119434f4a271f9a`。

## 在另一台机器取回

```bash
git clone --branch rdna3_w7900_qwen3.8_flash_dev --single-branch https://github.com/zihaomu/radiance.git
cd radiance
git log -4 --oneline
```

恢复源码和阅读数据不需要 GPU。要重跑服务，需要可用的 gfx1100 ROCm 环境和另行提供的 Qwen3.8 Flash Next IQR-V4 `.rad` 模型。模型（121678372864 字节，SHA-256 `25aff8c950f4b823bc8b77c8184ed48827cb47058ffce08ae2c83b12e021fea0`）、容器镜像和编译产物不在 Git 中。

## 实现与结果

已保存的源码提交（按依赖顺序）：`d942273`（TP2 MTP 算子及测试）、`ac43eb0`（单行与四行 QSA radix）、`aa93302`（AQL 信号延迟退休对照）。查看单项改动可用 `git show --stat <提交>`。这些是实验候选，未启用新默认值。

- TP2 MTP 候选：验证词表头、四行 dense/MoE/HC、草稿投影、IQR 与 rerank 的优化及 GPU 对照测试。开启实验时原有八个 `RAD_MTP_*` 开关保持不变。
- QSA Top-512：单行 radix 使用 `RAD_QSA_SELECT_RADIX=1`。在此基础上，`RAD_QSA_SELECT_RADIX4=1` 允许单序列四行验证每行独立选页；关闭第二个开关即可只保留单行路径。四行使用真实 GPU/参考库逐字节比对，并有单行回归。
- AQL：`RADIANCE_XP_DEFER_RIDE_RETIRE=1` 只用于专家预取辅助流的延迟退休对照，默认关闭；没有证据支持推广为加速选项。当前 TP2 不启动这条预取路径。
- 32K/256、无前缀命中、单流：同构建 MTP3 从 26.81 升至 62.51 tok/s。最新独立复测 TP2/MTP0、TP2/MTP3 分别为 55.27、62.50 tok/s；整请求分别为 107.02、151.42 秒。解码吞吐不含预填充。
- 真正的拓扑对照须固定 MTP 深度：20 GiB 主机专家池，MTP0 单卡、双卡分别为 38.30、55.31 tok/s。16 GiB 下 TP1/MTP0 在此配置装载失败；TP1/MTP3 有未覆盖的全词表等算子形状，没有 32K 速度成绩。

以上是特定请求的小样本测量。独立请求数、设备回放数和 GPU 检查项不能相互替代。

## 浏览报告

- [QSA 选块原理](browse/doc/W7900_TP2_MTP0_MTP3_QSA_TopK_加速原理.md)和[四行 radix 实施记录](browse/doc/W7900_TP2_MTP3_QSA_Radix4_实施.md)
- [最新 TP2 长短输入实测](browse/results/tp2-radix4-retest-20261126/report.md)和[固定 MTP0 的 TP1/TP2 32K 实测](browse/results/tp1-tp2-32k-20261126/report.md)
- [MTP3 设备轨迹归因](browse/results/32k-mtp3-verify-20261125/report.md)和[AQL 三臂对照](browse/doc/AQL_辅助等待与延迟信号回收_对照.md)

<a id="evidence"></a>

## 解包与校验

从仓库根目录运行以下 Linux 命令。需要 Git、GNU tar、zstd、sha256sum；数值校验还需要 Perl。归档是普通 Git 文件，不需要 LFS、额外下载链接或原机器。

```bash
cd docs/w7900-checkpoint-20261126
sha256sum -c SHA256SUMS
mkdir restored
tar --zstd --no-same-owner --no-same-permissions -xf evidence.tar.zst -C restored
cd restored
sha256sum --quiet -c CHECKPOINT-SHA256SUMS
perl results/mtp0-topk-radix-20261125/pages-oracle.pl \
	results/mtp0-topk-radix-20261125/pages-A/*.bin \
	results/mtp0-topk-radix-20261125/p2-pages-*-snapshots/*.bin
```

预期：包中 3,011 个精选文件（另附逐文件校验清单），336 个真实选块结果通过独立 oracle。包括冻结请求、逐次输出与计时、选定 GPU CSV 轨迹、数值夹具、技术报告及历史运行脚本。浏览版文档把包内引用标为恢复路径，未收录的历史引用另行标注。

这是**项目恢复点而非全机镜像**。公开包使用通用工作区和锁目录路径代替原机绝对路径，并将个别 PCI 地址改为代号；少数日志仅保留诊断行，运行环境仅保留镜像指纹和白名单开关。原可执行文件指纹用于辨认版本，不代表包内附有二进制。旧脚本若依赖模型、镜像、设备锁、完整容器检查记录或完整日志，不能直接在此公开子集上重跑，需重新配置或取得这些外部前提。