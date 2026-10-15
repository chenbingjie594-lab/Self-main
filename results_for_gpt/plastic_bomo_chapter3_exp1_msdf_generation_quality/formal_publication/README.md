# Experiment1 正式结果与本地复核

正式状态：`MSDF_VISUAL_METRICS_NOT_IMPROVED`。实验已完整完成，仅评价冻结 Raw SD2 与 MSDF 的视觉指标。

## 论文主结果：缺陷区域

| Method | ROI FID ↓ | ROI KID ↓（未缩放） | ROI LPIPS（多样性） |
| --- | ---: | ---: | ---: |
| Raw SD2 | 172.160 ± 2.273 | 0.034258 ± 0.002864 | 0.098340 ± 0.051545 |
| MSDF | 188.548 ± 4.922 | 0.046853 ± 0.002047 | 0.115825 ± 0.070651 |

FID/KID 为 A/B/C 三个独立评价 bank 的均值和样本 SD；LPIPS 为同 slot 三种 bank 配对距离先取均值，再对 80 个 slot 汇总的均值和样本 SD。二者 SD 的统计单位不同。KID 是主指标，FID 为小样本辅助指标。LPIPS 更高只表示外观差异更大，不表示真实性更好。

总体 ROI FID 与 KID 在三个 bank 中均高于 Raw SD2，不能支持 MSDF 的总体视觉分布质量提升。Flash 的 ROI FID 从 195.462 降至 183.059，但 ROI KID 从 0.062484 到 0.062408，变化极小，不能据此声称稳定或显著提升；其 LPIPS 下降。Black 的 ROI FID 从 204.714 升至 257.124，ROI KID 从 0.045212 升至 0.099039，其 LPIPS 上升。总体结论必须与这些类别差异一起呈现。

整图 FID 从 28.427 升至 28.741，整图 KID 均值略有降低。这不能替代预先固定的 ROI KID 主分析。没有统计显著性检验，也不把 bank SD 解释为独立训练种子的不确定性。

## 输入与协议

- 复用源结果 commit：`3332e6cbfb50d9f44b648ec006c329137831a021`。
- Raw SD2 和 MSDF 各 240 张，80 个 slot × A/B/C，每个 bank 每类 40 张，全部参与评价；不重新生成、不筛图、不 reroll。
- 真实参考为 48 张图、60 个原有标注框（Flash 33、Black 27），称为 historically observed frozen held-out visual reference set，不称为 untouched test set。
- Raw、MSDF、Real 均按原始 bbox 每侧扩展该 bbox 宽/高的 25%，再取整并裁至图像边界；未更改 crop 或原标注。
- 标准 torch-fidelity 0.3.0 Inception-V3 2048 维特征与官方 LPIPS AlexNet v0.1。包版本、实现源码 SHA256、权重 SHA256 见 `../metric_runtime.json`。

## 下载后只读复核

31 项测试通过；后台命令通过 Bash 语法检查。22 个服务器结果文件的 SHA256 全部匹配原有 `experiment1_artifact_manifest.json`。重新核对 17 个冻结输入元数据、480 张生成图、48 张真实图和 48 个真实标签文件的哈希，完成结果保护检查返回退出码 0。

独立核对了 36 组 bank/region/class 指标记录、160 条 LPIPS slot 记录（480 个配对距离）、所有 bank/slot 均值与样本 SD、两张论文 CSV、分类摘要，以及 18 组 KID 的 100 次固定子集抽样和全输入覆盖。已下载特征缓存为 1068 × 2048，数值有限且哈希匹配。原始指标文件和表格保持不变。

本地复核没有载入 torch 或任何神经网络，没有重新提取特征或重算 FID/KID/LPIPS。只检查已有记录、哈希和描述性统计。表格技能用于核对统计单位、分组覆盖、原始数值与 CSV 展示值的一致性；没有重写 CSV 或新增工作簿。

## 归档范围与停止条件

归档评价代码、冻结配置、测试、命令、JSON/CSV/MD 和指标图。`inception_features.npz` 留在本地/服务器，不提交 Git；其 SHA256 保留在原缓存清单和本地复核记录中。未提交模型权重或原始图片。

服务器输出中的 `GitHub_submission = NOT_PERFORMED_BY_EVALUATOR` 描述的是评价脚本自身未操作 Git，保留原样；本次仓库提交记录才是正式归档证据。

本阶段到此停止。不修改生成器，不启动其他基线、消融、BootstrapGuard、DWBG、detector training 或人工挑图。
