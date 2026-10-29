# MSDF-v3 两项训练级消融固定样本可视化报告

状态：**PASS（固定样本生成、真实文件配对审计和可视化完成）**。本轮没有训练、KID/FID 或检测器实验。

## 冻结协议与样本

六个正式 checkpoint（A0/A1/A2 × Flash/Black）均成功加载完整 SD2 Inpainting pipeline，并以 `strict=True` 加载对应 MSDF。checkpoint 内的 `training_ablation` 分别为 `full`、`no_latent`、`no_morph`；每个模型四层 projection weight 范数均非零。完整模型文件 SHA256 绑定见 `checkpoint_preflight.json`。

每类从原始 Stage20B 40 个冻结槽位按原顺序均匀选择索引 `0,13,26,39`，Bank 固定为 `A,B,C,A`。该清单在生成前写入 `fixed_selection.json`，没有读取生成结果进行选图。三个模型逐槽使用同一 normal、reference、Mask、BBox、prompt、DDIM scheduler、50 steps、CFG 7.5、eta 0、strength 1、sample seed、bank seed、裁切粘贴和 lossless PNG 设置；只改变正式 checkpoint。

共保留 24 张正式 PNG（3 模型 × 8 槽位），没有重抽、替换或质量筛选。输入 crop、Mask、原始源图、checkpoint 和输出 PNG 均记录 SHA256。

## 配对和有效性审计

`figures/pairing_audit.json` 为 PASS，三组各 8 条记录，0 个不一致。审计键为 `(bank,class,slot_id)`，同时核验 manifest 字段和实际文件字节：generated PNG、normal source、reference source、normal crop、reference crop 和 mask。prompt、完整推理参数、生成协议 SHA、seed、gamma、reroll 和质量筛选标记也必须一致。

每个正式 checkpoint 还在该类首个固定槽位上执行 gamma=1 active 与 gamma=0 bypass 的同 seed 对照。六组 RNG 结束状态完全相同，六组输出均非逐像素相同，证明非零 projection 在真实完整推理中产生了有效输出变化。全画布 mean absolute uint8 差异范围为 0.00318–0.00719，最大通道差异为 7–58；详细 changed-pixel 数见 `checks/active_vs_bypass.json`。全画布均值被 4096×1024 背景稀释，因此只用于排除 exact no-op。

## 可见差异

完整图、带上下文 ROI 图和统一色标 `abs(generated-normal)` 热图分别见 `figures/full_comparison`、`figures/roi_zoom` 和 `figures/diff_heatmap`。热图全局固定范围为 0–0.48235，没有逐图拉伸。

- Black 的差异不呈单一方向。Bank A `black_ann_2461…` 中 Full 的缺陷变化更强，A1 次之，A2 较弱；Bank C `black_ann_b6bc…` 中 A1 最强而 A2 最弱。另一个 Bank A 样本中 Full 与 A2 接近。
- Flash 中 A2 在三个可辨样本上通常产生更亮或更强的局部变化，尤其 `flash_ann_2904…`、`flash_ann_a2c6…` 和 `flash_ann_fd60…`；Full 与 A1 在 `2904…` 上较弱。这个现象只说明可见注入结果不同，不能据此认定结构更真实或质量更好。
- exact bbox ROI 的描述性 mean absolute uint8 变化，在 Black 四样本均值中为 Full 9.70、A1 10.49、A2 7.96；Flash 为 Full 8.83、A1 9.31、A2 12.21。该数值不是质量指标，也不用于模型排序。
- 没有观察到 Full 在八个固定样本上稳定优于两个消融版本；不同槽位的强弱次序会改变。

## 失败样本与限制

`black_ann_d88c…` 的三组目标框内都只有很弱的黑点变化，可视为三模型共同的缺陷显著性失败。`flash_ann_8b4d…` 在三组中也都较弱。`flash_ann_2904…` 中 Full 和 A1 的 Flash 不明显，而 A2 可见；这些样本按预定规则全部保留，没有重新抽样。

本报告只有每类 4 个固定样本，覆盖 A/B/C Bank，适合机制可视化和失败案例检查，不支持总体性能结论。Reference 是条件参考图，不是逐像素 ground truth。历史 A0 缺少训练前文件 SHA manifest、donor 顺序和初始 RNG ledger，因此其训练级严格因果可比性限制仍然存在。本轮没有运行 KID/FID，也没有据此宣称 Full 或任一消融方案最优。

