# Chapter 3 Experiment 4 — Three-Method Qualitative Comparison

## 1. 实验目的

在完全冻结的 Raw SD2、MSDF 与 AnomalyDiffusion 输出上，制作可审计的三方法定性对比材料。本实验未训练、未重新生成、未调整模型或生成参数，也未计算新指标。

## 2. 数据来源

输入来自 Stage20B Raw SD2、Stage25 MSDF、Experiment 2 AnomalyDiffusion，以及 Experiment 1 已冻结的视觉匹配清单和真实参考清单。服务器端内部清单保存路径、样本身份和逐文件 SHA256；公开摘要不包含这些字段。

## 3. 三方法匹配规则

以 `bank + slot_id` 为唯一匹配键。每组要求 class、conditioning normal、bbox、frozen slot 和 bank 完全一致。三方法共匹配 240 组；所有 720 张生成图在抽样前均按原始清单核验 SHA256。

## 4. 抽样规则

固定种子为 2026。按 Flash/Black 和 Bank A/B/C 分层，每层抽取 4 组，共 24 组；每类 12 个 slot_id 均不重复。抽样只使用 bank、slot_id、class 和 frozen metadata。抽样名单在任何图像像素被打开前冻结。论文图固定使用每个 class/bank 的第 01 组，共 6 组。

## 5. ROI 裁剪协议

完全复用 Experiment 1 的 ROI 坐标：以 normalized xywh bbox 为基础，每侧扩展 25% context margin，左上界向下取整、右下界向上取整并裁剪到画布。所有方法与 conditioning normal 使用相同 ROI；展示缩放统一采用 bilinear 插值，ROI 为 256×256。

## 6. 论文图片组成

`qualitative_full_set/` 含 24 张四列两行审计图；`paper_figures/` 含 Flash 与 Black 各一张论文组合图；`qualitative_residuals/` 含 24 张固定 0–255 显示范围的绝对 RGB 残差图；另含一张明确标注为非配对真值的真实缺陷参考图。所有图片均保存为 300 DPI PNG。

## 7. 审计结果

审计状态：**PASS**。24 组完整存在，Flash=12、Black=12，每类各 Bank=4；匹配、ROI、冻结前抽样、源文件 SHA256 复核、输出文件数量和 PNG 元数据检查均通过。训练、生成、质量筛选、人工换样、源图修改及新指标计数均为 0。

## 8. 观察记录模板

下表留给人工查看后填写。这里没有使用 FID/KID/LPIPS 自动生成定性结论。

| 类别 | Bank | 缺陷清晰度 | 形态是否符合类别 | 背景融合 | 纹理/结构伪影 | bbox 一致性 | 三方法可见差异 |
|---|---|---|---|---|---|---|---|
| Flash | A |  |  |  |  |  |  |
| Flash | B |  |  |  |  |  |  |
| Flash | C |  |  |  |  |  |  |
| Black | A |  |  |  |  |  |  |
| Black | B |  |  |  |  |  |  |
| Black | C |  |  |  |  |  |  |

## 9. 结果解释边界

这些图用于观察固定样本上的可见现象，不构成统计显著性、总体优越性或因果结论。残差图是诊断材料，不是新评价指标。真实参考样本只展示类别外观，与 synthetic slot 不是一一配对的 ground truth。具体视觉结论应由研究者查看全部 24 组后记录。
