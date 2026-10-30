# Plastic_Bomo DefectFill 对比实验

Status: `COMPLETE_PARTIAL_REIMPLEMENTATION`

该仓库是非官方且不完整复现：三项损失与 LoRA 实际参与训练，但原生 8 候选 Low-Fidelity Selection 没有端到端实现。本轮只执行 Primary Strict（每 slot 第一个且唯一候选，无筛选/重抽）。

训练 Mask 是 `COARSE_BBOX_MASK`，不是像素级 GT。Black/Flash 各 2000 updates，LoRA rank=8/alpha=16。

## Bankwise

|Bank|Population|KID|FID|
|---|---|---:|---:|
|A|Black|0.0786821070|206.325461|
|A|Flash|0.0680308683|192.838618|
|A|Overall|0.0431822931|163.758742|
|B|Black|0.0770644870|207.230108|
|B|Flash|0.0662830215|191.730084|
|B|Overall|0.0449547953|166.551205|
|C|Black|0.0810058092|211.321101|
|C|Flash|0.0706817508|195.078418|
|C|Overall|0.0472755519|166.082275|

## Pooled（拼接特征重算）

|Scope|Population|KID|FID|
|---|---|---:|---:|
|B+C|Black|0.0789503377|191.003669|
|B+C|Flash|0.0717118849|179.162752|
|B+C|Overall|0.0472722386|150.273498|
|A+B+C|Black|0.0802005233|183.348991|
|A+B+C|Flash|0.0711044812|174.790315|
|A+B+C|Overall|0.0456529020|144.535950|

## 对比解释

`comparison.csv` 同时包含 Raw SD2、MSDF Full、MSDF Max-Support、SD2 Inpainting + LoRA r16 与本方法。DefectFill 使用 UNet+文本编码器 LoRA r8 和三项损失；LoRA 基线仅使用 UNet Attention LoRA r16，因此差异不构成纯 Loss 消融。
