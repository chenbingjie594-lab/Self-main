# Plastic_Bomo SD2 Inpainting + LoRA 对比实验

Status: `COMPLETE`

本轮只执行标准 UNet Attention LoRA；未修改或训练 MSDF。Black 与 Flash 各自独立训练一次，均完成 2000 次 successful optimizer updates。
基础网络为冻结的 `sd2-community/stable-diffusion-2-inpainting` 9 通道 Inpainting UNet；LoRA rank/alpha 均为 16。

Flash 首次物理执行在 453 updates 后因外部托管会话消失而中断；当时没有 LoRA 或优化器 checkpoint。该不完整尝试已单独保留审计，不纳入结果。经用户授权后，Flash 使用完全相同的 seed、数据顺序和超参数从 step 0 重跑；重跑前 23 步与中断日志逐行一致，最终完成 2000 updates。Black checkpoint 未重训。

## LoRA 逐 Bank 指标

| Bank | Population | KID ↓ | KID subset std | FID ↓ | Generated |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Black | 0.1121249848 | 0.0133444334 | 229.571663 | 40 |
| A | Flash | 0.0615885714 | 0.0046706248 | 162.433749 | 40 |
| A | Overall | 0.0652443889 | 0.0116429815 | 171.806993 | 80 |
| B | Black | 0.1108525582 | 0.0132105715 | 228.891620 | 40 |
| B | Flash | 0.0651246158 | 0.0045476180 | 162.015589 | 40 |
| B | Overall | 0.0669440592 | 0.0123658837 | 174.994281 | 80 |
| C | Black | 0.1006047552 | 0.0127010690 | 227.080242 | 40 |
| C | Flash | 0.0614698242 | 0.0047247729 | 160.773203 | 40 |
| C | Overall | 0.0609652930 | 0.0112535506 | 172.119987 | 80 |

## LoRA Pooled（从拼接特征重算）

| Scope | Population | KID ↓ | KID subset std | FID ↓ | Generated |
| --- | --- | ---: | ---: | ---: | ---: |
| B+C | Black | 0.1070422492 | 0.0121060917 | 212.530976 | 80 |
| B+C | Flash | 0.0673293664 | 0.0072757816 | 150.572374 | 80 |
| B+C | Overall | 0.0652635325 | 0.0129596656 | 160.770113 | 160 |
| A+B+C | Black | 0.1101715408 | 0.0142703570 | 207.147703 | 120 |
| A+B+C | Flash | 0.0660861750 | 0.0073196149 | 147.972725 | 120 |
| A+B+C | Overall | 0.0655519303 | 0.0127380360 | 155.939060 | 240 |

## 对比口径

`comparison.csv` 并列 Raw SD2、MSDF Full（Full Mean-Support）、MSDF Max-Support 与本次 LoRA。Raw/MSDF 使用历史 SHA 绑定的同协议指标；LoRA 使用本轮 240 张图重新提取的特征。Raw/MSDF 是基础 UNet 全量微调路线，LoRA 是仅 UNet Attention 的低秩微调。Max-Support 只改变 Black 分支，因此其 Flash 行与 MSDF Full 使用同一冻结 Flash 输出，并已在 `metric_source` 中标明。

Pooled B+C 与 A+B+C 均将对应图片的 Inception-2048 特征拼接后重新计算 KID/FID，绝非逐 Bank 指标平均。

## 完整性

- 训练数据：V2 TRAIN，Black=97、Flash=75；未读取 official final-eval 内容。
- 生成：A/B/C 每 Bank 每类 40 张，共 240 张；DDIM 50、CFG 7.5；无筛选、Best-of-N 或 reroll。
- 私有图片、LoRA 权重和 Inception 特征缓存只保留在服务器，不应提交 GitHub。
- KID subset std 是 100 个内部子集的波动，不是跨 Bank 置信区间。
