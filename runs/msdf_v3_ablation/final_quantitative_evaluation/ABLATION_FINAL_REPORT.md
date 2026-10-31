# MSDF-v3 最终训练级消融定量评测

## 核心结果（A+B+C pooled）

| 实验配置 | Black KID↓ | Black FID↓ | Flash KID↓ | Flash FID↓ | Overall KID↓ | Overall FID↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full MSDF-v3 | **0.04443** | **152.91** | **0.06371** | **141.71** | **0.03836** | **128.55** |
| w/o Latent Semantic | 0.05830 | 160.54 | 0.07068 | 149.31 | 0.04506 | 136.72 |
| w/o Morphology Guidance | 0.04796 | 157.54 | 0.07007 | 142.12 | 0.04160 | 131.84 |

## 相对 Full 的变化

负值表示消融模型指标更低（更好）。

| 消融 | 指标 | 绝对差值 | 百分比变化 |
| --- | --- | ---: | ---: |
| w/o Latent Semantic | Black_KID | +0.01387527 | +31.23% |
| w/o Latent Semantic | Black_FID | +7.63127992 | +4.99% |
| w/o Latent Semantic | Flash_KID | +0.00697177 | +10.94% |
| w/o Latent Semantic | Flash_FID | +7.59973581 | +5.36% |
| w/o Latent Semantic | Overall_KID | +0.00669598 | +17.45% |
| w/o Latent Semantic | Overall_FID | +8.17530087 | +6.36% |
| w/o Morphology Guidance | Black_KID | +0.00352665 | +7.94% |
| w/o Morphology Guidance | Black_FID | +4.62530353 | +3.02% |
| w/o Morphology Guidance | Flash_KID | +0.00635818 | +9.98% |
| w/o Morphology Guidance | Flash_FID | +0.41536617 | +0.29% |
| w/o Morphology Guidance | Overall_KID | +0.00323185 | +8.42% |
| w/o Morphology Guidance | Overall_FID | +3.29069708 | +2.56% |

## 科学结论

### Latent Semantic 分支

去除 Latent Semantic 后，Black、Flash 与 Overall 的 pooled KID/FID 六项指标全部变差；其中 Overall KID 增加 0.00670（+17.45%），Overall FID 增加 8.18（+6.36%）。同一方向也出现在 A/B/C 三个 Bank 的 Black、Flash、Overall KID/FID 全部 18 组比较中。因此，在本轮冻结协议和 V2 TRAIN ROI 参考集内，结果支持潜空间语义分支及双分支融合带来稳定的生成质量增益。

### Morphology Guidance

去除 Morphology Guidance 后六项 pooled 指标同样全部变差；Overall KID 增加 0.00323（+8.42%），Overall FID 增加 3.29（+2.56%）。Black 退化较明确（KID +7.94%，FID +3.02%）；Flash KID 明显退化（+9.98%），但 Flash FID 仅增加 0.42（+0.29%），说明形态引导对 Flash 的 FID 收益较弱。尽管如此，A/B/C 的全部 18 组同类指标仍均由 Full 取得更低值，因此当前结果支持形态引导有效，但不应把 Flash FID 的微小差异表述为强效应。

### 跨 Bank 稳定性

Full 在 A、B、C 每个 Bank 的 Black、Flash、Overall KID 和 FID 上均优于两个消融模型，共 36/36 个逐 Bank 成对指标方向一致，并非由单个 Bank 拉低 pooled 结果。Full 的三 Bank 均值±样本标准差为：Black KID 0.04159±0.00240、FID 172.62±2.62；Flash KID 0.06001±0.00128、FID 154.57±4.43；Overall KID 0.03690±0.00064、FID 144.30±2.10。逐 Bank 原始值及 KID 内部 subset 标准差见 `ablation_bankwise_metrics.csv`；这里的跨 Bank 标准差只描述三个生成种子的离散程度，不是置信区间。

## 结果证实与尚未验证

**结果证实：** 在相同冻结 slot、Normal、Reference、BBox/Coarse Mask、Prompt、sample/bank seed、DDIM 50 steps、CFG 7.5、ROI 特征和 V2 TRAIN 真实参考协议下，Full 的本轮生成质量指标稳定优于两个训练级消融模型。

**尚未验证：** 这些结果不证明独立测试集泛化、下游检测收益或跨数据集有效性，也不能量化训练随机性带来的不确定性；每种配置只有一次正式训练。A0 历史训练还缺少完整初始 RNG/训练文件哈希账本，因此模块因果表述应限定在现有 checkpoint 和冻结评价协议内。

## 证据边界

结果使用 V2 TRAIN 的 172 个真实 ROI；它不是独立测试集，且可能与生成器训练数据重叠。A0 历史训练缺少完整初始 RNG/训练文件哈希账本，因此模块因果结论应保持条件性。未读取 final_eval。
