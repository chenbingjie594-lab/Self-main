# 历史方案：MSDF 与 DHFG

> 更新说明：DHFG v1-v5 的检测反馈容易把教师捷径写入生成外观，出现白点黑影和黑点缺失，因此不再作为当前第二创新点。当前第二创新点已改为 [DASR 检测感知的合成缺陷重平衡](./创新点二-DASR.md)。本文件保留 DHFG 的探索记录，MSDF 仍为第一创新点。

> 当前状态：MSDF v3 已完成生成质量与下游检测初测；DHFG v1 出现检测器捷径，v2 又因粗 Mask 特征被背景稀释而产生缺陷缺失，现已重构为 DHFG v3 的真实核心—背景特征差匹配方案。  
> 研究主线：面向高分辨率工业图像中的微小瑕疵生成，优先提升真实缺陷保真度和下游检测增益。  
> 版本原则：所有方法从同一份原始 SD2 Inpainting 权重开始训练；CARF v4 作为历史对照和区域约束基础，不再单独承担主要创新。

## 1. 总体问题

现有 SD2 Inpainting baseline 主要存在两个相互关联的问题：

1. **微小缺陷在潜空间中发生信息消失。** 512×512 图像被 VAE 压缩到约 64×64 后，小黑点、微小亮斑和细边缘可能只剩一个或不足一个稳定特征单元。提高 Mask 内损失只能增强颜色或亮度，无法稳定恢复真实形态。
2. **生成目标与检测目标不一致。** KID 或视觉效果较好的样本不一定能提高检测器；过亮、过黑或过于简单的合成缺陷反而会改变训练分布，使 Precision、Recall 和 mAP 出现方向不一致的变化。

因此设计两个互补模块：

- **MSDF** 解决“微小缺陷能否被生成并保留真实结构”；
- **DHFG** 解决“生成缺陷是否具有真实、适度且有利于检测的判别信息”。

## 2. 创新点一：MSDF 多尺度微小缺陷保真模块

MSDF（Multi-Scale Defect Fidelity）通过全局扩散分支和高分辨率局部缺陷分支协同建模，解决微小缺陷经过 VAE 和 U-Net 多次下采样后形态退化的问题。

### 2.1 输入与结构

输入包括正常图像、粗 Mask、缺陷类别、扩散时间步和当前噪声 latent。

1. **全局扩散分支**：沿用 SD2 Inpainting U-Net，负责产品整体纹理、光照、Mask 外背景以及缺陷与产品的自然融合。
2. **局部缺陷分支**：以粗 Mask 为中心裁取外扩 2～4 倍的局部区域，并缩放到固定分辨率；轻量 CNN 或局部 U-Net 提取边缘、形状和上下文三级特征。
3. **多尺度特征注入**：将局部特征分别注入主 U-Net 的高、中、低分辨率层，而不是只在最终损失中提高 Mask 权重。
4. **类别原型**：训练阶段从同类真实缺陷中学习局部形态原型；推理时使用类别原型、类别条件和噪声预测局部特征，不依赖真实目标缺陷图。

### 2.2 尺度与时间步门控

根据粗 Mask 相对面积计算缺陷尺度：

```text
r = mask_area / image_area
```

各层注入权重由缺陷尺度 `r`、时间步 `t` 和当前局部响应共同预测：

```text
g_l = Gate_l(r, t, local_response)
F_l = F_l_global + g_l * Project_l(F_l_local)
```

- 微小缺陷优先增强高分辨率层，避免边缘和小黑点被淹没；
- 较大缺陷增加中低分辨率特征，维持完整轮廓；
- 去噪前期确定位置和低频结构；
- 去噪中期注入缺陷形状；
- 去噪后期恢复边缘、高频纹理并强化背景保持。

这里不使用统一的固定注入强度上限，而是按样本尺度和扩散阶段动态控制。

### 2.3 训练目标

```text
L_MSDF = L_noise
       + lambda_local * L_local
       + lambda_edge  * L_edge
       + lambda_bg    * L_background
       + lambda_scale * L_scale_consistency
```

- `L_noise`：标准扩散去噪损失；
- `L_local`：真实缺陷局部区域的重建或感知一致性；
- `L_edge`：梯度、边缘和高频结构保持；
- `L_background`：粗区域外产品纹理保持；
- `L_scale_consistency`：全局图与局部高分辨率分支的多尺度特征一致性。

CARF 的核心区、上下文区和背景约束可以并入 `L_local` 与 `L_background`，作为辅助组件而不是主要贡献。

### 2.4 训练与推理

MSDF 在训练和推理阶段都启用。训练阶段学习局部分支、尺度门控、类别原型和 U-Net 融合层；推理阶段根据正常图、粗 Mask、缺陷类别和随机噪声完成多尺度注入。

## 3. 创新点二：DHFG 检测反馈式困难缺陷生成

DHFG（Detector-guided Hard Defect Generation）使用冻结的真实数据教师检测器，让生成缺陷的多尺度中间特征接近对应真实缺陷，同时仅以弱置信度下限保证缺陷可检测，使生成质量与下游需求形成闭环。

### 3.1 教师检测器

先仅使用真实训练集训练一个教师检测器 `D`。扩散训练时冻结 `D`，将可靠时间步得到的干净图估计送入检测器，提取：

- 目标类别置信度；
- Mask 区域的缺陷特征；
- Mask 外的误检响应；
- 预测区域与粗 Mask 的空间关系。

教师检测器只提供监督，不参与参数更新。

### 3.2 逐样本置信度容差匹配

生成缺陷与当前真实参考缺陷的教师 logit 做逐样本匹配，并保留容差区间：

```text
abs(score_generated - score_reference) <= tolerance
```

超出容差时使用 Huber 损失，既防止缺陷消失，也抑制置信度超过真实参考后继续增长，避免模型通过过亮、过黑或硬边缘迎合检测器。

### 3.3 真实核心—背景多尺度特征差匹配

DHFG v1 的类别平均原型导致 Flash Point 收敛成重复的“白头黑尾”；v2 在整个粗 Mask 内平均匹配，微小缺陷又被大面积背景稀释。DHFG v3 先在真实参考图的教师响应图中选取 Mask 内 top-k 位置作为缺陷核心，并在核心外构造局部背景环带。该支持停止梯度且同时用于真实图和生成图，生成结果不能自行移动注意区域。

每个检测尺度计算：

- `gap = mean(feature_core) - mean(feature_context)`；
- 真实与生成 `gap` 的方向和幅值；
- 核心区域通道标准差。

核心—背景差显式消除了局部材质基线，既突出真正的微小缺陷信号，又保留逐样本形态差异。

### 3.4 背景与多样性约束

- `L_boundary`：在粗 Mask 外侧环带直接对齐生成图与对应真实图，抑制黑影、双点和边缘泄漏；
- `L_false_bg`：抑制 Mask 外的缺陷检测响应，防止条纹、光照和背景接缝被学习为异常；
- 逐样本真实特征目标替代类别均值原型，降低同类样本的模板坍缩。

### 3.5 训练目标

```text
L_feature_gap = L_gap_direction + eta * (L_gap_magnitude + L_core_std)

L_DHFG = alpha * L_feature_gap
       + beta  * L_confidence_pair
       + gamma * L_boundary
       + delta * L_false_bg

L_total = L_MSDF + lambda_detector * L_DHFG
```

DHFG 主要在训练阶段启用。正常推理不需要加载教师检测器；批量生成后的检测器筛选只能作为辅助质量控制，不能替代训练反馈。

### 3.6 DHFG v3 代码实现

当前实现不对 YOLO 的 NMS、最终框坐标或阈值化结果反向传播，而是使用检测头之前的多尺度特征和每类响应图：

1. 在 `yolo11` 环境中，用 `tools/export_dhfg_teacher.py` 将 Real-only 的 `best.pt` 导出为带元数据的 TorchScript 教师；
2. 扩散训练环境只加载 TorchScript，不需要安装 Ultralytics；教师参数始终冻结；
3. 先用该类别的真实缺陷局部区域标定真实置信度下界以及背景响应 90% 上界；缺陷分数只聚合粗 Mask 内最高的少量位置，避免微小目标响应被大量低值稀释；
4. 在可靠的低噪声时间步恢复预测 `x0`，围绕粗 Mask 裁取局部 latent，再以局部 VAE 解码得到教师输入；
5. 从真实教师响应中选取缺陷核心并构造上下文环带，用 `L_feature_gap` 匹配逐样本核心—背景差；
6. `L_confidence_pair` 将生成分数约束在真实样本附近的容差区间，`L_boundary` 保护 Mask 外侧环带，`L_false_bg` 使用 Mask 外 top 1% 响应抑制局部误检；
7. 教师每隔若干 micro-step 且仅在低噪声时间步启用，以适配 24 GB 显存；反馈使用 warmup 和 `alpha_cumprod` 可靠性，避免训练初期过强。

训练启动时还会检查真实缺陷标定概率。默认要求80%分位上界至少达到1%；若低于该值会直接终止，并提示提高教师导出分辨率或重新训练面向512×512局部缺陷的教师，防止模型被引导为“检测器不可见”的缺陷。

逐样本置信度差在 logit 空间采用带容差的 Huber 损失：差距很大时保持线性增长，进入真实参考容差后梯度为零。教师反馈乘以当前扩散时间步的 `alpha_cumprod` 可靠性，并且不对间隔采样做峰值倍增，从而降低单个活跃 micro-step 引发 fp16 梯度溢出的风险。

关键文件：

- 教师反馈模块：`diffusers/pipelines/stable_diffusion/dhfg_guidance.py`
- 教师导出：`tools/export_dhfg_teacher.py`
- 训练核心：`train_rda.py` 中的 `--enable_dhfg` 路径
- 批量训练入口：`run_train_dhfg.py`
- DHFG-only 推理配置：`configs/dhfg.json`
- 联合推理配置：`configs/msdf_dhfg.json`

DHFG-only 与联合版本均从原始 SD2 Inpainting 权重重新训练。DHFG 的效果写入训练后的 U-Net，因此 DHFG-only 推理不加载教师；联合版本推理只需要随 checkpoint 保存的 `msdf.pt`。

## 4. 核心实验设计

### 4.1 公平对比

| 编号 | 方法 | 作用 |
|---|---|---|
| M0 | Real Only | 下游检测下限，不使用合成图 |
| M1 | SD2 Baseline | 去掉全部创新模块 |
| M2 | CARF v4 | 历史方法对照 |
| M3 | SD2 + MSDF | 验证创新点一 |
| M4 | SD2 + DHFG | 验证创新点二的独立有效性 |
| M5 | SD2 + MSDF + DHFG | 完整方法 |

M1～M5 必须统一 SD2 初始权重、真实训练数据、训练步数、学习率、batch size、prompt、推理 normal、粗 Mask、scheduler、采样步数和 seed。新方法不能从训练好的 Baseline 或 CARF checkpoint 继续训练。

### 4.2 MSDF 必要消融

| 实验 | 局部分支 | 多尺度注入 | 尺度门控 | 时间步门控 |
|---|---:|---:|---:|---:|
| A0 | × | × | × | × |
| A1 | √ | √ | × | × |
| A2 | √ | √ | √ | × |
| A3 | √ | √ | √ | √ |

A3 为完整 MSDF。重点判断高分辨率局部分支是否真正改善微小缺陷形态，而不是只提高亮度或对比度。

### 4.3 DHFG 必要消融

| 实验 | 核心—背景特征差 | 逐样本置信度匹配 | 边界/背景约束 |
|---|---:|---:|---:|
| B0 | × | × | × |
| B1 | √ | × | × |
| B2 | √ | √ | √ |

B2 为完整 DHFG。若时间有限，不继续拆分每一个损失权重。

### 4.4 生成质量评测

优先使用经典指标：

- `KID↓`：生成集与真实缺陷集的分布距离；
- `IC-LPIPS↑`：同类生成样本的多样性；
- `LPIPS↓`：适用于存在配对参考时的感知距离；
- `SSIM↑`、`PSNR↑`：在 Mask 外区域评估背景保持。

同时固定抽样数量，人工统计有效生成率、缺陷不可见率和明显伪影率。人工指标只作补充，不能替代经典指标。

### 4.5 粗 Mask 鲁棒性

固定同一组 normal 和 seed，比较以下 Mask：

```text
original / erode20 / dilate20 / dilate40 / shift8
```

对 Baseline、CARF v4、MSDF 和完整方法分别生成，比较生成指标及其下游检测性能下降幅度。

### 4.6 下游检测增强

建立以下训练组：

```text
Real Only
Real + Baseline
Real + CARF v4
Real + MSDF
Real + DHFG
Real + MSDF + DHFG
```

实验要求：

- 所有组使用相同真实训练集；
- 每组加入相同数量、类别平衡的唯一合成样本；
- 合成图不得进入验证集或测试集；
- 验证集和测试集只使用固定真实图；
- YOLO11s 的权重、分辨率、epoch、seed 和其他训练参数完全一致；
- 报告 Precision、Recall、mAP50、mAP50-95 和各类别 AP；
- 最终核心结果至少运行 3 个 seed，报告均值与标准差。

### 4.7 数据集安排

- **Plastic_Bomo**：主要实验，完整比较生成质量、粗 Mask 鲁棒性和下游检测增益；
- **DeepPCB**：公开数据集泛化，先选择 `pinhole`、`mousebite`、`short`，确认有效后再扩展全部类别。

## 5. 实施顺序与停止条件

1. 先实现 MSDF，不接入检测器；
2. 从原始 SD2 权重在 Plastic_Bomo 上训练，与 Baseline 和 CARF v4 公平比较；
3. 若微小形态、KID 或下游 mAP50-95 均没有可复现改善，先修正多尺度结构，不进入 DHFG；
4. MSDF 有效后训练真实数据教师检测器并实现 DHFG；
5. 完成 Baseline、MSDF、DHFG、MSDF+DHFG 四组核心对比；
6. 最后进行 DeepPCB 泛化与 3-seed 复现实验。

## 6. MSDF v3 代码落地

v1 虽然能够稳定生成微小缺陷，但与 CARF v4 的平均 KID 差距不足 1%，并且 Big Black Spot 容易收敛为相似的小圆点。原因是跨样本参考与另一张目标图的监督不一致，模型容易忽略参考；同时将参考特征直接拉伸到目标粗 Mask，会削弱参考形态。

v2 采用“像素级缺陷残差 + latent 语义残差 + 可学习参考形态支持 + 保持长宽比对齐 + U-Net 四级分频注入”：

- 在 VAE 压缩前用像素分支提取微小边缘和高频残差；
- 在 latent 分支提取缺陷主体和局部语义；
- 两个分支融合后预测参考形态支持图；
- 粗 Mask 只给出中心和最大范围，不再决定最终缺陷轮廓；
- 对齐过程保持参考缺陷长宽比和占用面积比例；
- 低分辨率 U-Net 层注入主体/光晕，高分辨率层逐级加强边缘；
- 70% 自参考学习真实形态编码，30% 跨参考用于类别正则；
- 形态支持损失只作用于自参考样本，避免错误地强迫跨参考匹配另一张目标图。

推理使用训练集中的同类别参考图和参考 Mask，不读取待评测样本的真实缺陷图。

- 核心模块：`diffusers/pipelines/stable_diffusion/msdf_guidance.py`
- 训练核心：`train_rda.py` 中的独立 `--enable_msdf` 路径
- 专用训练入口：`run_train_msdf.py`
- 推理入口：`inference.py` 与自定义 Inpainting Pipeline
- 推理配置：`configs/msdf.json`
- 每类附加权重：`msdf.pt`，格式版本为 3；v1/v2 权重不能直接加载，必须重新训练
- 最小单元测试：`tests/test_msdf_adapter.py`

MSDF-only 模式固定关闭 RDA 与 CARF；U-Net 和 MSDF 从原始 SD2 Inpainting 权重共同训练。各尺度输出投影采用零初始化，因此训练开始时严格不改变原始 U-Net 输出。MSDF 在 CFG 中只注入条件分支，无条件分支保持为零残差。v3 默认使用独立目录 `model/msdf_v3_sd2_2000_s42`，不会覆盖 v1/v2。

v3 针对 v2 的失败补充三项结构机制：每个 up-block 的注入通过当前 U-Net 特征 RMS 自适应限幅；像素与潜空间分支采用模态 dropout，并给像素 support head 独立形态监督，避免高分辨率分支被旁路；训练、checkpoint 加载及推理去噪链路均执行有限值检查，禁止 NaN/Inf 被静默保存为纯黑结果。

## 7. 预期论文表述

创新点一：

> 针对微小工业缺陷在潜空间压缩和多级下采样过程中发生特征消失的问题，提出尺度与时间步联合控制的多尺度局部—全局缺陷保真模块，在保持产品背景的同时强化微小缺陷的形态和高频结构。

创新点二：

> 针对生成质量目标与下游检测目标不一致的问题，提出检测反馈式困难缺陷生成机制，通过真实缺陷原型、适度困难度和背景误检约束，引导扩散模型生成真实、多样且具有检测训练价值的微小缺陷。

## 8. DHFG v4：极性感知的分布校准反馈

v4 不再把生成结果逐张匹配到随机真实参考的教师置信度，而是在训练开始前从该类别全部真实缺陷中标定置信度、亮响应、暗响应、色度变化、有效覆盖率和空间范围的 20%～80% 分位区间。生成属性只有超出真实区间时才产生 Huber 惩罚，从而同时抑制缺陷消失和过度强化。

真实缺陷核心由教师响应与局部物理对比共同形成连续软权重图。局部对比相对于粗 Mask 外侧上下文计算，并分解为正亮度响应和负亮度响应，因此无需按类别硬编码即可描述亮缺陷、暗缺陷以及明暗混合缺陷。教师多尺度核心—上下文特征差降为辅助目标，防止模型只生成最容易被检测器识别的固定模板。

```text
L_DHFG-v4 = 0.25 L_feature_relation
          + 0.50 L_confidence_band
          + 0.75 L_dual_polarity
          + 0.50 L_coverage_shape
          + 0.50 L_boundary
          + 0.10 L_false_background
```

DHFG v4 仍然只在训练阶段加载冻结教师。反馈被学习进 U-Net，普通推理不加载教师、不进行梯度引导，也不增加推理开销。默认 checkpoint 目录为 `model/dhfg_v4_sd2_2000_s42`，推理实验名为 `sd2_2000_s42_dhfg_v4`。

## 9. DHFG v5：反事实差分反馈

v4 的绝对图像统计量会把粗 Mask 内原有条纹和暗纹误认为有效缺陷，因此背景纹理即使没有生成缺陷也可能落入真实分布区间。v5 为真实图和生成图分别构造局部反事实图：利用真实参考的教师响应与物理对比确定连续软支持区，然后用周围局部上下文平滑替换该区域。监督对象由绝对响应改为原图与反事实图之间的增量：

```text
Delta_real = Teacher(real) - Teacher(erase(real))
Delta_gen  = Teacher(generated) - Teacher(erase(generated))
```

背景材质在同图差分中被抵消；缺陷消失时 `Delta_gen` 接近零，缺陷过强时又会超过真实增量。v5 同时匹配多尺度教师特征增量、响应图增量、置信度增量，以及生成残差的明暗极性和软形态。反事实分支采用 stop-gradient，只保留原始生成分支的教师反向图，因此显存规模与 v3/v4 接近。

```text
L_DHFG-v5 = 0.25 L_feature_delta
          + 0.75 L_response_delta
          + 0.50 L_confidence_delta
          + 0.50 L_residual_polarity
          + 0.25 L_residual_shape
          + 0.50 L_boundary
          + 0.10 L_false_background
```

默认 checkpoint 目录为 `model/dhfg_v5_sd2_2000_s42`，推理实验名为 `sd2_2000_s42_dhfg_v5`。v5 仍是训练期反馈模块，推理不加载检测教师。
