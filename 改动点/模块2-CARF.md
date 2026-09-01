# 创新点二：粗标注驱动的缺陷区域自精细化（CARF）

> Coarse-Annotation Region Refinement  
> 当前独立版本：CARF-only v4

## 1. 提出原因

工业缺陷数据常用 bbox、点标注或由 bbox 缩小得到的粗 Mask。它只能说明缺陷大致位于哪里，不能当作精确分割 GT。Baseline 直接使用粗 Mask 去噪，容易把框的大小和形状学成缺陷形状，造成缺陷过大、过小或边界生硬。

CARF 将粗 Mask 视为“缺陷搜索区域”，在扩散训练与每一步去噪中，根据当前 latent 证据预测更合适的 soft region。它不是生成后的图像后处理。

## 2. v3 精细化结构

输入为：

```text
noisy latent + masked-image latent + coarse prior + diffusion timestep
```

模块包含三部分：

1. 空间编码器：提取缺陷响应及其与背景的局部差异。
2. Core/Context 残差头：Core 控制高置信缺陷主体，Context 描述边缘和光晕过渡。
3. 形态自适应门控：根据粗区域面积、latent 残差均值/方差和时间步，为每张图动态调节 Core 与 Context，不再让所有缺陷使用相同收缩强度。

v3 不直接重写粗 Mask，而采用残差精细化：

```text
refined prior = coarse prior + bounded residual correction
```

残差头零初始化，因此训练开始时 `refined prior = coarse prior`，与 Baseline 起点完全一致。`min_support` 是真实生效的下界，可避免区域坍缩成一个亮点；所有修正仍被约束在粗标注搜索区域内。

## 3. 训练监督

- 可靠证据损失：真实缺陷图中的高响应区域作为正证据。
- 包含约束：精细区域不能离开粗标注搜索范围。
- 当前样本面积约束：只使用当前缺陷估计目标面积，不再混入随机参考缺陷的面积。
- 核心覆盖损失：高置信缺陷核心必须达到最低覆盖率，防止过度收缩。
- 时间一致性：相邻噪声时间步应得到相近的精细区域。
- 平滑与 Core/Context 分解损失：减少碎片，并分离主体和渐变边缘。
- 缺陷区域加权去噪：避免微小缺陷在全图平均扩散损失中被背景淹没。
- 干净 latent 重建：约束缺陷颜色和局部响应，抑制 Black Spot 的异常彩色光晕。
- 局部结构一致性：保持缺陷核心与边缘梯度，减少 Flash Point 模糊和碎裂。
- 区域外背景约束：限制粗区域外的无关变化和扩散伪影。

v4 不改变 CARF 的区域精细化网络，而是修正 v3 的训练目标。v3 主要回答“在哪里生成”，v4 同时约束“生成成什么样”。高噪声时间步的干净 latent 估计会按噪声可靠度降权，避免辅助损失破坏标准扩散训练。

## 4. 相比 CARF v2 的关键修正

| 项目 | v2 | v3 |
|---|---|---|
| 初始区域 | 零初始化仍把 prior 压到约 0.625 | 严格等于 coarse prior |
| `min_support` | 参数存在但未使用 | 实际限制最小保留比例 |
| 面积目标 | 当前样本与随机参考各占 0.5 | 仅由当前样本决定 |
| 区域强度 | 所有样本共享 | 按形态和时间步动态门控 |
| 防坍缩 | 无直接约束 | 高置信核心覆盖损失 |

## 5. 文件与开关

- 核心模块：`diffusers/pipelines/stable_diffusion/carf_guidance.py`
- 训练：`train_rda.py`、`run_train_rda.py`
- v4 推理配置：`configs/carf_only_v4.json`
- 每类权重：`carf.pt`（格式版本 4）

旧 CARF v2/v3 权重仍可加载；v4 使用独立目录，不覆盖旧结果。

## 6. CARF-only v4 训练

```bash
nohup env CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_train_rda.py \
  --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train \
  --pretrained_model_name_or_path /mnt/sda/cbj/ImageGeneration/Self-main/sd2-community/stable-diffusion-2-inpainting \
  --output_root ./model/carf_only_v4_sd2_2000_s42 \
  --rda_mode none --enable_carf \
  --categories Plastic_Bomo \
  --train_steps 2000 --batch_size 1 --gradient_accumulation_steps 4 \
  --unet_learning_rate 5e-6 --carf_learning_rate 1e-4 \
  --carf_hidden_dim 64 --carf_weight 0.5 \
  --carf_consistency_weight 0.25 \
  --carf_min_support 0.35 \
  --carf_max_core_adjustment 0.75 \
  --carf_max_context_adjustment 0.50 \
  --carf_coverage_weight 0.50 --carf_min_core_coverage 0.90 \
  --carf_region_weight 0.50 \
  --carf_reconstruction_weight 0.50 \
  --carf_structure_weight 0.20 \
  --carf_background_weight 0.20 \
  --unet_train_scope full --seed 42 --mixed_precision fp16 \
  > logs/train_carf_only_v4.log 2>&1 &
```

## 7. 推理

```bash
nohup env CUDA_VISIBLE_DEVICES=2 python inference.py \
  --config configs/carf_only_v4.json \
  --model_ckpt_root ./model/carf_only_v4_sd2_2000_s42 \
  --ddim_scheduler_root ./scheduler \
  --categories Plastic_Bomo --dataset_type mvtec \
  --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train \
  --output_name ./results/carf_only_v4_sd2_2000_s42 \
  > logs/inference_carf_only_v4.log 2>&1 &
```

最终应在固定 seed、normal 和 Mask 的条件下比较 Baseline、CARF-only v3 与 CARF-only v4，并报告生成指标和同一独立真实验证集上的 YOLO11 下游检测指标。
