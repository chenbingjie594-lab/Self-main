# RDA v4 冻结复现清单

> 状态：最终主线，冻结。v5–v8 不得覆盖本配置、权重或结果目录。

## 1. 数据与初始化

- 主数据集：`Plastic_Bomo_split70_seed42`
- 划分：70% 训练、30% held-out，`seed=42`
- 类别：`01_Flash_point`、`02_Big_black_spots`
- 初始权重：原始 Stable Diffusion 2 Inpainting，不从 baseline checkpoint 继续训练
- 每个缺陷类别独立训练 2000 steps
- 输入分辨率：512
- prompt：`a photo of a sks defect`

## 2. 冻结训练参数

```yaml
rda_mode: full
train_steps: 2000
batch_size: 1
gradient_accumulation_steps: 4
unet_learning_rate: 5.0e-6
rda_learning_rate: 1.0e-4
adapter_hidden_dim: 128
token_grid: 4
max_token_scale: 0.25
distribution_weight: 0.25
background_weight: 0.25
region_weight: 1.0
conditional_weight: 1.0
latent_reconstruction_weight: 1.0
structure_weight: 0.25
unet_train_scope: full
saliency_low_quantile: 0.60
saliency_high_quantile: 0.95
mask_jitter_radius: 24
mask_blur_radius: 2
max_grad_norm: 1.0
mixed_precision: fp16
gradient_checkpointing: true
use_8bit_adam: false
seed: 42
```

训练入口为 `run_train_rda.py`，底层为 `train_rda.py`。正式权重目录固定为：

```text
model/rda_v4_sd2_2000_s42/Plastic_Bomo/<defect>/
```

## 3. 冻结推理参数

正式配置使用 `configs/rda_v4_frozen.json`；`configs/rda.json` 是兼容旧命令的同参数配置。

```yaml
pipeline_mode: custom
seed: 42
dtype: float16
num_inference_steps: 50
guidance_scale: 7.5
blur_factor: 0
rda_checkpoint: rda.pt
normal_filter:
  black_threshold: 20
  min_mean_luminance: 30.0
  min_nonblack_ratio: 0.60
```

除 RDA 外，prompt perturbation、spatial guidance、CAMA、DDIM noise 和 MDAP 全部关闭。

## 4. 正式结果与消融

相同 held-out 划分、相同 Inception/LPIPS 实现下：

| 方法 | Flash KID×1000↓ | Black KID×1000↓ | 宏平均 KID×1000↓ | IC-LPIPS×100 |
|---|---:|---:|---:|---:|
| Matched baseline | 49.18 | 52.99 | 51.09 | 26.4 |
| E1：仅辅助损失 | 51.65 | **38.85** | 45.25 | 26.5 |
| RDA v4 | **48.39** | 39.64 | **44.02** | 26.2 |

RDA v4 相比 baseline 的宏平均 KID 降低 13.8%。E1 已贡献约 82.5% 的 KID 改善量；完整参考适配器继续改善宏平均结果，并缓解 E1 在 Flash 类上的退化。

## 5. 复现命令

```bash
nohup env CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python run_train_rda.py \
  --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train \
  --pretrained_model_name_or_path /mnt/sda/cbj/ImageGeneration/Self-main/sd2-community/stable-diffusion-2-inpainting \
  --output_root ./model/rda_v4_sd2_2000_s42 \
  --rda_mode full \
  --categories Plastic_Bomo \
  --train_steps 2000 --batch_size 1 --gradient_accumulation_steps 4 \
  --unet_learning_rate 5e-6 --rda_learning_rate 1e-4 \
  --adapter_hidden_dim 128 --token_grid 4 --max_token_scale 0.25 \
  --distribution_weight 0.25 --background_weight 0.25 \
  --region_weight 1.0 --conditional_weight 1.0 \
  --latent_reconstruction_weight 1.0 --structure_weight 0.25 \
  --unet_train_scope full --saliency_low_quantile 0.60 \
  --saliency_high_quantile 0.95 --mask_jitter_radius 24 \
  --seed 42 --mixed_precision fp16 \
  > logs/train_rda_v4_frozen.log 2>&1 &
```

```bash
nohup env CUDA_VISIBLE_DEVICES=2 python inference.py \
  --config configs/rda_v4_frozen.json \
  --model_ckpt_root ./model/rda_v4_sd2_2000_s42 \
  --ddim_scheduler_root ./scheduler \
  --categories Plastic_Bomo --dataset_type mvtec \
  --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train \
  --output_name ./results/rda_v4_sd2_2000_s42_frozen \
  > logs/inference_rda_v4_frozen.log 2>&1 &
```

注意：`CUDA_VISIBLE_DEVICES=2` 后，程序内部显示的 `cuda:0` 是可见设备中的第 0 张，物理卡仍是 GPU 2。
