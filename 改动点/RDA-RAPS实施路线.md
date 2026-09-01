# RDA 与 RAPS 实施路线

> 当前主线：已验证的 RDA v4（`format_version=4`）。  
> RAPS 尚未实现；baseline、RDA 权重和生成结果分别保存，不互相覆盖。

## 1. 固定实验基准

所有对比固定数据划分、SD2 初始权重、训练步数、学习率、prompt、scheduler、CFG、采样步数、normal 输入、粗空间先验和随机种子。评价至少包括 KID、IC-LPIPS、背景保持、失败率和下游检测/分割指标。

## 2. RDA v4 主线

RDA v4 从同类别真实缺陷和粗 mask 中提取 reference tokens，并只向 classifier-free guidance 的条件分支追加这些 token。逐样本门控根据参考缺陷的对比度、纹理、面积和包围区域尺度，自适应限制参考信息的注入强度。

训练时：

- 从原始 SD2 inpainting 权重开始，与 baseline 使用相同的 2000 steps；
- 冻结 VAE 和文本编码器，联合训练 U-Net 与 RDA；
- 真实图像和真实粗标注作为监督目标，不使用 baseline 生成结果作为 teacher；
- 对粗 mask 随机膨胀，降低模型记忆固定标注边界的风险；
- 使用区域扩散、条件扩散、latent 重建、结构、分布和背景保持损失。

已验证的宏平均结果：

| 方法 | KID×1000 ↓ | IC-LPIPS×100 ↑ |
|---|---:|---:|
| Baseline | 51.1 | **26.4** |
| RDA v4 | **44.0** | 26.2 |

v4 相对 baseline 的 KID 降低 13.8%，同时多样性基本保持。v5–v8 属于探索性消融，未超过 v4，不再作为代码主线。

## 3. 训练命令

```bash
nohup env CUDA_VISIBLE_DEVICES=2 python run_train_rda.py \
  --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train \
  --pretrained_model_name_or_path /mnt/sda/cbj/ImageGeneration/Self-main/sd2-community/stable-diffusion-2-inpainting \
  --output_root ./model/rda_v4_sd2_2000_s42 \
  --categories Plastic_Bomo \
  --train_steps 2000 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --unet_learning_rate 5e-6 \
  --rda_learning_rate 1e-4 \
  --adapter_hidden_dim 128 \
  --token_grid 4 \
  --max_token_scale 0.25 \
  --distribution_weight 0.25 \
  --unet_train_scope full \
  --mixed_precision fp16 \
  > logs/train_rda_v4.log 2>&1 &
```

## 4. 推理命令

```bash
nohup env CUDA_VISIBLE_DEVICES=2 python inference.py \
  --config configs/rda.json \
  --model_ckpt_root ./model/rda_v4_sd2_2000_s42 \
  --ddim_scheduler_root ./scheduler \
  --categories Plastic_Bomo \
  --dataset_type mvtec \
  --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train \
  --output_name ./results/rda_v4_sd2_2000_s42 \
  > logs/inference_rda_v4.log 2>&1 &
```

`configs/rda.json` 中的 `modules.rda.root` 必须与训练的 `--output_root` 一致。当前加载器只接受 v4 权重，防止误把 v5–v8 权重载入主线。

## 5. 后续 RAPS

RAPS 只在 RDA v4 主线稳定后实现。它为同一条件生成多个候选，并按参考相似性、有效异常、mask 外背景一致性、空间合理性、硬失败和批次多样性进行筛选。RAPS 不覆盖原始生成结果，也不代替 RDA 的训练能力。

## 6. 回退和版本管理

- baseline 配置和权重保持只读；
- RDA checkpoint 独立保存；
- 每次实验记录配置、代码版本和种子；
- 关闭 RDA 时不得残留隐式特征调制；
- v5–v8 结果目录可保留作消融证据，但不再用于训练或推理命令。
