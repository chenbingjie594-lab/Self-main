# RDA 必做实验

> 方法版本：RDA v4（`format_version=4`）  
> 目标：用尽量少的实验证明 RDA 有效、提升来自参考模块，并且合成数据具有实际价值。  
> 当前不做门控形式和 `max_token_scale` 消融。

## 1. 统一设置

所有方法固定：

- 同一份 SD2 inpainting 初始权重；
- 相同训练集和 held-out 测试集；
- 2000 training steps；
- 相同学习率、batch size、prompt、scheduler、CFG和采样步数；
- 相同normal、粗mask和推理seed；
- 每个类别生成相同数量的完整图像。

测试集不能作为训练图或参考图。黑图、过曝等失败结果必须计入主结果，不能先过滤再评价。

核心指标：

```text
KID：真实性，越低越好
IC-LPIPS：多样性，越高越好
Background LPIPS：背景保持，越低越好
失败率：黑图、过曝、严重色偏和无缺陷比例，越低越好
```

## 2. 实验一：Baseline与RDA主对比

这是主体实验，比较：

```text
Matched Baseline
RDA v4
一个可正确复现的相关方法（优先SEAS或MAGIC）
```

已有 Plastic_Bomo 结果：

| 方法 | KID×1000 ↓ | IC-LPIPS×100 ↑ |
|---|---:|---:|
| Baseline | 51.1 | **26.4** |
| RDA v4 | **44.0** | 26.2 |

SEAS 当前结果中每类有70%的黑底图，疑似未合成的缺陷层，不能直接作为正式竞品结果。需要先生成或合成完整产品图，再进行相同评价。

主结果表：

| 类别 | 方法 | KID↓ | IC-LPIPS↑ | BG-LPIPS↓ | 失败率↓ |
|---|---|---:|---:|---:|---:|
| | Baseline | | | | |
| | 相关方法 | | | | |
| | RDA v4 | | | | |

## 3. 实验二：最小核心消融

只训练三个版本：

| 编号 | 方法 | Reference tokens | RDA辅助损失 | 目的 |
|---|---|---:|---:|---|
| E0 | Baseline | × | × | 原始基准 |
| E1 | Loss-only | × | √ | 判断提升是否只来自额外损失 |
| E2 | Full RDA v4 | √ | √ | 完整方法 |

必须保证三个版本都从相同 SD2 权重训练相同2000 steps。

E1 已通过训练开关实现：

```text
--rda_mode loss_only
```

该模式不创建、不训练、不保存 RDA adapter，也不向条件分支追加 reference tokens；U-Net 仍使用与完整 RDA 相同的区域、重建、结构、分布和背景辅助损失。推理时使用 `configs/rda_e1_loss_only.json`，确保 RDA 完全关闭。

E1训练命令：

```bash
nohup env CUDA_VISIBLE_DEVICES=2 python run_train_rda.py \
  --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train \
  --pretrained_model_name_or_path /mnt/sda/cbj/ImageGeneration/Self-main/sd2-community/stable-diffusion-2-inpainting \
  --output_root ./model/rda_e1_loss_only_sd2_2000_s42 \
  --rda_mode loss_only \
  --categories Plastic_Bomo \
  --train_steps 2000 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --unet_learning_rate 5e-6 \
  --distribution_weight 0.25 \
  --background_weight 0.25 \
  --region_weight 1.0 \
  --conditional_weight 1.0 \
  --latent_reconstruction_weight 1.0 \
  --structure_weight 0.25 \
  --unet_train_scope full \
  --mixed_precision fp16 \
  > logs/train_rda_e1_loss_only.log 2>&1 &
```

E1推理命令：

```bash
nohup env CUDA_VISIBLE_DEVICES=2 python inference.py \
  --config configs/rda_e1_loss_only.json \
  --model_ckpt_root ./model/rda_e1_loss_only_sd2_2000_s42 \
  --ddim_scheduler_root ./scheduler \
  --categories Plastic_Bomo \
  --dataset_type mvtec \
  --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train \
  --output_name ./results/rda_e1_loss_only_sd2_2000_s42 \
  > logs/inference_rda_e1_loss_only.log 2>&1 &
```

核心判断：

- E1优于E0：辅助损失有效；
- E2进一步优于E1：reference tokens提供了额外贡献；
- 如果E2与E1接近，则不能把提升主要归因于参考适配器。

不再拆分每一种辅助损失，也不做门控强度消融。

## 4. 实验三：跨类别验证

当前只有 Plastic_Bomo 两个缺陷类别，至少再增加3种不同形态的缺陷：

```text
一种线状或裂纹缺陷
一种不规则纹理缺陷
一种物体表面缺陷
```

可以从 MVTec AD 中选择代表类别。每个新类别只需要比较 matched baseline 和完整 RDA v4，不需要重新运行全部消融。

最终分别报告每个类别的指标和宏平均，不能只报告总体平均。

| 数据集/类别 | Baseline KID↓ | RDA KID↓ | Baseline IC-LPIPS↑ | RDA IC-LPIPS↑ |
|---|---:|---:|---:|---:|
| Plastic_Bomo/Flash | | | | |
| Plastic_Bomo/Black | | | | |
| 新类别1 | | | | |
| 新类别2 | | | | |
| 新类别3 | | | | |
| 宏平均 | | | | |

## 5. 实验四：下游增益

使用同一个检测、分割或异常检测模型，保持训练参数一致，只改变训练数据：

| 编号 | 下游训练数据 |
|---|---|
| D0 | 仅真实训练数据 |
| D1 | 真实数据 + Baseline合成数据 |
| D2 | 真实数据 + RDA合成数据 |

D1和D2加入的合成图片数量必须相同，并在真实held-out测试集上评价。

根据下游任务选择指标：

```text
目标检测：mAP、AP、Recall
缺陷分割：mIoU、Dice、pixel AUROC
异常检测：image AUROC、pixel AUROC、AP
```

最关键的判断是 D2 是否优于 D1。它可以证明 RDA 的生成改进能够转化为真实应用收益。

## 6. 附带做一个参考响应图

这不单独作为大实验，也不需要重新训练。

固定同一张normal、同一个粗mask和同一个seed，只替换三张同类别参考图，展示生成缺陷是否随参考图变化。再增加一张关闭RDA的结果作为对照。

论文图按下面顺序排列：

```text
normal | 粗mask | reference A/B/C | baseline | RDA A/B/C
```

该图用于直观证明模型确实读取了参考缺陷，而不是忽略reference tokens。

## 7. 最终执行清单

- [ ] 完成 Baseline、RDA和一个相关方法的主对比；
- [ ] 完成 E0、E1、E2 三组最小消融；
- [ ] 增加至少3种不同形态缺陷的 baseline/RDA 对比；
- [ ] 完成 D0、D1、D2 下游增益实验；
- [ ] 整理一张参考图切换的定性对比图；

完成以上四组核心实验后即可冻结 RDA v4，开始第二创新点，不再继续扩展 RDA 实验。
