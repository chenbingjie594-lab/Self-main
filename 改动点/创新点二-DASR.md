# 创新点二：DASR 检测感知的合成缺陷重平衡

## 1. 提出原因

MSDF 解决微小缺陷在扩散模型下采样和去噪过程中形态丢失的问题，但“看起来像缺陷”不等于“适合训练检测器”。把全部生成图直接加入 YOLO 会引入三类问题：缺陷可能没有真正形成；粗 Mask 不能作为精确检测框；大量相似样本会改变类别与尺度分布。因此，局部生成质量提高后，下游 Precision、Recall 和 mAP 仍可能下降。

DASR（Detector-Aware Synthetic Rebalancing）在生成与检测之间建立面向训练价值的数据闭环：只保留真实、有效、覆盖检测器薄弱尺度且彼此不同的合成样本，再以真实物理尺度回贴到高分辨率背景。

## 2. 核心流程

1. **真实先验建模**：从真实训练集统计两类缺陷的亮/暗极性、颜色偏移、框面积和长宽比分布，不读取最终验证集。
2. **真实残差标注**：比较生成图与对应 normal 图，提取实际发生变化的缺陷支持域；粗 Mask 只限定搜索范围，不直接充当框标注。未生成缺陷的样本会被淘汰。
3. **质量门控**：过滤背景泄漏、残差面积过大、属性超出真实训练分布和类别极性错误的样本。
4. **检测薄弱性估计**：用真实训练集的三折 OOF 检测结果统计各类别、各尺度区间的漏检程度，漏检较多的区间获得更高采样权重。最终真实验证集始终不可见。
5. **适度困难度与多样性选择**：Real-only 探针只评估候选，优先选择中等置信度样本，并通过特征空间最远点选择抑制近重复缺陷。
6. **高分辨率重组**：根据真实框尺度与长宽比把缺陷残差回贴到 4096×1024 无缺陷背景，最终 YOLO 框由回贴后的实际支持域重新计算。
7. **固定配额训练**：合成图只进入训练集，两类使用相同预设配额；Real-only 与 Real+DASR 共用完全相同的真实验证集和 YOLO 初始权重。

DASR 不再次微调扩散模型。需要训练的是三折 OOF 弱点估计器、一个 Real-only 候选探针和最终对照检测器。生成模型仍使用已经训练好的 MSDF；DASR 负责把生成结果转化为真正对检测有价值的数据。

## 3. 代码

- `tools/create_dasr_yolo_folds.py`：创建无验证泄漏的三折真实训练集。
- `tools/profile_dasr_weakness.py`：生成类别—尺度弱点画像。
- `tools/build_dasr_detection.py`：残差提取、质量门控、检测感知选择、高分辨率回贴和 YOLO 数据集构建。
- `inference.py`：显式 `--seed` 优先于配置文件 seed，用于构建真正独立的多 seed 候选池。

## 4. 服务器实验流程

先设置路径：

```bash
PROJECT=/mnt/sda/cbj/ImageGeneration/Self-main
REAL=/mnt/sda/cbj/dataset/Plastic_Bomo/GLRR_detection_no_hgrr_plus_black35_s42/real_only
BG=/mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train/Plastic_Bomo/train/good
FOLDS=/mnt/sda/cbj/dataset/Plastic_Bomo/DASR_oof_s42
cd "$PROJECT"
```

### 4.1 三折 OOF 弱点估计器

```bash
python tools/create_dasr_yolo_folds.py \
  --real_root "$REAL" --output "$FOLDS" --folds 3 --seed 42

mkdir -p logs/dasr_oof
for fold in 0 1 2; do
  yolo detect train \
    model=pretrained/yolo11s.pt data="$FOLDS/fold_${fold}.yaml" \
    epochs=150 imgsz=1536 batch=1 rect=False device=0 workers=4 \
    seed=42 deterministic=True patience=40 \
    project=runs/dasr_oof name=fold_${fold}_s42 exist_ok=False \
    > "logs/dasr_oof/fold_${fold}.log" 2>&1
done

python tools/profile_dasr_weakness.py \
  --fold "$FOLDS/fold_0/images/val" "$FOLDS/fold_0/labels/val" runs/dasr_oof/fold_0_s42/weights/best.pt \
  --fold "$FOLDS/fold_1/images/val" "$FOLDS/fold_1/labels/val" runs/dasr_oof/fold_1_s42/weights/best.pt \
  --fold "$FOLDS/fold_2/images/val" "$FOLDS/fold_2/labels/val" runs/dasr_oof/fold_2_s42/weights/best.pt \
  --output results/dasr_weakness_s42.json --device 0 --imgsz 1536
```

三个 fold 都必须从同一个 `yolo11s.pt` 开始，不能继承其他 fold 的权重。

### 4.2 生成独立的 MSDF 候选池

同一 checkpoint 用多个 seed 推理，不需要重新训练 MSDF：

```bash
for seed in 42 3407 7859 12011; do
  CUDA_VISIBLE_DEVICES=0 python inference.py \
    --config configs/msdf.json --seed "$seed" \
    --model_ckpt_root ./model/msdf_v3_sd2_2000_s42 \
    --ddim_scheduler_root ./scheduler \
    --categories Plastic_Bomo --dataset_type mvtec \
    --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_split70_seed42/train \
    --output_name "results/dasr_pool_seed${seed}"
done
```

禁止通过复制同一张生成图凑配额。

### 4.3 构建 Real-only 与 Real+DASR

`--probe_weights` 使用只在真实训练集上训练的 Real-only 模型。它只给候选打分；最终 Real+DASR 仍从原始 `yolo11s.pt` 训练。

```bash
python tools/build_dasr_detection.py \
  --real_root "$REAL" \
  --generated_root results/dasr_pool_seed42 \
  --generated_root results/dasr_pool_seed3407 \
  --generated_root results/dasr_pool_seed7859 \
  --generated_root results/dasr_pool_seed12011 \
  --background_dir "$BG" \
  --output /mnt/sda/cbj/dataset/Plastic_Bomo/GLRR_DASR_80_80_s42 \
  --flash_count 80 --black_count 80 \
  --probe_weights runs/real_only_probe_s42/weights/best.pt \
  --probe_device 0 --probe_imgsz 1536 \
  --weakness_profile results/dasr_weakness_s42.json --seed 42
```

若有效候选不足，程序会列出每类淘汰原因。此时增加独立 seed，不应复制样本。

### 4.4 最终公平对照

```bash
DATA=/mnt/sda/cbj/dataset/Plastic_Bomo/GLRR_DASR_80_80_s42
mkdir -p logs/dasr_final
for group in real_only real_dasr; do
  yolo detect train \
    model=pretrained/yolo11s.pt data="$DATA/${group}.yaml" \
    epochs=150 imgsz=1536 batch=1 rect=False device=0 workers=4 \
    seed=42 deterministic=True patience=40 \
    project=runs/dasr_final name=${group}_s42 exist_ok=False \
    > "logs/dasr_final/${group}_s42.log" 2>&1
done
```

## 5. 必须报告的实验

主表报告 Real-only、Real+MSDF（全部直接加入）和 Real+DASR 三组的 Precision、Recall、mAP50、mAP50-95，并给出两个类别的 AP50-95。三组必须使用相同真实验证集。

核心消融只保留三项：去掉质量门控、去掉检测薄弱性权重、去掉多样性选择。所有消融保持相同合成配额。
