# MSDF-v3 两项消融运行命令

先设置本机路径：

```bash
export REPO=/path/to/Self-main
export DATA_ROOT=/path/to/Plastic_Bomo_split70_seed42/train
export PYTHON=/path/to/msdf-python
export FIGURE_PYTHON=/path/to/python-with-matplotlib
```

以下命令从同一原始 SD2 Inpainting 权重初始化，训练 Plastic_Bomo 的 Flash 和 Black 两个类别。`CUDA_VISIBLE_DEVICES` 可替换为空闲物理卡。命令使用当前实际包装器 `run_train_rda.py`；它会依次运行两个类别并写独立 checkpoint。

## A1：w/o Latent Semantic

```bash
cd $REPO
mkdir -p runs/msdf_v3_ablation/train/no_latent
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -u run_train_rda.py \
  --base_dir $DATA_ROOT \
  --pretrained_model_name_or_path $REPO/sd2-community/stable-diffusion-2-inpainting \
  --categories Plastic_Bomo --rda_mode none --enable_msdf \
  --msdf_ablation no_latent --output_root model/msdf_v3_no_latent_sd2_2000_s42 \
  --train_steps 2000 --batch_size 1 --gradient_accumulation_steps 4 \
  --unet_learning_rate 5e-6 --msdf_learning_rate 1e-4 \
  --msdf_hidden_dim 64 --msdf_context_scale 1.75 --msdf_max_injection 0.75 \
  --msdf_max_residual_ratio 0.25 --msdf_branch_dropout 0.20 \
  --msdf_pixel_support_weight 0.50 --msdf_region_weight 0.75 \
  --msdf_reconstruction_weight 0.50 --msdf_structure_weight 0.25 \
  --msdf_background_weight 0.20 --msdf_scale_weight 0.25 --msdf_support_weight 0.25 \
  --unet_train_scope full --seed 42 --mixed_precision fp16 --use_8bit_adam \
  > runs/msdf_v3_ablation/train/no_latent/train.log 2>&1 &
echo $!
```

## A2：w/o Reference Morphology Alignment

```bash
cd $REPO
mkdir -p runs/msdf_v3_ablation/train/no_morph
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  $PYTHON -u run_train_rda.py \
  --base_dir $DATA_ROOT \
  --pretrained_model_name_or_path $REPO/sd2-community/stable-diffusion-2-inpainting \
  --categories Plastic_Bomo --rda_mode none --enable_msdf \
  --msdf_ablation no_morph --output_root model/msdf_v3_no_morph_sd2_2000_s42 \
  --train_steps 2000 --batch_size 1 --gradient_accumulation_steps 4 \
  --unet_learning_rate 5e-6 --msdf_learning_rate 1e-4 \
  --msdf_hidden_dim 64 --msdf_context_scale 1.75 --msdf_max_injection 0.75 \
  --msdf_max_residual_ratio 0.25 --msdf_branch_dropout 0.20 \
  --msdf_pixel_support_weight 0.50 --msdf_region_weight 0.75 \
  --msdf_reconstruction_weight 0.50 --msdf_structure_weight 0.25 \
  --msdf_background_weight 0.20 --msdf_scale_weight 0.25 --msdf_support_weight 0.25 \
  --unet_train_scope full --seed 42 --mixed_precision fp16 --use_8bit_adam \
  > runs/msdf_v3_ablation/train/no_morph/train.log 2>&1 &
echo $!
```

## 训练前验收

```bash
cd $REPO
$PYTHON tools/check_msdf_v3_ablation_routes.py \
  --output runs/msdf_v3_ablation/checks/route_unit.json
CUDA_VISIBLE_DEVICES=0 $PYTHON \
  tools/smoke_msdf_v3_training_ablations.py --device cuda:0 \
  --image-dir "$DATA_ROOT/Plastic_Bomo/test/02_Big_black_spots" \
  --mask-dir "$DATA_ROOT/Plastic_Bomo/ground_truth/02_Big_black_spots" \
  --output runs/msdf_v3_ablation/checks
```

## 冻结生成说明

正式 checkpoint 完成后，必须把 A0/A1/A2 的两类 checkpoint 分别绑定到现有 Stage20B 固定 slots 与 RNG manifest，再沿用 `tools/stage25_msdf_generator.py` 的 `generate()` 实现逐 bank 生成。现有 Stage25 launcher 会校验其自身冻结代码与 checkpoint tree，不能直接拿新 checkpoint 路径冒充旧 run；因此本轮没有给出一个会绕过绑定审计的伪命令。生成绑定应固定 A=42、B=3407、C=2026，每 bank Flash/Black 各 40，且只改变 checkpoint。

## 可视化

三个参数均应指向完成生成后包含相同 `class + slot_id` 的真实 manifest。脚本按每类排序后的共同 slot 均匀抽取 4 个，不做质量筛选。

```bash
cd $REPO
MPLCONFIGDIR=/tmp/msdf-mpl $FIGURE_PYTHON \
  scripts/render_msdf_ablation_grid.py \
  --full-manifest runs/msdf_v3_ablation/generation/full/formal_synthetic_manifest.json \
  --no-latent-manifest runs/msdf_v3_ablation/generation/no_latent/formal_synthetic_manifest.json \
  --no-morph-manifest runs/msdf_v3_ablation/generation/no_morph/formal_synthetic_manifest.json \
  --slots-per-class 4 --zoom 4 \
  --output-dir runs/msdf_v3_ablation/figures
```

输出为 `full_comparison.{png,pdf}`、`roi_zoom.{png,pdf}`、`diff_heatmap.{png,pdf}` 和带全部输入 SHA256 的 `figure_manifest.json`。
