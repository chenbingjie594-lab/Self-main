# CARF 粗 Mask 扰动实验

## 已准备的数据

本机目录：`dataset/Plastic_Bomo_mask_robustness_s42`

- 每个缺陷类别固定 20 个样本，共 40 个样本；
- 每个样本包含相同的 paired normal；
- 五种 Mask：`original`、`erode20`、`dilate20`、`dilate40`、`shift8`；
- `manifest.json` 保存来源、扰动方向和实际面积；
- 每个变体的文件名和推理顺序一致。

上传至服务器：

```text
/mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_mask_robustness_s42
```

## Baseline 批量推理

```bash
mkdir -p logs/mask_robustness

for variant in original erode20 dilate20 dilate40 shift8; do
  nohup env CUDA_VISIBLE_DEVICES=2 python inference.py \
    --config configs/baseline.json \
    --model_ckpt_root ./model/baseline_sd2_2000_s42_noise_0.0 \
    --ddim_scheduler_root ./scheduler \
    --categories Plastic_Bomo \
    --dataset_type mvtec \
    --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_mask_robustness_s42/${variant} \
    --use_paired_normal \
    --output_name ./results/mask_robustness/${variant}/baseline \
    > logs/mask_robustness/baseline_${variant}.log 2>&1
done &
```

## CARF 批量推理

建议等 Baseline 完成后再运行，或为其指定另一张空闲显卡。

```bash
for variant in original erode20 dilate20 dilate40 shift8; do
  nohup env CUDA_VISIBLE_DEVICES=2 python inference.py \
    --config configs/carf_only.json \
    --model_ckpt_root ./model/carf_only_v2_sd2_2000_s42 \
    --ddim_scheduler_root ./scheduler \
    --categories Plastic_Bomo \
    --dataset_type mvtec \
    --base_dir /mnt/sda/cbj/dataset/Plastic_Bomo/Plastic_Bomo_mask_robustness_s42/${variant} \
    --use_paired_normal \
    --output_name ./results/mask_robustness/${variant}/sd2_2000_s42_carf_only_v2 \
    > logs/mask_robustness/carf_${variant}.log 2>&1
done &
```

外层循环本身已放入后台，循环内部的 `nohup` 不加 `&`，因此五种扰动会依次运行，避免同一张显卡同时加载五份模型。

## 完成检查

每个方法、每种扰动应生成两类共 40 张图：

```bash
for method in baseline sd2_2000_s42_carf_only_v2; do
  for variant in original erode20 dilate20 dilate40 shift8; do
    count=$(find "results/mask_robustness/${variant}/${method}/Plastic_Bomo" \
      -path '*/image/*' -type f | wc -l)
    echo "${method} ${variant}: ${count}"
  done
done
```

所有输出均应为 `40`。

