# DeepPCB Stage 4 Multi-Seed Report

Selector status: **DEEPPCB_SELECTOR_TRANSFER_CONFIRMED**
Augmentation status: **AUGMENTATION_NEGATIVE**

Seeds: 42 (reused without retraining), 3407, 2026. All 12 checkpoints were independently validated against the same frozen 500-image, 3,140-instance official RAW validation set; validation manifest SHA-256 is `875f4086574f1d6f79e5ba450c33987beb0674d626c1c3b05f6cc4b3ac3d9c44`. Stage3R selections, synthetic budget (40/class), model initialization and all training settings other than seed were frozen.

## mAP50-95 by training seed

| Method | 42 | 3407 | 2026 | Mean ± population std |
|---|---:|---:|---:|---:|
| Real-only | 0.671547 | 0.642101 | 0.646036 | 0.653228 ± 0.013052 |
| Random | 0.599280 | 0.648838 | 0.629772 | 0.625963 ± 0.020410 |
| DiversityQuality | 0.575592 | 0.647180 | 0.638886 | 0.620553 ± 0.031972 |
| BootstrapGuard | 0.639801 | 0.655081 | 0.620840 | 0.638574 ± 0.014006 |

## Paired seed-42/3407/2026 differences

| Comparison | 42 | 3407 | 2026 | Mean ± population std | Wins |
|---|---:|---:|---:|---:|---:|
| BG − Random | +0.040521 | +0.006243 | −0.008933 | +0.012610 ± 0.020685 | 2/3 |
| BG − DQ | +0.064209 | +0.007900 | −0.018047 | +0.018021 ± 0.034335 | 2/3 |
| BG − Real-only | −0.031745 | +0.012979 | −0.025197 | −0.014654 ± 0.019722 | 1/3 |

The pre-specified selector rule is met: BG has positive mean differences and at least 2/3 paired wins against both synthetic controls. This is a **relative selector transfer** finding, not proof of robust absolute augmentation gain. Seed 2026 reverses both selector comparisons; with only three seeds, the confirmation should be described as criterion-based rather than uniformly stable. Against Real-only, BG loses on average and in two of three seeds, hence the separate negative augmentation status.

## Six-class BG − DQ strict-IoU AP

| Class | Mean AP50-95 difference |
|---|---:|
| open | +0.028243 |
| short | +0.002910 |
| mousebite | +0.021149 |
| spur | +0.009575 |
| spurious_copper | +0.008416 |
| pinhole | +0.037833 |

All six class means are positive, but the class margins are smaller than in seed42 alone. Full per-method mean and std for each class are in `per_class_multiseed.json`.

## Plastic_Bomo comparison

Only within-dataset paired deltas are compared. Plastic_Bomo BG − Random/DQ/Real-only means are +0.016172/+0.030872/+0.034089 (3/3 wins for each). DeepPCB corresponding means are +0.012610/+0.018021/−0.014654 (2/3, 2/3 and 1/3 wins). Absolute mAP across the two datasets is not compared.

Seed42 checkpoint hashes and validation metrics were matched against the original Stage4 results. Seven requested plots and all raw/aggregate tables are included. No selector, candidate pool, synthetic budget or validation protocol was changed.

| Method | mAP50-95 mean | std |
|---|---:|---:|
| real_only | 0.653228 | 0.013052 |
| random | 0.625963 | 0.020410 |
| diversity_quality | 0.620553 | 0.031972 |
| bootstrap_guard | 0.638574 | 0.014006 |

BG-Random: mean +0.012610, wins 2/3
BG-DQ: mean +0.018021, wins 2/3
BG-Real: mean -0.014654, wins 1/3
