# DeepPCB Stage 4 Report

Status: **DEEPPCB_BG_SIGNAL_POSITIVE**

This is a single-seed (42) positive cross-dataset pilot signal; it does not confirm cross-dataset generalization.

## Protocol integrity

- Training: 100 RAW tested pair images with all 775 real GT boxes. No converted per-instance crop was used as a real detector image.
- Validation: 500 official RAW test images with 3,140 GT boxes (open 659, short 478, mousebite 586, spur 483, spurious_copper 464, pinhole 470).
- Train/validation pair overlap: 0; invalid validation boxes: 0. The 30-image spatial sanity sample passed image-coordinate and tested/template pixel-change checks.
- All arms use the same validation manifest (`875f4086574f1d6f79e5ba450c33987beb0674d626c1c3b05f6cc4b3ac3d9c44`) and the same YOLO11s seed-42 training protocol.
- Random, DiversityQuality and BootstrapGuard each contain 240 frozen Stage3R candidates: exactly 40 per class, 40 unique target pairs per class and at most 8 samples per generation seed.

## Independent best-checkpoint validation

| Method | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| real_only | 0.910338 | 0.864832 | 0.927615 | 0.671547 |
| random | 0.889217 | 0.839908 | 0.899752 | 0.599280 |
| diversity_quality | 0.895896 | 0.827380 | 0.898022 | 0.575592 |
| bootstrap_guard | 0.928035 | 0.862442 | 0.926635 | 0.639801 |

BG - Random mAP50-95: +0.040521
BG - DiversityQuality mAP50-95: +0.064209
BG - Real-only mAP50-95: -0.031745

Under the pre-registered rule, BG beats both synthetic controls and therefore receives the positive status. However, all three synthetic arms remain below Real-only. Thus the result supports **distribution-aware selection over blind or quality-only synthetic selection**, while this seed does **not** show that adding synthetic data improves over the 100-image real-only baseline.

## Per-class BG versus DiversityQuality

| Class | DQ AP50-95 | BG AP50-95 | Delta | DQ recall | BG recall |
|---|---:|---:|---:|---:|---:|
| open | 0.532360 | 0.573994 | +0.041635 | 0.811457 | 0.905918 |
| short | 0.437985 | 0.474495 | +0.036510 | 0.679916 | 0.713389 |
| mousebite | 0.540421 | 0.636241 | +0.095820 | 0.817083 | 0.856655 |
| spur | 0.554652 | 0.625864 | +0.071212 | 0.760573 | 0.834698 |
| spurious_copper | 0.749793 | 0.813039 | +0.063246 | 0.929293 | 0.930995 |
| pinhole | 0.638342 | 0.715175 | +0.076833 | 0.965957 | 0.932997 |

BG improves strict-IoU AP over DQ in all six classes. The two highlighted classes improve by +0.063246 (spurious_copper) and +0.076833 (pinhole), although pinhole recall decreases by 0.032960.

## Cross-dataset interpretation

Within-dataset seed-42 deltas point in the same direction for selector comparison: BG minus Random/DQ is +0.013382/+0.030738 on Plastic_Bomo and +0.040521/+0.064209 on DeepPCB. Absolute mAP values are not compared across datasets. Multi-seed confirmation remains a separate, manually authorized next step.

Validation: 500 RAW images, 3140 instances, train overlap 0.
Training protocol: YOLO11s, seed 42, imgsz 512, epochs 150, patience 40, batch 8.
