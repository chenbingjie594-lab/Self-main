# BootstrapGuard Cross-Detector Downstream Generalization Pilot

Status: **CROSS_DETECTOR_SIGNAL_POSITIVE**

| Arm | Precision | Recall | AP50 | AP50-95 | Best epoch |
|---|---:|---:|---:|---:|---:|
| real_only | 0.8248 | 0.8614 | 0.8525 | 0.3764 | 58 |
| random | 0.7524 | 0.8500 | 0.8386 | 0.3754 | 54 |
| diversity_quality | 0.8305 | 0.8864 | 0.8904 | 0.3802 | 74 |
| bootstrap_guard | 0.8113 | 0.8614 | 0.8402 | 0.3965 | 72 |

BootstrapGuard - Random AP50-95: **+0.021090**.
BootstrapGuard - DiversityQuality AP50-95: **+0.016307**.

This single-seed pilot supports cross-detector transferability of the frozen synthetic composition. It does not prove detector independence; confirmation requires the planned multi-seed extension.

Training protocol: Faster R-CNN ResNet50-FPN, FasterRCNN_ResNet50_FPN_Weights.DEFAULT (COCO), seed 42, 1536 px input.
