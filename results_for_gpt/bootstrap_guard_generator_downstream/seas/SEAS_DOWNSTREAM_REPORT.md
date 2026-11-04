# SeaS downstream detection

**Comparison classification: B — native-generation extension comparison.** This is `NATIVE_GENERATION_NON_POSITION_MATCHED` and must not be described in a thesis as a strictly position-paired controlled generator comparison.

SeaS uses its formal SD1.4 Separation-and-Sharing checkpoint and native 512×512 outputs. YOLO labels are computed from each generated mask; frozen Raw-SD2 boxes are not moved or reused.

| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---:|---:|---:|---:|
| Real-only | 0.8742 ± 0.0268 | 0.8444 ± 0.0393 | 0.8379 ± 0.0546 | 0.4065 ± 0.0260 |
| RealRepeat | 0.8753 ± 0.0533 | 0.8831 ± 0.0443 | 0.8739 ± 0.0439 | 0.4182 ± 0.0091 |
| Raw SD2 Bank A | 0.8749 ± 0.0421 | 0.8430 ± 0.0304 | 0.8610 ± 0.0196 | 0.4061 ± 0.0108 |
| SD2+LoRA Bank A | 0.9021 ± 0.0250 | 0.8817 ± 0.0277 | 0.8800 ± 0.0192 | 0.4114 ± 0.0177 |
| TF-IDG Bank A | 0.8613 ± 0.0414 | 0.8559 ± 0.0532 | 0.8665 ± 0.0235 | 0.4179 ± 0.0109 |
| MAGIC Bank A | 0.8334 ± 0.0787 | 0.8290 ± 0.0500 | 0.8443 ± 0.0325 | 0.3916 ± 0.0069 |
| DefectFill (Partial Reimplementation) | 0.7988 ± 0.1099 | 0.8054 ± 0.0520 | 0.8158 ± 0.0557 | 0.3822 ± 0.0187 |
| BootstrapGuard | 0.9041 ± 0.0098 | 0.8740 ± 0.0392 | 0.8746 ± 0.0160 | 0.4406 ± 0.0012 |
| SeaS (Native Generation, Non-Position-Matched) | 0.8188 ± 0.0454 | 0.8563 ± 0.0141 | 0.8355 ± 0.0096 | 0.3755 ± 0.0229 |

## Analysis

SeaS (Native Generation, Non-Position-Matched) reaches 0.3755 ± 0.0229 mAP50-95. Flash AP50-95 is 0.3718 ± 0.0278; Black is 0.3792 ± 0.0277.

Absolute differences: SeaS − RealRepeat -0.0427; SeaS − Raw SD2 -0.0306; SeaS − TF-IDG -0.0424; SeaS − BootstrapGuard -0.0651.

All 80 native Bank-A images were retained without selection or reroll. Per-epoch exposure is matched; early stopping may make total compute differ.
