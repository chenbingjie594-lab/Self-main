# DefectFill (Partial Reimplementation) downstream detection

This experiment is **not** labeled as the official complete DefectFill implementation. The paper-native eight-candidate Low-Fidelity Selection is absent and was not added.

| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---:|---:|---:|---:|
| Real-only | 0.8742 ± 0.0268 | 0.8444 ± 0.0393 | 0.8379 ± 0.0546 | 0.4065 ± 0.0260 |
| RealRepeat | 0.8753 ± 0.0533 | 0.8831 ± 0.0443 | 0.8739 ± 0.0439 | 0.4182 ± 0.0091 |
| Raw SD2 Bank A | 0.8749 ± 0.0421 | 0.8430 ± 0.0304 | 0.8610 ± 0.0196 | 0.4061 ± 0.0108 |
| SD2+LoRA Bank A | 0.9021 ± 0.0250 | 0.8817 ± 0.0277 | 0.8800 ± 0.0192 | 0.4114 ± 0.0177 |
| TF-IDG Bank A | 0.8613 ± 0.0414 | 0.8559 ± 0.0532 | 0.8665 ± 0.0235 | 0.4179 ± 0.0109 |
| MAGIC Bank A | 0.8334 ± 0.0787 | 0.8290 ± 0.0500 | 0.8443 ± 0.0325 | 0.3916 ± 0.0069 |
| BootstrapGuard | 0.9041 ± 0.0098 | 0.8740 ± 0.0392 | 0.8746 ± 0.0160 | 0.4406 ± 0.0012 |
| DefectFill (Partial Reimplementation) | 0.7988 ± 0.1099 | 0.8054 ± 0.0520 | 0.8158 ± 0.0557 | 0.3822 ± 0.0187 |

## Analysis

DefectFill (Partial Reimplementation) reaches 0.3822 ± 0.0187 mAP50-95. Flash AP50-95 is 0.3565 ± 0.0171; Black is 0.4079 ± 0.0416.

Absolute mAP50-95 differences: DefectFill − RealRepeat -0.0360; DefectFill − Raw SD2 -0.0239; DefectFill − LoRA -0.0293; DefectFill − TF-IDG -0.0357; DefectFill − BootstrapGuard -0.0584.

All 80 Primary Strict Bank-A images were retained without selection, replacement, or reroll. Early stopping means per-epoch exposure is matched but total compute may differ.
