# Leakage-fixed protocol

Only the clean rerun below is eligible for the chapter table. The invalid historical result is excluded.

# Poisson Blending downstream detection

Traditional OpenCV `NORMAL_CLONE` with bbox-derived masks (not human pixel masks). No selection, reroll, Copy-Paste, or Poisson variant search was performed.

| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---:|---:|---:|---:|
| Real-only | 0.8742 ± 0.0268 | 0.8444 ± 0.0393 | 0.8379 ± 0.0546 | 0.4065 ± 0.0260 |
| RealRepeat | 0.8753 ± 0.0533 | 0.8831 ± 0.0443 | 0.8739 ± 0.0439 | 0.4182 ± 0.0091 |
| Raw SD2 | 0.8749 ± 0.0421 | 0.8430 ± 0.0304 | 0.8610 ± 0.0196 | 0.4061 ± 0.0108 |
| SD2+LoRA | 0.9021 ± 0.0250 | 0.8817 ± 0.0277 | 0.8800 ± 0.0192 | 0.4114 ± 0.0177 |
| TF-IDG | 0.8613 ± 0.0414 | 0.8559 ± 0.0532 | 0.8665 ± 0.0235 | 0.4179 ± 0.0109 |
| MAGIC | 0.8334 ± 0.0787 | 0.8290 ± 0.0500 | 0.8443 ± 0.0325 | 0.3916 ± 0.0069 |
| DefectFill (Partial Reimplementation) | 0.7988 ± 0.1099 | 0.8054 ± 0.0520 | 0.8158 ± 0.0557 | 0.3822 ± 0.0187 |
| SeaS (Native Generation) | 0.8188 ± 0.0454 | 0.8563 ± 0.0141 | 0.8355 ± 0.0096 | 0.3755 ± 0.0229 |
| Random | 0.8923 ± 0.0096 | 0.8291 ± 0.0365 | 0.8529 ± 0.0138 | 0.4244 ± 0.0114 |
| BootstrapGuard | 0.9041 ± 0.0098 | 0.8740 ± 0.0392 | 0.8746 ± 0.0160 | 0.4406 ± 0.0012 |
| Poisson Blending | 0.8803 ± 0.0102 | 0.8647 ± 0.0169 | 0.8769 ± 0.0153 | 0.4175 ± 0.0151 |

## Analysis

Poisson Blending reaches 0.4175 ± 0.0151 mAP50-95. Flash AP50-95 is 0.4053 ± 0.0305; Black is 0.4297 ± 0.0061.

Absolute differences: Poisson − RealRepeat -0.0007; Poisson − Raw SD2 +0.0114; Poisson − Random -0.0069; Poisson − BootstrapGuard -0.0231.

Generation visibility audit: {'VISIBLE_SIGNAL': 80}. All generated samples were retained. Per-epoch exposure is matched; early stopping may make total compute differ.
