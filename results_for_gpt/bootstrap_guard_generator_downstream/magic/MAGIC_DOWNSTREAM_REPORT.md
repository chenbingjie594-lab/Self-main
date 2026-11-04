# MAGIC downstream detection

| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---:|---:|---:|---:|
| Real-only | 0.8742 ± 0.0268 | 0.8444 ± 0.0393 | 0.8379 ± 0.0546 | 0.4065 ± 0.0260 |
| RealRepeat | 0.8753 ± 0.0533 | 0.8831 ± 0.0443 | 0.8739 ± 0.0439 | 0.4182 ± 0.0091 |
| Raw SD2 Bank A | 0.8749 ± 0.0421 | 0.8430 ± 0.0304 | 0.8610 ± 0.0196 | 0.4061 ± 0.0108 |
| SD2+LoRA Bank A | 0.9021 ± 0.0250 | 0.8817 ± 0.0277 | 0.8800 ± 0.0192 | 0.4114 ± 0.0177 |
| TF-IDG Bank A | 0.8613 ± 0.0414 | 0.8559 ± 0.0532 | 0.8665 ± 0.0235 | 0.4179 ± 0.0109 |
| BootstrapGuard | 0.9041 ± 0.0098 | 0.8740 ± 0.0392 | 0.8746 ± 0.0160 | 0.4406 ± 0.0012 |
| MAGIC Bank A | 0.8334 ± 0.0787 | 0.8290 ± 0.0500 | 0.8443 ± 0.0325 | 0.3916 ± 0.0069 |

## Analysis

MAGIC Bank A reaches 0.3916 ± 0.0069 mAP50-95. Flash AP50-95 is 0.3668 ± 0.0084; Black is 0.4164 ± 0.0219.

Absolute mAP50-95 differences: MAGIC − RealRepeat -0.0266; MAGIC − Raw SD2 -0.0145; MAGIC − LoRA -0.0199; MAGIC − TF-IDG -0.0263; MAGIC − BootstrapGuard -0.0490.

MAGIC is crop-native. Each audited 512×512 output was pasted back into its SHA-verified frozen normal canvas with the frozen inverse crop transform. No failed sample, empty CAMA mask, or weak result was removed. Early stopping means only per-epoch exposure is matched, not strict total compute.
