# TF-IDG downstream detection

| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---:|---:|---:|---:|
| Real-only | 0.8742 ± 0.0268 | 0.8444 ± 0.0393 | 0.8379 ± 0.0546 | 0.4065 ± 0.0260 |
| RealRepeat | 0.8753 ± 0.0533 | 0.8831 ± 0.0443 | 0.8739 ± 0.0439 | 0.4182 ± 0.0091 |
| Raw SD2 Bank A | 0.8749 ± 0.0421 | 0.8430 ± 0.0304 | 0.8610 ± 0.0196 | 0.4061 ± 0.0108 |
| SD2+LoRA Bank A | 0.9021 ± 0.0250 | 0.8817 ± 0.0277 | 0.8800 ± 0.0192 | 0.4114 ± 0.0177 |
| BootstrapGuard | 0.9041 ± 0.0098 | 0.8740 ± 0.0392 | 0.8746 ± 0.0160 | 0.4406 ± 0.0012 |
| TF-IDG Bank A | 0.8613 ± 0.0414 | 0.8559 ± 0.0532 | 0.8665 ± 0.0235 | 0.4179 ± 0.0109 |

## Analysis

TF-IDG is evaluated because it is training-free: AnyDoor and DINOv2 remained frozen, while the downstream detector and its frozen validation protocol match the other generator controls. The audited Bank A contributes 40 Flash and 40 Black full-canvas images; no Best-of-N, reroll, or quality selection was used.

TF-IDG reaches 0.4179 ± 0.0109 mAP50-95. Its absolute differences are +0.0114 versus Real-only, +0.0118 versus Raw SD2, +0.0065 versus LoRA, -0.0003 versus RealRepeat, and -0.0227 versus BootstrapGuard. Class AP50-95 is 0.4206 ± 0.0275 for Flash and 0.4152 ± 0.0072 for Black.

Thus, TF-IDG improves over Real-only and Raw SD2 in this three-seed experiment, but does not demonstrate an additional gain over RealRepeat and remains below BootstrapGuard. This is an empirical comparison, not a significance claim. Early stopping produced 90–137 epochs (339–499 audited optimizer steps), so only the per-epoch dataset size is matched; total compute is not strictly matched.
