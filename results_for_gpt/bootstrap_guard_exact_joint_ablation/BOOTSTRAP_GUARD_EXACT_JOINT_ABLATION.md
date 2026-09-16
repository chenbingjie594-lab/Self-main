# BootstrapGuard Exact Joint-Component Ablation

Status: **JOINT_GUIDANCE_PARTIAL**

Old MarginalGuard downstream results are excluded as invalid for causal ablation.

| Method | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| DiversityQuality | 0.8724 | 0.8818 | 0.8828 | 0.4186 +/- 0.0131 |
| ExactMarginal | 0.8879 | 0.8769 | 0.8847 | 0.4244 +/- 0.0248 |
| FullJointGuard | 0.9041 | 0.8740 | 0.8746 | 0.4406 +/- 0.0012 |

FullJointGuard - ExactMarginal paired mAP50-95: 0.016164 +/- 0.023621; wins 2/3.

The intervention changes joint bootstrap guidance for Black selection. Flash IDs are identical and serve as a sanity check; class effects can still interact through the shared detector.
