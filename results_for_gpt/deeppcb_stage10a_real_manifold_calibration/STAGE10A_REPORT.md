# DeepPCB Stage10A Real-Manifold Calibration

Status: **INSTANCE_ADAPTIVE_TRUST_REGION_SUPPORTED**
Frozen normalization protocol for Stage10B: **instance-adaptive**
Variation floor: **NONZERO_VARIATION_FLOOR_SUPPORTED**

This calibration used 775 corrected Stage3R OOF real instances only. Every feature came from a teacher whose training split excluded the instance's physical pair. Official validation, synthetic samples, downstream metrics, Stage9E, Stage9F, and Stage9G were not used for numerical calibration.

The audit covered 100 physical pairs and all six classes. Each pair occurs in exactly one holdout fold, no holdout pair appears in its teacher's training split, and all three corrected banks have 896-dimensional features whose manifest order and class IDs match the NPZ contents.

## Radius definition and normalization

The primary local radius is the mean cosine distance to the five nearest real neighbors from the same class and same OOF fold. k=3 and k=10 are retained in the output as robustness alternatives; k=5 is a frozen first candidate, not claimed to be universally optimal.

Instance normalization materially reduced class-scale differences:

| normalization | mean pairwise Wasserstein | class median spread | mean within-class CV |
|---|---:|---:|---:|
| global | 0.089210 | 0.197296 | 0.472507 |
| class-adaptive | 0.063871 | 0.067997 | 0.472507 |
| instance-adaptive | 0.010159 | 0.007926 | 0.165089 |

According to the pre-registered simplicity rule, neither global nor class-adaptive normalization was within 10% of the instance-adaptive Wasserstein result. The future Stage10B protocol is therefore frozen as instance-adaptive.

## Frozen trust-region candidates

| band | lower beta | upper beta | real top5 coverage |
|---|---:|---:|---:|
| conservative | 0.546280 | 1.106665 | 0.7399 |
| medium | 0.546280 | 1.184703 | 0.8898 |
| wide | 0.546280 | 1.235615 | 0.9399 |

The lower beta is the real-only q05 of nearest-neighbor distance divided by the source's k5 radius. Its value (0.546280) and normalized nearest-neighbor median (0.788821) support a nonzero variation floor rather than forced source copying. Upper bounds are the q75, q90, and q95 of 3,875 real top5-neighbor relations. These values were not fitted to synthetic quality or detector utility.

The future Task-Manifold Variation Band is a per-generated-sample generator constraint. BootstrapGuard remains a separate selected-dataset/set-composition mechanism; the two innovations are not merged.

Future candidate only (not implemented): `beta_L * r_i <= d_i <= beta_U * r_i` with band loss `max(0, beta_L*r_i-d_i)^2 + max(0, d_i-beta_U*r_i)^2`.
