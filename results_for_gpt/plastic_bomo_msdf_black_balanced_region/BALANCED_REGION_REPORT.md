# Experiment 11 — Gradient-Balanced Region Supervision

Status: `BALANCED_REGION_EXPLORATORY_IMPROVED`.

## Phase A — frozen Region calibration

Calibration used the exact 12 Black sample/reference pairs frozen by Experiment 10 (8 self-reference and 4 cross-reference) and the frozen BackgroundOnly step2000 checkpoint. It restored the same RNG state for the base and Region forwards and performed zero optimizer updates.

The calibration passed. `lambda_region = 0.0012344430664000732` from `min(0.02, r_unet, r_msdf)`, with `r_unet = 0.025107955891676272` and `r_msdf = 0.0012344430664000732`. The full-precision value was frozen before training.

| Parameter group | Median base norm | Median Region norm | Base/Region ratio | Mean cosine | Valid | Zero | Nonfinite |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UNet | 0.3606600666 | 14.36437391 | 0.02510795589 | 0.028156 | 12 | 0 | 0 |
| MSDF | 0.0009617077267 | 0.7790620344 | 0.001234443066 | 0.549825 | 12 | 0 | 0 |

## Phase B — BalancedRegion training

G3 was trained independently from the same public SD2 base for exactly 2000 successful optimizer updates. The frozen objective was `L_denoising + 0.20 * L_background + 0.0012344430664000732 * L_region`; Reconstruction, Structure, Multiscale and Support weights were zero. Morphology alignment, both branches, dynamic gate and up-block injection remained enabled, with gamma 1.0.

## Black results

| Method | Black KID ↓ | Black FID ↓ |
| --- | ---: | ---: |
| Raw SD2 | 0.0379734447 | 161.766924 |
| Original MSDF | 0.0800462518 | 208.478914 |
| MSDF-NoAux | 0.0672672960 | 199.143610 |
| MSDF-BackgroundOnly | 0.0610055755 | 191.966915 |
| MSDF-BalancedRegion | 0.0593810876 | 187.683578 |

## Overall results

The hybrid combines the frozen gamma=0 Flash checkpoint/images with the independently trained BalancedRegion Black checkpoint/images; it is not one jointly trained model.

| Method | Overall KID ↓ | Overall FID ↓ |
| --- | ---: | ---: |
| Raw SD2 | 0.0352225710 | 149.427503 |
| Original Class-Adaptive MSDF | 0.0498916065 | 165.131136 |
| NoAux-Black Hybrid MSDF | 0.0458934929 | 160.520351 |
| BackgroundOnly-Black Hybrid MSDF | 0.0429452473 | 154.905368 |
| BalancedRegion-Black Hybrid MSDF | 0.0416453387 | 152.782300 |

## Comparisons

Negative differences mean BalancedRegion/Hybrid is lower and better.

| Comparison | Δ KID | Relative KID | Δ FID | Relative FID |
| --- | ---: | ---: | ---: | ---: |
| G3 BalancedRegion minus G1 Full Black | -0.0206651642 | -25.817% | -20.795336 | -9.975% |
| G3 BalancedRegion minus G2 NoAux Black | -0.0078862084 | -11.724% | -11.460032 | -5.755% |
| G3 BalancedRegion minus BackgroundOnly Black | -0.0016244879 | -2.663% | -4.283336 | -2.231% |
| G3 BalancedRegion minus G0 Raw Black | +0.0214076429 | +56.375% | +25.916654 | +16.021% |
| BalancedRegion Hybrid minus Original Adaptive Overall | -0.0082462678 | -16.528% | -12.348836 | -7.478% |
| BalancedRegion Hybrid minus NoAux Hybrid Overall | -0.0042481541 | -9.257% | -7.738051 | -4.821% |
| BalancedRegion Hybrid minus BackgroundOnly Hybrid Overall | -0.0012999086 | -3.027% | -2.123068 | -1.371% |
| BalancedRegion Hybrid minus Raw Overall | +0.0064227677 | +18.235% | +3.354797 | +2.245% |

## Required findings

1. The actual frozen `lambda_region` was `0.0012344430664000732`.
2. Region gradient calibration passed with finite, nonzero gradients in both UNet and MSDF groups.
3. Black KID was below 0.06100558 (0.0593810876).
4. Black FID was below 191.966915 (187.683578).
5. Overall KID was below 0.04294525 (0.0416453387).
6. Overall FID was below 154.905368 (152.782300).
7. Relative to Raw SD2, the Black KID gap changed from +0.0230321308 to +0.0214076429, and the Black FID gap changed from +30.199990 to +25.916654.
8. Adding calibrated Region supervision improved both Black distribution metrics relative to BackgroundOnly.
9. Further work on gradient-balanced auxiliary supervision is supported for independent validation.

## Interpretation boundary

This is a Bank-A development ablation evaluated against V2 TRAIN references that may overlap generator training. It is not an independent test result and cannot establish final-paper superiority.

Experiment 10 recorded initialization and RNG summaries. Experiment 11 reused the same initialization rule, seed, sample-order rule and reference rule, but complete byte-for-byte trajectory equivalence is not assumed.

The experiment stops here. No other loss restoration, weight search, new Flash model, architecture change, extra seed, Bank B/C, BootstrapGuard or detector training was started.
