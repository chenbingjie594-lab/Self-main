# Experiment 10 — MSDF Auxiliary Gradient Conflict and Background-Only Restoration

Status: `BACKGROUND_ONLY_EXPLORATORY_IMPROVED`.

## Phase A — gradient conflict diagnosis

The probe used the frozen Experiment 9 NoAux step2000 model, 12 preselected Black samples from 12 source groups, identical RNG restoration for every loss, and zero optimizer steps. Support was NOT_APPLICABLE for four cross-reference samples.

The most negative mean cosine was support on MSDF (-0.096419). The largest mean historical-weighted gradient ratio was support on MSDF (846.892643).

Background cosine: UNet mean=0.252040, negative fraction=0.000; MSDF mean=0.083110, negative fraction=0.417. The pre-frozen severe-conflict stop rule was not triggered.

Gradient conflict is a local optimization observation and is not treated as proof of KID/FID causality.

## Phase B — BackgroundOnly

G3 was trained independently from the same public SD2 base for exactly 2000 successful optimizer updates. Morphology alignment, both branches, dynamic gate and up-block injection remained enabled; gamma remained 1.0. The objective was conditional_loss + 0.20 * msdf_background_loss; the other five auxiliary weights were zero.

## Black results

| Method | Black KID ↓ | Black FID ↓ |
| --- | ---: | ---: |
| Raw SD2 | 0.0379734447 | 161.766924 |
| Original MSDF | 0.0800462518 | 208.478914 |
| MSDF-NoAux | 0.0672672960 | 199.143610 |
| MSDF-BackgroundOnly | 0.0610055755 | 191.966915 |

## Overall results

The hybrid combines the frozen gamma=0 Flash checkpoint/images with the independently trained BackgroundOnly Black checkpoint/images; it is not one jointly trained model.

| Method | Overall KID ↓ | Overall FID ↓ |
| --- | ---: | ---: |
| Raw SD2 | 0.0352225710 | 149.427503 |
| Original Class-Adaptive MSDF | 0.0498916065 | 165.131136 |
| NoAux-Black Hybrid MSDF | 0.0458934929 | 160.520351 |
| BackgroundOnly-Black Hybrid MSDF | 0.0429452473 | 154.905368 |

## Comparisons

Negative differences mean BackgroundOnly/Hybrid is lower and better.

| Comparison | Δ KID | Relative KID | Δ FID | Relative FID |
| --- | ---: | ---: | ---: | ---: |
| G3 BackgroundOnly minus G1 Full Black | -0.0190406762 | -23.787% | -16.512000 | -7.920% |
| G3 BackgroundOnly minus G2 NoAux Black | -0.0062617205 | -9.309% | -7.176695 | -3.604% |
| G3 BackgroundOnly minus G0 Raw Black | +0.0230321308 | +60.653% | +30.199990 | +18.669% |
| BackgroundOnly Hybrid minus Original Adaptive Overall | -0.0069463592 | -13.923% | -10.225768 | -6.193% |
| BackgroundOnly Hybrid minus NoAux Hybrid Overall | -0.0029482456 | -6.424% | -5.614983 | -3.498% |
| BackgroundOnly Hybrid minus Raw Overall | +0.0077226763 | +21.925% | +5.477865 | +3.666% |

## Required findings

1. BackgroundOnly lowered Black KID versus NoAux (0.0610055755 vs 0.0672672960).
2. BackgroundOnly lowered Black FID versus NoAux (191.966915 vs 199.143610).
3. Versus Full MSDF, BackgroundOnly changed Black KID by -0.0190406762 and Black FID by -16.512000.
4. The Black gap to Raw changed from KID +0.0292938513 to +0.0230321308, and FID +37.376686 to +30.199990.
5. The BackgroundOnly hybrid changed Overall KID by -0.0069463592 versus original class-adaptive MSDF and -0.0029482456 versus NoAux hybrid.
6. The BackgroundOnly hybrid changed Overall FID by -10.225768 versus original class-adaptive MSDF and -5.614983 versus NoAux hybrid.
7. Phase A found Background UNet/MSDF mean cosines of 0.252040/0.083110; Phase B improved both metrics versus NoAux. This is descriptive consistency, not causal proof.
8. Further gradient-conflict mitigation research is supported for independent validation.

## Interpretation boundary

This is a Bank-A development ablation evaluated against V2 TRAIN references that may overlap generator training. It is not an independent test result and cannot establish final-paper superiority.

Experiment 9 recorded initialization and RNG summaries. Experiment 10 reused the same seed, sample-order rule and reference rule, but complete byte-for-byte trajectory equivalence is not assumed.

The experiment stops here. No other loss restoration, weight search, new Flash model, architecture change, extra seed, Bank B/C, BootstrapGuard or detector training was started.
