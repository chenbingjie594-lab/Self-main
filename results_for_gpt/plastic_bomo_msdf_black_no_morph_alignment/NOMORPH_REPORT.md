# Experiment 8 — MSDF Black Morphology Alignment Ablation

Status: `NOMORPH_EXPLORATORY_MIXED`.

G2 was trained independently from the same public SD2 base for exactly 2000 successful optimizer updates. Morphology alignment was disabled in training and checkpoint reload; gamma remained 1.0. No other ablation, reroll, filtering, checkpoint selection, extra seed or extra bank was used.

## Black results

| Method | Black KID ↓ | Black FID ↓ |
| --- | ---: | ---: |
| Raw SD2 | 0.0379734447 | 161.766924 |
| Original MSDF | 0.0800462518 | 208.478914 |
| MSDF-NoMorph | 0.0810946196 | 203.537590 |

## Overall results

The hybrid combines the frozen gamma=0 Flash checkpoint/images with the independently trained NoMorph Black checkpoint/images; it is not one jointly trained model.

| Method | Overall KID ↓ | Overall FID ↓ |
| --- | ---: | ---: |
| Raw SD2 | 0.0352225710 | 149.427503 |
| Original Class-Adaptive MSDF | 0.0498916065 | 165.131136 |
| NoMorph-Black Hybrid MSDF | 0.0486682954 | 162.473638 |

## Comparisons

Negative differences mean NoMorph/Hybrid is lower and better.

| Comparison | Δ KID | Relative KID | Δ FID | Relative FID |
| --- | ---: | ---: | ---: | ---: |
| G2 NoMorph minus G1 Full Black | +0.0010483678 | +1.310% | -4.941324 | -2.370% |
| G2 NoMorph minus G0 Raw Black | +0.0431211749 | +113.556% | +41.770666 | +25.822% |
| Hybrid minus Original Adaptive Overall | -0.0012233111 | -2.452% | -2.657498 | -1.609% |
| Hybrid minus Raw Overall | +0.0134457245 | +38.174% | +13.046135 | +8.731% |

## Required findings

1. Disabling morphology alignment did not lower Black KID (0.0810946196 vs 0.0800462518).
2. It also lowered Black FID (203.537590 vs 208.478914).
3. The Black gap to Raw changed from KID +0.0420728071 to +0.0431211749, and FID +46.711990 to +41.770666.
4. The hybrid changed Overall KID by -0.0012233111 and Overall FID by -2.657498 versus original class-adaptive MSDF.
5. Further morphology research is supported for independent validation.

## Interpretation boundary

This is a Bank-A development ablation evaluated against V2 TRAIN references that may overlap generator training. It is not an independent test result and cannot establish final-paper superiority.

The historical G1 run did not preserve a step-0 adapter SHA or full RNG-state ledger. The same seed, sorted sample order, epoch permutation, reference rule, runtime and initialization code were reused, but exact byte-for-byte G1/G2 random-trajectory identity cannot be independently proven. The result therefore remains an exploratory controlled ablation rather than a strict causal estimate.

The experiment stops here. No new structure search, loss search, extra seed, Bank B/C, BootstrapGuard, detector training or other generator experiment was started.
