# Experiment 9 — MSDF Black No Auxiliary Losses

Status: `NOAUX_EXPLORATORY_IMPROVED`.

G3 was trained independently from the same public SD2 base for exactly 2000 successful optimizer updates. Morphology alignment, both adapter branches, support head, dynamic gate and up-block injection remained enabled; gamma remained 1.0. Only the six auxiliary training-loss weights were set to zero, so the optimized scalar objective equaled conditional_loss.

## Black results

| Method | Black KID ↓ | Black FID ↓ |
| --- | ---: | ---: |
| Raw SD2 | 0.0379734447 | 161.766924 |
| Original MSDF | 0.0800462518 | 208.478914 |
| MSDF-NoMorph | 0.0810946196 | 203.537590 |
| MSDF-NoAux | 0.0672672960 | 199.143610 |

## Overall results

The hybrid combines the frozen gamma=0 Flash checkpoint/images with the independently trained NoAux Black checkpoint/images; it is not one jointly trained model.

| Method | Overall KID ↓ | Overall FID ↓ |
| --- | ---: | ---: |
| Raw SD2 | 0.0352225710 | 149.427503 |
| Original Class-Adaptive MSDF | 0.0498916065 | 165.131136 |
| NoMorph-Black Hybrid MSDF | 0.0486682954 | 162.473638 |
| NoAux-Black Hybrid MSDF | 0.0458934929 | 160.520351 |

## Comparisons

Negative differences mean NoAux/Hybrid is lower and better.

| Comparison | Δ KID | Relative KID | Δ FID | Relative FID |
| --- | ---: | ---: | ---: | ---: |
| G3 NoAux minus G1 Full Black | -0.0127789558 | -15.964% | -9.335304 | -4.478% |
| G3 NoAux minus G2 NoMorph Black | -0.0138273236 | -17.051% | -4.393980 | -2.159% |
| G3 NoAux minus G0 Raw Black | +0.0292938513 | +77.143% | +37.376686 | +23.105% |
| NoAux Hybrid minus Original Adaptive Overall | -0.0039981137 | -8.014% | -4.610785 | -2.792% |
| NoAux Hybrid minus NoMorph Hybrid Overall | -0.0027748026 | -5.701% | -1.953287 | -1.202% |
| NoAux Hybrid minus Raw Overall | +0.0106709219 | +30.296% | +11.092848 | +7.424% |

## Required findings

1. Removing the six auxiliary losses lowered Black KID (0.0672672960 vs 0.0800462518).
2. It lowered Black FID (199.143610 vs 208.478914).
3. Versus NoMorph, NoAux changed Black KID by -0.0138273236 and Black FID by -4.393980.
4. The Black gap to Raw changed from KID +0.0420728071 to +0.0292938513, and FID +46.711990 to +37.376686.
5. The NoAux hybrid changed Overall KID by -0.0039981137 versus original class-adaptive MSDF and -0.0027748026 versus NoMorph hybrid.
6. The NoAux hybrid changed Overall FID by -4.610785 versus original class-adaptive MSDF and -1.953287 versus NoMorph hybrid.
7. Auxiliary-loss redesign is supported for independent validation.

## Interpretation boundary

This is a Bank-A development ablation evaluated against V2 TRAIN references that may overlap generator training. It is not an independent test result and cannot establish final-paper superiority.

The historical G1 run did not preserve a step-0 adapter SHA or full RNG-state ledger. The same seed, sorted sample order, epoch permutation, reference rule, runtime and initialization code were reused, but exact byte-for-byte G1/G3 random-trajectory identity cannot be independently proven. The result therefore remains an exploratory controlled ablation rather than a strict causal estimate.

The experiment stops here. No per-loss restoration, loss-weight search, new Flash model, architecture change, extra seed, Bank B/C, BootstrapGuard, detector training or other generator experiment was started.
