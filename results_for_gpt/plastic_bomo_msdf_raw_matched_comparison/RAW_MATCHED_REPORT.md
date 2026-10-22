# Experiment 7 — Raw SD2 vs Class-Adaptive MSDF

Status: `EXPLORATORY_NOT_IMPROVED`.

No training, generation, replacement, filtering, gamma search, MSDF modification, BootstrapGuard, or detector training was performed.

## Matched protocol

All four groups use the same 80 frozen Bank-A slots (Flash 40, Black 40), conditions, bboxes, prompts, sample seeds, bank seed and ROI geometry. Raw SD2 uses its independently trained Raw checkpoint. MSDF gamma=0 only disables the final residual contribution and is not Raw SD2.

All metrics use the same 172 V2 TRAIN real ROI features. Overall KID/FID was recomputed from each complete 80-image distribution. Raw features were reused only after the Experiment 1 cache, images, slots and crops passed SHA/binding checks; no new Inception forward was needed.

## Complete matched results

| Group | Method | Overall KID ↓ | Overall FID ↓ | Flash KID ↓ | Flash FID ↓ | Black KID ↓ | Black FID ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G0 | Raw SD2 | 0.0352225710 | 149.427503 | 0.0609828974 | 181.297468 | 0.0379734447 | 161.766924 |
| G1 | MSDF Uniform gamma=0 | 0.0510316436 | 166.880548 | 0.0501908097 | 161.149091 | 0.0828684324 | 211.209226 |
| G2 | MSDF Uniform gamma=1 | 0.0515171327 | 165.268961 | 0.0533428952 | 161.288652 | 0.0800462518 | 208.478914 |
| G3 | MSDF Class-Adaptive gamma | 0.0498916065 | 165.131136 | 0.0501908097 | 161.149091 | 0.0800462518 | 208.478914 |

## Differences

Negative values mean the first method is lower/better for KID or FID.

| Comparison | Δ Overall KID | Relative KID | Δ Overall FID | Relative FID | Δ Flash KID | Δ Black KID |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G1 minus Raw | +0.0158090726 | +44.883% | +17.453044 | +11.680% | -0.0107920878 | +0.0448949877 |
| G2 minus Raw | +0.0162945617 | +46.262% | +15.841458 | +10.601% | -0.0076400022 | +0.0420728071 |
| G3 minus Raw | +0.0146690356 | +41.647% | +15.703633 | +10.509% | -0.0107920878 | +0.0420728071 |
| G3 minus G2 | -0.0016255262 | -3.155% | -0.137825 | -0.083% | -0.0031520855 | +0.0000000000 |

## Required findings

1. G3 Overall KID is not lower than Raw SD2 (0.0498916065 vs 0.0352225710).
2. G3 Overall FID is not lower than Raw SD2 (165.131136 vs 149.427503).
3. Black KID is +0.0420728071 (+110.795%) and Black FID is +46.711990 (+28.876%) relative to Raw.
4. Flash KID is -0.0107920878 (-17.697%) and Flash FID is -20.148376 (-11.113%) relative to Raw.
5. G3's change from G2 comes entirely from Flash because both groups use the same gamma=1 Black images and features.
6. Overall KID does not improve while Black KID worsens versus Raw.

## Interpretation boundary and next step

This is a Bank-A development comparison using V2 TRAIN references that may overlap generator training. The class gamma was selected on this development reference. These values are not directly comparable to historical final_eval and do not establish final-paper superiority or SOTA.

Stop gamma tuning and move to training-level morphology-guidance or dual-branch-fusion ablation.

The experiment stops here; no follow-up experiment was started.
