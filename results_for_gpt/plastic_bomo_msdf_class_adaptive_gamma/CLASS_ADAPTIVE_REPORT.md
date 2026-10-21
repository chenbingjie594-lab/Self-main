# MSDF Class-Adaptive Gamma Experiment

Status: `COMPLETE`. No training, sampling, regeneration, image copying, image encoding, or historical-result modification was performed.

## Protocol boundary

G2 references the existing 40 Flash images at gamma=0 and 40 Black images at gamma=1 from the same frozen Bank A. Every selected PNG, slot, ROI and seed passed SHA/binding checks. Overall KID/FID was recomputed from the combined 80-image feature distribution against the same 172 V2 TRAIN ROIs; it is not an average of class scores.

The V2 TRAIN reference may overlap checkpoint training data, and the class-specific gamma choice was made after observing this development set. Results are exploratory and cannot be compared directly with historical final_eval values.

## Three-group comparison

| Method | Overall KID ↓ | Overall FID ↓ | Flash KID ↓ | Black KID ↓ | Flash FID ↓ | Black FID ↓ | Overall LPIPS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Uniform gamma=0 | 0.0510316436 | 166.880548 | 0.0501908097 | 0.0828684324 | 161.149091 | 211.209226 | 0.481713 |
| Uniform gamma=1 | 0.0515171327 | 165.268961 | 0.0533428952 | 0.0800462518 | 161.288652 | 208.478914 | 0.481834 |
| Class-Adaptive gamma | 0.0498916065 | 165.131136 | 0.0501908097 | 0.0800462518 | 161.149091 | 208.478914 | 0.483068 |

## Differences: Class-Adaptive minus baseline

| Baseline | Δ Overall KID | Δ Overall FID | Δ Overall LPIPS |
| --- | ---: | ---: | ---: |
| Uniform gamma=0 | -0.0011400371 | -1.749411 | +0.001355 |
| Uniform gamma=1 | -0.0016255262 | -0.137825 | +0.001233 |

## Findings

1. Class-adaptive gamma does beat both uniform baselines on Overall KID.
2. Class-adaptive gamma does beat both uniform baselines on Overall FID.
3. Best Overall KID: **Class-Adaptive gamma**. Best Overall FID: **Class-Adaptive gamma**.
4. No KID improvement/FID degradation tradeoff occurs against either baseline.
5. Independent, source-group-isolated validation is warranted only as confirmation; this experiment cannot establish final performance.

The experiment stops here. No wider gamma search, regeneration, retraining, BootstrapGuard, morphology ablation, or downstream detector training was started.
