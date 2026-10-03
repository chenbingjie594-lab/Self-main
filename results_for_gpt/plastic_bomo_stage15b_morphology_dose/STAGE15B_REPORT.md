# Plastic_Bomo Stage15B — Morphology Dose vs Class Interaction

Status: `MORPHOLOGY_SIGNAL_PRESENT_DOSE_HYPOTHESIS_NOT_CONFIRMED`.

BalancedMorph50 is a mechanistic probe built from frozen HighMorph/Random lists. It is not a proposed final selector. Only three M50 detectors were trained; all twelve Stage15A metrics were reused. No Stage16 process was started.

## Paired downstream effects

Primary outcome: mAP50-95. Values are percentage-point deltas, mean ± sample std over seeds 42, 3407, 2026.

| Effect | seed42 | seed3407 | seed2026 | Mean ± std | Positive / negative |
|---|---:|---:|---:|---:|---:|
| D50_00 | -2.056 | +2.141 | +1.938 | +0.674 ± 2.367 | 2/3 / 1/3 |
| D_balance | -3.731 | -1.521 | +1.967 | -1.095 ± 2.873 | 1/3 / 2/3 |
| D80_50 | +4.342 | -0.845 | -2.392 | +0.368 ± 3.527 | 1/3 / 2/3 |

D50_00=M50−M00; D_balance=M50−(M10+M01)/2; D80_50=M11−M50.

## Registered gates

- A_balanced_gain_ge_0_005: PASS.
- B_balanced_wins_ge_2: PASS.
- C_same_dose_mean_ge_minus_0_005: FAIL.
- D_excess_mean_le_minus_0_005: FAIL.
- E_excess_losses_ge_2: PASS.

## Class-specific and secondary outcomes

| Outcome | M50−M00 mean ± std (pp) | Wins | M11−M50 mean ± std (pp) | Negative seeds |
|---|---:|---:|---:|---:|
| map50 | +1.019 ± 1.549 | 2/3 | +2.477 ± 6.195 | 1/3 |
| recall | -1.005 ± 2.395 | 1/3 | +4.418 ± 5.638 | 0/3 |
| flash_ap50_95 | +2.076 ± 4.222 | 2/3 | +1.330 ± 6.421 | 2/3 |
| black_ap50_95 | -0.727 ± 0.863 | 1/3 | -0.594 ± 0.659 | 2/3 |

Class-specific assessment: `RESIDUAL_CLASS_DEPENDENCE_PRESENT`.

| Arm | Flash AP50-95 by seed (42, 3407, 2026), % | Black AP50-95 by seed, % |
|---|---|---|
| M00 | 36.444, 30.932, 36.333 | 44.441, 42.908, 42.990 |
| M10 | 41.804, 36.452, 36.445 | 47.506, 45.302, 40.862 |
| M01 | 32.690, 36.874, 38.361 | 46.471, 43.703, 42.864 |
| M11 | 42.257, 35.546, 36.124 | 43.199, 40.887, 42.291 |
| M50 | 33.741, 36.230, 39.966 | 43.032, 41.892, 43.234 |

## Training-free concentration diagnostics

Features use the existing DWBG real-only OOF teacher checkpoints and P3/P4/P5 detection-head-input ROIs. Distances and effective rank are computed within each fold and then averaged, never between incompatible teacher spaces. Historical real feature banks contain teacher-training records; they are not mislabeled as holdout features.

| Group | Diagnostic | M11 | Mean M10/M01/M50 | M11 minus comparison |
|---|---|---:|---:|---:|
| overall | mean_pairwise_cosine_distance | 0.142951 | 0.149823 | -0.006872 |
| overall | median_nearest_neighbor_distance | 0.020499 | 0.020916 | -0.000417 |
| overall | covariance_effective_rank | 9.356127 | 9.490370 | -0.134243 |
| overall | normalized_ER | 0.010442 | 0.010592 | -0.000150 |
| flash | mean_pairwise_cosine_distance | 0.127622 | 0.138678 | -0.011057 |
| flash | median_nearest_neighbor_distance | 0.020081 | 0.020104 | -0.000023 |
| flash | covariance_effective_rank | 7.503307 | 7.456429 | +0.046877 |
| flash | normalized_ER | 0.008374 | 0.008322 | +0.000052 |
| black | mean_pairwise_cosine_distance | 0.126352 | 0.125640 | +0.000712 |
| black | median_nearest_neighbor_distance | 0.023051 | 0.022300 | +0.000751 |
| black | covariance_effective_rank | 7.876002 | 8.211157 | -0.335154 |
| black | normalized_ER | 0.008790 | 0.009164 | -0.000374 |

| Group | Morphology std M11 | Mean std M10/M01/M50 |
|---|---:|---:|
| overall | 0.280771 | 0.529823 |
| flash | 0.193842 | 0.322636 |
| black | 0.305613 | 0.560112 |

Concentration metrics are separate descriptive outcomes and do not determine the registered downstream gates. No weighted quality score or feedback into selection was used.

## Dose membership and evidence limits

| Arm | Nominal HighMorph role count | Actual membership in frozen HighMorph list |
|---|---:|---:|
| M00 | 0 | 14 |
| M10 | 40 | 50 |
| M01 | 40 | 44 |
| M11 | 80 | 80 |
| M50 | 40 | 40 |

Nominal role dose is matched; frozen High/Random overlap means the historical arms do not have exact physical High-membership doses of 0/40/80. Report this limitation without changing any arm.

New M50 runs export raw args.yaml, 150-epoch CSVs, primary image draws/batches and optimizer attempts/successes. Historical arms retain only their original scheduled-budget evidence; exact historical optimizer-step and pretrained-byte equality are not asserted. Mosaic source-image usage is not equated with primary loader draws.

This single probe uses existing sample sets and three detector seeds. Even if the dose gates pass, compression is a mechanism hypothesis to assess against the separate diagnostics, not a proven consequence of morphology similarity alone. No additional dose sweep, generation, DeepPCB experiment or BootstrapGuard modification is authorized by this finalizer.
## Downloaded-result independent review


CPU-only review: `DOWNLOADED_RESULTS_REVIEW_PASS`, 846 checks passed. All 15 arm/seed metric records, 12 unchanged historical records, 3 raw final validation exports, 3 epoch ledgers/args/CSVs, frozen sample selection and all 3 feature NPZ caches were cross-checked. Paired effects/sample std were independently recomputed; feature rank was recomputed with SVD rather than the production Gram eigendecomposition.

Registered gates A/B/E pass; C/D fail. M50−M00 is +0.674 ± 2.367 pp, but M50−mean(M10,M01) is −1.095 ± 2.873 pp. M11−M50 is +0.368 ± 3.527 pp: two negative seeds do not establish the required negative mean dose penalty. Flash improves by +2.076 pp on average, while Black decreases by −0.727 pp. Neither Stage16 generator direction is authorized.

Successful updates are 532/534/534 despite identical 544 attempted updates. The review does not upgrade schedule equivalence into exact successful-compute equality.

### Review limitations

- No model checkpoint or image bytes are re-executed by this CPU review. Server-recorded hashes are cross-checked against downloaded ledgers and NPZs; original training images/checkpoints and validation images are not independently rehashed locally.
- The freeze hashes prove internal content consistency, not an independently timestamped pre-training freeze. Zero selection/tuning validation usage is supported by code and recorded metadata, not an external execution trace.
- Historical Stage15A runtime optimizer counters and pretrained bytes remain unavailable. New M50 runs share 544 attempted updates but have 532/534/534 successful updates due to AMP skips. Exact successful-update equality is not claimed.
- All three raw training CSVs have 15 header columns but only 8 columns at epoch150: bypassed final framework validation returned no metric placeholders, so the three learning rates appear under metric headers. The first five training columns and epoch count remain valid. Final detector APs use separately exported last.pt validation JSON, never the ragged CSV metric columns. Original CSVs are preserved unchanged.
- Historical nominal HighMorph doses differ from actual High-list membership because Random/High lists overlap. M00/M10/M01/M11/M50 actual membership is 14/50/44/80/40. Parent reuse and generation-seed concentrations also differ; this is not a one-variable randomized causal dose experiment.
- M11 is narrower in overall pairwise feature distance and morphology std, but classwise feature diagnostics are not uniformly narrower. Concentration alone does not demonstrate a downstream penalty or choose a generator mechanism.

Machine-readable checks, hashes and explicit per-seed CSV format diagnostics are retained in `downloaded_results_review.json`. No raw metrics, CSVs, thresholds, samples or historical results were changed.
