# Plastic_Bomo Stage15A — Class-Conditional Morphology Causal Isolation

Status: `CLASS_INTERACTION_DOMINATES_MORPHOLOGY_SIGNAL`.

Only M10/M01 were newly trained (six runs). M00/M11 reproduce the exact Stage14B Random/High metrics and checkpoint hashes. No Stage15B design is authorized and no Stage15B experiment has started.

## Paired primary effects

All entries below are mAP50-95 percentage-point deltas. Mean ± std uses sample std across seeds 42, 3407, 2026.

| Effect | seed42 | seed3407 | seed2026 | Mean ± std | Positive seeds |
|---|---:|---:|---:|---:|---:|
| EF_B0 | +4.213 | +3.957 | -1.008 | +2.387 ± 2.943 | 2/3 |
| EF_B1 | +3.148 | -2.072 | -1.405 | -0.110 ± 2.841 | 1/3 |
| EB_F0 | -0.862 | +3.368 | +0.951 | +1.152 ± 2.122 | 2/3 |
| EB_F1 | -1.927 | -2.660 | +0.554 | -1.344 ± 1.685 | 1/3 |
| ME_F | +3.680 | +0.943 | -1.206 | +1.139 ± 2.449 | 2/3 |
| ME_B | -1.394 | +0.354 | +0.753 | -0.096 ± 1.142 | 2/3 |
| interaction | -1.065 | -6.028 | -0.397 | -2.497 ± 3.077 | 0/3 |

EF_B0=M10−M00; EF_B1=M11−M01; EB_F0=M01−M00; EB_F1=M11−M10. ME_F/ME_B average the respective conditional effects. Interaction=M11−M10−M01+M00.

## Interpretation

Flash-only replacement passes primary gates A–D: overall mAP50-95 improves by +2.387 pp (2/3 positive), and Flash AP50-95 by +3.664 pp (3/3 positive). Conditional robustness fails: under High-Black, the overall Flash effect is −0.110 pp (1/3 positive). The interaction is −2.497 pp (all three seeds negative), exceeding the registered 0.50 pp magnitude threshold.

Black-only replacement improves overall mAP50-95 by +1.152 pp and Black AP50-95 by +0.899 pp (both 2/3 positive). The gate for Black morphology being non-beneficial does not pass. These observations do not support a general rule that Flash should always receive morphology guidance while Black should always remain baseline.

## Class-specific and secondary effects

| Outcome | EF_B0 mean ± std (pp) | Positive seeds | EF_B1 mean ± std (pp) | Positive seeds |
|---|---:|---:|---:|---:|
| Flash AP50-95 | +3.664 ± 3.077 | 3/3 | +2.001 ± 6.568 | 1/3 |
| Black AP50-95 | +1.111 ± 2.824 | 2/3 | -2.220 ± 1.445 | 0/3 |
| mAP50 | +2.907 ± 3.215 | 3/3 | +0.273 ± 4.652 | 1/3 |
| Recall | +1.976 ± 1.978 | 3/3 | -0.139 ± 4.359 | 1/3 |

Full per-arm precision, recall, AP50 and AP50-95 remain in factorial_all_arms_metrics.json; all seven effects for each outcome remain in the effect JSON files.

## Manifest and training audit

All four arms contain the prescribed 40 Flash + 40 Black candidates. Every candidate ID, path, generation seed and parent/source ID agrees with the original pool. Flash High/Random overlap is 4 and Black overlap is 10; neither was removed. The full manifest digest and per-arm ID digests match.

The downloaded budget declares 138 real + 80 synthetic, 150 epochs, batch 1, and 32,700 scheduled draws/batches per run. Six new args.yaml hashes and final-last-only evaluation metadata are present. These are recorded protocol values rather than locally verified runtime counters.

## Evidence limits

- Downloaded files contain scheduled budget metadata and args.yaml hashes, but not training CSVs, actual batch counters, optimizer-step logs, args.yaml contents, or pretrained checkpoint bytes. Actual equal compute and identical initialization cannot be independently verified locally.
- The manifest hash verifies downloaded content against its recorded digest; it does not independently prove the pre-training freeze timestamp or unchanged server image/label bytes.
- This 2x2 intervention changes frozen class-specific sample sets. It establishes conditional subset effects; morphology-specific causality remains subject to residual sample differences and three-seed uncertainty.

The registered status is preserved without changing metrics, thresholds, frozen subsets or training protocol. No additional training, generation or official-validation evaluation was performed by this review.
