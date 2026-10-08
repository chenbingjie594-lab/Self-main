# Stage17A Vanilla Synthetic Utility Stability Calibration

SYNTHETIC_HEADROOM_NOT_REPRODUCIBLE

Primary: epoch150 last.pt mAP50-95. Detector seed=42; three generation banks, not three detector seeds.

## Final metrics

| Arm | mAP50-95 (%) | Delta vs RR (pp) |
|---|---:|---:|
| RR | 39.7499 | +0.0000 |
| VA | 36.2966 | -3.4533 |
| VB | 34.3187 | -5.4312 |
| VC | 33.3872 | -6.3627 |

Mean gain: -5.0824 pp; sample SD: 1.4857 pp; wins: 0/3.
Generation nuisance SD: 1.4857 pp; range: 2.9094 pp.
Future method minimum gain: 2.9715 pp (does not change current gates).

## Compute and exposure

Successful optimizer steps exactly equal: False; range: 2.
Equal scheduled draws, batches, epochs and optimizer attempts are audited separately from successful AMP updates. A/B/C primary and secondary augmentation fetch sequences are audited.
RR repeats complete donor parent images and complete labels. Extra class-role counts match; exact bbox-instance exposure is not claimed.

## Interpretation limits

Utility is measured only under the current frozen Plastic_Bomo annotation protocol. This is not proof of true physical-defect recognition.
88 Flash + 30 Black boxes have XML lineage; two Black conversions have approximately 1px edge clipping differences. Supplemental Black has 25 images / 50 boxes with historical prelabel import and no instance-level human verification evidence. Suspected omissions and Small/Big Black semantic boundaries remain unchanged.
The generator is the explicitly authorized train-only rebuild, not the unavailable historical checkpoint. Six known related-validation normal sources were excluded in a separately frozen background pool; unknown acquisition relationships are not ruled out.
Stage16 stopped records and historical Stage14–15 conclusions remain unchanged. No new modules, selectors, BootstrapGuard or DeepPCB were used. Stage17B is not automatically started.
