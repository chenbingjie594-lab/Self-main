# Stage25 formal result — MSDF-V2 improvement not confirmed

This is the final-result index, not an authorization to rerun or start a new stage.

Authoritative completed run: [`detector_recovery_20261014_032214`](../detector_recovery_20261014_032214/).

**Status: `MSDF_GENERATOR_IMPROVEMENT_NOT_CONFIRMED`.** The runtime qualified and the experiment completed; this is a negative downstream result, not an unfinished run or a runtime-qualification failure.

The older status/report at the Stage25 root records the original **local Windows P0** stop. It is retained unchanged for provenance and is not the latest formal server result. `server_package_preparation/` and `server_console_revision/` are also historical preparation records, not final performance conclusions.

## Frozen-protocol results

Primary metric: mAP50–95, shown as percentages. Differences are percentage points (pp).

| Generation bank | Raw | MSDF | MSDF − Raw |
| --- | ---: | ---: | ---: |
| A | 29.7100 | 25.9814 | −3.7286 pp |
| B | 33.0818 | 32.9551 | −0.1267 pp |
| C | 37.3483 | 33.6790 | −3.6693 pp |
| Mean | 33.3800 | 30.8718 | −2.5082 pp |

Real-repeat (RR): 35.9702%; mean MSDF − RR: −5.0983 pp. MSDF strict wins are **0/3 against Raw** and **0/3 against RR**.

All three primary gates failed: mean improvement ≥0.5 pp, at least two strict matched-bank wins, and both class-mean deltas ≥−2 pp. Class-mean MSDF − Raw: flash −0.8361 pp; black −4.1803 pp. The larger degradation is observed in black; this experiment does **not** identify a particular internal MSDF branch or loss as its cause.

No generator-improvement claim or augmentation-utility claim is supported by this run. This is limited to the current frozen protocol, not a proof about every possible MSDF variant or dataset.

## Completion and interruption accounting

- The qualified Linux/RTX3090 runtime is `server_runtime_20261013_080546_552698/`.
- Both generators completed 2,000 successful updates. All 240 planned images (A/B/C, each 40 flash + 40 black) were retained without selection or reroll.
- `formal_20261013_085920_253633/` contains the original generation run and interrupted detector A. Its nine complete detector epochs and partial next epoch remain preserved; they are not treated as a completed primary detector.
- Explicit manual recovery reused the same two frozen generators and all 240 byte-verified images. No additional generator training or sampling occurred. Each primary detector A/B/C restarted from the original public initialization and seed 42, epoch 1 through 150; this was not epoch 10/optimizer resume.
- Each MSDF detector used 32,400 draws and 600 scheduled optimizer attempts: 592 successful updates and 8 AMP skips. The reused Raw A/B/C and RR comparators have the same schedule and actual update/skip counts. No skipped-update compensation was applied.
- All three epoch 150 native EMA `last.pt` files were frozen before the single unified evaluation. No best-checkpoint selection or validation-based tuning was introduced.
- New evaluation was 3 × 48 images = 144 image forwards. The four qualified historical comparator metrics were reused without new comparator training or evaluation.
- The final_eval set was historically observed in Stage20B. It is **not an untouched independent test set**. Three generation banks are not three detector seeds and do not establish detector-seed uncertainty or statistical significance.

The additional physical compute of the interrupted A attempt is recorded separately in [`manual_detector_recovery.json`](../detector_recovery_20261014_032214/manual_detector_recovery.json) and its source snapshots. The source files were not overwritten. The reported SSH-disconnection cause is not independently established.

## Review entry points

- [Formal report](../detector_recovery_20261014_032214/STAGE25_REPORT.md)
- [Full metrics, classwise deltas and gates](../detector_recovery_20261014_032214/stage25_tradeoff_summary.json)
- [Detector budget/checkpoint audit](../detector_recovery_20261014_032214/detector_compute_audit.json)
- [Unified final_eval access audit](../detector_recovery_20261014_032214/final_eval_access_audit.json)
- [Original 77-file result artifact ledger](../detector_recovery_20261014_032214/stage25_result_artifact_manifest.json)
- [Read-only local review record](local_review.json)

Local post-execution review independently recomputed the gates from the seven metric records; checked all 97,200 primary draw records against the frozen role/augmentation schedule; and verified 2,463 local file hashes with no missing files or mismatches. It also checked the historical comparator metric authorities and all four comparator checkpoints. The 42 P1 tests and 27 manual-recovery tests passed again before publication (stdlib/fixture/contract checks, not model forwards). No local model was loaded, no local GPU forward was performed, and no training, generation or new experiment was started.

## What is published to Git

This publication contains the corresponding source/configuration/tests/command documents, exact result JSON/Markdown, checkpoint metadata, prediction JSON, source snapshots and necessary text logs (including per-draw JSONL). The original frozen artifacts are not rewritten to make the negative result pass.

**Model weights, generated/real raster images, dataset labels/caches, and temporary training/final_eval input directories are not uploaded.** Their hashes and role/checkpoint manifests remain in the audit records; keep the original assets on the server for byte-level review. A metadata-only clone is not a self-contained training dataset/model package. No Stage26 or other experiment is authorized or automatically started by this publication.
