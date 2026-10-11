# Stage20B — Raw SD2 baseline utility calibration

Status: `GENERATOR_BASELINE_HEADROOM_NOT_CONFIRMED`.

| Arm | mAP50-95 | Delta vs RR (pp) |
|---|---:|---:|
| RR | 0.359702 | — |
| RawA | 0.297100 | -6.260204 |
| RawB | 0.330818 | -2.888354 |
| RawC | 0.373483 | +1.378126 |

Mean delta: -2.590144 pp; sample SD: 3.827887 pp; strict wins: 1/3.
Gates A/B/C: False/False/False.
Flash mean AP delta: +0.035728 pp; Black: -5.216016 pp.
Generation-bank nuisance sample SD/range: 3.827887/7.638330 pp.

All three raw banks use the same frozen80 slots and two class final_step2000 pipelines. No quality scoring, selector, reroll or discarded raw output.
Four detectors share seed42, initialization, role order, augmentation seeds, 32400 draws and600 scheduled attempts. Actual AMP skips/success counts are preserved in detector_compute_audit.json; no compensatory training.
Primary checkpoints are the pinned native EMA representation of epoch150 last.pt. No best.pt, early stopping or validation-based checkpoint selection.
Only after all four checkpoint SHA freezes did one unified evaluation round open the48 V2 final_eval images and60 dataset-defined boxes.

## Limits

physical_source_complete=false: no KNOWN source-group leakage, not proof of absence of unknown physical relationships.
Labels are dataset-defined supervision, not universally human-verified physical truth. Rectangular bbox masks are weak geometry, not segmentation ground truth.
Bank variation is generation-RNG variation, not detector-seed confirmation. No comparisons to historical Stage14–18 absolute mAP.
BootstrapGuard=0; DeepPCB=0; TDCRG implementation=0. No automatic Stage20C/TDCRG, regardless of outcome.

## Review files

formal_protocol_binding.json binds immutable Stage20A/P0/R0 and runtime/source/model bytes.
generator_checkpoint_binding.json and detector_checkpoint_binding.json identify every primary pipeline/checkpoint.
formal_synthetic_manifest.json binds all240 PNG/labels; final_eval_access_audit.json records the evaluation barrier/population.
Full GPU update/draw logs and large weights remain under this run directory on the server.

## Explicit manual interruption recovery

Source stopped run: `formal_20261009_060552`. Original files/status were not overwritten.
User reported a possible accidental process kill; its external cause is not independently verified.
Generator checkpoints, all raw240 images and completed RR/A/B were reused after SHA verification.
Only C restarted from the original public initialization and seed42, epoch1 through150, with the unchanged frozen schedule. No epoch77 resume or added budget for the primary C checkpoint.
The interrupted C attempt is preserved separately, including its observed update/draw counters (potentially lower bounds). That extra physical work is not hidden or counted as matched primary exposure.
See manual_arm_C_recovery.json and manual_recovery/source_snapshot/. Inherited state-history entries refer to the source execution, not fresh generator/RR/A/B execution.
Immutable heavy files may share hardlinks with the source; do not edit either copy. No automatic next stage.
