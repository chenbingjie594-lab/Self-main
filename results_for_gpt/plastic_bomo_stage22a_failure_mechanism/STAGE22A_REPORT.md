# Stage22A — Synthetic Failure Mechanism Audit

Status: `SYNTHETIC_OPTIMIZATION_MECHANISM_NOT_IDENTIFIED`.

All 109 Stage21B internal_train real images and 61 matched frozen slots in each of A/B/C were probed at the same seed-42 public YOLO11s initialization, with deterministic no-augmentation letterbox preprocessing at 1536 and no optimizer step. The native box/cls/dfl loss and selected-layer gradients were measured on 292 forward/backward passes. Model parameters and buffers were restored/verified unchanged. Real class references use annotation membership; real direction controls use leave-one-out.

This is descriptive evidence, not a causal attribution. Differences in loss, gradient, conflict, or detector features do not by themselves distinguish pixel fidelity, annotation geometry, domain shift, or optimizer interaction. Three banks are insufficient for a correlation claim. Decision: No unique signal passed all predeclared A-E and annotation-count gates; frozen-initialization contrasts remain descriptive, not causal. Strongest unique candidate: none; no Stage22B or new method was started.

## Reproducibility

- Protocol: `diagnostic_protocol.json`; immutable roles and source hashes: `diagnostic_data_manifest.json`.
- Exact selected parameters and initialization binding: `model_parameter_binding.json`.
- Per-image loss, group norm, direction and sign conflict: respective audit JSON files.
- Per-slot A/B/C variation: `matched_triplet_audit.json`; ROI activation moments: `feature_distribution_audit.json`.
- Flash/Black summaries: `classwise_mechanism_audit.json`; decision limits: `mechanism_candidate_summary.json`.
- Official final_eval content/forward: 0; internal_dev: 0; generator/detector training: 0; optimizer steps: 0.

Limitations: the diagnostic uses a frozen initialization, no augmentation, one detector architecture and no intervention. The existing Stage21B utility ordering is historical context only; it was not used to select images, layers, thresholds, or metrics.

## Classwise diagnostic summary

| Class | Group | Total loss median | Selected-gradient L2 median | Same-class cosine median | Same-class conflict fraction |
|---|---|---:|---:|---:|---:|
| flash | real | 136.117 | 898.62 | 0.397541 | 0.0000 |
| flash | A | 137.876 | 973.488 | 0.292401 | 0.0000 |
| flash | B | 138.818 | 1002.79 | 0.288351 | 0.0000 |
| flash | C | 138.027 | 1018.16 | 0.286979 | 0.0000 |
| black | real | 135.896 | 875.637 | 0.283237 | 0.0000 |
| black | A | 138.22 | 921.021 | 0.312502 | 0.0000 |
| black | B | 139.454 | 948.939 | 0.302728 | 0.0000 |
| black | C | 138.296 | 920.446 | 0.310444 | 0.0000 |

Real rows use leave-one-out references. Each bank uses the same 61 slots; real target multiplicity is audited separately.

| Class | A/B/C within-slot total-loss range median | Gradient-norm range median | Same-class cosine range median |
|---|---:|---:|---:|
| flash | 4.49129 | 139.078 | 0.0247646 |
| black | 4.50423 | 113.78 | 0.0285504 |

Feature distributions and all raw per-image/per-slot records are in the linked JSON audits. No metric is a utility predictor.
