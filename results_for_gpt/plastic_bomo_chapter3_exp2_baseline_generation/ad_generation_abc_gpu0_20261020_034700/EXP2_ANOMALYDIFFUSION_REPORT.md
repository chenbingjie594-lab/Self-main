# Plastic_Bomo Chapter 3 Experiment 2 — AnomalyDiffusion generation

Status: `ANOMALYDIFFUSION_GENERATION_COMPLETE`.

This artifact records completion of the AnomalyDiffusion arm only. It does not claim that DefectFill, the full four-method comparison, visual metrics, or detector evaluation is complete.

## Execution

- Method label: `AnomalyDiffusion (explicit512 resolution adaptation)`
- Training budget: 2,000 optimizer steps, batch size 1, one GPU.
- Formal generation: 240 images from 80 frozen slots across three banks.
- Banks: A=42, B=3407, C=2026.
- Per bank: Flash 40, Black 40.
- Sampling: native DDIM, 200 steps, eta 1.0, adaptive mask disabled.
- Quality selection: disabled.
- Reroll count: 0.
- Started: `2026-10-20T03:41:08Z`
- Finished: `2026-10-20T05:47:50Z`
- Service result: success, exit code 0.

## Validation

- Generation status: `COMPLETE`, `240/240`.
- Formal validity: `PASS`.
- Exact `(bank, slot)` coverage: true.
- Generated PNG count: 240; YOLO label count: 240.
- A/B/C manifests: complete, 80 records each.
- Class counts in every bank: Flash 40, Black 40.
- Quality selection: false; reroll count: 0.
- Runtime log completion marker present; no `Traceback` or `FAILED`.

## Publication boundary

Generated images, checkpoints, runtime logs, training metrics, per-sample manifests, server paths, and data identities remain server-only. Git stores source, launchers, frozen protocol metadata, source repair hashes, and a path-free completion summary. The retained runtime log has SHA256 `3711ab0905aa51af14858c8168a981024a867949d3f614e33f92b312745f03bb` and size 3656780 bytes.

This run establishes successful fixed-slot generation. It does not establish comparative visual quality or downstream detector utility.
