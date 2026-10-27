# MAGIC — Plastic_Bomo

Complete official MAGIC BBox-Mask adaptation experiment.

- Black/Flash were trained independently for 5,000 steps with seed 42.
- Official Prompt Perturbation, Spatially Adaptive Guidance and CAMA were enabled.
- Banks A/B/C contain 40 Black and 40 Flash slots per bank (240 total).
- `metrics.csv` contains bankwise, B+C pooled, and A+B+C pooled KID/FID plus Raw SD2/MSDF comparisons.
- `generation_diagnostics.csv` records background change, target-BBox coverage and structural failures.
- This is a rectangular BBox-Mask weak-label adaptation, not a segmentation-mask experiment.
- Model checkpoints and generated runtime images are intentionally excluded from GitHub.

Reproduction entry point: `tools/run_plastic_bomo_baseline_magic.py`.
