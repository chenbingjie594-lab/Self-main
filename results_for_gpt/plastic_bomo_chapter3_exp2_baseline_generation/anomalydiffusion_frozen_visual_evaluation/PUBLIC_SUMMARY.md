# Experiment 2: frozen three-method visual evaluation

The frozen 240-image AnomalyDiffusion output was evaluated without retraining, regeneration, selection, reroll, or detector inference. The evaluation reused Experiment 1's real Inception features and exact FID/KID/LPIPS protocol.

| Method | ROI FID ↓ | ROI KID ↓ | ROI LPIPS diversity |
| --- | ---: | ---: | ---: |
| Raw SD2 | 172.160 ± 2.273 | 0.034258 ± 0.002864 | 0.098340 ± 0.051545 |
| MSDF | 188.548 ± 4.922 | 0.046853 ± 0.002047 | 0.115825 ± 0.070651 |
| AnomalyDiffusion | 305.032 ± 0.117 | 0.182471 ± 0.001171 | 0.067577 ± 0.045722 |

AnomalyDiffusion is worse than both frozen baselines on defect-region FID and the primary KID metric. The result holds separately for Flash and Black. Its lower LPIPS indicates less appearance diversity across A/B/C, not better realism.

- Flash: FID 294.712 ± 1.926, KID 0.189921 ± 0.001754, LPIPS 0.071523 ± 0.050925.
- Black: FID 386.899 ± 2.977, KID 0.298508 ± 0.005294, LPIPS 0.063631 ± 0.040116.

FID/KID use three independently evaluated banks with 80 images per bank and 40 per class. LPIPS uses 80 frozen slot means from A-B, A-C, and B-C ROI comparisons. The full protocol and unrounded values are in the accompanying report and status JSON.
