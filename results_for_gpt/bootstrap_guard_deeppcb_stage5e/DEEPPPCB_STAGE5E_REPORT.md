# DeepPCB Stage 5E — Matched No-Mosaic Multi-Seed Confirmation

Selector status: **BOOTSTRAP_SELECTOR_PARTIAL**.
Augmentation status: **SYNTHETIC_AUGMENTATION_CONFIRMED**.
Seeds: 42, 3407, 2026. Seed42 reused; eight new runs trained.
Mosaic 0, synthetic arms 260 real + 80 synthetic draws per epoch; Real-only traverses 100 images.

| Method | s42 | s3407 | s2026 | mean ± sample std mAP50-95 |
|---|---:|---:|---:|---:|
| real_only | 0.711847 | 0.721416 | 0.712280 | 0.715181 ± 0.005404 |
| random | 0.737334 | 0.726829 | 0.731637 | 0.731933 ± 0.005259 |
| dq | 0.723775 | 0.719945 | 0.729517 | 0.724412 ± 0.004818 |
| bg | 0.722119 | 0.727546 | 0.733077 | 0.727580 ± 0.005479 |

| Paired comparison | mean delta | wins/3 |
|---|---:|---:|
| random - real_only | +0.016752 | 3/3 |
| dq - real_only | +0.009231 | 2/3 |
| bg - real_only | +0.012399 | 3/3 |
| bg - random | -0.004353 | 2/3 |
| bg - dq | +0.003168 | 2/3 |

| Class | Random mean AP50-95 | BG mean AP50-95 | Random wins/3 |
|---|---:|---:|---:|
| open | 0.639539 | 0.630500 | 3/3 |
| short | 0.531651 | 0.524583 | 3/3 |
| mousebite | 0.741423 | 0.740699 | 2/3 |
| spur | 0.693446 | 0.684541 | 3/3 |
| spurious_copper | 0.923055 | 0.919856 | 2/3 |
| pinhole | 0.862487 | 0.865304 | 2/3 |
Random stably exceeds BG in 5/6 classes (positive mean and >=2 paired wins).

Only within-protocol paired seed comparisons support selector conclusions.
No selector, candidate pool, real/validation split or exposure ratio was modified.
