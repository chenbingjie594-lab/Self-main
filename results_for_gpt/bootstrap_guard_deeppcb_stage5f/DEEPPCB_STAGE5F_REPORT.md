# DeepPCB Stage 5F — Compute-Matched Real-Only Control

Status: **SYNTHETIC_CONTENT_GAIN_NOT_SUPPORTED**. Seeds: 42, 3407, 2026.
RealRepeat340 uses only 100 frozen real images, sampled 340 times per epoch with replacement.
All synthetic arms were reused, not retrained.

| Method | s42 | s3407 | s2026 | mean ± sample std mAP50-95 |
|---|---:|---:|---:|---:|
| real_only | 0.711847 | 0.721416 | 0.712280 | 0.715181 ± 0.005404 |
| real_repeat340 | 0.728628 | 0.733876 | 0.738143 | 0.733549 ± 0.004766 |
| random | 0.737334 | 0.726829 | 0.731637 | 0.731933 ± 0.005259 |
| dq | 0.723775 | 0.719945 | 0.729517 | 0.724412 ± 0.004818 |
| bg | 0.722119 | 0.727546 | 0.733077 | 0.727580 ± 0.005479 |

| Paired comparison | mean delta | wins/3 |
|---|---:|---:|
| real_repeat340 - real_only | +0.018368 | 3/3 |
| random - real_repeat340 | -0.001616 | 1/3 |
| dq - real_repeat340 | -0.009137 | 0/3 |
| bg - real_repeat340 | -0.005969 | 0/3 |

| Class | RealRepeat mean AP50-95 | Random mean AP50-95 | Random wins/3 |
|---|---:|---:|---:|
| open | 0.648519 | 0.639539 | 1/3 |
| short | 0.528520 | 0.531651 | 2/3 |
| mousebite | 0.737834 | 0.741423 | 2/3 |
| spur | 0.698871 | 0.693446 | 1/3 |
| spurious_copper | 0.926295 | 0.923055 | 1/3 |
| pinhole | 0.861255 | 0.862487 | 2/3 |

Compute matching is per epoch: 340 draws, 43 batches of eight with a final batch of four.
Cumulative batches can differ because early stopping occurs at different epochs; optimizer steps were not directly instrumented.
No selector or validation data was modified.
