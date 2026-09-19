# DeepPCB Stage 5D — Matched Balanced-Exposure Selector Pilot

Status: **BALANCED_SELECTOR_NO_ADVANTAGE**. Seed42 only; no multi-seed confirmation.
Stage5C Real-only and BG checkpoints were reused, not retrained.
Random and DQ used exactly 260 real + 80 synthetic draws per epoch with Mosaic off.

| Arm | mAP50-95 | Recall | Best epoch | Stop epoch |
|---|---:|---:|---:|---:|
| real_only | 0.711847 | 0.886925 | 127 | 150 |
| random | 0.737334 | 0.891220 | 130 | 150 |
| dq | 0.723775 | 0.883173 | 52 | 92 |
| bg | 0.722119 | 0.892671 | 95 | 135 |

BG - DQ mAP50-95: -0.001656.
BG - Random mAP50-95: -0.015215.

| Class | Random AP50-95 | DQ AP50-95 | BG AP50-95 |
|---|---:|---:|---:|
| open | 0.642924 | 0.630825 | 0.620133 |
| short | 0.531721 | 0.511436 | 0.522842 |
| mousebite | 0.752643 | 0.737741 | 0.735554 |
| spur | 0.702974 | 0.671835 | 0.677653 |
| spurious_copper | 0.921321 | 0.918015 | 0.909232 |
| pinhole | 0.872422 | 0.872799 | 0.867300 |

Stage4M used Mosaic ON and uniform image exposure; Stage5D uses Mosaic OFF and balanced exposure.
Cross-protocol absolute differences do not isolate a selector effect.
Do not modify selectors or automatically launch more seeds based on this result.
