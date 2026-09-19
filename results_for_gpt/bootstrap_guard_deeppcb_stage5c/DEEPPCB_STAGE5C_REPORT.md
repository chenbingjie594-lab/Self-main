# DeepPCB Stage 5C — Clean No-Mosaic Exposure Pilot

Status: EXPOSURE_CAUSAL_SIGNAL_SUPPORTED. Clean single-seed exposure signal only; not multi-seed confirmation.
Mosaic calls: 0. Additional image reads: 0 across all completed epochs.

| Arm | mAP50-95 | Recall | Best epoch | Stop epoch |
|---|---:|---:|---:|---:|
| real_only | 0.711847 | 0.886925 | 127 | 150 |
| uniform | 0.687554 | 0.881694 | 90 | 130 |
| balanced | 0.722119 | 0.892671 | 95 | 135 |

Balanced - Uniform ΔmAP50-95: +0.034565.
Balanced - Real-only ΔmAP50-95: +0.010272.

| Class | Uniform AP50-95 | Balanced AP50-95 | Delta |
|---|---:|---:|---:|
| open | 0.600858 | 0.620133 | +0.019275 |
| short | 0.495529 | 0.522842 | +0.027314 |
| mousebite | 0.710648 | 0.735554 | +0.024906 |
| spur | 0.643656 | 0.677653 | +0.033997 |
| spurious_copper | 0.858198 | 0.909232 | +0.051034 |
| pinhole | 0.816437 | 0.867300 | +0.050862 |

Stage5B comparison is descriptive only; different training trajectories do not identify a Mosaic effect.
No selector, synthetic pool, ratio, validation split or additional seed was changed.
