# DeepPCB Stage 8B morphology training ablation

Generator gate: **NOMORPH_REPAIR_SUPPORTED**.
Stage status: **GENERATOR_GATE_PASSED_AWAITING_SEED42_DETECTOR**.
No weighted score was used; all generator axes are reported independently.
LPIPS improved: True; edge improved: True; boundary improved: False; task feature no-collapse: True.
Class breadth: LPIPS 6/6; structural 5/6; short consistent improvement: True.

## Seed42 detector pilot

Primary checkpoint: epoch150 `last.pt`, with the frozen Stage7C training and validation protocol.

| Generator | mAP50-95 |
|---|---:|
| Vanilla SD2 | 0.720157 |
| Full MSDF-v3 | 0.722848 |
| MSDF-NoMorph | 0.714740 |

NoMorph minus Full MSDF: **-0.008107**. NoMorph minus SD2: **-0.005417**.

NoMorph improved short over Full by +0.005204 AP50-95, but remained -0.006261 below SD2. It regressed versus Full in five of six classes, including spurious copper (-0.022374), mousebite (-0.010643), and pinhole (-0.008599).

Detector gate: **NOMORPH_DETECTOR_SEED42_NOT_SUPPORTED**.

Final status: **MSDF_REPAIR_NOT_SUPPORTED**. Seeds 3407/2026 are not authorized, and no MSDF-v4 module is proposed.
