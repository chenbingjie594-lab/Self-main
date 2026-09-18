# DeepPCB Stage 2 Report

## Decision

Status: **`DEEPPCB_POOL_READY`**

The frozen instance-level candidate pool contains all 4,650 expected valid candidates. No generation failure, duplicate image, invisible generation, invalid metadata, or validation-set leakage was observed. No candidate selection or detector training was performed.

## Registry and candidate counts

| Class | Real instances | Expected candidates | Valid candidates |
|---|---:|---:|---:|
| open | 130 | 780 | 780 |
| short | 129 | 774 | 774 |
| mousebite | 129 | 774 | 774 |
| spur | 129 | 774 | 774 |
| spurious_copper | 129 | 774 | 774 |
| pinhole | 129 | 774 | 774 |
| **Total** | **775** | **4,650** | **4,650** |

- Unique target-instance coverage: 100%.
- Generation/audit failures: 0.
- Each candidate contains exactly one target label and uses an independent bbox-derived instance mask.
- Seeds 42, 3407, 2026, 7859, 12011, and 15683 each cover every target instance.

## Reference policy

The deterministic SHA256 policy produced 3,209 self references and 1,441 same-class cross references: **69.01% self / 30.99% cross** globally.

| Class | Self | Cross | Unique targets | Unique references |
|---|---:|---:|---:|---:|
| open | 67.18% | 32.82% | 130 | 130 |
| short | 69.12% | 30.88% | 129 | 129 |
| mousebite | 72.09% | 27.91% | 129 | 129 |
| spur | 70.03% | 29.97% | 129 | 129 |
| spurious_copper | 66.54% | 33.46% | 129 | 129 |
| pinhole | 69.12% | 30.88% | 129 | 129 |

## Artifact diagnostics

- Duplicate candidate IDs: 0.
- Duplicate image hashes: 0 groups / 0 images beyond the first.
- Targets producing identical images across seeds: 0.
- `NO_VISIBLE_GENERATION`: 0.
- Outside-mask MAE median/q95: 2.4761 / 4.5351.
- Near-mask-ring MAE median/q95: 3.8363 / 9.9111.
- Inside-mask MAE median/q95: 33.7896 / 62.7774.

These MAE values are diagnostics only and were not used to filter the pool.

## Frozen attribute-bin distributions

Counts are bin 0 / bin 1 / bin 2. Stage 0.5 train-only thresholds and `tau_sigma = 58.7430814883903` were reused exactly; no threshold was estimated from this synthetic pool.

| Class | Scale bins | Contrast bins | Morphology bins |
|---|---|---|---|
| open | 258 / 258 / 264 | 265 / 210 / 305 | 252 / 264 / 264 |
| short | 258 / 258 / 258 | 252 / 276 / 246 | 258 / 258 / 258 |
| mousebite | 258 / 258 / 258 | 197 / 287 / 290 | 258 / 258 / 258 |
| spur | 252 / 264 / 258 | 232 / 236 / 306 | 258 / 258 / 258 |
| spurious_copper | 258 / 258 / 258 | 239 / 202 / 333 | 258 / 258 / 258 |
| pinhole | 240 / 270 / 264 | 199 / 213 / 362 | 258 / 252 / 264 |

## Provenance and leakage

- Target, reference, and paired-normal background provenance: frozen 100-pair low-data train only.
- Official validation use count: 0.
- Leakage audit: PASS.

Stage 2 is complete. The raw pool is frozen for a later selection stage; this stage performed no Random, DiversityQuality, BootstrapGuard, detector-manifold, FID, KID, LPIPS, or CLIP selection.
