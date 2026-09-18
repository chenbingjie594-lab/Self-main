# DeepPCB Stage 3R Report

## Verdict

- Status: **DEEPPCB_STAGE3R_READY**
- Manifold assessment: **VALID_OOF_GTBOX_MANIFOLD**
- Scope: corrected RAW pair-level OOF reconstruction and frozen three-arm selection only. **No downstream detector was trained.**

The old Stage 3 manifold is invalid and must not be used. Its apparent single-class collapse came from pairing one converted per-instance crop with all labels from the RAW pair. Stage 3R rebuilt the carrier images and annotations in one coordinate system and reused the original fold membership and unchanged Stage 2 candidate pool.

## Integrity gates

- RAW low-data set: 100 tested pair images, 775 annotations; class counts are 130/129/129/129/129/129.
- Image identity: all 100 copied training images match their RAW sources by SHA-256.
- Annotation retention: 775 retained, 0 dropped, 0 extra and 0 class mismatches.
- Leakage: official validation use count is 0; original 34/33/33 pair-level folds were reused.
- Overfit sanity gate: PASS on 12 pair images; recall 1.000, mAP50 0.995 and mAP50-95 0.834. Every class has recall 1.000.
- Coordinate round trip: maximum absolute error 5.68e-14 pixels and minimum IoU 0.999999999999997.

## Corrected OOF detector

| Fold | Recall | mAP50 | mAP50-95 | Old recall | Old mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.782 | 0.924 | 0.550 | 0.093 | 0.055 |
| 1 | 0.886 | 0.937 | 0.652 | 0.118 | 0.069 |
| 2 | 0.811 | 0.897 | 0.517 | 0.125 | 0.068 |

All six classes now produce predictions. Class-aware/localization recall is: open 0.962/0.992, short 0.868/0.953, mousebite 0.791/0.984, spur 0.760/0.992, spurious_copper 0.884/0.977 and pinhole 0.922/1.000. This confirms that the previous five-class zero-recall pattern was caused by the malformed carrier/label construction, not an inherent detector collapse.

## Corrected manifold and frozen selectors

| Class | Valid candidates | Valid ratio | Unique target pairs |
|---|---:|---:|---:|
| open | 702/780 | 0.900 | 98 |
| short | 748/774 | 0.966 | 100 |
| mousebite | 761/774 | 0.983 | 99 |
| spur | 713/774 | 0.921 | 98 |
| spurious_copper | 749/774 | 0.968 | 100 |
| pinhole | 720/774 | 0.930 | 98 |

Random, Diversity+Quality and BootstrapGuard each select 40 candidates per class (240 total), use 40 unique target pairs per class and satisfy the maximum-eight-per-seed cap. BootstrapGuard has zero marginal or joint guardrail violations in all six classes. Its corrected selection overlaps the invalid old selection by only 15/240 items (Jaccard 0.032), so the old manifests cannot be treated as interchangeable.

## Decision

Stage 3R is ready as the corrected OOF GT-box manifold and selection input for the next downstream comparison. This result validates data integrity, OOF detector behavior, feature-coordinate mapping, candidate eligibility and selector fairness; it does not yet establish downstream performance gains.
