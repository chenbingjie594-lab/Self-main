# DeepPCB Stage 3 Report

## Decision

Status: **`DEEPPCB_SELECTION_READY`**

The three selection arms are complete and fairness checks pass. Candidate-pool SHA256: `1b2eb1abaeb886b00e49ea64cf322e075c88105b80fe67bc4e69200b6b851a14`.

## OOF probe audit

| Fold | Holdout pairs | Recall | mAP50-95 |
|---|---:|---:|---:|
| 0 | 34 | 0.0927 | 0.0547 |
| 1 | 33 | 0.1179 | 0.0691 |
| 2 | 33 | 0.1251 | 0.0682 |

All banks are holdout-only and contain at least 40 GT-box features per fold/class (minimum 40, maximum 48). Official validation use is zero. Probe quality is weak: only `open` has substantial recall; the other five classes have zero measured recall. The manifold therefore serves only as the predeclared GT-box feature eligibility floor, not as evidence of reliable six-class detection.

## Manifold eligibility

| Class | Valid / total | Valid ratio | Unique target pairs |
|---|---:|---:|---:|
| open | 750 / 780 | 96.15% | 100 |
| short | 744 / 774 | 96.12% | 98 |
| mousebite | 759 / 774 | 98.06% | 100 |
| spur | 762 / 774 | 98.45% | 99 |
| spurious_copper | 749 / 774 | 96.77% | 98 |
| pinhole | 756 / 774 | 97.67% | 99 |

Every class exceeds the frozen requirements of 40 valid candidates and 40 unique target pairs.

## Three selection arms

RandomMatched, DiversityQuality, and BootstrapGuard each contain exactly 240 candidates: 40 per class, 40 unique target pairs per class, and no generation seed contributes more than 8 candidates per class. All use the same frozen pool and `manifold_valid == true` eligibility.

Average of the six class-level manifold score summaries:

| Arm | Mean score | Mean class median |
|---|---:|---:|
| RandomMatched | 0.6829 | 0.6960 |
| DiversityQuality | 0.8106 | 0.8051 |
| BootstrapGuard | 0.6817 | 0.7049 |

DQ achieves the expected higher manifold quality. BG preserves frozen real-distribution plausibility and uses manifold quality only as a tie-break.

## Frozen guardrail violations

| Arm | Marginal + joint violations |
|---|---:|
| RandomMatched | 12 |
| DiversityQuality | 15 |
| BootstrapGuard | 0 |

The high-contrast drift is clearest for `spurious_copper` and `pinhole`: DQ selects 21/40 and 20/40 high-contrast samples respectively, while BG selects 14/40 for both. BG keeps all marginal and joint statistics within the frozen Stage 0.5 q05–q95 intervals.

## DQ versus BG overlap

| Class | Count | Jaccard |
|---|---:|---:|
| open | 0 | 0.0000 |
| short | 3 | 0.0390 |
| mousebite | 5 | 0.0667 |
| spur | 3 | 0.0390 |
| spurious_copper | 1 | 0.0127 |
| pinhole | 0 | 0.0000 |
| **Overall** | **12** | **0.0256** |

## Fairness and scope

- Pair-level 3-fold OOF split: PASS; holdout union is all 100 frozen pairs and intersections are empty.
- Class mapping consistency: PASS.
- Same candidate pool and eligibility: PASS.
- Class count, unique target-pair rule, and seed cap: PASS.
- Official validation use: 0.
- BootstrapGuard marginal/joint feasibility: PASS.

No candidate regeneration, downstream dataset construction, or downstream detector training was performed.
