# DeepPCB Stage9G Downstream Failure Attribution

Status: **BOTH_LIMITATIONS_SUPPORTED**
Selection representativeness: **RANDOM80_REPRESENTATIVE**

FrozenAdapter variation significantly higher: **True**
FrozenAdapter manifold departure significantly larger: **True**
FrozenAdapter downstream utility lower than FineTuned-SD2: **True**
Repeated synthetic-content gain supported: **False**

## Frozen selection and representativeness

The exact Stage9F selection was retained: 80 unique source-seed keys (40 short and 40 pinhole), RNG 2026, with no metric-based reselection or regeneration. For all three generator arms, none of the seven audited selected80 means fell outside its stratified random-80 full-pool 95% interval. The Stage9F result is therefore not explained by an obvious random-selection shift.

## Variation overshoot evidence

Relative to FineTuned-SD2 across all 258 paired sources, FrozenAdapter increased:

- IC-LPIPS by 0.02850 (Wilcoxon p = 6.82e-44).
- Corrected feature variance by 8.20e-05 (p = 2.15e-43).
- Corrected pairwise feature distance by 0.11021 (p = 2.15e-43).

This extra variation was accompanied by greater task-manifold departure. The full-pool normalized pair-gap increase was 0.43949 (p = 9.10e-70); selected80 independently showed an increase of 0.44035 (p = 2.57e-09). On selected80, the fraction above historical corrected Stage3R real q95 rose from 21.25% for FineTuned-SD2 to 46.25% for FrozenAdapter. The fraction above 2x real q95 rose from 2.5% to 15.0%.

Within FrozenAdapter's full pool, corrected feature variation was positively associated with mean normalized manifold gap (Spearman rho = 0.485, p = 1.19e-16). This is descriptive association only: it does not show that diversity caused detector degradation.

## Detector attribution

FrozenAdapter versus FineTuned-SD2 lost 1.004 pp AP50-95 on short and 3.109 pp on pinhole, so pinhole was the larger target-class loss. The effect was not target-only: open, mousebite, spur, and spurious-copper AP50-95 also all declined, producing a detector-level `broad negative transfer` diagnostic.

Training dynamics do not support a simple late-overfitting explanation. FrozenAdapter was unstable in the middle of training and later recovered, while its last-50-epoch overall trend remained positive; the appropriate status is `no clear dynamics pattern`. Epoch150 `last.pt` remains the primary checkpoint.

## Synthetic-content ceiling

Two directly exposure-controlled stages independently failed to support synthetic-content gain: Stage5F random synthetic minus RealRepeat averaged -0.162 pp with 1/3 wins, and Stage9F FineTuned-SD2 minus RealRepeat80 was -0.019 pp overall and -0.617 pp on Target2. Stage6H and Stage7C provide supporting negative evidence under different protocols and were not numerically pooled.

Route: retire FrozenResidualAdapter as a paper innovation candidate; any future first innovation must restart from utility-constrained generation.

All correlations are descriptive associations. No detector or generator was trained, no image was regenerated, and no sample was reselected. Per-source generated-identity R2 correlation is not identifiable from the frozen artifacts and was not fabricated.
