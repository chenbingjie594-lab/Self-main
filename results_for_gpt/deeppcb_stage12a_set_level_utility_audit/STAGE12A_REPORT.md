# DeepPCB Stage12A Historical Set-Level Marginal Utility Audit

## Decision

Status: **SET_LEVEL_UTILITY_STRUCTURE_SUPPORTED**<br>
Stage12B mechanism design authorized: **True**<br>
Stage12B automatically started: **false**

## Evidence boundary

- Primary comparable families: F2, F3, F4, F6 (4 independent families).
- F1: `FAMILY_INCOMPLETE`; excluded because its actual set and equal fixed-budget protocol cannot both be reconstructed.
- F5: secondary exploratory only; excluded from every primary gate.
- Historical extra records reconstructed: 1840 across 15 arms.
- Frozen feature/gradient extraction only; no generation, selection rerun, training, detector rerun, or new official-validation evaluation.
- All utility comparisons and descriptor ranks are within-family. Absolute mAP and raw descriptor levels were not pooled across families.

## Primary result

`marginal_real_coverage` is the only single descriptor passing the preregistered gate:

- pair-weighted winner consistency: 0.786 (required >= 0.70)
- pooled within-family rank Spearman rho: 0.788 (required |rho| >= 0.50)
- LOFO rho range: 0.723 to 0.895, positive in all four exclusions
- family-unit bootstrap WC 95% interval: [0.750, 1.000]

Support alignment did not pass because pair-weighted WC was 0.643, below 0.70, despite positive rho. Feature ESS, composition balance, and diagnostic gradient alignment did not pass.

## Multi-axis checks

- Best preregistered unweighted dominance relation WC: 0.800.
- Three-axis Pareto winner-on-frontier fraction: 0.750.
- Winner strictly dominated fraction: 0.250.

These are associative historical findings, not causal evidence. Gradient geometry remains diagnostic and is not part of the innovation claim.
