# TDCRG — future specification only

Current authorization: **FALSE**. No implementation, teacher training, generator
training, sampling or ablation belongs to Stage20A. All Stage20B baseline gates
must pass before a separate research decision authorizes development.

## Isolation and supervision

Use only frozen V2 TRAIN source groups. A donor's OOF detector teacher must never
have trained on that donor's group, including aliases, siblings and derivatives.
Teachers are used only during generator training. Generator inference is
detector-free. No official/final evaluation input may select targets, tune loss
weights, gates or generator architecture. Labels are dataset-defined supervision,
not proof of exhaustive physical defects or universally human-verified masks.

## Distribution calibration

Extract real defect ROI features using train-only OOF teachers. Estimate each
class's mean, covariance and covariance effective rank. Candidate distribution
losses align generated and real class means and covariances, not each generated
sample to a nearest real instance. Feature layer, normalization, estimator,
numerical safeguards and loss weights require a later train-only frozen protocol;
this document does not assign untested values.

## Diversity preservation

Monitor within-class pairwise feature distance, nearest-neighbor distance and
covariance effective rank separately. Increased feature correspondence does not
justify mode collapse. Do not combine these into a weighted quality score or use
them for post-generation Top-K. A future diversity-preservation term and its
acceptance limits must be specified before experiments, using TRAIN evidence only.

## Unified class/timestep bounded residual

A shared learned model may condition gates on class and diffusion timestep:
`F'_l = F_l + g_l(c,t) * R_l`. An explicit residual norm bound or regularization
is mandatory. No manual rule such as strong Flash / weak Black. Bound, gate
parameterization and insertion locations remain future design decisions, not
implemented in Stage20A.

## Planned cumulative ablation sequence (not executed)

1. Frozen raw baseline.
2. Baseline + distribution mean alignment.
3. + mean/covariance alignment.
4. + diversity preservation.
5. + unified class-adaptive residual gate.

All variants must share frozen slots, generator initialization, source groups,
RNG protocol, exposure and detector evaluation. No quality-based filtering,
best-of-N, nearest-real copying or validation-based tuning.

## Independent innovations

BootstrapGuard is independent Innovation 2. It is forbidden during validation of
Innovation 1; combinations are allowed only after TDCRG is completely frozen.
DeepPCB is a future external generalization dataset only after Plastic_Bomo
Innovation 1 is frozen. Neither is modified or run at Stage20A.
