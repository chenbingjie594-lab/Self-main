# Stage17A independent train-only baseline

New 88 flash / 80 black instance crops from the unchanged frozen 138-image train set. This is a changed source protocol, NOT historical split70 baseline reproduction. Historical crop/mask functions are hash-pinned; all same-class ellipses on a parent are unioned before cropping.

Validation image bytes/pixels were accessed solely for exact parent isolation, never val annotations, selection, task loss or metrics. No proof of unknown acquisition-group independence or annotation accuracy is claimed. Supplemental Black annotation-review limitations remain.

No old checkpoint or labels changed. Generation remains blocked pending new-checkpoint/runtime, conditioning-normal lineage, slot/bbox transform and paired-RNG audits.
