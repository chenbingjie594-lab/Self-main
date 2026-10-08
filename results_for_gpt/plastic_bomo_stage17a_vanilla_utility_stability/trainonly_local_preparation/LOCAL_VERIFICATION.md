# Independent train-only baseline verification

The local full-data preparation completed: unchanged 138 frozen parent images,
168 unchanged annotations (flash 88, black 80). Parent-name, exact file hash and
decoded RGB hash overlap with the current local validation directory were all
absent. Validation images were read for identity checks only, not their labels,
metrics or generator selection.

The 84 prior uniquely train-bound historical crops were compared by annotation
identity against this new pool: 84/84 JPEG image hashes and 84/84 PNG mask hashes
match exactly. This checks reuse of the recovered historical transform, not just
shape compatibility. All 168 instances are included, not a selected subset.

Preparation scope is LOCAL_TEST and cannot authorize training on the server.
The server command must independently pass the same frozen-source and isolation
checks, then match the original pretrained SD2 base files before training. No GPU
training was performed locally. Existing model checkpoints and historical status
files were not overwritten. Acquisition-group independence and annotation
accuracy remain unproven; this audit establishes exact frozen-parent isolation.

This is a new train-only source protocol, not recovery of the old baseline.
Generation, detector training, normal-conditioning isolation and final Stage17A
utility results are still pending.
