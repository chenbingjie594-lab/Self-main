# BootstrapGuard DeepPCB Feasibility Audit

Status: **DEEPPCB_NEEDS_CANDIDATE_GENERATION**

Official data: train=1000 images/6873 instances; val=500 images/3140 instances; classes=6.

Candidate pool: **CANDIDATE_POOL_MISSING**.
Recommended budget: {'open': 100, 'short': 100, 'mousebite': 100, 'spur': 100, 'spurious_copper': 100, 'pinhole': 100} (total 600 per synthetic arm).

MSDF-v3 code-flow compatibility: **supported after full call-chain review**. The raw automatic check reported a false negative because it inspected only one adapter file; `inference.py` and the inpainting pipeline provide class-specific reference image/mask and pass reference pixels plus the target mask. Checkpoint availability was not assumed.
Leakage audit: NONE_DETECTED_IN_REAL_SPLIT.

Scale and morphology are directly transferable using DeepPCB-specific train tertiles. Contrast is computable, but near-zero context variance creates extreme normalized values in some classes; its train-only denominator-floor policy must be audited before candidate bins are frozen.

No synthetic data was generated and no detector was trained.
