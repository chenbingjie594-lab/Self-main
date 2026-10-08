# Stage17A generator training / validation exact crop audit

Scope: LOCAL_EXACT_AUDIT. Exact sources: 27/27; unique validation parents: 25; per class: {'flash': 14, 'black': 13}.

Reconstruction used only previously recovered crop geometry and a fixed +/-1 center-rounding ambiguity. SHA256 equality of reconstructed JPEG bytes establishes identity, not perceptual similarity. Actual generator-training membership is bound to the rebuild source/mask hashes.

Validation pixels were read solely for isolation diagnosis; no validation labels, detector losses, metrics or scores were accessed. No slots were selected or removed using these results.

Do not run the current rebuilt checkpoint as a leakage-free Stage17A baseline. Dropping suspect generation slots cannot undo generator training exposure. A separately authorized, explicitly versioned train-only reconstruction/retraining protocol is required before clean downstream claims. Historical experiment statuses are not rewritten by this diagnostic.
