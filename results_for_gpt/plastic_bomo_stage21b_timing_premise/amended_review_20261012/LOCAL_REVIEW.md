# Stage21B A1 downloaded-result review

Review: `LOCAL_DOWNLOADED_STAGE21B_A1_REVIEW_PASS`. Experiment: `SYNTHETIC_EXPOSURE_TIMING_PREMISE_NOT_SUPPORTED`.

- Original entry audit and split bytes unchanged; only the four authorized conditioning-group exclusions, Flash32 / Black29.
- All 53 server metadata files, six downloaded last.pt files and 153000 draw-log records verified. No checkpoint loading/forward and no dev/final-eval image or label content opened by this review.
- Actual augmentation pairing, frozen initialization, optimizer groups and all 450-attempt learning-rate sequences match.
- Each arm: 150 epochs, 25500 draws, 450 scheduled attempts; actual successful updates 442–443, disclosed, not strictly equal and not compensated.
- Curriculum-minus-Uniform: A -2.5002, B -0.2377, C -4.4869 pp; mean -2.4082 pp; wins 0/3.
- Flash mean -1.0856 pp; Black mean -3.7309 pp.
- Across-bank sample SD (ddof=1): Uniform 3.0491, Curriculum 0.9395 pp.
- Gate A=False, B=False, C=False, D=True. Lower spread does not offset lower utility or Black-class degradation.

This is **DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR**, not an independent pipeline holdout: the frozen generator historically used 33 dev-group annotation inputs (Black23 / Flash10). The result rejects this one fixed late schedule under this protocol; it does not reject every possible curriculum or establish an official V2 final-eval outcome. File-access claims rely on pinned code and reported counters, not an OS sandbox.

No bound server file was rewritten. No generation, training, evaluation rerun, new selector, weighting, BootstrapGuard, TDCRG, DeepPCB or next stage was started. Stop after GitHub archival.
