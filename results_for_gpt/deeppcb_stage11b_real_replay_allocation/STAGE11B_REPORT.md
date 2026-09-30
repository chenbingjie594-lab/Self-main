# DeepPCB Stage11B Real-Replay Allocation Pilot

Status: **GRADIENT_ALLOCATION_REAL_REPLAY_NOT_SUPPORTED**.

Primary checkpoint: epoch150 `last.pt`; detector seed: 42; architecture: YOLO11s.

## Frozen gates

- A_beat_uniform_by_0_5pp: **False**
- B_strictly_beat_feature: **True**
- C_strictly_beat_hardness: **True**
- D_no_broad_class_collapse: **False**

## Primary mAP50-95

- uniform: 0.736003
- feature: 0.717191
- hardness: 0.714273
- gradient: 0.722463

Gradient - Uniform: -1.354 pp.

Cross-architecture confirmation authorized: **False**.

All arms used the same frozen 150-epoch schedule, 57,000 draws, 7,200 batches, sequence hashes, and training arguments. Successful optimizer updates were 915-917 and are disclosed as an AMP/data-dependent variation, not hidden or relabeled as equal.

No checkpoint was retrained or re-evaluated during finalization. No synthetic generation or Stage11B-R was started.
