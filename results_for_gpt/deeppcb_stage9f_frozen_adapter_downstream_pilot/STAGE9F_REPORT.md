# DeepPCB Stage9F Fixed-Budget Downstream Utility Pilot

Status: **HIGH_FIDELITY_SD2_BETTER**.
Single detector seed42 diagnostic only; no significance or stability claim.
Primary checkpoint is epoch150 last.pt; best.pt is secondary.

The status means that FineTuned-SD2 outperformed FrozenAdapter on both preregistered Target2 and overall mAP50-95. It does **not** establish a synthetic augmentation gain: FineTuned-SD2 remained 0.019 pp below RealRepeat80 overall and 0.617 pp below it on Target2.

| arm | overall mAP50-95 | Target2 mAP50-95 | short AP | pinhole AP |
|---|---:|---:|---:|---:|
| real_only | 0.724266 | 0.687365 | 0.537384 | 0.837346 |
| real_repeat80 | 0.727186 | 0.697555 | 0.531816 | 0.863293 |
| pretrained_noadapt_syn80 | 0.706845 | 0.654149 | 0.494725 | 0.813574 |
| finetuned_sd2_syn80 | 0.726993 | 0.691381 | 0.536588 | 0.846174 |
| frozen_adapter_syn80 | 0.707805 | 0.670818 | 0.526550 | 0.815086 |

## Primary paired differences

- FrozenAdapter - FineTuned-SD2: -1.919 pp overall and -2.056 pp Target2.
- FineTuned-SD2 - RealRepeat80: -0.019 pp overall and -0.617 pp Target2.
- FrozenAdapter - RealRepeat80: -1.938 pp overall and -2.674 pp Target2.
- Pretrained-NoAdapt - RealRepeat80: -2.034 pp overall and -4.341 pp Target2.

The ordering supports the need for task adaptation relative to an unadapted pretrained generator, but the repaired frozen adapter's additional variation did not translate into downstream utility in this seed. RealRepeat80 remained the strongest primary arm.

## Protocol audit

- All five arms completed 150 epochs; primary evaluation used epoch150 `last.pt`.
- Real-only used 260 draws and 33 batches per epoch (4,950 batches total).
- The other four arms used 260 real plus 80 matched extra draws and 43 batches per epoch (6,450 batches total).
- The four 340-draw arms had identical source exposure frequencies.
- The three synthetic arms shared the same 80 frozen source-seed keys and labels.
- Selection was uniform random with RNG 2026 and did not use quality, feature, task, or validation metrics.

Best.pt reverses a primary comparison direction: **False**.
No selector, generator retraining, detector tuning, or validation-driven decision was used.
