# DeepPCB Stage 7C matched downstream benchmark

Status: **MSDF_DOWNSTREAM_NOT_SUPPORTED**.
Primary checkpoint: epoch150 last.pt; best.pt is secondary only.
Both arms consumed 150 epochs and 6450 forward/backward batches per seed.
AMP successful optimizer updates equal in all seed pairs: **False** (raw counts retained in training_budget_audit.json).

| seed | SD2 mAP50-95 | MSDF mAP50-95 | delta |
|---:|---:|---:|---:|
| 42 | 0.720157 | 0.722848 | +0.002690 |
| 3407 | 0.721823 | 0.712551 | -0.009271 |
| 2026 | 0.724283 | 0.718204 | -0.006080 |

SD2 mean +/- std: 0.722088 +/- 0.002076.
MSDF mean +/- std: 0.717868 +/- 0.005156.
Paired delta mean +/- std: -0.004220 +/- 0.006194; wins 1/3.
No generator, selector, generation seed, or validation protocol was changed.
