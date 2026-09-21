# DeepPCB Stage 6H fixed-150-epoch novelty confirmation

Status: FIXED_BUDGET_BATCH_MATCHED_STEP_COUNT_UNVERIFIED. Batch-matched result: NOVELTY_UTILITY_FIXED_BUDGET_NOT_SUPPORTED.
All six runs completed 150 epochs and 6450 audited batches; three reused runs lack direct optimizer-step counts.
Primary: independently validated epoch-150 last.pt. Stage6G best.pt is secondary only.

| seed | inner last mAP50-95 | novel last mAP50-95 | delta |
|---:|---:|---:|---:|
| 42 | 0.716877 | 0.711162 | -0.005715 |
| 3407 | 0.710540 | 0.711769 | +0.001229 |
| 2026 | 0.717399 | 0.718395 | +0.000996 |

Inner mean±std: 0.714939±0.003818.
Novel mean±std: 0.713775±0.004012.
Paired mean±std: -0.001163±0.003943; wins 2/3.

Per-class AP50-95 delta / wins:
- open: -0.001792; 1/3
- short: -0.001732; 2/3
- mousebite: +0.002431; 2/3
- spur: +0.001478; 1/3
- spurious_copper: -0.000401; 2/3
- pinhole: -0.006963; 2/3
