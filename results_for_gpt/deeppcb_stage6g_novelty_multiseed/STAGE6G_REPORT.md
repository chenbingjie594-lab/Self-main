# DeepPCB Stage 6G multi-seed paired novelty utility

Status: **NOVELTY_UTILITY_MULTISEED_SUPPORTED**.
Only four new runs: inner/novel at detector seeds 3407 and 2026; seed42 reused.
Frozen candidates and same within-seed paired schedules; generation-seed distribution remains a fixed residual nuisance.

| Detector seed | Inner mAP50-95 | Novel mAP50-95 | Novel - Inner | Inner best/stop | Novel best/stop |
|---:|---:|---:|---:|---:|---:|
| 42 | 0.707056 | 0.726288 | +0.019231 | 43/83 | 105/145 |
| 3407 | 0.724647 | 0.723758 | -0.000889 | 81/121 | 127/150 |
| 2026 | 0.721916 | 0.723666 | +0.001750 | 144/150 | 142/150 |

Inner mean±sample std: 0.717873 ± 0.009467.
Novel mean±sample std: 0.724570 ± 0.001488.
Paired mean delta±sample std: +0.006697 ± 0.010935; wins 2/3.
The same maximum budget and early-stopping rule were used, but cumulative optimizer batches may differ; this is not a fixed-step effect.
No new candidate, selector, or generator change.
