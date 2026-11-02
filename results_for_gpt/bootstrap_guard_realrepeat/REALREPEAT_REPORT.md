# Plastic_Bomo RealRepeat control

## Main result (three seeds, mean ± sample standard deviation)

| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---:|---:|---:|---:|
| Real-only | 0.8742 ± 0.0268 | 0.8444 ± 0.0393 | 0.8379 ± 0.0546 | 0.4065 ± 0.0260 |
| RealRepeat | 0.8753 ± 0.0533 | 0.8831 ± 0.0443 | 0.8739 ± 0.0439 | 0.4182 ± 0.0091 |
| Random | 0.8923 ± 0.0096 | 0.8291 ± 0.0365 | 0.8529 ± 0.0138 | 0.4244 ± 0.0114 |
| DWBG | 0.8877 ± 0.0249 | 0.8832 ± 0.0389 | 0.8901 ± 0.0335 | 0.4097 ± 0.0078 |
| BootstrapGuard | 0.9041 ± 0.0098 | 0.8740 ± 0.0392 | 0.8746 ± 0.0160 | 0.4406 ± 0.0012 |

## Why

RealRepeat isolates whether the synthetic-data gain can be explained by 80 additional real-image exposures per epoch. Every epoch contains all 138 frozen real images plus 80 deterministic replacement draws; no synthetic image or altered label is used.

## Result

RealRepeat − Real-only mAP50-95: +0.0117. BootstrapGuard − RealRepeat: +0.0224 (+5.35%), with BootstrapGuard winning 3/3 paired seeds. Mean class AP50-95 (RealRepeat → BootstrapGuard): Flash 0.3973 → 0.4186; Black 0.4392 → 0.4625.

## Conclusion

The numerical direction above is the result; no favorable outcome was assumed. This is an exact per-epoch exposure match, not a strict total-compute match, because Early Stopping determines cumulative epochs and historical exact optimizer-step counters were not retained. See `compute_match_audit.json` for the minimal strict-compute follow-up; it was not run.
