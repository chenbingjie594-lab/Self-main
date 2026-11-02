# Raw SD2 Bank A downstream detection

| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |
|---|---:|---:|---:|---:|
| Real-only | 0.8742 ± 0.0268 | 0.8444 ± 0.0393 | 0.8379 ± 0.0546 | 0.4065 ± 0.0260 |
| RealRepeat | 0.8753 ± 0.0533 | 0.8831 ± 0.0443 | 0.8739 ± 0.0439 | 0.4182 ± 0.0091 |
| Random | 0.8923 ± 0.0096 | 0.8291 ± 0.0365 | 0.8529 ± 0.0138 | 0.4244 ± 0.0114 |
| BootstrapGuard | 0.9041 ± 0.0098 | 0.8740 ± 0.0392 | 0.8746 ± 0.0160 | 0.4406 ± 0.0012 |
| Raw SD2 Bank A | 0.8749 ± 0.0421 | 0.8430 ± 0.0304 | 0.8610 ± 0.0196 | 0.4061 ± 0.0108 |

## Conclusion

Raw SD2 − Real-only mAP50-95: -0.0004. Raw SD2 − RealRepeat: -0.0121. These are observed three-seed results under the frozen protocol; no hyperparameter or sample selection was changed.
