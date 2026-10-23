# Experiment 10 Phase A — MSDF Auxiliary Gradient Conflict Analysis

Twelve Black TRAIN samples from twelve distinct source groups were frozen before diagnostics: 8 self-reference and 4 cross-reference. No optimizer step was called.

| Parameter group | Loss | Historical weight | Mean norm | Median norm | Mean cosine vs denoising | Median cosine | Negative fraction | Weighted ratio | Valid | Invalid |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| UNet | Denoising | 1.00 | 0.031812 | 0.0240307 | 1 | 1 | 0 | 1 | 12 | 0 |
| UNet | Region | 0.75 | 15.2136 | 10.1545 | 0.231714 | 0.235201 | 0 | 304.19 | 12 | 0 |
| UNet | Reconstruction | 0.50 | 12.3354 | 13.1182 | 0.202324 | 0.168676 | 0 | 265.769 | 12 | 0 |
| UNet | Structure | 0.25 | 2.60666 | 0.0675094 | 0.121093 | 0.113962 | 0.181818 | 15.6178 | 11 | 1 |
| UNet | Background | 0.20 | 1.48685 | 1.51 | 0.25204 | 0.180774 | 0 | 26.1998 | 12 | 0 |
| UNet | Multiscale | 0.25 | 4.54751 | 4.75819 | 0.220256 | 0.201126 | 0 | 47.0593 | 12 | 0 |
| UNet | Support | 0.25 | NA | NA | NA | NA | NA | NA | 0 | 12 |
| MSDF | Denoising | 1.00 | 0.00067516 | 0.000661776 | 1 | 1 | 0 | 1 | 12 | 0 |
| MSDF | Region | 0.75 | 0.726622 | 0.523608 | 0.549249 | 0.542348 | 0 | 759.076 | 12 | 0 |
| MSDF | Reconstruction | 0.50 | 0.620928 | 0.485275 | 0.462101 | 0.444979 | 0 | 731.862 | 12 | 0 |
| MSDF | Structure | 0.25 | 0.0921903 | 0.00419421 | 0.143754 | 0.0659952 | 0.181818 | 30.1726 | 11 | 1 |
| MSDF | Background | 0.20 | 0.00052526 | 0.000400472 | 0.0831102 | 0.110432 | 0.416667 | 0.268624 | 12 | 0 |
| MSDF | Multiscale | 0.25 | 0.216703 | 0.170157 | 0.486986 | 0.459434 | 0 | 126.983 | 12 | 0 |
| MSDF | Support | 0.25 | 1.32267 | 1.27152 | -0.0964192 | -0.0739801 | 0.625 | 846.893 | 8 | 4 |
| LR_SCALED_COMBINED | Denoising | 1.00 | 7.55137e-08 | 6.97152e-08 | 1 | 1 | 0 | 1 | 12 | 0 |
| LR_SCALED_COMBINED | Region | 0.75 | 7.44888e-05 | 5.34535e-05 | 0.484926 | 0.472079 | 0 | 638.339 | 12 | 0 |
| LR_SCALED_COMBINED | Reconstruction | 0.50 | 6.35107e-05 | 5.18409e-05 | 0.410566 | 0.391017 | 0 | 578.501 | 12 | 0 |
| LR_SCALED_COMBINED | Structure | 0.25 | 9.66005e-06 | 4.24819e-07 | 0.138277 | 0.0665569 | 0.181818 | 28.3305 | 11 | 1 |
| LR_SCALED_COMBINED | Background | 0.20 | 1.488e-06 | 1.51039e-06 | 0.10431 | 0.075453 | 0 | 6.69426 | 12 | 0 |
| LR_SCALED_COMBINED | Multiscale | 0.25 | 2.22339e-05 | 1.84462e-05 | 0.434379 | 0.435491 | 0 | 100.161 | 12 | 0 |
| LR_SCALED_COMBINED | Support | 0.25 | 0.000132267 | 0.000127152 | -0.077506 | -0.0685453 | 0.625 | 586.358 | 8 | 4 |

## Findings

- The most negative mean direction was Support on MSDF (mean cosine -0.096419; negative fraction 0.625).
- The largest mean historical-weighted gradient ratio was Support on MSDF (846.892643).
- Background: UNet mean cosine=0.252040, negative fraction=0.000, weighted ratio=26.199776; MSDF mean cosine=0.083110, negative fraction=0.417, weighted ratio=0.268624.
- Support is marked NOT_APPLICABLE for the four cross-reference samples, matching the original morphology_support_loss implementation.
- At least one auxiliary/group combination had negative cosine on at least half of valid samples: TRUE.

Pre-frozen severe Background conflict stop rule triggered: **FALSE**.

Cosine conflict is a local optimization observation and does not by itself establish a causal effect on KID or FID.
