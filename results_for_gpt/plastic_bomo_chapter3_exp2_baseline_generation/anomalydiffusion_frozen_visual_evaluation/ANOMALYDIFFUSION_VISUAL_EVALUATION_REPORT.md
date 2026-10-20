# Experiment 2 — AnomalyDiffusion frozen visual evaluation

Status: `COMPLETE`.

No training, image generation, selection, reroll, or detector evaluation ran. The evaluator reused the frozen Experiment 1 real Inception features and Raw SD2/MSDF results, then evaluated all 240 AnomalyDiffusion images under the same protocol.

## Defect-region primary table

| Method | FID (bank mean ± SD) | KID (bank mean ± SD, unscaled) | LPIPS (slot mean ± SD) |
| --- | ---: | ---: | ---: |
| Raw SD2 | 172.16 ± 2.27332 | 0.0342579 ± 0.00286449 | 0.0983397 ± 0.0515454 |
| MSDF | 188.548 ± 4.92193 | 0.046853 ± 0.0020467 | 0.115825 ± 0.0706508 |
| AnomalyDiffusion | 305.032 ± 0.117458 | 0.182471 ± 0.00117126 | 0.0675771 ± 0.0457217 |

Primary analysis is defect-region KID. FID is auxiliary because the reference sample is small. LPIPS measures A/B/C appearance diversity within each frozen slot and is not a realism score.

## Class-wise defect-region results

| Method | Flash FID | Flash KID | Flash LPIPS | Black FID | Black KID | Black LPIPS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw SD2 | 195.462 ± 3.75103 | 0.0624838 ± 0.00692506 | 0.11303 ± 0.0537968 | 204.714 ± 4.55456 | 0.0452116 ± 0.0038828 | 0.0836497 ± 0.0452241 |
| MSDF | 183.059 ± 5.10518 | 0.0624083 ± 0.00453255 | 0.0899103 ± 0.0442879 | 257.124 ± 10.9142 | 0.099039 ± 0.00449029 | 0.14174 ± 0.0822924 |
| AnomalyDiffusion | 294.712 ± 1.92562 | 0.189921 ± 0.00175353 | 0.0715234 ± 0.0509246 | 386.899 ± 2.97686 | 0.298508 ± 0.00529354 | 0.0636308 ± 0.0401158 |

FID/KID were computed independently for banks A/B/C, with 80 generated images per bank and 40 per class. SD uses ddof=1 across the three banks. KID uses the same fixed seed 42, 100 subsets, and subset size min(50, n_real, n_generated) as Experiment 1.

The ROI is the original normalized defect box expanded by 25% on every side, then floor/ceil rounded and clipped to the image. LPIPS AlexNet v0.1 compares A-B, A-C, and B-C crops for each slot at 256×256 and summarizes the 80 per-slot means.
