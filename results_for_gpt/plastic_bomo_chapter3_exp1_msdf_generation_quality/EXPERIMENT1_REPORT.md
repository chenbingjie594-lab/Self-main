# Experiment1 — MSDF visual quality baseline comparison

Status: `MSDF_VISUAL_METRICS_NOT_IMPROVED`.

Only frozen Raw SD2 and MSDF generation outputs were evaluated. No generator/detector training, sampling, selection, BootstrapGuard or DWBG ran.

## Defect-region primary table

| Method | FID (bank mean ± SD) | KID (bank mean ± SD, unscaled) | LPIPS (slot mean ± SD) |
| --- | ---: | ---: | ---: |
| Raw SD2 | 172.16 ± 2.27332 | 0.0342579 ± 0.00286449 | 0.0983397 ± 0.0515454 |
| MSDF | 188.548 ± 4.92193 | 0.046853 ± 0.0020467 | 0.115825 ± 0.0706508 |

Primary analysis: defect-region KID. FID is auxiliary because reference samples are small and FID is sample-size biased. KID negative unbiased estimates are retained, not clipped.

Crop: original normalized bbox expanded by25% of its width/height on each side, floor/ceil integer bounds and image-boundary clipping. The same rule is used for all methods and real reference boxes. No minimum-size crop enlargement or manual adjustment.

Reference: 48 unique V2 final_eval images, 60 original boxes; class boxes {'black': 27, 'flash': 33}. This frozen held-out visual reference set was historically observed and is not an untouched test set.

Full-image class-wise references include unique images containing the class; a multi-class image can enter both class subsets. ROI references include every original annotation. No class balancing or image deletion was introduced. Counts are saved in each bankwise record.

FID/KID are computed separately for A/B/C (80 images per bank,40 per class). SD across three banks uses ddof1 and is not detector-seed uncertainty or proof of statistical significance. KID uses100 fixed-seed subsets of min(50,n_real,n_generated); full subset coverage is audited. Within-bank subset SD (ddof0) is not the paper bank SD.

LPIPS AlexNet v0.1 compares A-B/A-C/B-C defect crops within the same slot. Each slot is the mean of three distances, then80 slot means (40 per class) are summarized with sample SD. Inputs are bilinear-resized256x256 RGB and normalized to[-1,1]. LPIPS measures appearance diversity, not realism or a guarantee of higher quality.

## Class-wise ROI directions

- flash: MSDF_VISUAL_METRICS_IMPROVED; MSDF−Raw FID=-12.4035, KID=-7.54569e-05, LPIPS=-0.0231194.
- black: MSDF_VISUAL_METRICS_NOT_IMPROVED; MSDF−Raw FID=52.4106, KID=0.0538274, LPIPS=0.0580904.

## Interpretation

The descriptive status uses only directions of Overall ROI bank-mean FID/KID, not a newly weighted quality score. Class-wise outcomes above must accompany any overall statement.
Measured appearance diversity increases (LPIPS MSDF−Raw=0.0174855). No arbitrary threshold is used to claim preservation or significance.

Metric implementations, actual package versions, source-file hashes and pretrained-weight hashes are frozen before metric forwards in metric_runtime.json. The standard library implementations perform all FID/KID calculations; no alternative distance or feature network was implemented.

No downstream mAP record was read to explain these results. No image is ranked, filtered or rerolled. No subsequent experiment is automatically authorized.

## Primary implementation sources

- https://torch-fidelity.readthedocs.io/en/v0.3.0/api.html
- https://github.com/toshas/torch-fidelity/blob/v0.3.0/torch_fidelity/metric_fid.py
- https://github.com/toshas/torch-fidelity/blob/v0.3.0/torch_fidelity/metric_kid.py
- https://github.com/toshas/torch-fidelity/blob/v0.4.0/torch_fidelity/metric_fid.py
- https://github.com/toshas/torch-fidelity/blob/v0.4.0/torch_fidelity/metric_kid.py
- https://github.com/richzhang/PerceptualSimilarity
