# Plastic_Bomo SeaS baseline

Status: `COMPLETE`

SeaS was reproduced with its official SD1.4 backbone and Separation-and-Sharing mechanism. The generation and RMP stages each completed 1600 optimizer updates (two jointly trained anomaly categories × 800), using batch 2 and gradient accumulation 2. No MSDF component or SD2 backbone was introduced.

Plastic_Bomo V2 supplies bounding boxes, not human pixel segmentations. This experiment is explicitly a `COARSE_BBOX_MASK_ADAPTER`, not the paper's pixel-GT protocol. Training used 97 Black and 75 Flash V2 TRAIN anomalies plus 60 normals; no test defect or test GT mask was used.

## Results

| Scope | Population | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Black | 0.14901477 | 248.4111 |
| A | Flash | 0.09392446 | 214.8472 |
| A | Overall | 0.07741907 | 185.0694 |
| B | Black | 0.13854123 | 249.5333 |
| B | Flash | 0.10199668 | 231.2705 |
| B | Overall | 0.06985237 | 189.2895 |
| C | Black | 0.17142524 | 267.4442 |
| C | Flash | 0.09972812 | 219.9453 |
| C | Overall | 0.09097272 | 201.5235 |
| B+C pooled | Black | 0.15259283 | 239.1884 |
| B+C pooled | Flash | 0.10163709 | 210.8963 |
| B+C pooled | Overall | 0.08124556 | 179.0618 |
| A+B+C pooled | Black | 0.15481329 | 229.6379 |
| A+B+C pooled | Flash | 0.09954417 | 202.5279 |
| A+B+C pooled | Overall | 0.07888011 | 170.3380 |

Pooled metrics were recomputed from concatenated Inception-2048 features and are not averages of Bank metrics. `comparison.csv` includes Raw SD2, MSDF Full, MSDF Max-Support, and SeaS. SeaS uses SD1.4 while MSDF uses SD2, so this is a method-level comparison rather than a same-backbone ablation.

## Generation and localization

All 240 RGB images and masks were retained without quality filtering, Best-of-N, or reroll. SeaS uses native generation and cannot control each frozen target BBox; no claim of position or latent alignment with SD2 is made. Mean predicted-mask/target-BBox IoU was 0.455595 and mean centroid shift was 2.5132 pixels; no generated mask was empty.

The inference compatibility change only moves the loaded RMP to the requested CUDA device and sets evaluation mode. It changes no weights, architecture, loss, or sampling mathematics. Images, checkpoints, detailed logs, and feature caches remain server-only.
