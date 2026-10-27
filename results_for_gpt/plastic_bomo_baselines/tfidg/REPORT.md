# Plastic_Bomo TF-IDG baseline

Status: `COMPLETE`

TF-IDG was evaluated as a training-free method. AnyDoor and DINOv2 remained frozen; no Plastic_Bomo model, MSDF module, detector, or selector was trained.
Reference and target masks are coarse frozen BBox masks. All 240 outputs were retained without Best-of-N, reroll, or quality filtering.

## Bankwise ROI metrics

| Bank | Cohort | KID ↓ | KID subset std | FID ↓ | Generated |
| --- | --- | ---: | ---: | ---: | ---: |
| A | Black | 0.1443038339 | 0.0110015593 | 244.209547 | 40 |
| A | Flash | 0.1771600985 | 0.0080713278 | 270.479583 | 40 |
| A | Overall | 0.1235892460 | 0.0086046658 | 229.564518 | 80 |
| B | Black | 0.1610814981 | 0.0122623519 | 258.046382 | 40 |
| B | Flash | 0.1567887137 | 0.0075350926 | 251.927312 | 40 |
| B | Overall | 0.1258132301 | 0.0094529647 | 230.739877 | 80 |
| C | Black | 0.1466083123 | 0.0116225596 | 249.013167 | 40 |
| C | Flash | 0.1558158896 | 0.0078251330 | 258.534781 | 40 |
| C | Overall | 0.1170513076 | 0.0086181249 | 227.987758 | 80 |

## Pooled ROI metrics (recomputed from concatenated features)

| Banks | Cohort | KID ↓ | KID subset std | FID ↓ | Generated |
| --- | --- | ---: | ---: | ---: | ---: |
| B+C | Black | 0.1526995969 | 0.0098812536 | 237.125711 | 80 |
| B+C | Flash | 0.1571156403 | 0.0069388347 | 242.766802 | 80 |
| B+C | Overall | 0.1236532859 | 0.0094822137 | 214.887661 | 160 |
| A+B+C | Black | 0.1497581044 | 0.0109949360 | 227.684108 | 120 |
| A+B+C | Flash | 0.1644234772 | 0.0084828481 | 244.292807 | 120 |
| A+B+C | Overall | 0.1239550342 | 0.0102041543 | 210.194676 | 240 |

## Diagnostics

- Mean changed-pixel fraction outside target BBox: `0.00049491`.
- Mean defect-change centroid shift: `1.2362` crop pixels.
- Mean resized-BBox reference-copy SSIM: `0.359608`.
- Background-pollution flags: `0/240`.

## Comparison and protocol

`comparison.csv` compares TF-IDG with frozen same-protocol Raw SD2, MSDF Full, and MSDF Max-Support metrics. Raw/MSDF are full-UNet fine-tuning routes; TF-IDG is training-free.
B+C and A+B+C metrics were recomputed from concatenated Inception-2048 features and are not averages of bank metrics.
KID uses 100 fixed-seed subsets (seed 42), subset size min(50, n_real, n_generated). Real references are the same 172 V2 TRAIN ROIs (Black 97, Flash 75).
Images, pretrained weights, and feature caches remain server-only.

## Runtime note

An initial formal run stopped after 5 images because a local 60% PyTorch VRAM cap rejected an allocation while physical VRAM remained free. The run resumed from the SHA-bound manifest after raising only that allocator cap to 95% and adding per-sample cache collection. No algorithm, model structure, inference parameter, reference, seed, image, or completed record was changed or rerolled.
