# DeepPCB Stage 5A Diagnosis

Status: **IMAGE_LEVEL_SHIFT_STRONGLY_SUPPORTED**

Offline descriptive diagnosis only; no candidate selection, generation, training, or validation was performed.

## Key image-level contrast

| Population | Images | GT | Objects/image | Classes/image |
|---|---:|---:|---:|---:|
| Real train | 100 | 775 | 7.750 | 6.000 |
| Official RAW val | 500 | 3,140 | 6.280 | 4.954 |
| BG synthetic only | 240 | 240 | 1.000 | 1.000 |
| Real + BG | 340 | 1,015 | 2.985 | 2.471 |

The same 240 single-object, single-class image design is used in Random, DQ and BG. Synthetic images are 70.59% of augmented training images but only 23.65% of annotated objects. With batch size 8, uniform image sampling would expose approximately 5.65 synthetic and 2.35 real images per batch; this is an exposure estimate, not a measured gradient contribution.

| Distance to official RAW val | Real train | Real + BG |
|---|---:|---:|
| Object-count Wasserstein | 1.470 | 3.580 |
| Normalized class co-occurrence L1 | 10.754 | 13.952 |
| Normalized class co-occurrence Frobenius | 1.894 | 2.382 |

Mean union-bbox foreground fraction also falls from 2.562% in real train (2.129% in val) to 1.117% in Real + BG. These statistics show that the augmented image population is substantially less like the validation population in object density and class co-occurrence than the real-only population is.

## Other descriptive checks

- Background reuse: 240 BG candidates use 96 distinct physical background pairs; a used pair contributes 2.50 synthetic images on average, maximum 6. This is exposure, not isolated evidence that repetition caused the performance change. Random uses 97 pairs; DQ uses 86.
- Corrected best-checkpoint epoch means across seeds: Real-only 79.7, Random 58.3, DQ 54.3, BG 70.7. Individual seeds vary; there is no consistent evidence that every synthetic arm reaches its best checkpoint earlier. The epoch calculation uses the same 0.1×mAP50 + 0.9×mAP50-95 fitness rule as the saved checkpoints.
- Plastic_Bomo has 138 real images with 1.217 objects/image and 80 synthetic images (36.70% of augmented images); its augmented mean is 1.138 objects/image. DeepPCB starts at 7.750 objects/image and drops to 2.985 after augmentation. This is a within-dataset composition comparison, not a comparison of absolute mAP across datasets.

## Evidence ranking and interpretation

1. Image-level object-density shift: supported by the increased validation distance.
2. Class co-occurrence collapse: supported by the increased validation distance.
3. Background repetition: observed, causal impact not isolated.
4. Training-seed variance: observed in Stage 4M.
5. Synthetic realism gap: **NOT_ESTABLISHED** by this analysis.

`IMAGE_LEVEL_SHIFT_STRONGLY_SUPPORTED` means the measured composition shift is strong; it does **not** mean the shift has been causally proven to explain the entire BG-versus-Real-only performance gap. The two pre-registered future alternatives are density/co-occurrence-preserving synthetic composition and real/synthetic exposure balancing. Neither was executed here.
