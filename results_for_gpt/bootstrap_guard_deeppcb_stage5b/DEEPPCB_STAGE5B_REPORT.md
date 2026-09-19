# DeepPCB Stage 5B Exposure Pilot

Status: **EXPOSURE_BALANCE_SIGNAL_POSITIVE**
Beats Real-only: **False**

Only seed42 was trained; images, labels, model and validation remained frozen.
Sampler: 260 real + 80 synthetic draws per epoch, 127 completed epochs.
The quota governs dataloader base-image draws. Unchanged YOLO Mosaic augmentation may load additional images, so effective post-augmentation pixel exposure is not proven to equal 260/80.
Uniform BG mAP50-95/Recall: 0.639801/0.862442.
Exposure-balanced BG mAP50-95/Recall: 0.645491/0.869028.
Exposure-balanced minus uniform BG mAP50-95: +0.005690.
Exposure-balanced minus Real-only mAP50-95: -0.026055.

## Protocol and exposure audit

- The source probability was derived only from training objects: 240/(775+240) = 0.236453 synthetic. The frozen rule rounds 340 draws to 260 real and 80 synthetic per epoch. No validation statistic was used to choose the ratio.
- All 127 completed epochs passed the exact 260/80 quota and consumed-base-image sequence check: 33,020 real and 10,160 synthetic base-image draws. Across training, all 100 real and 240 synthetic images appeared at least once.
- Cumulative synthetic class draws: open 1,678; short 1,740; mousebite 1,715; spur 1,677; spurious_copper 1,686; pinhole 1,664. Uniform sampling within the BG source was preserved.
- Frozen BG manifest, training/validation dataset hashes, baseline checkpoint hashes and checked YOLO hyperparameters passed. Original Uniform BG and Real-only seed42 were reused without retraining. The official RAW validation manifest remained unchanged.

## Independent validation

| Method | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Real-only | 0.910338 | 0.864832 | 0.927615 | 0.671547 |
| Uniform BG | 0.928035 | 0.862442 | 0.926635 | 0.639801 |
| Exposure-balanced BG | 0.933292 | 0.869028 | 0.930633 | 0.645491 |

The primary metric improved by **+0.005690** over Uniform BG, meeting the pre-registered positive rule, but remained **−0.026055** below Real-only. This small single-seed gain is not evidence that exposure balancing eliminates DeepPCB's negative transfer.

| Class | Uniform BG AP50-95 | Exposure-balanced AP50-95 | Delta |
|---|---:|---:|---:|
| open | 0.573994 | 0.616310 | +0.042316 |
| short | 0.474495 | 0.491468 | +0.016973 |
| mousebite | 0.636241 | 0.634042 | −0.002199 |
| spur | 0.625864 | 0.618665 | −0.007199 |
| spurious_copper | 0.813039 | 0.818739 | +0.005700 |
| pinhole | 0.715175 | 0.693724 | −0.021450 |

Three classes improved and three declined. In particular, pinhole AP50-95 fell by 0.021450 even though the overall metric rose. The status therefore reflects the primary pre-registered mAP50-95 rule, not uniform per-class improvement.

## Interpretation limits

The intervention changed the training dataloader's *base-image* source exposure only. Original YOLO augmentation, including Mosaic, remained enabled and may read extra images outside the 260/80 base-image quota; actual augmented-pixel source composition was not measured. Training also stopped after 127 epochs rather than the full 150 because the frozen early-stopping rule remained in force. This is one seed, not multi-seed confirmation. Neither the 260/80 ratio nor the selector should be retuned from this result.
