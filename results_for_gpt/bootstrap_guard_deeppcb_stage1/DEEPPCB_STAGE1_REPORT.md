# DeepPCB Stage 1 Report

## Conclusion

**Status: `DEEPPCB_STAGE1_READY_FOR_POOL_GENERATION`**

The frozen low-data protocol passed its integrity gate, all six class-specific MSDF-v3 models completed training with finite losses, and the 30-image smoke generation passed artifact and leakage checks. Stage 1 is ready to proceed to candidate-pool generation; this report does not define or launch that later stage.

## Frozen low-data protocol integrity

- Selected split: 100 unique physical PCB pairs (100 unique defect paths and 100 unique paired-normal paths).
- Frozen pair-ID SHA256: `1ec83d6310aca8b0a4db688bfd50c2d74f95b5bcdc5e98a386f6951aa08cf999`.
- Selected annotations: 775; retained: 775; dropped: 0; extra: 0; class-label mismatches: 0.
- Every selected PCB pair contains all six defect classes. Independent per-class unique-image counts are 100 for every class.
- Instance counts: open 130; short, mousebite, spur, spurious_copper, and pinhole 129 each.
- Frozen stable-contrast threshold: `tau_sigma = 58.7430814883903`.
- Official validation overlap: 0 pairs.

## Frozen training protocol

- Initialization: the same original Stable Diffusion 2 Inpainting checkpoint for every class; no Plastic_Bomo initialization.
- Architecture: frozen MSDF-v3 bounded dual-branch injection.
- Independent models: open, short, mousebite, spur, spurious_copper, and pinhole.
- Optimizer: AdamW; 2,000 successful optimizer updates; batch size 1; gradient accumulation 4; seed 42; fp16.
- Learning rates: U-Net `5e-6`; MSDF `1e-4`.
- Checkpoint rule: final checkpoint after 2,000 updates, with no validation-based selection.

## Checkpoint audit

| Class | Status | Final loss | Minimum loss | Checkpoint |
|---|---:|---:|---:|---|
| open | READY | 0.328081 | 0.023068 | `model/msdf_v3_deeppcb_lowdata_s42/open/msdf.pt` |
| short | READY | 0.386541 | 0.020174 | `model/msdf_v3_deeppcb_lowdata_s42/short/msdf.pt` |
| mousebite | READY | 0.180215 | 0.011302 | `model/msdf_v3_deeppcb_lowdata_s42/mousebite/msdf.pt` |
| spur | READY | 0.253785 | 0.014865 | `model/msdf_v3_deeppcb_lowdata_s42/spur/msdf.pt` |
| spurious_copper | READY | 0.210377 | 0.009524 | `model/msdf_v3_deeppcb_lowdata_s42/spurious_copper/msdf.pt` |
| pinhole | READY | 0.226493 | 0.002683 | `model/msdf_v3_deeppcb_lowdata_s42/pinhole/msdf.pt` |

All six processes exited normally, all recorded losses are finite, and every checkpoint is 1,364,198 bytes.

## Smoke-generation audit

- Seeds: 42, 3407, 2026, 7859, and 12011.
- Output count: 5 candidates per class, 30 total.
- All generated records have finite area, morphology, stable-contrast, and context-sigma attributes.
- All masks, bounding boxes, and YOLO labels are valid.
- All source, reference, and background provenance is restricted to the frozen low-data training split.
- Official validation data was not used.

## Decision

DeepPCB Stage 1 is complete and passes the prerequisite checks for the next candidate-pool generation stage. The smoke run verifies pipeline correctness and artifact validity; it is not itself a diversity, quality, or downstream-performance conclusion.
