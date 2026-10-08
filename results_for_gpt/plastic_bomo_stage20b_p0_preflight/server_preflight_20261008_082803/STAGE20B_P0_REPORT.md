# Stage20B-P0 — Raw Generator Baseline Structural Preflight

Status: `GENERATOR_TRAIN_SPLIT_CONTAMINATION`. Scope: `SERVER_PREFLIGHT`. Execution authorized: `False`. NO training, sampling, model loading or final_eval forward occurred. This is not Stage17 rescue, selector/task-guidance/morphology or BootstrapGuard work.

Stage20A is bound to commit aa183e564a12c4d33d1283616921f370e4ba655d and its original frozen SHA256 ledger, unchanged. All split/slot decisions use V2 TRAIN metadata only. The final_eval files themselves are not opened at P0. Only no **KNOWN** source-group leakage is supported; `physical_source_complete=false`.

Annotations remain dataset-defined supervision, not universally human-verified physical ground truth. Rectangular bbox masks are weak spatial geometry, not segmentation ground truth. Historical provenance and the two clipping issues remain unchanged.

V2 train/final_eval differ from Stage14-18 evaluation protocols. Absolute mAP comparisons across those protocols are forbidden; future interpretation is RR vs raw banks within V2 only. Stage17 nuisance SD/threshold are not transferred.

The protocols specify one new class-specific full-UNet SD2 baseline from the hash-matched public original initialization, 2000 successful updates/class, deterministic 512 crops, no normal-only training and no new loss. A crop that truncates its target is structurally rejected in the predeclared ordering before slot freeze, never moved to rescue it. No mask/crop training images have been exported or synthesized at P0.

80 slots (40/40) use source-group round-robin and stable hashes, without quality metrics. A/B/C differ only in generation RNG; normals, geometry, prompt, class checkpoint ROLE, scheduler and preprocessing are shared. The future trained checkpoints do not exist yet: their hashes MUST be shared and frozen across banks before sampling. All successful raw outputs must be used. Structural failures after freeze stop the comparison, with no reroll, replacement or budget reduction.

RR is 80 replay exposures of full donor images and complete labels, NOT 80 independent new real images. The real base is identical 136 images. Training roles and per-position augmentation seeds are shared. YOLO11s uses fixed explicit AdamW and no composite augmentation, 150 epochs, batch1, imgsz1536, accumulation64 with a 24-draw epoch tail: 32400 draws/batches and 600 scheduled optimizer attempts. This is a predeclared V2 budget, not a retroactive adjustment. Successful AMP updates may differ and must be reported. The future trainer must explicitly suppress early stopping AND its automatic final evaluation; val=False alone is not sufficient.

All four epoch150 last.pt checkpoints must be frozen before any unified V2 final_eval evaluation. Gates remain mean bank-RR >=0.50pp, at least2 strict bank wins, each class mean >=-2.0pp. Failure stops with no automatic generator/CFG/steps/prompt/bank/selector fallback. TDCRG remains unauthorized.

Local results are NOT server weight/runtime verification. A server P0 run must hash-bind the same Stage20A assets, public SD2 weights, pretrained detector, code and actual runtime package/source files. Missing or changed assets are stop gates, not invitations to replace data. P0 does not test GPU numerical determinism or differentiation: these are runtime checks to enforce before the first future formal launch, never evidence claimed here. No model weights were downloaded. Public reference metadata: https://huggingface.co/sd2-community/stable-diffusion-2-inpainting/tree/5f74973cbb64c8568780732c17f43eb269d63a0d
