# TF-IDG model card for this comparison

TF-IDG is used as a training-free reference-guided diffusion baseline. The public AnyDoor and DINOv2 ViT-g weights are frozen. Plastic_Bomo contributes only V2 TRAIN reference defects and frozen normal/BBox slots. No MSDF component, detector, selection, fine-tuning, or test GT is used.

Reference and target masks are frozen rectangular BBox masks (`COARSE_BBOX_MASK`), not dense human segmentations. Images and weights remain server-only.

## Local compatibility adapter

The official inference path was kept algorithmically intact. The local source adapter fixes the repository's `checkPoint/` path capitalization, reads `use_interactive_seg` from the loaded inference config, supplies the frozen per-slot seed to Python/NumPy/PyTorch, and retains the official 50-step DDIM, energy-guidance, adaptive-mask, strength, start-step, and AdaIN settings.

The server copy originally imposed an artificial 60% per-process CUDA-memory cap. It was changed to an environment-controlled cap (`TFIDG_CUDA_MEMORY_FRACTION`, run value `0.95`) and the runner releases cached allocations after each batch-1 sample. This is memory management only: model structure, weights, masks, reference choice, seeds, and inference equations were not changed.
