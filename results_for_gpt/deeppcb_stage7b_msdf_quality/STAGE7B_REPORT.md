# DeepPCB Stage 7B matched image-quality benchmark

Status: **MSDF_IMAGE_FIDELITY_NOT_SUPPORTED**.
Evaluation is training-reference fidelity, not unseen generalization.
Local-KID x1000 (primary): SD2 2.5182; MSDF 4.1750.
Mask-crop LPIPS: SD2 0.028639; MSDF 0.035063; MSDF wins 22.06%.
SSIM: SD2 0.977768; MSDF 0.975368. Edge cosine: SD2 0.930212; MSDF 0.923605.
Global-KID x1000 (secondary): SD2 -2.1608; MSDF -1.9846.
Fidelity and diversity are reported separately; no weighted quality score was constructed.
No detector, selector, candidate generation, or generator modification was performed.
