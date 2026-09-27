# Stage9A local method difference

| Method | Backbone update | Allowed condition | Stage9A role |
|---|---|---|---|
| Vanilla SD2 | Full UNet | mask, normal image, text, timestep | existing control |
| Full MSDF-v3 | Full UNet + MSDF | includes defect reference | existing diagnostic control |
| FrozenResidualAdapter | none | mask, paired normal context, timestep | new prototype |

DualAnoDiff, AnomalyDiffusion and DefectFill require external baseline verification; no unverified architectural claims are made.
