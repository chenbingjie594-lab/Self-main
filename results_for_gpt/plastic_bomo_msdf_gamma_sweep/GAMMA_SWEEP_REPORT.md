# MSDF Feature Residual Injection Strength Sweep

Status: `COMPLETE`. This was an inference-only Bank-A development sweep; no model was trained.

## Protocol and interpretation limits

All five gamma values used the same 80 frozen Bank-A slots, normals, bboxes, prompts, seeds, steps, scheduler, CFG and checkpoints. Gamma scales only the final tanh-bounded residual immediately before it is added to each UNet up-block. At gamma=1.0 the multiply is skipped and two class-specific end-to-end PNGs exactly matched the prior frozen MSDF outputs.

The reference is all 172 original V2 TRAIN ROIs (Flash 75, Black 97). It may overlap the data used to train these checkpoints. These scores are exploratory screening results, not independent validation or final paper performance. `final_eval` was not read or used. Historical final_eval Raw SD2/MSDF scores are not directly comparable to this table.

LPIPS is an exploratory Bank-A diversity reference from 20 deterministic adjacent slot pairs per class. It differs from the historical same-slot A/B/C LPIPS protocol and is not directly comparable to it.

## Complete results

| gamma | Flash KID ↓ | Black KID ↓ | Overall KID ↓ | Flash FID ↓ | Black FID ↓ | Overall FID ↓ | Flash LPIPS ↑ | Black LPIPS ↑ | Overall LPIPS ↑ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.05019081 | 0.08286843 | 0.05103164 | 161.14909 | 211.20923 | 166.88055 | 0.487301 | 0.476124 | 0.481713 |
| 0.25 | 0.05039118 | 0.08280198 | 0.05124602 | 161.69725 | 211.02437 | 167.05994 | 0.486704 | 0.477276 | 0.481990 |
| 0.50 | 0.05213159 | 0.08131966 | 0.05161607 | 162.22027 | 209.84517 | 166.54391 | 0.486162 | 0.478246 | 0.482204 |
| 0.75 | 0.05370484 | 0.08008924 | 0.05182682 | 163.22390 | 208.91387 | 166.54244 | 0.485504 | 0.478675 | 0.482089 |
| 1.00 | 0.05334290 | 0.08004625 | 0.05151713 | 161.28865 | 208.47891 | 165.26896 | 0.484835 | 0.478834 | 0.481834 |

## Required findings

1. Lower residual strength does not improve Black KID. Gamma=1.00 is best at 0.08004625; gamma=0.75 is +0.00004299 worse and gamma=0 is +0.00282218 worse.
2. The trend is monotonic in the opposite direction: Black KID falls at every step as gamma increases (0.08286843 → 0.08280198 → 0.08131966 → 0.08008924 → 0.08004625). Reducing gamma therefore steadily hurts Black KID on this development set.
3. Flash does not show a consistent negative response to lower gamma. Relative to gamma=1.00, Flash KID improves at gamma=0, 0.25 and 0.50, but worsens by +0.00036194 at gamma=0.75. Flash FID is best at gamma=0 and is worse at gamma=0.25–0.75. Because no lower gamma improves Black, there is no useful Black/Flash tradeoff candidate in this sweep.
4. Gamma=0 is not better than gamma=1 for Black KID (0.08286843 vs 0.08004625).
5. This TRAIN-overlap screen does not show evidence of excessive residual injection at gamma=1.00. This conclusion requires multiple banks and source-group-isolated validation before formal use.

The sweep stops here. No wider gamma search, new training, BootstrapGuard, multi-bank confirmation or second optimization round was started.
