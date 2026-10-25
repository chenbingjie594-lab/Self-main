# Experiment 15 — Training-Level PixelOnly Ablation

Status: `TRAINING_BRANCH_MIXED`.

One PixelOnly Black model was trained from the same public SD2 base for exactly 2000 successful updates. Its latent encoder remained in the checkpoint structure but was frozen, excluded from AdamW, and its feature contribution was zeroed. Pixel support, fusion, morphology alignment, dynamic gating and all four up-block injections remained enabled. Branch dropout was fixed to 0.0.

Compared with the historical Full model, this changes both the branch configuration and the necessary single-branch dropout setting. It also changes random-number consumption. The comparison is therefore not a strict one-variable causal experiment.

## Bankwise Black

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0379734447 | 161.766924 |
| A | BackgroundOnly Gamma=0 | 0.0889144026 | 215.575033 |
| A | Full Dual-Branch | 0.0610055755 | 191.966915 |
| A | PixelOnly Inference | 0.0693023369 | 197.265217 |
| A | PixelOnly Training | 0.0736958968 | 199.053764 |
| B | Raw SD2 | 0.0451930130 | 172.446636 |
| B | BackgroundOnly Gamma=0 | 0.0843191988 | 214.712135 |
| B | Full Dual-Branch | 0.0589119648 | 183.167155 |
| B | PixelOnly Inference | 0.0627188459 | 189.162660 |
| B | PixelOnly Training | 0.0653272386 | 191.057712 |
| C | Raw SD2 | 0.0396594895 | 163.039112 |
| C | BackgroundOnly Gamma=0 | 0.0819206051 | 208.363846 |
| C | Full Dual-Branch | 0.0625615813 | 192.421727 |
| C | PixelOnly Inference | 0.0680597325 | 198.444529 |
| C | PixelOnly Training | 0.0692697182 | 188.823004 |

## Bankwise Overall

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0352225710 | 149.427503 |
| A | BackgroundOnly Gamma=0 | 0.0535316454 | 167.818726 |
| A | Full Dual-Branch | 0.0429452473 | 154.905368 |
| A | PixelOnly Inference | 0.0468402547 | 158.665499 |
| A | PixelOnly Training | 0.0485407234 | 158.832706 |
| B | Raw SD2 | 0.0364288648 | 153.024285 |
| B | BackgroundOnly Gamma=0 | 0.0587844988 | 170.261286 |
| B | Full Dual-Branch | 0.0474332942 | 155.388943 |
| B | PixelOnly Inference | 0.0494397057 | 158.103821 |
| B | PixelOnly Training | 0.0501353270 | 160.775759 |
| C | Raw SD2 | 0.0403267710 | 152.163042 |
| C | BackgroundOnly Gamma=0 | 0.0579059031 | 166.153269 |
| C | Full Dual-Branch | 0.0487462165 | 158.366801 |
| C | PixelOnly Inference | 0.0503736127 | 160.849206 |
| C | PixelOnly Training | 0.0516689898 | 158.916353 |

## Pooled metrics

Pooled metrics were recomputed from concatenated features; bank and class scores were not averaged.

| Scope | Population | Method | Images | KID ↓ | FID ↓ |
| --- | --- | --- | ---: | ---: | ---: |
| A+B+C | Black | Raw SD2 | 120 | 0.0436378602 | 144.564373 |
| A+B+C | Black | BackgroundOnly Gamma=0 | 120 | 0.0860462536 | 187.528169 |
| A+B+C | Black | Full Dual-Branch | 120 | 0.0626064865 | 165.678980 |
| A+B+C | Black | PixelOnly Inference | 120 | 0.0679663956 | 170.411014 |
| A+B+C | Black | PixelOnly Training | 120 | 0.0707290324 | 168.764660 |
| A+B+C | Overall | Raw SD2 | 240 | 0.0384155666 | 133.427437 |
| A+B+C | Overall | BackgroundOnly Gamma=0 | 240 | 0.0568490017 | 149.018226 |
| A+B+C | Overall | Full Dual-Branch | 240 | 0.0467225114 | 138.162539 |
| A+B+C | Overall | PixelOnly Inference | 240 | 0.0490596889 | 140.783956 |
| A+B+C | Overall | PixelOnly Training | 240 | 0.0503590571 | 140.754751 |
| B+C | Black | Raw SD2 | 80 | 0.0431993046 | 152.531956 |
| B+C | Black | BackgroundOnly Gamma=0 | 80 | 0.0834684092 | 193.255567 |
| B+C | Black | Full Dual-Branch | 80 | 0.0604039292 | 170.796441 |
| B+C | Black | PixelOnly Inference | 80 | 0.0647112949 | 176.112176 |
| B+C | Black | PixelOnly Training | 80 | 0.0666873966 | 171.468224 |
| B+C | Overall | Raw SD2 | 160 | 0.0409881956 | 138.961555 |
| B+C | Overall | BackgroundOnly Gamma=0 | 160 | 0.0606951058 | 154.444034 |
| B+C | Overall | Full Dual-Branch | 160 | 0.0510395576 | 143.488476 |
| B+C | Overall | PixelOnly Inference | 160 | 0.0526714940 | 145.678649 |
| B+C | Overall | PixelOnly Training | 160 | 0.0539061835 | 145.771695 |

## Required findings

1. G3 B+C pooled Black KID/FID are 0.0666873966 / 171.468224; A+B+C pooled values are 0.0707290324 / 168.764660.
2. Versus PixelOnly Inference on B+C, retraining changed KID by +0.0019761017 and FID by -4.643952.
3. Versus Full on B+C, PixelOnly Training changed KID by +0.0062834674 and FID by +0.671783.
4. B/C bankwise direction is recorded as `mixed directions`.
5. On A+B+C pooled Black, the best KID method is `raw` and the best FID method is `raw`.
6. Versus Raw on B+C, G3 differs by KID +0.0234880921 (+54.37%) and FID +18.936268 (+12.41%).
7. Pixel/Fusion and UNet gradients were finite and effective during training; latent encoder gradients remained None and its final SHA equals initialization.
8. Pixel Support was nonzero and finite in preflight and remained enabled in training and inference.
9. Support for retaining the existing dual-branch training configuration follows the declared status: `TRAINING_BRANCH_MIXED`.
10. The next-step recommendation is: do not infer latent uselessness from one seed; inspect fusion behavior before authorizing any new training

## Research boundary

A/B/C share slots, V2 TRAIN references may overlap generator training, B/C have been observed in prior work, and only one training seed was run. These are development results, not independent validation or evidence of cross-seed statistical significance.

No LatentOnly training, second seed, Full retraining, fusion change, loss search, gamma search, BootstrapGuard or detector training was started.
