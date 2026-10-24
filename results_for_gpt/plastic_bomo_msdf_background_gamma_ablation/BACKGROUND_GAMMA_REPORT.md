# Experiment 13 — MSDF BackgroundOnly Residual Injection Ablation

Status: `RESIDUAL_INJECTION_CONSISTENT_BENEFIT`.

This experiment changes only the final inference-time MSDF residual injection gamma on one frozen BackgroundOnly checkpoint. Gamma=0 retains the trained UNet, MSDF checkpoint, reference preparation, branches, morphology alignment and dynamic gate; it is not Raw SD2 or a training-level ablation.

## Bankwise Black

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0379734447 | 161.766924 |
| A | BackgroundOnly Gamma=0 | 0.0889144026 | 215.575033 |
| A | BackgroundOnly Gamma=1 | 0.0610055755 | 191.966915 |
| B | Raw SD2 | 0.0451930130 | 172.446636 |
| B | BackgroundOnly Gamma=0 | 0.0843191988 | 214.712135 |
| B | BackgroundOnly Gamma=1 | 0.0589119648 | 183.167155 |
| C | Raw SD2 | 0.0396594895 | 163.039112 |
| C | BackgroundOnly Gamma=0 | 0.0819206051 | 208.363846 |
| C | BackgroundOnly Gamma=1 | 0.0625615813 | 192.421727 |

## Bankwise Overall

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0352225710 | 149.427503 |
| A | BackgroundOnly Gamma=0 | 0.0535316454 | 167.818726 |
| A | BackgroundOnly Gamma=1 | 0.0429452473 | 154.905368 |
| B | Raw SD2 | 0.0364288648 | 153.024285 |
| B | BackgroundOnly Gamma=0 | 0.0587844988 | 170.261286 |
| B | BackgroundOnly Gamma=1 | 0.0474332942 | 155.388943 |
| C | Raw SD2 | 0.0403267710 | 152.163042 |
| C | BackgroundOnly Gamma=0 | 0.0579059031 | 166.153269 |
| C | BackgroundOnly Gamma=1 | 0.0487462165 | 158.366801 |

## Pooled metrics

Pooled metrics were recomputed from concatenated features, not averaged across banks or classes.

| Scope | Population | Method | Images | KID ↓ | FID ↓ |
| --- | --- | --- | ---: | ---: | ---: |
| A+B+C | Black | Raw SD2 | 120 | 0.0436378602 | 144.564373 |
| A+B+C | Black | BackgroundOnly Gamma=0 | 120 | 0.0860462536 | 187.528169 |
| A+B+C | Black | BackgroundOnly Gamma=1 | 120 | 0.0626064865 | 165.678980 |
| A+B+C | Overall | Raw SD2 | 240 | 0.0384155666 | 133.427437 |
| A+B+C | Overall | BackgroundOnly Gamma=0 | 240 | 0.0568490017 | 149.018226 |
| A+B+C | Overall | BackgroundOnly Gamma=1 | 240 | 0.0467225114 | 138.162539 |
| B+C | Black | Raw SD2 | 80 | 0.0431993046 | 152.531956 |
| B+C | Black | BackgroundOnly Gamma=0 | 80 | 0.0834684092 | 193.255567 |
| B+C | Black | BackgroundOnly Gamma=1 | 80 | 0.0604039292 | 170.796441 |
| B+C | Overall | Raw SD2 | 160 | 0.0409881956 | 138.961555 |
| B+C | Overall | BackgroundOnly Gamma=0 | 160 | 0.0606951058 | 154.444034 |
| B+C | Overall | BackgroundOnly Gamma=1 | 160 | 0.0510395576 | 143.488476 |

## Finding

Gamma=1 beat Gamma=0 across A/B/C in Black KID on 3/3 banks, Black FID on 3/3, Overall KID on 3/3 and Overall FID on 3/3.

For the held-out generation seeds B+C, pooled Black KID and FID were both improved by injection: `True`.

Against Raw SD2 on A+B+C pooled features, BackgroundOnly Gamma=1 still has 43.47% higher Black KID, 14.61% higher Black FID, 21.62% higher Overall KID and 3.55% higher Overall FID. The residual injection is beneficial within this frozen BackgroundOnly model, but it does not close the Raw SD2 gap.

All Gamma=1 minus Gamma=0 and Raw comparisons, including absolute and relative changes, are in `background_gamma_pairwise_deltas.csv`. Negative means the first method is lower and better.

Residual diagnostics report nonzero per-block tensor statistics, dynamic-gate ranges, finiteness and proximity to the tanh bound. Nonzero residuals do not by themselves establish a quality benefit.

A later controlled experiment may examine training-level Pixel/Latent dual-branch fusion because the useful inference residual remains insufficient relative to Raw SD2. No follow-up experiment was started here.

## Research boundary

Inference-time final residual ablation on one trained model; not a training-level MSDF or branch ablation.

A/B/C share slots and V2 TRAIN references; not independent-test or cross-dataset validation.

No training, Gamma search, loss change, Region recalibration, architecture change, final_eval, BootstrapGuard or detector training was run. The experiment stops here.
