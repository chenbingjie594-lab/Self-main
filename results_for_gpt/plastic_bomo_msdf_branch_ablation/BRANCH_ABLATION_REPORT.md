# Experiment 14 — MSDF Pixel/Latent Dual-Branch Ablation

Status: `DUAL_BRANCH_CONSISTENT_BENEFIT`.

This is a frozen-checkpoint inference ablation. Full, PixelOnly and LatentOnly use identical weights and gamma=1. PixelOnly disables the latent encoder contribution. LatentOnly disables both the pixel encoder contribution and pixel support, which also changes downstream morphology, support and dynamic gating. The results therefore measure coupled branch-path effects, not pure independent encoder contributions.

## Bankwise Black

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0379734447 | 161.766924 |
| A | BackgroundOnly Gamma=0 | 0.0889144026 | 215.575033 |
| A | Full Dual-Branch | 0.0610055755 | 191.966915 |
| A | PixelOnly | 0.0693023369 | 197.265217 |
| A | LatentOnly | 0.0682713123 | 198.444827 |
| B | Raw SD2 | 0.0451930130 | 172.446636 |
| B | BackgroundOnly Gamma=0 | 0.0843191988 | 214.712135 |
| B | Full Dual-Branch | 0.0589119648 | 183.167155 |
| B | PixelOnly | 0.0627188459 | 189.162660 |
| B | LatentOnly | 0.0696429198 | 205.816954 |
| C | Raw SD2 | 0.0396594895 | 163.039112 |
| C | BackgroundOnly Gamma=0 | 0.0819206051 | 208.363846 |
| C | Full Dual-Branch | 0.0625615813 | 192.421727 |
| C | PixelOnly | 0.0680597325 | 198.444529 |
| C | LatentOnly | 0.0630567467 | 196.419812 |

## Bankwise Overall

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0352225710 | 149.427503 |
| A | BackgroundOnly Gamma=0 | 0.0535316454 | 167.818726 |
| A | Full Dual-Branch | 0.0429452473 | 154.905368 |
| A | PixelOnly | 0.0468402547 | 158.665499 |
| A | LatentOnly | 0.0454889221 | 157.798422 |
| B | Raw SD2 | 0.0364288648 | 153.024285 |
| B | BackgroundOnly Gamma=0 | 0.0587844988 | 170.261286 |
| B | Full Dual-Branch | 0.0474332942 | 155.388943 |
| B | PixelOnly | 0.0494397057 | 158.103821 |
| B | LatentOnly | 0.0518532680 | 164.536917 |
| C | Raw SD2 | 0.0403267710 | 152.163042 |
| C | BackgroundOnly Gamma=0 | 0.0579059031 | 166.153269 |
| C | Full Dual-Branch | 0.0487462165 | 158.366801 |
| C | PixelOnly | 0.0503736127 | 160.849206 |
| C | LatentOnly | 0.0476928314 | 159.501857 |

## Pooled metrics

These metrics were recomputed from concatenated Inception features; no class or bank metric was averaged.

| Scope | Population | Method | Images | KID ↓ | FID ↓ |
| --- | --- | --- | ---: | ---: | ---: |
| A+B+C | Black | Raw SD2 | 120 | 0.0436378602 | 144.564373 |
| A+B+C | Black | BackgroundOnly Gamma=0 | 120 | 0.0860462536 | 187.528169 |
| A+B+C | Black | Full Dual-Branch | 120 | 0.0626064865 | 165.678980 |
| A+B+C | Black | PixelOnly | 120 | 0.0679663956 | 170.411014 |
| A+B+C | Black | LatentOnly | 120 | 0.0679206121 | 174.662221 |
| A+B+C | Overall | Raw SD2 | 240 | 0.0384155666 | 133.427437 |
| A+B+C | Overall | BackgroundOnly Gamma=0 | 240 | 0.0568490017 | 149.018226 |
| A+B+C | Overall | Full Dual-Branch | 240 | 0.0467225114 | 138.162539 |
| A+B+C | Overall | PixelOnly | 240 | 0.0490596889 | 140.783956 |
| A+B+C | Overall | LatentOnly | 240 | 0.0484051395 | 141.236236 |
| B+C | Black | Raw SD2 | 80 | 0.0431993046 | 152.531956 |
| B+C | Black | BackgroundOnly Gamma=0 | 80 | 0.0834684092 | 193.255567 |
| B+C | Black | Full Dual-Branch | 80 | 0.0604039292 | 170.796441 |
| B+C | Black | PixelOnly | 80 | 0.0647112949 | 176.112176 |
| B+C | Black | LatentOnly | 80 | 0.0661469439 | 182.293431 |
| B+C | Overall | Raw SD2 | 160 | 0.0409881956 | 138.961555 |
| B+C | Overall | BackgroundOnly Gamma=0 | 160 | 0.0606951058 | 154.444034 |
| B+C | Overall | Full Dual-Branch | 160 | 0.0510395576 | 143.488476 |
| B+C | Overall | PixelOnly | 160 | 0.0526714940 | 145.678649 |
| B+C | Overall | LatentOnly | 160 | 0.0521174777 | 147.567000 |

## Findings

On B+C pooled Black, Full minus PixelOnly is -0.0043073657 KID and -5.315735 FID; Full is better when these values are negative.

On B+C pooled Black, Full minus LatentOnly is -0.0057430147 KID and -11.496990 FID.

The larger B+C Black KID closure effect came from `pixel_branch_and_coupled_support`. For Pixel closure this includes the coupled pixel-support path.

B/C bankwise Full beat both single branches on every Black KID and FID check: `True`. B+C pooled Full beat both on both metrics: `True`.

Best B+C pooled Black methods: KID `raw`, FID `raw`. Best A+B+C Overall methods: KID `raw`, FID `raw`.

Full versus Raw on A+B+C pooled Black: KID +0.0189686263 (+43.47%), FID +21.114607 (+14.61%).

Whether the two branches should be retained follows the declared status above. A training-level single-branch ablation would be a separate experiment and is warranted only to test training behavior; this inference experiment cannot establish it.

All requested pairwise absolute and relative changes are in `branch_pairwise_deltas.csv`. Feature, support, gate and up-block residual diagnostics are in `branch_feature_diagnostics.json`.

## Direct answers

1. **Full vs PixelOnly:** Full is better on Black KID and FID in both B and C and in B+C pooled results.
2. **Full vs LatentOnly:** Full is better on Black KID and FID in both B and C and in B+C pooled results.
3. **Larger closure effect:** Closing the Pixel branch and its coupled Pixel Support path has the larger B+C Black impact: Full–LatentOnly is -0.0057430147 KID and -11.496990 FID, versus Full–PixelOnly at -0.0043073657 KID and -5.315735 FID. This cannot be assigned to pixel encoder features alone.
4. **B/C consistency:** Yes. Full beats both single-branch variants on Black KID and FID in each held-out-seed bank.
5. **Best pooled Black:** Raw SD2 is best among all methods. Full is best among the three MSDF branch configurations for A+B+C and B+C on both KID and FID.
6. **Best Overall:** Raw SD2 is best among all methods for pooled Overall KID and FID. Full is best among the three MSDF branch configurations.
7. **Full vs Raw:** On A+B+C pooled Black, Full is +0.0189686263 KID (+43.47%) and +21.114607 FID (+14.61%) behind Raw. On Overall, it is +0.0083069448 KID (+21.62%) and +4.735103 FID (+3.55%) behind Raw.
8. **Retain dual branch:** Yes for this frozen inference configuration because Full wins every declared B/C and B+C Black comparison. This is not a training-level architecture claim.
9. **Training-level single-branch ablation:** It is worth running as a separate next experiment to distinguish learned branch value from inference-time support/gate coupling. It was not started here.

## Research boundary

Inference-time coupled branch ablation only. Pixel disablement also removes pixel support and changes downstream morphology/support/gating; results are not pure independent encoder contributions.

Banks share slots and V2 TRAIN references may overlap training; no independent-test or statistical-significance claim.

No training, architecture or fusion change, loss change, gamma search, seed search, final_eval use, BootstrapGuard or detector training occurred. The experiment stops here.
