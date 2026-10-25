# Experiment 16 — MSDF Pixel Feature / Pixel Support Factorial Decoupling

Status: `PIXEL_SUPPORT_PATH_PROMISING`.

This is a frozen-checkpoint 2×2 inference intervention. F controls the direct pixel-feature input to Fusion; S controls the explicit pixel-support term. F can still alter fused support and downstream morphology/gating, so the paths are not fully independent causal channels.

## Bankwise Black

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0379734447 | 161.766924 |
| A | BackgroundOnly Gamma=0 | 0.0889144026 | 215.575033 |
| A | F1S1 Full Dual-Branch | 0.0610055755 | 191.966915 |
| A | F0S0 LatentOnly | 0.0682713123 | 198.444827 |
| A | F0S1 (Pixel Feature OFF; Support ON) | 0.0618318202 | 195.641516 |
| A | F1S0 (Pixel Feature ON; Support OFF) | 0.0700751724 | 197.993778 |
| B | Raw SD2 | 0.0451930130 | 172.446636 |
| B | BackgroundOnly Gamma=0 | 0.0843191988 | 214.712135 |
| B | F1S1 Full Dual-Branch | 0.0589119648 | 183.167155 |
| B | F0S0 LatentOnly | 0.0696429198 | 205.816954 |
| B | F0S1 (Pixel Feature OFF; Support ON) | 0.0597471269 | 185.239525 |
| B | F1S0 (Pixel Feature ON; Support OFF) | 0.0620360040 | 201.988711 |
| C | Raw SD2 | 0.0396594895 | 163.039112 |
| C | BackgroundOnly Gamma=0 | 0.0819206051 | 208.363846 |
| C | F1S1 Full Dual-Branch | 0.0625615813 | 192.421727 |
| C | F0S0 LatentOnly | 0.0630567467 | 196.419812 |
| C | F0S1 (Pixel Feature OFF; Support ON) | 0.0614769226 | 193.289073 |
| C | F1S0 (Pixel Feature ON; Support OFF) | 0.0644988162 | 195.644068 |

## Bankwise Overall

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0352225710 | 149.427503 |
| A | BackgroundOnly Gamma=0 | 0.0535316454 | 167.818726 |
| A | F1S1 Full Dual-Branch | 0.0429452473 | 154.905368 |
| A | F0S0 LatentOnly | 0.0454889221 | 157.798422 |
| A | F0S1 (Pixel Feature OFF; Support ON) | 0.0427888648 | 156.286384 |
| A | F1S0 (Pixel Feature ON; Support OFF) | 0.0465731625 | 158.047760 |
| B | Raw SD2 | 0.0364288648 | 153.024285 |
| B | BackgroundOnly Gamma=0 | 0.0587844988 | 170.261286 |
| B | F1S1 Full Dual-Branch | 0.0474332942 | 155.388943 |
| B | F0S0 LatentOnly | 0.0518532680 | 164.536917 |
| B | F0S1 (Pixel Feature OFF; Support ON) | 0.0477423291 | 156.018769 |
| B | F1S0 (Pixel Feature ON; Support OFF) | 0.0481763959 | 162.645804 |
| C | Raw SD2 | 0.0403267710 | 152.163042 |
| C | BackgroundOnly Gamma=0 | 0.0579059031 | 166.153269 |
| C | F1S1 Full Dual-Branch | 0.0487462165 | 158.366801 |
| C | F0S0 LatentOnly | 0.0476928314 | 159.501857 |
| C | F0S1 (Pixel Feature OFF; Support ON) | 0.0480919716 | 158.302796 |
| C | F1S0 (Pixel Feature ON; Support OFF) | 0.0480630553 | 158.948700 |

## Pooled metrics

All pooled and Overall metrics were recomputed from concatenated features.

| Scope | Population | Method | Images | KID ↓ | FID ↓ |
| --- | --- | --- | ---: | ---: | ---: |
| A+B+C | Black | Raw SD2 | 120 | 0.0436378602 | 144.564373 |
| A+B+C | Black | BackgroundOnly Gamma=0 | 120 | 0.0860462536 | 187.528169 |
| A+B+C | Black | F1S1 Full Dual-Branch | 120 | 0.0626064865 | 165.678980 |
| A+B+C | Black | F0S0 LatentOnly | 120 | 0.0679206121 | 174.662221 |
| A+B+C | Black | F0S1 (Pixel Feature OFF; Support ON) | 120 | 0.0622256151 | 166.769833 |
| A+B+C | Black | F1S0 (Pixel Feature ON; Support OFF) | 120 | 0.0669875753 | 173.324776 |
| A+B+C | Overall | Raw SD2 | 240 | 0.0384155666 | 133.427437 |
| A+B+C | Overall | BackgroundOnly Gamma=0 | 240 | 0.0568490017 | 149.018226 |
| A+B+C | Overall | F1S1 Full Dual-Branch | 240 | 0.0467225114 | 138.162539 |
| A+B+C | Overall | F0S0 LatentOnly | 240 | 0.0484051395 | 141.236236 |
| A+B+C | Overall | F0S1 (Pixel Feature OFF; Support ON) | 240 | 0.0465207821 | 138.251575 |
| A+B+C | Overall | F1S0 (Pixel Feature ON; Support OFF) | 240 | 0.0477221366 | 140.771128 |
| B+C | Black | Raw SD2 | 80 | 0.0431993046 | 152.531956 |
| B+C | Black | BackgroundOnly Gamma=0 | 80 | 0.0834684092 | 193.255567 |
| B+C | Black | F1S1 Full Dual-Branch | 80 | 0.0604039292 | 170.796441 |
| B+C | Black | F0S0 LatentOnly | 80 | 0.0661469439 | 182.293431 |
| B+C | Black | F0S1 (Pixel Feature OFF; Support ON) | 80 | 0.0600207387 | 171.760536 |
| B+C | Black | F1S0 (Pixel Feature ON; Support OFF) | 80 | 0.0630556862 | 179.753136 |
| B+C | Overall | Raw SD2 | 160 | 0.0409881956 | 138.961555 |
| B+C | Overall | BackgroundOnly Gamma=0 | 160 | 0.0606951058 | 154.444034 |
| B+C | Overall | F1S1 Full Dual-Branch | 160 | 0.0510395576 | 143.488476 |
| B+C | Overall | F0S0 LatentOnly | 160 | 0.0521174777 | 147.567000 |
| B+C | Overall | F0S1 (Pixel Feature OFF; Support ON) | 160 | 0.0507123319 | 143.479382 |
| B+C | Overall | F1S0 (Pixel Feature ON; Support OFF) | 160 | 0.0505007110 | 146.249135 |

## B+C pooled Black findings

- Raw SD2: KID `0.0431993046`, FID `152.531956`.
- BackgroundOnly Gamma=0: KID `0.0834684092`, FID `193.255567`.
- F1S1 Full Dual-Branch: KID `0.0604039292`, FID `170.796441`.
- F0S0 LatentOnly: KID `0.0661469439`, FID `182.293431`.
- F0S1 (Pixel Feature OFF; Support ON): KID `0.0600207387`, FID `171.760536`.
- F1S0 (Pixel Feature ON; Support OFF): KID `0.0630556862`, FID `179.753136`.

Requested signed comparisons (first minus second; negative is better):

- f0s1-f1s1: KID `-0.0003831905`, FID `+0.964095`.
- f1s0-f1s1: KID `+0.0026517570`, FID `+8.956695`.
- f0s0-f0s1: KID `+0.0061262052`, FID `+10.532895`.
- f0s0-f1s0: KID `+0.0030912577`, FID `+2.540295`.
- f0s1-raw: KID `+0.0168214341`, FID `+19.228580`.
- f1s0-raw: KID `+0.0198563817`, FID `+27.221181`.

## Mechanism diagnostics

The larger mean Aligned Support area change relative to Full came from `S` (|ΔF|=0.00000000, |ΔS|=0.00036621).

The larger mean up-block residual L2 change relative to Full came from `S` (|ΔF|=0.39959231, |ΔS|=2.84181468).

Step/sample/block diagnostics are preserved in the three factorial diagnostic CSV files. Support means alone are not treated as quality evidence; conclusions use KID/FID together with bankwise consistency.

## Required questions

1. **Can F0S1 strictly retain Pixel Support?** Yes. Across all 600 diagnostic steps, the raw and explicit Pixel Support means matched exactly; the support was recomputed from the current Pixel Encoder output while the Fusion pixel half was exactly zero.

2. **Can F1S0 strictly retain Pixel features?** Yes. The Pixel Feature entering Fusion was nonzero at all 600 diagnostic steps and matched the raw Pixel Encoder L2, while the explicit Pixel Support contribution was exactly zero.

3. **Did F1S1 regress to historical Full?** Yes. The fixed Bank A/B/C PNG SHA256 checks all passed.

4. **Did F0S0 regress to historical LatentOnly?** Yes. The fixed Bank A/B/C PNG SHA256 checks all passed. All six strict regression checks passed.

5. **What changed after disabling Pixel features while retaining Support?** F0S1 improved over F0S0 on B+C Black by `-0.0061262052` KID and `-10.532895` FID. Against Full, it improved KID by `-0.0003831905` but worsened FID by `+0.964095`.

6. **What changed after retaining Pixel features while disabling Support?** F1S0 improved over F0S0 by `-0.0030912577` KID and `-2.540295` FID, but was worse than Full by `+0.0026517570` KID and `+8.956695` FID.

7. **Do F and S show conditional complementarity?** The B+C Black descriptive interaction is positive for KID (`0.0034744482`) and FID (`1.576200`), but the conditional Pixel Feature KID effect changes direction when S is enabled. This is non-additivity on the fixed set, not stable complementarity or statistical significance.

8. **Which path changes Aligned Support more?** S. Removing S reduced mean Aligned Support area from `0.0008544922` to `0.0004882812`; removing F while keeping S caused no mean area change.

9. **Which path changes final up-block residuals more?** S. Relative to Full, removing S changed mean residual L2 by `2.84181468`; removing F while retaining S changed it by `0.39959231`.

10. **Is there clear Support-range shrinkage?** Yes when S is off: mean Aligned Support area fell `42.86%`, and final Support area fell `41.94%`. No diagnostic step crossed the fixed abnormal-small threshold, so this is systematic shrinkage rather than collapse.

11. **Is Full still the best MSDF configuration?** It remains the strongest balanced configuration, but it is not the unanimous metric winner. F0S1 has slightly lower B+C Black KID (`0.0600207387` versus `0.0604039292`), while Full has lower FID (`170.796441` versus `171.760536`). Full wins both metrics on Bank B; Bank C is mixed.

12. **Did the gap to Raw SD2 shrink?** Mixed. F0S1 narrows the KID gap by `0.0003831905` relative to Full but widens the FID gap by `0.964095`. F0S1 still trails Raw by `38.94%` KID and `12.61%` FID on B+C Black.

13. **Is there enough evidence to design a new Fusion mechanism?** No. The direct Pixel Feature effect is small and inconsistent across banks/metrics. The explicit Support path accounts for the larger Support-area and residual changes.

14. **What single mechanism is most worth improving next?** The explicit Pixel Support composition/alignment mechanism, especially the fixed combination of fused support and pixel support before morphology. This recommendation is for a separate preregistered experiment; no follow-up was started.

## Research boundary

No model was retrained and no parameter, gamma, scheduler, seed, morphology algorithm, dynamic-gate formula, or residual injection formula changed. Banks share frozen slots; B/C were observed previously; V2 TRAIN references may overlap model training. This is not an independent test or a cross-dataset generalization claim.

The experiment stops here without BootstrapGuard, detector training, or another optimization round.
