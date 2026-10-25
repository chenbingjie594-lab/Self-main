# Experiment 17 — Confidence-Preserving Max-Support Fusion

Status: `MAX_SUPPORT_POOLED_IMPROVED_NOT_CONSISTENT`.

This frozen-checkpoint experiment changes only Reference Support composition from the mean to the pixelwise maximum. Morphology alignment uses Reference Support for source boxes and occupied-area placement, so downstream geometry and residuals may also change.

## Bankwise Black

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0379734447 | 161.766924 |
| A | Full Mean-Support | 0.0610055755 | 191.966915 |
| A | Full Max-Support | 0.0576759565 | 189.294246 |
| A | F0S1 (Pixel Feature OFF; Support ON) | 0.0618318202 | 195.641516 |
| A | F1S0 (Pixel Feature ON; Support OFF) | 0.0700751724 | 197.993778 |
| B | Raw SD2 | 0.0451930130 | 172.446636 |
| B | Full Mean-Support | 0.0589119648 | 183.167155 |
| B | Full Max-Support | 0.0597358768 | 179.521104 |
| B | F0S1 (Pixel Feature OFF; Support ON) | 0.0597471269 | 185.239525 |
| B | F1S0 (Pixel Feature ON; Support OFF) | 0.0620360040 | 201.988711 |
| C | Raw SD2 | 0.0396594895 | 163.039112 |
| C | Full Mean-Support | 0.0625615813 | 192.421727 |
| C | Full Max-Support | 0.0621582562 | 191.344810 |
| C | F0S1 (Pixel Feature OFF; Support ON) | 0.0614769226 | 193.289073 |
| C | F1S0 (Pixel Feature ON; Support OFF) | 0.0644988162 | 195.644068 |

## Bankwise Overall

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0352225710 | 149.427503 |
| A | Full Mean-Support | 0.0429452473 | 154.905368 |
| A | Full Max-Support | 0.0410493840 | 154.364903 |
| A | F0S1 (Pixel Feature OFF; Support ON) | 0.0427888648 | 156.286384 |
| A | F1S0 (Pixel Feature ON; Support OFF) | 0.0465731625 | 158.047760 |
| B | Raw SD2 | 0.0364288648 | 153.024285 |
| B | Full Mean-Support | 0.0474332942 | 155.388943 |
| B | Full Max-Support | 0.0472636843 | 155.609406 |
| B | F0S1 (Pixel Feature OFF; Support ON) | 0.0477423291 | 156.018769 |
| B | F1S0 (Pixel Feature ON; Support OFF) | 0.0481763959 | 162.645804 |
| C | Raw SD2 | 0.0403267710 | 152.163042 |
| C | Full Mean-Support | 0.0487462165 | 158.366801 |
| C | Full Max-Support | 0.0480114916 | 157.903399 |
| C | F0S1 (Pixel Feature OFF; Support ON) | 0.0480919716 | 158.302796 |
| C | F1S0 (Pixel Feature ON; Support OFF) | 0.0480630553 | 158.948700 |

## Pooled metrics

Pooled metrics were recomputed from concatenated features.

| Scope | Population | Method | Images | KID ↓ | FID ↓ |
| --- | --- | --- | ---: | ---: | ---: |
| A+B+C | Black | Raw SD2 | 120 | 0.0436378602 | 144.564373 |
| A+B+C | Black | Full Mean-Support | 120 | 0.0626064865 | 165.678980 |
| A+B+C | Black | Full Max-Support | 120 | 0.0612624700 | 162.531720 |
| A+B+C | Black | F0S1 (Pixel Feature OFF; Support ON) | 120 | 0.0622256151 | 166.769833 |
| A+B+C | Black | F1S0 (Pixel Feature ON; Support OFF) | 120 | 0.0669875753 | 173.324776 |
| A+B+C | Overall | Raw SD2 | 240 | 0.0384155666 | 133.427437 |
| A+B+C | Overall | Full Mean-Support | 240 | 0.0467225114 | 138.162539 |
| A+B+C | Overall | Full Max-Support | 240 | 0.0459628962 | 137.374563 |
| A+B+C | Overall | F0S1 (Pixel Feature OFF; Support ON) | 240 | 0.0465207821 | 138.251575 |
| A+B+C | Overall | F1S0 (Pixel Feature ON; Support OFF) | 240 | 0.0477221366 | 140.771128 |
| B+C | Black | Raw SD2 | 80 | 0.0431993046 | 152.531956 |
| B+C | Black | Full Mean-Support | 80 | 0.0604039292 | 170.796441 |
| B+C | Black | Full Max-Support | 80 | 0.0598580121 | 167.719746 |
| B+C | Black | F0S1 (Pixel Feature OFF; Support ON) | 80 | 0.0600207387 | 171.760536 |
| B+C | Black | F1S0 (Pixel Feature ON; Support OFF) | 80 | 0.0630556862 | 179.753136 |
| B+C | Overall | Raw SD2 | 160 | 0.0409881956 | 138.961555 |
| B+C | Overall | Full Mean-Support | 160 | 0.0510395576 | 143.488476 |
| B+C | Overall | Full Max-Support | 160 | 0.0503690454 | 142.892372 |
| B+C | Overall | F0S1 (Pixel Feature OFF; Support ON) | 160 | 0.0507123319 | 143.479382 |
| B+C | Overall | F1S0 (Pixel Feature ON; Support OFF) | 160 | 0.0505007110 | 146.249135 |

## Primary findings

B+C Black Max minus Mean: KID `-0.0005459171`, FID `-3.076695`.

A+B+C Black Max: KID `0.0612624700` versus Mean `0.0626064865`; FID `162.531720` versus Mean `165.678980`.

B+C Overall Max: KID `0.0503690454`, FID `142.892372`.

B+C Black Max remains above Raw by `+0.0166587076` KID and `+15.187790` FID.

## Mechanism diagnostics

Mean Reference Support mean/area: `0.00128929` / `0.00146484`; Max: `0.00135805` / `0.00146484`.

Mean Aligned/Final Support area: `0.00085449` / `0.00378418`; Max: `0.00085449` / `0.00378418`.

Mean/Max residual L2: `5.87042783` / `6.37550573`.

| Up-block | Mean residual L2 | Max residual L2 | Change |
| ---: | ---: | ---: | ---: |
| 0 | 0.00061051 | 0.00065908 | +7.96% |
| 1 | 4.28025241 | 4.85890462 | +13.52% |
| 2 | 13.81932727 | 14.80692183 | +7.15% |
| 3 | 5.38152112 | 5.83553737 | +8.44% |

Detailed per-bank, per-sample and per-step records plus mean/median/min/max summaries are retained in the diagnostic CSVs.

## Required answers

1. **Max-Support executed:** Yes. CPU checks verified exact `torch.maximum`, and formal generation completed 120/120 images.
2. **Mean-Support historical regression:** PASS, with exact image SHA256 agreement for the fixed A/B/C checks (3/3).
3. **Reference Support response:** Its mean rose from `0.0012892897` to `0.0013580550` (+5.33%), while area above 0.05 stayed unchanged.
4. **Source BBox:** No change; mean width/height remained `3.00` / `2.00` latent pixels.
5. **Occupied Ratio:** Mean increased from `0.4137591124` to `0.4424406886` (+6.93%); neither clamp bound triggered.
6. **Aligned Support:** Thresholded area stayed `0.0008544922`, but mean intensity increased from `0.0004020901` to `0.0004314199` (+7.29%).
7. **Up-block residuals:** Mean residual L2 increased in all four blocks; the table above reports each change. Tanh-near-limit fractions remained extremely small.
8. **Black KID:** B+C pooled improved by `-0.0005459171` (-0.90%). A and C improved; B worsened by `+0.0008239120`.
9. **Black FID:** B+C pooled improved by `-3.076695` (-1.80%). A, B and C all improved.
10. **B/C direction:** Not fully consistent. B Black KID worsened while B FID improved; both C metrics improved.
11. **Overall:** B+C pooled improved in KID by `-0.0006705121` and FID by `-0.596104`. Bank B Overall FID worsened slightly; the other bankwise comparisons improved.
12. **Gap to Raw SD2:** On B+C Black, Max remains worse by `+0.0166587076` KID and `+15.187790` FID. On B+C Overall, the gaps are `+0.0093808499` and `+3.930816`.
13. **Better formula:** Max-Support is better on the pooled development metrics, but the bankwise inconsistency prevents treating it as uniformly superior.
14. **Adaptive Support fusion:** There is a basis for a future bounded study because one global Max rule helps pooled metrics but does not help every bank and metric.
15. **Training-level validation:** Max-Support is worth including as a candidate in a controlled next-round comparison. It should not become the default before independent validation, and Raw SD2 still leads clearly.

## Research boundary

This is an inference-only intervention on a fixed trained model. It does not establish joint-training benefit, independent-test performance, support-head accuracy, statistical significance, or cross-class generalization. No follow-up experiment was started.
