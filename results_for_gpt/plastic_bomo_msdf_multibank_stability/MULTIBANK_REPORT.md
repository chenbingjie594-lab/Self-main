# Experiment 12 — MSDF BalancedRegion Multi-Bank Stability

Status: `MULTIBANK_MIXED`.

No model was trained, continued, recalibrated or structurally modified. Bank A was reused; only missing Bank B/C Flash gamma=0, BackgroundOnly Black and BalancedRegion Black images were generated from frozen slots and RNG manifests.

## Bankwise Black

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0379734447 | 161.766924 |
| A | Original Class-Adaptive MSDF | 0.0800462518 | 208.478914 |
| A | BackgroundOnly Hybrid | 0.0610055755 | 191.966915 |
| A | BalancedRegion Hybrid | 0.0593810876 | 187.683578 |
| B | Raw SD2 | 0.0451930130 | 172.446636 |
| B | Original Class-Adaptive MSDF | 0.0873685743 | 220.087779 |
| B | BackgroundOnly Hybrid | 0.0589119648 | 183.167155 |
| B | BalancedRegion Hybrid | 0.0629516938 | 185.781092 |
| C | Raw SD2 | 0.0396594895 | 163.039112 |
| C | Original Class-Adaptive MSDF | 0.0819643425 | 202.319080 |
| C | BackgroundOnly Hybrid | 0.0625615813 | 192.421727 |
| C | BalancedRegion Hybrid | 0.0678789201 | 189.759601 |

## Bankwise Flash

Original, BackgroundOnly and BalancedRegion hybrids share the exact same frozen Stage25 Flash gamma=0 images within each bank.

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0609828974 | 181.297468 |
| A | Original Class-Adaptive MSDF | 0.0501908097 | 161.149091 |
| A | BackgroundOnly Hybrid | 0.0501908097 | 161.149091 |
| A | BalancedRegion Hybrid | 0.0501908097 | 161.149091 |
| B | Raw SD2 | 0.0535829509 | 179.584994 |
| B | Original Class-Adaptive MSDF | 0.0609961308 | 168.576025 |
| B | BackgroundOnly Hybrid | 0.0609961308 | 168.576025 |
| B | BalancedRegion Hybrid | 0.0609961308 | 168.576025 |
| C | Raw SD2 | 0.0679392286 | 184.625930 |
| C | Original Class-Adaptive MSDF | 0.0571273848 | 167.556955 |
| C | BackgroundOnly Hybrid | 0.0571273848 | 167.556955 |
| C | BalancedRegion Hybrid | 0.0571273848 | 167.556955 |

## Bankwise Overall

| Bank | Method | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: |
| A | Raw SD2 | 0.0352225710 | 149.427503 |
| A | Original Class-Adaptive MSDF | 0.0498916065 | 165.131136 |
| A | BackgroundOnly Hybrid | 0.0429452473 | 154.905368 |
| A | BalancedRegion Hybrid | 0.0416453387 | 152.782300 |
| B | Raw SD2 | 0.0364288648 | 153.024285 |
| B | Original Class-Adaptive MSDF | 0.0567716790 | 173.683948 |
| B | BackgroundOnly Hybrid | 0.0474332942 | 155.388943 |
| B | BalancedRegion Hybrid | 0.0481079333 | 156.915121 |
| C | Raw SD2 | 0.0403267710 | 152.163042 |
| C | Original Class-Adaptive MSDF | 0.0532413289 | 165.013765 |
| C | BackgroundOnly Hybrid | 0.0487462165 | 158.366801 |
| C | BalancedRegion Hybrid | 0.0504812897 | 158.007584 |

## Pooled metrics

Pooled metrics were recomputed from all generated features and are not averages of bankwise KID/FID.

| Scope | Population | Method | Images | KID ↓ | FID ↓ |
| --- | --- | --- | ---: | ---: | ---: |
| A+B+C | Black | Raw SD2 | 120.0 | 0.0436378602 | 144.564373 |
| A+B+C | Flash | Raw SD2 | 120.0 | 0.0636730276 | 165.274682 |
| A+B+C | Overall | Raw SD2 | 240.0 | 0.0384155666 | 133.427437 |
| A+B+C | Black | Original Class-Adaptive MSDF | 120.0 | 0.0858180704 | 185.936021 |
| A+B+C | Flash | Original Class-Adaptive MSDF | 120.0 | 0.0594131306 | 151.375893 |
| A+B+C | Overall | Original Class-Adaptive MSDF | 240.0 | 0.0533934965 | 149.021969 |
| A+B+C | Black | BackgroundOnly Hybrid | 120.0 | 0.0626064865 | 165.678980 |
| A+B+C | Flash | BackgroundOnly Hybrid | 120.0 | 0.0594131306 | 151.375893 |
| A+B+C | Overall | BackgroundOnly Hybrid | 240.0 | 0.0467225114 | 138.162539 |
| A+B+C | Black | BalancedRegion Hybrid | 120.0 | 0.0653054536 | 165.012990 |
| A+B+C | Flash | BalancedRegion Hybrid | 120.0 | 0.0594131306 | 151.375893 |
| A+B+C | Overall | BalancedRegion Hybrid | 240.0 | 0.0469490693 | 137.789175 |
| B+C | Black | Raw SD2 | 80.0 | 0.0431993046 | 152.531956 |
| B+C | Flash | Raw SD2 | 80.0 | 0.0637731743 | 169.098334 |
| B+C | Overall | Raw SD2 | 160.0 | 0.0409881956 | 138.961555 |
| B+C | Black | Original Class-Adaptive MSDF | 80.0 | 0.0836452972 | 192.295072 |
| B+C | Flash | Original Class-Adaptive MSDF | 80.0 | 0.0622924784 | 157.426178 |
| B+C | Overall | Original Class-Adaptive MSDF | 160.0 | 0.0581209745 | 154.817505 |
| B+C | Black | BackgroundOnly Hybrid | 80.0 | 0.0604039292 | 170.796441 |
| B+C | Flash | BackgroundOnly Hybrid | 80.0 | 0.0622924784 | 157.426178 |
| B+C | Overall | BackgroundOnly Hybrid | 160.0 | 0.0510395576 | 143.488476 |
| B+C | Black | BalancedRegion Hybrid | 80.0 | 0.0646576990 | 171.537316 |
| B+C | Flash | BalancedRegion Hybrid | 80.0 | 0.0622924784 | 157.426178 |
| B+C | Overall | BalancedRegion Hybrid | 160.0 | 0.0521871660 | 144.149782 |

## Stability finding

Across held-out generation Banks B/C, BalancedRegion beat BackgroundOnly in Black KID on 0/2 banks, Black FID on 1/2, Overall KID on 0/2, and Overall FID on 1/2.

On A+B+C pooled features, BalancedRegion versus BackgroundOnly changed Black KID by +0.0026989671 (+4.31%) and Black FID by -0.665990 (-0.40%). Overall KID changed by +0.0002265579 (+0.48%) and Overall FID by -0.373364 (-0.27%).

When Bank A is excluded, B+C pooled BalancedRegion is worse than BackgroundOnly by +0.0042537698 Black KID, +0.740875 Black FID, +0.0011476084 Overall KID and +0.661306 Overall FID.

Against Raw SD2 on A+B+C pooled features, BalancedRegion remains higher by +0.0216675934 Black KID (+49.65%), +20.448617 Black FID (+14.14%), +0.0085335027 Overall KID (+22.21%) and +4.361739 Overall FID (+3.27%).

All absolute and relative deltas, including comparisons with Raw SD2, are recorded in `multibank_pairwise_deltas.csv`. A negative delta means the first method is lower and better.

## Recommendation

Do not promote BalancedRegion as a stable replacement or advance it alone to independent validation. Its Bank-A improvement did not reproduce in B/C: KID lost on both held-out banks, and B+C pooled Black/Overall KID and FID are all worse than BackgroundOnly. Preserve the frozen evidence and stop this Region-weight direction; a future authorized study should return to a training-level module ablation such as morphology guidance or dual-branch fusion before independent validation.

## Research boundary

A/B/C change only the generation seed and share the same slots. The real reference remains V2 TRAIN and may overlap generator training. Multi-bank stability is not cross-dataset generalization or independent-test validation.

KID subset standard deviation describes the internal 100-subset computation. It is not uncertainty across three banks and is not a 95% confidence interval.

The experiment stops here. No final_eval, loss search, recalibration, retraining, Gamma change, architecture change, BootstrapGuard or detector training was started.
