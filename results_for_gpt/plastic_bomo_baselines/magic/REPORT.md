# Plastic_Bomo MAGIC baseline

Status: `COMPLETE`.

The official MAGIC training and inference entry points were retained. Prompt Perturbation, Spatially Adaptive Guidance, and CAMA were enabled for all 240 outputs.

## Metrics

| Pool | Class | Images | KID ↓ | FID ↓ |
| --- | --- | ---: | ---: | ---: |
| A | Black | 40 | 0.2523645544 | 329.414385 |
| A | Flash | 40 | 0.1631907199 | 272.566443 |
| A | Overall | 80 | 0.1652379319 | 280.642884 |
| B | Black | 40 | 0.2616307631 | 328.541143 |
| B | Flash | 40 | 0.1597662904 | 267.639093 |
| B | Overall | 80 | 0.1675920306 | 277.672068 |
| C | Black | 40 | 0.2618999220 | 335.075069 |
| C | Flash | 40 | 0.1567609725 | 269.868120 |
| C | Overall | 80 | 0.1678457797 | 282.801947 |
| B+C | Black | 80 | 0.2605336887 | 322.691857 |
| B+C | Flash | 80 | 0.1624445441 | 259.766695 |
| B+C | Overall | 160 | 0.1707792171 | 271.684097 |
| A+B+C | Black | 120 | 0.2605942652 | 318.020929 |
| A+B+C | Flash | 120 | 0.1633206478 | 258.708110 |
| A+B+C | Overall | 240 | 0.1698128986 | 268.782973 |

## Data and CAMA limitations

The available annotations are BBoxes, not human pixel masks, so this is explicitly a BBox-Mask adaptation experiment.
U-2-Net on transparent film is out of domain. CAMA produced 786 valid records and 3,774 line/object-mask non-intersections; the official CAMA fallback remained enabled and CAMA was never bypassed.

## Structural diagnostics

- Black: background MAE=0.018629, target-BBox coverage=0.011123, structural failures=48/120.
- Flash: background MAE=0.018556, target-BBox coverage=0.016407, structural failures=51/120.
- Overall: background MAE=0.018592, target-BBox coverage=0.013765, structural failures=99/240.

Raw SD2 and Original Class-Adaptive MSDF comparison rows in `metrics.csv` are reused from the SHA-verified unified multibank protocol; pooled values are recomputed distributions, not bank averages.

No model weights are part of the GitHub publication set.
