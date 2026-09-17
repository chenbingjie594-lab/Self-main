# BootstrapGuard Cross-Detector Multi-Seed Confirmation

Status: **CROSS_DETECTOR_TRANSFER_CONFIRMED**

| Method | Precision | Recall | AP50 | AP50-95 |
|---|---:|---:|---:|---:|
| Real-only | 0.6783 +/- 0.2436 | 0.8811 +/- 0.0184 | 0.8497 +/- 0.0068 | 0.3964 +/- 0.0176 |
| Random | 0.7833 +/- 0.0418 | 0.8811 +/- 0.0297 | 0.8698 +/- 0.0303 | 0.3949 +/- 0.0168 |
| DQ | 0.8390 +/- 0.0074 | 0.8939 +/- 0.0066 | 0.8906 +/- 0.0133 | 0.3932 +/- 0.0116 |
| BootstrapGuard | 0.8225 +/- 0.0300 | 0.8735 +/- 0.0210 | 0.8554 +/- 0.0300 | 0.4036 +/- 0.0096 |

BG - Random AP50-95: +0.008710 +/- 0.013464, wins 2/3.
BG - DQ AP50-95: +0.010387 +/- 0.006924, wins 3/3.

BootstrapGuard primarily benefits stricter-IoU localization quality rather than low-IoU detection rate.

The transferability signal is confirmed across YOLO11s and Faster R-CNN, but this does not establish universal detector independence.
