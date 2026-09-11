# Multidetector Synthetic View Diagnosis

Status: **MULTIVIEW_SYNTHETIC_SIGNAL_PARTIAL**

Pilot records: 223. Categories are based on architecture median IoU and detector-specific empirical-CDF confidence percentile. Raw confidence is never averaged across detectors. This is a diagnosis only; no final selection or downstream training was performed.

## Category ratios

```json
{
  "0": {
    "selected_dwbg": {
      "CONSENSUS_HARD": 0.575,
      "YOLO_SPECIFIC_HARD": 0.025,
      "FRCNN_SPECIFIC_HARD": 0.4,
      "CONSENSUS_EASY": 0.0
    },
    "selected_random": {
      "CONSENSUS_HARD": 0.8,
      "YOLO_SPECIFIC_HARD": 0.05,
      "FRCNN_SPECIFIC_HARD": 0.15,
      "CONSENSUS_EASY": 0.0
    },
    "additional": {
      "CONSENSUS_HARD": 0.825,
      "YOLO_SPECIFIC_HARD": 0.0,
      "FRCNN_SPECIFIC_HARD": 0.125,
      "CONSENSUS_EASY": 0.05
    }
  },
  "1": {
    "selected_dwbg": {
      "CONSENSUS_HARD": 0.2,
      "YOLO_SPECIFIC_HARD": 0.175,
      "FRCNN_SPECIFIC_HARD": 0.075,
      "CONSENSUS_EASY": 0.55
    },
    "selected_random": {
      "CONSENSUS_HARD": 0.25,
      "YOLO_SPECIFIC_HARD": 0.125,
      "FRCNN_SPECIFIC_HARD": 0.025,
      "CONSENSUS_EASY": 0.6
    },
    "additional": {
      "CONSENSUS_HARD": 0.425,
      "YOLO_SPECIFIC_HARD": 0.275,
      "FRCNN_SPECIFIC_HARD": 0.05,
      "CONSENSUS_EASY": 0.25
    }
  }
}
```

## Real distribution

```json
{
  "0": {
    "CONSENSUS_HARD": 0.136364,
    "YOLO_SPECIFIC_HARD": 0.136364,
    "FRCNN_SPECIFIC_HARD": 0.159091,
    "CONSENSUS_EASY": 0.568182
  },
  "1": {
    "CONSENSUS_HARD": 0.15,
    "YOLO_SPECIFIC_HARD": 0.125,
    "FRCNN_SPECIFIC_HARD": 0.1125,
    "CONSENSUS_EASY": 0.6125
  }
}
```

## Fold vote counts

```json
{
  "yolo": {
    "0": 38,
    "1": 40,
    "2": 61,
    "3": 84
  },
  "frcnn": {
    "0": 40,
    "1": 33,
    "2": 26,
    "3": 124
  }
}
```
