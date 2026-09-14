# CDCC Pilot Diagnosis

Status: **CDCC_OFFLINE_FEASIBLE**

This is an offline coverage pilot. Cross-detector categories are descriptive only and do not estimate downstream training utility.

## Task-level weakness conclusion

Flash has no stable consensus-supported attribute bin; its scale/contrast/morphology rank-1 results are architecture-dependent. Black has two stable bins: `tiny` scale and `low` contrast. The old elongated target is not retained as a consensus constraint.

## Consensus weakness map

```json
{
  "flash": {
    "scale_bin": {
      "bins": [
        {
          "bin": "tiny",
          "n": 30,
          "recall_yolo": 0.7666666666666667,
          "recall_frcnn": 0.6333333333333333,
          "weakness_yolo": 0.23333333333333328,
          "weakness_frcnn": 0.3666666666666667,
          "consensus_weakness": 0.23333333333333328,
          "model_specific_gap": 0.13333333333333341,
          "rank": 1
        },
        {
          "bin": "large",
          "n": 29,
          "recall_yolo": 0.5517241379310345,
          "recall_frcnn": 0.9310344827586207,
          "weakness_yolo": 0.4482758620689655,
          "weakness_frcnn": 0.06896551724137934,
          "consensus_weakness": 0.06896551724137934,
          "model_specific_gap": 0.3793103448275862,
          "rank": 2
        },
        {
          "bin": "medium",
          "n": 29,
          "recall_yolo": 0.7586206896551724,
          "recall_frcnn": 0.9310344827586207,
          "weakness_yolo": 0.24137931034482762,
          "weakness_frcnn": 0.06896551724137934,
          "consensus_weakness": 0.06896551724137934,
          "model_specific_gap": 0.1724137931034483,
          "rank": 3
        }
      ],
      "rank1": "tiny",
      "architecture_dependent": true,
      "consensus_constraint_bin": null,
      "rule": "rank1 for both architectures and model_specific_gap <= consensus_weakness"
    },
    "contrast_bin": {
      "bins": [
        {
          "bin": "medium",
          "n": 29,
          "recall_yolo": 0.7586206896551724,
          "recall_frcnn": 0.7931034482758621,
          "weakness_yolo": 0.24137931034482762,
          "weakness_frcnn": 0.2068965517241379,
          "consensus_weakness": 0.2068965517241379,
          "model_specific_gap": 0.034482758620689724,
          "rank": 1
        },
        {
          "bin": "high",
          "n": 29,
          "recall_yolo": 0.6551724137931034,
          "recall_frcnn": 0.8275862068965517,
          "weakness_yolo": 0.3448275862068966,
          "weakness_frcnn": 0.1724137931034483,
          "consensus_weakness": 0.1724137931034483,
          "model_specific_gap": 0.1724137931034483,
          "rank": 2
        },
        {
          "bin": "low",
          "n": 30,
          "recall_yolo": 0.6666666666666666,
          "recall_frcnn": 0.8666666666666667,
          "weakness_yolo": 0.33333333333333337,
          "weakness_frcnn": 0.1333333333333333,
          "consensus_weakness": 0.1333333333333333,
          "model_specific_gap": 0.20000000000000007,
          "rank": 3
        }
      ],
      "rank1": "medium",
      "architecture_dependent": true,
      "consensus_constraint_bin": null,
      "rule": "rank1 for both architectures and model_specific_gap <= consensus_weakness"
    },
    "morphology_bin": {
      "bins": [
        {
          "bin": "medium",
          "n": 29,
          "recall_yolo": 0.7241379310344828,
          "recall_frcnn": 0.7586206896551724,
          "weakness_yolo": 0.27586206896551724,
          "weakness_frcnn": 0.24137931034482762,
          "consensus_weakness": 0.24137931034482762,
          "model_specific_gap": 0.03448275862068961,
          "rank": 1
        },
        {
          "bin": "elongated",
          "n": 29,
          "recall_yolo": 0.7241379310344828,
          "recall_frcnn": 0.8275862068965517,
          "weakness_yolo": 0.27586206896551724,
          "weakness_frcnn": 0.1724137931034483,
          "consensus_weakness": 0.1724137931034483,
          "model_specific_gap": 0.10344827586206895,
          "rank": 2
        },
        {
          "bin": "compact",
          "n": 30,
          "recall_yolo": 0.6333333333333333,
          "recall_frcnn": 0.9,
          "weakness_yolo": 0.3666666666666667,
          "weakness_frcnn": 0.09999999999999998,
          "consensus_weakness": 0.09999999999999998,
          "model_specific_gap": 0.2666666666666667,
          "rank": 3
        }
      ],
      "rank1": "medium",
      "architecture_dependent": true,
      "consensus_constraint_bin": null,
      "rule": "rank1 for both architectures and model_specific_gap <= consensus_weakness"
    }
  },
  "black": {
    "scale_bin": {
      "bins": [
        {
          "bin": "tiny",
          "n": 27,
          "recall_yolo": 0.7037037037037037,
          "recall_frcnn": 0.8148148148148148,
          "weakness_yolo": 0.2962962962962963,
          "weakness_frcnn": 0.18518518518518523,
          "consensus_weakness": 0.18518518518518523,
          "model_specific_gap": 0.11111111111111105,
          "rank": 1
        },
        {
          "bin": "large",
          "n": 27,
          "recall_yolo": 0.8888888888888888,
          "recall_frcnn": 0.9259259259259259,
          "weakness_yolo": 0.11111111111111116,
          "weakness_frcnn": 0.07407407407407407,
          "consensus_weakness": 0.07407407407407407,
          "model_specific_gap": 0.03703703703703709,
          "rank": 2
        },
        {
          "bin": "medium",
          "n": 26,
          "recall_yolo": 0.9230769230769231,
          "recall_frcnn": 1.0,
          "weakness_yolo": 0.07692307692307687,
          "weakness_frcnn": 0.0,
          "consensus_weakness": 0.0,
          "model_specific_gap": 0.07692307692307687,
          "rank": 3
        }
      ],
      "rank1": "tiny",
      "architecture_dependent": false,
      "consensus_constraint_bin": "tiny",
      "rule": "rank1 for both architectures and model_specific_gap <= consensus_weakness"
    },
    "contrast_bin": {
      "bins": [
        {
          "bin": "low",
          "n": 27,
          "recall_yolo": 0.6666666666666666,
          "recall_frcnn": 0.8148148148148148,
          "weakness_yolo": 0.33333333333333337,
          "weakness_frcnn": 0.18518518518518523,
          "consensus_weakness": 0.18518518518518523,
          "model_specific_gap": 0.14814814814814814,
          "rank": 1
        },
        {
          "bin": "medium",
          "n": 26,
          "recall_yolo": 0.8846153846153846,
          "recall_frcnn": 0.9615384615384616,
          "weakness_yolo": 0.11538461538461542,
          "weakness_frcnn": 0.038461538461538436,
          "consensus_weakness": 0.038461538461538436,
          "model_specific_gap": 0.07692307692307698,
          "rank": 2
        },
        {
          "bin": "high",
          "n": 27,
          "recall_yolo": 0.9629629629629629,
          "recall_frcnn": 0.9629629629629629,
          "weakness_yolo": 0.03703703703703709,
          "weakness_frcnn": 0.03703703703703709,
          "consensus_weakness": 0.03703703703703709,
          "model_specific_gap": 0.0,
          "rank": 3
        }
      ],
      "rank1": "low",
      "architecture_dependent": false,
      "consensus_constraint_bin": "low",
      "rule": "rank1 for both architectures and model_specific_gap <= consensus_weakness"
    },
    "morphology_bin": {
      "bins": [
        {
          "bin": "compact",
          "n": 27,
          "recall_yolo": 0.8888888888888888,
          "recall_frcnn": 0.8888888888888888,
          "weakness_yolo": 0.11111111111111116,
          "weakness_frcnn": 0.11111111111111116,
          "consensus_weakness": 0.11111111111111116,
          "model_specific_gap": 0.0,
          "rank": 1
        },
        {
          "bin": "elongated",
          "n": 27,
          "recall_yolo": 0.7407407407407407,
          "recall_frcnn": 0.8888888888888888,
          "weakness_yolo": 0.2592592592592593,
          "weakness_frcnn": 0.11111111111111116,
          "consensus_weakness": 0.11111111111111116,
          "model_specific_gap": 0.14814814814814814,
          "rank": 2
        },
        {
          "bin": "medium",
          "n": 26,
          "recall_yolo": 0.8846153846153846,
          "recall_frcnn": 0.9615384615384616,
          "weakness_yolo": 0.11538461538461542,
          "weakness_frcnn": 0.038461538461538436,
          "consensus_weakness": 0.038461538461538436,
          "model_specific_gap": 0.07692307692307698,
          "rank": 3
        }
      ],
      "rank1": "compact",
      "architecture_dependent": true,
      "consensus_constraint_bin": null,
      "rule": "rank1 for both architectures and model_specific_gap <= consensus_weakness"
    }
  }
}
```

## Bootstrap q05/q95

```json
{
  "flash": {
    "scale_bin:large": {
      "q05": 0.2,
      "q95": 0.45
    },
    "scale_bin:medium": {
      "q05": 0.2,
      "q95": 0.45
    },
    "scale_bin:tiny": {
      "q05": 0.225,
      "q95": 0.475
    },
    "contrast_bin:high": {
      "q05": 0.2,
      "q95": 0.45
    },
    "contrast_bin:low": {
      "q05": 0.225,
      "q95": 0.475
    },
    "contrast_bin:medium": {
      "q05": 0.2,
      "q95": 0.45
    },
    "morphology_bin:compact": {
      "q05": 0.225,
      "q95": 0.475
    },
    "morphology_bin:elongated": {
      "q05": 0.2,
      "q95": 0.45
    },
    "morphology_bin:medium": {
      "q05": 0.2,
      "q95": 0.45
    },
    "weakness_count:0": {
      "q05": 1.0,
      "q95": 1.0
    },
    "weakness_count:1": {
      "q05": 0.0,
      "q95": 0.0
    },
    "weakness_count:2": {
      "q05": 0.0,
      "q95": 0.0
    },
    "weakness_count:3": {
      "q05": 0.0,
      "q95": 0.0
    },
    "weakness_count:2+": {
      "q05": 0.0,
      "q95": 0.0
    },
    "weakness_count:3-way": {
      "q05": 0.0,
      "q95": 0.0
    }
  },
  "black": {
    "scale_bin:large": {
      "q05": 0.225,
      "q95": 0.45
    },
    "scale_bin:medium": {
      "q05": 0.2,
      "q95": 0.45
    },
    "scale_bin:tiny": {
      "q05": 0.225,
      "q95": 0.45
    },
    "contrast_bin:high": {
      "q05": 0.225,
      "q95": 0.45
    },
    "contrast_bin:low": {
      "q05": 0.225,
      "q95": 0.45
    },
    "contrast_bin:medium": {
      "q05": 0.2,
      "q95": 0.45
    },
    "morphology_bin:compact": {
      "q05": 0.225,
      "q95": 0.45
    },
    "morphology_bin:elongated": {
      "q05": 0.225,
      "q95": 0.475
    },
    "morphology_bin:medium": {
      "q05": 0.2,
      "q95": 0.45
    },
    "weakness_count:0": {
      "q05": 0.325,
      "q95": 0.6
    },
    "weakness_count:1": {
      "q05": 0.275,
      "q95": 0.525
    },
    "weakness_count:2": {
      "q05": 0.05,
      "q95": 0.225
    },
    "weakness_count:3": {
      "q05": 0.0,
      "q95": 0.0
    },
    "weakness_count:2+": {
      "q05": 0.05,
      "q95": 0.225
    },
    "weakness_count:3-way": {
      "q05": 0.0,
      "q95": 0.0
    }
  }
}
```

## Random / DWBG / CDCC distributions

```json
{
  "flash": {
    "random": {
      "n": 40,
      "scale_bin": {
        "large": 0.375,
        "medium": 0.275,
        "tiny": 0.35
      },
      "contrast_bin": {
        "high": 0.325,
        "low": 0.35,
        "medium": 0.325
      },
      "morphology_bin": {
        "compact": 0.175,
        "elongated": 0.65,
        "medium": 0.175
      },
      "weakness_count": {
        "0": 1.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0,
        "2+": 0.0,
        "3-way": 0.0
      }
    },
    "dwbg": {
      "n": 40,
      "scale_bin": {
        "large": 0.325,
        "medium": 0.325,
        "tiny": 0.35
      },
      "contrast_bin": {
        "high": 0.275,
        "low": 0.475,
        "medium": 0.25
      },
      "morphology_bin": {
        "compact": 0.275,
        "elongated": 0.525,
        "medium": 0.2
      },
      "weakness_count": {
        "0": 1.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0,
        "2+": 0.0,
        "3-way": 0.0
      }
    },
    "cdcc": {
      "n": 40,
      "scale_bin": {
        "large": 0.325,
        "medium": 0.325,
        "tiny": 0.35
      },
      "contrast_bin": {
        "high": 0.325,
        "low": 0.35,
        "medium": 0.325
      },
      "morphology_bin": {
        "compact": 0.35,
        "elongated": 0.325,
        "medium": 0.325
      },
      "weakness_count": {
        "0": 1.0,
        "1": 0.0,
        "2": 0.0,
        "3": 0.0,
        "2+": 0.0,
        "3-way": 0.0
      }
    }
  },
  "black": {
    "random": {
      "n": 40,
      "scale_bin": {
        "large": 0.175,
        "medium": 0.375,
        "tiny": 0.45
      },
      "contrast_bin": {
        "high": 0.5,
        "low": 0.175,
        "medium": 0.325
      },
      "morphology_bin": {
        "compact": 0.175,
        "elongated": 0.7,
        "medium": 0.125
      },
      "weakness_count": {
        "0": 0.425,
        "1": 0.525,
        "2": 0.05,
        "3": 0.0,
        "2+": 0.05,
        "3-way": 0.0
      }
    },
    "dwbg": {
      "n": 40,
      "scale_bin": {
        "large": 0.15,
        "medium": 0.375,
        "tiny": 0.475
      },
      "contrast_bin": {
        "high": 0.25,
        "low": 0.5,
        "medium": 0.25
      },
      "morphology_bin": {
        "compact": 0.075,
        "elongated": 0.775,
        "medium": 0.15
      },
      "weakness_count": {
        "0": 0.225,
        "1": 0.575,
        "2": 0.2,
        "3": 0.0,
        "2+": 0.2,
        "3-way": 0.0
      }
    },
    "cdcc": {
      "n": 40,
      "scale_bin": {
        "large": 0.325,
        "medium": 0.35,
        "tiny": 0.325
      },
      "contrast_bin": {
        "high": 0.35,
        "low": 0.325,
        "medium": 0.325
      },
      "morphology_bin": {
        "compact": 0.3,
        "elongated": 0.375,
        "medium": 0.325
      },
      "weakness_count": {
        "0": 0.475,
        "1": 0.4,
        "2": 0.125,
        "3": 0.0,
        "2+": 0.125,
        "3-way": 0.0
      }
    }
  }
}
```

## Out-of-range distributions

```json
{
  "flash": {
    "random": {
      "morphology_bin:compact": {
        "value": 0.175,
        "q05": 0.225,
        "q95": 0.475,
        "inside": false
      },
      "morphology_bin:elongated": {
        "value": 0.65,
        "q05": 0.2,
        "q95": 0.45,
        "inside": false
      },
      "morphology_bin:medium": {
        "value": 0.175,
        "q05": 0.2,
        "q95": 0.45,
        "inside": false
      }
    },
    "dwbg": {
      "morphology_bin:elongated": {
        "value": 0.525,
        "q05": 0.2,
        "q95": 0.45,
        "inside": false
      }
    },
    "cdcc": {}
  },
  "black": {
    "random": {
      "scale_bin:large": {
        "value": 0.175,
        "q05": 0.225,
        "q95": 0.45,
        "inside": false
      },
      "contrast_bin:high": {
        "value": 0.5,
        "q05": 0.225,
        "q95": 0.45,
        "inside": false
      },
      "contrast_bin:low": {
        "value": 0.175,
        "q05": 0.225,
        "q95": 0.45,
        "inside": false
      },
      "morphology_bin:compact": {
        "value": 0.175,
        "q05": 0.225,
        "q95": 0.45,
        "inside": false
      },
      "morphology_bin:elongated": {
        "value": 0.7,
        "q05": 0.225,
        "q95": 0.475,
        "inside": false
      },
      "morphology_bin:medium": {
        "value": 0.125,
        "q05": 0.2,
        "q95": 0.45,
        "inside": false
      }
    },
    "dwbg": {
      "scale_bin:large": {
        "value": 0.15,
        "q05": 0.225,
        "q95": 0.45,
        "inside": false
      },
      "scale_bin:tiny": {
        "value": 0.475,
        "q05": 0.225,
        "q95": 0.45,
        "inside": false
      },
      "contrast_bin:low": {
        "value": 0.5,
        "q05": 0.225,
        "q95": 0.45,
        "inside": false
      },
      "morphology_bin:compact": {
        "value": 0.075,
        "q05": 0.225,
        "q95": 0.45,
        "inside": false
      },
      "morphology_bin:elongated": {
        "value": 0.775,
        "q05": 0.225,
        "q95": 0.475,
        "inside": false
      },
      "morphology_bin:medium": {
        "value": 0.15,
        "q05": 0.2,
        "q95": 0.45,
        "inside": false
      },
      "weakness_count:0": {
        "value": 0.225,
        "q05": 0.325,
        "q95": 0.6,
        "inside": false
      },
      "weakness_count:1": {
        "value": 0.575,
        "q05": 0.275,
        "q95": 0.525,
        "inside": false
      }
    },
    "cdcc": {}
  }
}
```

## Cross-detector view categories

```json
{
  "flash": {
    "random": {
      "available": 40,
      "missing": 0,
      "ratios": {
        "CONSENSUS_HARD": 0.8,
        "YOLO_SPECIFIC_HARD": 0.05,
        "FRCNN_SPECIFIC_HARD": 0.15,
        "CONSENSUS_EASY": 0.0
      }
    },
    "dwbg": {
      "available": 40,
      "missing": 0,
      "ratios": {
        "CONSENSUS_HARD": 0.575,
        "YOLO_SPECIFIC_HARD": 0.025,
        "FRCNN_SPECIFIC_HARD": 0.4,
        "CONSENSUS_EASY": 0.0
      }
    },
    "cdcc": {
      "available": 40,
      "missing": 0,
      "ratios": {
        "CONSENSUS_HARD": 0.9,
        "YOLO_SPECIFIC_HARD": 0.025,
        "FRCNN_SPECIFIC_HARD": 0.075,
        "CONSENSUS_EASY": 0.0
      }
    }
  },
  "black": {
    "random": {
      "available": 40,
      "missing": 0,
      "ratios": {
        "CONSENSUS_HARD": 0.25,
        "YOLO_SPECIFIC_HARD": 0.125,
        "FRCNN_SPECIFIC_HARD": 0.025,
        "CONSENSUS_EASY": 0.6
      }
    },
    "dwbg": {
      "available": 40,
      "missing": 0,
      "ratios": {
        "CONSENSUS_HARD": 0.2,
        "YOLO_SPECIFIC_HARD": 0.175,
        "FRCNN_SPECIFIC_HARD": 0.075,
        "CONSENSUS_EASY": 0.55
      }
    },
    "cdcc": {
      "available": 40,
      "missing": 0,
      "ratios": {
        "CONSENSUS_HARD": 0.3,
        "YOLO_SPECIFIC_HARD": 0.15,
        "FRCNN_SPECIFIC_HARD": 0.0,
        "CONSENSUS_EASY": 0.55
      }
    }
  }
}
```

## Overlap

```json
{
  "flash": {
    "cdcc_vs_random": {
      "count": 5,
      "jaccard": 0.06666666666666667,
      "candidate_ids": [
        "dwbg_0_19997_69_000470",
        "dwbg_0_24571_0_000503",
        "dwbg_0_24571_74_000557",
        "dwbg_0_30103_36_000612",
        "dwbg_0_42_13_000003"
      ]
    },
    "cdcc_vs_dwbg": {
      "count": 6,
      "jaccard": 0.08108108108108109,
      "candidate_ids": [
        "dwbg_0_12011_11_000002_c12012",
        "dwbg_0_12011_17_000262",
        "dwbg_0_12011_64_000298",
        "dwbg_0_19997_2_000425",
        "dwbg_0_24571_8_000559",
        "dwbg_0_3407_17_000091"
      ]
    }
  },
  "black": {
    "cdcc_vs_random": {
      "count": 9,
      "jaccard": 0.1267605633802817,
      "candidate_ids": [
        "dwbg_1_19997_7_000500",
        "dwbg_1_24571_12_000563",
        "dwbg_1_3407_23_000155",
        "dwbg_1_42_13_000064",
        "dwbg_1_42_23_000070",
        "dwbg_1_42_29_000074",
        "dwbg_1_42_4_000080",
        "dwbg_1_7859_28_000246",
        "dwbg_1_7859_8_000253"
      ]
    },
    "cdcc_vs_dwbg": {
      "count": 13,
      "jaccard": 0.19402985074626866,
      "candidate_ids": [
        "dwbg_1_19997_5_000499",
        "dwbg_1_19997_7_000500",
        "dwbg_1_24571_18_000568",
        "dwbg_1_24571_24_000574",
        "dwbg_1_24571_32_000580",
        "dwbg_1_3407_19_000151",
        "dwbg_1_3407_23_000155",
        "dwbg_1_42_23_000070",
        "dwbg_1_42_29_000074",
        "dwbg_1_42_2_000068",
        "dwbg_1_42_34_000078",
        "dwbg_1_42_35_000079",
        "dwbg_1_7859_16_000237"
      ]
    }
  }
}
```
