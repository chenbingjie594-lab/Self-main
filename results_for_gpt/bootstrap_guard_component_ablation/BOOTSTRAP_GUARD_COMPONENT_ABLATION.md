# BootstrapGuard Component Ablation

Status: **JOINT_GUARD_DOMINANT**

| Level | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| DiversityQuality | 0.8724¡À0.0753 | 0.8818¡À0.0411 | 0.8828¡À0.0264 | 0.4186¡À0.0131 |
| MarginalGuard | 0.9005¡À0.0311 | 0.8550¡À0.0508 | 0.8766¡À0.0241 | 0.4147¡À0.0108 |
| FullGuard | 0.9041¡À0.0098 | 0.8740¡À0.0392 | 0.8746¡À0.0160 | 0.4406¡À0.0012 |

## Paired FullGuard - MarginalGuard
```json
{
  "precision": {
    "by_seed": {
      "42": 0.045717708763667875,
      "3407": -0.0008203473794702321,
      "2026": -0.0340398097642981
    },
    "mean_delta": 0.003619183873299847,
    "std_delta": 0.04006366831360991,
    "wins": 1
  },
  "recall": {
    "by_seed": {
      "42": 0.11071201931505614,
      "3407": 0.008842156486355779,
      "2026": -0.0625938565420836
    },
    "mean_delta": 0.01898677308644277,
    "std_delta": 0.08709716749771289,
    "wins": 2
  },
  "map50": {
    "by_seed": {
      "42": 0.02995503136631794,
      "3407": -0.006684355705098954,
      "2026": -0.02935616632500293
    },
    "mean_delta": -0.002028496887927981,
    "std_delta": 0.029928453165655053,
    "wins": 1
  },
  "map5095": {
    "by_seed": {
      "42": 0.03243028395843206,
      "3407": 0.032011495924804734,
      "2026": 0.013383485987267463
    },
    "mean_delta": 0.025941755290168084,
    "std_delta": 0.01087779581383147,
    "wins": 3
  }
}
```

## Paired MarginalGuard - DiversityQuality
```json
{
  "precision": {
    "by_seed": {
      "42": -0.005695963097062773,
      "3407": -0.04008619368424371,
      "2026": 0.12996635974147852
    },
    "mean_delta": 0.028061400986724012,
    "std_delta": 0.08991183155773234,
    "wins": 1
  },
  "recall": {
    "by_seed": {
      "42": -0.07812352518820054,
      "3407": -0.07272727272727275,
      "2026": 0.07045454545454555
    },
    "mean_delta": -0.026798750820309247,
    "std_delta": 0.08426703158649314,
    "wins": 1
  },
  "map50": {
    "by_seed": {
      "42": -0.0038091402037045974,
      "3407": -0.04786810888877191,
      "2026": 0.03310480602975763
    },
    "mean_delta": -0.0061908143542396266,
    "std_delta": 0.04053896294209252,
    "wins": 1
  },
  "map5095": {
    "by_seed": {
      "42": -0.007939101048000885,
      "3407": -0.023179893935136253,
      "2026": 0.019424361264878676
    },
    "mean_delta": -0.0038982112394194877,
    "std_delta": 0.021587663910688032,
    "wins": 1
  }
}
```

## Per-class
```json
{
  "flash": {
    "diversity_quality": {
      "ap50": {
        "mean": 0.9172948815889993,
        "std": 0.033989654524151564
      },
      "ap5095": {
        "mean": 0.4046320164419847,
        "std": 0.01516784541961942
      },
      "recall": {
        "mean": 0.9833333333333334,
        "std": 0.028867513459481315
      }
    },
    "marginal_guard": {
      "ap50": {
        "mean": 0.8904408119113999,
        "std": 0.03983716655267104
      },
      "ap5095": {
        "mean": 0.40755927836728906,
        "std": 0.028125842806223825
      },
      "recall": {
        "mean": 0.9166666666666666,
        "std": 0.07637626158259735
      }
    },
    "bootstrap_guard": {
      "ap50": {
        "mean": 0.9046267176904431,
        "std": 0.018679808939272305
      },
      "ap5095": {
        "mean": 0.4186485276616521,
        "std": 0.01366361322914947
      },
      "recall": {
        "mean": 0.9449821051383277,
        "std": 0.0498850864226261
      }
    }
  },
  "black": {
    "diversity_quality": {
      "ap50": {
        "mean": 0.8482995915008734,
        "std": 0.018862888688388965
      },
      "ap5095": {
        "mean": 0.43247929708763544,
        "std": 0.031019954635523032
      },
      "recall": {
        "mean": 0.7803030303030303,
        "std": 0.05719571541871781
      }
    },
    "marginal_guard": {
      "ap50": {
        "mean": 0.8627720324699936,
        "std": 0.008651060093344409
      },
      "ap5095": {
        "mean": 0.4217556126834919,
        "std": 0.008931159453717566
      },
      "recall": {
        "mean": 0.7933721953290784,
        "std": 0.025913623625429393
      }
    },
    "bootstrap_guard": {
      "ap50": {
        "mean": 0.8445291329150946,
        "std": 0.02393000266518317
      },
      "ap5095": {
        "mean": 0.46254987396946506,
        "std": 0.014562049459360153
      },
      "recall": {
        "mean": 0.8030303030303031,
        "std": 0.03471648253754427
      }
    }
  }
}
```

## Offline candidate overlap
```json
{
  "flash": {
    "count": 6,
    "jaccard": 0.08108108108108109,
    "candidate_ids": [
      "dwbg_0_12011_64_000298",
      "dwbg_0_15683_44_000364",
      "dwbg_0_24571_74_000557",
      "dwbg_0_24571_8_000559",
      "dwbg_0_30103_62_000636",
      "dwbg_0_7859_75_000231"
    ]
  },
  "black": {
    "count": 22,
    "jaccard": 0.3793103448275862,
    "candidate_ids": [
      "dwbg_1_15683_20_000401",
      "dwbg_1_15683_22_000403",
      "dwbg_1_19997_15_000481",
      "dwbg_1_19997_22_000486",
      "dwbg_1_19997_32_000495",
      "dwbg_1_19997_35_000498",
      "dwbg_1_19997_7_000500",
      "dwbg_1_19997_8_000501",
      "dwbg_1_24571_18_000568",
      "dwbg_1_24571_24_000574",
      "dwbg_1_24571_30_000578",
      "dwbg_1_24571_32_000580",
      "dwbg_1_3407_19_000151",
      "dwbg_1_3407_31_000160",
      "dwbg_1_3407_34_000163",
      "dwbg_1_42_23_000070",
      "dwbg_1_42_29_000074",
      "dwbg_1_42_2_000068",
      "dwbg_1_7859_16_000237",
      "dwbg_1_7859_25_000243",
      "dwbg_1_7859_28_000246",
      "dwbg_1_7859_8_000253"
    ]
  }
}
```

## Offline composition
```json
{
  "flash": {
    "diversity_quality": {
      "distribution": {
        "n": 40,
        "scale_bin": {
          "large": 0.525,
          "medium": 0.2,
          "tiny": 0.275
        },
        "contrast_bin": {
          "high": 0.35,
          "low": 0.375,
          "medium": 0.275
        },
        "morphology_bin": {
          "compact": 0.225,
          "elongated": 0.525,
          "medium": 0.25
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
      "all_out_of_range": {
        "scale_bin:large": {
          "value": 0.525,
          "q05": 0.2,
          "q95": 0.45,
          "inside": false
        },
        "morphology_bin:elongated": {
          "value": 0.525,
          "q05": 0.2,
          "q95": 0.45,
          "inside": false
        }
      },
      "marginal_out_of_range": {
        "scale_bin:large": {
          "value": 0.525,
          "q05": 0.2,
          "q95": 0.45,
          "inside": false
        },
        "morphology_bin:elongated": {
          "value": 0.525,
          "q05": 0.2,
          "q95": 0.45,
          "inside": false
        }
      }
    },
    "marginal_guard": {
      "distribution": {
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
      },
      "all_out_of_range": {},
      "marginal_out_of_range": {}
    },
    "full_guard": {
      "distribution": {
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
      },
      "all_out_of_range": {},
      "marginal_out_of_range": {}
    }
  },
  "black": {
    "diversity_quality": {
      "distribution": {
        "n": 40,
        "scale_bin": {
          "large": 0.3,
          "medium": 0.375,
          "tiny": 0.325
        },
        "contrast_bin": {
          "high": 0.55,
          "low": 0.05,
          "medium": 0.4
        },
        "morphology_bin": {
          "compact": 0.15,
          "elongated": 0.6,
          "medium": 0.25
        },
        "weakness_count": {
          "0": 0.675,
          "1": 0.275,
          "2": 0.05,
          "3": 0.0,
          "2+": 0.05,
          "3-way": 0.0
        }
      },
      "all_out_of_range": {
        "contrast_bin:high": {
          "value": 0.55,
          "q05": 0.225,
          "q95": 0.45,
          "inside": false
        },
        "contrast_bin:low": {
          "value": 0.05,
          "q05": 0.225,
          "q95": 0.45,
          "inside": false
        },
        "morphology_bin:compact": {
          "value": 0.15,
          "q05": 0.225,
          "q95": 0.45,
          "inside": false
        },
        "morphology_bin:elongated": {
          "value": 0.6,
          "q05": 0.225,
          "q95": 0.475,
          "inside": false
        },
        "weakness_count:0": {
          "value": 0.675,
          "q05": 0.325,
          "q95": 0.6,
          "inside": false
        }
      },
      "marginal_out_of_range": {
        "contrast_bin:high": {
          "value": 0.55,
          "q05": 0.225,
          "q95": 0.45,
          "inside": false
        },
        "contrast_bin:low": {
          "value": 0.05,
          "q05": 0.225,
          "q95": 0.45,
          "inside": false
        },
        "morphology_bin:compact": {
          "value": 0.15,
          "q05": 0.225,
          "q95": 0.45,
          "inside": false
        },
        "morphology_bin:elongated": {
          "value": 0.6,
          "q05": 0.225,
          "q95": 0.475,
          "inside": false
        }
      }
    },
    "marginal_guard": {
      "distribution": {
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
          "compact": 0.325,
          "elongated": 0.35,
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
      },
      "all_out_of_range": {},
      "marginal_out_of_range": {}
    },
    "full_guard": {
      "distribution": {
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
      },
      "all_out_of_range": {},
      "marginal_out_of_range": {}
    }
  }
}
```

## Quantile sensitivity
```json
{
  "q025_q975": {
    "flash": {
      "feasible": true,
      "distribution": {
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
      },
      "overlap_with_formal_full": {
        "count": 6,
        "jaccard": 0.08108108108108109,
        "candidate_ids": [
          "dwbg_0_12011_64_000298",
          "dwbg_0_15683_44_000364",
          "dwbg_0_24571_74_000557",
          "dwbg_0_24571_8_000559",
          "dwbg_0_30103_62_000636",
          "dwbg_0_7859_75_000231"
        ]
      }
    },
    "black": {
      "feasible": true,
      "distribution": {
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
          "compact": 0.325,
          "elongated": 0.375,
          "medium": 0.3
        },
        "weakness_count": {
          "0": 0.475,
          "1": 0.4,
          "2": 0.125,
          "3": 0.0,
          "2+": 0.125,
          "3-way": 0.0
        }
      },
      "overlap_with_formal_full": {
        "count": 21,
        "jaccard": 0.3559322033898305,
        "candidate_ids": [
          "dwbg_1_15683_20_000401",
          "dwbg_1_15683_22_000403",
          "dwbg_1_19997_15_000481",
          "dwbg_1_19997_32_000495",
          "dwbg_1_19997_35_000498",
          "dwbg_1_19997_7_000500",
          "dwbg_1_19997_8_000501",
          "dwbg_1_24571_18_000568",
          "dwbg_1_24571_24_000574",
          "dwbg_1_24571_30_000578",
          "dwbg_1_30103_0_000651",
          "dwbg_1_3407_31_000160",
          "dwbg_1_3407_34_000163",
          "dwbg_1_3407_35_000164",
          "dwbg_1_42_13_000064",
          "dwbg_1_42_23_000070",
          "dwbg_1_42_29_000074",
          "dwbg_1_42_35_000079",
          "dwbg_1_42_4_000080",
          "dwbg_1_7859_16_000237",
          "dwbg_1_7859_8_000253"
        ]
      }
    }
  },
  "q05_q95": {
    "flash": {
      "feasible": true,
      "distribution": {
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
      },
      "overlap_with_formal_full": {
        "count": 6,
        "jaccard": 0.08108108108108109,
        "candidate_ids": [
          "dwbg_0_12011_64_000298",
          "dwbg_0_15683_44_000364",
          "dwbg_0_24571_74_000557",
          "dwbg_0_24571_8_000559",
          "dwbg_0_30103_62_000636",
          "dwbg_0_7859_75_000231"
        ]
      }
    },
    "black": {
      "feasible": true,
      "distribution": {
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
          "compact": 0.325,
          "elongated": 0.375,
          "medium": 0.3
        },
        "weakness_count": {
          "0": 0.475,
          "1": 0.4,
          "2": 0.125,
          "3": 0.0,
          "2+": 0.125,
          "3-way": 0.0
        }
      },
      "overlap_with_formal_full": {
        "count": 21,
        "jaccard": 0.3559322033898305,
        "candidate_ids": [
          "dwbg_1_15683_20_000401",
          "dwbg_1_15683_22_000403",
          "dwbg_1_19997_15_000481",
          "dwbg_1_19997_32_000495",
          "dwbg_1_19997_35_000498",
          "dwbg_1_19997_7_000500",
          "dwbg_1_19997_8_000501",
          "dwbg_1_24571_18_000568",
          "dwbg_1_24571_24_000574",
          "dwbg_1_24571_30_000578",
          "dwbg_1_30103_0_000651",
          "dwbg_1_3407_31_000160",
          "dwbg_1_3407_34_000163",
          "dwbg_1_3407_35_000164",
          "dwbg_1_42_13_000064",
          "dwbg_1_42_23_000070",
          "dwbg_1_42_29_000074",
          "dwbg_1_42_35_000079",
          "dwbg_1_42_4_000080",
          "dwbg_1_7859_16_000237",
          "dwbg_1_7859_8_000253"
        ]
      }
    }
  },
  "q10_q90": {
    "flash": {
      "feasible": true,
      "distribution": {
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
      },
      "overlap_with_formal_full": {
        "count": 6,
        "jaccard": 0.08108108108108109,
        "candidate_ids": [
          "dwbg_0_12011_64_000298",
          "dwbg_0_15683_44_000364",
          "dwbg_0_24571_74_000557",
          "dwbg_0_24571_8_000559",
          "dwbg_0_30103_62_000636",
          "dwbg_0_7859_75_000231"
        ]
      }
    },
    "black": {
      "feasible": true,
      "distribution": {
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
          "compact": 0.325,
          "elongated": 0.375,
          "medium": 0.3
        },
        "weakness_count": {
          "0": 0.475,
          "1": 0.4,
          "2": 0.125,
          "3": 0.0,
          "2+": 0.125,
          "3-way": 0.0
        }
      },
      "overlap_with_formal_full": {
        "count": 21,
        "jaccard": 0.3559322033898305,
        "candidate_ids": [
          "dwbg_1_15683_20_000401",
          "dwbg_1_15683_22_000403",
          "dwbg_1_19997_15_000481",
          "dwbg_1_19997_32_000495",
          "dwbg_1_19997_35_000498",
          "dwbg_1_19997_7_000500",
          "dwbg_1_19997_8_000501",
          "dwbg_1_24571_18_000568",
          "dwbg_1_24571_24_000574",
          "dwbg_1_24571_30_000578",
          "dwbg_1_30103_0_000651",
          "dwbg_1_3407_31_000160",
          "dwbg_1_3407_34_000163",
          "dwbg_1_3407_35_000164",
          "dwbg_1_42_13_000064",
          "dwbg_1_42_23_000070",
          "dwbg_1_42_29_000074",
          "dwbg_1_42_35_000079",
          "dwbg_1_42_4_000080",
          "dwbg_1_7859_16_000237",
          "dwbg_1_7859_8_000253"
        ]
      }
    }
  }
}
```

The formal q05¨Cq95 method remains frozen. Quantile variants are feasibility diagnostics only.
