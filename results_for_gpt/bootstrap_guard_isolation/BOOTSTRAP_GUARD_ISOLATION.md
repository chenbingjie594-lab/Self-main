# BootstrapGuard Mechanism Isolation

Status: **GUARD_MECHANISM_DISTINCT**

Selection was frozen before guardrails and cross-detector weakness labels were loaded. No validation, downstream, or detector score was used.

## Fairness
```json
{
  "same_frozen_candidate_pool": true,
  "same_eligibility": true,
  "same_class_count": true,
  "same_parent_cap": true,
  "same_seed_cap": true,
  "same_required_background_diversity": true,
  "same_manifold_definition": true,
  "only_mechanism_difference": "BootstrapGuard uses real bootstrap distribution guard; DiversityQuality does not",
  "passed": true
}
```

## Candidate overlap
```json
{
  "flash": {
    "count": 8,
    "jaccard": 0.1111111111111111,
    "candidate_ids": [
      "dwbg_0_12011_10_000001_c7007",
      "dwbg_0_12011_64_000298",
      "dwbg_0_15683_44_000364",
      "dwbg_0_19997_2_000425",
      "dwbg_0_19997_35_000440",
      "dwbg_0_24571_74_000557",
      "dwbg_0_3407_17_000091",
      "dwbg_0_7859_22_000181"
    ]
  },
  "black": {
    "count": 14,
    "jaccard": 0.21212121212121213,
    "candidate_ids": [
      "dwbg_1_15683_20_000401",
      "dwbg_1_19997_15_000481",
      "dwbg_1_19997_7_000500",
      "dwbg_1_24571_15_000566",
      "dwbg_1_24571_30_000578",
      "dwbg_1_30103_0_000651",
      "dwbg_1_30103_10_000653",
      "dwbg_1_3407_19_000151",
      "dwbg_1_3407_23_000155",
      "dwbg_1_3407_31_000160",
      "dwbg_1_3407_33_000162",
      "dwbg_1_42_34_000078",
      "dwbg_1_7859_1_000233",
      "dwbg_1_7859_8_000253"
    ]
  }
}
```

## Manifold quality
```json
{
  "flash": {
    "diversity_quality": {
      "mean": 0.6969777811948674,
      "median": 0.6895445302323222,
      "std": 0.03325716892347781,
      "min": 0.65559679198976,
      "max": 0.7845899969820378
    },
    "bootstrap_guard": {
      "mean": 0.5531649376373231,
      "median": 0.536733194552494,
      "std": 0.11259473243839166,
      "min": 0.37258058442490394,
      "max": 0.7812160229205956
    },
    "abs_mean_difference": 0.1438128435575443,
    "abs_median_difference": 0.1528113356798282
  },
  "black": {
    "diversity_quality": {
      "mean": 0.775370410641099,
      "median": 0.7738684116484917,
      "std": 0.023892437909896504,
      "min": 0.7395127056182795,
      "max": 0.8491924645744323
    },
    "bootstrap_guard": {
      "mean": 0.6980215068054706,
      "median": 0.7229449536206796,
      "std": 0.08898480018751306,
      "min": 0.38585117783150147,
      "max": 0.8491924645744323
    },
    "abs_mean_difference": 0.07734890383562842,
    "abs_median_difference": 0.050923458027812174
  }
}
```

## Source diversity
```json
{
  "flash": {
    "diversity_quality": {
      "unique_parent_source": 40,
      "unique_background": 40,
      "max_per_parent": 1,
      "max_per_background": 1,
      "seed_histogram": {
        "12011": 5,
        "15683": 8,
        "19997": 8,
        "24571": 3,
        "30103": 8,
        "3407": 2,
        "42": 3,
        "7859": 3
      },
      "max_per_seed": 8
    },
    "bootstrap_guard": {
      "unique_parent_source": 40,
      "unique_background": 40,
      "max_per_parent": 1,
      "max_per_background": 1,
      "seed_histogram": {
        "12011": 5,
        "15683": 2,
        "19997": 6,
        "24571": 5,
        "30103": 4,
        "3407": 3,
        "42": 7,
        "7859": 8
      },
      "max_per_seed": 8
    }
  },
  "black": {
    "diversity_quality": {
      "unique_parent_source": 40,
      "unique_background": 40,
      "max_per_parent": 1,
      "max_per_background": 1,
      "seed_histogram": {
        "12011": 2,
        "15683": 7,
        "19997": 3,
        "24571": 4,
        "30103": 6,
        "3407": 8,
        "42": 5,
        "7859": 5
      },
      "max_per_seed": 8
    },
    "bootstrap_guard": {
      "unique_parent_source": 40,
      "unique_background": 40,
      "max_per_parent": 1,
      "max_per_background": 1,
      "seed_histogram": {
        "12011": 1,
        "15683": 2,
        "19997": 7,
        "24571": 7,
        "30103": 2,
        "3407": 7,
        "42": 8,
        "7859": 6
      },
      "max_per_seed": 8
    }
  }
}
```

## Post-selection distribution audit
```json
{
  "flash": {
    "diversity_quality": {
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
    "bootstrap_guard": {
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
    "diversity_quality_out_of_range": {
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
    "bootstrap_guard_out_of_range": {}
  },
  "black": {
    "diversity_quality": {
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
    "bootstrap_guard": {
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
    "diversity_quality_out_of_range": {
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
    "bootstrap_guard_out_of_range": {}
  }
}
```
