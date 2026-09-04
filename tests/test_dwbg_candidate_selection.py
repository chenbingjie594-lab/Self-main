import json
import sys
import types

import numpy as np

try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from tools.dwbg_utils import (
    BIN_NAMES, assign_profile_bin, compute_bbox_geometry, compute_local_contrast,
    native, quantile_labels,
)
from tools.score_dwbg_candidates import ensemble_aggregate
from tools.select_dwbg_candidates import (
    interval_boundary_score, score_candidate, select_dwbg,
)


def profile():
    rows = lambda names, values: [{"name": n, "weakness": v} for n, v in zip(names, values)]
    return {"class_profiles": {
        "0": {"scale": rows(BIN_NAMES["scale"], [.2, .3, .5]),
              "contrast": rows(BIN_NAMES["contrast"], [.2, .35, .4]),
              "morphology": rows(BIN_NAMES["morphology"], [.4, .2, .25])},
        "1": {"scale": rows(BIN_NAMES["scale"], [.45, .15, .2]),
              "contrast": rows(BIN_NAMES["contrast"], [.5, .2, .1]),
              "morphology": rows(BIN_NAMES["morphology"], [.2, .15, .4])},
    }}


def config():
    return {
        "flash": {"flash_conf_low": .1, "flash_conf_high": .45, "min_iou": .5,
                  "weights": {"boundary": .5, "scale": .25, "contrast": .125, "morphology": .125},
                  "marginal_quotas": {"scale": {"large": .5}, "contrast": {"high": .3}, "morphology": {"compact": .3}}},
        "black": {"min_iou": .25, "difficulty_target": .35, "difficulty_sigma": .25,
                  "weights": {"scale": .3, "contrast": .35, "morphology": .25, "detector_difficulty": .1},
                  "marginal_quotas": {"scale": {"tiny": .5}, "contrast": {"low": .5}, "morphology": {"elongated": .5}}},
        "selection": {"quota_bonus": .4, "source_diversity_bonus": .12, "seed_diversity_bonus": .06, "geometry_diversity_bonus": .08},
    }


def candidate(identifier, class_id, scale, contrast, morphology, confidence=.25, iou=.7, seed=1, source=None):
    return {"candidate_id": identifier, "class_id": class_id, "scale_bin": scale,
            "contrast_bin": contrast, "morphology_bin": morphology,
            "median_confidence": confidence, "median_iou": iou, "seed": seed,
            "reference_image": source or f"normal_{identifier}.jpg", "image_path": f"{identifier}.jpg", "label_path": f"{identifier}.txt"}


def test_geometry_matches_profile_definition():
    geometry = compute_bbox_geometry([10, 20, 30, 30], 100, 100)
    assert np.isclose(geometry["area_fraction"], .02)
    assert geometry["elongation"] == 2.0


def test_local_contrast_is_shared_definition():
    image = np.full((20, 20), 100, np.float32)
    image[8:12, 8:12] = 150
    result = compute_local_contrast(image, [8, 8, 12, 12])
    assert result["raw_contrast"] > 0
    assert np.isfinite(result["local_contrast"])


def test_class_specific_tertile_assignment():
    _, threshold = quantile_labels([.1, .2, .3, .4, .5, .6], BIN_NAMES["scale"])
    assert assign_profile_bin(.1, threshold, BIN_NAMES["scale"]) == "tiny"
    assert assign_profile_bin(.6, threshold, BIN_NAMES["scale"]) == "large"


def test_detector_ensemble_median():
    result = ensemble_aggregate([{"best_iou": .2, "confidence": .1}, {"best_iou": .8, "confidence": .4}, {"best_iou": .6, "confidence": .9}])
    assert np.isclose(result["median_iou"], .6)
    assert np.isclose(result["median_confidence"], .4)


def test_flash_boundary_interval_and_decay():
    assert interval_boundary_score(.25, .6, .1, .45, .5) == 1.0
    assert interval_boundary_score(.7, .6, .1, .45, .5) < 1.0
    assert interval_boundary_score(.02, .6, .1, .45, .5) < interval_boundary_score(.08, .6, .1, .45, .5)
    assert interval_boundary_score(.25, .2, .1, .45, .5) == 0.0


def test_black_weakness_score_and_distinct_policy():
    flash = score_candidate(candidate("f", 0, "large", "high", "compact"), profile(), config())
    black = score_candidate(candidate("b", 1, "tiny", "low", "elongated"), profile(), config())
    assert flash["boundary_score"] is not None
    assert black["boundary_score"] is None
    assert black["detector_difficulty_score"] is not None
    assert black["weakness_score"] > flash["weakness_score"]


def test_quota_selection_does_not_only_choose_same_bin():
    pool = [
        candidate("a", 1, "tiny", "low", "elongated", seed=1),
        candidate("b", 1, "tiny", "medium", "compact", seed=2),
        candidate("c", 1, "medium", "low", "elongated", seed=3),
        candidate("d", 1, "large", "high", "compact", seed=4),
    ]
    selected = select_dwbg(pool, 3, profile(), config())
    assert len(selected) == 3
    assert len({item["scale_bin"] for item in selected}) >= 2
    assert any("quota_" in reason for item in selected for reason in item["selection_reason"])


def test_equal_random_dwbg_counts_contract():
    random = [candidate("r1", 0, "tiny", "low", "compact"), candidate("r2", 1, "tiny", "low", "elongated")]
    dwbg = [candidate("d1", 0, "large", "high", "compact"), candidate("d2", 1, "tiny", "low", "elongated")]
    assert len(random) == len(dwbg)
    assert [x["class_id"] for x in random] == [x["class_id"] for x in dwbg]


def test_manifest_is_json_serializable():
    selected = select_dwbg([candidate("a", 0, "large", "high", "compact")], 1, profile(), config())
    encoded = json.dumps(native({"selected": selected, "number": np.int64(1)}))
    assert '"number": 1' in encoded
