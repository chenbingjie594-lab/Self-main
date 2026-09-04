import json
import sys
import types
from pathlib import Path

import numpy as np

# The production profiler deliberately reuses DASR's shared helpers.  DASR
# imports OpenCV for image processing, while these pure-statistics tests do not
# need it; provide an import-only stub when pytest runs in a minimal CPU env.
try:
    import cv2  # noqa: F401
except ModuleNotFoundError:
    sys.modules["cv2"] = types.ModuleType("cv2")

from tools.profile_dwbg_weakness import (
    bin_statistics,
    build_profiles,
    context_ring_masks,
    local_contrast,
    match_prediction,
    native,
    quantile_labels,
    read_yolo_boxes,
)


def test_bbox_area_and_elongation(tmp_path: Path):
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.2 0.1\n", encoding="utf-8")
    item = read_yolo_boxes(label, width=100, height=100)[0]
    assert np.isclose(item["area_fraction"], 0.02)
    assert np.isclose(item["width_fraction"], 0.2)
    assert np.isclose(item["height_fraction"], 0.1)
    assert np.isclose(item["aspect_ratio"], 2.0)
    assert np.isclose(item["elongation"], 2.0)


def test_direction_independent_elongation(tmp_path: Path):
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.1 0.2\n", encoding="utf-8")
    item = read_yolo_boxes(label, width=100, height=100)[0]
    assert np.isclose(item["aspect_ratio"], 0.5)
    assert np.isclose(item["elongation"], 2.0)


def test_context_ring_clips_at_boundary():
    defect, context = context_ring_masks((8, 8), [0, 0, 2, 2], ring_scale=1.5)
    assert defect.shape == (8, 8)
    assert context.shape == (8, 8)
    assert defect.sum() == 4
    assert context.any()
    assert not np.logical_and(defect, context).any()


def test_tiny_bbox_contrast_is_finite():
    image = np.full((5, 5), 100, dtype=np.float32)
    image[0, 0] = 180
    result = local_contrast(image, [0, 0, 1, 1], ring_scale=1.5)
    assert np.isfinite(result["local_contrast"])
    assert np.isfinite(result["raw_contrast"])
    assert result["raw_contrast"] > 0


def test_match_uses_max_iou_then_its_confidence():
    boxes = np.asarray([
        [0, 0, 10, 10],       # perfect IoU, low confidence
        [0, 0, 8, 8],         # lower IoU, high confidence
        [0, 0, 10, 10],       # wrong class
    ], dtype=np.float32)
    result = match_prediction(
        [0, 0, 10, 10], boxes, np.asarray([0, 0, 1]),
        np.asarray([0.20, 0.99, 0.99]), class_id=0,
        match_iou=0.5, match_confidence=0.25,
    )
    assert np.isclose(result["best_iou"], 1.0)
    assert np.isclose(result["matched_confidence"], 0.20)
    assert result["detected"] is False


def test_no_same_class_prediction_returns_zero():
    result = match_prediction(
        [0, 0, 10, 10], np.asarray([[0, 0, 10, 10]]),
        np.asarray([1]), np.asarray([0.9]), class_id=0,
    )
    assert result == {"best_iou": 0.0, "matched_confidence": 0.0, "detected": False}


def test_duplicate_quantile_edges_are_merged():
    labels, metadata = quantile_labels([1.0, 1.0, 1.0], ("tiny", "medium", "large"))
    assert labels == ["medium", "medium", "medium"]
    assert metadata["active_bins"] == ["medium"]
    labels, metadata = quantile_labels([1.0, 1.0, 1.0, 2.0], ("low", "medium", "high"))
    assert len(labels) == 4
    assert set(labels).issubset({"low", "medium", "high"})


def test_laplace_smoothed_recall():
    records = [
        {"detected": True, "matched_confidence": 0.8, "best_iou": 0.7},
        {"detected": False, "matched_confidence": 0.1, "best_iou": 0.2},
    ]
    stats = bin_statistics(records)
    assert stats["recall"] == 0.5
    assert stats["smoothed_recall"] == 0.5
    assert stats["weakness"] == 0.5
    one = bin_statistics(records[:1])
    assert np.isclose(one["smoothed_recall"], 2 / 3)


def _instance(class_id, detected, area, contrast, elongation, confidence=0.3, iou=0.6):
    return {
        "class_id": class_id, "detected": detected,
        "area_fraction": area, "local_contrast": contrast,
        "elongation": elongation, "matched_confidence": confidence,
        "best_iou": iou,
    }


def test_joint_min_count_and_top_k():
    instances = [
        _instance(0, False, 0.01, 0.1, 1.0),
        _instance(0, False, 0.01, 0.1, 1.0),
        _instance(0, True, 0.01, 0.1, 1.0),
        _instance(0, True, 0.10, 2.0, 3.0),
        _instance(1, True, 0.02, 0.5, 1.2),
    ]
    profiles, modes = build_profiles(instances, min_joint_count=3, top_k_modes=1)
    assert len(modes["0"]) == 1
    assert modes["0"][0]["count"] == 3
    assert modes["0"][0]["eligible"] is True
    assert any(not item["eligible"] for item in profiles["0"]["joint_modes"])
    assert modes["1"] == []


def test_profile_is_json_serializable():
    instances = [_instance(0, np.bool_(True), np.float32(0.1), 1.0, 2.0)]
    profiles, modes = build_profiles(instances, min_joint_count=1, top_k_modes=10)
    payload = native({"version": np.int64(2), "instances": instances,
                      "class_profiles": profiles, "failure_modes": modes})
    encoded = json.dumps(payload)
    assert '"version": 2' in encoded
