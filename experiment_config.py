"""Configuration helpers for baseline and modular MAGIC experiments."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path


DEFAULT_CONFIG = {
    "experiment_name": "baseline",
    "pipeline_mode": "standard",
    "seed": 42,
    "device": "cuda",
    "dtype": "float16",
    "prompt": "a photo of a sks defect",
    "negative_prompt": None,
    "num_inference_steps": 50,
    "guidance_scale": 7.5,
    "blur_factor": 0,
    "normal_filter": {
        "enabled": True,
        "black_threshold": 20,
        "min_mean_luminance": 30.0,
        "min_nonblack_ratio": 0.60,
    },
    "modules": {
        "prompt_perturbation": {
            "enabled": False,
            "noise_scale": 0.0,
        },
        "spatial_guidance": {
            "enabled": False,
            "inside": 7.5,
            "outside": 7.5,
            "use_schedule": False,
            "schedule": "cosine",
            "minimum": 4.0,
            "maximum": 7.5,
            "power": 2.0,
            "exp_k": 3.0,
            "sigmoid_k": 8.0,
            "sample_per_step": False,
        },
        "cama": {"enabled": False},
        "ddim_noise": {
            "enabled": False,
            "eta": 0.0,
            "anomaly_strength_min": 0.0,
            "anomaly_strength_max": 0.0,
            "anomaly_stop_step": 999999,
            "use_random_mask": False,
        },
        "mdap": {
            "enabled": False,
            "strength": 0.35,
            "schedule": "cosine",
            "end_fraction": 0.7,
            "ring_width": 12,
            "threshold_sigma": 1.0,
            "blur_sigma": 2.0,
            "contrast_min": 0.5,
            "contrast_max": 2.0,
            "amplitude": 1.0,
            "modes": {
                "Flash_point": "bright",
                "Big_black_spots": "dark",
            },
        },
        "rda": {
            "enabled": False,
            "root": None,
            "filename": "rda.pt",
        },
        "carf": {
            "enabled": False,
            "root": None,
            "filename": "carf.pt",
        },
        "msdf": {
            "enabled": False,
            "root": None,
            "filename": "msdf.pt",
        },
    },
}


def _merge(base, override):
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path):
    config = deepcopy(DEFAULT_CONFIG)
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            user_config = json.load(handle)
        _merge(config, user_config)
    validate_config(config)
    return config


def validate_config(config):
    modules = config["modules"]
    spatial = modules["spatial_guidance"]
    noise = modules["ddim_noise"]
    mdap = modules["mdap"]
    rda = modules["rda"]
    carf = modules["carf"]
    msdf = modules["msdf"]
    normal_filter = config["normal_filter"]

    if config["num_inference_steps"] <= 0:
        raise ValueError("num_inference_steps must be greater than zero")
    if config["pipeline_mode"] not in {"standard", "custom"}:
        raise ValueError("pipeline_mode must be 'standard' or 'custom'")
    if spatial["minimum"] > spatial["maximum"]:
        raise ValueError("spatial_guidance.minimum cannot exceed maximum")
    if noise["anomaly_strength_min"] > noise["anomaly_strength_max"]:
        raise ValueError("ddim_noise anomaly_strength_min cannot exceed anomaly_strength_max")
    if not 0.0 <= mdap["end_fraction"] <= 1.0:
        raise ValueError("mdap.end_fraction must be in [0, 1]")
    if mdap["strength"] < 0:
        raise ValueError("mdap.strength cannot be negative")
    if rda["enabled"] and not rda["root"]:
        raise ValueError("rda.root is required when RDA is enabled")
    if rda["enabled"] and config["pipeline_mode"] != "custom":
        raise ValueError("RDA requires pipeline_mode='custom'")
    if rda["enabled"] and spatial["enabled"]:
        raise ValueError(
            "Enable either RDA or legacy spatial_guidance, not both"
        )
    if carf["enabled"] and not carf["root"]:
        raise ValueError("carf.root is required when CARF is enabled")
    if carf["enabled"] and config["pipeline_mode"] != "custom":
        raise ValueError("CARF requires pipeline_mode='custom'")
    if msdf["enabled"] and not msdf["root"]:
        raise ValueError("msdf.root is required when MSDF is enabled")
    if msdf["enabled"] and config["pipeline_mode"] != "custom":
        raise ValueError("MSDF requires pipeline_mode='custom'")
    if msdf["enabled"] and (rda["enabled"] or carf["enabled"]):
        raise ValueError("The first MSDF experiment is MSDF-only; disable RDA and CARF")
    if not 0 <= normal_filter["black_threshold"] <= 255:
        raise ValueError("normal_filter.black_threshold must be in [0, 255]")
    if not 0 <= normal_filter["min_nonblack_ratio"] <= 1:
        raise ValueError("normal_filter.min_nonblack_ratio must be in [0, 1]")


def enabled_modules(config):
    return [
        name for name, values in config["modules"].items()
        if isinstance(values, dict) and values.get("enabled", False)
    ]
