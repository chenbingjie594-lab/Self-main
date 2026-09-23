import json
import sys
from collections import Counter
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from run_deeppcb_stage5b import CLASSES
from run_deeppcb_stage7c import PairedSampler7C, frozen_selection, status

ROOT=Path(__file__).resolve().parents[1]


def test_frozen_formal_selection_maps_exactly_to_both_generators():
    manifest,classes,seeds=frozen_selection(
        ROOT/"results_for_gpt/bootstrap_guard_deeppcb_stage3r/random_matched_manifest_40x6_corrected.json",
        ROOT/"results_for_gpt/deeppcb_stage7a_sd2_baseline/sd2_candidate_pool_manifest.json",
        ROOT/"results_for_gpt/bootstrap_guard_deeppcb_stage2/candidate_pool_manifest.json")
    assert manifest["count"]==240
    assert Counter(classes)==Counter({c:40 for c in CLASSES})
    assert len({(x["source_instance_id"],x["generation_seed"]) for x in manifest["records"]})==240
    assert all(x["sd2_candidate_id"].startswith("sd2_") for x in manifest["records"])


def test_sampler_preserves_exact_frozen_schedule():
    names=[f"real_{i}" for i in range(100)]+[f"synthetic_{i:03d}" for i in range(240)]
    sequence=names[:260]+names[260:340]
    sampler=PairedSampler7C(names,[sequence]); sampler.set_epoch(0)
    assert [names[i] for i in sampler]==sequence
    assert len(sampler)==340


@pytest.mark.parametrize("mean,wins,expected",[(.01,2,"MSDF_DOWNSTREAM_SUPPORTED"),(-.01,1,"MSDF_DOWNSTREAM_NOT_SUPPORTED"),(.01,1,"MSDF_DOWNSTREAM_MIXED"),(-.01,2,"MSDF_DOWNSTREAM_MIXED")])
def test_objective_status(mean,wins,expected):
    assert status({"mean":mean,"wins":wins})==expected


def test_no_forbidden_selector_or_generator_training():
    source=(ROOT/"tools/run_deeppcb_stage7c.py").read_text(encoding="utf-8")
    assert "inference.py" not in source
    assert "BootstrapGuard" not in source
    assert "q50" not in source and "q95" not in source
    assert '"new_selector":False' in source
