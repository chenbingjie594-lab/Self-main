import sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from run_deeppcb_stage8a import VARIANTS

def test_frozen_variant_scope():
    assert set(VARIANTS)=={"full","no_latent","no_pixel","no_multiscale","residual_half","residual_zero"}
    assert "no_morphology" not in VARIANTS

def test_ablation_control_validation():
    pytest.importorskip("torch")
    from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter
    m=MSDFAdapter(4,64,[1280,1280,640,320])
    state=m.set_ablation({"pixel_branch":False,"residual_scale":.5})
    assert not state["pixel_branch"] and state["latent_branch"] and state["residual_scale"]==.5
    with pytest.raises(ValueError):m.set_ablation({"residual_scale":1.1})
    with pytest.raises(ValueError):m.set_ablation({"up_block_mask":[True]})
    with pytest.raises(ValueError):m.set_ablation({"invented":False})

def test_no_forbidden_stage8a_methods():
    root=Path(__file__).resolve().parents[1]
    text=(root/"tools/run_deeppcb_stage8a.py").read_text()+(root/"tools/evaluate_deeppcb_stage8a.py").read_text()
    for forbidden in ("BootstrapGuard","novelty filtering","weighted_quality_score ="):
        assert forbidden not in text
    assert "detector_training_count\":0" in text
