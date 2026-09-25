from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def test_single_training_variable_only():
    source=(ROOT/"tools/run_deeppcb_stage8b.py").read_text()
    assert source.count('"--msdf_disable_morphology_alignment"')==1
    assert '"--msdf_max_injection","0.75"' in source
    assert '"--msdf_branch_dropout","0.20"' in source
    assert '"--train_steps","2000"' in source
    assert '"--seed","42"' in source

def test_checkpoint_freezes_training_time_semantics():
    train=(ROOT/"train_rda.py").read_text(encoding="utf-8")
    guide=(ROOT/"diffusers/pipelines/stable_diffusion/msdf_guidance.py").read_text(encoding="utf-8")
    pipe=(ROOT/"diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint_magic.py").read_text(encoding="utf-8")
    assert '"morphology_alignment_enabled": msdf.ablation["morphology_alignment"]' in train
    assert 'checkpoint.get("morphology_alignment_enabled", True)' in guide
    assert "if msdf_ablation is not None" in pipe

def test_gate_does_not_train_detector_or_combine_metrics():
    source=(ROOT/"tools/evaluate_deeppcb_stage8b.py").read_text()
    assert '"weighted_score":False' in source
    assert "model.train(" not in source
    assert "NOMORPH_REPAIR_SUPPORTED" in source
    assert "NOMORPH_REPAIR_NOT_SUPPORTED" in source
    assert "NOMORPH_REPAIR_MIXED" in source

def test_nonfinite_boundary_values_are_missing_not_zero():
    source=(ROOT/"tools/evaluate_deeppcb_stage8b.py").read_text(encoding="utf-8")
    assert "np.isfinite(v) else None" in source
    assert '"missing"' in source
