from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_seed42_only_and_gate_required():
 s=(ROOT/"tools/run_deeppcb_stage8b_detector.py").read_text(encoding="utf-8")
 assert "SEED=42" in s
 assert "SEEDS=" not in s and "for seed in" not in s
 assert 'gate["status"]=="NOMORPH_REPAIR_SUPPORTED"' in s
def test_frozen_stage7c_detector_protocol():
 s=(ROOT/"tools/run_deeppcb_stage8b_detector.py").read_text(encoding="utf-8")
 for token in ('epochs=150','patience=151','batch=8','mosaic=0.0','real_draws":260','synthetic_draws":80'):
  assert token in s
 assert 'run/"weights/last.pt"' in s
