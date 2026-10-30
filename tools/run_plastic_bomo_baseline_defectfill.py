"""Audit, Primary-Strict generation, and unified ROI evaluation for DefectFill.

This adapter intentionally labels the third-party code PARTIAL_REIMPLEMENTATION:
the three training losses and LoRA paths execute, but native eight-candidate LFS
is not implemented end-to-end.  This run never performs LFS or best-of-N.
"""
from __future__ import annotations

import argparse, csv, gc, hashlib, json, math, os, sys, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
THIRD = REPO.parent / "third_party/defectfill"
OUT = REPO / "results_for_gpt/plastic_bomo_baselines/defectfill"
PRIVATE = OUT / "server_artifacts"
P0 = REPO / "results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115"
BASE = REPO / "sd2-community/stable-diffusion-2-inpainting"
LORA_OUT = REPO / "results_for_gpt/plastic_bomo_baselines/lora"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
sys.path[:0] = [str(THIRD), str(REPO), str(REPO / "tools")]

CLASSES = ("black", "flash")
BANKS = ("A", "B", "C")

def now(): return datetime.now(timezone.utc).isoformat()
def load(p): return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p, v):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(4 << 20), b""): h.update(b)
    return h.hexdigest()
def require(x, msg):
    if not x: raise RuntimeError(msg)
def write_csv(p, rows):
    with Path(p).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def audit():
    checkpoints = {}
    for cls in CLASSES:
        p = PRIVATE / cls / "checkpoints/checkpoint_final.pt"
        require(p.is_file(), f"missing {cls} final checkpoint")
        import torch
        c = torch.load(p, map_location="cpu", weights_only=False)
        require(c.get("step") == 2000 and c.get("unet_lora") and c.get("text_encoder_lora"), f"bad {cls} checkpoint")
        checkpoints[cls] = {"sha256": sha(p), "bytes": p.stat().st_size, "step": c["step"]}
    base_files = {}
    for rel in ("unet/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors", "text_encoder/model.safetensors"):
        base_files[rel] = {"sha256": sha(BASE / rel), "bytes": (BASE / rel).stat().st_size}
    save(OUT / "training_protocol.json", {
        "status": "TRAINING_COMPLETE", "method_identity": "PARTIAL_REIMPLEMENTATION",
        "official_implementation": False, "paper_venue": "CVPR 2025",
        "authenticity": {
            "defect_loss": "IMPLEMENTED_AND_BACKPROPAGATED", "object_loss": "IMPLEMENTED_AND_BACKPROPAGATED",
            "attention_loss": "IMPLEMENTED_AND_BACKPROPAGATED", "lora": "IMPLEMENTED_UNET_AND_TEXT_ENCODER",
            "low_fidelity_selection": "MISSING_END_TO_END; utility exists but inference CLI disables candidates",
        },
        "data": {"source": "Plastic_Bomo V2 TRAIN only", "counts": {"black": 97, "flash": 75},
                 "mask_source": "COARSE_BBOX_MASK", "pixel_level_GT": False},
        "training": {"separate_classes": True, "rank": 8, "alpha": 16, "steps_per_class": 2000,
          "batch_size": 1, "gradient_accumulation": 4, "seed": 42, "unet_lr": 2e-4,
          "text_encoder_lr": 4e-5, "warmup_steps": 100, "loss_weights": {"defect": .5, "object": .2, "attention": .05},
          "base": "sd2-community/stable-diffusion-2-inpainting", "in_channels": 9, "MSDF": False},
        "inference": {"protocol": "Primary Strict", "candidates_per_slot": 1, "selection": False,
          "reroll": False, "banks": list(BANKS), "slots_per_class_bank": 40, "steps": 50, "cfg": 7.5,
          "native_defectfill_protocol_executed": False},
        "adaptations": ["frozen-manifest loader", "Albumentations-free equivalent random scale/crop", "optional TensorBoard no-op fallback"],
    })
    save(OUT / "checkpoint_audit.json", {"status": "PASS_PARTIAL_REIMPLEMENTATION", "audited_at": now(),
        "base_checkpoint": base_files, "checkpoints": checkpoints})
    save(OUT / "experiment_status.json", {"status": "TRAINING_COMPLETE", "method_identity": "PARTIAL_REIMPLEMENTATION",
        "training_updates": {"black": 2000, "flash": 2000}, "generated": 0, "native_LFS_executed": False})

def roi(slot):
    cx, cy, bw, bh = slot["bbox"]; w, h = slot["geometry"]["final_canvas_wh"]
    return [max(0, math.floor((cx-.75*bw)*w)), max(0, math.floor((cy-.75*bh)*h)),
            min(w, math.ceil((cx+.75*bw)*w)), min(h, math.ceil((cy+.75*bh)*h))]

def generate(device_index=0):
    import numpy as np, torch
    from PIL import Image
    from torchvision.transforms.functional import pil_to_tensor
    from model import DefectFillModel
    from utils import load_checkpoint
    from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rng = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
    records = []
    manifest = OUT / "generation_manifest.json"
    save(manifest, {"status": "RUNNING", "records": records})
    device = torch.device(f"cuda:{device_index}"); torch.cuda.set_device(device)
    for cls in CLASSES:
        model = DefectFillModel(device=device, lora_rank=8, lora_alpha=16, seed=42,
            base_model=BASE, local_files_only=True, enable_lfs=False)
        ckpt = PRIVATE / cls / "checkpoints/checkpoint_final.pt"
        load_checkpoint(model, None, ckpt)
        model.pipeline.unet.eval(); model.pipeline.text_encoder.eval(); model.pipeline.vae.eval()
        prompt = f"A plastic_bomo with {model.placeholder_token}"
        for bank in BANKS:
            folder = PRIVATE / "images" / bank / cls; folder.mkdir(parents=True, exist_ok=True)
            for i, slot in enumerate(s for s in slots if s["class"] == cls):
                crop, mask = crop_and_mask(slot, use_normal=True)
                image_t = pil_to_tensor(crop.convert("RGB")).unsqueeze(0).to(device, torch.float16) / 127.5 - 1
                mask_t = pil_to_tensor(mask.convert("L")).unsqueeze(0).to(device, torch.float16) / 255
                seed = rng[slot["slot_id"]][bank]["sample_seed"]
                g = torch.Generator(device=device).manual_seed(seed)
                out = model.generate(image_t, mask_t, prompt, 50, 7.5, g)[0]
                arr = (out.float().clamp(0, 1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
                patch = Image.fromarray(arr, "RGB")
                with Image.open(slot["normal_path"]) as im: full = im.convert("RGB")
                x,y,_,_ = slot["geometry"]["fixed_crop_xyxy"]; l,t,r,b = slot["geometry"]["paste_valid_crop_region_xyxy"]
                full.paste(patch.crop((l-x,t-y,r-x,b-y)), (l,t))
                path = folder / f"{slot['slot_id']}.png"; full.save(path)
                records.append({"slot_id": slot["slot_id"], "class": cls, "bank": bank,
                  "frozen_prompt": slot["prompt"], "method_prompt": prompt, "sample_seed": seed,
                  "normal_id": slot["conditioning_normal_id"], "bbox_normalized_xywh": slot["bbox"],
                  "mask_source": "COARSE_BBOX_MASK", "candidate_index": 0, "candidate_count": 1,
                  "selection": False, "reroll": False, "roi_xyxy": roi(slot),
                  "image_path": path.relative_to(OUT).as_posix(), "image_SHA256": sha(path)})
                save(manifest, {"status":"RUNNING", "records":records, "count":len(records)})
                print(f"[generate] {cls} {bank} {i+1}/40 total={len(records)}/240", flush=True)
        del model; gc.collect(); torch.cuda.empty_cache()
    require(len(records)==240 and Counter((r["bank"],r["class"]) for r in records)==Counter({(b,c):40 for b in BANKS for c in CLASSES}), "generation count")
    save(manifest, {"status":"COMPLETE", "protocol":"Primary Strict", "count":240, "records":records,
      "candidate_count_total":240, "selection_count":0, "reroll_count":0, "native_LFS_executed":False})
    s=load(OUT/"experiment_status.json"); s.update(status="GENERATION_COMPLETE",generated=240); save(OUT/"experiment_status.json",s)

def evaluate(device_index=0):
    import torch
    import run_plastic_bomo_baseline_lora as lora
    import run_plastic_bomo_msdf_multibank_stability as x12
    from evaluate_plastic_bomo_msdf_gamma_sweep import runtime
    manifest=load(OUT/"generation_manifest.json"); require(manifest["status"]=="COMPLETE", "generation incomplete")
    lib, runtime_audit = runtime(f"cuda:{device_index}")
    lora.OUTPUT=OUT; features=lora.extract_lora_features(f"cuda:{device_index}",torch,lib,manifest)
    gamma=x12.array_features(GAMMA/"gamma_sweep_inception_features.npz",GAMMA/"gamma_sweep_inception_feature_cache.json",torch)
    refs=load(GAMMA/"gamma_sweep_protocol.json")["development_reference"]["records"]
    real={p:torch.stack([gamma["real:"+r["annotation_id"]] for r in refs if p=="Overall" or r["class"]==p.lower()]) for p in ("Black","Flash","Overall")}
    def fake(b,p):
        rows=[r for r in manifest["records"] if r["bank"]==b and (p=="Overall" or r["class"]==p.lower())]
        return torch.stack([features[f"lora:{r['bank']}:{r['class']}:{r['slot_id']}"] for r in rows])
    bw=[]; pooled=[]
    for b in BANKS:
      for p in ("Black","Flash","Overall"): bw.append({"bank":b,"population":p,**x12.metric(real[p],fake(b,p),lib),"metric_source":"DefectFill Primary Strict ROI"})
    for scope,banks in (("B+C","BC"),("A+B+C","ABC")):
      for p in ("Black","Flash","Overall"): pooled.append({"scope":scope,"population":p,**x12.metric(real[p],torch.cat([fake(b,p) for b in banks]),lib),"metric_source":"concatenated features; not bank average"})
    write_csv(OUT/"bankwise_metrics.csv",bw); write_csv(OUT/"pooled_metrics.csv",pooled)
    comparison=[]
    for r in lora.historical_comparison_rows(): comparison.append(r)
    for r in bw+pooled: comparison.append({"scope":r.get("bank",r.get("scope")),"population":r["population"],"method":"DefectFill PARTIAL Primary Strict","fine_tuning_type":"UNet + text-encoder LoRA r8 plus three losses",**{k:r[k] for k in ("KID","KID_subset_std","FID","reference_ROIs","generated_images","KID_subset_size")},"metric_source":r["metric_source"]})
    write_csv(OUT/"comparison.csv",comparison)
    lines=["# Plastic_Bomo DefectFill 对比实验","","Status: `COMPLETE_PARTIAL_REIMPLEMENTATION`","","该仓库是非官方且不完整复现：三项损失与 LoRA 实际参与训练，但原生 8 候选 Low-Fidelity Selection 没有端到端实现。本轮只执行 Primary Strict（每 slot 第一个且唯一候选，无筛选/重抽）。","","训练 Mask 是 `COARSE_BBOX_MASK`，不是像素级 GT。Black/Flash 各 2000 updates，LoRA rank=8/alpha=16。","","## Bankwise","","|Bank|Population|KID|FID|","|---|---|---:|---:|"]
    lines += [f"|{r['bank']}|{r['population']}|{r['KID']:.10f}|{r['FID']:.6f}|" for r in bw]
    lines += ["","## Pooled（拼接特征重算）","","|Scope|Population|KID|FID|","|---|---|---:|---:|"]+[f"|{r['scope']}|{r['population']}|{r['KID']:.10f}|{r['FID']:.6f}|" for r in pooled]
    (OUT/"REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    s=load(OUT/"experiment_status.json"); s.update(status="COMPLETE_PARTIAL_REIMPLEMENTATION",evaluation_complete=True,completed_at=now()); save(OUT/"experiment_status.json",s)

def main():
    p=argparse.ArgumentParser(); p.add_argument("stage",choices=("audit","generate","evaluate","all")); p.add_argument("--device",type=int,default=0); a=p.parse_args()
    os.environ.update(HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    if a.stage in ("audit","all"): audit()
    if a.stage in ("generate","all"): generate(a.device)
    if a.stage in ("evaluate","all"): evaluate(a.device)
if __name__ == "__main__": main()
