#!/usr/bin/env python3
"""Final MSDF-v3 training-level ablation evaluation.

This entry point deliberately reuses the frozen Stage20B slot/RNG protocol and
the project's V2-TRAIN ROI metric implementation.  It never trains a model.
The generation/evaluation phases are kept separate so an interrupted GPU run
can be resumed without replacing completed images.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "runs/msdf_v3_ablation/final_quantitative_evaluation"
P0 = REPO / "results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115"
GAMMA = REPO / "results_for_gpt/plastic_bomo_msdf_gamma_sweep"
VARIANTS = {
    "full": (REPO / "model/msdf_v3_sd2_2000_s42", "full", "Full MSDF-v3"),
    "no_latent": (REPO / "model/msdf_v3_no_latent_sd2_2000_s42", "no_latent", "w/o Latent Semantic"),
    "no_morph": (REPO / "model/msdf_v3_no_morph_sd2_2000_s42", "no_morph", "w/o Morphology Guidance"),
}
CLASS_DIR = {"flash": "01_Flash_point", "black": "02_Big_black_spots"}


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require(value, message):
    if not value: raise RuntimeError(message)


def roi_xyxy(slot, margin=.25):
    w, h = slot["geometry"]["final_canvas_wh"]; cx, cy, bw, bh = slot["bbox"]
    return [max(0, math.floor((cx-(.5+margin)*bw)*w)), max(0, math.floor((cy-(.5+margin)*bh)*h)),
            min(w, math.ceil((cx+(.5+margin)*bw)*w)), min(h, math.ceil((cy+(.5+margin)*bh)*h))]


def checkpoint_binding(root):
    files = sorted(p for p in root.rglob("*") if p.is_file())
    values = {p.relative_to(root).as_posix(): sha(p) for p in files}
    return {"files": values, "tree_sha256": canonical_sha(values), "file_count": len(values)}


def audit():
    import torch
    from diffusers.pipelines.stable_diffusion.msdf_guidance import MSDFAdapter
    slots_path, rng_path, protocol_path = (P0 / "stage20b_generation_slots.json",
        P0 / "stage20b_rng_manifest.json", P0 / "stage20b_generation_protocol.json")
    slots = load(slots_path)["slots"]
    require(len(slots) == 80 and sum(s["class"] == "flash" for s in slots) == 40, "invalid frozen slots")
    prior_preflight_path = REPO / "runs/msdf_v3_ablation/visual_evaluation_20261029_v2/checkpoint_preflight.json"
    prior_preflight = load(prior_preflight_path)
    require(prior_preflight["status"] == "PASS", "prior six-checkpoint runtime preflight is not PASS")
    result = {"status": "PASS", "training_performed": False, "checkpoint_modified": False,
              "strict_runtime_preflight": {"path": str(prior_preflight_path),
                                           "sha256": sha(prior_preflight_path), "status": "PASS"},
              "frozen_inputs": {"slots_sha256": sha(slots_path), "rng_sha256": sha(rng_path),
                                "generation_protocol_sha256": sha(protocol_path)}, "checkpoints": {}}
    for variant, (model_root, expected, _label) in VARIANTS.items():
        result["checkpoints"][variant] = {}
        for cls, directory in CLASS_DIR.items():
            root = model_root / "Plastic_Bomo" / directory
            cp = torch.load(root / "msdf.pt", map_location="cpu", weights_only=False)
            stored = cp.get("training_ablation", "full")
            require(stored == expected, f"{variant}/{cls}: expected {expected}, got {stored}")
            adapter = MSDFAdapter(int(cp["latent_channels"]), int(cp["hidden_dim"]), cp["block_channels"],
                float(cp["context_scale"]), float(cp["max_injection"]), float(cp["max_residual_ratio"]),
                float(cp["branch_dropout"]), float(cp["pixel_support_weight"]), training_ablation=stored)
            loaded = adapter.load_state_dict(cp["state_dict"], strict=True)
            require(not loaded.missing_keys and not loaded.unexpected_keys, "strict adapter load failed")
            metadata = load(root / "msdf_training.json")
            msdf_digest = sha(root / "msdf.pt")
            old_row = prior_preflight["checkpoints"][variant][cls]
            require(old_row["msdf_sha256"] == msdf_digest and old_row["strict_runtime_load"],
                    f"{variant}/{cls}: checkpoint changed since strict runtime preflight")
            result["checkpoints"][variant][cls] = {"training_ablation": stored,
                "msdf_sha256": msdf_digest, "tree_binding_from_strict_runtime_preflight": old_row["binding"],
                "successful_updates": metadata.get("successful_optimizer_updates", metadata.get("max_train_steps", 2000)),
                "strict_adapter_load": True, "training_contract": adapter.training_contract()}
    old = REPO / "results_for_gpt/plastic_bomo_stage25_msdf_v2_requalification/detector_recovery_20261014_032214/generator_checkpoint_binding.json"
    prior = load(old)["classes"]
    result["historical_A0_reuse"] = {"decision": "REJECTED_CHECKPOINT_SHA_MISMATCH",
        "historical_binding_file": str(old),
        "historical_msdf_sha256": {c: prior[c]["checkpoint"]["sha256"]["msdf.pt"] for c in ("flash", "black")},
        "formal_A0_msdf_sha256": {c: result["checkpoints"]["full"][c]["msdf_sha256"] for c in ("flash", "black")}}
    save(OUT / "generation_integrity_audit.json", result)
    protocol = load(protocol_path)
    save(OUT / "metric_protocol.json", {"status": "FROZEN", "reference": "V2 TRAIN only",
        "reference_count": 172, "reference_feature_source": str(GAMMA / "gamma_sweep_inception_features.npz"),
        "roi": "normalized bbox expanded 25% on each side; floor/ceil; canvas clip",
        "feature_extractor": "torch-fidelity inception-v3-compat 2048D float32",
        "KID": {"subsets": 100, "subset_size": "min(50,n_real,n_fake)", "seed": 42,
                "kernel": "polynomial degree=3 gamma=None coef0=1"},
        "FID": "torch-fidelity float64 feature statistics", "pooled": "recomputed from concatenated features",
        "generation": {k: protocol[k] for k in ("num_inference_steps", "guidance_scale", "scheduler",
            "scheduler_config", "eta", "strength", "height", "width", "negative_prompt")},
        "slots_sha256": sha(slots_path), "rng_sha256": sha(rng_path), "final_eval_used": False})
    print("[audit] PASS: six checkpoints strictly loaded; historical A0 reuse rejected", flush=True)


def generate(device):
    import numpy as np
    import torch
    from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
    from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
    from generate_plastic_bomo_stage20b_banks import sample
    from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
    require(torch.cuda.is_available(), "CUDA_UNAVAILABLE: generation requires a working NVIDIA device")
    audit_data = load(OUT / "generation_integrity_audit.json")
    protocol = load(P0 / "stage20b_generation_protocol.json")
    slots = load(P0 / "stage20b_generation_slots.json")["slots"]
    rngs = {r["slot_id"]: r["banks"] for r in load(P0 / "stage20b_rng_manifest.json")["records"]}
    for variant, (model_root, expected, _label) in VARIANTS.items():
        manifest_path = OUT / "generation" / variant / "generation_manifest.json"
        records = load(manifest_path).get("records", []) if manifest_path.exists() else []
        done = {(r["bank"], r["slot_id"]) for r in records}
        for cls, dirname in CLASS_DIR.items():
            root = model_root / "Plastic_Bomo" / dirname
            pipe = StableDiffusionInpaintPipeline.from_pretrained(str(root), local_files_only=True,
                use_safetensors=True, torch_dtype=torch.float16, safety_checker=None,
                feature_extractor=None, requires_safety_checker=False).to(device)
            pipe.scheduler = DDIMScheduler.from_config(protocol["scheduler_config"])
            pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
            adapter = load_msdf_adapter(root / "msdf.pt", pipe.unet, device)
            require(adapter.training_ablation == expected, "runtime ablation mismatch")
            for slot in (s for s in slots if s["class"] == cls):
                donor, mask = crop_and_mask({**slot, "path": slot["source_path"], "sha256": slot["source_sha256"]})
                pixels = torch.from_numpy(np.asarray(donor).copy()).permute(2,0,1)[None].to(device).float()/127.5-1
                mask_t = torch.from_numpy(np.asarray(mask).copy())[None,None].to(device).float()/255
                with torch.inference_mode():
                    reference = pipe.vae.encode(pixels.half()).latent_dist.mode()*pipe.vae.config.scaling_factor
                def prepare(_module, args, kwargs):
                    inp = args[0] if args else kwargs["sample"]; timestep = args[1] if len(args)>1 else kwargs["timestep"]
                    adapter.prepare(reference.float(), mask_t, inp[:1,4:5].float(), timestep,
                                    pipe.scheduler.config.num_train_timesteps, True, pixels)
                handle = pipe.unet.register_forward_pre_hook(prepare, with_kwargs=True)
                try:
                    adapter.set_injection_gamma(1.0)
                    for bank in "ABC":
                        if (bank, slot["slot_id"]) in done: continue
                        image, trace = sample(pipe, slot, rngs[slot["slot_id"]][bank], protocol, device)
                        path = OUT / "generation" / variant / "images" / bank / f"{slot['slot_id']}.png"
                        path.parent.mkdir(parents=True, exist_ok=True); require(not path.exists(), f"refusing overwrite: {path}")
                        image.save(path, format="PNG")
                        row = {"variant": variant, "class": cls, "bank": bank, "slot_id": slot["slot_id"],
                            "frozen_slot": slot, "image_path": str(path), "image_sha256": sha(path),
                            "roi_xyxy": roi_xyxy(slot), "checkpoint_msdf_sha256": audit_data["checkpoints"][variant][cls]["msdf_sha256"],
                            "sample_seed": rngs[slot["slot_id"]][bank]["sample_seed"],
                            "bank_seed": rngs[slot["slot_id"]][bank]["bank_seed"], **trace,
                            "quality_selection": False, "reroll": False}
                        records.append(row); done.add((bank, slot["slot_id"]))
                        save(manifest_path, {"status": "RUNNING", "records": records})
                        print(f"[generate] {variant} {cls} {bank} {len(records)}/240", flush=True)
                finally: handle.remove()
            adapter.detach(); del adapter, pipe; gc.collect(); torch.cuda.empty_cache()
        require(len(records) == 240, f"{variant}: expected 240, got {len(records)}")
        save(manifest_path, {"status": "COMPLETE", "count": 240, "records": records,
            "quality_selection": False, "reroll_count": 0})


def metric(real, fake, lib):
    from torch_fidelity.metric_fid import fid_features_to_statistics, fid_statistics_to_metric
    from torch_fidelity.metric_kid import kid_features_to_metric
    options = {"kid_subsets":100, "kid_subset_size":min(50,len(real),len(fake)), "rng_seed":42,
               "verbose":False, "kid_degree":3, "kid_gamma":None, "kid_coef0":1.0}
    k = kid_features_to_metric(real, fake, **options)
    f = fid_statistics_to_metric(fid_features_to_statistics(real), fid_features_to_statistics(fake), verbose=False)
    return {"KID":float(k["kernel_inception_distance_mean"]),
            "KID_subset_std":float(k["kernel_inception_distance_std"]),
            "FID":float(f["frechet_inception_distance"]), "real_count":len(real), "generated_count":len(fake)}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def evaluate(device):
    import numpy as np
    import torch
    from PIL import Image
    from evaluate_plastic_bomo_msdf_gamma_sweep import runtime
    lib, runtime_audit = runtime(device)
    ledger=load(GAMMA/"gamma_sweep_inception_feature_cache.json")
    require(ledger["sha256"] == sha(GAMMA/"gamma_sweep_inception_features.npz"), "real feature cache SHA mismatch")
    with np.load(GAMMA/"gamma_sweep_inception_features.npz", allow_pickle=False) as z: arr=z["features"]
    cached={k:torch.from_numpy(arr[i]).double().cpu() for i,k in enumerate(ledger["keys"])}
    refs=load(GAMMA/"gamma_sweep_protocol.json")["development_reference"]["records"]
    real={c:torch.stack([cached["real:"+r["annotation_id"]] for r in refs if c=="overall" or r["class"]==c])
          for c in ("black","flash","overall")}
    model=lib["Inception"]("inception-v3-compat",["2048"],
        feature_extractor_weights_path=str(lib["inception_weight"]), internal_dtype=torch.float32
        ).float().eval().requires_grad_(False).to(device)
    features={}
    for variant in VARIANTS:
        m=load(OUT/"generation"/variant/"generation_manifest.json"); require(m["status"]=="COMPLETE", f"{variant} incomplete")
        for i,row in enumerate(m["records"]):
            p=Path(row["image_path"]); require(p.is_file() and sha(p)==row["image_sha256"], "image SHA mismatch")
            roi=Image.open(p).convert("RGB").crop(tuple(row["roi_xyxy"]))
            x=torch.from_numpy(np.asarray(roi).copy()).permute(2,0,1)[None].to(device)
            with torch.inference_mode(): value=model(x)[0][0].double().cpu()
            features[(variant,row["bank"],row["class"],row["slot_id"])]=value
            print(f"[feature] {variant} {i+1}/240", flush=True)
    bank_rows=[]; pooled=[]
    slots=load(P0/"stage20b_generation_slots.json")["slots"]
    for variant,(_root,_expected,label) in VARIANTS.items():
        for bank in "ABC":
            for pop in ("black","flash","overall"):
                selected=[s for s in slots if pop=="overall" or s["class"]==pop]
                fake=torch.stack([features[(variant,bank,s["class"],s["slot_id"])] for s in selected])
                bank_rows.append({"variant":variant,"configuration":label,"bank":bank,"population":pop.title(),**metric(real[pop],fake,lib)})
        for scope,banks in (("B+C","BC"),("A+B+C","ABC")):
            for pop in ("black","flash","overall"):
                selected=[s for s in slots if pop=="overall" or s["class"]==pop]
                fake=torch.stack([features[(variant,b,s["class"],s["slot_id"])] for b in banks for s in selected])
                pooled.append({"variant":variant,"configuration":label,"scope":scope,"population":pop.title(),**metric(real[pop],fake,lib)})
    write_csv(OUT/"ablation_bankwise_metrics.csv",bank_rows); write_csv(OUT/"ablation_pooled_metrics.csv",pooled)
    publish(bank_rows, pooled, runtime_audit)


def publish(bank_rows, pooled, runtime_audit):
    main=[r for r in pooled if r["scope"]=="A+B+C"]
    def row(v,p): return next(r for r in main if r["variant"]==v and r["population"]==p)
    table=[]
    for v,(_r,_e,label) in VARIANTS.items():
        table.append({"configuration":label,"Black_KID":row(v,"Black")["KID"],"Black_FID":row(v,"Black")["FID"],
            "Flash_KID":row(v,"Flash")["KID"],"Flash_FID":row(v,"Flash")["FID"],
            "Overall_KID":row(v,"Overall")["KID"],"Overall_FID":row(v,"Overall")["FID"]})
    write_csv(OUT/"ablation_final_table.csv",table)
    full=table[0]; deltas=[]
    for item in table[1:]:
        for name in list(full)[1:]:
            d=item[name]-full[name]; deltas.append({"configuration":item["configuration"],"metric":name,
                "absolute_delta_vs_full":d,"percent_change_vs_full":100*d/full[name],"improved":d<0})
    write_csv(OUT/"ablation_deltas_vs_full.csv",deltas)
    best={k:min(r[k] for r in table) for k in list(table[0])[1:]}
    lines=["# MSDF-v3 最终训练级消融定量评测","","## 核心结果（A+B+C pooled）","",
        "| 实验配置 | Black KID↓ | Black FID↓ | Flash KID↓ | Flash FID↓ | Overall KID↓ | Overall FID↓ |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in table:
        vals=[]
        for k in list(r)[1:]:
            s=f"{r[k]:.5f}" if k.endswith("KID") else f"{r[k]:.2f}"
            vals.append(f"**{s}**" if r[k]==best[k] else s)
        lines.append("| "+r["configuration"]+" | "+" | ".join(vals)+" |")
    lines += ["","## 相对 Full 的变化","","负值表示消融模型指标更低（更好）。",
              "","| 消融 | 指标 | 绝对差值 | 百分比变化 |","| --- | --- | ---: | ---: |"]
    for d in deltas: lines.append(f"| {d['configuration']} | {d['metric']} | {d['absolute_delta_vs_full']:+.8f} | {d['percent_change_vs_full']:+.2f}% |")
    lines += ["","## 科学结论","",
        "结论必须依据上表与 `ablation_bankwise_metrics.csv`：KID 和 FID 若方向不一致，则记为混合证据；Full 仅在 A/B/C 多数 Bank 同向优于消融时才称为稳定。", "",
        "## 证据边界","","结果使用 V2 TRAIN 的 172 个真实 ROI；它不是独立测试集，且可能与生成器训练数据重叠。A0 历史训练缺少完整初始 RNG/训练文件哈希账本，因此模块因果结论应保持条件性。未读取 final_eval。",""]
    (OUT/"ABLATION_FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    save(OUT/"experiment_status.json",{"status":"COMPLETE","training_count":0,"generated_images":720,
        "feature_count":720,"final_eval_used":False,"runtime_audit":runtime_audit})


def main():
    p=argparse.ArgumentParser(); p.add_argument("phase",choices=("audit","generate","evaluate","all")); p.add_argument("--device",default="cuda:0")
    a=p.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    if a.phase in ("audit","all"): audit()
    if a.phase in ("generate","all"): generate(a.device)
    if a.phase in ("evaluate","all"): evaluate(a.device)


if __name__ == "__main__":
    try: main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True)
        save(OUT/"experiment_status.json",{"status":"BLOCKED","phase_error":f"{type(e).__name__}: {e}",
            "training_count":0,"checkpoint_modified":False,"final_eval_used":False,"pid":os.getpid()})
        raise
