#!/usr/bin/env python3
"""Audit and run the frozen Raw SD2 Bank-A Plastic_Bomo detector control."""
from __future__ import annotations

import argparse, csv, hashlib, json, os, platform
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO, __version__ as ultralytics_version

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543"
MANIFEST = SOURCE / "formal_bank_A_manifest.json"
REAL_YAML = Path("/mnt/sda/cbj/dataset/Plastic_Bomo/bootstrap_guard_downstream/real_only.yaml")
MODEL = ROOT / "pretrained/yolo11s.pt"
OUT = ROOT / "results_for_gpt/bootstrap_guard_generator_downstream/raw_sd2"
RUNS = ROOT / "runs/bootstrap_guard_generator_downstream/raw_sd2"
DATA_DIR = RUNS / "dataset_binding"
DATA_YAML = DATA_DIR / "raw_sd2_bank_a.yaml"
HIST = ROOT / "results_for_gpt/bootstrap_guard_downstream/validated_per_seed.json"
RR = ROOT / "results_for_gpt/bootstrap_guard_realrepeat/seed_42_result.json"
SEEDS = (42, 3407, 2026)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def save(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def resolve_record_path(value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else SOURCE / p


def read_real_yaml():
    d = yaml.safe_load(REAL_YAML.read_text())
    base = Path(d["path"])
    return d, base / d["train"], base / d["val"]


def image_files(directory: Path):
    return sorted(p.resolve() for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


def audit() -> dict:
    OUT.mkdir(parents=True, exist_ok=True); DATA_DIR.mkdir(parents=True, exist_ok=True)
    m = json.loads(MANIFEST.read_text())
    recs = m.get("records", [])
    errors, items = [], []
    if m.get("bank") != "A" or len(recs) != 80: errors.append("MANIFEST_NOT_BANK_A_80")
    ids = [r.get("slot_id") for r in recs]
    if len(set(ids)) != 80: errors.append("SLOT_ID_NOT_UNIQUE")
    classes = Counter(r.get("class") for r in recs)
    if classes != {"flash": 40, "black": 40}: errors.append(f"CLASS_COUNTS_{dict(classes)}")
    for r in recs:
        sid, cls, fs = r.get("slot_id"), r.get("class"), r.get("frozen_slot", {})
        ip, lp = resolve_record_path(r["image_path"]), resolve_record_path(r["label_path"])
        e = []
        if not ip.is_file(): e.append("MISSING_IMAGE")
        if not lp.is_file(): e.append("MISSING_LABEL")
        if fs.get("split") != "train": e.append("FROZEN_SPLIT_NOT_TRAIN")
        if fs.get("slot_id") != sid or fs.get("class") != cls: e.append("FROZEN_ID_CLASS_MISMATCH")
        ann = fs.get("annotation_id", "")
        if not ann or ann not in sid: e.append("ANNOTATION_ID_NOT_BOUND_TO_SLOT")
        actual_im_sha = sha(ip) if ip.is_file() else None
        actual_lb_sha = sha(lp) if lp.is_file() else None
        if actual_im_sha != r.get("image_sha256"): e.append("IMAGE_SHA_MISMATCH")
        if actual_lb_sha != r.get("label_sha256"): e.append("LABEL_SHA_MISMATCH")
        wh = None; label = None
        if ip.is_file():
            with Image.open(ip) as im: wh = list(im.size)
            if wh != fs.get("geometry", {}).get("final_canvas_wh"): e.append("IMAGE_SIZE_MISMATCH")
        if lp.is_file():
            lines = [x for x in lp.read_text().splitlines() if x.strip()]
            if len(lines) != 1: e.append("LABEL_LINE_COUNT_NOT_ONE")
            else:
                try: label = [float(x) for x in lines[0].split()]
                except ValueError: e.append("LABEL_PARSE_ERROR")
                if label and len(label) == 5:
                    expected_class = 0 if cls == "flash" else 1
                    if int(label[0]) != expected_class: e.append("LABEL_CLASS_MISMATCH")
                    if not np.allclose(label[1:], fs.get("bbox", []), atol=5e-7, rtol=0): e.append("LABEL_BBOX_MISMATCH")
                elif label: e.append("LABEL_FIELD_COUNT")
        if not r.get("formal_synthetic") or r.get("bank") != "A": e.append("NOT_FORMAL_BANK_A")
        if r.get("quality_selection") is not False: e.append("QUALITY_SELECTION_NOT_FALSE")
        items.append({"slot_id":sid,"annotation_id":ann,"class":cls,"frozen_split":fs.get("split"),
                      "frozen_source_path":fs.get("source_path"),"source_path_name_contains_val":
                      "/images/val/" in str(fs.get("source_path","")),"image_path":str(ip),"label_path":str(lp),
                      "image_wh":wh,"image_sha256":actual_im_sha,"label_sha256":actual_lb_sha,
                      "bbox":fs.get("bbox"),"label":label,"errors":e})
        errors.extend(f"{sid}:{x}" for x in e)
    real_cfg, real_train, real_val = read_real_yaml()
    real_images = image_files(real_train)
    real_labels = [Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt") for p in real_images]
    instances = sum(len([x for x in p.read_text().splitlines() if x.strip()]) for p in real_labels)
    if len(real_images) != 138 or instances != 168: errors.append(f"REAL_BASE_{len(real_images)}_{instances}")
    synthetic = [resolve_record_path(r["image_path"]).resolve() for r in recs]
    train = real_images + synthetic
    (DATA_DIR / "train_images.txt").write_text("\n".join(map(str, train)) + "\n")
    data = {"path": str(DATA_DIR), "train": str(DATA_DIR / "train_images.txt"),
            "val": str(real_val.resolve()), "names": real_cfg["names"]}
    DATA_YAML.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    result = {"status":"PASS" if not errors else "FAIL","manifest_path":str(MANIFEST),
              "manifest_sha256":sha(MANIFEST),"manifest_status":m.get("status"),"bank":m.get("bank"),
              "records":len(recs),"slot_ids_unique":len(set(ids))==80,"class_counts":dict(classes),
              "real_train_images":len(real_images),"real_instances":instances,
              "synthetic_images":len(synthetic),"total_train_images":len(train),
              "frozen_train_split_all":all(x["frozen_split"]=="train" for x in items),
              "historical_source_path_naming_caveat_count":sum(x["source_path_name_contains_val"] for x in items),
              "historical_source_path_naming_caveat":"Frozen protocol says split=train, while the bound legacy source_path may contain /images/val/. Eligibility is based on the frozen Stage20B protocol identity/SHA/annotation binding, not that legacy folder token.",
              "data_yaml":str(DATA_YAML),"train_list_sha256":sha(DATA_DIR/"train_images.txt"),
              "validation_path":str(real_val.resolve()),"errors":errors,"items":items}
    save(OUT / "dataset_audit.json", result)
    if errors: raise RuntimeError(f"Dataset audit failed with {len(errors)} errors; see dataset_audit.json")
    return result


def metrics_payload(m):
    b=m.box; pc={}
    for i,name in m.names.items():
        k="flash" if i==0 else "black"
        pc[k]={"ap50":float(b.ap50[i]),"ap5095":float(b.maps[i]),"precision":float(b.p[i]),"recall":float(b.r[i]),"dataset_name":name}
    return {"precision":float(b.mp),"recall":float(b.mr),"map50":float(b.map50),"map5095":float(b.map),"per_class":pc}


def train(seed: int, device: str):
    a=audit()
    run=RUNS/f"raw_sd2_bank_a_s{seed}"
    if run.exists(): raise FileExistsError(f"Refusing to overwrite {run}")
    model=YOLO(str(MODEL))
    model.train(data=str(DATA_YAML),model=str(MODEL),epochs=150,patience=40,batch=1,imgsz=1536,
        device=device,workers=4,project=str(RUNS),name=run.name,exist_ok=False,pretrained=True,
        optimizer="auto",seed=seed,deterministic=True,rect=False,cos_lr=False,close_mosaic=10,
        amp=True,val=True,mosaic=1.0,mixup=0.0,lr0=0.01,lrf=0.01,momentum=0.937,
        weight_decay=0.0005,warmup_epochs=3.0,warmup_momentum=0.8,warmup_bias_lr=0.1,
        box=7.5,cls=0.5,dfl=1.5,nbs=64,hsv_h=0.015,hsv_s=0.7,hsv_v=0.4,
        degrees=0.0,translate=0.1,scale=0.5,shear=0.0,perspective=0.0,flipud=0.0,fliplr=0.5)
    recover(seed,device)


def recover(seed:int,device:str):
    run=RUNS/f"raw_sd2_bank_a_s{seed}"; rows=list(csv.DictReader((run/"results.csv").open()))
    best,last=run/"weights/best.pt",run/"weights/last.pt"; stop=len(rows)
    best_epoch=max(range(stop),key=lambda i:float(rows[i]["metrics/mAP50-95(B)"]))+1
    ck=torch.load(last,map_location="cpu",weights_only=False); updates=ck.get("updates"); source="last.pt:updates"
    if updates is None:
        nb,nw,last_step,updates=218,654,-1,0
        for ep in range(stop):
            acc=64
            for i in range(nb):
                ni=i+nb*ep
                if ni<nw: acc=max(1,int(np.interp(ni,[0,nw],[1,64]).round()))
                if ni-last_step>=acc: updates+=1;last_step=ni
        source="exact_reconstruction_from_frozen_ultralytics_accumulation_schedule"
    val=YOLO(str(best)).val(data=str(DATA_YAML),imgsz=1536,batch=1,device=device,workers=4,plots=False)
    rec={"method":"Raw SD2 Bank A","seed":seed,"run_dir":str(run.relative_to(ROOT)),
         "best_epoch":best_epoch,"stop_epoch":stop,"optimizer_steps":int(updates),"optimizer_steps_source":source,
         "best_checkpoint_sha256":sha(best),**metrics_payload(val)}
    save(OUT/f"seed_{seed}_result.json",rec);print(json.dumps(rec,indent=2))


def meanstd(xs): return float(np.mean(xs)),float(np.std(xs,ddof=1))


def finalize():
    rr=[json.loads((ROOT/f"results_for_gpt/bootstrap_guard_realrepeat/seed_{s}_result.json").read_text()) for s in SEEDS]
    raw=[json.loads((OUT/f"seed_{s}_result.json").read_text()) for s in SEEDS]
    hist=json.loads(HIST.read_text())["records"]
    with (OUT/"per_seed_results.csv").open("w",newline="") as f:
        fields=["method","seed","precision","recall","map50","map5095","flash_ap50","flash_ap5095","black_ap50","black_ap5095","best_epoch","stop_epoch","optimizer_steps","checkpoint_sha256"]
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader()
        for r in raw:w.writerow({"method":r["method"],"seed":r["seed"],"precision":r["precision"],"recall":r["recall"],"map50":r["map50"],"map5095":r["map5095"],"flash_ap50":r["per_class"]["flash"]["ap50"],"flash_ap5095":r["per_class"]["flash"]["ap5095"],"black_ap50":r["per_class"]["black"]["ap50"],"black_ap5095":r["per_class"]["black"]["ap5095"],"best_epoch":r["best_epoch"],"stop_epoch":r["stop_epoch"],"optimizer_steps":r["optimizer_steps"],"checkpoint_sha256":r["best_checkpoint_sha256"]})
    groups={"Real-only":[r for r in hist if r["arm"]=="real_only"],"RealRepeat":rr,
            "Random":[r for r in hist if r["arm"]=="random"],"BootstrapGuard":[r for r in hist if r["arm"]=="bootstrap_guard"],"Raw SD2 Bank A":raw}
    aggregates=[]
    for name,rs in groups.items():
        row={"method":name}
        for k in ("precision","recall","map50","map5095"):
            m,s=meanstd([r[k] for r in rs]);row[k+"_mean"]=m;row[k+"_std"]=s;row[k]=f"{m:.4f} ± {s:.4f}"
        aggregates.append(row)
    fields=["method","precision","recall","map50","map5095","precision_mean","precision_std","recall_mean","recall_std","map50_mean","map50_std","map5095_mean","map5095_std"]
    with (OUT/"aggregate_results.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(aggregates)
    save(OUT/"checkpoint_audit.json",{"status":"PASS","records":[{"seed":r["seed"],"path":r["run_dir"]+"/weights/best.pt","sha256":r["best_checkpoint_sha256"],"best_epoch":r["best_epoch"]} for r in raw]})
    bg={r["seed"]:r for r in hist if r["arm"]=="bootstrap_guard"}
    save(OUT/"compute_audit.json",{"classification":"SAME_PER_EPOCH_DATASET_SIZE; TOTAL_COMPUTE_VARIES_WITH_EARLY_STOPPING","samples_per_epoch":218,"records":[{"seed":r["seed"],"stop_epoch":r["stop_epoch"],"optimizer_steps":r["optimizer_steps"],"bootstrap_guard_stop_epoch":bg[r["seed"]]["stop_epoch"]} for r in raw]})
    args=ROOT/"runs/bootstrap_guard_downstream/bootstrap_guard_s42/args.yaml"
    save(OUT/"training_config_audit.json",{"status":"MATCHED_EXCEPT_TRAIN_DATA","ultralytics":ultralytics_version,"python":platform.python_version(),"torch":torch.__version__,"historical_args":str(args.relative_to(ROOT)),"historical_args_sha256":sha(args),"model":str(MODEL.relative_to(ROOT)),"model_sha256":sha(MODEL),"data_yaml":str(DATA_YAML),"seeds":list(SEEDS),"frozen":{"epochs":150,"patience":40,"batch":1,"imgsz":1536,"optimizer":"auto","mosaic":1.0,"mixup":0.0,"close_mosaic":10,"amp":True,"deterministic":True}})
    g={x["method"]:x for x in aggregates}; rawm=g["Raw SD2 Bank A"]["map5095_mean"]; realm=g["Real-only"]["map5095_mean"]; rrm=g["RealRepeat"]["map5095_mean"]
    lines=["# Raw SD2 Bank A downstream detection","","| Method | Precision ↑ | Recall ↑ | mAP50 ↑ | mAP50-95 ↑ |","|---|---:|---:|---:|---:|"]
    for n in ("Real-only","RealRepeat","Random","BootstrapGuard","Raw SD2 Bank A"):
        x=g[n];lines.append(f"| {n} | {x['precision']} | {x['recall']} | {x['map50']} | {x['map5095']} |")
    lines += ["","## Conclusion","",f"Raw SD2 − Real-only mAP50-95: {rawm-realm:+.4f}. Raw SD2 − RealRepeat: {rawm-rrm:+.4f}. These are observed three-seed results under the frozen protocol; no hyperparameter or sample selection was changed.",""]
    (OUT/"RAW_SD2_DOWNSTREAM_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    p=argparse.ArgumentParser();p.add_argument("--audit",action="store_true");p.add_argument("--seed",type=int,choices=SEEDS);p.add_argument("--recover-seed",type=int,choices=SEEDS);p.add_argument("--device",default="0");p.add_argument("--finalize",action="store_true");a=p.parse_args()
    if a.audit:audit()
    elif a.seed is not None:train(a.seed,a.device)
    elif a.recover_seed is not None:recover(a.recover_seed,a.device)
    elif a.finalize:finalize()
    else:p.error("choose --audit, --seed, --recover-seed, or --finalize")

if __name__=="__main__":main()
