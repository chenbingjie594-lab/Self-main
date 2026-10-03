"""Freeze the exact 2x2 Stage15A manifests and build only M10/M01 datasets."""
from __future__ import annotations
import argparse,hashlib,json,os
from collections import Counter
from pathlib import Path
import yaml
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def digest(x):return hashlib.sha256(json.dumps(x,separators=(",",":"),sort_keys=True).encode()).hexdigest()
def link(src,dst):
 src=Path(src).resolve();dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() or dst.is_symlink():
  if dst.resolve()!=src:raise RuntimeError("STAGE15A_DATASET_CONFLICT: "+str(dst))
 else:os.symlink(src,dst)
def layout(p):
 p=Path(p);d=yaml.safe_load(p.read_text());base=Path(d.get("path",p.parent));base=base if base.is_absolute() else (p.parent/base).resolve();tr=Path(d["train"]);tr=tr if tr.is_absolute() else base/tr;val=Path(d["val"]);val=val if val.is_absolute() else base/val;return tr,Path(str(tr).replace("/images/","/labels/")),val
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--stage14a_subsets",type=Path,required=True);p.add_argument("--candidate_pool",type=Path,required=True);p.add_argument("--real_data_yaml",type=Path,required=True);p.add_argument("--dataset_root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);sub=load(a.stage14a_subsets)["subsets"];pool=load(a.candidate_pool);rows=pool.get("candidates",pool.get("selected",[]));byid={x["candidate_id"]:x for x in rows}
 def split(ids,c):return [i for i in ids if int(byid[i]["class_id"])==c]
 RF,RB=split(sub["random"],0),split(sub["random"],1);HF,HB=split(sub["high_fidelity"],0),split(sub["high_fidelity"],1)
 if not all(len(x)==40 for x in (RF,RB,HF,HB)):raise RuntimeError("FACTORIAL_MANIFEST_INVALID: source class counts")
 ids={"M00":RF+RB,"M10":HF+RB,"M01":RF+HB,"M11":HF+HB};records={}
 for arm,z in ids.items():records[arm]=[{"candidate_id":i,"class_name":"flash" if int(byid[i]["class_id"])==0 else "black","image_path":byid[i]["image_path"],"label_path":byid[i]["label_path"],"generation_seed":int(byid[i]["seed"]),"parent_source_id":f"{byid[i]['class_id']}:{Path(byid[i]['source_image']).stem}"} for i in z]
 checks={"M00_flash_equals_random_f":split(ids["M00"],0)==RF,"M00_black_equals_random_b":split(ids["M00"],1)==RB,"M10_flash_equals_high_f":split(ids["M10"],0)==HF,"M10_black_equals_random_b":split(ids["M10"],1)==RB,"M01_flash_equals_random_f":split(ids["M01"],0)==RF,"M01_black_equals_high_b":split(ids["M01"],1)==HB,"M11_flash_equals_high_f":split(ids["M11"],0)==HF,"M11_black_equals_high_b":split(ids["M11"],1)==HB}
 if not all(checks.values()):raise RuntimeError("FACTORIAL_MANIFEST_INVALID")
 manifest={"status":"FROZEN","interpretation":"morphology alignment, not broad fidelity","arms":records,"candidate_id_sha256":{k:digest(v) for k,v in ids.items()},"source_subset_path":str(a.stage14a_subsets)};save(a.output/"factorial_subset_definition.json",manifest);save(a.output/"factorial_manifest_freeze_audit.json",{"status":"PASS","checks":checks,"counts":{k:dict(Counter(x["class_name"] for x in v)) for k,v in records.items()},"manifest_sha256":digest(manifest),"new_synthetic_generation":0});save(a.output/"factorial_overlap_audit.json",{"high_f_vs_random_f":len(set(HF)&set(RF)),"high_b_vs_random_b":len(set(HB)&set(RB)),"overlap_removed":False})
 images,labels,val=layout(a.real_data_yaml);base={x.stem:x for x in images.iterdir() if x.is_file()};labs={x.stem:x for x in labels.glob("*.txt")};audit={"status":"PASS","arms":{},"scheduled":{"images_per_epoch":218,"epochs":150,"draws":32700,"batches":32700}}
 for arm in ("M10","M01"):
  root=a.dataset_root/arm
  for i,stem in enumerate(sorted(base)):link(base[stem],root/"images/train"/f"base_{i:03d}{base[stem].suffix}");link(labs[stem],root/"labels/train"/f"base_{i:03d}.txt")
  for i,r in enumerate(records[arm]):link(r["image_path"],root/"images/train"/f"extra_{i:03d}{Path(r['image_path']).suffix}");link(r["label_path"],root/"labels/train"/f"extra_{i:03d}.txt")
  counts=Counter(int(float(z.split()[0])) for i in range(80) for z in (root/"labels/train"/f"extra_{i:03d}.txt").read_text().splitlines())
  if len(base)!=138 or counts!={0:40,1:40}:raise RuntimeError(f"FACTORIAL_DATASET_INVALID_{arm}")
  (root/"data.yaml").write_text(f"path: {root.resolve()}\ntrain: images/train\nval: {val.resolve()}\nnc: 2\nnames: [flash, black]\n",encoding="utf-8");audit["arms"][arm]={"base_real":138,"synthetic_flash":40,"synthetic_black":40,"synthetic_total":80,"train_images":218,"data_yaml":str((root/"data.yaml").resolve())}
 save(a.output/"factorial_training_budget_audit.json",audit);save(a.output/"stage15a_protocol.json",cfg);print(json.dumps({"status":"STAGE15A_FACTORIAL_PREPARED","new_training_arms":["M10","M01"]},indent=2))
if __name__=="__main__":main()
