"""Build frozen equal-budget Stage14B confirmation datasets."""
from __future__ import annotations
import argparse,hashlib,json,os
from collections import Counter
from pathlib import Path
import yaml
from audit_plastic_bomo_stage14b import load,save

def link(src,dst):
 src=Path(src).resolve();dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() or dst.is_symlink():
  if dst.resolve()!=src:raise RuntimeError("STAGE14B_DATASET_CONFLICT: "+str(dst))
 else:os.symlink(src,dst)
def layout(p):
 p=Path(p);d=yaml.safe_load(p.read_text());base=Path(d.get("path",p.parent));base=base if base.is_absolute() else (p.parent/base).resolve();tr=Path(d["train"]);tr=tr if tr.is_absolute() else base/tr;val=Path(d["val"]);val=val if val.is_absolute() else base/val;return tr,Path(str(tr).replace("/images/","/labels/")),val
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--attribution",type=Path,required=True);p.add_argument("--candidate_pool",type=Path,required=True);p.add_argument("--real_data_yaml",type=Path,required=True);p.add_argument("--dataset_root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();cfg=load(a.protocol);definition=load(a.attribution/"matched_subset_definition.json");pool=load(a.candidate_pool);rows=pool.get("candidates",pool.get("selected",[]));byid={x["candidate_id"]:x for x in rows};matched=definition["status"]=="FROZEN_MATCHED_SUBSETS"
 source=definition["subsets"];ids={"random":source["random_matched" if matched else "random"],"high":source["high_morphology_alignment" if matched else "high_fidelity"],"low":source["low_morphology_alignment" if matched else "low_fidelity_diagnostic"]};images,labels,val=layout(a.real_data_yaml);base={x.stem:x for x in images.iterdir() if x.is_file()};labs={x.stem:x for x in labels.glob("*.txt")};singles={0:[],1:[]}
 for stem,lp in labs.items():
  z=[int(float(x.split()[0])) for x in lp.read_text().splitlines() if x.strip()]
  if len(z)==1 and z[0] in singles:singles[z[0]].append(stem)
 repeat=[(c,sorted(singles[c])[i%len(singles[c])]) for c in (0,1) for i in range(40)];arms={**ids,"real_repeat":repeat};audit={"status":"PASS","matched_subsets":matched,"arms":{},"equal_budget":{"base":138,"extra":80,"total":218,"flash_extra":40,"black_extra":40}}
 for arm,chosen in arms.items():
  root=a.dataset_root/arm
  for i,stem in enumerate(sorted(base)):link(base[stem],root/"images/train"/f"base_{i:03d}{base[stem].suffix}");link(labs[stem],root/"labels/train"/f"base_{i:03d}.txt")
  if arm=="real_repeat":
   for i,(c,stem) in enumerate(chosen):link(base[stem],root/"images/train"/f"extra_{i:03d}{base[stem].suffix}");link(labs[stem],root/"labels/train"/f"extra_{i:03d}.txt")
  else:
   for i,cid in enumerate(chosen):r=byid[cid];link(r["image_path"],root/"images/train"/f"extra_{i:03d}{Path(r['image_path']).suffix}");link(r["label_path"],root/"labels/train"/f"extra_{i:03d}.txt")
  counts=Counter(int(float(z.split()[0])) for i in range(80) for z in (root/"labels/train"/f"extra_{i:03d}.txt").read_text().splitlines());
  if counts!={0:40,1:40}:raise RuntimeError(f"STAGE14B_CLASS_EXPOSURE_MISMATCH_{arm}_{dict(counts)}")
  (root/"data.yaml").write_text(f"path: {root.resolve()}\ntrain: images/train\nval: {val.resolve()}\nnc: 2\nnames: [flash, black]\n",encoding="utf-8");audit["arms"][arm]={"images":218,"extra":80,"class_counts":{"flash":40,"black":40},"data_yaml":str((root/"data.yaml").resolve())}
 save(a.output/"training_budget_audit.json",audit);print(json.dumps({"status":"STAGE14B_DATASETS_PREPARED","matched":matched},indent=2))
if __name__=="__main__":main()
