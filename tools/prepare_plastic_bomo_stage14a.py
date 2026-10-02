"""Build compute-matched Plastic_Bomo Stage14A datasets from frozen subsets."""
from __future__ import annotations
import argparse,hashlib,json,os
from collections import Counter
from pathlib import Path
import yaml

PRIMARY=("random","high_fidelity","valid_novel","high_context_compatibility")
DIAGNOSTIC=("low_fidelity_diagnostic","low_context_diagnostic")
ARMS=("real_repeat",)+PRIMARY+DIAGNOSTIC
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def sha(p):
 h=hashlib.sha256();h.update(Path(p).read_bytes());return h.hexdigest()
def link(src,dst):
 src=Path(src).resolve();dst=Path(dst);dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists() or dst.is_symlink():
  if dst.resolve()!=src:raise RuntimeError("STAGE14A_DATASET_CONFLICT: "+str(dst))
 else:os.symlink(src,dst)
def layout(data_yaml):
 p=Path(data_yaml);d=yaml.safe_load(p.read_text());base=Path(d.get("path",p.parent));base=base if base.is_absolute() else (p.parent/base).resolve()
 def get(k):
  z=d[k];z=z[0] if isinstance(z,list) else z;z=Path(z);return z if z.is_absolute() else base/z
 train=get("train");val=get("val");labels=Path(str(train).replace("/images/","/labels/"));return d,train,labels,val
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--subsets",type=Path,required=True);p.add_argument("--candidate_pool",type=Path,required=True);p.add_argument("--real_data_yaml",type=Path,required=True);p.add_argument("--dataset_root",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);sub=load(a.subsets);pool=load(a.candidate_pool);rows=pool.get("candidates",pool.get("selected",[]));byid={x["candidate_id"]:x for x in rows};ydoc,images,labels,val=layout(a.real_data_yaml)
 if sub["status"]!="FROZEN_BEFORE_STAGE14A_SCREENING":raise RuntimeError("STAGE14A_SUBSETS_NOT_FROZEN")
 allowed=list(PRIMARY)+list(DIAGNOSTIC)
 if not all(sub["eligible_equal_budget"].get(x) for x in allowed):raise RuntimeError("STAGE14A_REQUIRED_SUBSET_NOT_EQUAL_BUDGET")
 base={x.stem:x for x in images.iterdir() if x.is_file()};labs={x.stem:x for x in labels.glob("*.txt")};
 if len(base)!=138 or set(base)!=set(labs):raise RuntimeError(f"STAGE14A_REAL_TRAIN_INVALID: images={len(base)} labels={len(labs)}")
 singles={0:[],1:[]}
 for stem,lp in labs.items():
  ids=[int(float(z.split()[0])) for z in lp.read_text().splitlines() if z.strip()]
  if len(ids)==1 and ids[0] in singles:singles[ids[0]].append(stem)
 if any(not singles[c] for c in singles):raise RuntimeError("STAGE14A_REALREPEAT_SINGLE_OBJECT_SOURCE_MISSING")
 repeat=[]
 for c in (0,1):
  z=sorted(singles[c]);repeat.extend((c,z[i%len(z)]) for i in range(40))
 audit={"status":"PASS","arms":{},"real_data_yaml":str(a.real_data_yaml),"real_train_images":138,"extra_images_per_arm":80,"total_images_per_arm":218,"class_exposure":{"flash":40,"black":40},"validation":str(val),"validation_unchanged":True}
 for arm in ARMS:
  root=a.dataset_root/arm
  for i,stem in enumerate(sorted(base)):link(base[stem],root/"images/train"/f"base_{i:03d}{base[stem].suffix}");link(labs[stem],root/"labels/train"/f"base_{i:03d}.txt")
  ids=[]
  if arm=="real_repeat":
   for i,(c,stem) in enumerate(repeat):link(base[stem],root/"images/train"/f"extra_{i:03d}{base[stem].suffix}");link(labs[stem],root/"labels/train"/f"extra_{i:03d}.txt");ids.append(f"real:{stem}:{c}")
  else:
   chosen=sub["subsets"][arm]
   for i,cid in enumerate(chosen):
    r=byid[cid];link(r["image_path"],root/"images/train"/f"extra_{i:03d}{Path(r['image_path']).suffix}");link(r["label_path"],root/"labels/train"/f"extra_{i:03d}.txt");ids.append(cid)
  counts=Counter()
  for i in range(80):
   lp=root/"labels/train"/f"extra_{i:03d}.txt"
   for z in lp.read_text().splitlines():counts[int(float(z.split()[0]))]+=1
  if counts!={0:40,1:40}:raise RuntimeError(f"STAGE14A_EXTRA_CLASS_EXPOSURE_MISMATCH_{arm}: {dict(counts)}")
  (root/"data.yaml").write_text(f"path: {root.resolve()}\ntrain: images/train\nval: {val.resolve()}\nnc: 2\nnames: [flash, black]\n",encoding="utf-8")
  audit["arms"][arm]={"train_images":218,"base_images":138,"extra_images":80,"extra_label_instances":{"flash":counts[0],"black":counts[1]},"extra_ids_sha256":hashlib.sha256(json.dumps(ids,separators=(",",":")).encode()).hexdigest(),"data_yaml":str((root/"data.yaml").resolve())}
 save(a.output/"subset_training_budget_audit.json",audit);save(a.output/"stage14a_dataset_manifest.json",{"status":"FROZEN","arms":audit["arms"],"subset_definition_sha256":sha(a.subsets),"candidate_pool_sha256":sha(a.candidate_pool),"deep_pcb_inputs":0});print(json.dumps({"status":"STAGE14A_DATASETS_PREPARED","arms":list(ARMS)},indent=2))
if __name__=="__main__":main()
