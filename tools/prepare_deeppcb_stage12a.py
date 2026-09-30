"""Stage12A preparation: reconstruct frozen historical extra sets and utility."""
from __future__ import annotations
import argparse,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()
def records(p):
 x=load(p);return x.get("records",x if isinstance(x,list) else [])
def metric_rows(p):return load(p)["records"]
def main():
 p=argparse.ArgumentParser()
 for n in ("protocol","registry","folds","stage6f","stage6h","stage7a","stage7c","stage9f","stage10c","stage11b","output"):p.add_argument("--"+n,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);cfg=load(a.protocol);reg=load(a.registry)["instances"];byid={x["instance_id"]:x for x in reg};folds=load(a.folds);foldof={pid:int(f["fold"]) for f in folds["folds"] for pid in f["holdout_pair_ids"]};out=[];arms={}
 def add(fid,arm,row,path,label=None,kind="synthetic"):
  iid=row.get("instance_id",row.get("source_instance_id"));r=byid[iid];box=row.get("bbox_xyxy",r["bbox_xyxy"]);z={"family_id":fid,"arm":arm,"sample_id":f"{fid}:{arm}:{len(out)}","source_instance_id":iid,"physical_pair_id":row.get("pair_id",r["pair_id"]),"class_name":row.get("class_name",r["class_name"]),"class_id":int(r["class_id"]),"fold":foldof[r["pair_id"]],"image_path":str(path),"label_path":str(label) if label else None,"bbox_xyxy":[int(v) for v in box],"generation_seed":row.get("generation_seed"),"sample_type":kind,"historical_selection_source":fid};out.append(z);arms.setdefault((fid,arm),[]).append(z)
 # F2 exact frozen all-eligible manifests.
 for arm in ("inner","novel"):
  for x in records(a.stage6f/f"{arm}_candidate_manifest.json"):add("F2",arm,x,x["image_path"],x.get("label_path"))
 # F3 exact paired SD2/MSDF selection.
 for x in records(a.stage7c/"source_seed_selection_manifest.json"):
  iid=x.get("source_instance_id",x.get("instance_id"));base={**x,"instance_id":iid,"pair_id":byid[iid]["pair_id"],"class_name":byid[iid]["class_name"],"generation_seed":x.get("generation_seed",x.get("seed"))}
  add("F3","sd2",base,x.get("sd2_image_path",x.get("sd2_image",x.get("sd2"))),x.get("sd2_label_path",x.get("sd2_label")));add("F3","msdf",base,x.get("msdf_image_path",x.get("msdf_image",x.get("msdf"))),x.get("msdf_label_path",x.get("msdf_label")))
 # F4 four actual extra sets; real repeat is an explicit real-image replay set.
 for x in records(a.stage9f/"stage9f_frozen_selection_manifest.json"):
  for arm,key,kind in (("real_repeat80","real","real_replay"),("finetuned_sd2_syn80","class_finetuned_sd2","synthetic"),("pretrained_noadapt_syn80","pretrained_no_adaptation","synthetic"),("frozen_adapter_syn80","repaired_frozen_adapter","synthetic")):add("F4",arm,x,x[key],kind=kind)
 # F5 remains secondary/exploratory.
 sd2={(x.get("source_instance_id",x.get("instance_id")),int(x.get("generation_seed",x.get("seed")))):x for x in records(a.stage7a/"sd2_candidate_pool_manifest.json")}
 for x in records(a.stage10c/"frozen_source_seed_selection.json"):
  add("F5","tmvb_syn80",x,x["image_path"]);add("F5","real_repeat80",x,x["real_path"],kind="real_replay");s=sd2[(x["instance_id"],int(x["generation_seed"]))];add("F5","sd2_syn80",x,s.get("image_path",s.get("synthetic_path")))
 # F6 frozen replay selections mapped to isolation artifacts.
 iso={x["instance_id"]:x for x in records(a.stage11b/"replay_isolation_audit.json")}
 for arm in ("uniform","feature","hardness","gradient"):
  for x in records(a.stage11b/f"{arm}_selection.json"):add("F6",arm,x,iso[x["instance_id"]]["image_path"],iso[x["instance_id"]]["label_path"],"real_replay")
 manifest=[]
 for (fid,arm),z in sorted(arms.items()):
  manifest.append({"family_id":fid,"arm":arm,"count":len(z),"source_count":len({x["source_instance_id"] for x in z}),"pair_count":len({x["physical_pair_id"] for x in z}),"class_counts":dict(Counter(x["class_name"] for x in z)),"generation_seed_counts":dict(Counter(str(x["generation_seed"]) for x in z)),"records_sha256":digest(z)})
 # Utilities: primary epoch150 last.pt only. Multi-seed arms are averaged.
 utility=[]
 def utilities(fid,rows,role="primary"):
  g=defaultdict(list)
  for x in rows:g[x["arm"]].append(float(x["map50_95"]))
  for arm,v in g.items():utility.append({"family_id":fid,"arm":arm,"role":role,"map50_95":sum(v)/len(v),"seed_values":v,"seed_count":len(v)})
 utilities("F2",metric_rows(a.stage6h/"last150_per_seed_results.json"));utilities("F3",metric_rows(a.stage7c/"per_seed_metrics.json"));utilities("F4",[x for x in metric_rows(a.stage9f/"primary_last_metrics.json") if x["arm"]!="real_only"]);utilities("F5",metric_rows(a.stage10c/"primary_last_metrics.json"),"secondary_exploratory");utilities("F6",metric_rows(a.stage11b/"primary_last_metrics.json"))
 families=[
  {"family_id":"F1","name":"Stage5F historical selector family","role":"incomplete","reason":"actual per-instance extra sets and equal fixed-budget checkpoint protocol are not jointly reconstructable; historical stop epochs differ"},
  {"family_id":"F2","name":"Stage6H inner vs novel","role":"primary","comparable":True,"detector":"YOLO11s","detector_seeds":[42,3407,2026],"extra_budget":160,"epochs":150,"primary_checkpoint":"last.pt","arms":["inner","novel"]},
  {"family_id":"F3","name":"Stage7C SD2 vs MSDF-v3","role":"primary","comparable":True,"detector":"YOLO11s","detector_seeds":[42,3407,2026],"extra_budget":240,"epochs":150,"primary_checkpoint":"last.pt","arms":["sd2","msdf"]},
  {"family_id":"F4","name":"Stage9F frozen-adapter pilot","role":"primary","comparable":True,"detector":"YOLO11s","detector_seeds":[42],"extra_budget":80,"epochs":150,"primary_checkpoint":"last.pt","arms":["real_repeat80","finetuned_sd2_syn80","pretrained_noadapt_syn80","frozen_adapter_syn80"]},
  {"family_id":"F5","name":"Stage10C TMVB pilot","role":"secondary_exploratory","reason":"historical downstream run was exploratory/unauthorized for formal support","arms":["real_repeat80","sd2_syn80","tmvb_syn80"]},
  {"family_id":"F6","name":"Stage11B real replay allocation","role":"primary","comparable":True,"detector":"YOLO11s","detector_seeds":[42],"extra_budget":120,"epochs":150,"primary_checkpoint":"last.pt","arms":["uniform","feature","hardness","gradient"]}]
 validity={"status":"PASS_WITH_DECLARED_EXCLUSIONS","primary_family_count":4,"minimum_required":cfg["minimum_primary_families"],"families":families,"no_cross_family_raw_metric_pooling":True,"official_validation_use_count":0,"detector_rerun_count":0,"generator_run_count":0}
 save(a.output/"historical_family_registry.json",{"families":families});save(a.output/"family_validity_audit.json",validity);save(a.output/"historical_extra_set_manifest.json",{"records":out,"arms":manifest,"record_count":len(out),"frozen_sha256":digest(out)});save(a.output/"historical_family_utility.json",{"primary_metric":"mAP50-95","checkpoint":"epoch150 last.pt","records":utility,"within_family_only":True});print(json.dumps({"status":"STAGE12A_PREPARATION_COMPLETE","records":len(out),"arms":len(manifest)},indent=2))
if __name__=="__main__":main()
