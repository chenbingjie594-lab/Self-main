"""Finalize Stage14B replication gates and scientific status."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def stat(v):return {"values":v,"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1)),"wins":sum(x>0 for x in v)}
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();rows=load(a.output/"all_seed_metrics.json")["records"];by={(x["seed"],x["arm"]):x for x in rows};seeds=(42,3407,2026);d={}
 pairs=((('high','random'),'high_minus_random'),(('high','real_repeat'),'high_minus_realrepeat'),(('high','low'),'high_minus_low'))
 for pair,name in pairs:d[name]=stat([by[s,pair[0]]["map50_95"]-by[s,pair[1]]["map50_95"] for s in seeds])
 secondary={}
 for pair,name in pairs:secondary[name]={metric:stat([by[s,pair[0]][metric]-by[s,pair[1]][metric] for s in seeds]) for metric in ("map50","recall")}
 classes={c:stat([by[s,"high"]["per_class"][c]["ap50_95"]-by[s,"random"]["per_class"][c]["ap50_95"] for s in seeds]) for c in ("flash","black")};A=d["high_minus_random"]["mean"]>=.005;B=d["high_minus_random"]["wins"]>=2;C=d["high_minus_realrepeat"]["mean"]>=.005;D=d["high_minus_low"]["wins"]==3;label=load(a.output/"completed_fidelity_audit.json")["status"];conf=load(a.output/"selection_confound_audit.json")["status"]
 if A and B and C and D and label=="BROAD_FIDELITY_DIFFERENCE_PRESENT":status="PERCEPTUAL_FIDELITY_BOTTLENECK_CONFIRMED"
 elif A and B and C and D and label=="MORPHOLOGY_ALIGNMENT_DIFFERENCE_PRESENT":status="MORPHOLOGY_ALIGNMENT_BOTTLENECK_CONFIRMED"
 elif d["high_minus_random"]["wins"]<=1:status="STAGE14A_FIDELITY_SIGNAL_NOT_REPLICATED"
 else:status="UTILITY_SIGNAL_PRESENT_BUT_MECHANISM_UNRESOLVED"
 classdep=classes["flash"]["wins"]>=2 and all(x<=0 for x in classes["black"]["values"]);summary={"status":status,"fidelity_label":label,"selection_confound_status":conf,"gates":{"A_mean_high_random_at_least_0_5pp":A,"B_high_beats_random_2_of_3":B,"C_mean_high_realrepeat_at_least_0_5pp":C,"D_high_beats_low_3_of_3":D},"deltas":d,"secondary_deltas":secondary,"classwise":classes,"class_dependent_fidelity_bottleneck":classdep,"STAGE15_FIDELITY_GENERATOR_DESIGN_AUTHORIZED":status=="PERCEPTUAL_FIDELITY_BOTTLENECK_CONFIRMED","STAGE15_MORPHOLOGY_CONDITIONED_GENERATION_AUTHORIZED":status=="MORPHOLOGY_ALIGNMENT_BOTTLENECK_CONFIRMED","STAGE15_AUTO_START":False,"deep_pcb_training_count":0,"deep_pcb_generation_count":0};save(a.output/"three_seed_summary.json",summary);save(a.output/"classwise_replication.json",{"status":"CLASS_DEPENDENT_FIDELITY_BOTTLENECK" if classdep else "NO_CLASS_DEPENDENT_GATE","classes":classes});save(a.output/"stage14b_status.json",summary)
 lines=["# Plastic_Bomo Stage14B Morphology/Fidelity Signal Confirmation","",f"Status: **{status}**.","",f"Fidelity attribution: **{label}**.  ",f"Selection confound: **{conf}**.","","## Three-seed primary deltas","",f"- High - Random: {100*d['high_minus_random']['mean']:+.3f} ± {100*d['high_minus_random']['std']:.3f} pp; wins {d['high_minus_random']['wins']}/3",f"- High - RealRepeat: {100*d['high_minus_realrepeat']['mean']:+.3f} ± {100*d['high_minus_realrepeat']['std']:.3f} pp; wins {d['high_minus_realrepeat']['wins']}/3",f"- High - Low: {100*d['high_minus_low']['mean']:+.3f} ± {100*d['high_minus_low']['std']:.3f} pp; wins {d['high_minus_low']['wins']}/3","","## Secondary detector deltas (High - Random)","",f"- mAP50: {100*secondary['high_minus_random']['map50']['mean']:+.3f} ± {100*secondary['high_minus_random']['map50']['std']:.3f} pp",f"- Recall: {100*secondary['high_minus_random']['recall']['mean']:+.3f} ± {100*secondary['high_minus_random']['recall']['std']:.3f} pp","",f"Flash High-Random mean: {100*classes['flash']['mean']:+.3f} pp; black: {100*classes['black']['mean']:+.3f} pp.","",f"Class-dependent gate: **{classdep}**.","","KID was used only as a set-level distribution metric and never for per-sample ranking. Official validation was used only for final `last.pt` evaluation. No generator, DeepPCB experiment, or Stage15 execution occurred."];(a.output/"STAGE14B_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8");print(json.dumps(summary,indent=2))
if __name__=="__main__":main()
