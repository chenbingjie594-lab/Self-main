"""Compute the pre-registered Stage15A factorial effects and causal gates."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

SEEDS=(42,3407,2026);METRICS=("map50_95","map50","recall")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+"\n",encoding="utf-8")
def summary(v):return {"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1)),"wins":sum(x>0 for x in v),"values":v}
def effects(v):
 m00,m10,m01,m11=(v[x] for x in ("M00","M10","M01","M11"));return {"EF_B0":m10-m00,"EF_B1":m11-m01,"EB_F0":m01-m00,"EB_F1":m11-m10,"ME_F":((m10-m00)+(m11-m01))/2,"ME_B":((m01-m00)+(m11-m10))/2,"interaction":m11-m10-m01+m00}
def main():
 p=argparse.ArgumentParser();p.add_argument("--protocol",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();cfg=load(a.protocol);rows=load(a.output/"factorial_all_arms_metrics.json")["records"];by={(x["seed"],x["arm"]):x for x in rows}
 if len(rows)!=12 or len(by)!=12 or any((s,m) not in by for s in SEEDS for m in ("M00","M10","M01","M11")):raise RuntimeError("STAGE15A_FACTORIAL_METRICS_INCOMPLETE")
 per=[]
 for s in SEEDS:
  z={"seed":s,"overall":{}}
  for k in METRICS:z["overall"][k]=effects({m:by[s,m][k] for m in ("M00","M10","M01","M11")})
  z["flash_ap50_95"]=effects({m:by[s,m]["per_class"]["flash"]["ap50_95"] for m in ("M00","M10","M01","M11")});z["black_ap50_95"]=effects({m:by[s,m]["per_class"]["black"]["ap50_95"] for m in ("M00","M10","M01","M11")});per.append(z)
 save(a.output/"factorial_effects_by_seed.json",{"effects":per})
 overall={k:{e:summary([x["overall"][k][e] for x in per]) for e in effects({m:0 for m in ("M00","M10","M01","M11")})} for k in METRICS};save(a.output/"factorial_effects_summary.json",overall)
 flash={e:summary([x["flash_ap50_95"][e] for x in per]) for e in per[0]["flash_ap50_95"]};black={e:summary([x["black_ap50_95"][e] for x in per]) for e in per[0]["black_ap50_95"]};save(a.output/"flash_class_effect.json",flash);save(a.output/"black_class_effect.json",black)
 ef=overall["map50_95"]["EF_B0"];conditional=overall["map50_95"]["EF_B1"];interaction=overall["map50_95"]["interaction"];gates={"A_overall_mean_ge_0_005":ef["mean"]>=cfg["flash_overall_threshold"],"B_overall_wins_ge_2":ef["wins"]>=2,"C_flash_ap_mean_ge_0_01":flash["EF_B0"]["mean"]>=cfg["flash_class_threshold"],"D_flash_ap_wins_ge_2":flash["EF_B0"]["wins"]>=2,"conditional_mean_positive":conditional["mean"]>0,"conditional_wins_ge_2":conditional["wins"]>=2};confirmed=all(gates.values());black_bad=overall["map50_95"]["EB_F0"]["mean"]<0 and sum(v<0 for v in black["EB_F0"]["values"])>=2;interaction_status="WEAK_CLASS_INTERACTION" if abs(interaction["mean"])<cfg["interaction_threshold"] else "MATERIAL_CLASS_INTERACTION"
 if confirmed:status="CLASS_DEPENDENT_MORPHOLOGY_BOTTLENECK_CONFIRMED" if black_bad else "FLASH_MORPHOLOGY_CAUSAL_SIGNAL_CONFIRMED"
 elif all(gates[k] for k in ("A_overall_mean_ge_0_005","B_overall_wins_ge_2","C_flash_ap_mean_ge_0_01","D_flash_ap_wins_ge_2")) and interaction_status=="MATERIAL_CLASS_INTERACTION" and not (gates["conditional_mean_positive"] and gates["conditional_wins_ge_2"]):status="CLASS_INTERACTION_DOMINATES_MORPHOLOGY_SIGNAL"
 else:status="FLASH_MORPHOLOGY_CAUSAL_SIGNAL_NOT_CONFIRMED"
 save(a.output/"interaction_analysis.json",{"status":interaction_status,"primary_map50_95":interaction,"threshold":cfg["interaction_threshold"]});state={"status":status,"flash_gates":gates,"FLASH_MORPHOLOGY_CAUSAL_SIGNAL_CONFIRMED":confirmed,"black_status":"BLACK_MORPHOLOGY_ALIGNMENT_NOT_BENEFICIAL" if black_bad else "BLACK_MORPHOLOGY_EFFECT_NOT_NEGATIVE_BY_GATE","interaction_status":interaction_status,"STAGE15B_CLASS_ADAPTIVE_GENERATOR_DESIGN":confirmed,"STAGE15B_AUTO_START":False,"new_synthetic_generation":0,"generator_training":0,"deep_pcb_training":0,"deep_pcb_generation":0};save(a.output/"stage15a_status.json",state)
 report=f"""# Plastic_Bomo Stage15A Report\n\n## Status\n\n`{status}`\n\nThis experiment isolates **morphology alignment**, not broad perceptual fidelity. M00 and M11 reuse Stage14B; only M10 and M01 were newly trained.\n\n## Primary effects (mAP50-95)\n\n- Flash under Random-Black (M10-M00): {ef['mean']*100:+.3f} ± {ef['std']*100:.3f} pp; wins {ef['wins']}/3.\n- Flash under High-Black (M11-M01): {conditional['mean']*100:+.3f} ± {conditional['std']*100:.3f} pp; wins {conditional['wins']}/3.\n- Black under Random-Flash (M01-M00): {overall['map50_95']['EB_F0']['mean']*100:+.3f} pp.\n- Interaction: {interaction['mean']*100:+.3f} pp (`{interaction_status}`).\n\n## Class effects\n\n- Flash AP50-95 M10-M00: {flash['EF_B0']['mean']*100:+.3f} pp; wins {flash['EF_B0']['wins']}/3.\n- Black AP50-95 M01-M00: {black['EB_F0']['mean']*100:+.3f} pp; positive wins {black['EB_F0']['wins']}/3.\n\nNo Stage15B experiment was automatically started.\n"""; (a.output/"STAGE15A_REPORT.md").write_text(report,encoding="utf-8");print(json.dumps(state,indent=2))
if __name__=="__main__":main()
