"""Re-evaluate Gate-3 with successful-update and stable-tail semantics."""
from __future__ import annotations
import argparse,json
from pathlib import Path

CLASSES=("short","pinhole")
def load(p):return json.loads(Path(p).read_text(encoding="utf-8"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False),encoding="utf-8")
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();doc=load(a.output/"tmvb_50step_stability.json");audit={};passed=True
 for cls in CLASSES:
  x=doc["classes"][cls];rows=x["records"];tail=rows[-20:];checks={"fifty_successful_updates":x["successful_steps"]==x["target_successful_steps"]==50,"parameter_changed":x["parameter_delta_l2"]>0,"grad_scaler_nonzero":bool(rows) and rows[-1]["scale_after"]>0,"last_20_attempts_all_successful":len(tail)==20 and all(not r["skipped"] and r["gradient_finite"] and r["parameter_changed"] for r in tail),"checkpoint_not_saved":x["checkpoint_saved"] is False};ok=all(checks.values());passed&=ok;audit[cls]={"status":"PASS" if ok else "FAIL","checks":checks,"attempted_steps":x["attempted_steps"],"successful_steps":x["successful_steps"],"amp_guarded_skips":x["skipped_steps"],"last_skip_attempt":max((r["attempt"] for r in rows if r["skipped"]),default=None),"consecutive_successful_tail":next((i for i,r in enumerate(reversed(rows)) if r["skipped"]),len(rows)),"final_grad_scaler_scale":rows[-1]["scale_after"],"parameter_delta_l2":x["parameter_delta_l2"]}
 status="TMVB_50STEP_STABLE" if passed else "TMVB_50STEP_UNSTABLE";doc["status"]=status;doc["stability_acceptance"]={"rule":"50 successful updates, nonzero parameter delta, nonzero scaler, and final 20 attempts finite/unskipped/updated","audit":audit,"early_amp_guarded_skips_are_not_successful_updates":True};save(a.output/"tmvb_50step_stability.json",doc);gate=load(a.output/"stage10b_gate.json");gate.update({"status":"GATE3_PASSED_READY_FOR_FORMAL_TRAINING" if passed else status,"GATE3_PASSED":passed,"FORMAL_TRAINING_AUTHORIZED":passed,"STAGE10C_DETECTOR_PILOT_AUTHORIZED":False});save(a.output/"stage10b_gate.json",gate);save(a.output/"gate3_stability_recheck.json",{"status":status,"audit":audit,"formal_checkpoint_saved":False});print(json.dumps({"status":status,"audit":audit},indent=2));
 if not passed:raise RuntimeError(status)
if __name__=="__main__":main()
