"""Diagnostic-only Vanilla SD2 UNet parameter drift from the common base."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import torch
from diffusers import UNet2DConditionModel

CLASSES=("open","short","mousebite","spur","spurious_copper","pinhole")
def group(name):
 if "attn" in name or "transformer" in name:return "attention"
 if "down_blocks" in name:return "down"
 if "mid_block" in name:return "mid"
 if "up_blocks" in name:return "up"
 if "conv" in name:return "convolution"
 return "other"
def main():
 p=argparse.ArgumentParser();p.add_argument("--base_model",type=Path,required=True);p.add_argument("--sd2_models",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);base=UNet2DConditionModel.from_pretrained(a.base_model,subfolder="unet",torch_dtype=torch.float32);base_state={k:v.detach().cpu() for k,v in base.state_dict().items()};del base;result={}
 for cls in CLASSES:
  model=UNet2DConditionModel.from_pretrained(a.sd2_models/cls,subfolder="unet",torch_dtype=torch.float32);acc={}
  for name,value in model.state_dict().items():
   delta=(value.detach().cpu()-base_state[name]).float();g=group(name);row=acc.setdefault(g,{"squared_delta":0.0,"squared_base":0.0,"absolute_delta":0.0,"parameters":0});row["squared_delta"]+=float(delta.square().sum());row["squared_base"]+=float(base_state[name].float().square().sum());row["absolute_delta"]+=float(delta.abs().sum());row["parameters"]+=delta.numel()
  total={k:sum(x[k] for x in acc.values()) for k in ("squared_delta","squared_base","absolute_delta","parameters")}
  def finish(x):return {"relative_l2":(x["squared_delta"]/max(x["squared_base"],1e-30))**.5,"mean_absolute_delta":x["absolute_delta"]/x["parameters"],"parameters":x["parameters"]}
  result[cls]={"global":finish(total),"groups":{k:finish(v) for k,v in acc.items()}};del model
 (a.output/"unet_parameter_drift.json").write_text(json.dumps({"diagnostic_only":True,"not_a_quality_score":True,"base_model":str(a.base_model),"classes":result},indent=2));print(json.dumps({c:result[c]["global"] for c in CLASSES},indent=2))
if __name__=="__main__":main()
