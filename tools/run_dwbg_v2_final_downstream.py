"""Print (or explicitly run) frozen, matched YOLO11s downstream commands."""
from __future__ import annotations
import argparse, subprocess
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset_root',type=Path,required=True); p.add_argument('--experiment',choices=['real_only','random_matched','dwbg_v2','all'],default='all'); p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='0'); p.add_argument('--run',action='store_true'); a=p.parse_args()
 mapping={'real_only':'real_only','random_matched':'real_random_matched','dwbg_v2':'real_dwbg_v2'}; names=[a.experiment] if a.experiment!='all' else list(mapping)
 for exp in names:
  group=mapping[exp]; cmd=['yolo','detect','train','model=pretrained/yolo11s.pt',f'data={a.dataset_root/(group+".yaml")}', 'epochs=150','imgsz=1536','batch=1','rect=False',f'device={a.device}','workers=4',f'seed={a.seed}','deterministic=True','patience=40','project=runs/dwbg_v2_final_40_40',f'name={group}_s{a.seed}','exist_ok=False']
  print(' '.join(map(str,cmd)))
  if a.run: subprocess.run(cmd,check=True)
if __name__=='__main__': main()
