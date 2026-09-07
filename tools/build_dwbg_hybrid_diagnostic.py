"""Build the post-hoc Flash-DWBG + Black-Random diagnostic dataset."""
from __future__ import annotations
import argparse,json,shutil
from pathlib import Path
try: from .build_dwbg_v2_final_datasets import add,yaml
except ImportError: from build_dwbg_v2_final_datasets import add,yaml
def main():
 p=argparse.ArgumentParser();p.add_argument('--real_only',type=Path,required=True);p.add_argument('--hybrid_manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 m=json.loads(a.hybrid_manifest.read_text());s=m['selected']; assert len([x for x in s if x['class_id']==0])==40 and len([x for x in s if x['class_id']==1])==40
 shutil.copytree(a.real_only,a.output); counts=add(a.hybrid_manifest,a.output,'syn'); (a.output.parent/'real_hybrid_flash_dwbg_black_random.yaml').write_text(yaml(a.output,'hybrid'))
 print(json.dumps({'synthetic':counts,'real_train_copied_from':str(a.real_only),'validation_copied_unchanged':True},indent=2))
if __name__=='__main__':main()
