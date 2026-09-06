"""Build frozen Real/Random/DWBG YOLO datasets; never changes validation."""
from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
IMG={'.jpg','.jpeg','.png','.bmp'}
def files(p): return sorted(x for x in Path(p).iterdir() if x.suffix.lower() in IMG)
def copytree(src,dst): shutil.copytree(src,dst,dirs_exist_ok=False)
def label_class(path):
 lines=Path(path).read_text().strip().splitlines()
 ids={int(x.split()[0]) for x in lines if x.strip()}
 if len(ids)!=1: raise ValueError(f'non-single-class label: {path}')
 return ids.pop()
def yaml(root,name):
 return f'path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: Flash point\n  1: Big black spots\n'
def add(manifest,root,prefix):
 d=json.loads(Path(manifest).read_text()); counts={0:0,1:0}
 for x in d['selected']:
  image,label=Path(x['image_path']),Path(x['label_path'])
  if not image.is_file() or not label.is_file(): raise FileNotFoundError(f'{image} / {label}')
  cid=label_class(label)
  if cid!=int(x['class_id']): raise ValueError(f'class mismatch: {x["candidate_id"]}')
  stem=f'{prefix}_{x["candidate_id"]}'; shutil.copy2(image,root/'images/train'/f'{stem}{image.suffix.lower()}'); shutil.copy2(label,root/'labels/train'/f'{stem}.txt'); counts[cid]+=1
 if counts!={0:40,1:40}: raise ValueError(f'synthetic counts invalid: {counts}')
 return counts
def main():
 p=argparse.ArgumentParser(); p.add_argument('--real_root',type=Path,required=True); p.add_argument('--random_manifest',type=Path,required=True); p.add_argument('--dwbg_manifest',type=Path,required=True); p.add_argument('--output_root',type=Path,required=True); a=p.parse_args()
 if a.output_root.exists(): raise FileExistsError(a.output_root)
 for needed in ('images/train','labels/train','images/val','labels/val'):
  if not (a.real_root/needed).is_dir(): raise FileNotFoundError(a.real_root/needed)
 a.output_root.mkdir(parents=True)
 groups={'real_only':None,'real_random_matched':a.random_manifest,'real_dwbg_v2':a.dwbg_manifest}
 report={}
 for name,manifest in groups.items():
  target=a.output_root/name; copytree(a.real_root,target)
  syn={0:0,1:0} if manifest is None else add(manifest,target,'syn')
  (a.output_root/f'{name}.yaml').write_text(yaml(target,name))
  report[name]={'synthetic':{'flash':syn[0],'black':syn[1]},'real_train_images':len(files(target/'images/train'))-sum(syn.values()),'val_images':len(files(target/'images/val'))}
 (a.output_root/'manifests').mkdir(); shutil.copy2(a.random_manifest,a.output_root/'manifests/random_matched_manifest_40_40_s42.json'); shutil.copy2(a.dwbg_manifest,a.output_root/'manifests/dwbg_v2_manifest_40_40.json')
 (a.output_root/'build_report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))
if __name__=='__main__': main()
