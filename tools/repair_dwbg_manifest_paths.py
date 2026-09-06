"""Restore image/label path metadata in a frozen manifest without reselection."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser(); p.add_argument('--manifest',type=Path,required=True); p.add_argument('--pool',type=Path,required=True); a=p.parse_args()
 manifest=json.loads(a.manifest.read_text(encoding='utf-8')); pool=json.loads(a.pool.read_text(encoding='utf-8'))
 index={x['candidate_id']:x for x in pool['candidates']}
 before=[x['candidate_id'] for x in manifest['selected']]
 for item in manifest['selected']:
  source=index.get(item['candidate_id'])
  if source is None: raise KeyError(f'candidate absent from frozen pool: {item["candidate_id"]}')
  item['image_path']=source['image_path']; item['label_path']=source['label_path']
 after=[x['candidate_id'] for x in manifest['selected']]
 assert before==after and len(before)==80
 manifest['path_metadata_restored_from_frozen_pool']=True
 a.manifest.write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 print(f'Restored paths for {len(before)} frozen candidates; IDs unchanged.')
if __name__=='__main__': main()
