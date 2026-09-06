"""Write immutable DWBG-v2 final experiment metadata and manifest hashes."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--dwbg',type=Path,required=True); p.add_argument('--random',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 d=json.loads(a.dwbg.read_text()); r=json.loads(a.random.read_text())
 for cid in (0,1):
  s=[x for x in d['selected'] if x['class_id']==cid]; h=[x for x in s if x['selection_group']=='hard']; an=[x for x in s if x['selection_group']=='anchor']; assert len(s)==40 and len(h)==28 and len(an)==12
  assert all(x['manifold_valid'] and x['median_iou']>=.5 for x in h); assert cid or all(x['median_confidence']>=.05 for x in h); assert all(x['median_iou']>=.5 for x in an)
 assert [sum(x['class_id']==i for x in r['selected']) for i in (0,1)]==[40,40]
 out={'experiment_version':'DWBG-v2-final-s42','candidate_pool_frozen':True,'real_train_instances':{'flash':88,'black':80},'synthetic_count':{'random_matched':{'flash':40,'black':40},'dwbg_v2':{'flash':40,'black':40}},'dwbg_hard_anchor':{'flash':{'hard':28,'anchor':12},'black':{'hard':28,'anchor':12}},'final_hard_gate':{'flash':{'manifold_valid':True,'min_median_iou':.5,'min_median_confidence':.05},'black':{'manifold_valid':True,'min_median_iou':.5}},'random_seed':42,'selection_frozen_before_downstream_validation':True,'msdf_modified':False,'yolo_architecture_modified':False,'real_validation_modified':False,'dwbg_manifest_sha256':sha(a.dwbg),'random_manifest_sha256':sha(a.random)}
 a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
if __name__=='__main__': main()
