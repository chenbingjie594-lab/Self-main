"""Make only the frozen Flash-DWBG + Black-Random post-hoc diagnostic manifest."""
from __future__ import annotations
import argparse,json
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--dwbg',type=Path,required=True);p.add_argument('--random',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.dwbg.read_text());r=json.loads(a.random.read_text());flash=[x for x in d['selected'] if x['class_id']==0];black=[x for x in r['selected'] if x['class_id']==1]
 assert len(flash)==40 and len(black)==40
 out={'purpose':'exploratory diagnostic only','defined_after_observing_s42_downstream_results':True,'post_hoc':True,'single_seed':True,'primary_claim_eligible':False,'flash_source':'DWBG-v2 frozen selection','black_source':'Random-matched frozen selection','flash_identity_match':{x['candidate_id'] for x in flash}=={x['candidate_id'] for x in d['selected'] if x['class_id']==0},'black_identity_match':{x['candidate_id'] for x in black}=={x['candidate_id'] for x in r['selected'] if x['class_id']==1},'selected':flash+black,'counts':{'flash':40,'black':40}}
 assert out['flash_identity_match'] and out['black_identity_match'];a.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({k:out[k] for k in ('flash_identity_match','black_identity_match','counts')},indent=2))
if __name__=='__main__':main()
