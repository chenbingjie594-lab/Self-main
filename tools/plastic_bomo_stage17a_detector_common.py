"""CPU-only invariants for the frozen Stage17A detector experiment."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from types import FunctionType
from prepare_plastic_bomo_stage17a_trainonly import load, save, sha, EXTS

ARMS = ('RR','VA','VB','VC')

def function_with_global(function, name, expected, replacement):
    """Reuse the installed builder verbatim with a private constructor binding."""
    if not isinstance(function,FunctionType) or function.__globals__.get(name) is not expected:
        raise RuntimeError('UNSUPPORTED_NATIVE_DATASET_BUILDER_BINDING:'+name)
    namespace=dict(function.__globals__); namespace[name]=replacement
    copied=FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    copied.__kwdefaults__=function.__kwdefaults__
    return copied

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def schedule(names, epoch):
    result = sorted(names)
    random.Random(42 + int(epoch)).shuffle(result)
    return result

def detector_protocol(cfg):
    expected = {'model':'pretrained/yolo11s.pt','seed':42,'imgsz':1536,'epochs':150,
                'patience':151,'batch':1,'optimizer':'auto','mosaic':1.0,'mixup':0.0,
                'copy_paste':0.0,'close_mosaic':10,'deterministic':True,'rect':False,
                'augment':False,'validation_during_training':False,'primary_checkpoint':'epoch150 last.pt'}
    if cfg['detector'] != expected or cfg['arms'] != list(ARMS):
        raise RuntimeError('FROZEN_STAGE17A_DETECTOR_PROTOCOL_CHANGED')
    if cfg['gates'] != {'mean_gain_min_raw':.005,'wins_min':2,'worst_mean_class_gain_min_raw':-.02}:
        raise RuntimeError('FROZEN_STAGE17A_GATES_CHANGED')
    return expected

def population(images):
    images=Path(images); labels=images.parent.parent/'labels'/images.name
    return [{'name':p.name,'image_sha256':sha(p),'label_sha256':sha(labels/(p.stem+'.txt'))}
            for p in sorted(images.iterdir()) if p.is_file() and p.suffix.lower() in EXTS]

def verify_generation(banks, slots_dir, preprobe):
    from generate_plastic_bomo_stage17a_frozen_banks import verify_preprobe, reproducibility
    from PIL import Image
    path=slots_dir/'generation_slot_manifest.json'; frozen=load(path); slot_hash=sha(path)
    status=load(banks/'formal_generation_status.json'); manifest=load(banks/'source_seed_manifest.json')
    if status['status']!='FROZEN_BANK_GENERATION_AND_REPRO_PASS' or manifest['slot_manifest_sha256']!=slot_hash:
        raise RuntimeError('PAIRED_GENERATION_PASS_REQUIRED')
    _,pre=verify_preprobe(preprobe,frozen['slots'],slot_hash)
    kept=set(manifest['retained_slot_ids']); slots=[s for s in frozen['slots'] if s['slot_id'] in kept]
    by={s['slot_id']:s for s in slots}; records=manifest['records']
    if len(slots)!=len(kept) or any(Counter(s['class'] for s in slots)[c]<35 for c in ('flash','black')):
        raise RuntimeError('INVALID_RETAINED_SLOTS')
    for bank in 'ABC':
        rows=[r for r in records if r['bank']==bank]
        if len(rows)!=len(slots) or {r['slot_id'] for r in rows}!=kept:
            raise RuntimeError('BANK_PAIRING_INVALID')
    post=load(banks/'post_generation_probe_records.json')['records']
    for phase,rows in (('formal',records),('post_probe',post)):
        for r in rows:
            f=banks/phase/r['bank']/Path(r['image_path'].replace('\\','/')).name
            if sha(f)!=r['image_file_sha256']: raise RuntimeError('GENERATED_FILE_CHANGED')
            with Image.open(f) as im:
                if im.mode!='RGB' or im.size!=(512,512) or hashlib.sha256(im.tobytes()).hexdigest()!=r['rgb_sha256']:
                    raise RuntimeError('GENERATED_RGB_CHANGED')
            s=by[r['slot_id']]
            if any(r[k]!=s[k] for k in ('class','source_id','real_parent','conditioning_id','synthetic_labels')) or r['sample_seed']!=s['bank_sample_seeds'][r['bank']]:
                raise RuntimeError('SLOT_MAPPING_CHANGED')
    audit=reproducibility(pre,records,post)
    if audit['status']!='PRE_FORMAL_POST_RGB_EXACT_PASS' or audit!=load(banks/'generation_reproducibility_audit.json'):
        raise RuntimeError('GENERATION_REPRODUCIBILITY_FAILED')
    return sorted(slots,key=lambda s:s['slot_id']), records

def verify_prepared(prepared):
    manifest=load(prepared/'detector_dataset_manifest.json')
    protocol=load(prepared/'stage17a_protocol.json'); detector_protocol(protocol)
    if digest(protocol)!=manifest['protocol_content_sha256']: raise RuntimeError('PROTOCOL_CHANGED')
    for arm in ARMS:
        images=Path(manifest['dataset_root'])/arm/'images/train'
        if population(images)!=manifest['dataset_assets'][arm]: raise RuntimeError('PREPARED_DATASET_CHANGED:'+arm)
        if sha(images.parent.parent/'data.yaml')!=manifest['training_yaml_sha256'][arm]: raise RuntimeError('TRAINING_YAML_CHANGED')
    if sha(manifest['official_data_yaml'])!=manifest['official_data_yaml_sha256']:
        raise RuntimeError('OFFICIAL_DATA_YAML_CHANGED')
    if population(manifest['official_val_images'])!=manifest['official_val_assets']:
        raise RuntimeError('OFFICIAL_VALIDATION_CHANGED')
    return protocol,manifest

class EpochSampler:
    def __init__(self,names):
        self.names=tuple(names); self.index={n:i for i,n in enumerate(names)}; self.epoch=0
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __len__(self): return len(self.names)
    def __iter__(self): return iter(self.index[n] for n in schedule(self.names,self.epoch))

class TraceDatasetMixin:
    """Carry secondary augmentation fetches through collate, not worker side effects."""
    def get_image_and_label(self,index):
        if getattr(self,'_stage17_calls',None) is not None:
            self._stage17_calls.append(Path(self.im_files[index]).stem)
        return super().get_image_and_label(index)
    def __getitem__(self,index):
        self._stage17_calls=[]
        try:
            result=super().__getitem__(index)
            result['stage17_fetches']=tuple(self._stage17_calls)
            return result
        finally: self._stage17_calls=None

def headroom(rows):
    import statistics
    by={r['arm']:r for r in rows}
    if set(by)!=set(ARMS) or len(rows)!=4: raise RuntimeError('FOUR_FINAL_METRICS_REQUIRED')
    values=[by[a]['map50_95'] for a in ARMS[1:]]
    deltas=[v-by['RR']['map50_95'] for v in values]
    classes={c:statistics.mean(by[a]['per_class'][c]['ap50_95']-by['RR']['per_class'][c]['ap50_95'] for a in ARMS[1:]) for c in ('flash','black')}
    gates={'A_mean_gain':statistics.mean(deltas)>=.005,'B_two_wins':sum(d>0 for d in deltas)>=2,'C_class_safety':min(classes.values())>=-.02}
    status=('SYNTHETIC_HEADROOM_NOT_REPRODUCIBLE' if not(gates['A_mean_gain'] and gates['B_two_wins']) else
            'SYNTHETIC_HEADROOM_CLASS_UNSTABLE' if not gates['C_class_safety'] else 'VANILLA_SYNTHETIC_HEADROOM_REPRODUCIBLE')
    return {'status':status,'gates':gates,'bank_deltas_raw':dict(zip('ABC',deltas)),
            'mean_delta':statistics.mean(deltas),'std_delta':statistics.stdev(deltas),
            'min_delta':min(deltas),'max_delta':max(deltas),'wins_vs_RR':sum(d>0 for d in deltas),
            'mean_class_deltas_raw':classes,'vanilla_std_pp':100*statistics.stdev(values),
            'vanilla_range_pp':100*(max(values)-min(values)),
            'future_method_min_gain_pp':max(.5,200*statistics.stdev(values)),
            'std_ddof':1,'detector_seed':42,'GENERATOR_MECHANISM_DEVELOPMENT_AUTHORIZED':all(gates.values()),'stage17b_auto_start':False}
