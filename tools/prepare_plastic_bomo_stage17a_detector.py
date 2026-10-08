"""Prepare four matched detector datasets without scores or training."""
import argparse
from collections import Counter
from pathlib import Path
import shutil
from plastic_bomo_stage17a_detector_common import ARMS, load, save, sha, digest, population, detector_protocol, verify_generation
from prepare_plastic_bomo_stage17a_trainonly import yaml_directories, verify_inputs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('protocol','banks','slots','preprobe','frozen','prepared_generator','real_data_yaml','dataset_root','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args(); a.dataset_root=a.dataset_root.resolve(); a.output=a.output.resolve()
    cfg=load(a.protocol); detector_protocol(cfg)
    slots,records=verify_generation(a.banks,a.slots,a.preprobe)
    dirs=yaml_directories(a.real_data_yaml)
    rows,train,val=verify_inputs(a.frozen,dirs['train'],dirs['val'])
    isolation=load(a.prepared_generator/'source_isolation_audit.json')
    if sha(a.frozen)!=isolation['frozen_manifest_sha256'] or train!=isolation['train_images']:
        raise RuntimeError('REAL_TRAIN_DIFFERS_FROM_FROZEN_TRAINONLY_GENERATOR_POOL')
    if sha(a.real_data_yaml)!=isolation['official_data_yaml_sha256']:
        raise RuntimeError('OFFICIAL_DATA_YAML_DIFFERS_FROM_TRAINONLY_ISOLATION')
    if val!=isolation['validation_images']: raise RuntimeError('VALIDATION_IMAGE_SET_DIFFERS_FROM_TRAINONLY_ISOLATION')
    if a.output.exists() or a.dataset_root.exists(): raise RuntimeError('USE_NEW_OUTPUT_AND_DATASET_DIRECTORY_NO_OVERWRITE')
    a.output.mkdir(parents=True); a.dataset_root.mkdir(parents=True)
    bases=sorted(train); extra_order=[s['slot_id'] for s in slots]
    by={(r['bank'],r['slot_id']):r for r in records}
    assets={}; yaml_hashes={}; index={}
    for arm in ARMS:
        root=a.dataset_root/arm; images=root/'images/train'; labels=root/'labels/train'
        images.mkdir(parents=True); labels.mkdir(parents=True); mapping=[]
        for i,name in enumerate(bases):
            stem=f'base_{i:03d}'; dest=images/(stem+Path(name).suffix)
            shutil.copyfile(dirs['train']/name,dest)
            shutil.copyfile(dirs['train'].parent.parent/'labels/train'/(Path(name).stem+'.txt'),labels/(stem+'.txt'))
            mapping.append({'position':stem,'role':'base','real_parent':name})
        for i,s in enumerate(slots):
            stem=f'extra_{i:03d}'; label=labels/(stem+'.txt')
            if arm=='RR':
                parent=s['real_parent']; dest=images/(stem+Path(parent).suffix)
                shutil.copyfile(dirs['train']/parent,dest)
                shutil.copyfile(dirs['train'].parent.parent/'labels/train'/(Path(parent).stem+'.txt'),label)
            else:
                r=by[arm[-1],s['slot_id']]; dest=images/(stem+'.png')
                source=a.banks/'formal'/arm[-1]/Path(r['image_path'].replace('\\','/')).name
                shutil.copyfile(source,dest)
                lines=[]
                for box in s['synthetic_labels']:
                    xywh=box['yolo_xywh']
                    if min(xywh[2:])<=0 or not all(0<=v<=1 for v in xywh): raise RuntimeError('INVALID_FROZEN_PROJECTED_LABEL')
                    lines.append(str(box['class_id'])+' '+' '.join(format(v,'.17g') for v in xywh))
                label.write_text('\n'.join(lines)+'\n',encoding='utf-8')
            mapping.append({'position':stem,'role':'extra','slot_id':s['slot_id'],'class_role':s['class'],'real_parent':s['real_parent']})
        import yaml
        # Internal validation refers to TRAIN ONLY, even if the framework constructs a val loader.
        (root/'data.yaml').write_text(yaml.safe_dump({'path':str(root),'train':'images/train','val':'images/train','names':{0:'flash',1:'black'}},sort_keys=False),encoding='utf-8')
        assets[arm]=population(images); yaml_hashes[arm]=sha(root/'data.yaml'); index[arm]=mapping
    protocol=dict(cfg,checkpoint_root='model/baseline_stage17a_trainonly_s42_noise_0.0',
                  baseline_identity='user-authorized train-only rebuild; not recovered historical baseline',
                  conditioning_policy='frozen user-authorized 174-normal isolated pool; unchanged baseline filter',
                  training_order='same positional metadata order; Random(42+zero_based_epoch) shuffle; all primary draws exactly once per epoch',
                  old_realrepeat_reused=False)
    save(a.output/'stage17a_protocol.json',protocol)
    manifest={'dataset_root':str(a.dataset_root),'official_data_yaml':str(a.real_data_yaml.resolve()),
              'official_data_yaml_sha256':sha(a.real_data_yaml),'official_val_images':str(dirs['val']),
              'official_val_assets':population(dirs['val']),'frozen_train_assets':train,
              'dataset_assets':assets,'training_yaml_sha256':yaml_hashes,'positions':index,
              'slot_manifest_sha256':sha(a.slots/'generation_slot_manifest.json'),
              'bank_manifest_sha256':sha(a.banks/'source_seed_manifest.json'),
              'protocol_content_sha256':digest(protocol),'draws_per_epoch':138+len(slots),
              'extra_class_role_counts':dict(Counter(s['class'] for s in slots))}
    save(a.output/'detector_dataset_manifest.json',manifest)
    save(a.output/'realrepeat_equivalence_audit.json',{'status':'NEW_MATCHED_RR_REQUIRED_AND_PREPARED',
         'historical_reuse_blocked_reason':'no exact order/runtime/validation-timing equivalence evidence',
         'base_images':138,'base_annotations':len(rows),'extra_full_original_image_replays':len(slots),
         'extra_class_role_counts':manifest['extra_class_role_counts'],
         'full_original_labels_repeated':True,'target_isolation':False,
         'class_role_not_equal_actual_bbox_count':'full image replay can contain multiple annotations; synthetic crops contain 85 projected labels; do not claim exact bbox-instance exposure equality',
         'same_draw_budget':True,'runtime_compute_pending':True})
    save(a.output/'detector_preparation_status.json',{'status':'MATCHED_FOUR_ARM_DATASETS_PREPARED',
         'labels_modified':False,'training_count':0,'official_validation_model_forwards':0,
         'validation_labels_used_for_selection':False,'stage17b_auto_start':False})
    print('MATCHED_FOUR_ARM_DATASETS_PREPARED',flush=True)

if __name__=='__main__': main()
