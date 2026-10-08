"""Train a new baseline, never replace or claim identity to historical weights."""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
CLASSES={'flash':'01_Flash_point','black':'02_Big_black_spots'}


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def load(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def command(base, images, masks, output, ledger):
    return [sys.executable,str(REPO/'train_dreambooth_noise.py'),
            '--pretrained_model_name_or_path',str(base),'--instance_data_dir',str(images),
            '--mask_data_dir',str(masks),'--output_dir',str(output),
            '--instance_prompt','a photo of a sks defect','--resolution','512',
            '--train_batch_size','1','--gradient_accumulation_steps','4',
            '--learning_rate','0.000001','--lr_scheduler','constant','--lr_warmup_steps','0',
            '--max_train_steps','2000','--text_noise_scale','0.0','--seed','42',
            '--mixed_precision','fp16','--gradient_checkpointing','--mvtecad','--center_crop',
            '--training_audit_path',str(ledger)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base_model','train_root','split_manifest','model_root','work_root','output','logs'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--prepare_only',action='store_true')
    a=p.parse_args()
    for name in ('base_model','train_root','split_manifest','model_root','work_root','output','logs'):
        setattr(a,name,getattr(a,name).resolve())
    if a.model_root.name!='baseline_stage17a_rebuild_split70_s42_noise_0.0':
        raise RuntimeError('NEW_BASELINE_DIRECTORY_REQUIRED')
    for path in (a.model_root,a.work_root,a.output):
        if path.exists():raise RuntimeError('OUTPUT_EXISTS_USE_CLEAN_NEW_DIRECTORY: '+str(path))
    split=load(a.split_manifest)
    if split.get('seed')!=42 or split.get('train_ratio')!=0.7:raise RuntimeError('FROZEN_SPLIT_MISMATCH')
    for part in ('model_index.json','unet/config.json','vae/config.json','text_encoder/config.json','tokenizer/tokenizer_config.json','tokenizer/vocab.json','tokenizer/merges.txt','scheduler/scheduler_config.json'):
        f=a.base_model/part
        if not f.is_file() or not f.stat().st_size:raise RuntimeError('BASE_MODEL_INCOMPLETE: '+part)
        if f.suffix=='.json':load(f)
    for part in ('unet','vae','text_encoder'):
        weights=list((a.base_model/part).glob('*.safetensors'))+list((a.base_model/part).glob('*.bin'))
        if not weights or any(not f.stat().st_size for f in weights):raise RuntimeError('BASE_MODEL_WEIGHTS_MISSING_OR_EMPTY:'+part)
    unet=load(a.base_model/'unet/config.json')
    if unet.get('in_channels')!=9:raise RuntimeError('NINE_CHANNEL_INPAINTING_BASE_REQUIRED')
    pairs={}
    for cls,folder in CLASSES.items():
        entry=split['categories']['Plastic_Bomo'][folder];rows=entry['train']
        if {r['image'] for r in rows}&{r['image'] for r in entry['eval']}:raise RuntimeError('TRAIN_EVAL_FILENAME_OVERLAP')
        if not rows:raise RuntimeError('EMPTY_TRAIN_SOURCE_POOL')
        pairs[cls]=[]
        for index,r in enumerate(rows):
            for k in ('image','mask'):
                if Path(r[k]).name!=r[k] or '/' in r[k] or '\\' in r[k]:raise RuntimeError('UNSAFE_MANIFEST_FILENAME')
            image=a.train_root/'Plastic_Bomo/test'/folder/r['image']
            mask=a.train_root/'Plastic_Bomo/ground_truth'/folder/r['mask']
            if not image.is_file() or not mask.is_file():raise RuntimeError('SOURCE_PAIR_MISSING:'+str(image))
            pairs[cls].append({'image':str(image),'mask':str(mask),'image_sha256':sha(image),'mask_sha256':sha(mask),
                               'staging_id':f'{index:05d}','staging_image_extension':image.suffix.lower(),'staging_mask_extension':mask.suffix.lower()})
    # No held-out image/mask content read; only existing train paths above are hashed.
    a.output.mkdir(parents=True);a.work_root.mkdir(parents=True);a.logs.mkdir(parents=True,exist_ok=True)
    base_files={str(p.relative_to(a.base_model)):sha(p) for p in sorted(a.base_model.rglob('*')) if p.is_file()}
    audit={'status':'PREPARED_NOT_TRAINED','baseline_identity':'NEW_REBUILD_NOT_HISTORICAL_RECOVERY',
           'split_manifest_sha256':sha(a.split_manifest),'base_model':str(a.base_model),'base_model_files_sha256':base_files,
           'training_entrypoint_sha256':sha(REPO/'train_dreambooth_noise.py'),'source_pairs':pairs,
           'generation_count':0,'detector_training_count':0,'validation_content_read':0,'labels_modified':False,
           'model_root':str(a.model_root),'commands':{}}
    for cls,folder in CLASSES.items():
        images=a.work_root/cls/'images';masks=a.work_root/cls/'masks';images.mkdir(parents=True);masks.mkdir()
        for row in pairs[cls]:
            shutil.copyfile(row['image'],images/(row['staging_id']+row['staging_image_extension']))
            shutil.copyfile(row['mask'],masks/(row['staging_id']+row['staging_mask_extension']))
            if sha(images/(row['staging_id']+row['staging_image_extension']))!=row['image_sha256'] or sha(masks/(row['staging_id']+row['staging_mask_extension']))!=row['mask_sha256']:
                raise RuntimeError('STAGED_SOURCE_HASH_MISMATCH')
        audit['commands'][cls]=command(a.base_model,images,masks,a.model_root/'Plastic_Bomo'/folder,a.output/f'{cls}_training_runtime.json')
    save(a.output/'baseline_rebuild_preparation_audit.json',audit)
    save(a.output/'baseline_rebuild_status.json',{'status':audit['status'],'generation_authorized':False,'detector_training_authorized':False})
    if a.prepare_only:
        print('PREPARED_NOT_TRAINED; use a separate clean work/output directory for actual launch.');return
    for cls,folder in CLASSES.items():
        log=a.logs/f'train_{cls}.log'
        if log.exists():raise RuntimeError('LOG_EXISTS: '+str(log))
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run(audit['commands'][cls],cwd=REPO,stdout=stream,stderr=subprocess.STDOUT,check=True)
        runtime=load(a.output/f'{cls}_training_runtime.json')
        if runtime['status']!='FINAL_PIPELINE_SAVED' or runtime['global_step']!=2000 or runtime['seed']!=42 or runtime['train_text_encoder']:
            raise RuntimeError('TRAINING_RUNTIME_PROTOCOL_MISMATCH')
        folder_path=a.model_root/'Plastic_Bomo'/folder
        files={str(p.relative_to(folder_path)):sha(p) for p in sorted(folder_path.rglob('*')) if p.is_file() and 'logs' not in p.relative_to(folder_path).parts}
        if any(not p.stat().st_size for p in folder_path.rglob('*.safetensors')):raise RuntimeError('EMPTY_SAVED_WEIGHTS')
        save(a.output/f'{cls}_checkpoint_asset_audit.json',{'checkpoint':str(folder_path),'files_sha256':files,'runtime':runtime})
        print(cls+' new baseline trained and saved',flush=True)
    for rows in pairs.values():
        for row in rows:
            if sha(row['image'])!=row['image_sha256'] or sha(row['mask'])!=row['mask_sha256']:raise RuntimeError('SOURCE_CHANGED')
    if sha(a.split_manifest)!=audit['split_manifest_sha256']:raise RuntimeError('SPLIT_CHANGED')
    if any(sha(a.base_model/k)!=v for k,v in base_files.items()):raise RuntimeError('BASE_MODEL_CHANGED')
    save(a.output/'baseline_rebuild_status.json',{'status':'NEW_BASELINE_TRAINING_COMPLETE_RECONSTRUCTION_AUDIT_REQUIRED',
         'old_checkpoint_recovered':False,'generation_authorized':False,'detector_training_authorized':False,
         'generator_training_classes':2,'deep_pcb_training':0,'bootstrap_guard_modification':0,'validation_content_read':0})


if __name__=='__main__':main()
