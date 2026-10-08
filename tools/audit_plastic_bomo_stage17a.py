"""Stage17A CPU reconstruction gate. Never generate, train, or access validation."""
import argparse
import hashlib
import json
from pathlib import Path


def load(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def sample_seed(bank, slot):
    value = json.dumps([bank, slot], separators=(',', ':'), ensure_ascii=False).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:8], 'big') % (2**63-1)


def baseline_errors(b):
    errors = []
    expected = {'pipeline_mode':'custom', 'prompt':'a photo of a sks defect',
                'num_inference_steps':50, 'guidance_scale':7.5, 'blur_factor':0,
                'negative_prompt':None, 'dtype':'float16'}
    for k,v in expected.items():
        if b.get(k)!=v: errors.append('FROZEN_BASELINE_FIELD_MISMATCH:'+k)
    for k in ('prompt_perturbation','spatial_guidance','cama','ddim_noise','mdap','rda'):
        if b.get('modules',{}).get(k,{}).get('enabled') is not False:
            errors.append('BASELINE_MODULE_NOT_DISABLED:'+k)
    if b.get('normal_filter')!={'enabled':True,'black_threshold':20,'min_mean_luminance':30.0,'min_nonblack_ratio':0.60}:
        errors.append('BASELINE_NORMAL_FILTER_MISMATCH')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=Path('configs/plastic_bomo_stage17a.json'))
    p.add_argument('--repo_root', type=Path, default=Path(__file__).resolve().parents[1])
    for key in ('checkpoint_root','scheduler','generation_train_root','split_manifest','real_train_root','detector_model'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--scope', choices=('LOCAL_PREFLIGHT','SERVER_PREFLIGHT'), default='LOCAL_PREFLIGHT')
    a = p.parse_args(); repo = a.repo_root.resolve(); cfg=load(a.protocol)
    # Never overwrite downloaded server audits or previous attempts.
    a.output.mkdir(parents=True, exist_ok=False)
    historical = {}
    for folder in ('plastic_bomo_stage14a_generator_bottleneck','plastic_bomo_stage14b_fidelity_confirmation',
                   'plastic_bomo_stage15a_class_conditional_morphology','plastic_bomo_stage15b_morphology_dose',
                   'plastic_bomo_stage16a_task_guidance','plastic_bomo_stage16a_r0_traceability_preflight'):
        for path in (repo/'results_for_gpt'/folder).glob('*status*.json'):
            historical[str(path)] = sha(path)
    baseline = load(repo/cfg['baseline_config']); experiment=load(repo/cfg['baseline_experiment'])
    errors=baseline_errors(baseline)
    previous=load(repo/'configs/plastic_bomo_stage15b.json')['detector']
    detector_matches=all(cfg['detector'].get(k)==v for k,v in previous.items() if k!='primary_checkpoint')
    detector_matches &= previous['primary_checkpoint']=='last.pt' and cfg['detector']['primary_checkpoint']=='epoch150 last.pt'
    if not detector_matches: errors.append('FROZEN_DETECTOR_PROTOCOL_MISMATCH')
    if a.checkpoint_root.name!=Path(cfg['checkpoint_root']).name:
        errors.append('EXACT_CHECKPOINT_DIRECTORY_REQUIRED_NO_ALIAS_SUBSTITUTION')
    checkpoints={}
    for cls,folder in cfg['classes'].items():
        root=a.checkpoint_root/'Plastic_Bomo'/folder
        missing=[]
        for relative in ('model_index.json','unet/config.json','vae/config.json','text_encoder/config.json','tokenizer/tokenizer_config.json','tokenizer/vocab.json','tokenizer/merges.txt'):
            if not (root/relative).is_file(): missing.append(relative)
        for component in ('unet','vae','text_encoder'):
            files=list((root/component).glob('*.safetensors'))+list((root/component).glob('*.bin'))
            if not files: missing.append(component+'/WEIGHTS')
        files={str(f.relative_to(root)):sha(f) for f in sorted(root.rglob('*')) if f.is_file()} if root.is_dir() else {}
        checkpoints[cls]={'path':str(root.resolve()),'missing_components':missing,'files_sha256':files,
                          'checkpoint_tree_sha256':hashlib.sha256(json.dumps(files,sort_keys=True).encode()).hexdigest() if files else None}
        if missing: errors.append('EXACT_BASELINE_CHECKPOINT_UNAVAILABLE:'+cls)
    scheduler=a.scheduler/'scheduler_config.json'
    if not scheduler.is_file(): errors.append('FROZEN_SCHEDULER_CONFIG_UNAVAILABLE')
    source_status={'train_root':str(a.generation_train_root.resolve()),'manifest':str(a.split_manifest.resolve()),'metadata_available':False}
    if not a.split_manifest.is_file():
        errors.append('FROZEN_GENERATION_SPLIT_MANIFEST_UNAVAILABLE')
    else:
        split=load(a.split_manifest)
        source_status['manifest_sha256']=sha(a.split_manifest)
        if split.get('seed')!=42 or split.get('train_ratio')!=0.7: errors.append('FROZEN_GENERATION_SPLIT_IDENTITY_MISMATCH')
        counts={}
        for cls,folder in cfg['classes'].items():
            rows=split.get('categories',{}).get('Plastic_Bomo',{}).get(folder,{}).get('train',[])
            counts[cls]=len(rows)
            for r in rows:
                for field,sub in (('image','test'),('mask','ground_truth')):
                    name=r.get(field,'')
                    if not name or '/' in name or '\\' in name or name in ('.','..'):
                        errors.append('INVALID_TRAIN_SOURCE_FILENAME');continue
                    asset=a.generation_train_root/'Plastic_Bomo'/sub/folder/name
                    if not asset.is_file(): errors.append('FROZEN_GENERATION_TRAIN_ASSET_UNAVAILABLE:'+str(asset))
            if not rows: errors.append('EMPTY_FROZEN_GENERATION_SOURCE_POOL:'+cls)
        source_status.update({'metadata_available':True,'train_source_counts':counts})
    if not (a.generation_train_root/'Plastic_Bomo/train/good').is_dir():
        errors.append('FROZEN_NORMAL_POOL_UNAVAILABLE')
    real={}
    for image in sorted((a.real_train_root/'images/train').glob('*')):
        if image.suffix.lower() not in ('.jpg','.jpeg','.png','.bmp'): continue
        label=a.real_train_root/'labels/train'/(image.stem+'.txt')
        if not label.is_file(): errors.append('REAL_TRAIN_LABEL_UNAVAILABLE:'+image.name);continue
        real[image.name]={'image_sha256':sha(image),'label_sha256':sha(label)}
    frozen=load(repo/'results_for_gpt/plastic_bomo_stage16a_r0_traceability_preflight/annotation_provenance_audit.json')['records']
    expected={r['image_id']:{'image_sha256':r['image_sha256'],'label_sha256':r['annotation_sha256']} for r in frozen}
    real_equal=real==expected and len(real)==138
    if not real_equal: errors.append('FROZEN_138_REAL_TRAIN_HASH_MISMATCH')
    if not a.detector_model.is_file(): errors.append('PRETRAINED_YOLO11S_UNAVAILABLE')
    code={name:sha(repo/name) for name in ('inference.py','magic_ddim.py','experiment_config.py','diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint_magic.py') if (repo/name).is_file()}
    if len(code)!=4: errors.append('BASELINE_IMPLEMENTATION_UNAVAILABLE')
    audit={'phase':'CPU_ASSET_RECONSTRUCTION_ONLY','baseline_config':baseline,'baseline_experiment':experiment,
           'baseline_config_sha256':sha(repo/cfg['baseline_config']),'implementation_sha256':code,
           'checkpoints':checkpoints,'scheduler_path':str(scheduler.resolve()),'scheduler_sha256':sha(scheduler) if scheduler.is_file() else None,
           'conditioning_logic':'inference.py chooses random normal after unchanged filter; Stage17A must freeze the same normal per slot before bank RNG',
           'mask_logic':'frozen training source mask; nearest Resize(shorter side512)+CenterCrop512; threshold127; CAMA/random mask disabled',
           'image_preprocessing':'bilinear Resize(shorter side512)+CenterCrop512','output_resolution':[512,512],
           'class_mapping':{'01_Flash_point':0,'02_Big_black_spots':1},'generation_entrypoint':'historical inference.py; bank-aware wrapper NOT YET IMPLEMENTED',
           'sources':source_status,'exact_baseline_reconstructed':False,'errors':errors,
           'runtime_call_equivalence_verified':False,'generation_reproducibility_verified':False}
    status={'status':'VANILLA_BASELINE_PROTOCOL_NOT_RECONSTRUCTABLE' if errors else 'BASELINE_ASSETS_AVAILABLE_RUNTIME_RECONSTRUCTION_REQUIRED',
            'scope':a.scope,'errors':errors,'GENERATION_AUTHORIZED':False,'DETECTOR_TRAINING_AUTHORIZED':False,
            'GENERATOR_MECHANISM_DEVELOPMENT_AUTHORIZED':False,'sampling_count':0,'detector_training_count':0,
            'official_validation_use_count':0,'deep_pcb_training':0,'deep_pcb_generation':0,
            'bootstrap_guard_modification':0,'stage17b_auto_start':False,'full_experiment_complete':False}
    save(a.output/'stage17a_protocol.json',cfg);save(a.output/'vanilla_generator_protocol_audit.json',audit)
    save(a.output/'training_environment_audit.json',{'status':'PREFLIGHT_ONLY_NOT_TRAINED','frozen_detector_protocol_matches':detector_matches,
         'detector':cfg['detector'],'pretrained_sha256':sha(a.detector_model) if a.detector_model.is_file() else None,
         'same_138_real_train_verified':real_equal,'successful_optimizer_step_equality':'NOT_MEASURED'})
    save(a.output/'realrepeat_equivalence_audit.json',{'status':'HISTORICAL_REUSE_NOT_AUTHORIZED','reason':'No evidence yet of exact slot/class-role exposure, ordering, initialization byte identity and validation timing equivalence. No historical metric silently reused.', 'retrained':False})
    save(a.output/'validation_usage_audit.json',{'official_validation_use_count':0,'validation_content_read':0,'selection_uses':0,'planned_final_evaluations':4,'must_wait_for_all_arms_complete':True})
    save(a.output/'historical_status_immutability_audit.json',{'before':historical,'after':{p:sha(p) for p in historical},'equal':all(sha(p)==v for p,v in historical.items())})
    save(a.output/'stage17a_status.json',status)
    (a.output/'STAGE17A_REPORT.md').write_text('# Stage17A — Vanilla Utility Stability\n\nStatus: '+status['status']+'\n\nThis is an asset preflight, not a completed experiment. No generation or training was run. Other checkpoint aliases are not substitutes for the exact frozen baseline.\n\n'+ '\n'.join('- '+e for e in errors)+'\n\nGenerator runtime call equivalence, source lineage into the legal frozen train pool, 80 slots, conditioning/bbox freezing, generation reproducibility and paired-bank audits remain required. No detector training is authorized. A structurally successful server preflight is not a generation pass.\n\nFinal intended arms: RR/VA/VB/VC; detector seed42 only; 150 epochs, batch1, imgsz1536. Validation only after all arms finish, last.pt only. Sample std uses ddof1; generation-bank variability at fixed detector seed is not general detector-seed uncertainty.\n\nLabels remain frozen. XML correspondence does not establish annotation accuracy. Supplemental Black prelabels lack instance-level human review evidence, and suspected omissions are unchanged. Claims are limited to utility under the current frozen annotation protocol, not physical-defect ground truth. Stage16 stop records are preserved, not interpreted as task-guidance failure.\n\nNo final metrics/status or GitHub completion is claimed before the required experiment is actually finished.\n',encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))
    raise SystemExit(2 if errors else 0)


if __name__=='__main__': main()
