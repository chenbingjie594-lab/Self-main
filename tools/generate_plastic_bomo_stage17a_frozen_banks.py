"""Generate frozen A/B/C banks and reload probes; never train/evaluate detectors."""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys

from prepare_plastic_bomo_stage17a_trainonly import REPO, FOLDERS, load, save, sha
from probe_plastic_bomo_stage17a_trainonly import verify_checkpoints, paired_rgb_pass
from probe_plastic_bomo_stage17a_frozen_slots import fixed_probe_slots, verify_slot_inputs
from probe_plastic_bomo_stage17a_rebuilt import historical_runtime


def verify_preprobe(probe, slots, slot_hash):
    status = load(probe / 'pregeneration_probe_status.json')
    fixed = load(probe / 'fixed_pregeneration_probe_slots.json')
    records = load(probe / 'pregeneration_probe_records.json')['records']
    if (status['status'] != 'FROZEN_SLOT_PREGEN_PROBE_PASS'
            or not status['reload_rgb_exact'] or not status['new_isolated_normal_pool_used']
            or status['slot_manifest_sha256'] != slot_hash
            or fixed['slot_manifest_sha256'] != slot_hash
            or fixed['slot_ids'] != [r['slot_id'] for r in fixed_probe_slots(slots)]
            or not paired_rgb_pass(records)):
        raise RuntimeError('EXACT_FROZEN_PREGEN_PROBE_REQUIRED')
    by = {r['slot_id']: r for r in slots}
    from PIL import Image
    for r in records:
        if r['slot_id'] not in fixed['slot_ids'] or r['class'] != by[r['slot_id']]['class']:
            raise RuntimeError('PREGEN_SLOT_MISMATCH')
        if r['sample_seed'] != by[r['slot_id']]['bank_sample_seeds'][r['bank']]:
            raise RuntimeError('PREGEN_SEED_MISMATCH')
        # Relocated downloads are allowed only if bytes and RGB match exactly.
        path = probe / Path(r['image_path'].replace('\\', '/')).name
        if sha(path) != r['image_file_sha256']:
            raise RuntimeError('PREGEN_IMAGE_FILE_CHANGED')
        with Image.open(path) as im:
            if im.mode != 'RGB' or im.size != (512, 512) or hashlib.sha256(im.tobytes()).hexdigest() != r['rgb_sha256']:
                raise RuntimeError('PREGEN_IMAGE_RGB_CHANGED')
    return fixed, records


def retained_slots(slots, failures):
    failed = {r['slot_id'] for r in failures}
    kept = [r for r in slots if r['slot_id'] not in failed]
    counts = Counter(r['class'] for r in kept)
    if any(counts[c] < 35 for c in FOLDERS):
        raise RuntimeError('INSUFFICIENT_MATCHED_SLOTS_NO_REPLACEMENT')
    return kept


def reproducibility(pre, formal, post):
    before = {(r['bank'], r['slot_id']): r['rgb_sha256'] for r in pre}
    generated = {(r['bank'], r['slot_id']): r['rgb_sha256'] for r in formal}
    rows = []
    for r in post:
        key = r['bank'], r['slot_id']
        rows.append({'bank': key[0], 'slot_id': key[1], 'before_rgb_sha256': before.get(key),
                     'formal_rgb_sha256': generated.get(key), 'after_rgb_sha256': r['rgb_sha256'],
                     'exact': before.get(key) == generated.get(key) == r['rgb_sha256']})
    complete = len(before) == 15 and len(rows) == 15 and len({(r['bank'], r['slot_id']) for r in rows}) == 15
    return {'status': 'PRE_FORMAL_POST_RGB_EXACT_PASS' if complete and all(r['exact'] for r in rows) else 'PRE_FORMAL_POST_RGB_EXACT_FAILED',
            'records': rows, 'complete_15_groups': complete, 'comparison': 'decoded RGB, not perceptual tolerance'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('slots', 'coverage', 'preprobe', 'normal_pool', 'training', 'checkpoint_root', 'scheduler', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--device', default='0')
    a = p.parse_args()
    a.output = a.output.resolve(); a.checkpoint_root = a.checkpoint_root.resolve()
    if a.checkpoint_root.name != 'baseline_stage17a_trainonly_s42_noise_0.0':
        raise RuntimeError('NEW_TRAINONLY_MODEL_REQUIRED')
    slot_path = a.slots / 'generation_slot_manifest.json'
    slot_hash = sha(slot_path); manifest = load(slot_path); slots = manifest['slots']
    freeze = load(a.slots / 'slot_freeze_status.json')
    coverage_path = a.coverage / 'slot_mask_label_coverage_audit.json'; coverage = load(coverage_path)
    if (freeze['scope'] != 'SERVER_FREEZE' or freeze['slot_manifest_sha256'] != slot_hash
            or len(slots) != 80 or len({r['slot_id'] for r in slots}) != 80
            or Counter(r['class'] for r in slots) != {'flash': 40, 'black': 40}
            or any(not r['structural_valid'] for r in slots)):
        raise RuntimeError('EXACT_80_SERVER_FROZEN_SLOTS_REQUIRED')
    if (coverage['status'] != 'SLOT_MASK_LABEL_COVERAGE_PASS' or coverage['scope'] != 'SERVER_COVERAGE'
            or coverage['slot_manifest_sha256'] != slot_hash or coverage['failed_slot_ids']):
        raise RuntimeError('SERVER_MASK_LABEL_COVERAGE_REQUIRED')
    normal_path = a.normal_pool / 'isolated_normal_pool_manifest.json'
    normal_status = load(a.normal_pool / 'isolated_normal_pool_status.json')
    normals = load(normal_path)
    if (sha(normal_path) != manifest['normal_pool_manifest_sha256']
            or normal_status['status'] != 'ISOLATED_NORMAL_POOL_PREPARED'
            or normal_status['scope'] != 'SERVER_AUDIT' or len(normals['records']) != 174):
        raise RuntimeError('EXACT_ISOLATED_NORMAL_POOL_REQUIRED')
    normal_by = {r['normal_filename']: r for r in normals['records']}
    for r in slots:
        if normal_by[r['conditioning_id']]['normal_sha256'] != r['normal_sha256']:
            raise RuntimeError('NORMAL_POOL_SLOT_BINDING_FAILED')
    verify_slot_inputs(slots)
    fixed, pre = verify_preprobe(a.preprobe, slots, slot_hash)
    baseline = REPO / 'configs/baseline_frozen_split70_s42.json'
    if sha(baseline) != manifest['baseline_config_sha256'] or sha(baseline) != fixed['baseline_config_sha256']:
        raise RuntimeError('BASELINE_CONFIG_CHANGED')
    if sha(a.scheduler / 'scheduler_config.json') != fixed['scheduler_sha256']:
        raise RuntimeError('SCHEDULER_CHANGED')
    for name, expected in fixed['implementation_sha256'].items():
        if sha(REPO / name) != expected: raise RuntimeError('INFERENCE_IMPLEMENTATION_CHANGED:' + name)
    assets = verify_checkpoints(a.training, a.checkpoint_root)
    if assets != manifest['checkpoint_assets']: raise RuntimeError('FROZEN_CHECKPOINT_CHANGED')
    a.output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'FROZEN_BANK_GENERATION_IN_PROGRESS', 'formal_sampling_count': 0,
              'post_probe_sampling_count': 0, 'formal_generation_authorized': True,
              'authorization_basis': 'frozen slots + server coverage + exact pre-generation probes',
              'detector_training_authorized': False, 'validation_labels_read': 0,
              'official_validation_detector_forwards': 0, 'stage17b_auto_start': False,
              'slot_manifest_sha256': slot_hash, 'quality_filtering': False, 'replacement_count': 0}
    save(a.output / 'formal_generation_status.json', status)
    records, post, failures = [], [], []
    input_digests = {str(q): sha(q) for q in (slot_path, coverage_path, normal_path,
                     a.preprobe / 'pregeneration_probe_status.json', a.preprobe / 'fixed_pregeneration_probe_slots.json',
                     a.preprobe / 'pregeneration_probe_records.json', baseline, a.scheduler / 'scheduler_config.json')}
    save(a.output / 'generation_protocol_audit.json', {'checkpoint_identity': a.checkpoint_root.name,
         'checkpoint_assets': assets, 'input_files_sha256': input_digests,
         'implementation_sha256': fixed['implementation_sha256'], 'generator_script_sha256': sha(__file__),
         'bank_seeds': {'A':42,'B':3407,'C':2026}, 'expected_slot_count_per_bank': 80,
         'baseline_parameters': load(baseline), 'normal_pool_count':174,
         'failure_policy':'numeric/output invalidity drops identical slot in all banks, no replacement; fewer than35/class stops; other runtime failures abort',
         'labels_policy':'copy frozen projected labels without alteration', 'validation_used': False})
    try:
        import numpy as np
        import torch
        from PIL import Image
        from diffusers.configuration_utils import FrozenDict
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        versions = {}
        for package in ('torch','diffusers','transformers','numpy','Pillow'):
            versions[package] = importlib.metadata.version(package)
        save(a.output / 'generation_environment.json', {'python':sys.version,'platform':platform.platform(),
             'packages':versions,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(int(a.device)),
             'deterministic_algorithms':True,'cudnn_benchmark':False})
        runtime = historical_runtime(a.checkpoint_root, baseline)

        def render(pipe, r, bank, destination):
            verify_slot_inputs([r]); seed = r['bank_sample_seeds'][bank]
            random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            normal = runtime.image_resize_center_crop(Image.open(r['normal_path']).convert('RGB'))
            mask = runtime.mask_resize_center_crop(Image.open(r['mask_path']).convert('L'))
            binary = (np.asarray(mask) > 127).astype('uint8') * 255
            if hashlib.sha256(binary.tobytes()).hexdigest() != r['preprocessed_mask_sha256']:
                raise RuntimeError('RUNTIME_MASK_TRANSFORM_MISMATCH')
            image = runtime.inpaint(pipe, normal, runtime.args.prompt, mask=Image.fromarray(binary), n_samples=1,
                device='cuda:'+a.device, blur_factor=runtime.args.blur_factor,
                guidance_scale_inside=runtime.args.guidance_scale_inside, guidance_scale_outside=runtime.args.guidance_scale_outside,
                num_inference_steps=runtime.args.num_inference_steps, guidance_scale=runtime.args.guidance_scale)[0].convert('RGB')
            if image.size != (512,512) or not np.isfinite(np.asarray(image)).all():
                raise RuntimeError('INVALID_GENERATED_OUTPUT')
            image.save(destination)
            return {'slot_id':r['slot_id'],'class':r['class'],'bank':bank,'sample_seed':seed,
                    'image_path':str(destination),'image_file_sha256':sha(destination),
                    'rgb_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
                    'source_id':r['source_id'],'real_parent':r['real_parent'],
                    'conditioning_id':r['conditioning_id'],'synthetic_labels':r['synthetic_labels']}

        for phase in ('formal', 'post_probe'):
            for cls, folder in FOLDERS.items():
                pipe = runtime.StableDiffusionInpaintPipeline_dynamic.from_pretrained(
                    str(a.checkpoint_root/'Plastic_Bomo'/folder), torch_dtype=torch.float16, local_files_only=True)
                pipe.scheduler = runtime.DDIMScheduler.from_pretrained(str(a.scheduler))
                config = dict(pipe.scheduler.config)
                keys = ('eta_mask_use_schedule','eta_mask_schedule','eta_mask_min','eta_mask_max','eta_mask_power',
                        'eta_mask_exp_k','eta_mask_sigmoid_k','eta_mask_guard','eta_mask_guard_margin','eta_mask_segmented','eta_mask_stop_step')
                config.update({k:getattr(runtime.args,k) for k in keys}); pipe.scheduler._internal_dict = FrozenDict(config)
                pipe.to('cuda:'+a.device); pipe.set_progress_bar_config(disable=True)
                def finite(_module, _inputs, output):
                    x = output.sample if hasattr(output,'sample') else output
                    if isinstance(x,tuple): x=x[0]
                    if torch.is_tensor(x) and not bool(torch.isfinite(x).all()):
                        raise RuntimeError('NONFINITE_PIPELINE_TENSOR')
                hooks = [pipe.unet.register_forward_hook(finite), pipe.vae.decoder.register_forward_hook(finite)]
                try:
                    selected = slots if phase == 'formal' else fixed_probe_slots(slots)
                    for bank in ('A','B','C'):
                        directory = a.output/phase/bank; directory.mkdir(parents=True,exist_ok=True)
                        for r in (r for r in selected if r['class']==cls):
                            if phase == 'formal' and r['slot_id'] in {f['slot_id'] for f in failures}: continue
                            status['formal_sampling_count' if phase=='formal' else 'post_probe_sampling_count'] += 1
                            try:
                                row = render(pipe,r,bank,directory/(r['slot_id']+'.png'))
                            except RuntimeError as e:
                                if phase != 'formal' or str(e) not in ('NONFINITE_PIPELINE_TENSOR','INVALID_GENERATED_OUTPUT'): raise
                                failures.append({'slot_id':r['slot_id'],'class':cls,'bank':bank,'reason':str(e)})
                                save(a.output/'generation_failures.json',{'records':failures})
                                retained_slots(slots,failures)
                                continue
                            (records if phase=='formal' else post).append(row)
                            save(a.output/('formal_generation_raw_records.json' if phase=='formal' else 'post_generation_probe_records.json'),{'records':records if phase=='formal' else post})
                            save(a.output/'formal_generation_status.json',status)
                            print(f'{phase} {cls} {bank} {len(records) if phase=="formal" else len(post)}',flush=True)
                finally:
                    for h in hooks: h.remove()
                    del pipe; torch.cuda.empty_cache()
        kept = retained_slots(slots,failures); kept_ids = {r['slot_id'] for r in kept}
        valid = [r for r in records if r['slot_id'] in kept_ids]
        pairs = Counter((r['bank'],r['slot_id']) for r in valid)
        if len(valid)!=3*len(kept) or any(pairs[b,s]!=1 for b in ('A','B','C') for s in kept_ids):
            raise RuntimeError('PAIRED_BANK_COMPLETENESS_FAILED')
        audit = reproducibility(pre,valid,post)
        save(a.output/'generation_reproducibility_audit.json',audit)
        if audit['status']!='PRE_FORMAL_POST_RGB_EXACT_PASS': raise RuntimeError('PRE_FORMAL_POST_RGB_MISMATCH')
        verify_slot_inputs(slots)
        if any(sha(q)!=d for q,d in input_digests.items()) or verify_checkpoints(a.training,a.checkpoint_root)!=assets:
            raise RuntimeError('FROZEN_INPUT_OR_MODEL_CHANGED_DURING_GENERATION')
        for name,digest in fixed['implementation_sha256'].items():
            if sha(REPO/name)!=digest: raise RuntimeError('IMPLEMENTATION_CHANGED_DURING_GENERATION')
        for row in records+post:
            if sha(row['image_path'])!=row['image_file_sha256']: raise RuntimeError('GENERATED_FILE_CHANGED')
        for bank in ('A','B','C'):
            save(a.output/f'bank_{bank}_manifest.json',{'bank':bank,'slot_manifest_sha256':slot_hash,
                 'records':[r for r in valid if r['bank']==bank], 'class_counts':dict(Counter(r['class'] for r in kept))})
        save(a.output/'source_seed_manifest.json',{'status':'PAIRED_FROZEN_BANKS_COMPLETE','slot_manifest_sha256':slot_hash,
             'checkpoint_identity':a.checkpoint_root.name,'records':valid,'retained_slot_ids':[r['slot_id'] for r in kept],
             'dropped_slot_ids':sorted(set(r['slot_id'] for r in failures)),'no_replacement':True})
        save(a.output/'bank_pairing_audit.json',{'status':'PASS','same_slots':True,'same_normal_mask_prompt_labels':True,
             'only_bank_rng_changes':True,'class_counts_per_bank':dict(Counter(r['class'] for r in kept)),
             'retained_slots':len(kept),'failures':failures,'quality_filtering':False,
             'excluded_files_preserved_in_raw_records':True,'annotation_semantic_correctness_proven':False})
        status.update(status='FROZEN_BANK_GENERATION_AND_REPRO_PASS',retained_slots_per_bank=len(kept),
                      retained_images=len(valid),post_generation_reproducibility='PASS',
                      detector_training_authorized=False,next_gate='Review downloaded generation audits before preparing matched detector workflow')
        save(a.output/'formal_generation_status.json',status); print(json.dumps(status,indent=2))
    except Exception as e:
        status.update(status='FROZEN_BANK_GENERATION_FAILED',error=f'{type(e).__name__}: {e}',formal_generation_authorized=False)
        save(a.output/'formal_generation_status.json',status)
        raise


if __name__ == '__main__': main()
