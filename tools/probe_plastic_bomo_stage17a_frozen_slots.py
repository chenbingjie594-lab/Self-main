"""Pre-generation RGB repeatability test on five of the immutable 80 slots."""
import argparse
import hashlib
import json
from pathlib import Path
import random

from prepare_plastic_bomo_stage17a_trainonly import REPO, FOLDERS, load, save, sha
from probe_plastic_bomo_stage17a_trainonly import verify_checkpoints, paired_rgb_pass
from probe_plastic_bomo_stage17a_rebuilt import historical_runtime
from audit_plastic_bomo_stage17a import sample_seed


def fixed_probe_slots(slots):
    result = []
    for cls, count in (('flash', 3), ('black', 2)):
        rows = sorted((r for r in slots if r['class'] == cls), key=lambda r: hashlib.sha256(
            json.dumps(['stage17a_pre_generation_probe', r['slot_id']], separators=(',', ':')).encode()).hexdigest())
        if len(rows) < count: raise RuntimeError('NOT_ENOUGH_FROZEN_SLOTS')
        result.extend(rows[:count])
    return result


def verify_slot_inputs(slots):
    for r in slots:
        for field in ('image', 'mask', 'normal'):
            if sha(r[field + '_path']) != r[field + '_sha256']:
                raise RuntimeError('FROZEN_SLOT_INPUT_CHANGED:' + r['slot_id'] + ':' + field)
        if r['bank_sample_seeds'] != {b:sample_seed(seed, r['slot_id']) for b,seed in {'A':42,'B':3407,'C':2026}.items()}:
            raise RuntimeError('FROZEN_SLOT_SEED_CHANGED')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('slots', 'coverage', 'training', 'checkpoint_root', 'reference_assets', 'scheduler', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--device', default='0')
    a = p.parse_args()
    a.output = a.output.resolve(); a.checkpoint_root = a.checkpoint_root.resolve()
    if a.checkpoint_root.name != 'baseline_stage17a_trainonly_s42_noise_0.0': raise RuntimeError('NEW_TRAINONLY_MODEL_REQUIRED')
    slot_path = a.slots / 'generation_slot_manifest.json'
    slot_hash = sha(slot_path); manifest = load(slot_path); freeze = load(a.slots / 'slot_freeze_status.json')
    if freeze['scope'] != 'SERVER_FREEZE' or slot_hash != freeze['slot_manifest_sha256']:
        raise RuntimeError('SERVER_FROZEN_SLOTS_REQUIRED')
    coverage = load(a.coverage / 'slot_mask_label_coverage_audit.json')
    if coverage['status'] != 'SLOT_MASK_LABEL_COVERAGE_PASS' or coverage['scope'] != 'SERVER_COVERAGE' or coverage['slot_manifest_sha256'] != slot_hash:
        raise RuntimeError('SERVER_COVERAGE_PASS_REQUIRED')
    slots = manifest['slots']
    if len(slots) != 80 or len({r['slot_id'] for r in slots}) != 80 or any(not r['structural_valid'] for r in slots):
        raise RuntimeError('FROZEN_SLOT_STRUCTURE_INVALID_NO_REPLACEMENT')
    verify_slot_inputs(slots)
    baseline = REPO / 'configs/baseline_frozen_split70_s42.json'
    reference = load(a.reference_assets / 'vanilla_generator_protocol_audit.json')
    if sha(baseline) != manifest['baseline_config_sha256'] or sha(baseline) != reference['baseline_config_sha256'] or sha(a.scheduler / 'scheduler_config.json') != reference['scheduler_sha256']:
        raise RuntimeError('FROZEN_GENERATION_CONFIG_OR_SCHEDULER_CHANGED')
    for name, expected in reference['implementation_sha256'].items():
        if sha(REPO / name) != expected: raise RuntimeError('INFERENCE_IMPLEMENTATION_CHANGED:' + name)
    a.output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'FROZEN_SLOT_PREGEN_PROBE_IN_PROGRESS', 'phase': 'BEFORE_FORMAL_GENERATION',
              'slot_manifest_sha256': slot_hash, 'probe_generation_count': 0, 'formal_slots_generated': 0,
              'formal_generation_authorized': False, 'detector_training_authorized': False,
              'official_validation_detector_forwards': 0, 'validation_labels_read': 0,
              'post_generation_reproducibility': 'NOT_RUN', 'stage17b_auto_start': False}
    save(a.output / 'pregeneration_probe_status.json', status)
    try:
        assets = verify_checkpoints(a.training, a.checkpoint_root)
        if assets != manifest['checkpoint_assets']: raise RuntimeError('FROZEN_CHECKPOINT_CHANGED')
        probes = fixed_probe_slots(slots)
        save(a.output / 'fixed_pregeneration_probe_slots.json', {'slot_ids': [r['slot_id'] for r in probes],
             'selection': 'predefined metadata hash, 3 flash / 2 black; no result or quality information',
             'slot_manifest_sha256': slot_hash, 'baseline_config_sha256': sha(baseline),
             'scheduler_sha256': reference['scheduler_sha256'], 'implementation_sha256': reference['implementation_sha256']})
        import numpy as np
        import torch
        from PIL import Image
        from diffusers.configuration_utils import FrozenDict
        runtime = historical_runtime(a.checkpoint_root, baseline)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        records = []
        for repeat in (0, 1):
            for cls, folder in FOLDERS.items():
                pipe = runtime.StableDiffusionInpaintPipeline_dynamic.from_pretrained(
                    str(a.checkpoint_root / 'Plastic_Bomo' / folder), torch_dtype=torch.float16, local_files_only=True)
                pipe.scheduler = runtime.DDIMScheduler.from_pretrained(str(a.scheduler))
                config = dict(pipe.scheduler.config)
                keys = ('eta_mask_use_schedule','eta_mask_schedule','eta_mask_min','eta_mask_max','eta_mask_power',
                        'eta_mask_exp_k','eta_mask_sigmoid_k','eta_mask_guard','eta_mask_guard_margin','eta_mask_segmented','eta_mask_stop_step')
                config.update({k:getattr(runtime.args,k) for k in keys}); pipe.scheduler._internal_dict = FrozenDict(config)
                pipe.to('cuda:' + a.device); pipe.set_progress_bar_config(disable=True)
                def finite(_module, _inputs, output):
                    x = output.sample if hasattr(output, 'sample') else output
                    if isinstance(x, tuple): x = x[0]
                    if torch.is_tensor(x) and not bool(torch.isfinite(x).all()): raise RuntimeError('NONFINITE_PIPELINE_TENSOR')
                hooks = [pipe.unet.register_forward_hook(finite), pipe.vae.decoder.register_forward_hook(finite)]
                try:
                    for bank in ('A','B','C'):
                        for r in (r for r in probes if r['class'] == cls):
                            verify_slot_inputs([r]); seed = r['bank_sample_seeds'][bank]
                            random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                            normal = runtime.image_resize_center_crop(Image.open(r['normal_path']).convert('RGB'))
                            mask = runtime.mask_resize_center_crop(Image.open(r['mask_path']).convert('L'))
                            mask_array = (np.asarray(mask) > 127).astype('uint8') * 255
                            if hashlib.sha256(mask_array.tobytes()).hexdigest() != r['preprocessed_mask_sha256']:
                                raise RuntimeError('RUNTIME_MASK_TRANSFORM_MISMATCH')
                            image = runtime.inpaint(pipe, normal, runtime.args.prompt, mask=Image.fromarray(mask_array), n_samples=1,
                                device='cuda:' + a.device, blur_factor=runtime.args.blur_factor,
                                guidance_scale_inside=runtime.args.guidance_scale_inside, guidance_scale_outside=runtime.args.guidance_scale_outside,
                                num_inference_steps=runtime.args.num_inference_steps, guidance_scale=runtime.args.guidance_scale)[0].convert('RGB')
                            array = np.asarray(image)
                            if image.size != (512,512) or not np.isfinite(array).all(): raise RuntimeError('INVALID_PROBE_OUTPUT')
                            dest = a.output / f"{bank}_{r['slot_id']}_repeat{repeat}.png"; image.save(dest)
                            records.append({'class':cls,'bank':bank,'probe_id':r['slot_id'],'slot_id':r['slot_id'],
                                'repeat':repeat,'sample_seed':seed,'rgb_sha256':hashlib.sha256(array.tobytes()).hexdigest(),
                                'image_path':str(dest),'image_file_sha256':sha(dest)})
                            status['probe_generation_count'] = len(records)
                            save(a.output / 'pregeneration_probe_records.json', {'records':records})
                            save(a.output / 'pregeneration_probe_status.json', status)
                            print(f'probe {len(records)}/30 {cls} {bank} repeat={repeat}', flush=True)
                finally:
                    for h in hooks: h.remove()
                    del pipe; torch.cuda.empty_cache()
        verify_slot_inputs(slots)
        if sha(slot_path) != slot_hash or verify_checkpoints(a.training, a.checkpoint_root) != assets:
            raise RuntimeError('FROZEN_PROTOCOL_CHANGED_DURING_PROBE')
        if not paired_rgb_pass(records): raise RuntimeError('VANILLA_GENERATION_NOT_DETERMINISTIC')
        status.update(status='FROZEN_SLOT_PREGEN_PROBE_PASS', reload_rgb_exact=True,
                      frozen_slot_coverage_pass=True, new_isolated_normal_pool_used=True)
        save(a.output / 'pregeneration_probe_status.json', status); print(json.dumps(status,indent=2))
    except Exception as e:
        status.update(status='FROZEN_SLOT_PREGEN_PROBE_FAILED', error=f'{type(e).__name__}: {e}')
        save(a.output / 'pregeneration_probe_status.json', status)
        raise


if __name__ == '__main__': main()
