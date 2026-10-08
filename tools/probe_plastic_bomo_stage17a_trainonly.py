"""Actual asset and reload-RGB audit for the new train-only checkpoint, not formal banks."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import struct
import sys

from prepare_plastic_bomo_stage17a_trainonly import REPO, FOLDERS, load, save, sha, rgb_hash
from train_plastic_bomo_stage17a_trainonly import verify_pool
from probe_plastic_bomo_stage17a_rebuilt import historical_runtime
from audit_plastic_bomo_stage17a import sample_seed


def choose_trainonly_probes(records):
    probes = []
    for cls, count in (('flash', 3), ('black', 2)):
        selected = sorted((r for r in records if r['class'] == cls), key=lambda r: r['source_id'])
        if len(selected) < count:
            raise RuntimeError('INSUFFICIENT_PROBE_SOURCES_NO_REPLACEMENT:' + cls)
        probes.extend(dict(r, probe_id=r['source_id']) for r in selected[:count])
    if len({r['probe_id'] for r in probes}) != 5:
        raise RuntimeError('DUPLICATE_PROBE_ID')
    return probes


def safetensor_layout(path):
    """Check header shapes and payload extent without unpickling/loading tensor data."""
    sizes = {'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2, 'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1, 'U8': 1, 'BOOL': 1}
    with Path(path).open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise RuntimeError('TRUNCATED_SAFETENSOR')
        length = struct.unpack('<Q', raw)[0]
        if length < 2 or length > 64 * 1024 * 1024:
            raise RuntimeError('INVALID_SAFETENSOR_HEADER_SIZE')
        header = json.loads(stream.read(length))
    intervals = []
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        elements = 1
        for dim in tensor['shape']:
            if not isinstance(dim, int) or dim < 0:
                raise RuntimeError('INVALID_TENSOR_SHAPE')
            elements *= dim
        start, end = tensor['data_offsets']
        if tensor['dtype'] not in sizes or start < 0 or end - start != elements * sizes[tensor['dtype']]:
            raise RuntimeError('INVALID_TENSOR_PAYLOAD')
        intervals.append((start, end))
    offset = 0
    for start, end in sorted(intervals):
        if start != offset:
            raise RuntimeError('NONCONTIGUOUS_SAFETENSOR_PAYLOAD')
        offset = end
    if not intervals or Path(path).stat().st_size != 8 + length + offset:
        raise RuntimeError('SAFETENSOR_PAYLOAD_EXTENT_MISMATCH')
    return {'tensor_count': len(intervals), 'payload_bytes': offset, 'header_bytes': length, 'file_bytes': Path(path).stat().st_size}


def verify_checkpoints(training, checkpoint_root):
    status = load(training / 'trainonly_training_status.json')
    if status['status'] != 'TRAINONLY_BASELINE_TRAINING_COMPLETE_RUNTIME_AUDIT_REQUIRED' or status['identity'] != checkpoint_root.name:
        raise RuntimeError('TRAINONLY_TRAINING_NOT_COMPLETE_OR_WRONG_MODEL')
    launch = load(training / 'trainonly_training_launch_audit.json')
    expected = {'seed': 42, 'max_train_steps': 2000, 'learning_rate': 1e-6, 'train_batch_size': 1,
                'gradient_accumulation_steps': 4, 'train_text_encoder': False, 'text_noise_scale': 0.,
                'mixed_precision': 'fp16', 'gradient_checkpointing': True, 'resolution': 512,
                'center_crop': True, 'with_prior_preservation': False, 'resume_from_checkpoint': None}
    results = {}
    for cls, folder in FOLDERS.items():
        runtime = load(training / f'{cls}_training_runtime.json')
        asset = load(training / f'{cls}_checkpoint_asset_audit.json')
        root = checkpoint_root / 'Plastic_Bomo' / folder
        if runtime != asset['runtime'] or runtime['status'] != 'FINAL_PIPELINE_SAVED' or runtime['global_step'] != 2000:
            raise RuntimeError('TRAINONLY_RUNTIME_RECORD_MISMATCH:' + cls)
        if any(runtime['arguments'][k] != v for k, v in expected.items()):
            raise RuntimeError('TRAINING_ARGUMENT_CHANGED:' + cls)
        if runtime['optimizer_step_attempts'] != runtime['successful_optimizer_steps'] + runtime['amp_skipped_steps']:
            raise RuntimeError('INVALID_UPDATE_LEDGER')
        if runtime['arguments']['pretrained_model_name_or_path'] != launch['base_model']:
            raise RuntimeError('INITIALIZATION_RECORD_MISMATCH')
        files = asset['files_sha256']
        layouts = {}
        for relative, digest in files.items():
            path = (root / relative).resolve()
            if root.resolve() not in path.parents or sha(path) != digest:
                raise RuntimeError('ACTUAL_CHECKPOINT_HASH_MISMATCH:' + str(path))
            if relative.endswith('.safetensors'):
                layouts[relative] = safetensor_layout(path)
        for part in ('unet', 'vae', 'text_encoder'):
            if load(root / part / 'config.json') is None or not any(k.startswith(part + '/') for k in layouts):
                raise RuntimeError('REQUIRED_MODEL_COMPONENT_MISSING:' + part)
        if load(root / 'unet/config.json')['in_channels'] != 9:
            raise RuntimeError('NON_INPAINTING_MODEL')
        results[cls] = {'files_sha256': files, 'safetensor_layout': layouts,
                        'optimizer_step_attempts': runtime['optimizer_step_attempts'],
                        'successful_optimizer_steps': runtime['successful_optimizer_steps'],
                        'amp_skipped_steps': runtime['amp_skipped_steps'], 'actual_file_hashes_verified': True}
    return results


def paired_rgb_pass(records):
    pairs = {}
    for r in records:
        pairs.setdefault((r['bank'], r['probe_id']), []).append(r)
    return (len(records) == 30 and len(pairs) == 15
            and all(len(v) == 2 and {r['repeat'] for r in v} == {0, 1}
                    and len({r['rgb_sha256'] for r in v}) == 1 for v in pairs.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'training', 'checkpoint_root', 'frozen', 'real_data_yaml', 'reference_assets', 'scheduler', 'normal_root', 'output'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--protocol', type=Path, default=REPO / 'configs/plastic_bomo_stage17a_trainonly.json')
    p.add_argument('--device', default='0')
    a = p.parse_args()
    for name in ('prepared', 'training', 'checkpoint_root', 'frozen', 'real_data_yaml', 'reference_assets', 'scheduler', 'normal_root', 'output', 'protocol'):
        setattr(a, name, getattr(a, name).resolve())
    if a.checkpoint_root.name != 'baseline_stage17a_trainonly_s42_noise_0.0':
        raise RuntimeError('ONLY_NEW_TRAINONLY_CHECKPOINT_ALLOWED')
    a.output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'TRAINONLY_RUNTIME_PROBE_IN_PROGRESS', 'formal_generation_authorized': False,
              'detector_training_authorized': False, 'formal_slots_generated': 0, 'probe_generation_count': 0,
              'validation_detector_forwards': 0, 'validation_labels_read': 0, 'normal_lineage_formal_gate': 'NOT_ESTABLISHED',
              'formal_pre_post_bank_reproducibility': 'NOT_RUN', 'stage17b_auto_start': False}
    save(a.output / 'runtime_probe_status.json', status)
    try:
        protocol, dataset, prepared_status = verify_pool(a.prepared, a.protocol, a.frozen, a.real_data_yaml)
        launch = load(a.training / 'trainonly_training_launch_audit.json')
        if launch['source_manifest_sha256'] != prepared_status['source_manifest_sha256'] or sha(a.protocol) != launch['protocol_sha256']:
            raise RuntimeError('TRAINING_POOL_LINK_MISMATCH')
        assets = verify_checkpoints(a.training, a.checkpoint_root)
        reference = load(a.reference_assets / 'vanilla_generator_protocol_audit.json')
        baseline = REPO / 'configs/baseline_frozen_split70_s42.json'
        if sha(baseline) != reference['baseline_config_sha256'] or sha(a.scheduler / 'scheduler_config.json') != reference['scheduler_sha256']:
            raise RuntimeError('BASELINE_INFERENCE_PARAMETERS_CHANGED')
        for name, digest in reference['implementation_sha256'].items():
            if sha(REPO / name) != digest:
                raise RuntimeError('INFERENCE_IMPLEMENTATION_CHANGED:' + name)
        save(a.output / 'actual_checkpoint_asset_audit.json', assets)
        import numpy as np
        import torch
        from PIL import Image
        from diffusers.configuration_utils import FrozenDict
        runtime = historical_runtime(a.checkpoint_root, baseline)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        normal_pool = runtime.get_valid_normal_images(str(a.normal_root), runtime.args.normal_filter)
        probes = choose_trainonly_probes(load(a.prepared / 'trainonly_source_manifest.json')['records'])
        rng = random.Random(42)
        isolation = load(a.prepared / 'source_isolation_audit.json')
        val_bytes = {r['image_sha256'] for r in isolation['validation_images'].values()}
        val_rgb = {r['decoded_rgb_sha256'] for r in isolation['validation_images'].values()}
        for r in probes:
            r['mask_path'] = str(dataset / r['mask_relative']); r['source_path'] = str(dataset / r['image_relative'])
            r['normal_path'] = str(Path(rng.choice(normal_pool)).resolve())
            r['normal_sha256'] = sha(r['normal_path']); _, rgb = rgb_hash(r['normal_path'])
            if r['normal_sha256'] in val_bytes or rgb in val_rgb:
                raise RuntimeError('SMOKE_NORMAL_EXACT_VALIDATION_IDENTITY_NO_REPLACEMENT')
            r['normal_decoded_rgb_sha256'] = rgb
            mask = runtime.mask_resize_center_crop(Image.open(r['mask_path']).convert('L'))
            binary = np.asarray(mask) > 127
            if not binary.any() or binary.all():
                raise RuntimeError('INVALID_PROBE_MASK_NO_REPLACEMENT')
        save(a.output / 'frozen_probe_conditions.json', {'scope': 'SMOKE_ONLY_NOT_80_SLOTS', 'probes': probes,
             'selection': 'source_id SHA metadata order, 3 flash / 2 black; historical normal filter + Random42; no quality scores',
             'normal_pool_files_sha256': {str(Path(n).resolve()): sha(n) for n in normal_pool},
             'normal_formal_lineage_verified': False,
             'normal_identity_screen': 'selected normals checked against frozen validation byte/RGB identities; any match aborts, never replaces'})
        records = []
        for repeat in (0, 1):
            for cls, folder in FOLDERS.items():
                pipe = runtime.StableDiffusionInpaintPipeline_dynamic.from_pretrained(
                    str(a.checkpoint_root / 'Plastic_Bomo' / folder), torch_dtype=torch.float16, local_files_only=True)
                pipe.scheduler = runtime.DDIMScheduler.from_pretrained(str(a.scheduler))
                scheduler_cfg = dict(pipe.scheduler.config)
                keys = ('eta_mask_use_schedule', 'eta_mask_schedule', 'eta_mask_min', 'eta_mask_max', 'eta_mask_power',
                        'eta_mask_exp_k', 'eta_mask_sigmoid_k', 'eta_mask_guard', 'eta_mask_guard_margin', 'eta_mask_segmented', 'eta_mask_stop_step')
                scheduler_cfg.update({k: getattr(runtime.args, k) for k in keys})
                pipe.scheduler._internal_dict = FrozenDict(scheduler_cfg)
                pipe.to('cuda:' + a.device); pipe.set_progress_bar_config(disable=True)
                def check_finite(_module, _inputs, output):
                    value = output.sample if hasattr(output, 'sample') else output
                    if isinstance(value, tuple):
                        value = value[0]
                    if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                        raise RuntimeError('NONFINITE_PIPELINE_TENSOR')
                hooks = [pipe.unet.register_forward_hook(check_finite), pipe.vae.decoder.register_forward_hook(check_finite)]
                try:
                    for bank, bank_seed in {'A': 42, 'B': 3407, 'C': 2026}.items():
                        for r in (r for r in probes if r['class'] == cls):
                            for key, expected in (('source_path', r['image_sha256']), ('mask_path', r['mask_sha256']), ('normal_path', r['normal_sha256'])):
                                if sha(r[key]) != expected:
                                    raise RuntimeError('FROZEN_PROBE_INPUT_CHANGED')
                            seed = sample_seed(bank_seed, r['probe_id'])
                            random.seed(seed); np.random.seed(seed % 2**32); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                            normal = runtime.image_resize_center_crop(Image.open(r['normal_path']).convert('RGB'))
                            mask = runtime.mask_resize_center_crop(Image.open(r['mask_path']).convert('L'))
                            mask = Image.fromarray((np.asarray(mask) > 127).astype('uint8') * 255)
                            image = runtime.inpaint(pipe, normal, runtime.args.prompt, mask=mask, n_samples=1,
                                device='cuda:' + a.device, blur_factor=runtime.args.blur_factor,
                                guidance_scale_inside=runtime.args.guidance_scale_inside, guidance_scale_outside=runtime.args.guidance_scale_outside,
                                num_inference_steps=runtime.args.num_inference_steps, guidance_scale=runtime.args.guidance_scale)[0].convert('RGB')
                            arr = np.asarray(image)
                            if image.size != (512, 512) or not np.isfinite(arr).all():
                                raise RuntimeError('INVALID_GENERATED_PROBE')
                            dst = a.output / f"{bank}_{r['probe_id']}_repeat{repeat}.png"; image.save(dst)
                            records.append({'class': cls, 'bank': bank, 'probe_id': r['probe_id'], 'repeat': repeat,
                                            'sample_seed': seed, 'rgb_sha256': hashlib.sha256(arr.tobytes()).hexdigest(),
                                            'image_path': str(dst), 'image_file_sha256': sha(dst)})
                            status['probe_generation_count'] = len(records)
                            save(a.output / 'runtime_probe_records.json', {'records': records})
                            save(a.output / 'runtime_probe_status.json', status)
                            print(f'probe {len(records)}/30 {cls} {bank} repeat={repeat}', flush=True)
                finally:
                    for h in hooks: h.remove()
                    del pipe; torch.cuda.empty_cache()
        verify_pool(a.prepared, a.protocol, a.frozen, a.real_data_yaml)
        verify_checkpoints(a.training, a.checkpoint_root)
        ok = paired_rgb_pass(records)
        if not ok:
            raise RuntimeError('TRAINONLY_RELOAD_RGB_NONDETERMINISTIC')
        status.update(status='TRAINONLY_RUNTIME_PROBE_PASS', reload_rgb_exact=True,
                      actual_checkpoint_hash_verified=True, source_to_frozen_parent_bound=True)
        save(a.output / 'runtime_probe_status.json', status); print(json.dumps(status, indent=2), flush=True)
    except Exception as e:
        status.update(status='TRAINONLY_RUNTIME_PROBE_FAILED', error=f'{type(e).__name__}: {e}')
        save(a.output / 'runtime_probe_status.json', status)
        raise


if __name__ == '__main__':
    main()
