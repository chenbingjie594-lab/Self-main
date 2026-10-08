"""Load/reload and exact-RGB smoke test; not formal Stage17A bank generation."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from audit_plastic_bomo_stage17a import load, save, sha, sample_seed


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def choose_probes(split, classes):
    """Three flash/two black, hash order only; no pixel/quality selection."""
    result = []
    for cls, count in (('flash', 3), ('black', 2)):
        folder = classes[cls]
        rows = split['categories']['Plastic_Bomo'][folder]['train']
        ordered = sorted(rows, key=lambda r: (hashlib.sha256(
            canonical([cls, r['image'], r['mask']]).encode()).hexdigest(),
            canonical(r)))
        if len(ordered) < count:
            raise RuntimeError('INSUFFICIENT_TRAIN_PROBE_SOURCES:' + cls)
        for row in ordered[:count]:
            result.append({'class': cls, 'folder': folder, **row,
                           'probe_id': hashlib.sha256(canonical(
                               [cls, row['image'], row['mask']]).encode()).hexdigest()})
    return result


def historical_runtime(model_root, config):
    # inference.py parses CLI at import. Supply only the frozen baseline CLI;
    # never execute its main(), which would generate its original entire pool.
    previous = sys.argv
    sys.argv = ['inference.py', '--model_ckpt_root', str(model_root),
                '--config', str(config)]
    try:
        return importlib.import_module('inference')
    finally:
        sys.argv = previous


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'asset_preflight', 'rebuild_audit', 'checkpoint_root',
                 'scheduler', 'generation_train_root', 'split_manifest', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--device', default='0')
    a = p.parse_args()
    cfg = load(a.protocol)
    preflight = load(a.asset_preflight / 'stage17a_status.json')
    assets = load(a.asset_preflight / 'vanilla_generator_protocol_audit.json')
    if preflight['errors'] or assets['errors']:
        raise RuntimeError('ASSET_PREFLIGHT_FAILED')
    if a.checkpoint_root.name != Path(cfg['checkpoint_root']).name:
        raise RuntimeError('WRONG_REBUILT_MODEL_ROOT')
    if load(a.rebuild_audit / 'baseline_rebuild_status.json')['status'] != 'NEW_BASELINE_TRAINING_COMPLETE_RECONSTRUCTION_AUDIT_REQUIRED':
        raise RuntimeError('REBUILD_NOT_COMPLETE')
    for cls, folder in cfg['classes'].items():
        runtime = load(a.rebuild_audit / (cls + '_training_runtime.json'))
        args = runtime['arguments']
        expected = {'seed': 42, 'max_train_steps': 2000, 'learning_rate': 1e-6,
                    'train_batch_size': 1, 'gradient_accumulation_steps': 4,
                    'train_text_encoder': False, 'text_noise_scale': 0.0}
        if runtime['status'] != 'FINAL_PIPELINE_SAVED' or runtime['global_step'] != 2000 or any(args[k] != v for k, v in expected.items()):
            raise RuntimeError('REBUILD_TRAINING_AUDIT_MISMATCH:' + cls)
        root = a.checkpoint_root / 'Plastic_Bomo' / folder
        recorded = load(a.rebuild_audit / (cls + '_checkpoint_asset_audit.json'))['files_sha256']
        for relative, digest in recorded.items():
            if sha(root / relative) != digest:
                raise RuntimeError('REBUILD_CHECKPOINT_HASH_MISMATCH:' + relative)
        for relative, digest in assets['checkpoints'][cls]['files_sha256'].items():
            if sha(root / relative) != digest:
                raise RuntimeError('PREFLIGHT_MODEL_CHANGED:' + relative)
    if sha(a.split_manifest) != assets['sources']['manifest_sha256'] or sha(a.scheduler / 'scheduler_config.json') != assets['scheduler_sha256']:
        raise RuntimeError('FROZEN_INPUT_CHANGED')
    for relative, digest in assets['implementation_sha256'].items():
        if sha(REPO / relative) != digest:
            raise RuntimeError('INFERENCE_IMPLEMENTATION_CHANGED:' + relative)
    baseline_path = REPO / cfg['baseline_config']
    if sha(baseline_path) != assets['baseline_config_sha256']:
        raise RuntimeError('BASELINE_CONFIG_CHANGED')
    a.output.mkdir(parents=True, exist_ok=False)
    status = {'status': 'REBUILT_RUNTIME_PROBE_IN_PROGRESS', 'formal_generation_authorized': False,
              'detector_training_authorized': False, 'formal_slots_generated': 0,
              'probe_generation_count': 0, 'official_validation_use_count': 0,
              'deep_pcb_generation': 0, 'stage17b_auto_start': False}
    save(a.output / 'runtime_probe_status.json', status)
    import numpy as np
    import torch
    from PIL import Image
    from diffusers.configuration_utils import FrozenDict
    runtime = historical_runtime(a.checkpoint_root, baseline_path)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    normal_pool = runtime.get_valid_normal_images(
        str(a.generation_train_root / 'Plastic_Bomo/train/good'), runtime.args.normal_filter)
    rng = random.Random(42)
    probes = choose_probes(load(a.split_manifest), cfg['classes'])
    for row in probes:
        row['source_path'] = str((a.generation_train_root / 'Plastic_Bomo/test' / row['folder'] / row['image']).resolve())
        row['mask_path'] = str((a.generation_train_root / 'Plastic_Bomo/ground_truth' / row['folder'] / row['mask']).resolve())
        row['normal_path'] = rng.choice(normal_pool)
        for field in ('source_path', 'mask_path', 'normal_path'):
            row[field + '_sha256'] = sha(row[field])
        mask = runtime.mask_resize_center_crop(Image.open(row['mask_path']).convert('L'))
        binary = np.asarray(mask) > 127
        if not binary.any() or binary.all():
            raise RuntimeError('PROBE_MASK_INVALID_NO_REPLACEMENT:' + row['probe_id'])
    save(a.output / 'frozen_probe_conditions.json', {
        'scope': 'SMOKE_TEST_ONLY_NOT_FORMAL_80_SLOTS', 'probes': probes,
        'normal_pool_sha256': {str(Path(n).name): sha(n) for n in normal_pool},
        'selection': 'train metadata hash only; unchanged normal filter; independent Random(42)',
        'source_to_detector_parent_mapping': 'NOT_YET_VERIFIED_FOR_FORMAL_SLOTS'})
    records = []
    try:
        for repeat in (0, 1):
            for cls, folder in cfg['classes'].items():
                pipe = runtime.StableDiffusionInpaintPipeline_dynamic.from_pretrained(
                    str(a.checkpoint_root / 'Plastic_Bomo' / folder), torch_dtype=torch.float16,
                    local_files_only=True)
                pipe.scheduler = runtime.DDIMScheduler.from_pretrained(str(a.scheduler))
                new_cfg = dict(pipe.scheduler.config)
                keys = ('eta_mask_use_schedule', 'eta_mask_schedule', 'eta_mask_min',
                        'eta_mask_max', 'eta_mask_power', 'eta_mask_exp_k',
                        'eta_mask_sigmoid_k', 'eta_mask_guard', 'eta_mask_guard_margin',
                        'eta_mask_segmented', 'eta_mask_stop_step')
                new_cfg.update({k: getattr(runtime.args, k) for k in keys})
                pipe.scheduler._internal_dict = FrozenDict(new_cfg)
                pipe.to('cuda:' + a.device)
                pipe.set_progress_bar_config(disable=True)
                def check_finite(_module, _inputs, output):
                    value = output.sample if hasattr(output, 'sample') else output
                    if isinstance(value, tuple):
                        value = value[0]
                    if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                        raise RuntimeError('NONFINITE_PIPELINE_TENSOR')
                hooks = [pipe.unet.register_forward_hook(check_finite),
                         pipe.vae.decoder.register_forward_hook(check_finite)]
                try:
                    for bank, bank_seed in cfg['banks'].items():
                        for row in (r for r in probes if r['class'] == cls):
                            for field in ('source_path', 'mask_path', 'normal_path'):
                                if sha(row[field]) != row[field + '_sha256']:
                                    raise RuntimeError('PROBE_INPUT_CHANGED')
                            seed = sample_seed(bank_seed, row['probe_id'])
                            random.seed(seed); np.random.seed(seed % 2**32)
                            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                            normal = runtime.image_resize_center_crop(Image.open(row['normal_path']).convert('RGB'))
                            mask = runtime.mask_resize_center_crop(Image.open(row['mask_path']).convert('L'))
                            mask = Image.fromarray(((np.asarray(mask) > 127).astype('uint8') * 255))
                            image = runtime.inpaint(
                                pipe, normal, runtime.args.prompt, mask=mask, n_samples=1,
                                device='cuda:' + a.device, blur_factor=runtime.args.blur_factor,
                                guidance_scale_inside=runtime.args.guidance_scale_inside,
                                guidance_scale_outside=runtime.args.guidance_scale_outside,
                                num_inference_steps=runtime.args.num_inference_steps,
                                guidance_scale=runtime.args.guidance_scale)[0].convert('RGB')
                            array = np.asarray(image)
                            if image.size != (512, 512) or not np.isfinite(array).all():
                                raise RuntimeError('PROBE_OUTPUT_INVALID')
                            dst = a.output / f"{bank}_{row['probe_id']}_repeat{repeat}.png"
                            image.save(dst)
                            records.append({'class': cls, 'bank': bank, 'probe_id': row['probe_id'],
                                            'sample_seed': seed, 'repeat': repeat,
                                            'rgb_sha256': hashlib.sha256(array.tobytes()).hexdigest(),
                                            'image_path': str(dst.resolve())})
                            status['probe_generation_count'] = len(records)
                            save(a.output / 'runtime_probe_records.json', {'records': records})
                            print(f"probe {len(records)}/30 {cls} bank={bank} repeat={repeat}", flush=True)
                finally:
                    for hook in hooks:
                        hook.remove()
                    del pipe
                    torch.cuda.empty_cache()
        pairs = {}
        for row in records:
            pairs.setdefault((row['bank'], row['probe_id']), []).append(row['rgb_sha256'])
        ok = len(records) == 30 and len(pairs) == 15 and all(len(v) == 2 and v[0] == v[1] for v in pairs.values())
        status.update(status='REBUILT_RUNTIME_PROBE_PASS' if ok else 'REBUILT_RUNTIME_NONDETERMINISTIC',
                      reload_rgb_exact=ok, formal_pre_post_bank_reproducibility='NOT_RUN',
                      detector_parent_mapping='NOT_YET_VERIFIED')
        save(a.output / 'runtime_probe_status.json', status)
        print(json.dumps(status, indent=2), flush=True)
        if not ok:
            raise RuntimeError('REBUILT_RUNTIME_NONDETERMINISTIC')
    except Exception as e:
        status.update(status='REBUILT_RUNTIME_PROBE_FAILED', error=f'{type(e).__name__}: {e}')
        save(a.output / 'runtime_probe_status.json', status)
        raise


if __name__ == '__main__':
    main()
