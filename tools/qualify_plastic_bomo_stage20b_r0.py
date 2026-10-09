"""Stage20B-R0 GPU qualification only: no optimizer steps or formal experiment."""
import argparse
from contextlib import contextmanager, ExitStack
import gc
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # Never select a site-packages diffusion pipeline over frozen repo code.
from stage20b_runtime_contract import (Stop, augmentation_seed, binding_recheck, canonical,
                                      generator_budget_dryrun, htext, load, require, save, sha)
from stage20b_detector_controller import Stage20BDetectorController, accumulation_dryrun

OUTPUTS = ('frozen_binding_recheck', 'runtime_environment', 'stage20b_runtime_environment', 'generator_model_load_audit',
           'generator_forward_backward_audit', 'generator_determinism_probe', 'generator_rng_probe',
           'generator_budget_dryrun', 'detector_model_load_audit', 'detector_forward_backward_audit',
           'detector_accumulation_dryrun', 'detector_augmentation_rng_audit', 'detector_no_final_eval_audit',
           'formal_execution_plan', 'formal_launcher_manifest')


def tensor_sha(tensor):
    t = tensor.detach().contiguous().reshape(-1)
    h = hashlib.sha256(str(tensor.dtype).encode() + canonical(list(tensor.shape)).encode())
    # Bounded CPU copies: no multi-GB state_dict clone or pickle serialization.
    chunk = max(1, (16 * 1024 * 1024) // t.element_size())
    for start in range(0, t.numel(), chunk):
        h.update(t[start:start + chunk].view(__import__('torch').uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def parameter_hashes(model):
    return {name: tensor_sha(p) for name, p in model.named_parameters()}


def gradient_audit(model):
    import torch
    rows = {'parameters': 0, 'non_none_gradients': 0, 'nonzero_gradients': 0,
            'all_gradients_finite': True, 'all_trainable_parameters_fp32': True}
    for p in model.parameters():
        rows['parameters'] += 1
        if p.requires_grad:
            rows['all_trainable_parameters_fp32'] &= p.dtype == torch.float32
        if p.grad is not None:
            rows['non_none_gradients'] += 1
            rows['all_gradients_finite'] &= bool(torch.isfinite(p.grad).all())
            rows['nonzero_gradients'] += int(bool((p.grad != 0).any()))
    return rows


@contextmanager
def network_forbidden():
    def forbidden(*args, **kwargs):
        raise Stop('FROZEN_RUNTIME_BINDING_CHANGED', 'network/download attempted during offline R0')
    with patch.object(socket.socket, 'connect', forbidden), patch.object(socket, 'create_connection', forbidden):
        yield


@contextmanager
def optimizer_steps_forbidden(torch, counts):
    def forbidden(*args, **kwargs):
        counts['forbidden_optimizer_step_calls'] += 1
        raise Stop('R0_OPTIMIZER_STEP_FORBIDDEN', 'optimizer.step() attempted')
    with patch.object(torch.optim.AdamW, 'step', forbidden), patch.object(torch.optim.SGD, 'step', forbidden):
        yield


@contextmanager
def final_eval_content_forbidden(stage20a, p0, counts):
    """Deny Python/audit-enabled native file opens on metadata-known final_eval assets.

    The physical old folder named val is NOT itself blocked: some of its assets
    are frozen V2 TRAIN. Build exact forbidden paths from V2 split metadata.
    No final_eval image/label is opened to build this guard.
    """
    allow = load(p0 / 'generator_train_allowlist.json')['inputs']
    real_roots = {Path(x['path']).parents[2] for x in allow}
    require(len(real_roots) == 1, 'FROZEN_RUNTIME_BINDING_CHANGED', 'ambiguous real root')
    real_root = next(iter(real_roots))
    denied = set()
    for row in load(stage20a / 'real_source_registry.json')['records']:
        if row['split_v2'] == 'final_eval' and row['asset_role'] == 'detector':
            name = row['current_path'].replace('\\', '/').rsplit('/', 1)[-1]
            denied.add(str((real_root / 'images' / row['split_membership_old'] / name).resolve()))
            denied.add(str((real_root / 'labels' / row['split_membership_old'] / (Path(name).stem + '.txt')).resolve()))
    require(len(denied) == 96, 'FROZEN_RUNTIME_BINDING_CHANGED', 'expected48 forbidden image/label pairs')
    armed = [True]

    def audit(event, args):
        if armed[0] and event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
            if str(Path(os.fsdecode(args[0])).resolve()) in denied:
                counts['forbidden_final_eval_open_attempts'] += 1
                raise Stop('R0_FINAL_EVAL_CONTENT_ACCESS_FORBIDDEN', 'blocked before open: ' + os.fsdecode(args[0]))

    sys.addaudithook(audit)
    try:
        yield {'exact_image_label_paths_denied': len(denied), 'guard': 'Python open audit hook; no eval dataset constructed',
               'native_decoder_inputs': 'explicit frozen TRAIN-only paths'}
    finally:
        armed[0] = False


def crop_and_mask(row, use_normal=False):
    from PIL import Image
    path = Path(row['normal_path'] if use_normal else row['path'])
    digest = row['normal_sha256'] if use_normal else row['sha256']
    require(path.is_file() and sha(path) == digest, 'FROZEN_RUNTIME_BINDING_CHANGED', str(path))
    with Image.open(path) as im:
        im = im.convert('RGB')
        require(list(im.size) == row['geometry']['original_canvas_wh'], 'FROZEN_RUNTIME_BINDING_CHANGED', 'canvas dimensions')
        crop = im.crop(tuple(row['geometry']['fixed_crop_xyxy']))
    mask = Image.new('L', (512, 512), 0)
    mask.paste(255, tuple(row['geometry']['rasterized_mask_xyxy_exclusive']))
    actual = hashlib.sha256(json.dumps([512, 512]).encode() + mask.tobytes()).hexdigest()
    require(actual == row['geometry']['mask_decoded_sha256'], 'FROZEN_RUNTIME_BINDING_CHANGED', 'mask rasterization')
    return crop, mask


def set_gpu_runtime(torch, device):
    require(torch.cuda.is_available(), 'GENERATOR_RUNTIME_MODEL_STATE_INVALID', 'CUDA unavailable; CPU cannot qualify R0')
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    __import__('numpy').random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


def generator_qualification(a, cfg, p0, out, counts):
    a.failure_status = 'GENERATOR_RUNTIME_MODEL_STATE_INVALID'
    import torch
    import numpy as np
    import torch.nn.functional as F
    from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler, DDIMScheduler
    from PIL import Image
    pipeline_file = Path(inspect.getfile(StableDiffusionInpaintPipeline)).resolve()
    require(pipeline_file == REPO / 'diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py',
            'FROZEN_RUNTIME_BINDING_CHANGED', {'imported_pipeline': str(pipeline_file)})
    require(Path(inspect.getfile(DDIMScheduler)).resolve() == REPO / 'diffusers/schedulers/scheduling_ddim.py',
            'FROZEN_RUNTIME_BINDING_CHANGED', 'scheduler import shadowing')
    print('R0: loading hash-bound public SD2', flush=True)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(str(a.base_model), local_files_only=True,
            use_safetensors=True, torch_dtype=torch.float32, safety_checker=None, feature_extractor=None, requires_safety_checker=False)
    pipe.to(a.device)
    pipe.vae.requires_grad_(False).eval().to(dtype=torch.float16)
    pipe.text_encoder.requires_grad_(False).eval().to(dtype=torch.float16)
    pipe.unet.requires_grad_(True).train().to(dtype=torch.float32)
    pipe.unet.enable_gradient_checkpointing()
    pipe.disable_xformers_memory_efficient_attention()
    state = {'unet_all_trainable': all(p.requires_grad for p in pipe.unet.parameters()),
             'unet_all_fp32_master_parameters': all(p.dtype == torch.float32 for p in pipe.unet.parameters()),
             'vae_frozen': all(not p.requires_grad for p in pipe.vae.parameters()),
             'text_encoder_frozen': all(not p.requires_grad for p in pipe.text_encoder.parameters()),
             'gradient_checkpointing': bool(pipe.unet.is_gradient_checkpointing),
             'unet_in_channels': pipe.unet.config.in_channels,
             'mixed_precision': 'fp16 autocast / FP32 trainable master parameters',
             'optimizer_steps': 0, 'source': str(a.base_model)}
    train_cfg = load(p0 / 'stage20b_generator_training_protocol.json')
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=train_cfg['learning_rate'],
              betas=(train_cfg['adam_beta1'], train_cfg['adam_beta2']),
              weight_decay=train_cfg['adam_weight_decay'], eps=train_cfg['adam_epsilon'])
    state['optimizer_master_parameters_fp32'] = all(p.dtype == torch.float32 for g in optimizer.param_groups for p in g['params'])
    state['optimizer_configuration'] = {k: v for k, v in optimizer.param_groups[0].items() if k != 'params'}
    noise_scheduler = DDPMScheduler.from_pretrained(str(a.base_model), subfolder='scheduler', local_files_only=True)
    state['loaded_components'] = ['unet', 'vae', 'text_encoder', 'tokenizer', 'scheduler']
    state['training_noise_scheduler_prediction_type'] = noise_scheduler.config.prediction_type
    model_ok = all(state[k] for k in ('unet_all_trainable', 'unet_all_fp32_master_parameters', 'vae_frozen',
                    'text_encoder_frozen', 'gradient_checkpointing', 'optimizer_master_parameters_fp32'))
    model_ok = model_ok and state['unet_in_channels'] == 9 and optimizer.param_groups[0]['lr'] == 1e-6 and not optimizer.state
    state['status'] = 'PASS' if model_ok else 'FAIL'
    save(out / 'generator_model_load_audit.json', state)
    require(model_ok, 'GENERATOR_RUNTIME_MODEL_STATE_INVALID', state)
    fb = []
    allow = load(p0 / 'generator_train_allowlist.json')['inputs']
    for cls in ('flash', 'black'):
        a.failure_status = 'GENERATOR_FORWARD_BACKWARD_RUNTIME_INVALID'
        random.seed(train_cfg['seed'])
        np.random.seed(train_cfg['seed'])
        torch.manual_seed(train_cfg['seed'])
        torch.cuda.manual_seed_all(train_cfg['seed'])
        row = min((x for x in allow if x['class'] == cls), key=lambda x: x['annotation_id'])
        print('R0: generator forward/backward ' + cls, flush=True)
        im, mask = crop_and_mask(row)
        pixels = torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1)[None].to(a.device).float() / 127.5 - 1
        mask_t = torch.from_numpy(np.asarray(mask).copy())[None, None].to(a.device).float() / 255
        before = {n: parameter_hashes(m) for n, m in (('unet', pipe.unet), ('vae', pipe.vae), ('text_encoder', pipe.text_encoder))}
        pipe.unet.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            latents = pipe.vae.encode(pixels.half()).latent_dist.sample() * pipe.vae.config.scaling_factor
            masked = pipe.vae.encode((pixels * (mask_t < .5)).half()).latent_dist.sample() * pipe.vae.config.scaling_factor
            ids = pipe.tokenizer(train_cfg['prompts'][cls], padding=False, truncation=True,
                                 max_length=pipe.tokenizer.model_max_length, return_tensors='pt').input_ids.to(a.device)
            text = pipe.text_encoder(ids)[0]
        noise = torch.randn_like(latents)
        timestep = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=a.device).long()
        inp = torch.cat([noise_scheduler.add_noise(latents, noise, timestep), F.interpolate(mask_t, latents.shape[-2:]), masked], dim=1)
        with torch.autocast('cuda', dtype=torch.float16):
            pred = pipe.unet(inp, timestep, encoder_hidden_states=text).sample
        target = noise if noise_scheduler.config.prediction_type == 'epsilon' else noise_scheduler.get_velocity(latents, noise, timestep)
        loss = F.mse_loss(pred.float(), target.float(), reduction='mean')
        loss_value = float(loss.detach()) if bool(torch.isfinite(loss)) else None
        if loss_value is not None:
            # No GradScaler.step(), clipping, schedule step or optimizer.step().
            (loss / train_cfg['gradient_accumulation_steps']).backward()
        grad = gradient_audit(pipe.unet)
        after = {n: parameter_hashes(m) for n, m in (('unet', pipe.unet), ('vae', pipe.vae), ('text_encoder', pipe.text_encoder))}
        ok = (loss_value is not None and grad['non_none_gradients'] > 0 and grad['nonzero_gradients'] > 0
              and grad['all_gradients_finite'] and grad['all_trainable_parameters_fp32'] and before == after
              and all(p.grad is None for m in (pipe.vae, pipe.text_encoder) for p in m.parameters()))
        fb.append({'class': cls, 'annotation_id': row['annotation_id'], 'source_sha256': row['sha256'],
                   'prompt': train_cfg['prompts'][cls], 'loss': loss_value, 'gradients': grad,
                   'parameter_hashes_before': before, 'parameter_hashes_after': after,
                   'frozen_component_gradients_all_None': all(p.grad is None for m in (pipe.vae, pipe.text_encoder) for p in m.parameters()),
                   'parameter_bytes_unchanged': before == after, 'backward_loss_normalizer': 4,
                   'backward_mode': 'unscaled backward only; no optimizer/GradScaler step', 'pass': ok})
        fb_status = 'FAIL' if not all(x['pass'] for x in fb) else ('PASS' if len(fb) == 2 else 'IN_PROGRESS')
        save(out / 'generator_forward_backward_audit.json', {'status': fb_status, 'classes': fb, 'optimizer_steps': 0})
        require(ok, 'GENERATOR_FORWARD_BACKWARD_RUNTIME_INVALID', {'class': cls, 'loss': loss_value, 'gradients': grad})
        pipe.unet.zero_grad(set_to_none=True)
        del pixels, mask_t, latents, masked, ids, text, noise, timestep, inp, pred, target, loss
        torch.cuda.empty_cache()
    del optimizer
    a.failure_status = 'GENERATOR_RUNTIME_NONDETERMINISTIC'
    pipe.unet.eval().to(dtype=torch.float16)
    module = importlib.import_module(StableDiffusionInpaintPipeline.__module__)
    original_randn = module.randn_tensor
    rng_manifest = {x['slot_id']: x['banks'] for x in load(p0 / 'stage20b_rng_manifest.json')['records']}
    slots = load(p0 / 'stage20b_generation_slots.json')['slots']
    infer = cfg['generation']
    det_rows, rng_rows = [], []
    probe_dir = out / 'runtime_probe'
    probe_dir.mkdir()
    (probe_dir / 'NOT_FORMAL_SYNTHETIC.txt').write_text('Base-initialization runtime probes ONLY. Forbidden as detector training inputs.\n', encoding='utf-8')

    class NoiseOnlyDone(Exception):
        pass

    for cls in ('flash', 'black'):
        slot = next(x for x in slots if x['class'] == cls)  # frozen slot order; no quality selection
        im, mask = crop_and_mask(slot, use_normal=True)
        metadata = dict(slot)  # Paths are also frozen non-RNG metadata, identical across banks.
        runs = []
        for bank, repeat in (('A', 0), ('A', 1), ('B', 0)):
            print(f'R0: {cls} base pipeline bank{bank} repeat{repeat} (B noise-only)', flush=True)
            gen = torch.Generator(device='cpu').manual_seed(rng_manifest[slot['slot_id']][bank]['sample_seed'])
            before_state = tensor_sha(gen.get_state())
            captures = []

            def trace_noise(*args, **kwargs):
                noise_before = tensor_sha(gen.get_state())
                tensor = original_randn(*args, **kwargs)
                captures.append({'before_noise_state_sha256': noise_before, 'after_noise_state_sha256': tensor_sha(gen.get_state()),
                                 'initial_noise_sha256': tensor_sha(tensor), 'shape': list(tensor.shape)})
                if bank == 'B':
                    raise NoiseOnlyDone()
                return tensor

            pipe.scheduler = DDIMScheduler.from_config(infer['scheduler_config'])
            require(all(getattr(pipe.scheduler.config, k) == v for k, v in infer['scheduler_config'].items()),
                    'GENERATOR_RUNTIME_MODEL_STATE_INVALID', 'DDIM config default substitution')
            with patch.object(module, 'randn_tensor', trace_noise):
                try:
                    output = pipe(prompt=slot['prompt'], image=im, mask_image=mask,
                        num_inference_steps=infer['num_inference_steps'], guidance_scale=infer['guidance_scale'],
                        eta=infer['eta'], strength=infer['strength'], height=512, width=512,
                        generator=gen, negative_prompt=infer['negative_prompt'], num_images_per_prompt=1,
                        output_type='np', clip_skip=None, padding_mask_crop=None)
                except NoiseOnlyDone:
                    require(bank == 'B' and len(captures) == 1, 'GENERATOR_RUNTIME_RNG_INVALID', 'noise trace abort')
                    output = None
            require(len(captures) == 1 and captures[0]['shape'] == [1, 4, 64, 64],
                    'GENERATOR_RUNTIME_RNG_INVALID', {'slot': slot['slot_id'], 'noise_captures': captures})
            run = {'bank': bank, 'repeat': repeat, 'CPU_generator': True,
                   'sample_seed': rng_manifest[slot['slot_id']][bank]['sample_seed'], 'before_state_sha256': before_state,
                   'after_state_sha256': tensor_sha(gen.get_state()), **captures[0],
                   'non_rng_metadata_sha256': htext(canonical(metadata)), 'noise_only': bank == 'B'}
            if output is not None:
                array = np.asarray(output.images)
                require(array.shape == (1, 512, 512, 3) and bool(np.isfinite(array).all()),
                        'GENERATOR_RUNTIME_NONDETERMINISTIC', 'nonfinite/invalid probe output')
                rgb = (array[0] * 255).round().clip(0, 255).astype(np.uint8)
                run['generated_patch_RGB_sha256'] = hashlib.sha256(rgb.tobytes()).hexdigest()
                image_path = probe_dir / f'{cls}_base_A_repeat{repeat}_NOT_FORMAL_SYNTHETIC.png'
                Image.fromarray(rgb).save(image_path, format='PNG')
                run['probe_image'] = image_path.relative_to(out).as_posix()
                run['probe_png_sha256'] = sha(image_path)
                counts['runtime_probe_generated_images'] += 1
            runs.append(run)
            save(out / 'generator_rng_probe.json', {'status': 'IN_PROGRESS', 'records': rng_rows + [{'slot_id': slot['slot_id'], 'runs': runs}]})
        same = (runs[0]['generated_patch_RGB_sha256'] == runs[1]['generated_patch_RGB_sha256']
                and runs[0]['before_state_sha256'] == runs[1]['before_state_sha256']
                and runs[0]['after_state_sha256'] == runs[1]['after_state_sha256']
                and runs[0]['initial_noise_sha256'] == runs[1]['initial_noise_sha256'])
        independent = (runs[0]['initial_noise_sha256'] != runs[2]['initial_noise_sha256']
                       and runs[0]['before_state_sha256'] != runs[2]['before_state_sha256']
                       and runs[0]['after_noise_state_sha256'] != runs[2]['after_noise_state_sha256']
                       and len({x['non_rng_metadata_sha256'] for x in runs}) == 1)
        det_rows.append({'class': cls, 'slot_id': slot['slot_id'], 'two_same_seed_runs_bit_identical': same,
                         'checkpoint': 'original public base, NOT future fine-tuned class checkpoint', 'runs': runs[:2]})
        rng_rows.append({'class': cls, 'slot_id': slot['slot_id'], 'A_B_initial_noise_different': independent, 'runs': runs})
        det_status = 'FAIL' if not all(x['two_same_seed_runs_bit_identical'] for x in det_rows) else ('PASS' if len(det_rows) == 2 else 'IN_PROGRESS')
        rng_status = 'FAIL' if not all(x['A_B_initial_noise_different'] for x in rng_rows) else ('PASS' if len(rng_rows) == 2 else 'IN_PROGRESS')
        save(out / 'generator_determinism_probe.json', {'status': det_status, 'records': det_rows,
             'NOT_FORMAL_SYNTHETIC': True, 'formal_class_checkpoint_tested': False})
        save(out / 'generator_rng_probe.json', {'status': rng_status, 'records': rng_rows,
             'final_A_B_image_difference_required': False})
        require(same, 'GENERATOR_RUNTIME_NONDETERMINISTIC', cls)
        require(independent, 'GENERATOR_RUNTIME_RNG_INVALID', cls)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def draw_rng(seed, transforms):
    import torch
    import numpy as np
    import cv2
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    gpu = torch.cuda.get_rng_state_all()
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cv2.setRNGSeed(seed % (2**31 - 1))
    # Installed Albumentations (if present) owns its RNG; seed its actual nested Compose too.
    seen = set()
    def seed_transform(obj):
        if obj is None or id(obj) in seen:
            return
        seen.add(id(obj))
        if callable(getattr(obj, 'set_random_seed', None)):
            obj.set_random_seed(seed % (2**32))
        for child in (getattr(obj, 'transforms', None) or []):
            seed_transform(child)
        seed_transform(getattr(obj, 'transform', None))
    seed_transform(transforms)
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(gpu)


def detector_qualification(a, cfg, p0, out, counts):
    a.failure_status = 'DETECTOR_FORWARD_BACKWARD_RUNTIME_INVALID'
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.engine.trainer import BaseTrainer
    from ultralytics.data.dataset import YOLODataset
    order = load(p0 / 'stage20b_training_order_spec.json')
    controller = Stage20BDetectorController(cfg, order)
    print('R0: loading frozen YOLO11s and deterministic 2-class training head', flush=True)
    torch.manual_seed(42)
    native = DetectionTrainer.__new__(DetectionTrainer)  # No native __init__/get_dataset/test_loader or training loop.
    native.args = SimpleNamespace(**load(p0 / 'stage20b_detector_resolved_runtime_arguments.json')['arguments'])
    native.data = {'nc': 2, 'names': {0: 'flash', 1: 'black'}, 'channels': 3}
    native.device = torch.device(a.device)
    original = YOLO(str(a.detector_model))  # Local file existence/SHA checked first; network forbidden.
    native.model = native.get_model(cfg=original.model.yaml, weights=original.model, verbose=False).to(a.device).float()
    model = native.model
    native.set_model_attributes()
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not ('.dfl' in name))
    native.freeze_layer_names = ['.dfl']
    native.stride = max(32, int(model.stride.max()))
    if native.args.channels_last is True or native.args.channels_last is None:
        model.to(memory_format=torch.channels_last)
    # Calls the actual hash-bound native optimizer builder without its training loop.
    optimizer = native.build_optimizer(model, name='AdamW', lr=cfg['detector']['lr0'],
                   momentum=cfg['detector']['momentum'], decay=cfg['detector']['weight_decay'])
    for group in optimizer.param_groups:
        group['initial_lr'] = cfg['detector']['lr0']
    save(out / 'detector_model_load_audit.json', {'status': 'PASS', 'pretrained_sha256': sha(a.detector_model),
         'ultralytics_version': __import__('importlib.metadata', fromlist=['version']).version('ultralytics'),
         'nc': model.model[-1].nc, 'names': native.data['names'], 'head_initialization_seed': 42,
         'training_model_parameter_sha256': parameter_hashes(model), 'FP32_parameters': all(p.dtype == torch.float32 for p in model.parameters()),
         'optimizer': type(optimizer).__name__, 'optimizer_groups': [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups],
         'optimizer_steps': 0, 'automatic_download_allowed': False})
    require(model.model[-1].nc == 2 and all(p.dtype == torch.float32 for p in model.parameters()),
            'DETECTOR_FORWARD_BACKWARD_RUNTIME_INVALID', 'class count or parameter dtype')
    # Copies only two allowed TRAIN inputs: native label-cache writes stay under runtime_probe.
    probe_data = out / 'runtime_probe/train_batch_inputs'
    (probe_data / 'images/train').mkdir(parents=True)
    (probe_data / 'labels/train').mkdir(parents=True)
    inputs = load(p0 / 'generator_train_allowlist.json')['inputs']
    chosen = [min((x for x in inputs if x['class'] == cls), key=lambda x: x['annotation_id']) for cls in ('flash', 'black')]
    for row in chosen:
        suffix = Path(row['path']).suffix
        shutil.copyfile(row['path'], probe_data / 'images/train' / (row['annotation_id'] + suffix))
        shutil.copyfile(row['full_label_path'], probe_data / 'labels/train' / (row['annotation_id'] + '.txt'))
    ds = YOLODataset(img_path=str(probe_data / 'images/train'), imgsz=1536, batch_size=1, augment=True,
                     hyp=native.args, rect=False, cache=False, stride=native.stride, task='detect', data=native.data)
    require(len(ds) == 2, 'DETECTOR_RUNTIME_AUGMENTATION_INVALID', 'unexpected probe dataset size')
    samples, seeds_rows, fb = [], [], []
    a.failure_status = 'DETECTOR_RUNTIME_AUGMENTATION_INVALID'
    for epoch, position in ((0, 0), (0, 63), (0, 64), (0, 215), (149, 215)):
        seed = augmentation_seed(cfg['protocol'], epoch, position)
        require(seed == order['epochs'][epoch]['augmentation_seeds'][position], 'DETECTOR_RUNTIME_AUGMENTATION_INVALID', 'seed mismatch')
        snapshots = []
        for repeat in range(2):
            with draw_rng(seed, ds.transforms):
                sample = ds[position % 2]
            snapshots.append({k: tensor_sha(sample[k]) for k in ('img', 'cls', 'bboxes', 'batch_idx')})
            if epoch == 0 and position == 0 and repeat == 0:
                samples.append(sample)
        seeds_rows.append({'epoch': epoch, 'position': position, 'seed_uint64': seed,
                           'python_torch_seed': seed, 'numpy_seed': seed % (2**32), 'cv2_seed': seed % (2**31 - 1),
                           'repeated_augmentation_tensor_hashes': snapshots, 'repeat_identical': snapshots[0] == snapshots[1]})
    save(out / 'detector_augmentation_rng_audit.json', {'status': 'PASS' if all(x['repeat_identical'] for x in seeds_rows) else 'FAIL',
         'records': seeds_rows, 'native_training_transforms': repr(ds.transforms),
         'mosaic_mixup_copy_paste_cutmix': [native.args.mosaic, native.args.mixup, native.args.copy_paste, native.args.cutmix],
         'per_draw_fresh_seed': True, 'official_final_eval_inputs': 0})
    require(all(x['repeat_identical'] for x in seeds_rows), 'DETECTOR_RUNTIME_AUGMENTATION_INVALID', 'same-seed augmentation mismatch')
    a.failure_status = 'DETECTOR_FORWARD_BACKWARD_RUNTIME_INVALID'
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    before = parameter_hashes(model)
    native._model_train()
    model.zero_grad(set_to_none=True)
    a.failure_status = 'DETECTOR_RUNTIME_SCHEDULE_INVALID'
    batch = native.preprocess_batch(ds.collate_fn(samples))
    with torch.autocast('cuda', dtype=torch.float16):
        loss, items = model(batch)
        objective = loss.sum()
    require(isinstance(items, dict) and all(k in items for k in ('box_loss', 'cls_loss', 'dfl_loss')),
            'DETECTOR_FORWARD_BACKWARD_RUNTIME_INVALID', 'frozen native loss schema')
    finite = bool(torch.isfinite(objective)) and all(bool(torch.isfinite(v).all()) for v in items.values())
    if finite:
        controller.backward(objective, group_length=64)
    grad = gradient_audit(model)
    after = parameter_hashes(model)
    # Training-mode BN running buffers are expected to change; restore those ephemeral probe buffers.
    with torch.no_grad():
        for n, b in model.named_buffers():
            b.copy_(buffers[n])
    ok = finite and grad['all_gradients_finite'] and grad['non_none_gradients'] > 0 and grad['nonzero_gradients'] > 0 and before == after
    fb.append({'loss': float(objective.detach()) if finite else None,
               'native_loss_items': {k: float(v) if bool(torch.isfinite(v).all()) else None for k, v in items.items()},
               'gradients': grad, 'parameter_hashes_before': before, 'parameter_hashes_after': after,
               'parameter_bytes_unchanged': before == after, 'probe_only_BN_buffers_restored': True,
               'backward_normalizer': 64, 'optimizer_steps': 0})
    save(out / 'detector_forward_backward_audit.json', {'status': 'PASS' if ok else 'FAIL', 'records': fb})
    require(ok, 'DETECTOR_FORWARD_BACKWARD_RUNTIME_INVALID', {'finite_loss': finite, 'gradients': grad})
    model.zero_grad(set_to_none=True)
    # Real autograd checks the exact shared backward helper, including tail24, without any optimizer step.
    autograd_groups = []
    for length in (64, 24):
        w = torch.tensor(2.0, device=a.device, requires_grad=True)
        for value in range(1, length + 1):
            controller.backward(w * value, length)
        expected = (length + 1) / 2
        require(abs(float(w.grad) - expected) < 1e-5, 'DETECTOR_RUNTIME_SCHEDULE_INVALID', 'tail autograd normalization')
        autograd_groups.append({'draws': length, 'normalizer': length, 'gradient': float(w.grad), 'expected_mean_gradient': expected})
    accumulation = accumulation_dryrun(cfg, order)
    accumulation['GPU_autograd_normalization_groups'] = autograd_groups
    accumulation['native_optimizer_group_LR_probes'] = [
        {'epoch': e, 'position': pos, 'group_names': [g['param_group'] for g in optimizer.param_groups],
         'learning_rates': controller.learning_rates(e, pos, optimizer.param_groups),
         'AdamW_betas_unchanged': [list(g['betas']) for g in optimizer.param_groups]}
        for e, pos in ((0, 0), (0, 63), (0, 64), (0, 215), (3, 0), (149, 215))]
    save(out / 'detector_accumulation_dryrun.json', accumulation)
    # Mock every native validation entry point during BOTH exit paths of the actual custom controller.
    a.failure_status = 'DETECTOR_AUTOMATIC_FINAL_EVAL_NOT_SUPPRESSED'
    forbidden = Mock(side_effect=Stop('DETECTOR_AUTOMATIC_FINAL_EVAL_NOT_SUPPRESSED', 'automatic validation called'))
    cleanup = Mock()
    with ExitStack() as stack:
        for owner, name in ((BaseTrainer, 'final_eval'), (BaseTrainer, 'validate'), (YOLO, 'val')):
            stack.enter_context(patch.object(owner, name, forbidden))
        native.validator = forbidden
        native.stopper = forbidden
        completed = controller.run(lambda *x: None, lambda *x: None, cleanup=cleanup)
        try:
            def raise_probe(*args):
                raise ValueError('R0 injected exit-path test')
            controller.run(raise_probe, lambda *x: None, cleanup=cleanup)
        except ValueError:
            pass
    save(out / 'detector_no_final_eval_audit.json', {'status': 'PASS' if forbidden.call_count == 0 and cleanup.call_count == 2 else 'FAIL',
         'native_final_eval_validate_YOLO_val_mock_calls': forbidden.call_count,
         'normal_exit_counts': completed, 'exception_exit_cleanup_calls': cleanup.call_count,
         'native_training_loop_invoked': False, 'custom_controller': 'Stage20BDetectorController.run',
         'early_stopper_not_called': True, 'no_val_dataset_or_validator_constructed': True})
    require(forbidden.call_count == 0 and cleanup.call_count == 2,
            'DETECTOR_AUTOMATIC_FINAL_EVAL_NOT_SUPPRESSED', 'exit-path mock')
    del model, native, original, optimizer, ds, batch, loss, objective, samples, buffers
    gc.collect()
    torch.cuda.empty_cache()


def formal_plan():
    return {'status': 'FROZEN_SEQUENCE_ONLY_NOT_STARTED', 'steps': [
        '1 train Flash exactly2000 successful updates', '2 train Black exactly2000 successful updates',
        '3 freeze all component bytes of both final_step2000 pipelines',
        '4 require both class checkpoints frozen before any formal sampling',
        '5 generate RawGenerator-A 80 frozen slots', '6 generate RawGenerator-B 80 frozen slots',
        '7 generate RawGenerator-C 80 frozen slots', '8 check all240 structural outputs; any failure stops without reroll',
        '9 prepare identical136-real-base RR/A/B/C with80 extra role exposures',
        '10 train all4 detectors and hash-freeze all epoch150 last.pt files',
        '11 only then perform one unified V2 final_eval round'],
        'formal_training_or_sampling_started': False, 'TDCRG_authorized': False,
        'success_gates_unchanged': {'mean_delta_pp_at_least': .50, 'strict_bank_wins_at_least': 2,
                                   'each_class_mean_delta_pp_at_least': -2.0}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage20a', type=Path, default=REPO / 'results_for_gpt/plastic_bomo_stage20a_v2_protocol')
    p.add_argument('--p0', type=Path, default=REPO / 'results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115')
    p.add_argument('--protocol', type=Path, default=REPO / 'configs/plastic_bomo_stage20b_p0.json')
    p.add_argument('--base_model', type=Path, required=True)
    p.add_argument('--detector_model', type=Path, required=True)
    p.add_argument('--device', default='0')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    for key in ('stage20a', 'p0', 'protocol', 'base_model', 'detector_model', 'output'):
        setattr(a, key, getattr(a, key).resolve())
    if a.output.exists():
        raise RuntimeError('R0_OUTPUT_EXISTS_USE_NEW_DIRECTORY_NO_OVERWRITE: ' + str(a.output))
    a.output.mkdir(parents=True)
    counts = {'optimizer_step_count': 0, 'forbidden_optimizer_step_calls': 0, 'forbidden_final_eval_open_attempts': 0,
              'generator_formal_training_count': 0, 'formal_synthetic_generation_count': 0,
              'generator_training_count': 0, 'synthetic_generation_count': 0,
              'detector_training_count': 0, 'final_eval_forward_count': 0,
              'official_final_eval_forward': 0, 'final_eval_image_or_label_content_read': 0,
              'official_final_eval_content_read': 0, 'runtime_probe_generated_images': 0,
              'DeepPCB': 0, 'BootstrapGuard': 0, 'TDCRG_implementation': 0}
    status = {'status': 'STAGE20B_R0_SERVER_RUNTIME_PENDING', 'STAGE20B_FORMAL_EXECUTION_READY': False,
              'TDCRG_DEVELOPMENT_AUTHORIZED': False, 'formal_auto_start': False}
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1',
                      CUBLAS_WORKSPACE_CONFIG=':4096:8', YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true',
                      NO_ALBUMENTATIONS_UPDATE='1')
    safety = ExitStack()
    phase = 'FROZEN_RUNTIME_BINDING_CHANGED'
    try:
        print('R0: frozen byte/runtime/allowed TRAIN asset recheck (not a P0 rerun)', flush=True)
        binding = binding_recheck(REPO, a.stage20a, a.p0, a.protocol, a.base_model, a.detector_model)
        save(a.output / 'frozen_binding_recheck.json', binding)
        content_guard = safety.enter_context(final_eval_content_forbidden(a.stage20a, a.p0, counts))
        cfg = load(a.protocol)
        train = load(a.p0 / 'stage20b_generator_training_protocol.json')
        class_inputs = {'flash': 75, 'black': 97}
        save(a.output / 'generator_budget_dryrun.json', {'status': 'PASS', 'protocol': train,
             'no_skip_scenario': generator_budget_dryrun(train, class_inputs),
             'injected_metadata_only_skip_scenario': generator_budget_dryrun(train, class_inputs, (1, 7, 31, 1999)),
             'optimizer_steps': 0, 'legacy_attempt_counted_main_not_used': True,
             'successful_update_counter_required_in_future_wrapper': True})
        save(a.output / 'formal_execution_plan.json', formal_plan())
        phase = 'GENERATOR_RUNTIME_MODEL_STATE_INVALID'
        a.failure_status = phase
        with network_forbidden():
            import torch
            a.device = 'cuda:' + a.device
            set_gpu_runtime(torch, a.device)
            environment = {'status': 'PASS', 'package_versions': binding['package_versions'],
                 'source_hashes': binding['runtime_source_files'], 'python': sys.version,
                 'torch_version': torch.__version__, 'CUDA_version': torch.version.cuda,
                 'cudnn_version': torch.backends.cudnn.version(), 'device': a.device,
                 'device_name': torch.cuda.get_device_name(a.device), 'capability': list(torch.cuda.get_device_capability(a.device)),
                 'TF32': False, 'cudnn_benchmark': False, 'deterministic_algorithms': True,
                 'CUBLAS_WORKSPACE_CONFIG': os.environ['CUBLAS_WORKSPACE_CONFIG'], 'package_change_count': 0,
                 'final_eval_content_guard': content_guard}
            save(a.output / 'runtime_environment.json', environment)
            save(a.output / 'stage20b_runtime_environment.json', environment)
            with optimizer_steps_forbidden(torch, counts):
                generator_qualification(a, cfg, a.p0, a.output, counts)
                phase = 'DETECTOR_FORWARD_BACKWARD_RUNTIME_INVALID'
                detector_qualification(a, cfg, a.p0, a.output, counts)
        # Frozen metadata/code may not be changed by probes; file ledger validation only, not P0 reslotting.
        a.failure_status = 'FROZEN_RUNTIME_BINDING_CHANGED'
        binding_after = binding_recheck(REPO, a.stage20a, a.p0, a.protocol, a.base_model, a.detector_model)
        require(binding == binding_after, 'FROZEN_RUNTIME_BINDING_CHANGED', 'probe changed frozen binding')
        binding['post_probe_binding_recheck_identical'] = True
        save(a.output / 'frozen_binding_recheck.json', binding)
        runtime_audits = [name for name in OUTPUTS if name not in ('formal_execution_plan', 'formal_launcher_manifest')]
        for name in runtime_audits:
            require(load(a.output / (name + '.json'))['status'] == 'PASS', 'R0_RUNTIME_CHECKS_INCOMPLETE', name)
        for name, key in (('generator_forward_backward_audit', 'classes'), ('generator_determinism_probe', 'records'),
                          ('generator_rng_probe', 'records')):
            require(len(load(a.output / (name + '.json'))[key]) == 2, 'R0_RUNTIME_CHECKS_INCOMPLETE', name)
        require(counts['forbidden_optimizer_step_calls'] == counts['forbidden_final_eval_open_attempts'] == 0,
                'R0_RUNTIME_CHECKS_INCOMPLETE', 'forbidden execution attempted')
        status.update(status='STAGE20B_RUNTIME_QUALIFICATION_PASS', STAGE20B_FORMAL_EXECUTION_READY=True)
        save(a.output / 'formal_launcher_manifest.json', {'status': 'NOT_CREATED_OR_STARTED',
             'reason': 'R0 only qualifies runtime; no executable formal launcher is silently created or run',
             'R0_runtime_qualified': True, 'formal_execution_plan': 'formal_execution_plan.json',
             'future_launcher_must_preserve': ['all frozen slots/banks/hyperparameters', 'both class checkpoint byte freezes before sampling',
                                             'all4 epoch150 last.pt byte freezes before final_eval'], 'auto_start': False})
    except Stop as error:
        status.update(status=error.status, detail=error.detail)
    except Exception as error:
        import traceback
        traceback.print_exc()
        status.update(status=getattr(a, 'failure_status', phase), detail={'exception_type': type(error).__name__, 'message': str(error)},
                      automatic_retry_or_protocol_change=False)
    finally:
        safety.close()
        status.update(counts)
        status.update(scope='SERVER_RUNTIME_QUALIFICATION', Stage20A_modified=False, Stage20B_P0_modified=False,
                      protocol_sha256=sha(a.protocol) if a.protocol.is_file() else None,
                      implementation_sha256={name: sha(REPO / 'tools' / name) for name in (
                          'qualify_plastic_bomo_stage20b_r0.py', 'stage20b_runtime_contract.py', 'stage20b_detector_controller.py')})
        for name in OUTPUTS:
            path = a.output / (name + '.json')
            if not path.exists():
                save(path, {'status': 'NOT_RUN', 'blocked_by': status['status']})
        # Snapshot partially finished audits, never call a partial class/phase result a complete R0 pass.
        save(a.output / 'stage20b_r0_status.json', status)
        report = f"""# Stage20B-R0 Runtime Qualification\n\nStatus: `{status['status']}`. Formal execution ready: `{status['STAGE20B_FORMAL_EXECUTION_READY']}`.\n\nOptimizer steps, formal generator training, formal synthetic generation, detector training and official final_eval remain0. Runtime probe images: {counts['runtime_probe_generated_images']}; these use the public base, are marked NOT_FORMAL_SYNTHETIC and cannot enter any detector dataset. A/B initial-noise independence does not require final-image differences.\n\nFrozen Stage20A and authoritative server P0 are unchanged. Failure stops without package changes, LR/CFG/steps/mask/seed changes, retry or reroll. Runtime qualification is not a utility result or evidence that future trained checkpoints are deterministic; their actual bytes must be bound before formal sampling.\n\nThe independent custom detector controller uses the frozen216-role schedule, group means64/64/64/24, and never invokes native .train(), validator, early stopper or final_eval. R0 checks its full150-epoch metadata exit path, both normal and exceptional exits, native TRAIN augmentation, native loss/backward and actual autograd tail normalization. It does not execute real600 optimizer attempts or150 GPU training epochs.\n\nOnly V2 TRAIN assets are accessed. Physical source completeness remains false; supervision is dataset-defined; rectangular bbox masks are weak geometry, not segmentation truth. V2 absolute mAP must not be compared with Stage14-18 absolute mAP. TDCRG, BootstrapGuard and DeepPCB remain untouched. Formal launchers/experiments are not automatically started.\n"""
        (a.output / 'STAGE20B_R0_REPORT.md').write_text(report, encoding='utf-8', newline='\n')
        save(a.output / 'r0_artifact_manifest.json', {'sha256': {x.relative_to(a.output).as_posix(): sha(x)
             for x in sorted(a.output.rglob('*')) if x.is_file()}, 'no_overwrite': True})
        print(json.dumps(status, indent=2, ensure_ascii=False), flush=True)
    if status['status'] != 'STAGE20B_RUNTIME_QUALIFICATION_PASS':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
