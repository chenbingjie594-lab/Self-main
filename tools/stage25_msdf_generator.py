"""Manifest-only V2 data shell; frozen MSDF v3 hooks and literal historical losses."""
import ast
from collections import Counter
import gc
import inspect
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

from stage25_msdf_contract import (BANKS, REPO, binding, safety_binding, guard, claim,
    load, save, sha, htext, canonical, require, tree_binding, verify_tree, verified_copy, update)

HELPERS = ('make_soft_prior', 'augment_coarse_mask', 'build_defect_saliency', 'multiscale_local_consistency_loss')
LOSSES = ('region', 'reconstruction', 'structure', 'background', 'scale', 'support')


def historical_nodes():
    """Select literal code, never import/execute generic train_rda main or its teachers."""
    source = (REPO / 'train_rda.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    helpers = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name in HELPERS]
    branches = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.If) or ast.unparse(node.test) != 'msdf is not None':
            continue
        assigned = {x.id for statement in node.body for x in ast.walk(statement) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Store)}
        if {'msdf_region_loss', 'diffusion_objective'} <= assigned:
            branches.append(node)
    require(len(helpers) == 4 and len(branches) == 1, 'STAGE25_FROZEN_BINDING_CHANGED', 'historical loss extraction ambiguous')
    return source, helpers, branches[0]


def kernel(torch, F):
    source, helpers, branch = historical_nodes()
    env = {'torch': torch, 'F': F, 'random': random}
    exec(compile(ast.Module(body=helpers, type_ignores=[]), 'train_rda.py:frozen_helpers', 'exec'), env)
    code = compile(ast.Module(body=branch.body, type_ignores=[]), 'train_rda.py:frozen_msdf_loss', 'exec')
    identity = {'train_rda_sha256': sha(REPO / 'train_rda.py'),
        'helpers': {n.name: htext(ast.get_source_segment(source, n)) for n in helpers},
        'auxiliary_loss_body_sha256': htext('\n'.join(ast.get_source_segment(source, x) for x in branch.body)),
        'generic_main_executed': False, 'teacher_or_selector_imported': False}
    return env, code, identity


def check_pipeline(pipe_class, scheduler_class):
    require(Path(inspect.getfile(pipe_class)).resolve() == REPO / 'diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py',
            'STAGE25_FROZEN_BINDING_CHANGED', 'pipeline import shadowing')
    require(Path(inspect.getfile(scheduler_class)).resolve() == REPO / 'diffusers/schedulers/scheduling_ddim.py',
            'STAGE25_FROZEN_BINDING_CHANGED', 'scheduler import shadowing')


def construct(torch, b, cfg, cls):
    from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler, DDIMScheduler
    from diffusers.pipelines.stable_diffusion.msdf_guidance import build_msdf_adapter
    import torch.nn.functional as F
    check_pipeline(StableDiffusionInpaintPipeline, DDIMScheduler)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(b['base_model'], local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float32, safety_checker=None,
        feature_extractor=None, requires_safety_checker=False).to(b['device'])
    pipe.vae.requires_grad_(False).eval().to(dtype=torch.float16)
    pipe.text_encoder.requires_grad_(False).eval().to(dtype=torch.float16)
    pipe.unet.requires_grad_(True).train().to(dtype=torch.float32)
    pipe.unet.enable_gradient_checkpointing()
    pipe.disable_xformers_memory_efficient_attention()
    d = cfg['MSDF_preserved_defaults']
    msdf = build_msdf_adapter(pipe.unet, 4, d['hidden_dim'], d['context_scale'], d['max_injection'],
        d['max_residual_ratio'], d['branch_dropout'], d['pixel_support_weight']).to(b['device'], dtype=torch.float32).train()
    require(msdf.attach(pipe.unet) == 4, 'STAGE25_MSDF_SHAPE_INVALID', 'four up blocks required')
    scheduler = DDPMScheduler.from_pretrained(b['base_model'], subfolder='scheduler', local_files_only=True)
    require(scheduler.config.prediction_type in ('epsilon', 'v_prediction'), 'STAGE25_MSDF_SHAPE_INVALID', 'prediction target')
    training = cfg['common_generator_training_target']
    ids = pipe.tokenizer(training['prompts'][cls], padding=False, truncation=True,
        max_length=pipe.tokenizer.model_max_length, return_tensors='pt').input_ids.to(b['device'])
    uncond = pipe.tokenizer('', padding='max_length', max_length=ids.shape[1], truncation=True,
        return_tensors='pt').input_ids.to(b['device'])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        text = torch.cat([pipe.text_encoder(uncond)[0], pipe.text_encoder(ids)[0]])
    env, code, identity = kernel(torch, F)
    identity['tokenizer_runtime_adaptation'] = 'conditional padding=False identical to Raw; unconditional padded to same conditional width; doubled CFG forward retained'
    args = SimpleNamespace(**{'msdf_' + n + '_weight': d[n + '_weight'] for n in LOSSES})
    return pipe, msdf, scheduler, text, env, code, args, identity


def microbatch(objects, row, reference_row, b, cfg, torch):
    import numpy as np
    import torch.nn.functional as F
    from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
    pipe, msdf, scheduler, text, env, code, args, _ = objects
    def tensors(r):
        image, mask = crop_and_mask(r)
        return (torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].to(b['device']).float() / 127.5 - 1,
                torch.from_numpy(np.asarray(mask).copy())[None, None].to(b['device']).float() / 255)
    pixels, original_mask = tensors(row)
    reference, reference_mask = tensors(reference_row)
    d = cfg['MSDF_preserved_defaults']
    mask = env['augment_coarse_mask'](original_mask, d['mask_jitter_radius'])
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        latents = pipe.vae.encode(pixels.half()).latent_dist.sample() * pipe.vae.config.scaling_factor
        masked = pipe.vae.encode((pixels * (mask < .5)).half()).latent_dist.sample() * pipe.vae.config.scaling_factor
        ref_latents = pipe.vae.encode(reference.half()).latent_dist.mode() * pipe.vae.config.scaling_factor
    soft_prior = env['make_soft_prior'](mask, latents.shape[-2:], d['mask_blur_radius'])
    saliency = env['build_defect_saliency'](pixels, original_mask, latents.shape[-2:],
        d['saliency_low_quantile'], d['saliency_high_quantile']).detach()
    noise = torch.randn_like(latents)
    timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=b['device']).long()
    noisy_latents = scheduler.add_noise(latents, noise, timesteps)
    inp = torch.cat([noisy_latents, soft_prior.to(noisy_latents.dtype), masked], 1)
    require(list(inp.shape) == [1, 9, 64, 64], 'STAGE25_MSDF_SHAPE_INVALID', list(inp.shape))
    msdf.prepare(ref_latents.float(), reference_mask, soft_prior, timesteps,
        scheduler.config.num_train_timesteps, classifier_free_guidance=True, reference_pixels=reference)
    with torch.autocast('cuda', dtype=torch.float16):
        prediction = pipe.unet(torch.cat([inp, inp]), torch.cat([timesteps, timesteps]), encoder_hidden_states=text).sample
        prediction_cond = prediction.chunk(2)[1]
        target = noise if scheduler.config.prediction_type == 'epsilon' else scheduler.get_velocity(latents, noise, timesteps)
        context = dict(env, args=args, msdf=msdf, scheduler=scheduler, device=torch.device(b['device']),
            pixel_error=(prediction_cond.float() - target.float()).square(),
            conditional_loss=F.mse_loss(prediction_cond.float(), target.float()), defect_saliency=saliency,
            reference_is_self=torch.tensor([row['annotation_id'] == reference_row['annotation_id']], device=b['device']),
            timesteps=timesteps, noisy_latents=noisy_latents, latents=latents,
            soft_prior=soft_prior, prediction_cond=prediction_cond,
            **{'msdf_' + n + '_loss': torch.zeros((), device=b['device']) for n in LOSSES})
        exec(code, context)
        loss = context['diffusion_objective']
    values = {n: float(context['msdf_' + n + '_loss'].detach()) for n in LOSSES}
    values['denoising'] = float(context['conditional_loss'].detach())
    require(bool(torch.isfinite(loss)) and all(math.isfinite(v) for v in values.values()), 'STAGE25_NONFINITE_LOSS', values)
    return loss, values, {'pixels': list(pixels.shape), 'mask': list(mask.shape), 'reference_latents': list(ref_latents.shape),
        'prior': list(soft_prior.shape), 'UNet_input': [2, *list(inp.shape[1:])], 'prediction': list(prediction.shape),
        'text_embeddings': list(text.shape), 'annotation_id': row['annotation_id'], 'reference_annotation_id': reference_row['annotation_id']}


def frozen_hashes(pipe):
    from qualify_plastic_bomo_stage20b_r0 import parameter_hashes
    return {n: parameter_hashes(m) for n, m in (('vae', pipe.vae), ('text_encoder', pipe.text_encoder))}


def gradient_record(module, torch):
    disconnected, nonfinite, nonzero, zero = [], [], [], []
    for n, p in module.named_parameters():
        if p.grad is None:
            disconnected.append(n)
        elif not bool(torch.isfinite(p.grad).all()):
            nonfinite.append(n)
        elif bool(torch.count_nonzero(p.grad)):
            nonzero.append(n)
        else:
            zero.append(n)
    return {'all_connected': not disconnected, 'all_finite': not nonfinite, 'disconnected': disconnected,
        'nonfinite': nonfinite, 'nonzero_parameter_names': nonzero, 'zero_but_connected_parameter_names': zero}


def safety(runtime, cls):
    runtime = Path(runtime)
    b, cfg = safety_binding(runtime)
    claim(runtime, 'safety_' + cls)
    result = {'status': 'NOT_RUN', 'class': cls, 'optimizer_steps': 0, 'sampling_count': 0, 'final_eval_content_read': 0}
    try:
        with guard(runtime, safety=True) as torch:
            torch.cuda.reset_peak_memory_stats()
            objects = construct(torch, b, cfg, cls)
            pipe, msdf = objects[:2]
            before = frozen_hashes(pipe)
            row = min((r for r in load(Path(b['p0_view']) / 'generator_train_allowlist.json')['inputs'] if r['class'] == cls), key=lambda r: r['annotation_id'])
            loss, values, shapes = microbatch(objects, row, row, b, cfg, torch)
            (loss / cfg['common_generator_training_target']['gradient_accumulation_steps']).backward()
            grads = {'unet': gradient_record(pipe.unet, torch), 'msdf': gradient_record(msdf, torch)}
            frozen_ok = (before == frozen_hashes(pipe) and all(not p.requires_grad and p.grad is None
                for module in (pipe.vae, pipe.text_encoder) for p in module.parameters()))
            ok = frozen_ok and all(r['all_connected'] and r['all_finite'] and r['nonzero_parameter_names'] for r in grads.values())
            require(all(p.requires_grad and p.dtype == torch.float32 for m in (pipe.unet, msdf) for p in m.parameters()),
                    'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', 'trainable FP32 masters')
            result.update(status='PASS' if ok else 'FAIL', loss=float(loss.detach()), losses=values, gradients=grads,
                tensor_shapes=shapes, VAE_text_frozen_unchanged=frozen_ok, historical_kernel=objects[-1],
                allocated_VRAM_bytes=torch.cuda.memory_allocated(), reserved_VRAM_bytes=torch.cuda.memory_reserved(),
                peak_VRAM_bytes=torch.cuda.max_memory_allocated(), peak_reserved_VRAM_bytes=torch.cuda.max_memory_reserved(),
                normal_allowlist_count=394, fresh_public_start=True, checkpoint_saved=False)
            require(ok, 'STAGE25_SERVER_RUNTIME_NOT_QUALIFIED', 'safety connectivity/finite/frozen check')
    finally:
        save(runtime / ('msdf_' + cls + '_safety.json'), result)


def train(run, cls):
    run = Path(run)
    b, cfg = binding(run)
    claim(run, 'msdf_train_' + cls)
    if cls == 'black':
        first = load(run / 'msdf_flash_training_audit.json')
        require(first['status'] == 'COMPLETE' and first['successful_updates'] == 2000, 'STAGE25_GENERATOR_BARRIER_FAILED', 'Flash first')
        verify_tree(run, first['checkpoint'])
    rows = sorted((r for r in load(Path(b['p0_view']) / 'generator_train_allowlist.json')['inputs'] if r['class'] == cls), key=lambda r: r['annotation_id'])
    audit = {'status': 'RUNNING', 'class': cls, 'successful_updates': 0, 'scheduled_attempts': 0, 'AMP_skips': 0,
        'microbatch_draws': 0, 'final_eval_content_read': 0, 'input_annotation_ids': [r['annotation_id'] for r in rows],
        'checkpoint_saved': False, 'automatic_resume': False, 'training_protocol_sha256': b['protocol_sha256']}
    started = time.monotonic()
    path = run / ('msdf_' + cls + '_training_audit.json')
    save(path, audit)
    update(run, **{'msdf_' + cls + '_training_started': True})
    try:
        with guard(run) as torch:
            objects = construct(torch, b, cfg, cls)
            pipe, msdf = objects[:2]
            before = frozen_hashes(pipe)
            t = cfg['common_generator_training_target']
            opt = torch.optim.AdamW([{'params': list(pipe.unet.parameters()), 'lr': t['learning_rate']},
                {'params': list(msdf.parameters()), 'lr': cfg['MSDF_preserved_defaults']['MSDF_learning_rate']}],
                betas=(t['adam_beta1'], t['adam_beta2']), weight_decay=t['adam_weight_decay'], eps=t['adam_epsilon'])
            scaler = torch.amp.GradScaler('cuda', enabled=True)
            actual = [0]
            hook = opt.register_step_post_hook(lambda *args, **kw: actual.__setitem__(0, actual[0] + 1))
            params = [p for g in opt.param_groups for p in g['params']]
            audit.update(historical_kernel=objects[-1], optimizer_groups=[{k: v for k, v in g.items() if k != 'params'} for g in opt.param_groups],
                epoch_order_sha256=[], precision='fp16 autocast, FP32 full UNet/MSDF masters', epoch_tail_divisor=4)
            epoch, skipped_tail = 0, 0
            with (run / 'logs' / ('msdf_' + cls + '_updates.jsonl')).open('x', encoding='utf-8') as log:
                while actual[0] < t['max_train_steps']:
                    order = torch.randperm(len(rows), generator=torch.Generator(device='cpu').manual_seed(t['seed'] + epoch)).tolist()
                    audit['epoch_order_sha256'].append(htext(canonical([rows[i]['annotation_id'] for i in order])))
                    for offset in range(0, len(order), t['gradient_accumulation_steps']):
                        opt.zero_grad(set_to_none=True)
                        records = []
                        for i in order[offset:offset + t['gradient_accumulation_steps']]:
                            j = i
                            if random.random() < cfg['MSDF_preserved_defaults']['reference_cross_probability']:
                                j = (i + random.randint(1, len(rows) - 1)) % len(rows)
                            loss, losses, shapes = microbatch(objects, rows[i], rows[j], b, cfg, torch)
                            scaler.scale(loss / t['gradient_accumulation_steps']).backward()
                            records.append({**shapes, 'loss': float(loss.detach()), 'losses': losses})
                            audit['microbatch_draws'] += 1
                            del loss
                        audit['scheduled_attempts'] += 1
                        scaler.unscale_(opt)
                        finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params)
                        norm = torch.nn.utils.clip_grad_norm_(params, t['max_grad_norm'])
                        require(not finite or bool(torch.isfinite(norm)), 'STAGE25_GENERATOR_TRAINING_FAILED', 'invalid clip norm')
                        step0, scale0 = actual[0], scaler.get_scale()
                        scaler.step(opt)
                        scaler.update()
                        success = actual[0] == step0 + 1
                        audit['successful_updates'] = actual[0]
                        audit['AMP_skips'] += int(not success)
                        skipped_tail = 0 if success else skipped_tail + 1
                        require(math.isfinite(scaler.get_scale()) and scaler.get_scale() > 0 and skipped_tail < 20,
                            'STAGE25_GENERATOR_TRAINING_FAILED', 'invalid scaler or20 consecutive skips; no rescue')
                        log.write(canonical({'epoch': epoch, 'offset': offset, 'attempt': audit['scheduled_attempts'],
                            'successful_updates': actual[0], 'AMP_skip': not success, 'gradients_finite': finite,
                            'gradient_norm': float(norm) if bool(torch.isfinite(norm)) else None,
                            'scale_before': scale0, 'scale_after': scaler.get_scale(), 'microbatches': records}) + '\n')
                        log.flush()
                        save(path, audit)
                        if actual[0] % 25 == 0 or not success:
                            print(f'MSDF {cls}: successful={actual[0]} attempts={audit["scheduled_attempts"]} skips={audit["AMP_skips"]}', flush=True)
                        if actual[0] == t['max_train_steps']:
                            require(success and finite and bool(torch.isfinite(norm)), 'STAGE25_GENERATOR_TRAINING_FAILED', 'last update invalid')
                            break
                    epoch += 1
            hook.remove()
            require(actual[0] == 2000 and all(bool(torch.isfinite(p).all()) for p in params), 'STAGE25_GENERATOR_TRAINING_FAILED', '2000/finiteness barrier')
            require(before == frozen_hashes(pipe), 'STAGE25_GENERATOR_TRAINING_FAILED', 'frozen VAE/text changed')
            dest = run / 'generator_checkpoints' / cls / 'final_step2000'
            require(not dest.exists(), 'STAGE25_OUTPUT_EXISTS', str(dest))
            pipe.unet.save_pretrained(str(dest / 'unet'), safe_serialization=True)
            for name, row in load(Path(b['p0_view']) / 'sd2_base_weight_binding.json')['files'].items():
                if name == 'unet/diffusion_pytorch_model.safetensors':
                    continue
                target = dest / name
                if target.exists():
                    require(name == 'unet/config.json', 'STAGE25_OUTPUT_EXISTS', name)
                    target.write_bytes((Path(b['base_model']) / name).read_bytes())
                else:
                    verified_copy(Path(b['base_model']) / name, target, row['sha256'])
                require(sha(target) == row['sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', name)
            torch.save({'format_version': 3, 'latent_channels': msdf.latent_channels, 'hidden_dim': msdf.hidden_dim,
                'block_channels': msdf.block_channels, 'context_scale': msdf.context_scale, 'max_injection': msdf.max_injection,
                'max_residual_ratio': msdf.max_residual_ratio, 'branch_dropout': msdf.branch_dropout,
                'pixel_support_weight': msdf.pixel_support_weight, 'morphology_alignment_enabled': True,
                'state_dict': {n: p.detach().cpu() for n, p in msdf.state_dict().items()}}, dest / 'msdf.pt')
            audit.update(status='COMPLETE', checkpoint_saved=True, checkpoint=tree_binding(dest, run),
                VAE_text_frozen_unchanged=True, frozen_component_hashes=before,
                final_grad_scaler_scale=scaler.get_scale(), peak_VRAM_bytes=torch.cuda.max_memory_allocated(),
                update_log_sha256=sha(run / 'logs' / ('msdf_' + cls + '_updates.jsonl')))
            msdf.detach()
    finally:
        audit['wall_seconds'] = time.monotonic() - started
        if audit['status'] != 'COMPLETE':
            audit['status'] = 'FAILED'
        save(path, audit)


def freeze_generators(run, raw=False):
    run = Path(run)
    classes = {}
    for cls in ('flash', 'black'):
        name = ('generator_training_' + cls) if raw else ('msdf_' + cls + '_training_audit')
        row = load(run / (name + '.json'))
        require(row['status'] == 'COMPLETE' and row['successful_updates'] == 2000, 'STAGE25_GENERATOR_BARRIER_FAILED', cls)
        verify_tree(run, row['checkpoint'])
        classes[cls] = {'successful_updates': 2000, 'checkpoint': row['checkpoint']}
    target = run / 'generator_checkpoint_binding.json'
    require(not target.exists(), 'STAGE25_OUTPUT_EXISTS', str(target))
    save(target, {'status': 'FROZEN', 'classes': classes, 'both_frozen_before_sampling': True})
    if not raw:
        save(run / 'msdf_checkpoint_binding.json', load(target))


def generate(run, bank, raw=False):
    run = Path(run)
    b, cfg = binding(run)
    frozen = load(run / 'generator_checkpoint_binding.json')
    require(frozen['status'] == 'FROZEN' and set(frozen['classes']) == {'flash', 'black'}, 'STAGE25_GENERATOR_BARRIER_FAILED', 'two frozen classes')
    checkpoints = {c: verify_tree(run, r['checkpoint']) for c, r in frozen['classes'].items()}
    claim(run, ('raw' if raw else 'msdf') + '_generate_' + bank)
    if bank != 'A':
        require(load(run / ('formal_bank_' + BANKS[BANKS.index(bank) - 1] + '_manifest.json'))['status'] == 'COMPLETE', 'STAGE25_BANK_SEQUENCE_INVALID', bank)
    p0 = Path(b['p0_view'])
    slots = load(p0 / 'stage20b_generation_slots.json')['slots']
    original = {s['slot_id']: s for s in load(Path(b['p0']) / 'stage20b_generation_slots.json')['slots']}
    rngs = {r['slot_id']: r['banks'] for r in load(p0 / 'stage20b_rng_manifest.json')['records']}
    g = load(p0 / 'stage20b_generation_protocol.json')
    root = run / ('synthetic_' + bank)
    root.mkdir()
    records = []
    manifest = run / ('formal_bank_' + bank + '_manifest.json')
    save(manifest, {'status': 'RUNNING', 'bank': bank, 'records': []})
    with guard(run) as torch:
        import numpy as np
        from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
        from diffusers.pipelines.stable_diffusion.msdf_guidance import load_msdf_adapter
        from qualify_plastic_bomo_stage20b_r0 import crop_and_mask
        from generate_plastic_bomo_stage20b_banks import sample
        from stage20b_formal_common import synthetic_label
        check_pipeline(StableDiffusionInpaintPipeline, DDIMScheduler)
        for cls in ('flash', 'black'):
            pipe = StableDiffusionInpaintPipeline.from_pretrained(str(checkpoints[cls]), local_files_only=True,
                use_safetensors=True, torch_dtype=torch.float16, safety_checker=None, feature_extractor=None,
                requires_safety_checker=False).to(b['device'])
            pipe.unet.eval(); pipe.vae.eval(); pipe.text_encoder.eval()
            pipe.disable_xformers_memory_efficient_attention()
            pipe.scheduler = DDIMScheduler.from_config(g['scheduler_config'])
            require(all(getattr(pipe.scheduler.config, k) == v for k, v in g['scheduler_config'].items()), 'STAGE25_FROZEN_BINDING_CHANGED', 'scheduler')
            msdf = None if raw else load_msdf_adapter(checkpoints[cls] / 'msdf.pt', pipe.unet, b['device'])
            for slot in [s for s in slots if s['class'] == cls]:
                rng = rngs[slot['slot_id']][bank]
                require(htext(canonical(original[slot['slot_id']])) == rng['non_rng_metadata_sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', 'original frozen slot')
                handle = None
                if msdf is not None:
                    donor, mask = crop_and_mask({**slot, 'path': slot['source_path'], 'sha256': slot['source_sha256']})
                    pixels = torch.from_numpy(np.asarray(donor).copy()).permute(2, 0, 1)[None].to(b['device']).float() / 127.5 - 1
                    mask_t = torch.from_numpy(np.asarray(mask).copy())[None, None].to(b['device']).float() / 255
                    with torch.inference_mode():
                        ref = pipe.vae.encode(pixels.half()).latent_dist.mode() * pipe.vae.config.scaling_factor
                    def prepare(module, args, kwargs):
                        inp = args[0] if args else kwargs['sample']
                        t = args[1] if len(args) > 1 else kwargs['timestep']
                        require(list(inp.shape)[1:] == [9, 64, 64] and inp.shape[0] == 2, 'STAGE25_MSDF_SHAPE_INVALID', 'inference input')
                        msdf.prepare(ref.float(), mask_t, inp[:1, 4:5].float(), t,
                            pipe.scheduler.config.num_train_timesteps, classifier_free_guidance=True, reference_pixels=pixels)
                    handle = pipe.unet.register_forward_pre_hook(prepare, with_kwargs=True)
                try:
                    image, trace = sample(pipe, slot, rng, g, b['device'])  # Exact Raw crop/RNG/paste implementation.
                finally:
                    if handle is not None:
                        handle.remove()
                image_path = root / (slot['slot_id'] + '.png')
                image.save(image_path, format='PNG')
                label_path = root / (slot['slot_id'] + '.txt')
                label_path.write_text(synthetic_label(slot), encoding='utf-8', newline='\n')
                records.append({'slot_id': slot['slot_id'], 'bank': bank, 'class': cls, 'role_id': 'extra_' + slot['slot_id'],
                    'frozen_slot': original[slot['slot_id']], 'runtime_slot': slot,
                    'image_path': image_path.relative_to(run).as_posix(), 'image_sha256': sha(image_path),
                    'label_path': label_path.relative_to(run).as_posix(), 'label_sha256': sha(label_path),
                    'checkpoint_tree_sha256': frozen['classes'][cls]['checkpoint']['tree_sha256'],
                    'quality_selection': False, 'formal_synthetic': True, 'MSDF_enabled': not raw, **trace})
                update(run, synthetic_generation_count=len(records) + BANKS.index(bank) * 80)
                save(manifest, {'status': 'RUNNING', 'bank': bank, 'records': records})
                print(f'{"Raw" if raw else "MSDF"} {bank}: {len(records)}/80 {slot["slot_id"]}', flush=True)
            if msdf is not None:
                msdf.detach()
            del pipe, msdf
            gc.collect(); torch.cuda.empty_cache()
    require(len({r['slot_id'] for r in records}) == len(records) == 80 and Counter(r['class'] for r in records) == {'flash': 40, 'black': 40},
            'STAGE25_SYNTHETIC_STRUCTURAL_FAILURE', 'all80 slots required')
    by_slot = {r['slot_id']: r for r in records}
    save(manifest, {'status': 'COMPLETE', 'bank': bank, 'records': [by_slot[s['slot_id']] for s in slots],
        'count': 80, 'quality_selection': False, 'reroll_count': 0, 'generation_protocol_sha256': sha(p0 / 'stage20b_generation_protocol.json')})
    if not raw:
        save(run / ('msdf_bank_' + bank + '_manifest.json'), load(manifest))


def validate_banks(run):
    from PIL import Image
    run = Path(run)
    records = []
    for bank in BANKS:
        row = load(run / ('formal_bank_' + bank + '_manifest.json'))
        require(row['status'] == 'COMPLETE' and row['count'] == 80, 'STAGE25_SYNTHETIC_STRUCTURAL_FAILURE', bank)
        for x in row['records']:
            require(sha(run / x['image_path']) == x['image_sha256'] and sha(run / x['label_path']) == x['label_sha256'], 'STAGE25_FROZEN_BINDING_CHANGED', 'generated asset')
            with Image.open(run / x['image_path']) as image:
                image.load()
                require(list(image.size) == x['frozen_slot']['geometry']['final_canvas_wh'], 'STAGE25_SYNTHETIC_STRUCTURAL_FAILURE', 'canvas')
            records.append(x)
    save(run / 'formal_synthetic_manifest.json', {'status': 'FROZEN', 'records': records, 'count': 240, 'quality_selection': False})
    save(run / 'formal_synthetic_validity.json', {'status': 'PASS', 'all_planned_outputs_retained': True, 'reroll': 0,
        'structural_checks_only': True, 'count': 240, 'manifest_sha256': sha(run / 'formal_synthetic_manifest.json')})
    if all(r['MSDF_enabled'] for r in records):
        save(run / 'msdf_generation_validity.json', load(run / 'formal_synthetic_validity.json'))
