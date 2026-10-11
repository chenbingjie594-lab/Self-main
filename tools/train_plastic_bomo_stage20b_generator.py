"""One V2 class, exactly 2000 successful updates; no legacy attempted-step main."""
import gc
import math
from pathlib import Path
import time

from stage20b_formal_common import (child_main, context, htext, canonical, load, require,
    runtime_guard, save, sha, tree_binding, update_status, verified_copy, verify_tree)


def train(a):
    expected = 'INIT' if a.class_name == 'flash' else 'FLASH_GENERATOR_TRAINED'
    b, p0, stage20a = context(a.run, (expected,))
    cls = a.class_name
    if cls == 'black':
        previous = load(a.run / 'generator_training_flash.json')
        require(previous['status'] == 'COMPLETE' and previous['successful_updates'] == load(p0 / 'stage20b_generator_training_protocol.json')['max_train_steps'],
                'FLASH_GENERATOR_TRAINING_FAILED', 'Flash must complete and freeze before Black')
        verify_tree(a.run, previous['checkpoint'])
    a.failure_reason = cls.upper() + '_GENERATOR_TRAINING_FAILED'
    cfg = load(p0 / 'stage20b_generator_training_protocol.json')
    rows = sorted((x for x in load(p0 / 'generator_train_allowlist.json')['inputs'] if x['class'] == cls),
                  key=lambda x: x['annotation_id'])
    require(len(rows) == {'flash': 75, 'black': 97}[cls] and all(x['split'] == 'train' for x in rows),
            a.failure_reason, 'TRAIN allowlist count')
    audit = {'status': 'RUNNING', 'class': cls, 'started_at_epoch': 0, 'attempted_updates': 0,
             'successful_updates': 0, 'AMP_skips': 0, 'nonfinite_losses': 0,
             'gradient_nonfinite_count': 0, 'microbatch_draws': 0, 'epochs_completed': 0,
             'last_step_legal': False, 'checkpoint_saved': False, 'protocol_sha256': sha(p0 / 'stage20b_generator_training_protocol.json'),
             'input_annotation_ids': [x['annotation_id'] for x in rows], 'TRAIN_instances': len(rows),
             'automatic_resume': False, 'official_final_eval_content_read': 0}
    path = a.run / ('generator_training_' + cls + '.json')
    save(path, audit)
    started = time.monotonic()
    try:
        with runtime_guard(a.run, b, p0, stage20a) as torch:
            import numpy as np
            import torch.nn.functional as F
            from diffusers import StableDiffusionInpaintPipeline, DDPMScheduler
            from qualify_plastic_bomo_stage20b_r0 import crop_and_mask, parameter_hashes
            # Same R0 model construction and exact public components; masters are NOT FP16.
            pipe = StableDiffusionInpaintPipeline.from_pretrained(b['base_model'], local_files_only=True,
                use_safetensors=True, torch_dtype=torch.float32, safety_checker=None, feature_extractor=None,
                requires_safety_checker=False).to(b['device'])
            pipe.vae.requires_grad_(False).eval().to(dtype=torch.float16)
            pipe.text_encoder.requires_grad_(False).eval().to(dtype=torch.float16)
            pipe.unet.requires_grad_(True).train().to(dtype=torch.float32)
            pipe.unet.enable_gradient_checkpointing()
            pipe.disable_xformers_memory_efficient_attention()
            update_status(a.run, generator_training_count=1 if cls == 'flash' else 2)
            require(all(p.dtype == torch.float32 and p.requires_grad for p in pipe.unet.parameters()),
                    a.failure_reason, 'UNet FP32 trainable master contract')
            frozen_before = {n: parameter_hashes(m) for n, m in (('vae', pipe.vae), ('text_encoder', pipe.text_encoder))}
            noise_scheduler = DDPMScheduler.from_pretrained(b['base_model'], subfolder='scheduler', local_files_only=True)
            require(noise_scheduler.config.prediction_type in ('epsilon', 'v_prediction'), a.failure_reason, 'prediction target')
            opt = torch.optim.AdamW(pipe.unet.parameters(), lr=cfg['learning_rate'],
                betas=(cfg['adam_beta1'], cfg['adam_beta2']), weight_decay=cfg['adam_weight_decay'], eps=cfg['adam_epsilon'])
            scaler = torch.amp.GradScaler('cuda', enabled=cfg['mixed_precision'] == 'fp16')
            actual = [0]
            hook = opt.register_step_post_hook(lambda *args, **kwargs: actual.__setitem__(0, actual[0] + 1))
            ids = pipe.tokenizer(cfg['prompts'][cls], padding=False, truncation=True,
                max_length=pipe.tokenizer.model_max_length, return_tensors='pt').input_ids.to(b['device'])
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                text = pipe.text_encoder(ids)[0]
            epoch = 0
            history = a.run / 'logs' / ('generator_' + cls + '_updates.jsonl')
            with history.open('x', encoding='utf-8') as log:
                while audit['successful_updates'] < cfg['max_train_steps']:
                    indices = torch.randperm(len(rows), generator=torch.Generator(device='cpu').manual_seed(cfg['seed'] + epoch)).tolist()
                    audit.setdefault('epoch_order_sha256', []).append(htext(canonical([rows[i]['annotation_id'] for i in indices])))
                    for offset in range(0, len(indices), cfg['gradient_accumulation_steps']):
                        opt.zero_grad(set_to_none=True)
                        group = indices[offset:offset + cfg['gradient_accumulation_steps']]
                        losses = []
                        for i in group:
                            image, mask = crop_and_mask(rows[i])
                            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].to(b['device']).float() / 127.5 - 1
                            mask_t = torch.from_numpy(np.asarray(mask).copy())[None, None].to(b['device']).float() / 255
                            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                                latents = pipe.vae.encode(pixels.half()).latent_dist.sample() * pipe.vae.config.scaling_factor
                                masked = pipe.vae.encode((pixels * (mask_t < .5)).half()).latent_dist.sample() * pipe.vae.config.scaling_factor
                            noise = torch.randn_like(latents)
                            t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=b['device']).long()
                            inp = torch.cat([noise_scheduler.add_noise(latents, noise, t), F.interpolate(mask_t, latents.shape[-2:]), masked], dim=1)
                            with torch.autocast('cuda', dtype=torch.float16):
                                pred = pipe.unet(inp, t, encoder_hidden_states=text).sample
                            target = noise if noise_scheduler.config.prediction_type == 'epsilon' else noise_scheduler.get_velocity(latents, noise, t)
                            loss = F.mse_loss(pred.float(), target.float(), reduction='mean')
                            audit['microbatch_draws'] += 1
                            finite = bool(torch.isfinite(loss))
                            audit['nonfinite_losses'] += int(not finite)
                            require(finite, a.failure_reason, 'nonfinite denoising loss; no rescue')
                            losses.append(float(loss.detach()))
                            # Frozen legacy generator accumulation divisor4, including epoch partial groups.
                            # The detector's separately frozen tail24 rule is NOT applied to generator training.
                            scaler.scale(loss / cfg['gradient_accumulation_steps']).backward()
                            del pixels, mask_t, latents, masked, noise, t, inp, pred, target, loss
                        audit['attempted_updates'] += 1
                        scaler.unscale_(opt)
                        grad_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in pipe.unet.parameters())
                        audit['gradient_nonfinite_count'] += int(not grad_finite)
                        norm = torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), cfg['max_grad_norm'])
                        require(not grad_finite or bool(torch.isfinite(norm)), a.failure_reason, 'finite gradients but invalid clipping norm')
                        before, scale_before = actual[0], scaler.get_scale()
                        scaler.step(opt)
                        scaler.update()
                        success = actual[0] == before + 1
                        audit['successful_updates'] += int(success)
                        audit['AMP_skips'] += int(not success)
                        audit['last_step_legal'] = success and grad_finite and bool(torch.isfinite(norm))
                        require(math.isfinite(scaler.get_scale()) and scaler.get_scale() > 0, a.failure_reason, 'GradScaler invalid')
                        row = {'epoch': epoch, 'draws': len(group), 'loss_normalizer': cfg['gradient_accumulation_steps'],
                               'attempt': audit['attempted_updates'], 'successful_updates': audit['successful_updates'],
                               'AMP_skip': not success, 'loss_mean': sum(losses) / len(losses),
                               'gradient_norm': float(norm) if bool(torch.isfinite(norm)) else None,
                               'scale_before': scale_before, 'scale_after': scaler.get_scale()}
                        log.write(canonical(row) + '\n')
                        log.flush()
                        update_status(a.run, **{f'generator_{cls}_successful_updates': audit['successful_updates']})
                        save(path, audit)
                        if audit['attempted_updates'] % 20 == 0 or audit['successful_updates'] == cfg['max_train_steps']:
                            print(f"generator {cls}: successful={audit['successful_updates']}/{cfg['max_train_steps']} attempts={audit['attempted_updates']} skips={audit['AMP_skips']}", flush=True)
                        if audit['successful_updates'] == cfg['max_train_steps']:
                            break
                    epoch += 1
                    audit['epochs_completed'] = epoch if offset + len(group) == len(indices) else epoch - 1
            hook.remove()
            require(audit['successful_updates'] == cfg['max_train_steps'] and audit['last_step_legal'], a.failure_reason, audit)
            require(all(bool(torch.isfinite(p).all()) for p in pipe.unet.parameters()), a.failure_reason, 'final parameter nonfinite')
            frozen_after = {n: parameter_hashes(m) for n, m in (('vae', pipe.vae), ('text_encoder', pipe.text_encoder))}
            require(frozen_before == frozen_after, a.failure_reason, 'frozen model parameters changed')
            audit['frozen_component_parameter_sha256'] = frozen_after
            audit['frozen_components_unchanged'] = True
            dest = a.run / 'generator_checkpoints' / cls / 'final_step2000'
            require(not dest.exists(), a.failure_reason, 'no overwrite checkpoint')
            pipe.unet.save_pretrained(str(dest / 'unet'), safe_serialization=True)
            # Preserve exact public FP32 VAE/text files, tokenizer and configs, rather than
            # serializing a rounded fp16 inference copy as a supposedly unchanged component.
            for name, row in load(p0 / 'sd2_base_weight_binding.json')['files'].items():
                if name == 'unet/diffusion_pytorch_model.safetensors':
                    continue
                target_path = dest / name
                if target_path.exists():  # save_pretrained's generated UNet config only
                    require(name == 'unet/config.json', a.failure_reason, name)
                    target_path.write_bytes((Path(b['base_model']) / name).read_bytes())
                    require(sha(target_path) == row['sha256'], a.failure_reason, name)
                else:
                    verified_copy(Path(b['base_model']) / name, target_path, row['sha256'])
            audit.update(status='COMPLETE', checkpoint_saved=True, checkpoint=tree_binding(dest, a.run),
                         checkpoint_role=cls + '_V2_final_step2000', final_grad_scaler_scale=scaler.get_scale(),
                         wall_seconds=time.monotonic() - started, unet_master_dtype='float32',
                         training_loss_normalizer=cfg['gradient_accumulation_steps'], precision='fp16 autocast',
                         peak_vram_bytes=torch.cuda.max_memory_allocated())
            save(path, audit)
            del opt, pipe, text, ids
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        audit['wall_seconds'] = time.monotonic() - started
        if audit['status'] != 'COMPLETE':
            audit['status'] = 'FAILED'
        save(path, audit)


if __name__ == '__main__':
    child_main(train, __doc__, lambda p: p.add_argument('--class_name', choices=('flash', 'black'), required=True))
