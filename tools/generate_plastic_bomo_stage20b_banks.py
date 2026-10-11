"""Frozen trained checkpoint sanity, then raw A/B/C (no scoring or reroll)."""
from collections import Counter
import gc
import hashlib

from stage20b_formal_common import (canonical, child_main, context, htext, load, require,
    runtime_guard, save, sha, synthetic_label, validate_bbox, verify_tree, update_status)


def sample(pipe, slot, rng, cfg, device):
    import numpy as np
    import torch
    from PIL import Image
    from qualify_plastic_bomo_stage20b_r0 import crop_and_mask, tensor_sha
    validate_bbox(slot)
    crop, mask = crop_and_mask(slot, use_normal=True)
    gen = torch.Generator(device='cpu').manual_seed(rng['sample_seed'])
    before = tensor_sha(gen.get_state())
    with torch.inference_mode():
        out = pipe(prompt=slot['prompt'], image=crop, mask_image=mask,
            num_inference_steps=cfg['num_inference_steps'], guidance_scale=cfg['guidance_scale'],
            eta=cfg['eta'], strength=cfg['strength'], height=cfg['height'], width=cfg['width'],
            generator=gen, negative_prompt=cfg['negative_prompt'], num_images_per_prompt=cfg['num_images_per_prompt'],
            output_type='np', clip_skip=cfg['clip_skip'], padding_mask_crop=cfg['padding_mask_crop'])
    array = np.asarray(out.images)
    require(array.shape == (1, cfg['height'], cfg['width'], 3) and bool(np.isfinite(array).all()),
            'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', slot['slot_id'] + ': nonfinite/invalid pipeline output')
    rgb = (array[0] * 255).round().clip(0, 255).astype(np.uint8)
    patch = Image.fromarray(rgb, 'RGB')
    with Image.open(slot['normal_path']) as im:
        full = im.convert('RGB')
    x, y, _, _ = slot['geometry']['fixed_crop_xyxy']
    l, t, r, bot = slot['geometry']['paste_valid_crop_region_xyxy']
    full.paste(patch.crop((l - x, t - y, r - x, bot - y)), (l, t))
    require(list(full.size) == slot['geometry']['final_canvas_wh'], 'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE', 'paste size')
    return full, {'sample_seed': rng['sample_seed'], 'bank_seed': rng['bank_seed'],
                  'RNG_before_sha256': before, 'RNG_after_sha256': tensor_sha(gen.get_state()),
                  'patch_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest(),
                  'decoded_RGB_sha256': hashlib.sha256(full.tobytes()).hexdigest(), 'finite_pipeline_pixels': True}


def generate(a):
    expected = {'sanity': 'GENERATOR_CHECKPOINTS_FROZEN', 'A': 'GENERATOR_CHECKPOINTS_FROZEN',
                'B': 'BANK_A_GENERATED', 'C': 'BANK_B_GENERATED'}[a.bank]
    b, p0, stage20a = context(a.run, (expected,))
    a.failure_reason = 'TRAINED_CHECKPOINT_RUNTIME_NONDETERMINISTIC' if a.bank == 'sanity' else 'FORMAL_SYNTHETIC_STRUCTURAL_FAILURE'
    checkpoints = load(a.run / 'generator_checkpoint_binding.json')
    require(checkpoints['status'] == 'FROZEN' and set(checkpoints['classes']) == {'flash', 'black'},
            'FORMAL_FROZEN_BINDING_CHANGED', 'two-class checkpoint barrier')
    paths = {c: verify_tree(a.run, row['checkpoint']) for c, row in checkpoints['classes'].items()}
    if a.bank != 'sanity':
        require(load(a.run / 'trained_checkpoint_determinism_audit.json')['status'] == 'PASS',
                a.failure_reason, 'sanity prerequisite')
    slots = load(p0 / 'stage20b_generation_slots.json')['slots']
    rngs = {r['slot_id']: r['banks'] for r in load(p0 / 'stage20b_rng_manifest.json')['records']}
    cfg = load(p0 / 'stage20b_generation_protocol.json')
    records = []
    root = a.run / ('checkpoint_sanity' if a.bank == 'sanity' else 'synthetic_' + a.bank)
    require(not root.exists(), a.failure_reason, 'no overwrites or sampling retries')
    root.mkdir()
    manifest = a.run / ('trained_checkpoint_determinism_audit.json' if a.bank == 'sanity' else 'formal_bank_' + a.bank + '_manifest.json')
    save(manifest, {'status': 'RUNNING', 'records': records})
    with runtime_guard(a.run, b, p0, stage20a) as torch:
        from diffusers import StableDiffusionInpaintPipeline, DDIMScheduler
        for cls in ('flash', 'black'):
            pipe = StableDiffusionInpaintPipeline.from_pretrained(str(paths[cls]), local_files_only=True,
                use_safetensors=True, torch_dtype=torch.float16, safety_checker=None, feature_extractor=None,
                requires_safety_checker=False).to(b['device'])
            pipe.unet.eval()
            pipe.vae.eval()
            pipe.text_encoder.eval()
            pipe.disable_xformers_memory_efficient_attention()
            pipe.scheduler = DDIMScheduler.from_config(cfg['scheduler_config'])
            require(all(getattr(pipe.scheduler.config, k) == v for k, v in cfg['scheduler_config'].items()),
                    'FORMAL_FROZEN_BINDING_CHANGED', 'DDIM config')
            selected = [s for s in slots if s['class'] == cls]
            if a.bank == 'sanity':
                selected = selected[:1]  # first frozen slot, never selected on image quality
            for slot in selected:
                bank = 'A' if a.bank == 'sanity' else a.bank
                rng = rngs[slot['slot_id']][bank]
                require(htext(canonical(slot)) == rng['non_rng_metadata_sha256'], 'FORMAL_FROZEN_BINDING_CHANGED', 'slot metadata')
                runs = []
                for repeat in range(2 if a.bank == 'sanity' else 1):
                    print(f"{a.bank} {slot['slot_id']} repeat={repeat}", flush=True)
                    image, trace = sample(pipe, slot, rng, cfg, b['device'])
                    name = slot['slot_id'] + (f'_repeat{repeat}_FORMAL_CHECKPOINT_SANITY_ONLY' if a.bank == 'sanity' else '')
                    image_path = root / (name + '.png')
                    image.save(image_path, format='PNG')
                    trace.update(image_path=image_path.relative_to(a.run).as_posix(), image_sha256=sha(image_path))
                    if a.bank == 'sanity':
                        runs.append(trace)
                    else:
                        label_path = root / (name + '.txt')
                        label_path.write_text(synthetic_label(slot), encoding='utf-8', newline='\n')
                        records.append({'slot_id': slot['slot_id'], 'bank': bank, 'class': cls,
                            'role_id': 'extra_' + slot['slot_id'], 'frozen_slot': slot,
                            'non_rng_metadata_sha256': rng['non_rng_metadata_sha256'],
                            'checkpoint_tree_sha256': checkpoints['classes'][cls]['checkpoint']['tree_sha256'],
                            'label_path': label_path.relative_to(a.run).as_posix(), 'label_sha256': sha(label_path),
                            'quality_selection': False, 'formal_synthetic': True, **trace})
                        count = len(records) + {'A': 0, 'B': len(slots), 'C': 2 * len(slots)}[bank]
                        update_status(a.run, formal_synthetic_generated=count, synthetic_generation_count=count)
                if a.bank == 'sanity':
                    identical = runs[0]['decoded_RGB_sha256'] == runs[1]['decoded_RGB_sha256']
                    records.append({'slot_id': slot['slot_id'], 'class': cls, 'runs': runs,
                                    'same_seed_RGB_identical': identical, 'role': 'FORMAL_CHECKPOINT_SANITY_ONLY',
                                    'detector_input_allowed': False})
                    save(manifest, {'status': 'RUNNING', 'records': records})
                    require(identical, 'TRAINED_CHECKPOINT_RUNTIME_NONDETERMINISTIC', slot['slot_id'])
                else:
                    save(manifest, {'status': 'RUNNING', 'records': records, 'bank': bank})
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
    if a.bank == 'sanity':
        require(len(records) == 2, a.failure_reason, 'sanity classes')
        save(manifest, {'status': 'PASS', 'records': records, 'sampling_count': 4,
                        'formal_synthetic_count': 0, 'quality_selection': False})
        update_status(a.run, checkpoint_sanity_sampling_count=4)
    else:
        by_slot = {x['slot_id']: x for x in records}
        require(len(by_slot) == len(records) == len(slots)
                and Counter(x['class'] for x in records) == {'flash': 40, 'black': 40},
                a.failure_reason, 'all frozen80 slots required')
        save(manifest, {'status': 'COMPLETE', 'bank': a.bank, 'records': [by_slot[s['slot_id']] for s in slots],
                        'generation_protocol_sha256': sha(p0 / 'stage20b_generation_protocol.json'),
                        'quality_selection': False, 'reroll_count': 0, 'count': len(records)})


if __name__ == '__main__':
    child_main(generate, __doc__, lambda p: p.add_argument('--bank', choices=('sanity', 'A', 'B', 'C'), required=True))
