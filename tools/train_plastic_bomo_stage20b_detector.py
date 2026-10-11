"""Four matched detector arms, using the R0 controller, not YOLO.train()."""
from collections import Counter
from copy import deepcopy
import gc
import math
from pathlib import Path
import time
from types import SimpleNamespace

from stage20b_formal_common import (canonical, child_main, context, htext, load, real_assets,
    require, runtime_guard, save, sha, update_status, verified_copy)
from stage20b_detector_controller import Stage20BDetectorController


def prepare_dataset(run, arm, p0, stage20a, synthetic):
    base = real_assets(stage20a, p0, 'train')
    require(len(base) == 136, 'FORMAL_FROZEN_BINDING_CHANGED', 'V2 TRAIN base136')
    if arm == 'RR':
        extras = [{'role_id': 'extra_' + r['slot_id'], 'image_path': r['full_image_path'],
                   'image_sha256': r['image_sha256'], 'label_path': r['full_labels_path'],
                   'label_sha256': r['label_sha256'], 'split': r['split']} for r in
                  load(p0 / 'stage20b_realrepeat_manifest.json')['records']]
        require(len({r['image_path'] for r in extras}) == 80, 'FORMAL_FROZEN_BINDING_CHANGED', 'RR unique full donors')
    else:
        extras = [{**r, 'image_path': str(run / r['image_path']), 'label_path': str(run / r['label_path']),
                   'split': 'train'} for r in synthetic['records'] if r['bank'] == arm]
    require(len(extras) == 80 and all(r['split'] == 'train' for r in base + extras),
            'FORMAL_FROZEN_BINDING_CHANGED', 'TRAIN-only216 roles')
    root = run / ('detector_' + arm) / 'train_inputs'
    require(not root.exists(), 'FORMAL_OUTPUT_EXISTS', str(root))
    root.mkdir(parents=True)
    roles = {}
    for row in base + extras:
        role = row['role_id']
        image = root / 'images/train' / (role + Path(row['image_path']).suffix)
        label = root / 'labels/train' / (role + '.txt')
        verified_copy(row['image_path'], image, row['image_sha256'])
        verified_copy(row['label_path'], label, row['label_sha256'])
        roles[role] = {**row, 'local_image': str(image), 'local_label': str(label)}
    require(len(roles) == 216, 'FORMAL_FROZEN_BINDING_CHANGED', 'duplicate roles')
    import yaml
    data = {'path': str(root), 'train': 'images/train', 'names': {0: 'flash', 1: 'black'}, 'nc': 2}
    (root / 'data.yaml').write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
    save(run / ('detector_' + arm) / 'train_role_binding.json', {'roles': roles, 'V2_TRAIN_only': True,
          'full_real_base': 136, 'extra_roles': 80, 'final_eval_reference_count': 0})
    return root, roles


def train(a):
    arm = a.arm
    b, p0, stage20a = context(a.run, ({'RR': 'SYNTHETIC_BANKS_VALIDATED', 'A': 'DETECTOR_RR_TRAINED',
                                      'B': 'DETECTOR_A_TRAINED', 'C': 'DETECTOR_B_TRAINED'}[arm],))
    a.failure_reason = 'DETECTOR_' + arm + '_TRAINING_FAILED'
    synthetic = load(a.run / 'formal_synthetic_manifest.json')
    require(synthetic['status'] == 'FROZEN' and load(a.run / 'formal_synthetic_validity.json')['status'] == 'PASS',
            a.failure_reason, 'synthetic freeze prerequisite')
    root, roles = prepare_dataset(a.run, arm, p0, stage20a, synthetic)
    cfg = load(Path(b['protocol']))
    frozen = load(p0 / 'stage20b_detector_protocol.json')['training']
    order = load(p0 / 'stage20b_training_order_spec.json')
    controller = Stage20BDetectorController(cfg, order)
    require(set(roles) == set(order['epochs'][0]['role_sequence']), a.failure_reason, 'frozen role IDs')
    audit = {'status': 'RUNNING', 'arm': arm, 'scheduled_draws': order['scheduled_draws'], 'actual_draws': 0,
             'scheduled_optimizer_attempts': order['scheduled_optimizer_attempts'], 'attempted_optimizer_updates': 0,
             'successful_optimizer_updates': 0, 'AMP_skipped_updates': 0, 'nonfinite_losses': 0,
             'gradient_nonfinite_count': 0, 'epochs': 0, 'epoch_records': [], 'official_final_eval_content_read': 0,
             'early_stopping_used': False, 'native_full_train_used': False, 'best_checkpoint_selection': False,
             'training_protocol_sha256': sha(p0 / 'stage20b_detector_protocol.json'),
             'order_spec_sha256': sha(p0 / 'stage20b_training_order_spec.json')}
    path = a.run / ('detector_arm_' + arm + '_training.json')
    started = time.monotonic()
    save(path, audit)
    try:
        with runtime_guard(a.run, b, p0, stage20a) as torch:
            from ultralytics import YOLO
            from ultralytics.models.yolo.detect import DetectionTrainer
            from ultralytics.data.dataset import YOLODataset
            from ultralytics.utils.torch_utils import ModelEMA
            from qualify_plastic_bomo_stage20b_r0 import draw_rng, parameter_hashes, tensor_sha
            require(sha(Path(b['detector_model'])) == b['P0_binding']['detector_sha256'],
                    'DETECTOR_INITIALIZATION_MISMATCH', 'pretrained bytes')
            torch.manual_seed(frozen['seed'])
            native = DetectionTrainer.__new__(DetectionTrainer)
            args = load(p0 / 'stage20b_detector_resolved_runtime_arguments.json')['arguments']
            args.update(data=str(root / 'data.yaml'), device=b['device'], project=str(a.run),
                        name='detector_' + arm, model=b['detector_model'])
            native.args = SimpleNamespace(**args)
            native.data = {'nc': 2, 'names': {0: 'flash', 1: 'black'}, 'channels': 3}
            native.device = torch.device(b['device'])
            original = YOLO(b['detector_model'])
            model = native.get_model(cfg=original.model.yaml, weights=original.model, verbose=False).to(b['device']).float()
            update_status(a.run, detector_training_count=('RR', 'A', 'B', 'C').index(arm) + 1)
            native.model = model
            native.set_model_attributes()
            for name, p in model.named_parameters():
                p.requires_grad_(not ('.dfl' in name))
            native.freeze_layer_names = ['.dfl']
            native.stride = max(32, int(model.stride.max()))
            if native.args.channels_last is True or native.args.channels_last is None:
                model.to(memory_format=torch.channels_last)
            initialization = {'pretrained_sha256': sha(Path(b['detector_model'])),
                'parameters': parameter_hashes(model), 'buffers': {n: tensor_sha(v) for n, v in model.named_buffers()},
                'head_seed': frozen['seed'], 'nc': model.model[-1].nc}
            require(initialization['parameters'] == load(Path(b['r0']) / 'detector_model_load_audit.json')['training_model_parameter_sha256'],
                    'DETECTOR_INITIALIZATION_MISMATCH', 'trained-head initialization differs from qualified R0')
            initfile = a.run / 'detector_initialization_comparison.json'
            if arm == 'RR':
                require(not initfile.exists(), 'DETECTOR_INITIALIZATION_MISMATCH', 'RR must initialize first')
                save(initfile, initialization)
            else:
                require(initialization == load(initfile), 'DETECTOR_INITIALIZATION_MISMATCH', arm + ' model/head parameters')
            require(all(p.dtype == torch.float32 for p in model.parameters()), a.failure_reason, 'FP32 master parameters')
            optimizer = native.build_optimizer(model, name=frozen['optimizer'], lr=frozen['lr0'],
                                               momentum=frozen['momentum'], decay=frozen['weight_decay'])
            for group in optimizer.param_groups:
                group['initial_lr'] = frozen['lr0']
            audit['optimizer_groups'] = [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups]
            scaler = torch.amp.GradScaler('cuda', enabled=frozen['amp'])
            actual_steps = [0]
            hook = optimizer.register_step_post_hook(lambda *args, **kwargs: actual_steps.__setitem__(0, actual_steps[0] + 1))
            # Preserve the hash-bound native trainer's EMA behavior and epoch150 last.pt
            # representation. No EMA selection, resync, NaN repair, best.pt or native save_model().
            ema = ModelEMA(model)
            audit['primary_model_representation'] = 'pinned native EMA at epoch150, not selected by validation'
            audit['EMA_update_rule'] = 'pinned native optimizer_step: update EMA after every scheduled scaler attempt, including skips'
            ds = YOLODataset(img_path=str(root / 'images/train'), imgsz=frozen['imgsz'], batch_size=frozen['batch'],
                augment=True, hyp=native.args, rect=frozen['rect'], cache=frozen['cache'], stride=native.stride,
                task=frozen['task'], data=native.data)
            lookup = {Path(f).stem: i for i, f in enumerate(ds.im_files)}
            require(len(ds) == 216 and set(lookup) == set(roles), a.failure_reason, 'native dataset dropped or changed roles')
            audit['native_training_transforms'] = repr(ds.transforms)
            native._model_train()
            optimizer.zero_grad(set_to_none=True)
            state = {'group_draws': 0, 'epoch_losses': [], 'observed': [], 'attempts': [], 'epoch': -1}
            with (a.run / 'logs' / ('detector_' + arm + '_draws.jsonl')).open('x', encoding='utf-8') as draws_log:
                def draw(epoch, pos, role, seed, length):
                    if pos == 0:
                        native._model_train()
                        state.update(epoch=epoch, epoch_losses=[], observed=[], attempts=[])
                    for group, lr in zip(optimizer.param_groups, controller.learning_rates(epoch, pos, optimizer.param_groups)):
                        group['lr'] = lr
                    with draw_rng(seed, ds.transforms):
                        sample = ds[lookup[role]]
                        require(Path(sample['im_file']).stem == role, a.failure_reason, 'actual sample role mismatch')
                        batch = native.preprocess_batch(ds.collate_fn([sample]))
                        with torch.autocast('cuda', dtype=torch.float16, enabled=frozen['amp']):
                            loss, items = model(batch)
                            objective = loss.sum()
                        finite = bool(torch.isfinite(objective))
                        audit['actual_draws'] += 1
                        audit['nonfinite_losses'] += int(not finite)
                        require(finite and isinstance(items, dict), a.failure_reason, 'nonfinite native loss; no rescue')
                        controller.backward(objective, length, scaler=scaler)
                    state['group_draws'] += 1
                    state['epoch_losses'].append(float(objective.detach()))
                    state['observed'].append(role)
                    draws_log.write(canonical({'epoch': epoch, 'position': pos, 'role': role,
                        'augmentation_seed': seed, 'loss': float(objective.detach()), 'normalizer': length,
                        'loss_items': {k: float(v.detach()) for k, v in items.items()}}) + '\n')

                def attempt(epoch, pos, length):
                    require(state['group_draws'] == length, a.failure_reason, 'actual accumulation draws')
                    audit['attempted_optimizer_updates'] += 1
                    scaler.unscale_(optimizer)
                    grad_finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                    audit['gradient_nonfinite_count'] += int(not grad_finite)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), frozen['max_grad_norm'])
                    require(not grad_finite or bool(torch.isfinite(norm)), a.failure_reason, 'invalid clipping norm')
                    before, scale0 = actual_steps[0], scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    success = actual_steps[0] == before + 1
                    audit['successful_optimizer_updates'] += int(success)
                    audit['AMP_skipped_updates'] += int(not success)
                    require(math.isfinite(scaler.get_scale()) and scaler.get_scale() > 0, a.failure_reason, 'GradScaler invalid')
                    optimizer.zero_grad(set_to_none=True)
                    ema.update(model)  # Native EMA schedule, without reading any validation.
                    state['attempts'].append({'position': pos, 'draws': length, 'normalizer': length,
                        'optimizer_step_executed': success, 'AMP_skip': not success,
                        'gradient_norm': float(norm) if bool(torch.isfinite(norm)) else None,
                        'scale_before': scale0, 'scale_after': scaler.get_scale(),
                        'learning_rates': [g['lr'] for g in optimizer.param_groups]})
                    state['group_draws'] = 0
                    update_status(a.run, **{f'detector_{arm}_successful_updates': audit['successful_optimizer_updates']})
                    save(path, audit)

                def epoch_end(epoch):
                    require(state['observed'] == order['epochs'][epoch]['role_sequence'], a.failure_reason, 'role order')
                    audit['epochs'] += 1
                    audit['epoch_records'].append({'epoch_0based': epoch, 'draws': len(state['observed']),
                        'actual_role_sequence_sha256': htext(canonical(state['observed'])),
                        'expected_role_sequence_sha256': order['epochs'][epoch]['sequence_sha256'],
                        'loss_mean': sum(state['epoch_losses']) / len(state['epoch_losses']),
                        'attempts': list(state['attempts'])})
                    draws_log.flush()
                    save(path, audit)
                    print(f"detector {arm}: epoch={epoch+1}/{frozen['epochs']} draws={audit['actual_draws']} successful={audit['successful_optimizer_updates']} skips={audit['AMP_skipped_updates']}", flush=True)

                totals = controller.run(draw, attempt, epoch_end=epoch_end,
                                        cleanup=lambda: optimizer.zero_grad(set_to_none=True))
            hook.remove()
            require(audit['actual_draws'] == totals['draws'] == order['scheduled_draws']
                    and audit['attempted_optimizer_updates'] == totals['scheduled_optimizer_attempts'] == order['scheduled_optimizer_attempts']
                    and audit['epochs'] == frozen['epochs'], a.failure_reason, 'actual budget mismatch')
            require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
                    and all(bool(torch.isfinite(v).all()) for v in ema.ema.state_dict().values()),
                    a.failure_reason, 'final model/EMA nonfinite; no resync')
            ema.update_attr(model, include=('yaml', 'nc', 'args', 'names', 'stride', 'class_weights'))
            snapshot = deepcopy(ema.ema).cpu().half().to(memory_format=torch.contiguous_format)
            if hasattr(snapshot, 'criterion'):
                snapshot.criterion = None
            require(all(bool(torch.isfinite(v).all()) for v in snapshot.state_dict().values()),
                    a.failure_reason, 'FP16 checkpoint serialization overflow; no clamp')
            dest = a.run / ('detector_' + arm) / 'last.pt'
            require(not dest.exists(), a.failure_reason, 'last.pt already exists')
            torch.save({'epoch': frozen['epochs'] - 1, 'model': None, 'ema': snapshot,
                        'updates': ema.updates, 'train_args': args, 'optimizer': None,
                        'version': __import__('ultralytics').__version__, 'stage20b_arm': arm,
                        'primary_checkpoint': 'epoch150 last.pt', 'validation_selection': False}, dest)
            audit.update(status='COMPLETE', last_checkpoint=dest.relative_to(a.run).as_posix(),
                last_sha256=sha(dest), EMA_updates=ema.updates, final_grad_scaler_scale=scaler.get_scale(),
                initial_parameter_binding_sha256=htext(canonical(initialization)),
                per_draw_log_sha256=sha(a.run / 'logs' / ('detector_' + arm + '_draws.jsonl')),
                wall_seconds=time.monotonic() - started, peak_vram_bytes=torch.cuda.max_memory_allocated())
            save(path, audit)
            del optimizer, model, original, native, ema, snapshot, ds
            gc.collect()
            torch.cuda.empty_cache()
    finally:
        audit['wall_seconds'] = time.monotonic() - started
        if audit['status'] != 'COMPLETE':
            audit['status'] = 'FAILED'
        save(path, audit)


if __name__ == '__main__':
    child_main(train, __doc__, lambda p: p.add_argument('--arm', choices=('RR', 'A', 'B', 'C'), required=True))
