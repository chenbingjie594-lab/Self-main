"""A single unified final_eval round, only after all four last.pt byte freezes."""
import gc
import math
from pathlib import Path

from stage20b_formal_common import (child_main, context, load, now, real_assets, require,
    runtime_guard, save, sha, update_status, verified_copy)


def metrics_record(metric):
    box = metric.box
    per_class = {}
    indices = [int(x) for x in box.ap_class_index]
    for i, cls in enumerate(('flash', 'black')):
        if i in indices:
            j = indices.index(i)
            per_class[cls] = {'precision': float(box.p[j]), 'recall': float(box.r[j]),
                              'ap50': float(box.ap50[j]), 'ap5095': float(box.ap[j])}
        else:
            # Native no-detections case, not an imputed missing metric or absent GT class.
            require(not indices and len(box.all_ap) == 0, 'FORMAL_METRICS_INVALID', 'partial class index')
            per_class[cls] = dict.fromkeys(('precision', 'recall', 'ap50', 'ap5095'), 0.0)
    result = {'status': 'COMPLETE', 'map5095': float(box.map), 'map50': float(box.map50),
              'precision': float(box.mp), 'recall': float(box.mr), 'per_class': per_class}
    values = [result[k] for k in ('map5095', 'map50', 'precision', 'recall')]
    values += [v for row in per_class.values() for v in row.values()]
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in values), 'FORMAL_METRICS_INVALID', result)
    return result


def evaluate(a):
    b, p0, stage20a = context(a.run, ('ALL_DETECTORS_FROZEN',))
    a.failure_reason = 'FORMAL_FINAL_EVAL_FAILED'
    binding = load(a.run / 'detector_checkpoint_binding.json')
    require(binding['status'] == 'FROZEN' and set(binding['arms']) == {'RR', 'A', 'B', 'C'},
            'FORMAL_FINAL_EVAL_BARRIER_FAILED', '4/4 frozen checkpoints required')
    for row in binding['arms'].values():
        require(row['epochs'] == 150 and sha(a.run / row['path']) == row['sha256'],
                'FORMAL_FROZEN_BINDING_CHANGED', row)
    claim = a.run / 'logs' / 'final_eval_one_round_claim.json'
    require(not claim.exists(), a.failure_reason, 'one round only; no retry')
    save(claim, {'started_at': now(), 'checkpoint_binding_sha256': sha(a.run / 'detector_checkpoint_binding.json')})
    audit = {'status': 'RUNNING', 'unified_rounds_started': 1, 'unified_rounds_completed': 0,
             'first_authorized_content_access_at': now(), 'state_at_first_access': 'ALL_DETECTORS_FROZEN',
             'checkpoint_binding_sha256': sha(a.run / 'detector_checkpoint_binding.json'),
             'source_assets_copied_and_SHA_verified': 0, 'source_images': 0, 'source_labels': 0,
             'official_final_eval_image_forwards': 0, 'arms': {}, 'warmup_uses_final_eval_images': False,
             'training_final_eval_content_reads': 0}
    save(a.run / 'final_eval_access_audit.json', audit)
    update_status(a.run, official_final_eval_rounds=1)
    rows = real_assets(stage20a, p0, 'final_eval')
    require(len(rows) == 48, a.failure_reason, 'frozen V2 final_eval48')
    root = a.run / 'final_eval_inputs'
    root.mkdir()
    try:
        with runtime_guard(a.run, b, p0, stage20a, evaluation=True) as torch:
            import yaml
            from ultralytics import YOLO
            from ultralytics.models.yolo.detect import DetectionValidator
            for row in rows:
                name = row['node_id']
                verified_copy(row['image_path'], root / 'images/final_eval' / (name + Path(row['image_path']).suffix), row['image_sha256'])
                audit['source_images'] += 1
                verified_copy(row['label_path'], root / 'labels/final_eval' / (name + '.txt'), row['label_sha256'])
                audit['source_labels'] += 1
                audit['source_assets_copied_and_SHA_verified'] += 2
                save(a.run / 'final_eval_access_audit.json', audit)
            valdir = str((root / 'images/final_eval').resolve())
            # Native YAML requires train+val keys. They are unused aliases here, created
            # only AFTER the barrier; split remains the frozen literal final_eval.
            data = {'path': str(root), 'train': None, 'val': valdir, 'final_eval': valdir,
                    'names': {0: 'flash', 1: 'black'}, 'nc': 2, 'channels': 3}
            yaml_path = root / 'data.yaml'
            yaml_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding='utf-8')
            expected = {x['node_id'] for x in rows}
            eval_cfg = load(p0 / 'stage20b_detector_protocol.json')['evaluation']
            native_cfg = load(p0 / 'stage20b_detector_resolved_runtime_arguments.json')['arguments']
            val_args = {**native_cfg, **{k: v for k, v in eval_cfg.items() if k != 'metrics'}}
            val_args.update(data=str(yaml_path), device=b['device'], workers=native_cfg['workers'],
                            project=str(a.run / 'final_eval_outputs'), mode='val', task='detect', exist_ok=False)
            for arm in ('RR', 'A', 'B', 'C'):
                observed, gt_classes = [], []

                class AuditedValidator(DetectionValidator):
                    def update_metrics(self, preds, batch):
                        names = [Path(f).stem for f in batch['im_file']]
                        require(all(n in expected for n in names), 'FORMAL_FINAL_EVAL_FAILED', 'unexpected evaluation input')
                        observed.extend(names)
                        gt_classes.extend(int(v) for v in batch['cls'].reshape(-1).cpu().tolist())
                        audit['official_final_eval_image_forwards'] += len(names)
                        audit['arms'][arm] = {'image_forwards': len(observed), 'node_ids': list(observed)}
                        save(a.run / 'final_eval_access_audit.json', audit)
                        super().update_metrics(preds, batch)

                checkpoint = a.run / binding['arms'][arm]['path']
                require(sha(checkpoint) == binding['arms'][arm]['sha256'], 'FORMAL_FROZEN_BINDING_CHANGED', arm)
                model = YOLO(str(checkpoint))
                metric = model.val(validator=AuditedValidator, **{**val_args, 'name': arm, 'model': str(checkpoint)})
                require(len(observed) == len(set(observed)) == 48 and set(observed) == expected
                        and gt_classes.count(0) == 33 and gt_classes.count(1) == 27,
                        a.failure_reason, 'evaluation image/annotation population mismatch')
                record = metrics_record(metric)
                record.update(arm=arm, checkpoint_sha256=binding['arms'][arm]['sha256'],
                              evaluation_protocol_sha256=sha(p0 / 'stage20b_detector_protocol.json'),
                              evaluation_image_count=48, target_counts={'flash': 33, 'black': 27},
                              primary_checkpoint='epoch150 last.pt')
                save(a.run / (('RR' if arm == 'RR' else 'Raw' + arm) + '_metrics.json'), record)
                del model, metric
                gc.collect()
                torch.cuda.empty_cache()
            audit.update(status='COMPLETE', unified_rounds_completed=1, completed_at=now(),
                         final_eval_population='frozen V2 metadata48; not historical physical val directory')
            require(audit['official_final_eval_image_forwards'] == 48 * 4, a.failure_reason, 'one four-arm unified round')
            save(a.run / 'final_eval_access_audit.json', audit)
    finally:
        if audit['status'] != 'COMPLETE':
            audit['status'] = 'FAILED_PARTIAL_NO_RETRY'
        save(a.run / 'final_eval_access_audit.json', audit)


if __name__ == '__main__':
    child_main(evaluate, __doc__)
