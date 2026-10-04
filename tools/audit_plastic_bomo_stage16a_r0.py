"""R0 annotation entry gate. Fail closed before any runtime/teacher/generation work.

No detector or generator imports. This entry-gate tool is not an implementation
of the later differentiability or generation stages. They remain NOT_RUN when
the provenance gate stops. An assertion that a dataset is real is not evidence
that its annotation rows were human produced or verified.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load(path):
    def reject(value):
        raise ValueError('Non-finite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def image_info(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        h = hashlib.sha256(json.dumps(list(im.size)).encode())
        h.update(im.tobytes())
        return {'image_sha256': sha(path), 'decoded_rgb_sha256': h.hexdigest(), 'image_size': list(im.size)}


def project_bbox(box, size):
    """Match torchvision Resize(int) dimensions and CenterCrop round offsets."""
    w, h = size
    if w <= 0 or h <= 0 or len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError('invalid bbox/image geometry')
    rw, rh = (512, int(512*h/w)) if w <= h else (int(512*w/h), 512)
    sx, sy = rw/w, rh/h
    ox, oy = int(round((rw-512)/2)), int(round((rh-512)/2))
    cx, cy, bw, bh = box
    original = [(cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h]
    projected = [original[0]*sx-ox, original[1]*sy-oy, original[2]*sx-ox, original[3]*sy-oy]
    finite = all(math.isfinite(v) for v in projected)
    retained = finite and 0 <= projected[0] < projected[2] <= 512 and 0 <= projected[1] < projected[3] <= 512
    return {'original_image_size': list(size), 'resize_scale': [sx, sy], 'resized_size': [rw, rh],
            'crop_offsets': [ox, oy], 'original_bbox': original, 'projected_bbox': projected,
            'fully_retained': retained, 'eligible': retained,
            'exclusion_reason': None if retained else 'BBOX_NOT_FULLY_RETAINED_OR_NONPOSITIVE'}


def lineage(images):
    """Known relations only; disconnected singleton is NOT proven source lineage."""
    names = sorted(images)
    parent = {n: n for n in names}
    def find(n):
        while parent[n] != n:
            parent[n] = parent[parent[n]]
            n = parent[n]
        return n
    edges = []
    def join(a, b, kind, evidence):
        ra, rb = find(a), find(b)
        parent[max(ra, rb)] = min(ra, rb)
        edges.append({'from': a, 'to': b, 'type': kind, 'evidence': evidence})
    hashes, stems = defaultdict(list), defaultdict(list)
    for name, row in images.items():
        hashes[row['decoded_rgb_sha256']].append(name)
        match = re.fullmatch(r'(img_.+)_pre_part_\d+', Path(name).stem)
        if match:
            stems[match.group(1)].append(name)
    for key, members in sorted(hashes.items()):
        for name in sorted(members)[1:]:
            join(sorted(members)[0], name, 'EXACT_CONTENT_ALIAS', {'decoded_rgb_sha256': key})
    for key, members in sorted(stems.items()):
        for name in sorted(members)[1:]:
            join(sorted(members)[0], name, 'KNOWN_PRE_PART_SIBLING', {'explicit_filename_source_key': key})
    members = defaultdict(list)
    for name in names:
        members[find(name)].append(name)
    groups, mapping = [], {}
    for values in sorted(members.values()):
        gid = 'known_group_' + digest(values)[:20]
        related = [e for e in edges if e['from'] in values and e['to'] in values]
        groups.append({'source_group_id': gid, 'member_files': values, 'relationship_evidence': related,
                       'confidence_type': 'KNOWN_SAME_SOURCE_GROUP' if related else 'SINGLETON_LINEAGE_UNRESOLVED',
                       'physical_original_recovered': False})
        mapping.update({n: gid for n in values})
    return {'nodes': [{'id': n, 'type': 'detector_training_image', **images[n]} for n in names],
            'edges': edges, 'connected_components': groups, 'conditioning_nodes': [],
            'conditioning_scan': 'NOT_RUN_ANNOTATION_GATE_FIRST', 'physical_original_recovery_claimed': False}, mapping


def provenance(row, evidence, evidence_root):
    """Evidence must bind the actual image, full annotation file and line/class.

    Supporting records still require a human provenance review; their hashes
    prove binding, not truth. No self-filled default acceptance is generated.
    """
    if evidence is None:
        return 'REJECTED_PROVENANCE_UNKNOWN', ['no instance-bound annotation origin/review evidence']
    errors = []
    for key in ('image_sha256', 'annotation_sha256', 'annotation_line_index', 'class_id'):
        if evidence.get(key) != row[key]:
            errors.append('evidence binding mismatch: '+key)
    support = evidence.get('supporting_file')
    if not support or not evidence.get('supporting_record_locator'):
        errors.append('missing inspectable supporting record')
    else:
        path = Path(support)
        if not path.is_absolute():
            path = evidence_root/path
        if not path.is_file() or sha(path) != evidence.get('supporting_file_sha256'):
            errors.append('supporting record unavailable or hash mismatch')
    if evidence.get('provenance_review_completed') is not True or not evidence.get('provenance_reviewer'):
        errors.append('independent provenance review not documented')
    origin = evidence.get('annotation_source')
    verified = evidence.get('human_verified') is True
    if errors:
        return 'REJECTED_PROVENANCE_UNKNOWN', errors
    if verified:
        return 'ACCEPTED_HUMAN_VERIFIED', []
    if origin == 'MANUAL_ANNOTATION':
        return 'ACCEPTED_MANUAL_SOURCE', []
    if origin == 'PRELABEL':
        return 'REJECTED_PRELABEL_UNVERIFIED', ['prelabel has no documented human verification']
    return 'REJECTED_PROVENANCE_UNKNOWN', ['annotation source is not established']


def run(args):
    cfg = load(args.protocol)
    if (cfg['minimum_accepted_unique_groups_per_class'] != 5 or cfg['generation_seed'] != 2026
            or cfg['maximum_donors_per_class'] != 10 or cfg['guidance_relative_rms'] != .005
            or any(cfg[k] for k in ('downstream_detector_training_authorized', 'new_teacher_training_authorized',
                                    'official_validation_authorized', 'formal_40_plus_40_generation_authorized',
                                    'stage16a_r_auto_start'))):
        raise ValueError('R0 frozen protocol changed')
    if not args.train_images.is_dir() or not args.train_labels.is_dir():
        raise FileNotFoundError('Explicit frozen training images/labels directories required')
    out = args.output
    existing = out/'stage16a_r0_status.json'
    if existing.exists() and load(existing).get('sampling_count', 0):
        raise RuntimeError('Do not overwrite a runtime experiment with an entry-gate audit')
    integrity = load(args.oof_integrity)
    paths = sorted(p for p in args.train_images.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES)
    if len(paths) != 138 or {p.name for p in paths} != set(integrity['original']['image_names']):
        raise ValueError('Actual train images differ from the frozen 138-image OOF universe')
    evidence_doc = load(args.annotation_evidence) if args.annotation_evidence else {'records': []}
    evidence = {}
    for r in evidence_doc['records']:
        key = (r['image_id'], r['annotation_line_index'])
        if key in evidence:
            raise ValueError('Duplicate annotation evidence key')
        evidence[key] = r
    evidence_root = args.annotation_evidence.parent if args.annotation_evidence else ROOT
    images = {p.name: {'image_path': str(p.resolve()), **image_info(p)} for p in paths}
    graph, groupof = lineage(images)
    records = []
    for path in paths:
        label = args.train_labels/(path.stem+'.txt')
        if not label.is_file():
            raise FileNotFoundError('Missing frozen annotation: '+str(label))
        labelhash = sha(label)
        for line_index, line in enumerate(label.read_text(encoding='utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f'Invalid native YOLO row: {label}:{line_index}')
            cls = int(fields[0]); box = [float(v) for v in fields[1:]]
            if str(cls) not in cfg['classes'] or not all(math.isfinite(v) for v in box) or box[2] <= 0 or box[3] <= 0:
                raise ValueError(f'Invalid class/box: {label}:{line_index}')
            row = {'donor_id': path.name+':line'+str(line_index), 'image_id': path.name,
                   **images[path.name], 'annotation_file': str(label.resolve()), 'annotation_sha256': labelhash,
                   'annotation_line_index': line_index, 'class_id': cls, 'class': cfg['classes'][str(cls)],
                   'bbox_xywh': box, 'source_group_id': groupof[path.name]}
            ev = evidence.get((path.name, line_index))
            status, reasons = provenance(row, ev, evidence_root)
            row.update(annotation_source=ev.get('annotation_source', 'UNKNOWN') if ev else 'UNKNOWN',
                       human_verified=ev.get('human_verified') if ev else None,
                       prelabel_model_if_known=ev.get('prelabel_model_if_known') if ev else None,
                       revision_history_if_known=ev.get('revision_history_if_known') if ev else None,
                       provenance_status=status, rejection_reasons=reasons, supporting_evidence=ev)
            records.append(row)
            graph['nodes'].append({'id': row['donor_id'], 'type': 'donor_annotation_instance', 'annotation_sha256': labelhash})
            graph['edges'].append({'from': path.name, 'to': row['donor_id'], 'type': 'KNOWN_SOURCE_RECORD',
                                   'evidence': {'label_file': str(label.resolve()), 'label_sha256': labelhash, 'line': line_index}})
    actual = Counter(str(r['class_id']) for r in records)
    if actual != Counter({str(k): int(v) for k, v in integrity['original_counts'].items()}):
        raise ValueError('Train annotation counts differ from frozen OOF integrity')
    unused = set(evidence)-{(r['image_id'], r['annotation_line_index']) for r in records}
    if unused:
        raise ValueError('Evidence contains rows outside the frozen training annotation universe')
    accepted = [r for r in records if r['provenance_status'] in cfg['annotation_acceptance']]
    counts = {c: len({r['source_group_id'] for r in accepted if r['class'] == c}) for c in cfg['classes'].values()}
    passed = all(n >= 5 for n in counts.values())
    # Even a future annotation pass cannot authorize unimplemented runtime gates.
    status = 'ANNOTATION_PROVENANCE_INSUFFICIENT' if not passed else 'R0_INCOMPLETE_REMAINING_GATES_NOT_RUN'
    annotation = {'status': 'PASS' if passed else status, 'scope': args.scope, 'image_count': len(paths),
                  'instance_count': len(records), 'class_counts': dict(actual),
                  'inputs': {'train_images': str(args.train_images.resolve()), 'train_labels': str(args.train_labels.resolve()),
                             'oof_integrity_sha256': sha(args.oof_integrity), 'protocol_sha256': sha(args.protocol),
                             'audit_script_sha256': sha(Path(__file__)),
                             'server_training_content_equality_claimed': False},
                  'provenance_counts': dict(Counter(r['provenance_status'] for r in records)),
                  'accepted_unique_source_groups_per_class': counts,
                  'minimum_groups_per_class': 5, 'evidence_manifest_sha256': sha(args.annotation_evidence) if args.annotation_evidence else None,
                  'visual_plausibility_used_as_evidence': False, 'validation_read': False, 'records': records}
    save(out/'stage16a_r0_protocol.json', cfg)
    save(out/'annotation_provenance_audit.json', annotation)
    save(out/'source_lineage_graph.json', {'status': 'DESCRIPTIVE_KNOWN_RELATIONS_ONLY', **graph,
                                        'parent_isolation_verified': False})
    later = ['oof_teacher_isolation_audit', 'conditioning_isolation_audit', 'bbox_transform_audit',
             'bbox_eligibility_bias_audit', 'spatial_semantics_audit', 'r0_donor_registry', 'mask_geometry_audit',
             'paired_rng_freeze', 'model_state_immutability_audit', 'differentiability_audit',
             'r0_task_guidance_step_audit', 'task_loss_effectiveness', 'paired_visual_validity', 'numerical_validity']
    for name in later:
        artifact = out/(name+'.json')
        if artifact.exists() and load(artifact).get('status') not in ('NOT_RUN', 'BASELINE_SPATIAL_SEMANTICS_UNRESOLVED'):
            raise RuntimeError('Existing executed later-stage audit requires a separate output directory: '+name)
        item = {'status': 'NOT_RUN', 'blocked_by': status, 'records': [], 'pass_claimed': False}
        if name == 'spatial_semantics_audit':
            item.update(status='BASELINE_SPATIAL_SEMANTICS_UNRESOLVED', audit_execution='NOT_RUN')
        save(artifact, item)
    gates = {g: 'NOT_RUN' for g in cfg['required_gates']}
    gates['ANNOTATION_PROVENANCE'] = 'PASS' if passed else 'FAIL'
    state = {'status': status, 'scope': args.scope, 'phase': 'ANNOTATION_ENTRY_GATE', 'gates': gates,
             'STAGE16A_R_FORMAL_GENERATION_AUTHORIZED': False, 'R0_GENERATION_AUTHORIZED': False,
             'DETECTOR_TRAINING_AUTHORIZED': False, 'STAGE16B_AUTO_START': False,
             'sampling_count': 0, 'detector_training_count': 0, 'new_teacher_training_count': 0,
             'optimizer_step_count': 0, 'official_validation_use_count': 0,
             'deep_pcb_training': 0, 'deep_pcb_generation': 0, 'bootstrap_guard_modification': 0,
             'original_stage16a_stop_record_modified': False,
             'accepted_unique_source_groups_per_class': counts,
             'later_runtime_implementation_complete': False}
    save(out/'stage16a_r0_status.json', state)
    report = f'''# Plastic_Bomo Stage16A-R0 entry-gate audit

Status: `{status}`. Scope: `{args.scope}`.

## Actual evidence

Read only the explicit frozen real-training directories: {len(paths)} images,
{len(records)} annotation instances. Counts: {dict(actual)}.
The local asset names and class counts match the historical OOF universe.
This local audit does not prove equality of current server training pixels or
the presence/absence of additional provenance records on the server.
Accepted unique known groups: {counts}.
Evidence manifest supplied: {bool(args.annotation_evidence)}.
Unknown provenance is rejected, not silently treated as manual annotation.
The known-relation graph groups explicit pre_part siblings and RGB aliases;
singleton nodes do not establish physical-original provenance or OOF isolation.

Historical Black annotation-building code includes prelabel conversion; it is
not instance-level evidence of later human verification. The local real-only
README explicitly mentions manual verification of additional validation labels,
not confirmation of every training annotation. Neither observation accepts a
training row. No official validation pixels or labels were read.

## Stop and limits

The entry gate requires five accepted unique source groups in EACH class.
Later gates are NOT_RUN, not failed numerical experiments and not passes.
All required filenames are emitted with explicit blocked placeholders where
there was no execution. No donor subset, random tensors, model states, task
gradients, visual scores, or downstream results have been produced.
The later R0 generation/runtime workflow is not implemented by this entry tool.

Do not claim task guidance is ineffective: it was not tested. Do not add manual
labels, fabricate a review ledger, alter the crop, or train a new teacher to
rescue this stop. Only previously existing, inspectable provenance records may
justify a separately reviewed rerun. Original Stage16A files remain unchanged.
'''
    (out/'STAGE16A_R0_REPORT.md').write_text(report, encoding='utf-8')
    return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/plastic_bomo_stage16a_r0.json')
    p.add_argument('--train_images', type=Path, required=True)
    p.add_argument('--train_labels', type=Path, required=True)
    p.add_argument('--oof_integrity', type=Path, default=ROOT/'results/dwbg/v2/fixed_v3/oof_integrity_report.json')
    p.add_argument('--annotation_evidence', type=Path, help='Existing human-reviewed provenance records, never an auto-filled acceptance template')
    p.add_argument('--output', type=Path, default=ROOT/'results_for_gpt/plastic_bomo_stage16a_r0_traceability_preflight')
    p.add_argument('--scope', choices=['LOCAL_TRAINING_ASSET_AUDIT', 'SERVER_PREFLIGHT'], default='SERVER_PREFLIGHT')
    state = run(p.parse_args())
    print(json.dumps(state, indent=2))
    raise SystemExit(2)


if __name__ == '__main__':
    main()
