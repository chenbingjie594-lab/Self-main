"""Verify downloaded R0 entry-gate stops; never executes later R0 stages."""
import argparse
from collections import Counter
from pathlib import Path

from audit_plastic_bomo_stage16a_r0 import load, save, sha


def review(server, local):
    checks = []
    def check(ok, name):
        checks.append({'check': name, 'pass': bool(ok)})
    state = load(server/'stage16a_r0_status.json')
    ann = load(server/'annotation_provenance_audit.json')
    before = load(local/'annotation_provenance_audit.json')
    graph = load(server/'source_lineage_graph.json')
    protocol = load(server/'stage16a_r0_protocol.json')
    check(state['scope'] == ann['scope'] == 'SERVER_PREFLIGHT', 'actual server scope')
    check(before['scope'] == 'LOCAL_TRAINING_ASSET_AUDIT', 'separate preserved local snapshot')
    check(state['status'] == ann['status'] == 'ANNOTATION_PROVENANCE_INSUFFICIENT', 'annotation stop consistency')
    check(state['phase'] == 'ANNOTATION_ENTRY_GATE', 'not a runtime experiment')
    check(ann['image_count'] == 138 and ann['instance_count'] == len(ann['records']) == 168, 'full annotation count')
    check(Counter(r['class'] for r in ann['records']) == {'flash': 88, 'black': 80}, 'class instance counts')
    check(len({r['donor_id'] for r in ann['records']}) == 168, 'no duplicate donor rows')
    check(len({r['image_id'] for r in ann['records']}) == 138, 'full annotated image universe')
    check(ann['evidence_manifest_sha256'] is None, 'no evidence supplied; not proof of evidence nonexistence')
    check(ann['provenance_counts'] == {'REJECTED_PROVENANCE_UNKNOWN': 168}, 'unknown provenance status count')
    check(state['accepted_unique_source_groups_per_class'] == ann['accepted_unique_source_groups_per_class'] == {'flash': 0, 'black': 0}, 'zero accepted groups')
    check(ann['minimum_groups_per_class'] == protocol['minimum_accepted_unique_groups_per_class'] == 5, 'frozen five-group gate')
    local_rows = {r['donor_id']: r for r in before['records']}
    immutable = ('image_sha256', 'decoded_rgb_sha256', 'image_size', 'annotation_sha256',
                 'annotation_line_index', 'class_id', 'bbox_xywh', 'source_group_id', 'provenance_status')
    check(set(local_rows) == {r['donor_id'] for r in ann['records']}, 'same local/server donor identities')
    for row in ann['records']:
        check(row['donor_id'] in local_rows and all(row[k] == local_rows[row['donor_id']][k] for k in immutable),
              'bound local/server image-label-geometry equality: '+row['donor_id'])
        check(row['annotation_source'] == 'UNKNOWN' and row['human_verified'] is None and row['supporting_evidence'] is None,
              'no fabricated annotation evidence: '+row['donor_id'])
    for key in ('oof_integrity_sha256', 'protocol_sha256', 'audit_script_sha256'):
        check(ann['inputs'][key] == before['inputs'][key], 'same audited input/code identity: '+key)
    check(ann['inputs']['train_images'].startswith('/mnt/') and ann['inputs']['train_labels'].startswith('/mnt/'), 'server training input paths')
    images = {n['id']: n for n in graph['nodes'] if n['type'] == 'detector_training_image'}
    check(len(images) == 138, 'known graph image nodes')
    groups = graph['connected_components']
    member_names = [name for g in groups for name in g['member_files']]
    check(Counter(member_names) == Counter({n: 1 for n in images}), 'graph components partition image universe')
    groupof = {n: g['source_group_id'] for g in groups for n in g['member_files']}
    for row in ann['records']:
        check(groupof.get(row['image_id']) == row['source_group_id'], 'annotation known-group binding: '+row['donor_id'])
    check(not graph['physical_original_recovery_claimed'] and not graph['parent_isolation_verified'], 'no physical-source or OOF isolation claim')
    check(all(not g['physical_original_recovered'] for g in groups), 'no recovered physical-original components')
    check(graph['conditioning_scan'] == 'NOT_RUN_ANNOTATION_GATE_FIRST' and not graph['conditioning_nodes'], 'conditioning not executed')
    for name, gate in state['gates'].items():
        check(gate == ('FAIL' if name == 'ANNOTATION_PROVENANCE' else 'NOT_RUN'), 'gate execution: '+name)
    for key in ('sampling_count', 'detector_training_count', 'new_teacher_training_count', 'optimizer_step_count',
                'official_validation_use_count', 'deep_pcb_training', 'deep_pcb_generation', 'bootstrap_guard_modification'):
        check(state[key] == 0, 'no unauthorized execution: '+key)
    for key in ('STAGE16A_R_FORMAL_GENERATION_AUTHORIZED', 'R0_GENERATION_AUTHORIZED', 'DETECTOR_TRAINING_AUTHORIZED',
                'STAGE16B_AUTO_START', 'original_stage16a_stop_record_modified', 'later_runtime_implementation_complete'):
        check(state[key] is False, 'no authorization/unsupported claim: '+key)
    check(not ann['visual_plausibility_used_as_evidence'] and not ann['validation_read'], 'no appearance/validation substitution')
    later = ['oof_teacher_isolation_audit', 'conditioning_isolation_audit', 'bbox_transform_audit',
             'bbox_eligibility_bias_audit', 'spatial_semantics_audit', 'r0_donor_registry', 'mask_geometry_audit',
             'paired_rng_freeze', 'model_state_immutability_audit', 'differentiability_audit',
             'r0_task_guidance_step_audit', 'task_loss_effectiveness', 'paired_visual_validity', 'numerical_validity']
    for name in later:
        item = load(server/(name+'.json'))
        correct = item['status'] == 'NOT_RUN'
        if name == 'spatial_semantics_audit':
            correct = item['status'] == 'BASELINE_SPATIAL_SEMANTICS_UNRESOLVED' and item['audit_execution'] == 'NOT_RUN'
        check(correct and not item['records'] and item['pass_claimed'] is False and item['blocked_by'] == state['status'],
              'honest blocked artifact: '+name)
    failures = [c for c in checks if not c['pass']]
    return {'status': 'SERVER_STOP_REVIEW_PASS' if not failures else 'SERVER_STOP_REVIEW_FAILED',
            'checks_passed': len(checks)-len(failures), 'checks_total': len(checks), 'failures': failures,
            'scope': 'DOWNLOADED_SERVER_ARTIFACT_REVIEW', 'experiment_status': state['status'],
            'inference': 'No instance-bound provenance evidence was supplied; this does not prove labels are wrong or that evidence cannot exist elsewhere.',
            'local_server_image_and_label_equality': not any(not c['pass'] for c in checks if 'local/server' in c['check']),
            'gpu_tests_performed': False, 'source_lineage_and_runtime_gates_verified': False,
            'known_component_count': len(groups),
            'server_raw_file_sha256': {p.name: sha(p) for p in sorted(server.iterdir()) if p.is_file()},
            'checks': checks}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server', type=Path, required=True)
    p.add_argument('--local', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = review(a.server, a.local)
    save(a.output, result)
    print(result['status'], result['checks_passed'], '/', result['checks_total'])
    if result['failures']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
