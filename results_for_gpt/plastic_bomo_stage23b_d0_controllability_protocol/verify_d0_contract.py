"""Read-only D0 source/metadata/contract review. No image or model imports/opens."""
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ALLOWED_INPUT_SUFFIXES = {'.py', '.json', '.md'}


def checked_bytes(path):
    path = path.resolve()
    if not path.is_relative_to(REPO) or path.suffix not in ALLOWED_INPUT_SUFFIXES:
        raise RuntimeError('D0_TEXT_INPUT_SCOPE_VIOLATION: ' + str(path))
    return path.read_bytes()


def load(path):
    return json.loads(checked_bytes(path).decode('utf-8'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))


def require(value, message):
    if not value:
        raise RuntimeError(message)


def function(path, name):
    tree = ast.parse(checked_bytes(path).decode('utf-8'))
    nodes = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name]
    require(len(nodes) == 1, 'FUNCTION_NOT_UNIQUE: ' + name)
    return nodes[0]


def main():
    source = load(HERE / 'source_evidence_index.json')
    for row in source['files']:
        require(hashlib.sha256(checked_bytes(REPO / row['path'])).hexdigest() == row['sha256'],
                'PINNED_SOURCE_CHANGED: ' + row['path'])
    # Hash all D0 artifacts without recursively traversing any dataset/model path.
    ledger = load(HERE / 'review_artifact_manifest.json')
    require(set(ledger['sha256']) == {p.name for p in HERE.iterdir()
            if p.is_file() and p.name != 'review_artifact_manifest.json'}, 'ARTIFACT_SET_CHANGED')
    for name, digest in ledger['sha256'].items():
        path = HERE / name
        require(path.parent == HERE and path.is_file(), 'BAD_ARTIFACT_PATH')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == digest, 'D0_ARTIFACT_CHANGED: ' + name)

    by_id = {r['id']: REPO / r['path'] for r in source['files']}
    env = load(by_id['M6'])
    for row in env['source_hashes']:
        if row['file'] in ('diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py',
                           'diffusers/schedulers/scheduling_ddim.py'):
            require(hashlib.sha256(checked_bytes(REPO / row['file'])).hexdigest() == row['sha256'],
                    'SERVER_SOURCE_SHA_MISMATCH')
    binding = load(by_id['M8'])
    for ident in ('C1', 'C2', 'C3', 'C9'):
        row = next(r for r in source['files'] if r['id'] == ident)
        expected = {**binding['formal_code_sha256'], **binding['R0_code_sha256']}[row['path']]
        require(row['sha256'] == expected, 'FORMAL_WRAPPER_SHA_MISMATCH')

    active = function(by_id['C5'], '__call__')
    calls = [ast.unparse(n.func) for n in ast.walk(active) if isinstance(n, ast.Call)]
    require('self.scheduler.step' in calls and 'self.vae.decode' in calls, 'ACTIVE_CHAIN_MISSING')
    require(not any('apply_overlay' in c or 'prepare_latents' in c or 'prepare_mask_latents' in c
                    for c in calls), 'ACTIVE_CHAIN_CHANGED')
    loop = next(n for n in ast.walk(active) if isinstance(n, ast.For)
                and ast.unparse(n.iter) == 'enumerate(self.scheduler.timesteps)')
    writes = [n for n in ast.walk(loop) if isinstance(n, ast.Assign)
              and any(isinstance(t, ast.Name) and t.id == 'latents' for t in n.targets)]
    require(len(writes) == 1 and 'self.scheduler.step' in ast.unparse(writes[0].value),
            'LATENT_RESTORE_OR_UPDATE_CHAIN_CHANGED')
    scheduler = function(by_id['C7'], 'step')
    require('mask_for_anomaly' not in [x.arg for x in scheduler.args.args], 'SCHEDULER_MASK_CHANGED')
    wrapper = function(by_id['C1'], 'sample')
    paste = [n for n in ast.walk(wrapper) if isinstance(n, ast.Call)
             and ast.unparse(n.func) == 'full.paste']
    require(len(paste) == 1 and len(paste[0].args) == 2 and not paste[0].keywords,
            'MASKED_COMPOSITE_OR_PASTE_CHANGED')
    require('patch.crop' in ast.unparse(paste[0].args[0]), 'PASTE_REGION_CHANGED')
    label = function(by_id['C3'], 'synthetic_label')
    require('validate_bbox(slot)' in ast.unparse(label), 'LABEL_CONTRACT_CHANGED')
    require(not any('Image' in ast.unparse(n) for n in ast.walk(label) if isinstance(n, ast.Call)),
            'LABEL_NOW_USES_PIXELS')

    cfg = load(by_id['M3'])
    require((cfg['height'], cfg['width'], cfg['strength'], cfg['eta'], cfg['padding_mask_crop'])
            == (512, 512, 1.0, 0.0, None), 'FROZEN_CALL_CHANGED')
    slots = load(by_id['M4'])['slots']
    require(len(slots) == 80, 'SLOT_PANEL_CHANGED')
    require(all(s['geometry']['valid'] and s['geometry']['generator_canvas_wh'] == [512, 512]
                and s['geometry']['resize_scale_xy'] == [1, 1]
                and s['geometry']['original_canvas_wh'] == s['geometry']['final_canvas_wh']
                and s['geometry']['mask_is_human_segmentation'] is False for s in slots),
            'GEOMETRY_METADATA_CHANGED')
    require(sum(s['geometry']['paste_valid_crop_region_xyxy'] != s['geometry']['fixed_crop_xyxy']
                for s in slots) == 22, 'PADDING_METADATA_CHANGED')
    rng = {r['slot_id']: r['banks'] for r in load(by_id['M5'])['records']}
    banks = [load(by_id[k]) for k in ('M10', 'M11', 'M12')]
    for bank in banks:
        require(bank['status'] == 'COMPLETE' and len(bank['records']) == 80, 'RAW_METADATA_INCOMPLETE')
        for row, slot in zip(bank['records'], slots):
            require(row['frozen_slot'] == slot and row['sample_seed'] == rng[slot['slot_id']][bank['bank']]['sample_seed'],
                    'SLOT_RNG_BINDING_CHANGED')
    for triplet in zip(*(b['records'] for b in banks)):
        require(all(all(r[k] == triplet[0][k] for k in ('frozen_slot', 'non_rng_metadata_sha256',
                    'checkpoint_tree_sha256', 'label_sha256')) for r in triplet), 'ABC_PAIRING_CHANGED')
    # No image_path, normal_path, label_path or checkpoint path is opened here.

    status = load(HERE / 'stage23b_d0_status.json')
    require(status['status'] == 'NO_MEASURABLE_NOVEL_CONTROLLABILITY_GAP', 'STATUS_CHANGED')
    audit = load(HERE / 'execution_audit.json')
    counters = [k for k in audit if k.endswith('_count')] + ['BootstrapGuard', 'DeepPCB',
                'official_final_eval_image_or_label_content_read', 'official_final_eval_forward']
    require(all(audit[k] == status[k] == 0 for k in counters), 'NONZERO_D0_EXECUTION_COUNT')
    require(not any(v for k, v in status.items() if k.endswith('_AUTHORIZED') or k.endswith('_REOPENED')),
            'EXECUTION_AUTHORIZATION_PRESENT')
    taxa = load(HERE / 'condition_axis_taxonomy.json')
    expected_axes = {'CLASS_IDENTITY', 'REQUESTED_CENTER', 'REQUESTED_WIDTH_HEIGHT', 'EDIT_SUPPORT_CENTER',
                     'EDIT_SUPPORT_EXTENT', 'DEFECT_VISIBILITY', 'LOCAL_CONTRAST', 'BOUNDARY_BLEND',
                     'OUTSIDE_BACKGROUND_CHANGE', 'IN_BOX_APPEARANCE_DIVERSITY', 'CROSS_RNG_CONDITION_VARIATION'}
    require({r['axis'] for r in taxa['axes']} == expected_axes and len(taxa['axes']) == 11, 'AXES_INCOMPLETE')
    require(all(r['classification'] in taxa['allowed_classifications'] for r in taxa['axes']), 'BAD_AXIS_CLASS')
    matrix = load(HERE / 'controllability_prior_art_gap_matrix.json')
    require(len(matrix['rows']) == 7, 'PRIOR_ART_MATRIX_INCOMPLETE')
    for row in matrix['rows']:
        require(set(row['cells']) == set(matrix['columns']), 'MISSING_PRIOR_ART_COLUMN')
        require(all(c['classification'] in matrix['allowed_classifications'] for c in row['cells'].values()),
                'BAD_PRIOR_ART_CLASS')
    gaps = load(HERE / 'protocol_worthy_gap_summary.json')
    require(not gaps['protocol_worthy_gaps'] and not gaps['primary_endpoints'], 'UNESTABLISHED_GAP_PROMOTED')
    require(all(not all(v is True for v in row['criteria'].values()) for row in gaps['candidates']),
            'GAP_CRITERIA_STATUS_CONTRADICTION')
    require(load(HERE / 'candidate_metric_definitions.json')['pass_fail_thresholds'] is None,
            'OUTPUT_FITTED_THRESHOLD_PRESENT')
    print(json.dumps({'status': 'D0_STATIC_CONTRACT_REVIEW_PASS', 'pinned_sources': len(source['files']),
                      'prior_art_cells': 56, 'axes': 11, 'metadata_slots': 80,
                      'artifact_hashes_verified': len(ledger['sha256']),
                      'images_or_models_opened': 0, 'residual_metrics_computed': 0,
                      'runtime_or_experiment_test': False}, indent=2))


if __name__ == '__main__':
    main()
