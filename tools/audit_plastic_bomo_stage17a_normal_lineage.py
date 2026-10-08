"""Read-only normal-origin audit. Stop on source-group overlap; never filter the pool."""
import argparse
import ast
import hashlib
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from prepare_plastic_bomo_stage17a_trainonly import REPO, load, save, sha

CONVERTER_SHA = '47bf12fe6ffce813789e85187af0680f96f5326dcf71a6bc1f2a09b0932a6dda'


def source_key(stem):
    match = re.fullmatch(r'(img_.+)_pre_part_\d+', stem)
    return match.group(1) if match else None


def historical_filter(normal_root, config):
    # Execute only the actual historical filter function, not inference imports/main.
    source = (REPO / 'inference.py').read_text(encoding='utf-8')
    funcs = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'get_valid_normal_images']
    if len(funcs) != 1:
        raise RuntimeError('HISTORICAL_NORMAL_FILTER_UNAVAILABLE')
    scope = {'os': os, 'np': np, 'Image': Image}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), '<historical_normal_filter_only>', 'exec'), scope)
    return scope['get_valid_normal_images'](str(normal_root), config)


def xml_source(path):
    root = ET.parse(path).getroot()
    return {'xml_sha256': sha(path), 'original_filename': root.findtext('filename'),
            'original_path': root.findtext('path')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('original_dataset', 'normal_root', 'runtime_probe', 'prepared', 'conversion_script', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise RuntimeError('NEW_AUDIT_DIRECTORY_REQUIRED')
    if sha(a.conversion_script) != CONVERTER_SHA:
        raise RuntimeError('RECOVERED_CONVERTER_CHANGED')
    reference = load(a.runtime_probe / 'frozen_probe_conditions.json')
    isolation = load(a.prepared / 'source_isolation_audit.json')
    expected = {Path(k).name: v for k, v in reference['normal_pool_files_sha256'].items()}
    cfg = load(REPO / 'configs/baseline_frozen_split70_s42.json')['normal_filter']
    accepted = historical_filter(a.normal_root, cfg)
    accepted_hashes = {Path(x).name: sha(x) for x in accepted}
    if accepted_hashes != expected:
        raise RuntimeError('NORMAL_POOL_DIFFERS_FROM_SERVER_RUNTIME')
    origins = sorted((a.original_dataset / 'empty_images').glob('*.jpg'))
    if len(origins) != 522:
        raise RuntimeError('ORIGINAL_NORMAL_CATALOG_CHANGED')
    # Historical converter: sorted original empty_images[:420] copied byte-for-byte.
    if {p.name for p in a.normal_root.glob('*.jpg')} != {f'{i:03d}.jpg' for i in range(420)}:
        raise RuntimeError('NUMBERED_NORMAL_CATALOG_CHANGED')
    records, overlaps, missing, val_metadata = [], [], [], {}
    used = {Path(r['normal_path']).name for r in reference['probes']}
    for index, original in enumerate(origins[:420]):
        name = f'{index:03d}.jpg'
        digest = sha(original)
        if sha(a.normal_root / name) != digest:
            raise RuntimeError('NORMAL_EXACT_ORIGIN_BINDING_FAILED:' + name)
        parent = original.stem.removeprefix('empty_')
        source_image = a.original_dataset / 'all-cut-empty' / (parent + '.jpg')
        source_xml = a.original_dataset / 'all-cut-empty' / (parent + '.xml')
        row = {'normal_filename': name, 'normal_sha256': digest, 'original_filename': original.name,
               'original_parent_stem': parent, 'accepted_by_unchanged_filter': name in expected,
               'used_in_smoke_probe': name in used, 'origin_file_bytes_exact': True,
               'source_key': source_key(parent), 'related_validation_parents': []}
        if not source_image.is_file() or not source_xml.is_file() or sha(source_image) != digest:
            missing.append(name)
            row['original_xml_binding'] = 'UNAVAILABLE'
        else:
            row['original_xml_binding'] = xml_source(source_xml)
        if name in expected:
            for val_name, val_asset in isolation['validation_images'].items():
                val_stem = Path(val_name).stem
                if not row['source_key'] or row['source_key'] != source_key(val_stem):
                    continue
                if val_name not in val_metadata:
                    val_image = a.original_dataset / 'all-cut' / val_name
                    val_xml = a.original_dataset / 'all-cut' / (val_stem + '.xml')
                    if not val_image.is_file() or not val_xml.is_file() or sha(val_image) != val_asset['image_sha256']:
                        raise RuntimeError('RELATED_VAL_ORIGINAL_BINDING_FAILED:' + val_name)
                    val_metadata[val_name] = {'validation_image_sha256': val_asset['image_sha256'], **xml_source(val_xml)}
                normal_xml = row['original_xml_binding']
                val_xml = val_metadata[val_name]
                same_xml = (isinstance(normal_xml, dict) and normal_xml['original_filename']
                            and normal_xml['original_path'] and normal_xml['original_filename'] == val_xml['original_filename']
                            and normal_xml['original_path'] == val_xml['original_path'])
                row['related_validation_parents'].append({'validation_filename': val_name,
                     'same_pre_part_source_key': True, 'same_original_xml_filename_and_path': bool(same_xml),
                     'normal_is_same_validation_partition': parent == val_stem, 'validation_metadata': val_xml})
            if row['related_validation_parents']:
                overlaps.append(row)
        records.append(row)
    status = {'status': 'NORMAL_CONDITIONING_SOURCE_GROUP_OVERLAP_CONFIRMED' if overlaps else 'NORMAL_LINEAGE_INCOMPLETE' if missing else 'NORMAL_PARENT_LINEAGE_AUDITED',
              'scope': 'LOCAL_ORIGINAL_METADATA_AND_BYTE_BINDING', 'original_normal_count': len(origins),
              'numbered_normals_exactly_bound': len(records), 'filtered_normal_count': len(expected),
              'filtered_normals_related_to_validation_source_groups': len(overlaps),
              'xml_group_overlap_confirmed': sum(any(h['same_original_xml_filename_and_path'] for h in r['related_validation_parents']) for r in overlaps),
              'overlap_normal_filenames': [r['normal_filename'] for r in overlaps],
              'smoke_probe_normals_with_group_overlap': [r['normal_filename'] for r in overlaps if r['used_in_smoke_probe']],
              'missing_original_xml_bindings': missing, 'validation_original_xml_files_read_for_source_metadata': len(val_metadata),
              'official_validation_yolo_labels_read': 0, 'validation_annotation_objects_used_for_tuning': False,
              'validation_image_pixels_decoded': 0, 'normal_pool_filtered_or_replaced': False,
              'trainonly_generator_training_used_normals': False, 'generator_retraining_required_by_this_finding': False,
              'runtime_repeat_pass_reinterpreted_as_quality_pass': False, 'formal_generation_authorized': False,
              'detector_training_authorized': False, 'sampling_count': 0, 'training_count': 0,
              'physical_original_image_pixels_recovered': False,
              'runtime_conditions_sha256': sha(a.runtime_probe / 'frozen_probe_conditions.json'),
              'isolation_audit_sha256': sha(a.prepared / 'source_isolation_audit.json'),
              'converter_sha256': CONVERTER_SHA, 'inference_sha256': sha(REPO / 'inference.py')}
    a.output.mkdir(parents=True)
    save(a.output / 'normal_lineage_status.json', status)
    save(a.output / 'normal_origin_manifest.json', {'records': records, 'overlap_records': overlaps})
    (a.output / 'NORMAL_LINEAGE_REPORT.md').write_text(
        '# Stage17A normal conditioning source isolation\n\n'
        f'Exact normal origin bindings: {len(records)}; unchanged filter accepts {len(expected)}. '
        f'Accepted normals sharing an explicit pre_part original source key with official validation: {len(overlaps)}.\n\n'
        'Numbered normal identity is proven by matching file bytes to original empty_images and all-cut-empty images, '
        'under the recovered converter ordering. Group relationships are supported by XML original filename/path '
        'metadata, with the corresponding all-cut validation-image bytes matching the server isolation audit. '
        'The unsplit physical original image is not recovered; no claim of same pixels across different partitions is made.\n\n'
        'Original validation XML files were read only for provenance metadata; current validation YOLO labels '
        'and metrics were not accessed. Historical results are unchanged. No normal was removed or substituted.\n\n'
        'Formal slots remain blocked. The train-only generator training used no normal images and is unaffected '
        'by this conditioning finding; runtime PASS still means loading and repeatability, not clean formal conditioning. '
        'A changed normal eligibility protocol requires explicit approval before implementing source-group exclusion '
        'or a new normal pool. Do not restart generator training or promote smoke samples to formal banks.\n', encoding='utf-8')
    print(__import__('json').dumps(status, indent=2))


if __name__ == '__main__':
    main()
