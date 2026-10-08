"""Verify imported original-lineage evidence against actual server normal bytes. No exclusion."""
import argparse
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify(a):
    evidence = load(a.normal_audit / 'normal_lineage_status.json')
    manifest = load(a.normal_audit / 'normal_origin_manifest.json')
    if evidence['status'] != 'NORMAL_CONDITIONING_SOURCE_GROUP_OVERLAP_CONFIRMED':
        raise RuntimeError('EXPECTED_ORIGINAL_LINEAGE_EVIDENCE_UNAVAILABLE')
    if sha(a.runtime_probe / 'frozen_probe_conditions.json') != evidence['runtime_conditions_sha256']:
        raise RuntimeError('RUNTIME_REFERENCE_CHANGED')
    if sha(a.prepared / 'source_isolation_audit.json') != evidence['isolation_audit_sha256']:
        raise RuntimeError('OFFICIAL_ISOLATION_REFERENCE_CHANGED')
    conditions = load(a.runtime_probe / 'frozen_probe_conditions.json')
    accepted = {Path(k).name: v for k, v in conditions['normal_pool_files_sha256'].items()}
    rows = manifest['records']
    if len(rows) != 420 or len({r['normal_filename'] for r in rows}) != 420:
        raise RuntimeError('NORMAL_CATALOG_INVALID')
    if {p.name for p in a.normal_root.iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png')} != {r['normal_filename'] for r in rows}:
        raise RuntimeError('SERVER_NORMAL_CATALOG_CHANGED')
    for r in rows:
        name = r['normal_filename']
        if Path(name).name != name or '/' in name or '\\' in name or sha(a.normal_root / name) != r['normal_sha256']:
            raise RuntimeError('SERVER_NORMAL_BYTES_CHANGED:' + name)
        if r['accepted_by_unchanged_filter'] != (name in accepted):
            raise RuntimeError('ACCEPTED_NORMAL_MEMBERSHIP_CHANGED')
        if name in accepted and accepted[name] != r['normal_sha256']:
            raise RuntimeError('RUNTIME_NORMAL_HASH_CHANGED')
    overlaps = [r for r in rows if r['accepted_by_unchanged_filter'] and r['related_validation_parents']]
    if [r['normal_filename'] for r in overlaps] != evidence['overlap_normal_filenames']:
        raise RuntimeError('IMPORTED_GROUP_EVIDENCE_INCONSISTENT')
    isolation = load(a.prepared / 'source_isolation_audit.json')
    for r in overlaps:
        for hit in r['related_validation_parents']:
            if not hit['same_original_xml_filename_and_path'] or hit['validation_metadata']['validation_image_sha256'] != isolation['validation_images'][hit['validation_filename']]['image_sha256']:
                raise RuntimeError('IMPORTED_XML_SOURCE_GROUP_EVIDENCE_MISMATCH')
    result = {'status': 'SERVER_NORMAL_SOURCE_GROUP_OVERLAP_EVIDENCE_BOUND' if a.scope == 'SERVER_AUDIT' else 'LOCAL_NORMAL_VERIFIER_TEST_PASS',
              'scope': a.scope,
              'normal_files_verified': 420, 'accepted_normal_count': len(accepted),
              'overlap_normal_filenames': [r['normal_filename'] for r in overlaps],
              'original_xml_evidence_location': 'local original dataset; not re-read on server',
              'origin_manifest_sha256': sha(a.normal_audit / 'normal_origin_manifest.json'),
              'normal_pool_changed': False, 'generator_retraining_required': False,
              'formal_generation_authorized': False, 'detector_training_authorized': False,
              'validation_images_or_labels_read': 0, 'sampling_count': 0, 'training_count': 0}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('normal_audit', 'normal_root', 'runtime_probe', 'prepared', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--scope', choices=('SERVER_AUDIT', 'LOCAL_TEST'), default='SERVER_AUDIT')
    a = p.parse_args()
    result = verify(a)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'server_normal_lineage_status.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
