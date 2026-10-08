"""Metadata/byte integrity verification only; never decode validation or open val labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('numbering_audit', 'rebuild_audit', 'generation_train_root', 'real_data_yaml', 'output'):
        p.add_argument('--' + key, required=True, type=Path)
    a = p.parse_args()
    import yaml
    cfg = yaml.safe_load(a.real_data_yaml.read_text(encoding='utf-8'))
    root = Path(cfg.get('path', a.real_data_yaml.parent))
    if not root.is_absolute():
        root = (a.real_data_yaml.parent / root).resolve()
    val = cfg['val']
    if not isinstance(val, str):
        raise RuntimeError('VALIDATION_DIRECTORY_REQUIRED_NO_GUESSING')
    directory = Path(val) if Path(val).is_absolute() else root / val
    if not directory.is_dir():
        raise RuntimeError('OFFICIAL_VALIDATION_DIRECTORY_UNAVAILABLE:' + str(directory))
    val_names = {p.stem for p in directory.iterdir()
                 if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')}
    numbering = load(a.numbering_audit)
    if numbering['conflicts']:
        raise RuntimeError('HISTORICAL_NUMBERING_CONFLICT')
    rows = numbering['records']
    rebuild = load(a.rebuild_audit / 'baseline_rebuild_preparation_audit.json')
    folders = {'flash': '01_Flash_point', 'black': '02_Big_black_spots'}
    checked = 0
    for cls, pairs in rebuild['source_pairs'].items():
        for pair in pairs:
            image = a.generation_train_root / 'Plastic_Bomo/test' / folders[cls] / Path(pair['image']).name
            mask = a.generation_train_root / 'Plastic_Bomo/ground_truth' / folders[cls] / Path(pair['mask']).name
            if sha(image) != pair['image_sha256'] or sha(mask) != pair['mask_sha256']:
                raise RuntimeError('GENERATOR_TRAINING_SOURCE_CHANGED')
            checked += 1
    if checked != 112 or len(rows) != 112:
        raise RuntimeError('SOURCE_CATALOG_SIZE_MISMATCH')
    overlap = [r for r in rows if r['parent_stem'] in val_names]
    result = {'status': 'GENERATOR_TRAIN_VALIDATION_METADATA_OVERLAP_REPRODUCED_ON_SERVER' if overlap else 'SERVER_METADATA_OVERLAP_NOT_OBSERVED_EXACT_LINEAGE_STILL_REQUIRED',
              'scope': 'SERVER_METADATA_AUDIT_NOT_PIXEL_IDENTITY_CONFIRMATION',
              'official_data_yaml_sha256': sha(a.real_data_yaml), 'official_validation_directory': str(directory.resolve()),
              'validation_filename_count': len(val_names),
              'all_112_source_and_mask_hashes_match_training_audit': True,
              'overlap_source_count': len(overlap),
              'overlap_unique_parent_names': len({r['parent_stem'] for r in overlap}),
              'per_class_overlap': dict(Counter(r['class'] for r in overlap)),
              'overlap_records': overlap, 'validation_image_pixels_read': 0, 'validation_labels_read': 0,
              'FORMAL_GENERATION_AUTHORIZED': False, 'DETECTOR_TRAINING_AUTHORIZED': False,
              'sampling_count': 0, 'optimizer_step_count': 0, 'protocol_changed': False}
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / 'server_isolation_status.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({k: v for k, v in result.items() if k != 'overlap_records'}, indent=2))


if __name__ == '__main__':
    main()
