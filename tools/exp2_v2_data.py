"""Frozen V2 data interface shared by the two baseline adapters; no ML imports."""
import hashlib
import json
from pathlib import Path


def within(path, root):
    # Native AnomalyDiffusion may use Python3.8 (Path.is_relative_to is3.9+).
    try:
        Path(path).relative_to(Path(root))
        return True
    except ValueError:
        return False


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_training_records(manifest, class_name=None):
    path = Path(manifest).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    binding = json.loads((path.parent / "prepared_binding.json").read_text(encoding="utf-8"))
    if binding.get('status') != 'FROZEN' or value.get("status") != "TRAIN_ONLY_ASSETS_PREPARED" or file_sha(path) != binding["training_manifest_sha256"]:
        raise ValueError("EXP2_TRAIN_MANIFEST_NOT_FROZEN")
    rows = value["records"]
    if len(rows) != 172 or len({r["annotation_id"] for r in rows}) != 172:
        raise ValueError("EXP2_TRAIN_POPULATION_CHANGED")
    # Bind positive TRAIN identity/geometry, not a caller-edited split field or
    # the physical name of an old train/val folder.
    import chapter3_exp1_quality_common as c
    cfg = c.load(c.REPO/'configs/plastic_bomo_chapter3_exp2.json')
    if value.get('source_bindings') != cfg['frozen_bindings_sha256']:
        raise ValueError('EXP2_SOURCE_METADATA_BINDING_CHANGED')
    old_cfg = c.load(c.resolve(cfg['exp1_config']))
    allowlist = c.resolve(old_cfg['p0_root'])/'generator_train_allowlist.json'
    relative = allowlist.relative_to(c.REPO).as_posix()
    c.checked_file(allowlist, cfg['frozen_bindings_sha256'][relative])
    origins = {r['annotation_id']: r for r in c.load(allowlist)['inputs']}
    if {r['annotation_id'] for r in rows} != set(origins):
        raise ValueError('EXP2_TRAIN_ANNOTATIONS_CHANGED')
    for row in rows:
        if row["split"] != "train" or row["canvas_wh"] != [512, 512] or row["mask_is_human_segmentation"]:
            raise ValueError("EXP2_TRAIN_RECORD_INVALID")
        old = origins[row['annotation_id']]
        if any(row[k] != old[k] for k in ('class','split','source_group_id','geometry')) or row['source_sha256'] != old['sha256']:
            raise ValueError('EXP2_TRAIN_LINEAGE_CHANGED')
    if class_name is not None:
        if class_name not in ("flash", "black"):
            raise ValueError("EXP2_CLASS_INVALID")
        rows = [r for r in rows if r["class"] == class_name]
    for row in rows:
        for role in ("image", "mask"):
            asset = (path.parent / row[role+"_path"]).resolve()
            if not within(asset, path.parent) or file_sha(asset) != row[role+"_sha256"]:
                raise ValueError("EXP2_TRAIN_ASSET_CHANGED: "+str(asset))
            row[role+"_path"] = str(asset)
    return rows
