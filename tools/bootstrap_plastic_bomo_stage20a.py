"""Build immutable V2 METADATA only. Does not copy/alter dataset images or labels.

Historical eval pixels and annotations are accessed for lineage/counts only, as
requested by Stage20A. They are never fed to a model or used for design tuning.
No torch/diffusion/detector imports; no training or synthetic generation.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
from PIL import Image

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASSES = {0: "flash", 1: "black"}


def digest(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def source_key(name):
    stem = Path(name).stem.removeprefix("empty_")
    m = re.fullmatch(r"(img_[^/]+)_pre_part_\d+", stem)
    return m.group(1) if m else None


def xml_metadata(path):
    if not path.is_file(): return None
    raw = path.read_bytes()
    if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
        raise ValueError("XML_ENTITY_NOT_ALLOWED:"+str(path))
    root = ET.fromstring(raw)
    return {"path": str(path), "sha256": digest(raw),
            "original_filename": root.findtext("filename"), "original_path": root.findtext("path"),
            "object_count": len(root.findall("object"))}


def valid_box(box):
    if len(box) != 4 or not all(math.isfinite(v) for v in box): return False
    x, y, w, h = box
    return w > 0 and h > 0 and 0 <= x <= 1 and 0 <= y <= 1 and x-w/2 >= -1e-5 and y-h/2 >= -1e-5 and x+w/2 <= 1+1e-5 and y+h/2 <= 1+1e-5


def image_record(task):
    role, base, path, old_split = task
    with Image.open(path) as raw:
        rgb = raw.convert("RGB"); size = list(rgb.size); pixels = rgb.tobytes()
    # Same size-prefixed RGB hash convention as the historical frozen-source audit.
    h = hashlib.sha256(json.dumps(size).encode()); h.update(pixels)
    file_hash = sha(path); canonical = "real_"+h.hexdigest()
    relative = path.relative_to(base).as_posix()
    node = "asset_"+digest((role+"|"+relative+"|"+file_hash).encode())
    meta = xml_metadata(path.with_suffix(".xml"))
    key = source_key(path.name)
    return {"node_id": node, "canonical_image_id": canonical, "asset_role": role,
            "current_path": str(path), "portable_locator": {"root_role": role, "relative_path": relative},
            "file_sha256": file_hash, "decoded_rgb_sha256": h.hexdigest(),
            "rgb_bytes_sha256": digest(pixels), "width": size[0], "height": size[1],
            "class_instances": [], "known_original_id": meta.get("original_filename") if meta else None,
            "known_parent_id": None, "known_source_group_id": None,
            "known_filename_source_key": key,
            "derivation_type": "PRIMARY_DETECTOR_REPRESENTATION" if role == "detector" else "HISTORICAL_REAL_EXPORT",
            "derivation_evidence": [meta] if meta else [], "possible_aliases": [], "possible_sibling_crops": [],
            "annotation_file": None, "annotation_source_family": None,
            "split_membership_old": old_split, "source_xml_metadata": meta,
            "physical_source_complete": False}


class UnionFind:
    def __init__(self, ids): self.parent = {x: x for x in ids}
    def find(self, x):
        if self.parent[x] != x: self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b: self.parent[max(a,b)] = min(a,b)


def stable_split(group_ids, protocol):
    ordered = sorted(group_ids, key=lambda x: (digest((protocol["dataset_protocol_name"]+x+str(protocol["split_seed"])).encode()), x))
    ntrain = math.floor(len(ordered)*protocol["train_group_fraction"])
    return {g: "train" if i < ntrain else "final_eval" for i, g in enumerate(ordered)}


def make_graph(records, explicit_edges):
    by_id = {r["node_id"]: r for r in records}; uf = UnionFind(by_id); edges = []
    def connect(a, b, kind, why):
        if a == b: return
        if a not in by_id or b not in by_id: raise ValueError("EDGE_NODE_UNAVAILABLE")
        uf.union(a,b); edges.append({"from": a, "to": b, "type": kind, "evidence": why})
    for field, kind in [("canonical_image_id", "EXACT_RGB_ALIAS"), ("known_filename_source_key", "KNOWN_SIBLING_CROP")]:
        groups = defaultdict(list)
        for r in records:
            if r[field]: groups[r[field]].append(r)
        for key, rows in sorted(groups.items()):
            for r in rows[1:]: connect(rows[0]["node_id"],r["node_id"],kind,{"field":field,"value":key})
    xmlgroups = defaultdict(list)
    for r in records:
        m = r["source_xml_metadata"]
        if m and m["original_filename"] and m["original_path"]:
            xmlgroups[(m["original_filename"], m["original_path"])].append(r)
    for key, rows in sorted(xmlgroups.items()):
        for r in rows[1:]: connect(rows[0]["node_id"],r["node_id"],"KNOWN_MANIFEST_PARENT",{"XML_original_filename_and_path":list(key)})
    for e in explicit_edges: connect(e["from"],e["to"],e["type"],e["evidence"])
    comps = defaultdict(list)
    for r in records: comps[uf.find(r["node_id"])].append(r)
    groups = []
    for rows in comps.values():
        # Content-derived ID, independent of local/server absolute path prefixes.
        anchor = sorted({r["canonical_image_id"] for r in rows})
        gid = "sg_"+digest(json.dumps(anchor,separators=(",",":")).encode())
        for r in rows: r["known_source_group_id"] = gid
        groups.append({"source_group_id":gid,"nodes":sorted(r["node_id"] for r in rows),
                       "canonical_images":anchor,"physical_source_complete":False,
                       "singleton":len(rows)==1,"knowledge_scope":"KNOWN_RELATIONS_ONLY"})
    for e in edges:
        if e["type"] == "EXACT_RGB_ALIAS":
            by_id[e["from"]]["possible_aliases"].append(e["to"]); by_id[e["to"]]["possible_aliases"].append(e["from"])
        if e["type"] == "KNOWN_SIBLING_CROP":
            by_id[e["from"]]["possible_sibling_crops"].append(e["to"]); by_id[e["to"]]["possible_sibling_crops"].append(e["from"])
    return sorted(groups,key=lambda x:x["source_group_id"]),edges


def source_evidence(repo, relative):
    p = repo / relative
    if p.is_file(): return load(p), {"path":str(p),"sha256":sha(p),"origin":"WORKTREE"}
    r = subprocess.run(["git","-C",str(repo),"show","23731281570f8bedadad50f058ec4a718fb83524:"+relative],capture_output=True)
    if r.returncode: return None, {"path":relative,"available":False}
    return json.loads(r.stdout), {"path":relative,"sha256":digest(r.stdout),"origin":"PINNED_GIT_BLOB"}


def provenance(records, xml_audit, black_audit, old_train_expected):
    xml_by = {x["donor_id"]:x for x in (xml_audit or {}).get("records",[])}
    black_by = {x["frozen_image"]:x for x in (black_audit or {}).get("records",[])}
    ledger = []; explicit = []; recordsha = defaultdict(list)
    for r in records: recordsha[r["file_sha256"]].append(r)
    for r in records:
        if r["asset_role"] != "detector": continue
        label = Path(r["current_path"]).parent.parent.parent/"labels"/r["split_membership_old"]/(Path(r["current_path"]).stem+".txt")
        if not label.is_file(): raise ValueError("PRIMARY_LABEL_MISSING:"+str(label))
        r["annotation_file"] = str(label); labelsha = sha(label)
        for index,line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(),1):
            if not line.strip(): continue
            v = line.split()
            if len(v)!=5 or int(v[0]) not in CLASSES: raise ValueError("INVALID_PRIMARY_LABEL:"+str(label))
            cls=CLASSES[int(v[0])]; box=[float(x) for x in v[1:]]
            if not valid_box(box): raise ValueError("INVALID_PRIMARY_BBOX:"+str(label))
            donor=Path(r["current_path"]).name+":line"+str(index); x=xml_by.get(donor)
            family="OTHER"; chain=[]; reproducible=False; issue=None; historical=None
            if x and x["frozen_image_sha256"]==r["file_sha256"] and x["frozen_bbox_xywh"]==box and x.get("closest_same_class_object"):
                m=x["closest_same_class_object"]; family="XML"; historical=m["xml"]
                chain=[{"source":"XML","sha256":m["xml_sha256"],"object_index":m["object_index"],"class_mapping":"Flash point->0; Big black spots->1"},
                       {"destination":"unchanged_frozen_YOLO","label_sha256":labelsha,"line":index,"coordinate_check_status":x["status"]}]
                reproducible=x["status"]=="XML_YOLO_COORDINATES_MATCH"
                if x["status"]=="CONSISTENT_WITH_ONE_PIXEL_EDGE_CLIPPING":
                    issue={"type":"APPROX_ONE_PIXEL_EDGE_CLIPPING","historical_xml_bbox":m["xml_bbox_xywh"],"frozen_bbox":box,"cause_confirmed_by_conversion_script":False}
            elif Path(r["current_path"]).name.startswith("blackinst_") and cls=="black":
                family="SUPPLEMENTAL_PRELABEL"; b=black_by.get(Path(r["current_path"]).name)
                historical="historical --prelabels import; original per-instance author/review event unresolved"
                chain=[{"entry":"finalize_black_instance_yolo.py --prelabels","human_verification_event":"UNRESOLVED"},
                       {"destination":"unchanged_frozen_YOLO","label_sha256":labelsha,"line":index}]
                if b and b["frozen_image_sha256"]==r["file_sha256"] and b["frozen_label_sha256"]==labelsha:
                    for c in b["candidates"]:
                        if c["exact_file_match"]:
                            for parent in recordsha[c["source_sha256"]]:
                                explicit.append({"from":parent["node_id"],"to":r["node_id"],"type":"KNOWN_EXPORT_TRANSFORM","evidence":c})
                                r["known_parent_id"]=parent["canonical_image_id"]
                    chain.append({"image_export_reproduced":b["status"],"box_conversion_reproducibility":"UNRESOLVED"})
            aid="ann_"+digest((r["canonical_image_id"]+"|"+str(index)+"|"+labelsha).encode())
            item={"annotation_id":aid,"node_id":r["node_id"],"canonical_image_id":r["canonical_image_id"],
                  "class":cls,"bbox":box,"bbox_format":"normalized_xywh","annotation_file":str(label),"label_sha256":labelsha,
                  "line":index,"source_family":family,"historical_source_path":historical,"conversion_chain":chain,
                  "conversion_reproducible":reproducible,"human_verification_status":"UNCONFIRMED" if family=="SUPPLEMENTAL_PRELABEL" else "UNKNOWN",
                  "known_conversion_issue":issue,"split_membership_old":r["split_membership_old"],
                  "dataset_defined_supervision":True,"physical_annotation_completeness_guaranteed":False}
            r["class_instances"].append(aid); ledger.append(item)
        r["annotation_source_family"]=sorted({x["source_family"] for x in ledger if x["node_id"]==r["node_id"]})
    old=[x for x in ledger if x["split_membership_old"]=="train"]
    if Counter(x["class"] for x in old)!=Counter(old_train_expected): raise ValueError("OLD_FROZEN_TRAIN_BOX_COUNTS_CHANGED")
    if Counter(x["source_family"] for x in old)!={"XML":118,"SUPPLEMENTAL_PRELABEL":50}: raise ValueError("OLD_168_PROVENANCE_NOT_BOUND")
    return ledger,explicit


def future_specs(repo, protocol):
    # Choose on code/input-contract feasibility only, never on historical mAP.
    code_paths=["train_dreambooth_noise.py","diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py"]
    evidence=[]
    for f in code_paths:
        r=subprocess.run(["git","-C",str(repo),"show","23731281570f8bedadad50f058ec4a718fb83524:"+f],capture_output=True)
        evidence.append({"path":f,"commit":"23731281570f8bedadad50f058ec4a718fb83524","available":r.returncode==0,"sha256":digest(r.stdout) if r.returncode==0 else None})
    baseline={"selected_family":"SD2_INPAINTING_RAW_V2","identity":"plastic_bomo_generation_v2_raw_sd2",
        "selection_basis":protocol["baseline_selection_basis"],"historical_metrics_or_visuals_used":False,
        "model_repository":"https://huggingface.co/sd2-community/stable-diffusion-2-inpainting",
        "model_card_is_community_mirror_not_affiliated_with_Stability_AI":True,
        "code_repository":"https://github.com/chenbingjie594-lab/Self-main","pinned_code":evidence,
        "pretrained_weights_policy":"Original public SD2 inpainting initialization only; no Stage14-18 fine-tuned checkpoint; Stage20B must hash-bind the actual base files before training",
        "required_training_inputs":["V2 TRAIN real images and unchanged labels","train-derived box/mask geometry","class prompt"],
        "allowed_conditioning":"V2 TRAIN normals with resolved source metadata; no eval fallback",
        "annotation_contract":"Fixed target mask/bbox geometry from train labels; bbox-derived masks are weak geometry, not human segmentation ground truth",
        "mask_policy":"No new mask module; deterministic rectangular bbox rasterization when segmentation unavailable; contract must be verified before Stage20B sampling",
        "training":"Separate new V2 run; full UNet baseline only, text encoder/VAE frozen; no teacher, adapter or new loss",
        "no_historical_checkpoint_reuse":True,"selector_dependency":False,"detector_at_inference":False,
        "raw_output_policy":"Use every successfully generated slot; structural failures logged without quality-based replacement; no best-of-N",
        "clean_train_only_protocol_defined":all(x["available"] for x in evidence),
        "runtime_contract_tested":False,"installation_or_download_performed":False,
        "alternatives_not_selected":{
          "AnomalyDiffusion":"Local code exists, but its native input/mask and detector bbox contract requires more reconstruction than the existing inpainting baseline; no quality or mAP comparison.",
          "DefectFill":"Local implementation is unofficial and permits test-mask fallback; fewer audited bindings for the V2 train-only contract. Not selected on visual or utility grounds."},
        "not_a_rescue":"Stage17A and Stage18A remain retired; shared public architecture does not reuse the old dataset protocol or checkpoint"}
    detector={"architecture":"YOLO11s","pretrained_initialization":"same hash-bound public yolo11s.pt in all arms",
        "detector_seed":42,"epochs":150,"batch":1,"imgsz":1536,"patience":151,"early_stopping_enabled":False,
        "primary_checkpoint":"epoch150 last.pt","best_pt_selection":False,"validation_during_training":False,
        "validation_based_tuning":False,"eval":"V2 final_eval only after all arms and checkpoints are frozen",
        "arms":["RR","RawGenerator-A","RawGenerator-B","RawGenerator-C"],
        "real_base":"ALL V2 TRAIN detector images, identical across arms","extra_images":80,
        "RR":"80 repeat exposures of predeclared train donors matched to synthetic slots, not 80 new independent samples",
        "epochs_draws_batches_attempted_optimizer_steps":"Must match all arms; audit successful AMP updates separately; never call unequal successful updates strict equal compute",
        "hyperparameters":"All remaining hyperparameters must be pinned before any Stage20B arm runs, shared across arms, and never adjusted after metrics",
        "automatic_training":False}
    gate={"metric":"mAP50-95","units":"percentage_points","A_mean_raw_bank_minus_RR_ge_pp":0.5,
        "B_strict_bank_wins_ge":2,"C_every_class_mean_delta_ge_pp":-2.0,"all_three_gates_required":True,
        "banks_are_generation_RNG_replicates_not_detector_seed_replicates":True,
        "failure_status":"GENERATOR_BASELINE_HEADROOM_NOT_CONFIRMED","automatic_fallback_generator":False,
        "if_all_pass":{"GENERATOR_BASELINE_HEADROOM_CONFIRMED":True,"TASK_DISTRIBUTION_CALIBRATION_DEVELOPMENT":True},
        "current_TDCRG_authorized":False,"Stage17A_nuisance_threshold_reused":False}
    slot={"schema_version":"V2.0","generation_count":0,"actual_slots_frozen":False,
        "required_fields":["slot_id","class","source_group_id","source_image_id","conditioning_normal_id","target_geometry","mask_if_available","annotation","prompt_or_class_condition"],
        "annotation_schema":{"format":"normalized_xywh plus pixel_xyxy","source":"V2 TRAIN provenance ledger","transform_evidence":"required"},
        "target_geometry_schema":{"canvas_wh":"required","crop_xyxy":"required if cropped","resize_scale_xy":"required if resized","placement_xy":"required if placed","bbox_transform":"required"},
        "invariants":["donor and normal must be V2 TRAIN","same slots/order/conditions/geometry/prompts across A/B/C","same source groups and class counts","banks differ ONLY in generation RNG"],
        "banks":protocol["generation_rng_banks"],"budget":protocol["budget"],
        "seed_rule":"Per-slot seed = first 64 bits SHA256(protocol_name + slot_id + bank_rng_key), shared deterministic derivation; persist generator states",
        "invalid_output_policy":"Only structural invalidity; log failure, never LPIPS/KID/confidence/manual quality rejection"}
    return baseline,detector,gate,slot


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root",type=Path,default=Path(".")); p.add_argument("--source_root",type=Path,required=True)
    p.add_argument("--real_root",type=Path,required=True); p.add_argument("--derived_root",type=Path)
    p.add_argument("--spec",type=Path,default=Path("data_protocols/plastic_bomo_generation_v2/bootstrap_spec.json"))
    p.add_argument("--output",type=Path,required=True); p.add_argument("--workers",type=int,default=4)
    a=p.parse_args(); repo=a.repo_root.resolve(); cfg=load(repo/a.spec)
    if a.output.exists(): raise RuntimeError("V2_OUTPUT_EXISTS_NEW_VERSION_REQUIRED")
    a.output.mkdir(parents=True); out=a.output
    evidence={}; inputs={}
    for key,f in cfg["historical_evidence"].items(): inputs[key],evidence[key]=source_evidence(repo,f)
    tasks=[]
    for split in ("train","val"):
        directory=a.real_root/"images"/split
        if not directory.is_dir(): raise RuntimeError("REAL_IMAGE_DIRECTORY_UNAVAILABLE:"+str(directory))
        tasks += [("detector",a.real_root,x,split) for x in sorted(directory.iterdir()) if x.suffix.lower() in EXTS]
    for role,sub in [("original_parts","all-cut"),("empty_parts","all-cut-empty"),("normal","empty_images"),("black_original","Black instance")]:
        base=a.source_root/sub
        if base.is_dir(): tasks += [(role,base,x,None) for x in sorted(base.iterdir()) if x.suffix.lower() in EXTS]
    print("Stage20A: hashing real assets and decoding only for lineage; image tasks="+str(len(tasks)),flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as pool: records=list(pool.map(image_record,tasks))
    if sum(x["asset_role"]=="detector" and x["split_membership_old"]=="train" for x in records)!=cfg["expected_old_train_images"]:
        raise RuntimeError("OLD_FROZEN_TRAIN_IMAGE_COUNT_CHANGED")
    ledger,explicit=provenance(records,inputs["xml_boxes"],inputs["black_exports"],cfg["expected_old_train_boxes"])
    parent_by={r["file_sha256"]:r for r in records if r["asset_role"]=="detector"}
    representations=[]
    if a.derived_root and inputs["train_derivatives"]:
        for d in inputs["train_derivatives"]["records"]:
            parent=parent_by.get(d["parent_image_sha256"])
            if not parent: continue
            im=a.derived_root/d["image_relative"]; mask=a.derived_root/d["mask_relative"]
            if im.is_file() and sha(im)==d["image_sha256"]:
                r=image_record(("train_derivative",a.derived_root,im,"train")); r["derivation_type"]="KNOWN_MANIFEST_CROP"
                r["known_parent_id"]=parent["canonical_image_id"]; r["derivation_evidence"].append({"manifest":evidence["train_derivatives"],"source_id":d["source_id"],"crop_source_xyxy":d["crop_source_xyxy"]})
                records.append(r); explicit.append({"from":parent["node_id"],"to":r["node_id"],"type":"KNOWN_DERIVED_FROM","evidence":r["derivation_evidence"]})
                if mask.is_file() and sha(mask)==d["mask_sha256"]:
                    representations.append({"representation_id":"mask_"+d["source_id"],"path":str(mask),"sha256":d["mask_sha256"],"parent_node_id":parent["node_id"],"type":"DERIVED_MASK"})
    print("Stage20A: building known-source graph and frozen hash split",flush=True)
    groups,edges=make_graph(records,explicit); split=stable_split([g["source_group_id"] for g in groups],cfg)
    by_id={r["node_id"]:r for r in records}
    for r in records:r["split_v2"]=split[r["known_source_group_id"]]
    for g in groups:g["split"]=split[g["source_group_id"]]
    for x in representations:
        parent=by_id[x["parent_node_id"]];x["source_group_id"]=parent["known_source_group_id"];x["split"]=parent["split_v2"]
    for x in ledger:
        r=by_id[x["node_id"]];x["source_group_id"]=r["known_source_group_id"];x["split_v2"]=r["split_v2"]
    class_audit={}
    for s in cfg["split_parts"]:
        primary=[r for r in records if r["asset_role"]=="detector" and r["split_v2"]==s]; anns=[x for x in ledger if x["split_v2"]==s]
        class_audit[s]={"image_count":len(primary),"detector_source_group_count":len({r["known_source_group_id"] for r in primary}),
            "all_source_group_count":sum(g["split"]==s for g in groups),"class_instance_counts":dict(Counter(x["class"] for x in anns)),
            "annotation_source_counts":dict(Counter(x["source_family"] for x in anns))}
    coverage=all(class_audit[s]["class_instance_counts"].get(c,0)>0 for s in cfg["split_parts"] for c in CLASSES.values())
    conflicts=[e for e in edges if by_id[e["from"]]["split_v2"]!=by_id[e["to"]]["split_v2"]]
    rgbsets={s:{r["decoded_rgb_sha256"] for r in records if r["split_v2"]==s} for s in cfg["split_parts"]}
    overlap=sorted(rgbsets["train"]&rgbsets["final_eval"])
    normals=[]; emptyhash=defaultdict(list)
    for r in records:
        if r["asset_role"]=="empty_parts":emptyhash[r["file_sha256"]].append(r)
    for r in records:
        if r["asset_role"]!="normal":continue
        origins=[x for x in emptyhash[r["file_sha256"]] if x["source_xml_metadata"] and x["source_xml_metadata"]["object_count"]==0 and x["known_filename_source_key"]]
        resolved=bool(origins) and all(x["known_source_group_id"]==r["known_source_group_id"] for x in origins)
        normals.append({"normal_id":r["canonical_image_id"],"node_id":r["node_id"],"path":r["current_path"],"sha256":r["file_sha256"],
            "known_source_group":r["known_source_group_id"],"split":r["split_v2"],"origin_evidence":[x["source_xml_metadata"] for x in origins],
            "status":"NORMAL_SOURCE_RESOLVED" if resolved else "NORMAL_SOURCE_UNRESOLVED", "confirmatory_allowed":resolved and r["split_v2"]=="train",
            "physical_source_complete":False,"brightness_or_quality_filter_used":False})
    normal_ok=any(x["confirmatory_allowed"] for x in normals)
    train_groups=sorted({r["known_source_group_id"] for r in records if r["asset_role"]=="detector" and r["split_v2"]=="train"})
    ordered=sorted(train_groups,key=lambda g:(digest((cfg["dataset_protocol_name"]+g+"2026OOF5").encode()),g))
    folds={g:i%cfg["oof_k"] for i,g in enumerate(ordered)}
    foldrecords=[{"canonical_image_id":r["canonical_image_id"],"node_id":r["node_id"],"source_group_id":r["known_source_group_id"],
                 "held_out_fold":folds[r["known_source_group_id"]],"guiding_teacher":"teacher_without_held_out_source_groups"}
                 for r in records if r["asset_role"]=="detector" and r["split_v2"]=="train"]
    fold_audits={str(i):{c:sum(x["class"]==c and folds[x["source_group_id"]]==i for x in ledger if x["split_v2"]=="train") for c in CLASSES.values()} for i in range(cfg["oof_k"])}
    oof_ok=all(v>0 for x in fold_audits.values() for v in x.values())
    baseline,detector,gate,slots=future_specs(repo,cfg)
    checks={"source_group_registry":bool(groups),"no_known_source_leakage":not conflicts and not overlap,
            "split_class_coverage":coverage,"train_only_normal_pool":normal_ok,"clean_baseline_spec":baseline["clean_train_only_protocol_defined"],"oof_fold_class_coverage":oof_ok}
    ready=all(checks.values())
    specific="V2_SOURCE_GROUP_LEAKAGE" if conflicts or overlap else "V2_SPLIT_CLASS_COVERAGE_INSUFFICIENT" if not coverage else None
    save(out/"v2_protocol.json",{**cfg,"evidence":evidence,"spec_sha256":sha(repo/a.spec),"split_frozen_before_metrics":True,"annotation_completeness_guaranteed":False})
    save(out/"real_source_registry.json",{"decoded_RGB_hash_convention":"SHA256(JSON([width,height]) bytes + RGB bytes)","physical_source_complete":False,"records":records,"derived_representations":representations})
    save(out/"source_group_graph.json",{"nodes":[{"node_id":r["node_id"],"canonical_image_id":r["canonical_image_id"]} for r in records],"edges":edges,"visual_similarity_edges_used":0,"physical_source_complete":False})
    save(out/"source_group_registry.json",{"records":groups,"physical_source_complete":False})
    save(out/"annotation_provenance_v2.json",{"records":ledger,"old_train_168_count":168,"old_train_source_counts":dict(Counter(x["source_family"] for x in ledger if x["split_membership_old"]=="train")),"additional_old_eval_boxes":sum(x["split_membership_old"]=="val" for x in ledger),"human_origin_inferred_from_XML":False,"labels_unchanged":True,"one_pixel_issues":[x["annotation_id"] for x in ledger if x["known_conversion_issue"]]})
    save(out/"v2_split_manifest.json",{"version":"V2.0","group_splits":split,"image_assignments":[{"node_id":r["node_id"],"canonical_image_id":r["canonical_image_id"],"source_group_id":r["known_source_group_id"],"split":r["split_v2"],"asset_role":r["asset_role"]} for r in records],"derived_representations":representations,"third_part_created":False,"final_eval_generator_design_use_forbidden":True})
    save(out/"v2_split_class_audit.json",{"status":"PASS" if coverage else "V2_SPLIT_CLASS_COVERAGE_INSUFFICIENT","splits":class_audit,"40_40_budget_changed":False,"train_class_instance_count_below_40":[c for c in CLASSES.values() if class_audit["train"]["class_instance_counts"].get(c,0)<40],"budget_feasibility_note":"80 synthetic need not imply 80 unique donors; future slot policy must explicitly freeze any donor reuse; budget not adjusted here"})
    group_sets={s:{r["known_source_group_id"] for r in records if r["split_v2"]==s} for s in cfg["split_parts"]}
    save(out/"v2_split_isolation_audit.json",{"status":"PASS" if not conflicts and not overlap else "V2_SOURCE_GROUP_LEAKAGE","cross_split_edges":conflicts,"cross_split_rgb_aliases":overlap,"cross_split_group_intersection":sorted(group_sets["train"]&group_sets["final_eval"]),"known_edge_types_checked":cfg["known_edge_types"],"physical_source_complete":False,"unknown_relations_not_proven_absent":True})
    save(out/"normal_source_registry.json",{"records":normals,"pool_count":len(normals),"allowed_future_normals":[x["normal_id"] for x in normals if x["confirmatory_allowed"]]})
    save(out/"normal_isolation_audit.json",{"status":"PASS" if normal_ok else "NORMAL_SOURCE_UNRESOLVED","counts":dict(Counter(("train_allowed" if x["confirmatory_allowed"] else "eval_excluded" if x["status"]=="NORMAL_SOURCE_RESOLVED" else "unresolved_excluded") for x in normals)),"allowed_normals_all_train":all(x["split"]=="train" and x["status"]=="NORMAL_SOURCE_RESOLVED" for x in normals if x["confirmatory_allowed"]),"final_eval_normals_allowed":0,"unknown_normals_default_excluded":True,"physical_source_complete":False})
    save(out/"v2_oof_fold_manifest.json",{"k":5,"folds":folds,"donor_records":foldrecords,"held_out_class_counts":fold_audits,"class_coverage_pass":oof_ok,"teachers_trained":0,"fold_unit":"source_group","fold_population":"V2 TRAIN groups containing labeled detector images; normal-only/auxiliary-only groups are not detector teacher training samples","teacher_training_groups":"V2 TRAIN labeled detector groups MINUS donor held_out_fold groups","final_eval_used":False})
    for fn,data in [("v2_generator_baseline_selection.json",baseline),("future_detector_protocol.json",detector),("future_baseline_gate.json",gate),("future_generation_slot_schema.json",slots)]:save(out/fn,data)
    method_spec=repo/"data_protocols/plastic_bomo_generation_v2/tdcrg_future_method_spec.md"
    (out/"tdcrg_future_method_spec.md").write_bytes(method_spec.read_bytes())
    status={"status":"PLASTIC_BOMO_GENERATION_V2_PROTOCOL_READY" if ready else "V2_PROTOCOL_NOT_READY","specific_stop_reason":specific,
            "gates":checks,"STAGE20B_RAW_GENERATOR_BASELINE_AUTHORIZED":ready,"TDCRG_DEVELOPMENT_AUTHORIZED":False,"stage20b_auto_start":False,
            "generator_training_count":0,"synthetic_generation_count":0,"detector_training_count":0,"teacher_training_count":0,"optimizer_step_count":0,
            "official_validation_use_count":0,"administrative_old_eval_images_registered":sum(r["asset_role"]=="detector" and r["split_membership_old"]=="val" for r in records),
            "administrative_old_eval_labels_read":sum(r["asset_role"]=="detector" and r["split_membership_old"]=="val" for r in records),
            "official_validation_use_definition":cfg["official_validation_use_definition"],"DeepPCB":0,"BootstrapGuard":0,"historical_files_modified":False,"physical_source_complete":False,
            "authorization_scope":"New V2 baseline calibration only; Stage20B must verify frozen assets, complete fixed training/generation/slot specifications and structural preflight before execution"}
    save(out/"stage20a_status.json",status)
    save(out/"execution_audit.json",{"utc":datetime.now(timezone.utc).isoformat(),"script_sha256":sha(Path(__file__)),"read_only_dataset":True,"counts":{"real_assets":len(records),"groups":len(groups),"edges":len(edges),"annotations":len(ledger)},"model_import_or_forward_count":0})
    report=f"""# Stage20A — Plastic_Bomo Generation Protocol V2 Bootstrap

Status: `{status['status']}`. This is a NEW protocol, not a rescue of Stage17/18. Their stop records remain unchanged.

1. Source lineage: {len(records)} real representations / {len(groups)} known-source groups / {len(edges)} evidenced edges. Exact RGB aliases, explicit pre_part siblings, XML-original metadata and hash-bound exports/crops are linked; no visual-similarity grouping. `physical_source_complete=false`: unknown physical relations are not guaranteed absent.
2. Split: fixed seed 2026, group-level SHA256 order, 75% train / 25% final_eval by GROUP. Only two parts: no independent third population. Known cross-split edge conflicts {len(conflicts)}, RGB alias overlap {len(overlap)}. All annotations and manifest-linked masks inherit their parent split. Splits are not chosen by mAP. See class audit for actual image/box counts.
3. Annotation boundary: unchanged 168 former train boxes, XML=118 (88 Flash / 30 Black), supplemental prelabels=50 (25 images); {sum(x['split_membership_old']=='val' for x in ledger)} former eval boxes separately included for V2 administrative regrouping. Two Black edge-clipping cases are explicitly ledgered, not corrected. XML is not proof of human origin; supplemental human verification remains unresolved. Claim only **utility under the frozen Plastic_Bomo V2 annotation protocol**, never physical completeness.
4. Normals: {len(normals)} registered, {sum(x['confirmatory_allowed'] for x in normals)} resolved V2 TRAIN normals eligible. Unresolved or final_eval normals excluded; no brightness/quality selector. Empty-part XML lineage proves the known source relation, not exhaustive physical acquisition independence.
5. Future baseline: `{baseline['selected_family']}`. Selected ONLY on pinned code reproducibility, train-only allowlist feasibility, spatial mask/bbox contract and RNG/slot compatibility. Start from original public pretrained weights, not historical fine-tuned weights. The [public model card](https://huggingface.co/sd2-community/stable-diffusion-2-inpainting) documents image+mask conditioning and identifies this repository as a community mirror. This is a protocol-based feasibility selection, not evidence of utility. No new model/module downloaded or trained.
6. Stage20B authorization: `{ready}`, not auto-started. 40 Flash + 40 Black budget unchanged. Full slots, generator training hyperparameters, shared detector hyperparameters, base-weight hashes and a structural mask/bbox/RNG preflight must be frozen before Stage20B execution. Raw samples only, no quality-based rejection, Top-K or best-of-N. Gates predeclared: mean >= +0.50 pp; strict wins >=2/3; each class mean >= -2.0 pp. No Stage17 nuisance threshold transferred.
7. TDCRG remains SPECIFICATION ONLY, unauthorized until all Stage20B gates pass. Five-fold OOF source-group manifests are definitions, not trained teachers. Future teacher only during generator training; inference detector-free; distribution mean/covariance rather than nearest-real; diversity diagnostics; unified learned class/timestep bounded residual gates; no BootstrapGuard until innovation is independently frozen.

Historical eval files were read ONLY for the explicitly requested registry, provenance and split counts. No evaluation forward, performance tuning, candidate selection, synthetic generation, training, optimizer update, DeepPCB or BootstrapGuard activity occurred. Original images, labels, splits and historical manifests were not changed.
"""
    (out/"STAGE20A_REPORT.md").write_text(report,encoding="utf-8",newline="\n")
    print(json.dumps({"status":status["status"],"gates":checks,"splits":class_audit,"train_normals":sum(x["confirmatory_allowed"] for x in normals),"STAGE20B_RAW_GENERATOR_BASELINE_AUTHORIZED":ready},indent=2))


if __name__=="__main__":main()
