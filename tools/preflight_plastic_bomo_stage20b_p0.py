"""CPU metadata/geometry/asset preflight ONLY; no torch imports, sampling or training."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys
from PIL import Image

OUTPUTS = ["stage20b_stage20a_binding_audit", "stage20b_split_recheck", "sd2_base_weight_binding",
           "stage20b_generator_training_protocol", "generator_train_allowlist", "stage20b_normal_pool_freeze",
           "stage20b_generation_slots", "stage20b_geometry_audit", "stage20b_rng_manifest",
           "stage20b_bank_pairing_audit", "stage20b_generation_protocol", "stage20b_realrepeat_manifest",
           "detector_initialization_binding", "stage20b_detector_protocol", "stage20b_training_order_spec"]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for x in iter(lambda:f.read(1024*1024),b""):h.update(x)
    return h.hexdigest()


def htext(s):return hashlib.sha256(s.encode()).hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False)
def load(p):return json.loads(Path(p).read_text(encoding="utf-8-sig"))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+"\n",encoding="utf-8",newline="\n")
def portable_name(path):return str(path).replace("\\","/").rstrip("/").rsplit("/",1)[-1]


class Stop(Exception):
    def __init__(self,status,detail):self.status=status;self.detail=detail


def require(ok,status,detail):
    if not ok:raise Stop(status,detail)


def resolve_detector_arguments(defaults, detector):
    custom={"architecture","primary_checkpoint","best_pt_selection","early_stopping","final_eval_during_or_at_train_end",
            "device_count","TF32","cudnn_benchmark","CUBLAS_WORKSPACE_CONFIG","gradient_accumulation","accumulation_rule",
            "max_grad_norm","augmentation_rng","runtime_version_policy","unsupported_args_policy"}
    native=set(detector)-custom
    require(native<=set(defaults),"DETECTOR_PROTOCOL_NOT_FULLY_FROZEN",{"unsupported_frozen_native_arguments":sorted(native-set(defaults))})
    result=dict(defaults);result.update({k:detector[k] for k in native})
    return result,sorted(native)


def geometry(ann, image, cfg):
    """Continuous exact bbox translation + independently rasterized weak mask."""
    w,h=image["width"],image["height"]; cx,cy,bw,bh=ann["bbox"]
    s=cfg["crop_side"]; left=math.floor(cx*w-s/2);top=math.floor(cy*h-s/2)
    original=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
    target=[original[0]-left,original[1]-top,original[2]-left,original[3]-top]
    valid=all(math.isfinite(v) for v in original+target) and bw>0 and bh>0 and min(original)>=0 and original[2]<=w and original[3]<=h
    valid=valid and target[0]>=0 and target[1]>=0 and target[2]<=s and target[3]<=s
    if not valid:return {"valid":False,"reason":"TARGET_CROPPED_OR_INVALID","original_bbox_xyxy":original,"fixed_crop_xyxy":[left,top,left+s,top+s]}
    box=[math.floor(target[0]),math.floor(target[1]),math.ceil(target[2]),math.ceil(target[3])]
    area=(box[2]-box[0])*(box[3]-box[1])
    mask=Image.new("L",(s,s),0);mask.paste(255,tuple(box))
    maskhash=hashlib.sha256(json.dumps([s,s]).encode()+mask.tobytes()).hexdigest()
    return {"valid":area>0,"original_canvas_wh":[w,h],"original_bbox_xyxy":original,
        "fixed_crop_xyxy":[left,top,left+s,top+s],"generator_canvas_wh":[s,s],"resize_scale_xy":[1,1],
        "generator_bbox_xyxy":target,"final_canvas_wh":[w,h],"final_bbox_normalized_xywh":ann["bbox"],
        "rasterized_mask_xyxy_exclusive":box,"nonzero_mask_pixels":area,"mask_decoded_sha256":maskhash,
        "mask_is_human_segmentation":False,"crop_shifted_to_rescue":False,
        "paste_valid_crop_region_xyxy":[max(0,left),max(0,top),min(w,left+s),min(h,top+s)],
        "inverse_transform":"x_final=x_generator+crop_left; y_final=y_generator+crop_top"}


def choose_donors(anns, by, cfg, budget):
    chosen=[];excluded=[]
    for cls,count in budget.items():
        grouped=defaultdict(list)
        for x in anns:
            if x["class"]==cls:grouped[x["source_group_id"]].append(x)
        key=lambda x:(htext(cfg["protocol"]+cls+x["source_group_id"]+x["annotation_id"]),x["annotation_id"])
        queues={g:sorted(z,key=key) for g,z in grouped.items()}
        groups=sorted(queues,key=lambda g:(key(queues[g][0]),g));n=0;round_id=0
        while n<count and any(round_id<len(queues[g]) for g in groups):
            for g in groups:
                if n==count:break
                if round_id>=len(queues[g]):continue
                x=queues[g][round_id];geom=geometry(x,by[x["node_id"]],cfg["geometry"])
                if not geom["valid"]:excluded.append({"annotation_id":x["annotation_id"],"class":cls,"geometry":geom});continue
                chosen.append((x,geom));n+=1
            round_id+=1
        require(n==count,"INSUFFICIENT_STRUCTURALLY_VALID_SLOTS",{"class":cls,"selected":n,"required":count,"excluded":excluded})
    return chosen,excluded


def model_binding(base, public):
    rows={};errors=[]
    for name,reference in public["weight_files"].items():
        f=base/name; actual=sha(f) if f.is_file() else None
        rows[name]={"path":str(f),"sha256":actual,"reference_sha256":reference["sha256"],"size":f.stat().st_size if f.is_file() else None}
        if actual!=reference["sha256"]:errors.append("PUBLIC_WEIGHT_MISMATCH:"+name)
    for name,blob in public["metadata_git_blobs"].items():
        f=base/name;raw=f.read_bytes() if f.is_file() else None
        gitblob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest() if raw is not None else None
        rows[name]={"path":str(f),"sha256":hashlib.sha256(raw).hexdigest() if raw is not None else None,"public_git_blob":blob,"actual_git_blob":gitblob}
        if gitblob!=blob:errors.append("PUBLIC_CONFIG_MISMATCH:"+name)
    # Never let from_pretrained silently pick an unbound alternate variant.
    for component in ("unet","vae","text_encoder"):
        weightnames=[f.relative_to(base).as_posix() for f in (base/component).glob("*") if f.suffix in (".bin",".safetensors")]
        if set(weightnames)!={n for n in public["weight_files"] if n.startswith(component+"/")}:
            errors.append("AMBIGUOUS_WEIGHT_VARIANTS:"+component)
    if (base/"unet/config.json").is_file() and load(base/"unet/config.json").get("in_channels")!=9:errors.append("NOT_NINE_CHANNEL_INPAINTING")
    return {"status":"PASS" if not errors else "SD2_BASE_WEIGHT_IDENTITY_UNRESOLVED","base_model":str(base),"public_reference":public,"files":rows,"errors":errors,"loaded_as_model":False}


def environment(repo):
    versions={};source={}
    for name in ("torch","ultralytics","transformers","accelerate","safetensors","Pillow","numpy","PyYAML"):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    try:
        dist=importlib.metadata.distribution("ultralytics")
        for f in ("ultralytics/cfg/default.yaml","ultralytics/engine/trainer.py","ultralytics/models/yolo/detect/train.py","ultralytics/utils/loss.py","ultralytics/utils/torch_utils.py"):
            path=dist.locate_file(f);source[f]={"path":str(path),"sha256":sha(path) if path.is_file() else None}
    except importlib.metadata.PackageNotFoundError:pass
    for f in ("train_dreambooth_noise.py","diffusers/__init__.py","diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_inpaint.py","diffusers/schedulers/scheduling_ddim.py"):
        path=repo/f;source[f]={"path":str(path),"sha256":sha(path) if path.is_file() else None}
    return {"versions":versions,"source_files":source,"python":sys.version,"platform":platform.platform(),"model_imported":False}


def run(a,cfg,out):
    repo=a.repo_root.resolve();root=a.stage20a;bound={};binding_errors=[]
    manifest=root/"frozen_artifact_manifest.json"
    require(manifest.is_file() and sha(manifest)==cfg["stage20a_frozen_manifest_sha256"],"STAGE20A_PROTOCOL_BINDING_FAILED","authoritative frozen manifest missing or modified")
    for fn,expected in load(manifest)["sha256"].items():
        f=root/fn;actual=sha(f) if f.is_file() else None;bound[fn]={"expected_sha256":expected,"actual_sha256":actual}
        if expected!=actual:binding_errors.append(fn)
    save(out/"stage20b_stage20a_binding_audit.json",{"status":"PASS" if not binding_errors else "STAGE20A_PROTOCOL_BINDING_FAILED","stage20a_commit":cfg["stage20a_commit"],"artifacts":bound,"stage20a_unchanged":True})
    require(not binding_errors,"STAGE20A_PROTOCOL_BINDING_FAILED",binding_errors)
    if a.scope=="SERVER_PREFLIGHT":
        local_manifest=a.local_frozen/"p0_frozen_artifact_manifest.json"
        require(local_manifest.is_file(),"GENERATION_SLOT_PAIRING_INVALID","upload complete frozen LOCAL P0 metadata first; server may not define different slots")
        for fn,expectedsha in load(local_manifest)["sha256"].items():
            path=a.local_frozen/fn
            require(path.is_file() and sha(path)==expectedsha,"GENERATION_SLOT_PAIRING_INVALID","local P0 frozen artifact changed:"+fn)
        local_status=load(a.local_frozen/"stage20b_p0_status.json")
        require(local_status.get("local_structural_preflight_pass") and local_status["protocol_sha256"]==sha(a.protocol),"DETECTOR_PROTOCOL_NOT_FULLY_FROZEN","uploaded P0 protocol must match frozen local config bytes")
    s=load(root/"stage20a_status.json");require(s["status"]=="PLASTIC_BOMO_GENERATION_V2_PROTOCOL_READY" and s["STAGE20B_RAW_GENERATOR_BASELINE_AUTHORIZED"],"STAGE20A_PROTOCOL_BINDING_FAILED","authoritative authorization absent")
    baseline=load(root/"v2_generator_baseline_selection.json")
    require(baseline["selected_family"]==cfg["generator_training"]["family"],"STAGE20A_PROTOCOL_BINDING_FAILED","baseline family changed")
    source=load(root/"real_source_registry.json")["records"];by={x["node_id"]:x for x in source}
    require(len(by)==len(source),"STAGE20A_PROTOCOL_BINDING_FAILED","duplicate nodes")
    assignments={x["node_id"]:x for x in load(root/"v2_split_manifest.json")["image_assignments"]}
    require(set(assignments)==set(by),"STAGE20A_PROTOCOL_BINDING_FAILED","registry/split node coverage mismatch")
    for x in source:
        y=assignments[x["node_id"]]
        require((x["split_v2"],x["known_source_group_id"],x["canonical_image_id"])==(y["split"],y["source_group_id"],y["canonical_image_id"]),"STAGE20A_PROTOCOL_BINDING_FAILED","registry/split schema inconsistent")
    ledger=load(root/"annotation_provenance_v2.json")["records"]
    groups={p:{r["known_source_group_id"] for r in source if r["split_v2"]==p} for p in ("train","final_eval")}
    rgb={p:{r["decoded_rgb_sha256"] for r in source if r["split_v2"]==p} for p in groups}
    cross=[e for e in load(root/"source_group_graph.json")["edges"] if by[e["from"]]["split_v2"]!=by[e["to"]]["split_v2"]]
    expected=cfg["expected"];actual={}
    for part,key in (("train","train"),("final_eval","final_eval")):
        actual[key+"_images"]=sum(x["asset_role"]=="detector" and x["split_v2"]==part for x in source)
        actual[key+"_boxes"]=dict(Counter(x["class"] for x in ledger if x["split_v2"]==part))
    ok=not groups["train"]&groups["final_eval"] and not rgb["train"]&rgb["final_eval"] and not cross
    ok=ok and all(actual[k]==expected[k] for k in actual)
    save(out/"stage20b_split_recheck.json",{"status":"PASS" if ok else "V2_SOURCE_ISOLATION_RECHECK_FAILED","counts":actual,"cross_source_groups":sorted(groups["train"]&groups["final_eval"]),"cross_rgb_aliases":sorted(rgb["train"]&rgb["final_eval"]),"cross_known_edges":cross,"physical_source_complete":False,"final_eval_content_read":0})
    require(ok,"V2_SOURCE_ISOLATION_RECHECK_FAILED",actual)
    train=[x for x in source if x["asset_role"]=="detector" and x["split_v2"]=="train"]
    anns=[x for x in ledger if x["split_v2"]=="train"]
    require(all(by[x["node_id"]]["split_v2"]=="train" and x["source_group_id"]==by[x["node_id"]]["known_source_group_id"] for x in anns),"GENERATOR_TRAIN_SPLIT_CONTAMINATION","annotation/source disagreement")
    normal_all=load(root/"normal_source_registry.json")["records"];normals=[x for x in normal_all if x["confirmatory_allowed"]]
    require(len(normals)==expected["train_normals"] and len(normal_all)-len(normals)==expected["excluded_normals"] and len({x["normal_id"] for x in normals})==len(normals),"TRAIN_NORMAL_POOL_INVALID","normal counts/IDs differ")
    require(all(x["split"]=="train" and x["status"]=="NORMAL_SOURCE_RESOLVED" and x["known_source_group"] in groups["train"] for x in normals),"TRAIN_NORMAL_POOL_INVALID","non-TRAIN/unknown normal")
    asset_requests={};mapped={};labelpaths={}
    for x in train:
        f=a.real_root/"images"/x["split_membership_old"]/portable_name(x["current_path"])
        lp=a.real_root/"labels"/x["split_membership_old"]/(f.stem+".txt")
        mapped[x["node_id"]]=str(f);labelpaths[x["node_id"]]=str(lp)
        asset_requests[str(f)]=x["file_sha256"]
        hashes={t["label_sha256"] for t in anns if t["node_id"]==x["node_id"]}
        require(len(hashes)==1,"GENERATOR_TRAIN_SPLIT_CONTAMINATION","ambiguous labels")
        asset_requests[str(lp)]=next(iter(hashes))
    for x in normals:asset_requests[str(a.normal_root/portable_name(x["path"]))]=x["sha256"]
    def check(item):
        f,expectedsha=item;f=Path(f)
        return None if f.is_file() and sha(f)==expectedsha else str(f)
    with ThreadPoolExecutor(max_workers=4) as pool:bad=[x for x in pool.map(check,asset_requests.items()) if x]
    require(not bad,"GENERATOR_TRAIN_SPLIT_CONTAMINATION",{"missing_or_changed_TRAIN_asset_files":bad})
    # Hash-bound metadata supplies dimensions; no evaluation image is opened.
    inputrows=[];excluded=[]
    for x in anns:
        r=by[x["node_id"]];g=geometry(x,r,cfg["geometry"])
        if not g["valid"]:excluded.append({"annotation_id":x["annotation_id"],"reason":g});continue
        inputrows.append({"annotation_id":x["annotation_id"],"canonical_image_id":x["canonical_image_id"],"source_group_id":x["source_group_id"],"split":"train","path":mapped[x["node_id"]],"sha256":r["file_sha256"],"class":x["class"],"annotation_ids":[x["annotation_id"]],"full_label_path":labelpaths[x["node_id"]],"full_label_sha256":x["label_sha256"],"geometry":g})
    save(out/"generator_train_allowlist.json",{"status":"PASS","inputs":inputrows,"counts":dict(Counter(x["class"] for x in inputrows)),"structural_exclusions":excluded,"final_eval_input_count":0,"unknown_split_input_count":0,"quality_filter_count":0,"source_assets_hash_verified":len(asset_requests)})
    normalrows=[{**x,"runtime_path":str(a.normal_root/portable_name(x["path"]))} for x in normals]
    save(out/"stage20b_normal_pool_freeze.json",{"status":"PASS","records":normalrows,"count":len(normals),"excluded_count":len(normal_all)-len(normals),"pool_changed":False,"brightness_or_quality_filter":False})
    training={**cfg["generator_training"],"base_model":str(a.base_model),"allowlist_sha256":sha(out/"generator_train_allowlist.json"),"mask_crop_policy":cfg["geometry"],"entrypoint":"train_dreambooth_noise.py, pinned by Stage20A baseline code evidence; future manifest-only launch wrapper, not launched at P0"}
    generation={**cfg["generation"],"prompts":cfg["generator_training"]["prompts"],"geometry_policy":cfg["geometry"],"raw_output_policy":cfg["raw_output_policy"]}
    save(out/"stage20b_generator_training_protocol.json",training);save(out/"stage20b_generation_protocol.json",generation)
    chosen,rejected=choose_donors(anns,by,cfg,cfg["budget"]);slots=[];rr=[]
    for ann,g in chosen:
        r=by[ann["node_id"]];sid=ann["class"]+"_"+ann["annotation_id"]
        ordered=sorted(normals,key=lambda n:(htext(sid+n["normal_id"]),n["normal_id"]))
        candidates=[n for n in ordered if [by[n["node_id"]]["width"],by[n["node_id"]]["height"]]==g["final_canvas_wh"]]
        require(bool(candidates),"TRAIN_NORMAL_POOL_INVALID",{"slot":sid,"required_canvas":g["final_canvas_wh"]})
        normal=candidates[0]
        slot={"slot_id":sid,"class":ann["class"],"annotation_id":ann["annotation_id"],"canonical_image_id":ann["canonical_image_id"],"node_id":ann["node_id"],"source_group_id":ann["source_group_id"],"split":"train","bbox":ann["bbox"],"source_path":mapped[ann["node_id"]],"source_sha256":r["file_sha256"],"conditioning_normal_id":normal["normal_id"],"normal_path":str(a.normal_root/portable_name(normal["path"])),"normal_sha256":normal["sha256"],"normal_source_group":normal["known_source_group"],"geometry":g,"mask_specification":cfg["geometry"]["mask"],"prompt":cfg["generator_training"]["prompts"][ann["class"]],"checkpoint_role":ann["class"]+"_V2_final_step2000","generation_protocol_sha256":sha(out/"stage20b_generation_protocol.json")}
        slots.append(slot)
        full=[x for x in ledger if x["node_id"]==ann["node_id"]]
        rr.append({"slot_id":sid,"class_role":ann["class"],"canonical_image_id":r["canonical_image_id"],"source_group_id":r["known_source_group_id"],"full_image_path":mapped[r["node_id"]],"image_sha256":r["file_sha256"],"full_labels_path":labelpaths[r["node_id"]],"label_sha256":ann["label_sha256"],"full_annotations":[{"annotation_id":x["annotation_id"],"class":x["class"],"bbox":x["bbox"]} for x in full],"bbox_crop_replay":False,"split":"train"})
    counts=dict(Counter(x["class"] for x in slots));require(counts==cfg["budget"],"INSUFFICIENT_STRUCTURALLY_VALID_SLOTS",counts)
    if a.scope=="SERVER_PREFLIGHT":
        local_slots=load(a.local_frozen/"stage20b_generation_slots.json")["slots"]
        without_paths=lambda x:{k:v for k,v in x.items() if k not in ("source_path","normal_path")}
        require([without_paths(x) for x in slots]==[without_paths(x) for x in local_slots],"GENERATION_SLOT_PAIRING_INVALID","server slot/normal/geometry/prompt metadata differs from local freeze")
        save(out/"stage20b_local_p0_binding_audit.json",{"status":"PASS","local_manifest_sha256":sha(a.local_frozen/"p0_frozen_artifact_manifest.json"),"config_bytes_identical":True,"all_80_slot_semantics_identical":True,"path_relocation_only":True})
    save(out/"stage20b_generation_slots.json",{"status":"PASS","frozen":True,"slots":slots,"counts":counts,"distinct_annotations":len({x["annotation_id"] for x in slots}),"groups_per_class":{c:len({x["source_group_id"] for x in slots if x["class"]==c}) for c in counts},"max_group_reuse":max(Counter(x["source_group_id"] for x in slots).values()),"post_freeze_replacement":False,"structural_candidates_skipped":rejected})
    save(out/"stage20b_geometry_audit.json",{"status":"PASS","records":[{"slot_id":x["slot_id"],**x["geometry"]} for x in slots],"valid_count":80,"mask_is_segmentation_ground_truth":False,"crop_reposition_count":0,"generated_images":0})
    reuse=Counter(x["canonical_image_id"] for x in rr)
    save(out/"stage20b_realrepeat_manifest.json",{"records":rr,"replay_exposures":80,"unique_donor_images":len(reuse),"unique_source_groups":len({x["source_group_id"] for x in rr}),"max_donor_reuse":max(reuse.values()),"class_role_counts":counts,"not_80_new_independent_images":True,"replay_counts":dict(reuse)})
    rng=[];paired=[]
    for slot in slots:
        metadata_hash=htext(canonical(slot));seedrows={}
        for bank,seed in cfg["banks"].items():
            sample_seed=int(htext("Plastic_Bomo_Generation_V2"+slot["slot_id"]+str(seed))[:16],16)
            seedrows[bank]={"bank_seed":seed,"sample_seed":sample_seed,"non_rng_metadata_sha256":metadata_hash}
        require(len({x["sample_seed"] for x in seedrows.values()})==3,"GENERATION_SLOT_PAIRING_INVALID",slot["slot_id"])
        rng.append({"slot_id":slot["slot_id"],"banks":seedrows});paired.append({"slot_id":slot["slot_id"],"non_rng_metadata_sha256":metadata_hash,"all_non_rng_fields_identical":True,"only_RNG_varies":True})
    save(out/"stage20b_rng_manifest.json",{"records":rng,"seed_rule":cfg["sample_seed_rule"],"torch_Generator_instantiated":False,"generation_count":0})
    save(out/"stage20b_bank_pairing_audit.json",{"status":"PASS","records":paired,"fields_checked":["source","class","bbox","mask","normal","prompt","checkpoint_role","scheduler","steps","CFG","preprocessing"],"actual_trained_checkpoint_binding":"PENDING_TRAINING; MUST hash-bind one class checkpoint shared across banks before any sampling","metadata_only":True})
    binding=model_binding(a.base_model,cfg["base_public_reference"]);save(out/"sd2_base_weight_binding.json",binding)
    require(not binding["errors"],"SD2_BASE_WEIGHT_IDENTITY_UNRESOLVED",binding["errors"])
    detector_sha=sha(a.detector_model) if a.detector_model.is_file() else None
    save(out/"detector_initialization_binding.json",{"status":"PASS" if detector_sha==cfg["detector_weight_sha256"] else "DETECTOR_INITIALIZATION_UNRESOLVED","model":str(a.detector_model),"sha256":detector_sha,"expected_sha256":cfg["detector_weight_sha256"],"evidence":cfg["detector_weight_evidence"],"same_for_all_arms":True,"model_loaded":False})
    require(detector_sha==cfg["detector_weight_sha256"],"DETECTOR_INITIALIZATION_UNRESOLVED",detector_sha)
    env=environment(repo)
    codebound=[]
    for code in baseline["pinned_code"]:
        actualsha=sha(repo/code["path"]) if (repo/code["path"]).is_file() else None
        codebound.append({**code,"actual_sha256":actualsha})
    require(all(x["actual_sha256"]==x["sha256"] for x in codebound),"DETECTOR_PROTOCOL_NOT_FULLY_FROZEN","pinned generator code differs from Stage20A")
    save(out/"stage20b_runtime_binding.json",{"environment":env,"generator_code":codebound,"scope":a.scope,"imports_models":False})
    if a.scope=="SERVER_PREFLIGHT":
        require(all(env["versions"][k] for k in ("torch","ultralytics","transformers","accelerate","safetensors","numpy","PyYAML")),"DETECTOR_PROTOCOL_NOT_FULLY_FROZEN","server runtime package metadata incomplete")
        require(bool(env["source_files"].get("ultralytics/engine/trainer.py",{}).get("sha256")),"DETECTOR_PROTOCOL_NOT_FULLY_FROZEN","server trainer source unavailable")
        import yaml
        default_path=Path(env["source_files"]["ultralytics/cfg/default.yaml"]["path"])
        resolved_args,override=resolve_detector_arguments(yaml.safe_load(default_path.read_text(encoding="utf-8")),cfg["detector"])
        resolved_args.update({"model":str(a.detector_model),"data":"ARM_SPECIFIC_V2_TRAIN_DATA_YAML_ONLY","task":"detect","mode":"train","device":"ONE_FROZEN_CUDA_DEVICE_PER_RUN","project":"NEW_STAGE20B_RUN_DIRECTORY","name":"ARM_SEED42_ROLE","exist_ok":False})
        require(resolved_args.get("distill_model") is None,"DETECTOR_PROTOCOL_NOT_FULLY_FROZEN","unexpected detector teacher default")
        save(out/"stage20b_detector_resolved_runtime_arguments.json",{"arguments":resolved_args,"all_remaining_defaults_materialized_and_frozen":True,"default_yaml_sha256":sha(default_path),"protocol_override_fields":sorted(override),"custom_enforced_rules":{k:cfg["detector"][k] for k in ("gradient_accumulation","accumulation_rule","early_stopping","final_eval_during_or_at_train_end","augmentation_rng","max_grad_norm")}})
    save(out/"stage20b_detector_protocol.json",{"status":"PASS","training":cfg["detector"],"evaluation":cfg["detector_evaluation"],"initialization_sha256":detector_sha,"runtime_binding_sha256":sha(out/"stage20b_runtime_binding.json"),"arms":["RR","RawGenerator-A","RawGenerator-B","RawGenerator-C"],"same_real_base_count":136,"extra_roles":80,"early_stopping_disabled":True,"all_last_pt_frozen_before_first_eval":True,"enforcement":"Future custom trainer must enforce scheduled optimizer attempts, per-draw RNG, 150 epochs and NO final_eval() call; no trainer implemented or launched at P0"})
    roles=["base_"+x["node_id"] for x in sorted(train,key=lambda x:x["node_id"])]+["extra_"+x["slot_id"] for x in slots]
    epochs=[]
    for epoch in range(150):
        sequence=sorted(roles,key=lambda r:(htext(cfg["protocol"]+"42"+str(epoch)+r),r))
        augment=[int(htext(cfg["protocol"]+"augmentation42"+str(epoch)+str(pos))[:16],16) for pos in range(216)]
        epochs.append({"epoch_0based":epoch,"role_sequence":sequence,"sequence_sha256":htext(canonical(sequence)),"augmentation_seeds":augment,"optimizer_attempt_after_positions_0based":[63,127,191,215],"accumulation_group_lengths":[64,64,64,24]})
    save(out/"stage20b_training_order_spec.json",{"spec":cfg["schedule"],"epochs":epochs,"scheduled_draws":32400,"scheduled_batches":32400,"scheduled_optimizer_attempts":600,"successful_updates_equal_required":False,"AMP_skips_logged_separately":True,"sample_order_shared_across_arms":True,"mosaic_mixup_cutmix_copy_paste_disabled_for_role_integrity":True,"augmentation_seeds_per_position_shared":True,"native_prefetch_or_sampler_iteration_not_proof_of_consumed_exposure":True})
    save(out/"stage20b_future_baseline_gate.json",load(root/"future_baseline_gate.json"))
    require(sha(manifest)==cfg["stage20a_frozen_manifest_sha256"],"STAGE20A_PROTOCOL_BINDING_FAILED","Stage20A changed during P0")
    return {"scope":a.scope,"status":"STAGE20B_STRUCTURAL_PREFLIGHT_PASS" if a.scope=="SERVER_PREFLIGHT" else "SERVER_ASSET_BINDING_PENDING","STAGE20B_EXECUTION_AUTHORIZED":a.scope=="SERVER_PREFLIGHT","local_structural_preflight_pass":True,"TDCRG_DEVELOPMENT_AUTHORIZED":False,"slots":80,"TRAIN_normals":394,"TRAIN_assets_hash_verified":len(asset_requests),"replay_unique_images":len(reuse),"max_replay_reuse":max(reuse.values()),"group_reuse_max":max(Counter(x["source_group_id"] for x in slots).values()),"generator_training_input_counts":dict(Counter(x["class"] for x in inputrows)),"runtime_preflight_is_not_forward_test":True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("stage20a","real_root","normal_root","base_model","detector_model","output"):p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--repo_root",type=Path,default=Path("."));p.add_argument("--protocol",type=Path,default=Path("configs/plastic_bomo_stage20b_p0.json"))
    p.add_argument("--scope",choices=["LOCAL_PREFLIGHT","SERVER_PREFLIGHT"],default="LOCAL_PREFLIGHT")
    p.add_argument("--local_frozen",type=Path,default=Path("results_for_gpt/plastic_bomo_stage20b_p0_preflight"))
    a=p.parse_args();cfg=load(a.protocol);out=a.output
    if out.exists():raise RuntimeError("P0_OUTPUT_EXISTS_USE_NEW_AUDIT_DIRECTORY_NO_OVERWRITE:"+str(out))
    out.mkdir(parents=True)
    try:status=run(a,cfg,out)
    except Stop as e:status={"status":e.status,"scope":a.scope,"detail":e.detail,"STAGE20B_EXECUTION_AUTHORIZED":False,"TDCRG_DEVELOPMENT_AUTHORIZED":False}
    status.update({"generator_training_count":0,"synthetic_generation_count":0,"detector_training_count":0,"official_final_eval_forward":0,"final_eval_image_or_label_content_read":0,"DeepPCB":0,"BootstrapGuard":0,"TDCRG_implementation":0,"Stage20A_modified":False,"stage20b_auto_start":False,"protocol_sha256":sha(a.protocol),"script_sha256":sha(Path(__file__))})
    for name in OUTPUTS:
        if not (out/(name+".json")).exists():save(out/(name+".json"),{"status":"NOT_RUN","blocked_by":status["status"]})
    save(out/"stage20b_p0_status.json",status)
    report=f"""# Stage20B-P0 — Raw Generator Baseline Structural Preflight

Status: `{status['status']}`. Scope: `{a.scope}`. Execution authorized: `{status['STAGE20B_EXECUTION_AUTHORIZED']}`. NO training, sampling, model loading or final_eval forward occurred. This is not Stage17 rescue, selector/task-guidance/morphology or BootstrapGuard work.

Stage20A is bound to commit aa183e564a12c4d33d1283616921f370e4ba655d and its original frozen SHA256 ledger, unchanged. All split/slot decisions use V2 TRAIN metadata only. The final_eval files themselves are not opened at P0. Only no **KNOWN** source-group leakage is supported; `physical_source_complete=false`.

Annotations remain dataset-defined supervision, not universally human-verified physical ground truth. Rectangular bbox masks are weak spatial geometry, not segmentation ground truth. Historical provenance and the two clipping issues remain unchanged.

V2 train/final_eval differ from Stage14-18 evaluation protocols. Absolute mAP comparisons across those protocols are forbidden; future interpretation is RR vs raw banks within V2 only. Stage17 nuisance SD/threshold are not transferred.

The protocols specify one new class-specific full-UNet SD2 baseline from the hash-matched public original initialization, 2000 successful updates/class, deterministic 512 crops, no normal-only training and no new loss. A crop that truncates its target is structurally rejected in the predeclared ordering before slot freeze, never moved to rescue it. No mask/crop training images have been exported or synthesized at P0.

80 slots (40/40) use source-group round-robin and stable hashes, without quality metrics. A/B/C differ only in generation RNG; normals, geometry, prompt, class checkpoint ROLE, scheduler and preprocessing are shared. The future trained checkpoints do not exist yet: their hashes MUST be shared and frozen across banks before sampling. All successful raw outputs must be used. Structural failures after freeze stop the comparison, with no reroll, replacement or budget reduction.

RR is 80 replay exposures of full donor images and complete labels, NOT 80 independent new real images. The real base is identical 136 images. Training roles and per-position augmentation seeds are shared. YOLO11s uses fixed explicit AdamW and no composite augmentation, 150 epochs, batch1, imgsz1536, accumulation64 with a 24-draw epoch tail: 32400 draws/batches and 600 scheduled optimizer attempts. This is a predeclared V2 budget, not a retroactive adjustment. Successful AMP updates may differ and must be reported. The future trainer must explicitly suppress early stopping AND its automatic final evaluation; val=False alone is not sufficient.

All four epoch150 last.pt checkpoints must be frozen before any unified V2 final_eval evaluation. Gates remain mean bank-RR >=0.50pp, at least2 strict bank wins, each class mean >=-2.0pp. Failure stops with no automatic generator/CFG/steps/prompt/bank/selector fallback. TDCRG remains unauthorized.

Local results are NOT server weight/runtime verification. A server P0 run must hash-bind the same Stage20A assets, public SD2 weights, pretrained detector, code and actual runtime package/source files. Missing or changed assets are stop gates, not invitations to replace data. P0 does not test GPU numerical determinism or differentiation: these are runtime checks to enforce before the first future formal launch, never evidence claimed here. No model weights were downloaded. Public reference metadata: https://huggingface.co/sd2-community/stable-diffusion-2-inpainting/tree/5f74973cbb64c8568780732c17f43eb269d63a0d
"""
    (out/"STAGE20B_P0_REPORT.md").write_text(report,encoding="utf-8",newline="\n")
    save(out/"p0_frozen_artifact_manifest.json",{"sha256":{x.relative_to(out).as_posix():sha(x) for x in sorted(out.rglob("*")) if x.is_file()},"no_overwrite":True,"scope":a.scope})
    print(json.dumps(status,indent=2,ensure_ascii=False))
    if status["status"] not in ("STAGE20B_STRUCTURAL_PREFLIGHT_PASS","SERVER_ASSET_BINDING_PENDING"):raise SystemExit(2)


if __name__=="__main__":main()
