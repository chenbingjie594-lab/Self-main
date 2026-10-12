"""Stage22A: read-only TRAIN-only diagnostics at one frozen public YOLO11s initialization.

The CPU preflight is intentionally runnable before GPU execution. It creates no
dataset, checkpoint, synthetic image, or detector run. The GPU path refuses to
overwrite an already completed Stage22A audit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

import numpy as np

from stage21b_a1_common import guard_for, load, sha


REPO = Path(__file__).resolve().parents[1]
CLASSES = ("flash", "black")
BANKS = ("A", "B", "C")


def fail(message):
    raise RuntimeError("STAGE22A_" + message)


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def ordered(rows, key):
    return sorted(rows, key=lambda row: (hashlib.sha256(str(row[key]).encode()).hexdigest(), str(row[key])))


def summary(values):
    a = np.asarray(list(values), dtype=np.float64)
    if not len(a):
        return {"n": 0}
    if not np.isfinite(a).all():
        fail("NONFINITE_METRIC")
    q = np.quantile(a, [0, .25, .5, .75, .9, .95, 1])
    return {"n": int(len(a)), "mean": float(a.mean()), "median": float(q[2]),
            "std": float(a.std(ddof=0)), "iqr": float(q[3]-q[1]),
            "q0": float(q[0]), "q25": float(q[1]), "q75": float(q[3]),
            "q90": float(q[4]), "q95": float(q[5]), "q100": float(q[6])}


def cosine(a, b):
    aa, bb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return None if aa == 0 or bb == 0 else float(np.dot(a, b) / (aa * bb))


def unit(a):
    n = float(np.linalg.norm(a))
    return a / n if n else a.copy()


def protocol_and_manifest(protocol_path, run, verify_files):
    cfg = load(protocol_path)
    if cfg["stage"] != "22A" or cfg["banks"] != list(BANKS) or cfg["real_count"] != 109 or cfg["synthetic_slots_per_bank"] != 61:
        fail("PROTOCOL_CHANGED")
    prior = load(run / "stage21b_status.json")
    if prior["status"] != "SYNTHETIC_EXPOSURE_TIMING_PREMISE_NOT_SUPPORTED" or prior["phase"] != "COMPLETE" or prior["internal_train_images"] != 109 or prior["synthetic_total"] != 61:
        fail("HISTORICAL_STAGE21B_STATUS_CHANGED")
    split = load(run / "stage21b_split_manifest.json")
    real = ordered([x for x in split["records"] if x["internal_split"] == "internal_train"], "role_id")
    slots = ordered(load(run / "stage21b_synthetic_allowlist_v2.json")["records"], "slot_id")
    if len(real) != 109 or len(slots) != 61 or Counter(x["class"] for x in slots) != cfg["synthetic_classes"]:
        fail("SOURCE_POPULATION_CHANGED")
    if len({x["role_id"] for x in real}) != 109 or len({x["slot_id"] for x in slots}) != 61:
        fail("DUPLICATE_ROLE_OR_SLOT")
    if any(not set(x["class_instances"]).issubset(CLASSES) or not x["class_instances"] for x in real):
        fail("REAL_CLASS_MEMBERSHIP_INVALID")
    real_ids = {x["role_id"] for x in real}
    slot_ids = {x["role_id"] for x in slots}
    bindings = {}
    for bank in BANKS:
        binding = load(run / "train_inputs" / bank / "role_binding.json")["roles"]
        if set(binding) != real_ids | slot_ids or len(binding) != 170:
            fail("ROLE_BINDING_CHANGED_" + bank)
        for row in real:
            b = binding[row["role_id"]]
            if b["image_sha256"] != row["image_sha256"] or b["label_sha256"] != row["label_sha256"]:
                fail("REAL_BINDING_CHANGED")
        for row in slots:
            b, asset = binding[row["role_id"]], row["banks"][bank]
            if b["image_sha256"] != asset["image_sha256"] or b["label_sha256"] != asset["label_sha256"]:
                fail("SYNTHETIC_BINDING_CHANGED")
        bindings[bank] = binding
    for row in slots:
        if len({row["banks"][bank]["label_sha256"] for bank in BANKS}) != 1:
            fail("MATCHED_SLOT_LABELS_DIFFER")
        if len({row["banks"][bank]["sample_seed"] for bank in BANKS}) != 3:
            fail("MATCHED_SLOT_RNG_NOT_DISTINCT")
    checked = set()
    for bank, binding in bindings.items():
        for role_id, row in binding.items():
            image, label = ((run / row[k]).resolve() for k in ("local_image", "local_label"))
            for path, expected in ((image, row["image_sha256"]), (label, row["label_sha256"])):
                if not path.is_relative_to(run.resolve()) or any(p in ("internal_dev_inputs", "final_eval_inputs") for p in path.parts):
                    fail("FORBIDDEN_SOURCE_PATH")
                if not path.is_file():
                    fail("TRAIN_INPUT_MISSING_" + bank + "_" + role_id)
                if verify_files and path not in checked:
                    if sha(path) != expected:
                        fail("TRAIN_ASSET_HASH_CHANGED_" + bank + "_" + role_id)
                    checked.add(path)
    manifest = {"stage21b_run": str(run.resolve()), "stage21b_split_manifest_sha256": sha(run / "stage21b_split_manifest.json"),
                "stage21b_synthetic_allowlist_sha256": sha(run / "stage21b_synthetic_allowlist_v2.json"),
                "role_binding_sha256": {b: sha(run / "train_inputs" / b / "role_binding.json") for b in BANKS},
                "real_order_method": "sha256(role_id), all 109", "real_role_ids": [x["role_id"] for x in real],
                "real_class_memberships": {x["role_id"]: sorted(set(x["class_instances"])) for x in real},
                "real_annotation_instances": {x["role_id"]: x["class_instances"] for x in real},
                "real_images_by_membership": {c: sum(c in x["class_instances"] for x in real) for c in CLASSES},
                "slot_order_method": "sha256(slot_id), all 61", "synthetic_slots": [
                    {"slot_id": x["slot_id"], "role_id": x["role_id"], "class": x["class"],
                     "source_group_id": x["source_group_id"], "conditioning_normal_id": x["conditioning_normal_id"],
                     "banks": {b: {"image_sha256": x["banks"][b]["image_sha256"],
                                   "label_sha256": x["banks"][b]["label_sha256"],
                                   "sample_seed": x["banks"][b]["sample_seed"]} for b in BANKS}} for x in slots],
                "same_slot_geometry_and_label_across_banks": True, "all_assets_sha256_verified": verify_files,
                "internal_dev_images": 0, "official_final_eval_images": 0}
    return cfg, manifest, real, slots, bindings


def batch_from_bound_asset(path, label, device, image_size, native_images, guard):
    import cv2
    import torch
    image = path.resolve()
    if str(image) not in native_images:
        fail("IMAGE_NOT_ALLOWLISTED")
    original = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if original is None:
        fail("IMAGE_DECODE_FAILED")
    h, w = original.shape[:2]
    ratio = min(image_size / h, image_size / w)
    nw, nh = round(w * ratio), round(h * ratio)
    dw, dh = image_size - nw, image_size - nh
    left, top = dw // 2, dh // 2
    resized = cv2.resize(original, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = cv2.copyMakeBorder(resized, top, dh-top, left, dw-left, cv2.BORDER_CONSTANT, value=(114,114,114))
    boxes, classes = [], []
    with label.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                fail("LABEL_FORMAT_INVALID")
            cls, xc, yc, bw, bh = map(float, parts)
            if cls not in (0., 1.) or not all(math.isfinite(v) for v in (xc,yc,bw,bh)) or bw <= 0 or bh <= 0:
                fail("LABEL_VALUE_INVALID")
            box = [(xc*w*ratio+left)/image_size, (yc*h*ratio+top)/image_size,
                   (bw*w*ratio)/image_size, (bh*h*ratio)/image_size]
            if box[0]-box[2]/2 < -1e-5 or box[0]+box[2]/2 > 1+1e-5 or box[1]-box[3]/2 < -1e-5 or box[1]+box[3]/2 > 1+1e-5:
                fail("LABEL_OUTSIDE_CANVAS")
            boxes.append(box); classes.append(int(cls))
    if not boxes:
        fail("EMPTY_TARGET")
    rgb = np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1))
    batch = {"img": torch.from_numpy(rgb[None]).to(device).float().div_(255),
             "cls": torch.tensor(classes, dtype=torch.float32, device=device).view(-1,1),
             "bboxes": torch.tensor(boxes, dtype=torch.float32, device=device),
             "batch_idx": torch.zeros(len(classes), dtype=torch.long, device=device)}
    return batch, classes, {"original_wh": [w,h], "resized_wh": [nw,nh], "padding_left_top": [left,top],
                            "annotation_instances": len(classes)}


def feature_roi_vectors(output, batch):
    import torch
    if not isinstance(output, torch.Tensor) or output.ndim != 4 or output.shape[0] != 1:
        fail("FEATURE_HOOK_TENSOR_INVALID")
    h, w = output.shape[-2:]
    vectors = defaultdict(list)
    for cls, box in zip(batch["cls"].reshape(-1), batch["bboxes"]):
        xc,yc,bw,bh = (float(v) for v in box)
        x0 = max(0, min(w-1, math.floor((xc-bw/2)*w)))
        x1 = max(x0+1, min(w, math.ceil((xc+bw/2)*w)))
        y0 = max(0, min(h-1, math.floor((yc-bh/2)*h)))
        y1 = max(y0+1, min(h, math.ceil((yc+bh/2)*h)))
        vectors[CLASSES[int(cls)]].append(output[0,:,y0:y1,x0:x1].detach().float().mean(dim=(1,2)).cpu().numpy())
    return {c: np.mean(v, axis=0).astype(np.float32) for c,v in vectors.items()}


def feature_stats(vectors):
    a = np.stack(vectors).astype(np.float64)
    mean = a.mean(axis=0)
    centered = a - mean
    # Singular values of centered n x d matrix give the nonzero covariance eigenvalues.
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eig = singular**2 / max(len(a)-1, 1)
    trace = float(eig.sum())
    positive = eig[eig > 0]
    prob = positive/positive.sum() if trace > 0 else np.asarray([])
    rank = float(np.exp(-(prob*np.log(prob)).sum())) if len(prob) else 0.
    return {"n": int(len(a)), "mean": mean, "covariance_trace": trace, "effective_rank": rank}


def wilson_interval(positive, total):
    if total <= 0:
        fail("EMPTY_BINOMIAL_GROUP")
    p, z = positive/total, 1.959963984540054
    denominator = 1+z*z/total
    center = (p+z*z/(2*total))/denominator
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denominator
    return [float(center-half),float(center+half)]


def run_gpu(cfg, manifest, real, slots, bindings, run, output, device):
    if (output / "stage22a_status.json").exists():
        fail("COMPLETED_OUTPUT_EXISTS")
    b = load(run / "execution_binding.json")
    b["device"] = "cuda:" + str(device)
    if b["device"] != load(run / "execution_binding.json")["device"]:
        fail("FROZEN_DEVICE_BINDING_CHANGED")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
        YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true", NO_ALBUMENTATIONS_UPDATE="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8", TOKENIZERS_PARALLELISM="false")
    import torch
    from execute_plastic_bomo_stage21b_a1 import make_model
    from qualify_plastic_bomo_stage20b_r0 import network_forbidden, optimizer_steps_forbidden, parameter_hashes, set_gpu_runtime, tensor_sha
    set_gpu_runtime(torch, b["device"])
    guard = guard_for(run,b,allow_dev=False)
    native_images = {str((run / row["local_image"]).resolve()) for bank in BANKS for row in bindings[bank].values()}
    guard.native_images = native_images
    selected_names = cfg["selected_parameters"]
    counts = {"forbidden_optimizer_step_calls": 0}
    raw, vectors, features = [], [], defaultdict(list)
    with ExitStack() as stack:
        stack.enter_context(guard.python_guard())
        stack.enter_context(guard.native_guard())
        stack.enter_context(network_forbidden())
        stack.enter_context(optimizer_steps_forbidden(torch, counts))
        native, model, _, _, init = make_model(run,b,run / "train_inputs/A",torch)
        if init != load(run / "initialization_reference.json"):
            fail("PUBLIC_INITIALIZATION_CHANGED")
        if any(isinstance(module, torch.nn.modules.dropout._DropoutNd) for module in model.modules()):
            fail("STOCHASTIC_DROPOUT_IN_FROZEN_MODEL")
        native._model_train()
        model.train()
        params = dict(model.named_parameters())
        if any(name not in params for name in selected_names.values()):
            fail("SELECTED_PARAMETER_MISSING")
        if any(not params[name].requires_grad for name in selected_names.values()):
            fail("SELECTED_PARAMETER_FROZEN")
        model_binding = {"model_source": "frozen public pretrained/yolo11s.pt reconstructed with Stage20B R0 native get_model",
            "stage21b_initialization_reference_sha256": sha(run / "initialization_reference.json"),
            "pretrained_sha256": init["pretrained_sha256"], "seed": 42,
            "full_parameter_hash_equal_frozen_reference": True,
            "selected_parameters": {group:{"name":name,"shape":list(params[name].shape),
                "initialization_sha256":init["parameters"][name]} for group,name in selected_names.items()},
            "feature_layers": cfg["feature_layers"], "training_checkpoints_used": 0,
            "optimizer_step_count": 0, "precision": "float32",
            "torch_version":torch.__version__,"ultralytics_version":__import__("ultralytics").__version__,
            "device_name":torch.cuda.get_device_name(b["device"])}
        save(output / "model_parameter_binding.json", model_binding)
        # BN state is reset after EVERY diagnostic forward, so ordering cannot
        # change later results despite native training-mode loss and hooks.
        buffer_state = {n: t.detach().clone() for n,t in model.named_buffers()}
        current = {"batch": None, "features": {}}
        hooks = []
        for layer, index in cfg["feature_layers"].items():
            def capture(_module, _input, output_tensor, feature_name=layer):
                current["features"][feature_name] = feature_roi_vectors(output_tensor, current["batch"])
            hooks.append(model.model[index].register_forward_hook(capture))

        def probe(bank, role, cls_memberships, slot=None):
            binding = bindings["A" if bank == "real" else bank][role]
            image, label = ((run / binding[k]).resolve() for k in ("local_image", "local_label"))
            batch, label_classes, geometry = batch_from_bound_asset(image,label,b["device"],cfg["image_size"],native_images,guard)
            if set(CLASSES[i] for i in label_classes) != set(cls_memberships):
                fail("LABEL_CLASS_MEMBERSHIP_CHANGED")
            model.zero_grad(set_to_none=True)
            current["batch"] = batch; current["features"] = {}
            loss, items = model(batch)
            objective = loss.sum()
            if not bool(torch.isfinite(objective)) or not isinstance(items,dict):
                fail("NATIVE_LOSS_INVALID")
            objective.backward()
            expected_loss_keys = {"box_loss","cls_loss","dfl_loss"}
            if set(items) != expected_loss_keys:
                fail("NATIVE_LOSS_ITEMS_CHANGED")
            fields = {key:float(value.detach()) for key,value in items.items()}
            fields["total_loss"] = float(objective.detach())
            if not all(math.isfinite(v) for v in fields.values()):
                fail("NONFINITE_LOSS")
            components = []
            norms = {}
            for group,name in selected_names.items():
                grad = params[name].grad
                if grad is None or not bool(torch.isfinite(grad).all()):
                    fail("NONFINITE_OR_MISSING_GRADIENT")
                flat = grad.detach().float().reshape(-1).cpu().numpy().copy()
                components.append(flat)
                norms[group] = float(np.linalg.norm(flat.astype(np.float64)))
            concat = np.concatenate(components).astype(np.float32)
            gn = float(np.linalg.norm(concat.astype(np.float64)))
            if gn == 0:
                fail("ZERO_SELECTED_GRADIENT")
            if set(current["features"]) != set(cfg["feature_layers"]):
                fail("FEATURE_HOOK_NOT_FIRED")
            record = {"role_id":role,"bank":bank,"slot_id":slot,"class_memberships":sorted(cls_memberships),
                "image_sha256":binding["image_sha256"],"label_sha256":binding["label_sha256"],
                "geometry":geometry,"loss":fields,"gradient_norm_by_group":norms,"gradient_norm_selected_concat":gn}
            row_index = len(raw)
            raw.append(record); vectors.append(unit(concat).astype(np.float32))
            for layer, group_map in current["features"].items():
                for cls, vec in group_map.items():
                    features[(bank,cls,layer)].append({"row_index":row_index,"vector":vec})
            with torch.no_grad():
                for n,t in model.named_buffers(): t.copy_(buffer_state[n])
            del loss,items,objective,batch
            model.zero_grad(set_to_none=True)
            print(f"Stage22A {len(raw)}/292 {bank} {role}", flush=True)

        for row in real:
            probe("real",row["role_id"],set(row["class_instances"]))
        for slot in slots:
            for bank in BANKS:
                probe(bank,slot["role_id"],{slot["class"]},slot["slot_id"])
        for hook in hooks: hook.remove()
        if len(raw) != 292 or counts["forbidden_optimizer_step_calls"] != 0:
            fail("PROBE_POPULATION_OR_OPTIMIZER_INVALID")
        if parameter_hashes(model) != init["parameters"] or {n:tensor_sha(t) for n,t in model.named_buffers()} != init["buffers"]:
            fail("FROZEN_MODEL_CHANGED")
    save(output / "file_access_guard.json",guard.counts)
    if guard.counts["official_final_eval_image_read"] or guard.counts["official_final_eval_label_read"] or guard.counts["official_blocked_open_attempts"] or guard.counts["internal_dev_blocked_open_attempts"]:
        fail("FORBIDDEN_CONTENT_ACCESS")
    analyze(cfg,manifest,raw,vectors,features,output,model_binding,guard.counts)


def analyze(cfg, manifest, raw, vectors, features, output, model_binding, guard_counts):
    bybank = {bank:[(i,r) for i,r in enumerate(raw) if r["bank"] == bank] for bank in ("real",)+BANKS}
    fields = ("box_loss","cls_loss","dfl_loss","total_loss")
    loss_audit = {"records": [{k:r[k] for k in ("role_id","bank","slot_id","class_memberships","loss")} for r in raw],
        "summary": {bank:{cls:{field:summary(r["loss"][field] for _,r in rows if cls is None or cls in r["class_memberships"])
                         for field in fields} for cls in (None,)+CLASSES} for bank,rows in bybank.items()},
        "class_none_means_all_images": True,"no_good_bad_threshold":True}
    save(output / "loss_distribution_audit.json",loss_audit)
    norm_fields = ("gradient_norm_selected_concat",)+tuple(cfg["selected_parameters"])
    norm_audit = {"records":[{"role_id":r["role_id"],"bank":r["bank"],"slot_id":r["slot_id"],
        "class_memberships":r["class_memberships"],"selected_concat":r["gradient_norm_selected_concat"],
        "by_group":r["gradient_norm_by_group"]} for r in raw],
        "summary":{bank:{cls:{field:summary([r["gradient_norm_selected_concat"] if field=="gradient_norm_selected_concat"
            else r["gradient_norm_by_group"][field] for _,r in rows if cls is None or cls in r["class_memberships"]])
            for field in norm_fields} for cls in (None,)+CLASSES} for bank,rows in bybank.items()},
        "selected_parameters":model_binding["selected_parameters"]}
    save(output / "gradient_norm_audit.json",norm_audit)
    real_indices = [i for i,r in enumerate(raw) if r["bank"] == "real"]
    references = {"global":unit(np.mean([vectors[i] for i in real_indices],axis=0))}
    for cls in CLASSES:
        references[cls] = unit(np.mean([vectors[i] for i in real_indices if cls in raw[i]["class_memberships"]],axis=0))
    direction_records = []
    for i,r in enumerate(raw):
        if r["bank"] == "real":
            # Leave-one-out avoids inflating the real control by self-inclusion.
            ref_global = unit(np.mean([vectors[j] for j in real_indices if j != i],axis=0))
            real_global = cosine(vectors[i],ref_global)
            same = {}
            for cls in r["class_memberships"]:
                peers = [vectors[j] for j in real_indices if j != i and cls in raw[j]["class_memberships"]]
                same[cls] = cosine(vectors[i],unit(np.mean(peers,axis=0)))
        else:
            real_global = cosine(vectors[i],references["global"])
            same = {cls:cosine(vectors[i],references[cls]) for cls in r["class_memberships"]}
        if real_global is None or any(v is None for v in same.values()):
            fail("ZERO_MEAN_REFERENCE_GRADIENT")
        direction_records.append({"role_id":r["role_id"],"bank":r["bank"],"slot_id":r["slot_id"],
            "class_memberships":r["class_memberships"],"cosine_real_global":real_global,
            "cosine_real_same_class":same,"global_conflict":real_global<0,
            "same_class_conflict":{cls:v<0 for cls,v in same.items()}})
    direction_summary = {bank:{cls:{"global":summary(d["cosine_real_global"] for d in direction_records
            if d["bank"]==bank and (cls is None or cls in d["class_memberships"])),
        "same_class":summary(d["cosine_real_same_class"][cls] for d in direction_records
            if d["bank"]==bank and cls is not None and cls in d["class_memberships"]) if cls else None}
        for cls in (None,)+CLASSES} for bank in ("real",)+BANKS}
    save(output / "gradient_direction_audit.json",{"reference_definition":cfg["gradient_reference"],
        "real_reference_counts":{"global":len(real_indices),**{cls:sum(cls in raw[i]["class_memberships"] for i in real_indices) for cls in CLASSES}},
        "real_reference_sha256":{k:hashlib.sha256(v.astype(np.float32).tobytes()).hexdigest() for k,v in references.items()},
        "records":direction_records,"summary":direction_summary,"cosine_is_diagnostic_only":True})
    conflict = {bank:{cls:{"global_conflict_count":sum(d["global_conflict"] for d in direction_records
            if d["bank"]==bank and (cls is None or cls in d["class_memberships"])),
        "global_conflict_fraction":float(np.mean([d["global_conflict"] for d in direction_records
            if d["bank"]==bank and (cls is None or cls in d["class_memberships"])])),
        "same_class_conflict_count":sum(d["same_class_conflict"][cls] for d in direction_records
            if d["bank"]==bank and cls is not None and cls in d["class_memberships"]) if cls else None,
        "same_class_conflict_fraction":float(np.mean([d["same_class_conflict"][cls] for d in direction_records
            if d["bank"]==bank and cls is not None and cls in d["class_memberships"]])) if cls else None}
        for cls in (None,)+CLASSES} for bank in ("real",)+BANKS}
    save(output / "gradient_conflict_audit.json",{"rule":"cosine < 0", "by_bank_and_class":conflict,
        "sample_filtering_or_weighting":False})
    slot_records = []
    for slot in manifest["synthetic_slots"]:
        ix = [next(i for i,r in enumerate(raw) if r["slot_id"]==slot["slot_id"] and r["bank"]==bank) for bank in BANKS]
        metrics = {"total_loss":[raw[i]["loss"]["total_loss"] for i in ix],
            "gradient_norm":[raw[i]["gradient_norm_selected_concat"] for i in ix],
            "cosine_global":[direction_records[i]["cosine_real_global"] for i in ix],
            "cosine_same_class":[direction_records[i]["cosine_real_same_class"][slot["class"]] for i in ix]}
        slot_records.append({"slot_id":slot["slot_id"],"class":slot["class"],
            "A_B_C_values":metrics,"within_slot":{k:{"range":float(np.ptp(v)),"std":float(np.std(v))} for k,v in metrics.items()},
            "same_label_sha256":slot["banks"]["A"]["label_sha256"],
            "sample_seeds":{bank:slot["banks"][bank]["sample_seed"] for bank in BANKS}})
    triplet = {"records":slot_records,"summary":{cls:{key:{stat:summary(row["within_slot"][key][stat]
        for row in slot_records if cls is None or row["class"]==cls) for stat in ("range","std")}
        for key in ("total_loss","gradient_norm","cosine_global","cosine_same_class")}
        for cls in (None,)+CLASSES},"only_generation_pixels_change_within_frozen_slot":True}
    save(output / "matched_triplet_audit.json",triplet)
    feature_rows = {}
    for layer in cfg["feature_layers"]:
        feature_rows[layer] = {}
        for cls in CLASSES:
            control = feature_stats([v["vector"] for v in features[("real",cls,layer)]])
            feature_rows[layer][cls] = {"real":{"n":control["n"],"covariance_trace":control["covariance_trace"],
                "effective_rank":control["effective_rank"]},"banks":{}}
            for bank in BANKS:
                current = feature_stats([v["vector"] for v in features[(bank,cls,layer)]])
                feature_rows[layer][cls]["banks"][bank] = {"n":current["n"],
                    "class_conditional_mean_l2_distance":float(np.linalg.norm(current["mean"]-control["mean"])),
                    "covariance_trace":current["covariance_trace"],"effective_rank":current["effective_rank"]}
    feature_audit = {"feature_layers":cfg["feature_layers"],"roi_protocol":cfg["feature_roi"],
        "per_image_class_roi_pooled":True,"summary":feature_rows,"no_fid_or_kid":True}
    save(output / "feature_distribution_audit.json",feature_audit)
    classwise = {cls:{"real_image_count":manifest["real_images_by_membership"][cls],
        "synthetic_slots_per_bank":sum(s["class"]==cls for s in manifest["synthetic_slots"]),
        "loss":{bank:loss_audit["summary"][bank][cls] for bank in ("real",)+BANKS},
        "gradient_norm":{bank:norm_audit["summary"][bank][cls] for bank in ("real",)+BANKS},
        "direction":{bank:direction_summary[bank][cls] for bank in ("real",)+BANKS},
        "conflict":{bank:conflict[bank][cls] for bank in ("real",)+BANKS},
        "feature":{layer:feature_rows[layer][cls] for layer in cfg["feature_layers"]},
        "matched_triplets":triplet["summary"][cls]} for cls in CLASSES}
    save(output / "classwise_mechanism_audit.json",classwise)
    # Predeclared descriptive gates: direction is never redefined from the
    # observed utility ordering. The strong-signal flag also requires that the
    # diagnostic contrast is not trivially confounded by target multiplicity.
    candidate_specs = {
        "loss_distribution": (lambda bank,cls:loss_audit["summary"][bank][cls]["total_loss"]["median"], "total_loss"),
        "gradient_norm": (lambda bank,cls:norm_audit["summary"][bank][cls]["gradient_norm_selected_concat"]["median"], "gradient_norm"),
        "gradient_direction_conflict": (lambda bank,cls:conflict[bank][cls]["same_class_conflict_fraction"], "cosine_same_class"),
    }
    candidates = {}
    multiplicity = {cls:{bank:summary(r["geometry"]["annotation_instances"] for _,r in bybank[bank]
        if cls in r["class_memberships"]) for bank in ("real",)+BANKS} for cls in CLASSES}
    for name,(value,triplet_key) in candidate_specs.items():
        rows = {}
        for cls in CLASSES:
            baseline = value("real",cls)
            medians = {bank:value(bank,cls) for bank in BANKS}
            deltas = {bank:medians[bank]-baseline for bank in BANKS}
            signs = {bank:int(np.sign(delta)) for bank,delta in deltas.items()}
            if name == "gradient_direction_conflict":
                ref = conflict["real"][cls]
                real_count = direction_summary["real"][cls]["same_class"]["n"]
                real_interval = wilson_interval(ref["same_class_conflict_count"],real_count)
                bank_intervals = {bank:wilson_interval(conflict[bank][cls]["same_class_conflict_count"],
                    direction_summary[bank][cls]["same_class"]["n"]) for bank in BANKS}
                clearly_separated = all(interval[0] > real_interval[1] or interval[1] < real_interval[0]
                    for interval in bank_intervals.values())
                criterion_detail = {"real_wilson95":real_interval,"bank_wilson95":bank_intervals}
            else:
                ref = (loss_audit["summary"] if name=="loss_distribution" else norm_audit["summary"])["real"][cls]
                ref = ref["total_loss" if name=="loss_distribution" else "gradient_norm_selected_concat"]
                clearly_separated = all(v < ref["q25"] or v > ref["q75"] for v in medians.values())
                criterion_detail = {"real_q25":ref["q25"],"real_q75":ref["q75"]}
            rows[cls] = {"real":baseline,"synthetic_median_or_fraction":medians,
                "delta_vs_real":deltas,"signs":signs,
                "clearly_separated_from_real_control":clearly_separated,"separation_details":criterion_detail,
                "all_three_banks_same_nonzero_direction":len(set(signs.values()))==1 and 0 not in signs.values(),
                "median_within_slot_range":triplet["summary"][cls][triplet_key]["range"]["median"]}
        consistent = all(row["all_three_banks_same_nonzero_direction"] for row in rows.values())
        same_class_sign = consistent and len({next(iter(row["signs"].values())) for row in rows.values()})==1
        rng_varies = all(row["median_within_slot_range"] > 0 for row in rows.values())
        def count_proportions(bank, cls):
            counts = Counter(r["geometry"]["annotation_instances"] for _,r in bybank[bank]
                if cls in r["class_memberships"])
            return {k:v/sum(counts.values()) for k,v in counts.items()}
        multiplicity_confounded = any(
            count_proportions("real",cls) != count_proportions(bank,cls)
            for cls in CLASSES for bank in BANKS)
        # Loss and norm directly respond to target count; direction/conflict
        # can too. A contrast at fixed initialization cannot rule that out.
        admissible = (all(row["clearly_separated_from_real_control"] for row in rows.values())
            and consistent and same_class_sign and rng_varies and not multiplicity_confounded)
        candidates[name] = {"by_class":rows,"A_real_synthetic_direction":all(row["clearly_separated_from_real_control"] for row in rows.values()),
            "B_three_bank_consistency":consistent,"C_class_asymmetry_noncontradictory":same_class_sign,
            "D_matched_rng_changes_signal":rng_varies,"E_no_final_eval_threshold_tuning":True,
            "real_vs_synthetic_annotation_multiplicity_differs":multiplicity_confounded,
            "all_predeclared_descriptive_gates_pass":admissible,
            "unique_unconfounded_mechanism_established":False,
            "interpretation":"descriptive direction only; target multiplicity, image domain, and annotation geometry are not independently controlled"}
    candidates["feature_activation"] = {"class_conditional_roi_distribution":feature_rows,
        "A_real_synthetic_direction":"mean distance is nonnegative by definition, not an adverse direction",
        "B_three_bank_consistency":"descriptive only", "C_class_asymmetry_noncontradictory":"see classwise audit",
        "D_matched_rng_changes_signal":"not measured for per-slot ROI distances",
        "E_no_final_eval_threshold_tuning":True,"unique_unconfounded_mechanism_established":False}
    passing = [name for name,row in candidates.items() if row.get("all_predeclared_descriptive_gates_pass")]
    strongest = passing[0] if len(passing)==1 else None
    if strongest:
        candidates[strongest]["unique_unconfounded_mechanism_established"] = True
    decision_reason = ("A single descriptive signal passes predeclared A-E and annotation-count gates; this is not causal proof."
        if strongest else "No unique signal passed all predeclared A-E and annotation-count gates; frozen-initialization contrasts remain descriptive, not causal.")
    ordering = {}
    for name,(value,_) in candidate_specs.items():
        ordering[name] = {}
        for cls in CLASSES:
            rank = sorted(BANKS,key=lambda bank:(-value(bank,cls),bank))
            ordering[name][cls] = {"descending_bank_order":rank,
                "comparison_to_historical_internal_dev_uniform":
                    "ORDERING_CONSISTENT_DIAGNOSTIC" if rank==cfg["historical_stage21b_internal_dev_bank_order_descending"]
                    else "ORDERING_NOT_MATCHED_DIAGNOSTIC"}
    for layer in cfg["feature_layers"]:
        ordering["feature_"+layer] = {}
        for cls in CLASSES:
            rank = sorted(BANKS,key=lambda bank:(-feature_rows[layer][cls]["banks"][bank]["class_conditional_mean_l2_distance"],bank))
            ordering["feature_"+layer][cls] = {"descending_bank_order":rank,
                "comparison_to_historical_internal_dev_uniform":
                    "ORDERING_CONSISTENT_DIAGNOSTIC" if rank==cfg["historical_stage21b_internal_dev_bank_order_descending"]
                    else "ORDERING_NOT_MATCHED_DIAGNOSTIC"}
    candidate_audit = {"criteria":{"A":"synthetic median outside real IQR in all banks/classes; conflict uses nonoverlapping Wilson 95% intervals",
        "B":"same nonzero direction in all A/B/C", "C":"no opposite Flash/Black directions",
        "D":"nonzero median within-slot range", "E":"no official final_eval tuning",
        "confound":"exact per-class annotation-count histograms must match"},
        "real_annotation_multiplicity":multiplicity,"candidates":candidates,"strongest_unique_candidate":strongest,
        "descriptive_bank_ordering":ordering,
        "historical_internal_dev_uniform_order_descending":cfg["historical_stage21b_internal_dev_bank_order_descending"],
        "reason":decision_reason,
        "bank_count":3,"pearson_spearman_computed":False,"utility_predictor_claim":False}
    save(output / "mechanism_candidate_summary.json",candidate_audit)
    status = {"status":"SYNTHETIC_OPTIMIZATION_MECHANISM_SIGNAL_IDENTIFIED" if strongest else "SYNTHETIC_OPTIMIZATION_MECHANISM_NOT_IDENTIFIED",
        "scope":"TRAIN_ONLY_FROZEN_INITIALIZATION_DIAGNOSTIC",
        "real_images":109,"synthetic_slots_per_bank":61,"probe_forwards":len(raw),"probe_backwards":len(raw),
        "generator_training_count":0,"generation_count":0,"detector_training_count":0,"optimizer_step_count":0,
        "official_final_eval_image_read":0,"official_final_eval_label_read":0,"official_final_eval_forward":0,
        "internal_dev_content_read":0,"BootstrapGuard":0,"DeepPCB":0,"stage22b_auto_start":False,
        "reason":candidate_audit["reason"],"model_parameter_binding_sha256":sha(output / "model_parameter_binding.json"),
        "diagnostic_protocol_sha256":sha(output / "diagnostic_protocol.json"),"data_manifest_sha256":sha(output / "diagnostic_data_manifest.json"),
        "guard_counts":guard_counts}
    report = f"""# Stage22A — Synthetic Failure Mechanism Audit\n\nStatus: `{status['status']}`.\n\nAll 109 Stage21B internal_train real images and 61 matched frozen slots in each of A/B/C were probed at the same seed-42 public YOLO11s initialization, with deterministic no-augmentation letterbox preprocessing at 1536 and no optimizer step. The native box/cls/dfl loss and selected-layer gradients were measured on 292 forward/backward passes. Model parameters and buffers were restored/verified unchanged. Real class references use annotation membership; real direction controls use leave-one-out.\n\nThis is descriptive evidence, not a causal attribution. Differences in loss, gradient, conflict, or detector features do not by themselves distinguish pixel fidelity, annotation geometry, domain shift, or optimizer interaction. Three banks are insufficient for a correlation claim. No unique mechanism passed the full A–E interpretation standard; no Stage22B or new method was started.\n\n## Reproducibility\n\n- Protocol: `diagnostic_protocol.json`; immutable roles and source hashes: `diagnostic_data_manifest.json`.\n- Exact selected parameters and initialization binding: `model_parameter_binding.json`.\n- Per-image loss, group norm, direction and sign conflict: respective audit JSON files.\n- Per-slot A/B/C variation: `matched_triplet_audit.json`; ROI activation moments: `feature_distribution_audit.json`.\n- Flash/Black summaries: `classwise_mechanism_audit.json`; decision limits: `mechanism_candidate_summary.json`.\n- Official final_eval content/forward: 0; internal_dev: 0; generator/detector training: 0; optimizer steps: 0.\n\nLimitations: the diagnostic uses a frozen initialization, no augmentation, one detector architecture and no intervention. The existing Stage21B utility ordering is historical context only; it was not used to select images, layers, thresholds, or metrics.\n"""
    report = report.replace("No unique mechanism passed the full A–E interpretation standard;", f"Decision: {decision_reason} Strongest unique candidate: {strongest or 'none'};")
    report += "\n## Classwise diagnostic summary\n\n"
    report += "| Class | Group | Total loss median | Selected-gradient L2 median | Same-class cosine median | Same-class conflict fraction |\n"
    report += "|---|---|---:|---:|---:|---:|\n"
    for cls in CLASSES:
        for bank in ("real",)+BANKS:
            report += (f"| {cls} | {bank} | {loss_audit['summary'][bank][cls]['total_loss']['median']:.6g} "
                f"| {norm_audit['summary'][bank][cls]['gradient_norm_selected_concat']['median']:.6g} "
                f"| {direction_summary[bank][cls]['same_class']['median']:.6g} "
                f"| {conflict[bank][cls]['same_class_conflict_fraction']:.4f} |\n")
    report += "\nReal rows use leave-one-out references. Each bank uses the same 61 slots; real target multiplicity is audited separately.\n\n"
    report += "| Class | A/B/C within-slot total-loss range median | Gradient-norm range median | Same-class cosine range median |\n"
    report += "|---|---:|---:|---:|\n"
    for cls in CLASSES:
        tr = triplet["summary"][cls]
        report += (f"| {cls} | {tr['total_loss']['range']['median']:.6g} "
            f"| {tr['gradient_norm']['range']['median']:.6g} "
            f"| {tr['cosine_same_class']['range']['median']:.6g} |\n")
    report += "\nFeature distributions and all raw per-image/per-slot records are in the linked JSON audits. No metric is a utility predictor.\n"
    (output / "STAGE22A_REPORT.md").write_text(report,encoding="utf-8")
    save(output / "stage22a_status.json",status)
    print(json.dumps({"status":status["status"],"probe_forwards":len(raw),"optimizer_steps":0,
        "official_final_eval_forward":0},indent=2),flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--protocol",type=Path,default=REPO/"configs/plastic_bomo_stage22a.json")
    p.add_argument("--stage21b",type=Path)
    p.add_argument("--output",type=Path,default=REPO/"results_for_gpt/plastic_bomo_stage22a_failure_mechanism")
    p.add_argument("--device",default="0")
    p.add_argument("--preflight_only",action="store_true")
    a = p.parse_args()
    protocol = a.protocol.resolve(); cfg = load(protocol)
    run = (a.stage21b or REPO/cfg["stage21b_run"]).resolve()
    output = a.output.resolve()
    if (output/"stage22a_status.json").exists(): fail("COMPLETED_OUTPUT_EXISTS")
    cfg, manifest, real, slots, bindings = protocol_and_manifest(protocol,run,verify_files=True)
    output.mkdir(parents=True,exist_ok=True)
    save(output/"diagnostic_protocol.json",{"frozen_protocol":cfg,"source_path":str(protocol),"source_sha256":sha(protocol),
        "predeclared_before_gpu":True,"historical_utility_not_used_for_selection":True})
    save(output/"diagnostic_data_manifest.json",manifest)
    save(output/"stage22a_preflight.json",{"status":"STAGE22A_TRAIN_ONLY_BINDING_PASS", "real_images":109,
        "synthetic_slots_per_bank":61,"all_source_assets_sha256_verified":True,"gpu_probe_requested":not a.preflight_only,
        "official_final_eval_access":0,"internal_dev_access":0})
    if a.preflight_only:
        print(json.dumps({"status":"STAGE22A_TRAIN_ONLY_BINDING_PASS","real_images":109,
            "synthetic_slots_per_bank":61,"gpu_probe_started":False},indent=2)); return
    run_gpu(cfg,manifest,real,slots,bindings,run,output,a.device)


if __name__ == "__main__": main()
