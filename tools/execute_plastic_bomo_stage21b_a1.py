"""Isolated GPU children: safety, one arm, or one unified internal_dev round."""
from collections import Counter
from copy import deepcopy
import argparse
import gc
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import traceback

from stage21b_a1_common import *


def prepare_bank(run, b, bank):
    root = run / "train_inputs" / bank
    binding = root / "role_binding.json"
    if binding.is_file():
        rows = load(binding)["roles"]
        expected = {r["role_id"]:{k:r[k] for k in ("image_sha256","label_sha256")}
                    for r in load(run / "stage21b_split_manifest.json")["records"] if r["internal_split"] == "internal_train"}
        expected.update({r["role_id"]:{k:r["banks"][bank][k] for k in ("image_sha256","label_sha256")}
                         for r in load(run / "stage21b_synthetic_allowlist_v2.json")["records"]})
        require(len(rows) == 170 and {r:{k:v[k] for k in ("image_sha256","label_sha256")} for r,v in rows.items()} == expected,
                "STAGE21B_DATASET_BINDING_CHANGED")
        for row in rows.values():
            require(sha(safe_relative(run,row["local_image"])) == row["image_sha256"]
                    and sha(safe_relative(run,row["local_label"])) == row["label_sha256"], "STAGE21B_DATASET_CHANGED")
        return root, rows
    require(not root.exists(), "STAGE21B_INCOMPLETE_DATASET_EXISTS")
    split = load(run / "stage21b_split_manifest.json")
    base = [r for r in split["records"] if r["internal_split"] == "internal_train"]
    records = []
    for row in base:
        records.append({"role_id":row["role_id"], "image_source":str(safe_relative(b["real_root"],row["image_relative_path"])),
            "label_source":str(safe_relative(b["real_root"],row["label_relative_path"])),
            "image_sha256":row["image_sha256"], "label_sha256":row["label_sha256"], "source_group":row["source_group_id"]})
    for row in load(run / "stage21b_synthetic_allowlist_v2.json")["records"]:
        asset = row["banks"][bank]
        records.append({"role_id":row["role_id"], "image_source":str(safe_relative(b["stage20b"],asset["image_path"])),
            "label_source":str(safe_relative(b["stage20b"],asset["label_path"])),
            "image_sha256":asset["image_sha256"], "label_sha256":asset["label_sha256"], "source_group":row["source_group_id"]})
    require(len(records) == 170 and len({r["role_id"] for r in records}) == 170, "STAGE21B_DATASET_ROLE_MISMATCH")
    rows = {}
    for row in records:
        role = row["role_id"]
        image = root / "images/train" / (role + Path(row["image_source"]).suffix)
        label = root / "labels/train" / (role + ".txt")
        verified_copy(row["image_source"], image, row["image_sha256"])
        verified_copy(row["label_source"], label, row["label_sha256"])
        rows[role] = {**row,"local_image":image.relative_to(run).as_posix(),"local_label":label.relative_to(run).as_posix()}
    import yaml
    (root / "data.yaml").write_text(yaml.safe_dump({"path":str(root),"train":"images/train",
         "names":{0:"flash",1:"black"},"nc":2},sort_keys=False),encoding="utf-8")
    save(binding,{"roles":rows,"internal_train_only":True,"official_eval_reference_count":0,"dev_reference_count":0})
    return root,rows


def make_model(run, b, root, torch):
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    from qualify_plastic_bomo_stage20b_r0 import parameter_hashes,tensor_sha
    frozen = load(run / "stage20b_detector_protocol_snapshot.json")["training"]
    require(sha(b["detector_model"]) == b["detector_sha256"],"STAGE21B_INITIALIZATION_CHANGED")
    torch.manual_seed(42)
    native = DetectionTrainer.__new__(DetectionTrainer)
    args = load(run / "stage20b_resolved_arguments_snapshot.json")["arguments"]
    args.update(data=str(root / "data.yaml"),device=b["device"],project=str(run),name="controlled_timing",
                model=b["detector_model"],val=False)
    native.args = SimpleNamespace(**args)
    native.data = {"nc":2,"names":{0:"flash",1:"black"},"channels":3}
    native.device = torch.device(b["device"])
    original = YOLO(b["detector_model"])
    model = native.get_model(cfg=original.model.yaml,weights=original.model,verbose=False).to(b["device"]).float()
    native.model = model; native.set_model_attributes()
    for name,p in model.named_parameters(): p.requires_grad_(".dfl" not in name)
    native.freeze_layer_names = [".dfl"]
    native.stride = max(32,int(model.stride.max()))
    if native.args.channels_last is True or native.args.channels_last is None: model.to(memory_format=torch.channels_last)
    initialization = {"parameters":parameter_hashes(model),"buffers":{n:tensor_sha(v) for n,v in model.named_buffers()},
                      "pretrained_sha256":sha(b["detector_model"]),"nc":model.model[-1].nc,"head_seed":42}
    require(initialization["parameters"] == load(run / "r0_initialization_snapshot.json")["training_model_parameter_sha256"]
            and initialization["nc"] == 2 and all(p.dtype == torch.float32 for p in model.parameters()),"STAGE21B_INITIALIZATION_CHANGED")
    ref = run / "initialization_reference.json"
    if not ref.exists(): save(ref,initialization)
    else: require(initialization == load(ref),"STAGE21B_INITIALIZATION_CHANGED")
    del original
    return native,model,frozen,args,initialization


def dataset(root,rows,native,guard):
    from ultralytics.data.dataset import YOLODataset
    guard.native_images = {str((root.parents[1] / row["local_image"]).resolve()) for row in rows.values()}
    ds = YOLODataset(img_path=str(root / "images/train"),imgsz=1536,batch_size=1,augment=True,
        hyp=native.args,rect=False,cache=False,stride=native.stride,task="detect",data=native.data)
    lookup = {Path(f).stem:i for i,f in enumerate(ds.im_files)}
    require(len(ds) == 170 and set(lookup) == set(rows),"STAGE21B_NATIVE_DATASET_DROPPED_ROLES")
    require(all(str(Path(f).resolve()) in guard.native_images for f in ds.im_files),"STAGE21B_NATIVE_IMAGE_NOT_ALLOWLISTED")
    return ds,lookup


def batch_hash(batch):
    from qualify_plastic_bomo_stage20b_r0 import tensor_sha
    # Exactly the model's augmented inputs/targets, before its loss mutates anything.
    return digest({k:tensor_sha(batch[k]) for k in ("img","cls","bboxes","batch_idx")})


def safety(run):
    b = child_context(run)
    require(load(run / "stage21b_status.json")["phase"] == "SAFETY", "STAGE21B_SAFETY_STATE_INVALID")
    audit = {"status":"RUNNING","optimizer_step_count":0,"internal_train_only":True,"schedules":{}}
    from qualify_plastic_bomo_stage20b_r0 import draw_rng,optimizer_steps_forbidden,parameter_hashes
    with gpu_guard(run,b,"safety") as (torch,guard):
        root,roles = prepare_bank(run,b,"A")
        for mode in ("uniform","curriculum"):
            native,model,frozen,args,init = make_model(run,b,root,torch)
            ds,lookup = dataset(root,roles,native,guard)
            schedule = load(run / (mode + "_schedule_v2.json"))
            role_index,k = schedule["epoch_draws"][0][0]
            role = schedule["roles"][role_index]
            require(role.startswith("base_"),"STAGE21B_SAFETY_MUST_USE_REAL_INTERNAL_TRAIN")
            seed = occurrence_seed(role,k,load(run / "stage21b_base_protocol_snapshot.json"))
            native._model_train(); model.zero_grad(set_to_none=True)
            counts = {"forbidden_optimizer_step_calls":0}
            with optimizer_steps_forbidden(torch,counts),draw_rng(seed,ds.transforms):
                batch = native.preprocess_batch(ds.collate_fn([ds[lookup[role]]]))
                with torch.autocast("cuda",dtype=torch.float16,enabled=frozen["amp"]):
                    loss,items = model(batch); objective = loss.sum()
                require(bool(torch.isfinite(objective)),"STAGE21B_SINGLE_BATCH_SAFETY_FAILED: loss")
                objective.backward()  # Deliberately NO optimizer/scaler step; discard the model afterwards.
            finite = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
            nonzero = any(p.grad is not None and bool((p.grad != 0).any()) for p in model.parameters())
            require(finite and nonzero and parameter_hashes(model) == init["parameters"],"STAGE21B_SINGLE_BATCH_SAFETY_FAILED: gradient/parameter")
            audit["schedules"][mode] = {"status":"PASS","role":role,"occurrence_index":k,"augmentation_seed":seed,
                "loss":float(objective.detach()),"finite_gradients":finite,"nonzero_gradients":nonzero,
                "parameters_unchanged":True,"initialization_sha256":digest(init),"optimizer_steps":0,
                "model_discarded_before_formal_training":True}
            del ds,native,model,batch,objective,loss,items; gc.collect(); torch.cuda.empty_cache()
    audit["status"] = "PASS"; save(run / "single_batch_safety_audit.json",audit)


def train(run,arm):
    b = child_context(run)
    s = load(run / "stage21b_status.json")
    require(s["phase"] == arm and s["completed_arms"] == list(ARMS[:ARMS.index(arm)])
            and load(run / "single_batch_safety_audit.json")["status"] == "PASS", "STAGE21B_ARM_ORDER_INVALID")
    path = run / (arm + "_training.json")
    require(not path.exists(),"STAGE21B_TRAINING_ALREADY_STARTED_NO_AUTOMATIC_RETRY")
    bank,mode = arm.split("_")
    schedule = load(run / (mode + "_schedule_v2.json")); base = load(run / "stage21b_base_protocol_snapshot.json")
    audit = {"status":"RUNNING","arm":arm,"epochs":0,"actual_draws":0,"attempted_optimizer_updates":0,
             "successful_optimizer_updates":0,"AMP_skipped_updates":0,"nonfinite_losses":0,"gradient_nonfinite_count":0,
             "epoch_records":[],"scheduled_draws":25500,"scheduled_optimizer_attempts":450,
             "early_stopping_used":False,"best_checkpoint_selection":False,"native_full_train_used":False,
             "internal_dev_content_read":0,"official_final_eval_content_read":0,"schedule_sha256":sha(run / (mode + "_schedule_v2.json")),
             "primary_model_representation":"pinned native EMA at epoch150","EMA_update_rule":"every scaler attempt, including AMP skip; unchanged from Stage20B"}
    save(path,audit); started = time.monotonic()
    try:
        with gpu_guard(run,b,arm) as (torch,guard):
            from ultralytics.utils.torch_utils import ModelEMA
            from qualify_plastic_bomo_stage20b_r0 import draw_rng
            root,roles = prepare_bank(run,b,bank)
            native,model,frozen,args,init = make_model(run,b,root,torch)
            ds,lookup = dataset(root,roles,native,guard)
            controller = TimingController(frozen,schedule,base)
            opt = native.build_optimizer(model,name=frozen["optimizer"],lr=frozen["lr0"],momentum=frozen["momentum"],decay=frozen["weight_decay"])
            for group in opt.param_groups: group["initial_lr"] = frozen["lr0"]
            audit["optimizer_groups"] = [{k:v for k,v in g.items() if k != "params"} for g in opt.param_groups]
            audit["initialization_sha256"] = digest(init); audit["native_transforms"] = repr(ds.transforms)
            scaler = torch.amp.GradScaler("cuda",enabled=frozen["amp"])
            steps = [0]
            hook = opt.register_step_post_hook(lambda *a,**kw: steps.__setitem__(0,steps[0] + 1))
            ema = ModelEMA(model); opt.zero_grad(set_to_none=True)
            state = {"group_draws":0,"observed":[],"attempts":[],"losses":[]}
            logpath = run / "logs" / (arm + "_draws.jsonl")
            with logpath.open("x",encoding="utf-8") as log:
                def draw(epoch,pos,role,seed,length):
                    if pos == 0:
                        native._model_train(); state.update(observed=[],attempts=[],losses=[])
                    i,k = schedule["epoch_draws"][epoch][pos]
                    require(schedule["roles"][i] == role and seed == occurrence_seed(role,k,base),"STAGE21B_OCCURRENCE_PAIRING_INVALID")
                    for group,lr in zip(opt.param_groups,controller.learning_rates(epoch,pos,opt.param_groups)): group["lr"] = lr
                    with draw_rng(seed,ds.transforms):
                        sample = ds[lookup[role]]
                        require(Path(sample["im_file"]).stem == role,"STAGE21B_ACTUAL_ROLE_MISMATCH")
                        batch = native.preprocess_batch(ds.collate_fn([sample]))
                        augmented_hash = batch_hash(batch)
                        with torch.autocast("cuda",dtype=torch.float16,enabled=frozen["amp"]):
                            loss,items = model(batch); objective = loss.sum()
                        finite = bool(torch.isfinite(objective)); audit["nonfinite_losses"] += int(not finite)
                        require(finite and isinstance(items,dict),"STAGE21B_NATIVE_LOSS_INVALID")
                        controller.backward(objective,length,scaler=scaler)
                    audit["actual_draws"] += 1; state["group_draws"] += 1
                    state["observed"].append([i,k]); state["losses"].append(float(objective.detach()))
                    log.write(canonical({"epoch":epoch,"position":pos,"role_index":i,"occurrence_index":k,
                        "augmentation_seed":seed,"augmentation_sha256":augmented_hash,"normalizer":length,
                        "loss":float(objective.detach()),"loss_items":{x:float(v.detach()) for x,v in items.items()}}) + "\n")

                def attempt(epoch,pos,length):
                    require(state["group_draws"] == length,"STAGE21B_ACCUMULATION_MISMATCH")
                    audit["attempted_optimizer_updates"] += 1; scaler.unscale_(opt)
                    gf = all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
                    audit["gradient_nonfinite_count"] += int(not gf)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(),frozen["max_grad_norm"])
                    require(not gf or bool(torch.isfinite(gn)),"STAGE21B_GRADIENT_CLIPPING_INVALID")
                    before,scale0 = steps[0],scaler.get_scale(); scaler.step(opt); scaler.update()
                    success = steps[0] == before + 1
                    audit["successful_optimizer_updates"] += int(success); audit["AMP_skipped_updates"] += int(not success)
                    require(math.isfinite(scaler.get_scale()) and scaler.get_scale() > 0,"STAGE21B_SCALER_INVALID")
                    opt.zero_grad(set_to_none=True); ema.update(model)
                    state["attempts"].append({"position":pos,"draws":length,"normalizer":length,"optimizer_step_executed":success,
                        "AMP_skip":not success,"gradient_norm":float(gn) if bool(torch.isfinite(gn)) else None,
                        "scale_before":scale0,"scale_after":scaler.get_scale(),"learning_rates":[g["lr"] for g in opt.param_groups]})
                    state["group_draws"] = 0

                def epoch_end(epoch):
                    require(state["observed"] == schedule["epoch_draws"][epoch],"STAGE21B_EXPOSURE_SEQUENCE_MISMATCH")
                    audit["epochs"] += 1
                    audit["epoch_records"].append({"epoch_0based":epoch,"draws":170,"sequence_sha256":digest(state["observed"]),
                        "loss_mean":sum(state["losses"])/170,"attempts":list(state["attempts"])})
                    log.flush(); save(path,audit)
                    print(f"{arm}: epoch={epoch+1}/150 draws={audit['actual_draws']} updates={audit['successful_optimizer_updates']} skips={audit['AMP_skipped_updates']}",flush=True)
                totals = controller.run(draw,attempt,epoch_end=epoch_end,cleanup=lambda:opt.zero_grad(set_to_none=True))
            hook.remove()
            require(totals == {"epochs":150,"draws":25500,"scheduled_optimizer_attempts":450}
                    and audit["actual_draws"] == 25500 and audit["attempted_optimizer_updates"] == 450,"STAGE21B_COMPUTE_PAIRING_INVALID")
            require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values())
                    and all(bool(torch.isfinite(v).all()) for v in ema.ema.state_dict().values()),"STAGE21B_FINAL_PARAMETERS_NONFINITE")
            ema.update_attr(model,include=("yaml","nc","args","names","stride","class_weights"))
            snapshot = deepcopy(ema.ema).cpu().half().to(memory_format=torch.contiguous_format)
            if hasattr(snapshot,"criterion"): snapshot.criterion = None
            require(all(bool(torch.isfinite(v).all()) for v in snapshot.state_dict().values()),"STAGE21B_SERIALIZATION_OVERFLOW")
            dest = run / "checkpoints" / arm / "last.pt"; dest.parent.mkdir(parents=True)
            torch.save({"epoch":149,"model":None,"ema":snapshot,"updates":ema.updates,"train_args":args,"optimizer":None,
                "version":__import__("ultralytics").__version__,"stage21b_arm":arm,"primary_checkpoint":"epoch150 last.pt","validation_selection":False},dest)
            audit.update(status="COMPLETE",last_checkpoint=dest.relative_to(run).as_posix(),last_sha256=sha(dest),
                EMA_updates=ema.updates,final_grad_scaler_scale=scaler.get_scale(),per_draw_log_sha256=sha(logpath),
                peak_vram_bytes=torch.cuda.max_memory_allocated())
    finally:
        audit["wall_seconds"] = time.monotonic() - started
        if audit["status"] != "COMPLETE": audit["status"] = "FAILED_NO_AUTOMATIC_RETRY"
        save(path,audit)


def evaluate(run):
    b = child_context(run)
    require(load(run / "stage21b_status.json")["phase"] == "ALL_SIX_FROZEN", "STAGE21B_EVALUATION_BARRIER_FAILED")
    frozen = checkpoint_barrier(run)
    require(frozen == load(run / "detector_checkpoint_binding.json") and load(run / "augmentation_pairing_audit.json")["status"] == "PASS",
            "STAGE21B_EVALUATION_BARRIER_FAILED")
    claim = run / "internal_dev_one_round_claim.json"
    require(not claim.exists(),"STAGE21B_INTERNAL_DEV_ROUND_ALREADY_STARTED_NO_RETRY")
    save(claim,{"checkpoint_binding_sha256":sha(run / "detector_checkpoint_binding.json"),"one_round_only":True})
    audit = {"status":"RUNNING","rounds_started":1,"rounds_completed":0,"internal_dev_forwards":0,"arms":{},
        "checkpoint_binding_sha256":sha(run / "detector_checkpoint_binding.json"),"state_at_first_dev_content_access":"ALL_SIX_FROZEN",
        "official_final_eval_forward":0,"official_final_eval_content_read":0,"dev_population":"DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR"}
    save(run / "internal_dev_evaluation_audit.json",audit); update(run,internal_dev_eval_rounds=1)
    try:
        with gpu_guard(run,b,"internal_dev",allow_dev=True) as (torch,guard):
            from ultralytics import YOLO
            from ultralytics.models.yolo.detect import DetectionValidator
            from evaluate_plastic_bomo_stage20b import metrics_record
            import yaml
            rows = [r for r in load(run / "stage21b_split_manifest.json")["records"] if r["internal_split"] == "internal_dev"]
            require(len(rows) == 27,"STAGE21B_DEV_POPULATION_CHANGED")
            root = run / "internal_dev_inputs"; require(not root.exists(),"STAGE21B_DEV_ALREADY_MATERIALIZED")
            for row in rows:
                for kind,suffix in (("image",Path(row["image_relative_path"]).suffix),("label",".txt")):
                    dest = root / ("images" if kind == "image" else "labels") / "internal_dev" / (row["node_id"] + suffix)
                    verified_copy(safe_relative(b["real_root"],row[kind + "_relative_path"]),dest,row[kind + "_sha256"])
            valdir = str((root / "images/internal_dev").resolve())
            data = {"path":str(root),"train":None,"val":valdir,"internal_dev":valdir,"names":{0:"flash",1:"black"},"nc":2,"channels":3}
            yaml_path = root / "data.yaml"; yaml_path.write_text(yaml.safe_dump(data,sort_keys=False),encoding="utf-8")
            guard.native_images = {str((root / "images/internal_dev" / (r["node_id"] + Path(r["image_relative_path"]).suffix)).resolve()) for r in rows}
            expected = {r["node_id"] for r in rows}; cls_counts = Counter(c for r in rows for c in r["class_instances"])
            frozen_protocol = load(run / "stage20b_detector_protocol_snapshot.json")
            args = {**load(run / "stage20b_resolved_arguments_snapshot.json")["arguments"],
                    **{k:v for k,v in frozen_protocol["evaluation"].items() if k != "metrics"}}
            args.update(data=str(yaml_path),device=b["device"],workers=0,project=str(run / "internal_dev_outputs"),
                        mode="val",task="detect",exist_ok=False,split="internal_dev")
            for arm in ARMS:
                observed,gt = [],[]
                class AuditedValidator(DetectionValidator):
                    def update_metrics(self,preds,batch):
                        names = [Path(f).stem for f in batch["im_file"]]
                        require(all(n in expected for n in names),"STAGE21B_EVAL_UNEXPECTED_INPUT")
                        observed.extend(names); gt.extend(int(x) for x in batch["cls"].reshape(-1).cpu().tolist())
                        audit["internal_dev_forwards"] += len(names)
                        audit["arms"][arm] = {"node_ids":list(observed),"image_forwards":len(observed)}
                        save(run / "internal_dev_evaluation_audit.json",audit)
                        super().update_metrics(preds,batch)
                checkpoint = run / frozen["arms"][arm]["path"]
                require(sha(checkpoint) == frozen["arms"][arm]["sha256"],"STAGE21B_CHECKPOINT_CHANGED")
                model = YOLO(str(checkpoint))
                metric = model.val(validator=AuditedValidator,**{**args,"name":arm,"model":str(checkpoint)})
                require(len(observed) == len(set(observed)) == 27 and set(observed) == expected
                        and Counter(gt) == {0:cls_counts["flash"],1:cls_counts["black"]},"STAGE21B_EVAL_POPULATION_MISMATCH")
                record = metrics_record(metric)
                record.update(arm=arm,checkpoint_sha256=frozen["arms"][arm]["sha256"],evaluation_image_count=27,
                    target_counts=dict(cls_counts),primary_checkpoint="epoch150 last.pt",dev_population=audit["dev_population"])
                save(run / (arm + "_metrics.json"),record)
                del model,metric; gc.collect(); torch.cuda.empty_cache()
            require(audit["internal_dev_forwards"] == 162,"STAGE21B_EVAL_ROUND_INVALID")
            audit.update(status="COMPLETE",rounds_completed=1)
    finally:
        if audit["status"] != "COMPLETE": audit["status"] = "FAILED_PARTIAL_NO_RETRY"
        save(run / "internal_dev_evaluation_audit.json",audit)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--run",type=Path,required=True)
    p.add_argument("--phase",choices=("safety","train","evaluate"),required=True); p.add_argument("--arm",choices=ARMS)
    a=p.parse_args(); a.run=a.run.resolve()
    try:
        if a.phase == "safety": safety(a.run)
        elif a.phase == "train":
            require(a.arm is not None,"arm required"); train(a.run,a.arm)
        else: evaluate(a.run)
    except Exception as error:
        update(a.run,status="STAGE21B_A1_STOPPED",reason=str(error),full_experiment_complete=False)
        print(traceback.format_exc(),flush=True); return 2
    return 0


if __name__ == "__main__": sys.exit(main())
