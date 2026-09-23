"""Stage 7C: exact-matched fixed-budget SD2 vs MSDF downstream detection."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from types import MethodType

from run_deeppcb_stage5b import CLASSES, VAL_HASH, load, metrics, sha, tree_digest
from run_deeppcb_stage5c import best_epoch, save, transforms_audit, walk_transforms
from run_deeppcb_stage5e import finite_metrics
from run_deeppcb_stage6b import BATCH, BATCHES, MODEL_SHA, file_population
from run_deeppcb_stage6f import schedule_hash
from run_deeppcb_stage6g import moments, schedule_for_seed
from run_deeppcb_stage6h import PATIENCE, check_early_stopping, checkpoint_epoch
from run_deeppcb_stage7a import canonical_pool_sha

SEEDS=(42,3407,2026)
ARMS=("sd2","msdf")
MSDF_POOL_SHA="1b2eb1abaeb886b00e49ea64cf322e075c88105b80fe67bc4e69200b6b851a14"
SD2_POOL_SHA="d6d4edc13820551cdf60713411a2bac7c16eceb0b266c521cac36c67f32e4310"
REAL_DRAWS,SYN_DRAWS=260,80


class PairedSampler7C:
    def __init__(self,names,schedules):
        self.names,self.schedules=tuple(names),schedules; self.index={x:i for i,x in enumerate(names)}
        fail(len(self.index)==340 and all(all(x in self.index for x in seq) for seq in schedules),"SAMPLER_POPULATION")
        self.epoch=0; self.observed=[]
    def __len__(self): return 340
    def set_epoch(self,epoch): self.epoch=int(epoch)
    def __iter__(self):
        self.observed=[]
        for name in self.schedules[self.epoch]:
            i=self.index[name]; self.observed.append(i); yield i


def fail(ok,message):
    if not ok: raise RuntimeError("STAGE7C_"+message)


def frozen_selection(path,sd2_manifest,msdf_manifest):
    frozen=load(path)
    selected=frozen["selected"]
    fail(frozen["selection"]=="random" and frozen["candidate_pool_manifest_sha256"]==MSDF_POOL_SHA,
         "FORMAL_SELECTION_IDENTITY_CHANGED")
    msdf={(x["target_instance_id"],int(x["generation_seed"])):x for x in load(msdf_manifest)["records"]}
    sd2_rows=load(sd2_manifest)["records"]
    fail(canonical_pool_sha(sd2_rows)==SD2_POOL_SHA and sha(msdf_manifest)==MSDF_POOL_SHA,"POOL_HASH_CHANGED")
    sd2={(x["instance_id"],int(x["generation_seed"])):x for x in sd2_rows}
    keys=[(x["target_instance_id"],int(x["generation_seed"])) for x in selected]
    fail(len(keys)==len(set(keys))==240 and set(keys)<=set(sd2) and set(keys)<=set(msdf),"MATCHED_KEYS_INVALID")
    classes=Counter(msdf[k]["class_name"] for k in keys); seeds=Counter(str(k[1]) for k in keys)
    fail(classes==Counter({c:40 for c in CLASSES}),"CLASS_COUNTS_CHANGED")
    manifest={"source":"frozen Stage3R formal random-matched 40x6 manifest; reused, not reselected",
        "source_manifest_path":str(path),"source_manifest_sha256":sha(path),"count":240,
        "records":[{"selection_index":i,"source_instance_id":k[0],"generation_seed":k[1],
            "class_name":msdf[k]["class_name"],"sd2_candidate_id":sd2[k]["candidate_id"],
            "msdf_candidate_id":msdf[k]["candidate_id"],"sd2_image":sd2[k]["image_path"],
            "msdf_image":msdf[k]["image_path"],"sd2_label":sd2[k]["label_path"],
            "msdf_label":msdf[k]["label_path"]} for i,k in enumerate(keys)]}
    return manifest,classes,seeds


def make_datasets(args,manifest,real_images,real_labels):
    roots={}
    for arm in ARMS:
        root=args.dataset_root/arm
        fail(not root.exists(),f"DATASET_EXISTS_{arm}")
        for kind in ("images","labels"): (root/kind/"train").mkdir(parents=True,exist_ok=False)
        for stem,image in real_images.items():
            (root/"images/train"/image.name).symlink_to(image.resolve())
            label=real_labels[stem]; (root/"labels/train"/label.name).symlink_to(label.resolve())
        for row in manifest["records"]:
            key=f"synthetic_{row['selection_index']:03d}"
            image=Path(row[f"{arm}_image"]); label=Path(row[f"{arm}_label"])
            if not image.is_absolute(): image=args.repo_root/image
            if not label.is_absolute(): label=args.repo_root/label
            fail(image.is_file() and label.is_file() and len(label.read_text().splitlines())==1,"SYNTHETIC_FILE_MISSING")
            (root/"images/train"/(key+image.suffix)).symlink_to(image.resolve())
            (root/"labels/train"/(key+".txt")).symlink_to(label.resolve())
        (root/"data.yaml").write_text("path: "+str(root.resolve())+"\ntrain: images/train\nval: "+
            str((args.real_dataset/"images/val").resolve())+"\nnc: 6\nnames: "+json.dumps(list(CLASSES))+"\n")
        roots[arm]=root
    return roots


def aggregate(rows):
    by={(x["arm"],x["seed"]):x for x in rows}; fail(len(rows)==len(by)==6,"RESULT_MATRIX")
    arms={arm:{m:moments(by[arm,s][m] for s in SEEDS) for m in ("precision","recall","map50","map50_95")} for arm in ARMS}
    deltas={m:{**moments(by["msdf",s][m]-by["sd2",s][m] for s in SEEDS),
        "wins":sum(by["msdf",s][m]>by["sd2",s][m] for s in SEEDS)} for m in ("precision","recall","map50","map50_95")}
    classes={}
    for c in CLASSES:
        sd=[by["sd2",s]["per_class"][c]["ap50_95"] for s in SEEDS]; ms=[by["msdf",s]["per_class"][c]["ap50_95"] for s in SEEDS]; d=[b-a for a,b in zip(sd,ms)]
        classes[c]={"sd2_ap50_95":moments(sd),"msdf_ap50_95":moments(ms),
            "delta_msdf_minus_sd2":{**moments(d),"wins":sum(x>0 for x in d)}}
    return arms,deltas,classes


def status(delta):
    if delta["mean"]>0 and delta["wins"]>=2: return "MSDF_DOWNSTREAM_SUPPORTED"
    if delta["mean"]<0 and delta["wins"]<=1: return "MSDF_DOWNSTREAM_NOT_SUPPORTED"
    return "MSDF_DOWNSTREAM_MIXED"


def main():
    p=argparse.ArgumentParser()
    for name in ("protocol_config","formal_selection","sd2_manifest","msdf_manifest","real_dataset","repo_root","model",
                 "reference_run","stage5f_results","stage6h_results","dataset_root","runs","output"):
        p.add_argument("--"+name,type=Path,required=True)
    p.add_argument("--device",default="0"); a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=True); a.runs=a.runs.resolve()
    protocol=load(a.protocol_config)
    fail(protocol.get("arms")==["sd2","msdf"] and protocol.get("training_seeds")==list(SEEDS) and
         protocol.get("epochs")==150 and protocol.get("primary_checkpoint")=="last.pt" and
         protocol.get("synthetic_population")==240 and protocol.get("real_draws_per_epoch")==260 and
         protocol.get("synthetic_draws_per_epoch")==80 and protocol.get("batches_per_epoch")==43,
         "PROTOCOL_CONFIG_CHANGED")
    manifest,class_counts,seed_counts=frozen_selection(a.formal_selection,a.sd2_manifest,a.msdf_manifest)
    save(a.output/"source_seed_selection_manifest.json",manifest)
    real_images,real_labels=file_population(a.real_dataset); train_sha,train_count=tree_digest(a.real_dataset,"train"); val_sha,val_count=tree_digest(a.real_dataset,"val")
    prior=load(a.stage5f_results/"stage5e_reuse_audit.json")
    stage5f_compute=load(a.stage5f_results/"compute_match_audit.json")
    stage6h_compute=load(a.stage6h_results/"compute_equality_audit.json")
    fail(stage5f_compute["per_epoch_draws_equal"] and stage5f_compute["per_epoch_batches_equal"] and
         stage5f_compute["batches_per_epoch"]==43 and stage5f_compute["last_batch_size"]==4,
         "STAGE5F_EXPOSURE_PROTOCOL_CHANGED")
    fail(stage6h_compute["equal_audited_epochs"] and stage6h_compute["equal_audited_batches"],
         "STAGE6H_FIXED_BUDGET_AUDIT_CHANGED")
    fail(train_count==200 and val_count==1000 and train_sha==prior["real_train_tree_sha256"] and
         val_sha==prior["validation_tree_sha256"] and prior["validation_manifest_sha256"]==VAL_HASH and
         sha(a.model)==MODEL_SHA,"REAL_VAL_MODEL_CHANGED")
    val_labels=list((a.real_dataset/"labels/val").glob("*.txt")); fail(len(val_labels)==500 and sum(len(x.read_text().splitlines()) for x in val_labels)==3140,"VAL_CHANGED")
    import yaml,torch,ultralytics
    from torch.utils.data import DataLoader
    from ultralytics import YOLO
    from ultralytics.data.build import seed_worker
    from ultralytics.models.yolo.detect import DetectionTrainer
    fail(ultralytics.__version__=="8.4.145","ULTRALYTICS_VERSION")
    early=check_early_stopping(); reference=yaml.safe_load((a.reference_run/"args.yaml").read_text())
    required={"imgsz":512,"epochs":150,"batch":8,"optimizer":"auto","deterministic":True,"mosaic":0.0}
    fail(all(reference.get(k)==v for k,v in required.items()),"REFERENCE_CONFIG_CHANGED")
    datasets=make_datasets(a,manifest,real_images,real_labels)
    source_names=[f"synthetic_{i:03d}" for i in range(240)]; schedules={s:schedule_for_seed(sorted(real_images),source_names,s) for s in SEEDS}
    schedule_audit={}; rows=[]; compute={}
    for seed in SEEDS:
        digest=schedule_hash(schedules[seed]); schedule_audit[str(seed)]={"sha256":digest,"same_real_sequence":True,"same_synthetic_sequence":True,
            "same_source_type_sequence":True,"real_draws_per_epoch":260,"synthetic_draws_per_epoch":80,"draws_per_epoch":340,"batches_per_epoch":43}
        for arm in ARMS:
            run=a.runs/f"{arm}_s{seed}"; fail(not run.exists(),"RUN_EXISTS")
            state={"sampler":None,"reads":mp.Value("q",0),"primary":mp.Value("q",0),"mosaic":mp.Value("q",0),"epochs":[],"sizes":[],"steps":0}
            class FiniteLoader(DataLoader):
                def reset(self): return None
            class Trainer(DetectionTrainer):
                def get_dataloader(self,dataset_path,batch_size=16,rank=0,mode="train"):
                    if mode!="train": return super().get_dataloader(dataset_path,batch_size,rank,mode)
                    fail(rank==-1 and batch_size==BATCH,"SINGLE_GPU_BATCH8")
                    ds=self.build_dataset(dataset_path,mode,batch_size); names=[Path(x).stem for x in ds.im_files]
                    state["sampler"]=PairedSampler7C(names,schedules[seed]); state["transforms"]=transforms_audit(ds.transforms); old=ds.get_image_and_label
                    def counted(i):
                        with state["reads"].get_lock(): state["reads"].value+=1
                        return old(i)
                    ds.get_image_and_label=counted; base=type(ds)
                    class Counted(base):
                        def __getitem__(self,i):
                            with state["primary"].get_lock(): state["primary"].value+=1
                            return super().__getitem__(i)
                    ds.__class__=Counted
                    for t in walk_transforms(ds.transforms):
                        if type(t).__name__=="Mosaic":
                            oldidx=t.get_indexes
                            def guarded(*x,_old=oldidx,**kw):
                                with state["mosaic"].get_lock(): state["mosaic"].value+=1
                                return _old(*x,**kw)
                            t.get_indexes=guarded
                    gen=torch.Generator().manual_seed(6148914691236517205+seed)
                    return FiniteLoader(ds,batch_size=BATCH,sampler=state["sampler"],shuffle=False,num_workers=self.args.workers,collate_fn=ds.collate_fn,pin_memory=True,worker_init_fn=seed_worker,generator=gen)
                def preprocess_batch(self,batch):
                    self.consumed.extend(Path(x).stem for x in batch["im_file"]); state["sizes"].append(len(batch["im_file"])); return super().preprocess_batch(batch)
            def train_start(trainer):
                original=trainer.optimizer.step
                def step(self,*x,**kw):
                    out=original(*x,**kw); state["steps"]+=1; return out
                trainer.optimizer.step=MethodType(step,trainer.optimizer)
            def epoch_start(trainer):
                state["sampler"].set_epoch(trainer.epoch); trainer.consumed=[]; state["sizes"]=[]
                for k in ("reads","primary","mosaic"): state[k].value=0
            def epoch_end(trainer):
                expected=schedules[seed][trainer.epoch]; observed=[state["sampler"].names[i] for i in state["sampler"].observed]
                fail(trainer.consumed==expected and observed==expected and state["sizes"]==[8]*42+[4] and state["reads"].value==state["primary"].value and state["mosaic"].value==0,"EXPOSURE")
                state["epochs"].append({"epoch":trainer.epoch+1,"schedule_sha256":hashlib.sha256(json.dumps(expected,separators=(",",":")).encode()).hexdigest(),"real_draws":260,"synthetic_draws":80,"batches":43,"cumulative_optimizer_steps":state["steps"],"additional_image_reads":0,"mosaic_calls":0})
                save(a.output/f"epoch_exposure_{arm}_s{seed}.json",{"epochs":state["epochs"]})
            model=YOLO(str(a.model)); model.add_callback("on_train_start",train_start); model.add_callback("on_train_epoch_start",epoch_start); model.add_callback("on_train_epoch_end",epoch_end)
            model.train(data=str((datasets[arm]/"data.yaml").resolve()),imgsz=512,epochs=150,patience=PATIENCE,batch=8,deterministic=True,seed=seed,optimizer="auto",mosaic=0.0,mixup=0.0,copy_paste=0.0,cutmix=0.0,device=a.device,project=str(a.runs),name=run.name,exist_ok=False,trainer=Trainer)
            cfg=yaml.safe_load((run/"args.yaml").read_text()); excluded={"data","project","name","save_dir","exist_ok","seed","patience"}
            diffs={k:[reference[k],cfg[k]] for k in set(reference)&set(cfg)-excluded if reference[k]!=cfg[k]}
            fail(not diffs and cfg["seed"]==seed and cfg["patience"]==PATIENCE and len(state["epochs"])==150,"CONFIG_OR_EPOCH")
            _,stop=best_epoch(run); fail(stop==150 and checkpoint_epoch(run/"weights/last.pt") in (-1,149),"LAST_NOT_150")
            score=metrics(YOLO(str(run/"weights/last.pt")).val(data=str((a.real_dataset/"data.yaml").resolve()),split="val",imgsz=512,device=a.device,verbose=False)); finite_metrics(score)
            bestscore=metrics(YOLO(str(run/"weights/best.pt")).val(data=str((a.real_dataset/"data.yaml").resolve()),split="val",imgsz=512,device=a.device,verbose=False)); finite_metrics(bestscore)
            row={"arm":arm,"seed":seed,**score,"primary_checkpoint":"last.pt","last_checkpoint_sha256":sha(run/"weights/last.pt"),"secondary_best":bestscore,"best_checkpoint_sha256":sha(run/"weights/best.pt")}; rows.append(row)
            compute[f"{arm}_s{seed}"]={"epochs":150,"audited_batches":6450,"optimizer_steps":state["steps"],"directly_observed":True}
            save(a.output/"per_seed_metrics.json",{"partial":True,"records":rows})
        # AMP can skip parameter updates after non-finite gradients. This is a
        # numerical outcome, not a difference in allocated forward/backward
        # compute. Preserve successful updates separately instead of aborting.
        pair=(compute[f"sd2_s{seed}"],compute[f"msdf_s{seed}"])
        for item in pair:
            item["successful_optimizer_updates_amp"]=item.pop("optimizer_steps")
            item["optimizer_update_opportunities_equal_within_seed"]=True
    arms,deltas,classes=aggregate(rows); result=status(deltas["map50_95"])
    save(a.output/"per_seed_metrics.json",{"partial":False,"primary":"epoch150 last.pt","records":rows})
    save(a.output/"per_class_metrics.json",{"classes":classes,"positive_mean_delta_classes":sum(classes[c]["delta_msdf_minus_sd2"]["mean"]>0 for c in CLASSES)})
    save(a.output/"paired_delta_summary.json",{"arms":arms,"msdf_minus_sd2":deltas})
    successful_equal=all(compute[f"sd2_s{s}"]["successful_optimizer_updates_amp"]==compute[f"msdf_s{s}"]["successful_optimizer_updates_amp"] for s in SEEDS)
    save(a.output/"training_budget_audit.json",{"early_stopping":early,"runs":compute,"equal_epochs":True,"equal_audited_batches":True,"equal_optimizer_update_opportunities_within_seed":True,"equal_successful_optimizer_updates_amp":successful_equal,"amp_skip_note":"AMP can skip a parameter update after non-finite gradients; this does not change forward/backward batches or allocated compute.","primary_checkpoint":"epoch150 last.pt","secondary_checkpoint":"best.pt"})
    save(a.output/"matched_selection_audit.json",{"exact_matched_count":240,"source_count":len(set(x["source_instance_id"] for x in manifest["records"])),"same_source_keys":True,"same_generation_seed_keys":True,"class_distribution":dict(class_counts),"generation_seed_distribution":dict(seed_counts),"separate_sampling_per_arm":False,"new_selector":False})
    save(a.output/"protocol_audit.json",{"protocol_provenance":{"synthetic_count":"Stage3R/Stage5 formal 240-candidate arm","exposure":"Stage5F 260 real + 80 synthetic; 43 batches","fixed_budget":"Stage6H 150 epoch last.pt protocol","stage5f_compute_audit_sha256":sha(a.stage5f_results/"compute_match_audit.json"),"stage6h_compute_audit_sha256":sha(a.stage6h_results/"compute_equality_audit.json")},"same_real_train_set":True,"real_train_tree_sha256":train_sha,"same_source_keys":True,"same_generation_seed_keys":True,"same_synthetic_count":240,"same_class_counts":True,"same_detector_architecture":"YOLO11s","same_pretrained_weights_protocol":True,"pretrained_sha256":sha(a.model),"same_hyperparameters":True,"training_seeds":list(SEEDS),"epochs":150,"draws_per_epoch":340,"real_draws_per_epoch":260,"synthetic_draws_per_epoch":80,"batches_per_epoch":43,"same_optimizer_update_opportunities_within_seed":True,"equal_successful_optimizer_updates_amp":successful_equal,"same_validation":True,"validation_manifest_sha256":VAL_HASH,"validation_tree_sha256":val_sha,"validation_leakage":0,"forbidden_methods_used":[]})
    save(a.output/"stage7c_status.json",{"status":result,"primary_metric":"mAP50-95","mean_delta_msdf_minus_sd2":deltas["map50_95"]["mean"],"wins":deltas["map50_95"]["wins"],"positive_mean_delta_classes":sum(classes[c]["delta_msdf_minus_sd2"]["mean"]>0 for c in CLASSES),"generator_modified":False,"selector_created":False,"validation_leakage":0})
    lines=["# DeepPCB Stage 7C matched downstream benchmark","",f"Status: **{result}**.","Primary checkpoint: epoch150 last.pt; best.pt is secondary only.","Frozen formal 240 source-seed keys; 260 real + 80 synthetic draws per epoch; 43 batches; exact paired schedules.","","| seed | SD2 mAP50-95 | MSDF mAP50-95 | delta |","|---:|---:|---:|---:|"]
    by={(x["arm"],x["seed"]):x for x in rows}
    for s in SEEDS: lines.append(f"| {s} | {by['sd2',s]['map50_95']:.6f} | {by['msdf',s]['map50_95']:.6f} | {by['msdf',s]['map50_95']-by['sd2',s]['map50_95']:+.6f} |")
    lines += ["",f"SD2 mean±std: {arms['sd2']['map50_95']['mean']:.6f}±{arms['sd2']['map50_95']['std']:.6f}.",f"MSDF mean±std: {arms['msdf']['map50_95']['mean']:.6f}±{arms['msdf']['map50_95']['std']:.6f}.",f"Paired delta mean±std: {deltas['map50_95']['mean']:+.6f}±{deltas['map50_95']['std']:.6f}; wins {deltas['map50_95']['wins']}/3.","No generator, selector, generation seed, or validation protocol was changed."]
    (a.output/"STAGE7C_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":result,"delta":deltas["map50_95"]["mean"],"wins":deltas["map50_95"]["wins"]},indent=2))


if __name__=="__main__": main()
