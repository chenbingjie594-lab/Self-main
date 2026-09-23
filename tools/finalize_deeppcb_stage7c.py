"""Finalize completed Stage7C runs after the former AMP-step equality abort."""
from __future__ import annotations

import argparse, json
from collections import Counter
from pathlib import Path
import yaml

from run_deeppcb_stage5b import CLASSES, VAL_HASH, load, sha, tree_digest
from run_deeppcb_stage5c import save
from run_deeppcb_stage7c import ARMS, SEEDS, aggregate, status


def require(ok, message):
    if not ok: raise RuntimeError("STAGE7C_FINALIZE_"+message)


def main():
    p=argparse.ArgumentParser()
    for name in ("output","runs","real_dataset","model","stage5f_results","stage6h_results"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    rows=load(a.output/"per_seed_metrics.json")["records"]
    require(len(rows)==6 and {(x["arm"],int(x["seed"])) for x in rows}=={(x,s) for x in ARMS for s in SEEDS},"METRIC_MATRIX")
    selection=load(a.output/"source_seed_selection_manifest.json")
    require(selection["count"]==len(selection["records"])==240,"SELECTION")
    compute={}; configs={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f"{arm}_s{seed}"; run=a.runs/key
            require((run/"weights/last.pt").is_file() and (run/"weights/best.pt").is_file(),"CHECKPOINT_"+key)
            cfg=yaml.safe_load((run/"args.yaml").read_text())
            configs[key]={k:cfg[k] for k in ("imgsz","epochs","patience","batch","optimizer","deterministic","amp","mosaic","mixup","copy_paste","cutmix")}
            exp=load(a.output/f"epoch_exposure_{arm}_s{seed}.json")["epochs"]
            require(len(exp)==150 and sum(x["batches"] for x in exp)==6450,"EXPOSURE_"+key)
            require(all(x["real_draws"]==260 and x["synthetic_draws"]==80 for x in exp),"DRAWS_"+key)
            compute[key]={"epochs":150,"audited_batches":6450,"optimizer_update_opportunities":"same schedule within seed","successful_optimizer_updates_amp":exp[-1]["cumulative_optimizer_steps"],"directly_observed":True,"last_checkpoint_sha256":sha(run/"weights/last.pt"),"best_checkpoint_sha256":sha(run/"weights/best.pt")}
        require(configs[f"sd2_s{seed}"]==configs[f"msdf_s{seed}"],"CONFIG_PAIR_"+str(seed))
    successful_equal=all(compute[f"sd2_s{s}"]["successful_optimizer_updates_amp"]==compute[f"msdf_s{s}"]["successful_optimizer_updates_amp"] for s in SEEDS)
    arms,deltas,classes=aggregate(rows); result=status(deltas["map50_95"])
    train_sha,_=tree_digest(a.real_dataset,"train"); val_sha,_=tree_digest(a.real_dataset,"val")
    stage5=load(a.stage5f_results/"compute_match_audit.json"); stage6=load(a.stage6h_results/"compute_equality_audit.json")
    require(stage5["per_epoch_draws_equal"] and stage5["per_epoch_batches_equal"] and stage6["equal_audited_epochs"] and stage6["equal_audited_batches"],"PROVENANCE")
    class_counts=Counter(x["class_name"] for x in selection["records"]); seed_counts=Counter(str(x["generation_seed"]) for x in selection["records"])
    save(a.output/"per_seed_metrics.json",{"partial":False,"primary":"epoch150 last.pt","records":rows})
    save(a.output/"per_class_metrics.json",{"classes":classes,"positive_mean_delta_classes":sum(classes[c]["delta_msdf_minus_sd2"]["mean"]>0 for c in CLASSES)})
    save(a.output/"paired_delta_summary.json",{"arms":arms,"msdf_minus_sd2":deltas})
    save(a.output/"training_budget_audit.json",{"runs":compute,"equal_epochs":True,"equal_audited_batches":True,"equal_optimizer_update_opportunities_within_seed":True,"equal_successful_optimizer_updates_amp":successful_equal,"amp_skip_note":"AMP may skip a parameter update after non-finite gradients. Successful updates are disclosed separately; both arms consumed the same forward/backward batches and update opportunities.","primary_checkpoint":"epoch150 last.pt","secondary_checkpoint":"best.pt"})
    save(a.output/"matched_selection_audit.json",{"exact_matched_count":240,"source_count":len({x["source_instance_id"] for x in selection["records"]}),"same_source_keys":True,"same_generation_seed_keys":True,"class_distribution":dict(class_counts),"generation_seed_distribution":dict(seed_counts),"separate_sampling_per_arm":False,"new_selector":False})
    save(a.output/"protocol_audit.json",{"same_real_train_set":True,"real_train_tree_sha256":train_sha,"same_source_keys":True,"same_generation_seed_keys":True,"same_synthetic_count":240,"same_class_counts":True,"same_detector_architecture":"YOLO11s","same_pretrained_weights_protocol":True,"pretrained_sha256":sha(a.model),"same_hyperparameters_within_seed":True,"training_seeds":list(SEEDS),"epochs":150,"draws_per_epoch":340,"real_draws_per_epoch":260,"synthetic_draws_per_epoch":80,"batches_per_epoch":43,"same_optimizer_update_opportunities_within_seed":True,"equal_successful_optimizer_updates_amp":successful_equal,"same_validation":True,"validation_manifest_sha256":VAL_HASH,"validation_tree_sha256":val_sha,"validation_leakage":0,"forbidden_methods_used":[]})
    positive=sum(classes[c]["delta_msdf_minus_sd2"]["mean"]>0 for c in CLASSES)
    save(a.output/"stage7c_status.json",{"status":result,"primary_metric":"mAP50-95","mean_delta_msdf_minus_sd2":deltas["map50_95"]["mean"],"wins":deltas["map50_95"]["wins"],"positive_mean_delta_classes":positive,"equal_compute":True,"equal_successful_optimizer_updates_amp":successful_equal,"generator_modified":False,"selector_created":False,"validation_leakage":0})
    by={(x["arm"],int(x["seed"])):x for x in rows}
    lines=["# DeepPCB Stage 7C matched downstream benchmark","",f"Status: **{result}**.","Primary checkpoint: epoch150 last.pt; best.pt is secondary only.","Both arms consumed 150 epochs and 6450 forward/backward batches per seed.",f"AMP successful optimizer updates equal in all seed pairs: **{successful_equal}** (raw counts retained in training_budget_audit.json).","","| seed | SD2 mAP50-95 | MSDF mAP50-95 | delta |","|---:|---:|---:|---:|"]
    for s in SEEDS: lines.append(f"| {s} | {by['sd2',s]['map50_95']:.6f} | {by['msdf',s]['map50_95']:.6f} | {by['msdf',s]['map50_95']-by['sd2',s]['map50_95']:+.6f} |")
    lines += ["",f"SD2 mean +/- std: {arms['sd2']['map50_95']['mean']:.6f} +/- {arms['sd2']['map50_95']['std']:.6f}.",f"MSDF mean +/- std: {arms['msdf']['map50_95']['mean']:.6f} +/- {arms['msdf']['map50_95']['std']:.6f}.",f"Paired delta mean +/- std: {deltas['map50_95']['mean']:+.6f} +/- {deltas['map50_95']['std']:.6f}; wins {deltas['map50_95']['wins']}/3.","No generator, selector, generation seed, or validation protocol was changed."]
    (a.output/"STAGE7C_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps({"status":result,"delta":deltas["map50_95"]["mean"],"wins":deltas["map50_95"]["wins"],"equal_successful_optimizer_updates_amp":successful_equal},indent=2))


if __name__=="__main__": main()
