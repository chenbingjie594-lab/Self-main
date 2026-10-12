"""Independent CPU review of downloaded Stage22A records; no model or image read."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
GROUPS = ("real", "A", "B", "C")
CLASSES = ("flash", "black")
REQUIRED = (
    "diagnostic_protocol.json", "diagnostic_data_manifest.json", "model_parameter_binding.json",
    "loss_distribution_audit.json", "gradient_norm_audit.json", "gradient_direction_audit.json",
    "gradient_conflict_audit.json", "matched_triplet_audit.json", "feature_distribution_audit.json",
    "classwise_mechanism_audit.json", "mechanism_candidate_summary.json", "stage22a_status.json",
    "file_access_guard.json", "STAGE22A_REPORT.md",
)


def require(condition, message):
    if not condition:
        raise RuntimeError("STAGE22A_REVIEW_FAILED: " + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def bad_constant(value):
        raise ValueError("nonfinite JSON token: " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad_constant)


def check_finite(value):
    if isinstance(value, dict):
        for child in value.values(): check_finite(child)
    elif isinstance(value, list):
        for child in value: check_finite(child)
    elif isinstance(value, float):
        require(math.isfinite(value), "nonfinite numeric value")


def near(a, b):
    return math.isclose(float(a), float(b), rel_tol=1e-7, abs_tol=1e-7)


def verify_summary(values, saved):
    a = np.asarray(list(values), dtype=np.float64)
    require(saved["n"] == len(a), "summary n")
    if not len(a): return
    q = np.quantile(a, [0, .25, .5, .75, .9, .95, 1])
    expected = {"mean":a.mean(), "median":q[2], "std":a.std(ddof=0), "iqr":q[3]-q[1],
        "q0":q[0], "q25":q[1], "q75":q[3], "q90":q[4], "q95":q[5], "q100":q[6]}
    require(all(near(v, saved[k]) for k,v in expected.items()), "summary differs from raw records")


def review(root):
    root = Path(root).resolve()
    require(all((root/name).is_file() for name in REQUIRED), "missing required artifact")
    data = {name:load(root/name) for name in REQUIRED if name.endswith(".json")}
    for value in data.values(): check_finite(value)
    cfg = load(REPO/"configs/plastic_bomo_stage22a.json")
    protocol = data["diagnostic_protocol.json"]
    require(protocol["frozen_protocol"] == cfg, "protocol semantic mismatch")
    require(protocol["source_sha256"] == sha(REPO/"configs/plastic_bomo_stage22a.json"), "protocol source bytes")
    prior = REPO/cfg["stage21b_run"]
    manifest = data["diagnostic_data_manifest.json"]
    require(manifest["stage21b_split_manifest_sha256"] == sha(prior/"stage21b_split_manifest.json"), "split binding")
    require(manifest["stage21b_synthetic_allowlist_sha256"] == sha(prior/"stage21b_synthetic_allowlist_v2.json"), "slot binding")
    source_real = [x for x in load(prior/"stage21b_split_manifest.json")["records"] if x["internal_split"]=="internal_train"]
    source_slots = load(prior/"stage21b_synthetic_allowlist_v2.json")["records"]
    require(len(source_real)==109 and len(source_slots)==61, "source population")
    require(set(manifest["real_role_ids"]) == {x["role_id"] for x in source_real}, "real IDs")
    require({x["slot_id"] for x in manifest["synthetic_slots"]} == {x["slot_id"] for x in source_slots}, "slot IDs")
    require(manifest["real_class_memberships"] == {x["role_id"]:sorted(set(x["class_instances"])) for x in source_real}, "real membership")
    require(manifest["real_annotation_instances"] == {x["role_id"]:x["class_instances"] for x in source_real}, "real target counts")
    source_slot_map = {x["slot_id"]:x for x in source_slots}
    for slot in manifest["synthetic_slots"]:
        source = source_slot_map[slot["slot_id"]]
        require(all(slot[k]==source[k] for k in ("role_id","class","source_group_id","conditioning_normal_id")), "matched donor/conditioning roles")
        for bank in GROUPS[1:]:
            require(all(slot["banks"][bank][k]==source["banks"][bank][k] for k in ("image_sha256","label_sha256","sample_seed")), "slot pixel/target/RNG hashes")
    expected, bound = {}, {}
    for bank in GROUPS[1:]:
        require(manifest["role_binding_sha256"][bank] == sha(prior/"train_inputs"/bank/"role_binding.json"), "role binding")
        bindings = load(prior/"train_inputs"/bank/"role_binding.json")["roles"]
        for x in source_slots:
            key = (bank,x["role_id"])
            expected[key] = ([x["class"]],x["slot_id"])
            bound[key] = bindings[x["role_id"]]
        if bank=="A":
            for x in source_real:
                expected[("real",x["role_id"])] = (sorted(set(x["class_instances"])),None)
                bound[("real",x["role_id"])] = bindings[x["role_id"]]
    losses, norms, directions = [data[name]["records"] for name in (
        "loss_distribution_audit.json", "gradient_norm_audit.json", "gradient_direction_audit.json")]
    maps = [{(r["bank"],r["role_id"]):r for r in records} for records in (losses,norms,directions)]
    for records,mapping in zip((losses,norms,directions),maps):
        require(len(records)==292 and len(mapping)==292 and set(mapping)==set(expected), "raw record population")
        for key,row in mapping.items():
            require(row["class_memberships"]==expected[key][0] and row["slot_id"]==expected[key][1], "raw role/class/slot")
    for key,(l,n,d) in ((key,tuple(mapping[key] for mapping in maps)) for key in expected):
        require(math.isclose(l["loss"]["total_loss"],sum(l["loss"][k] for k in ("box_loss","cls_loss","dfl_loss")),
            rel_tol=1e-6,abs_tol=1e-5), "native total loss")
        require(near(n["selected_concat"],math.sqrt(sum(v*v for v in n["by_group"].values()))), "concatenated gradient norm")
        require(d["global_conflict"] == (d["cosine_real_global"]<0), "global sign-conflict")
        require(all(d["same_class_conflict"][c] == (v<0) for c,v in d["cosine_real_same_class"].items()), "class sign-conflict")
        require(abs(d["cosine_real_global"]) <= 1.000001 and all(abs(v)<=1.000001 for v in d["cosine_real_same_class"].values()), "cosine domain")
    for bank in GROUPS:
        for cls in (None,)+CLASSES:
            key = "null" if cls is None else cls
            select = lambda rows:[r for r in rows if r["bank"]==bank and (cls is None or cls in r["class_memberships"])]
            ls,ns,ds = [select(rows) for rows in (losses,norms,directions)]
            for field,saved in data["loss_distribution_audit.json"]["summary"][bank][key].items():
                verify_summary((r["loss"][field] for r in ls),saved)
            for field,saved in data["gradient_norm_audit.json"]["summary"][bank][key].items():
                verify_summary((r["selected_concat"] if field=="gradient_norm_selected_concat" else r["by_group"][field] for r in ns),saved)
            saved = data["gradient_direction_audit.json"]["summary"][bank][key]
            verify_summary((r["cosine_real_global"] for r in ds),saved["global"])
            conflict = data["gradient_conflict_audit.json"]["by_bank_and_class"][bank][key]
            require(conflict["global_conflict_count"]==sum(r["global_conflict"] for r in ds), "global conflict count")
            require(near(conflict["global_conflict_fraction"],sum(r["global_conflict"] for r in ds)/len(ds)), "global conflict fraction")
            if cls:
                verify_summary((r["cosine_real_same_class"][cls] for r in ds),saved["same_class"])
                require(conflict["same_class_conflict_count"]==sum(r["same_class_conflict"][cls] for r in ds), "class conflict count")
                require(near(conflict["same_class_conflict_fraction"],sum(r["same_class_conflict"][cls] for r in ds)/len(ds)), "class conflict fraction")
    triplet = data["matched_triplet_audit.json"]
    require(len(triplet["records"])==61 and {r["slot_id"] for r in triplet["records"]}=={r["slot_id"] for r in source_slots}, "triplet population")
    for row in triplet["records"]:
        slot = next(x for x in source_slots if x["slot_id"]==row["slot_id"])
        require(row["class"]==slot["class"], "triplet class")
        keys = [(bank,slot["role_id"]) for bank in GROUPS[1:]]
        values = {"total_loss":[maps[0][key]["loss"]["total_loss"] for key in keys],
            "gradient_norm":[maps[1][key]["selected_concat"] for key in keys],
            "cosine_global":[maps[2][key]["cosine_real_global"] for key in keys],
            "cosine_same_class":[maps[2][key]["cosine_real_same_class"][slot["class"]] for key in keys]}
        for name,vals in values.items():
            require(row["A_B_C_values"][name]==vals, "triplet raw values")
            require(near(row["within_slot"][name]["range"],max(vals)-min(vals)), "triplet range")
            require(near(row["within_slot"][name]["std"],np.std(vals,ddof=0)), "triplet std")
        require(row["same_label_sha256"]==slot["banks"]["A"]["label_sha256"], "triplet label")
        require(row["sample_seeds"]=={b:slot["banks"][b]["sample_seed"] for b in GROUPS[1:]}, "triplet RNG seeds")
    for cls in (None,)+CLASSES:
        key = "null" if cls is None else cls
        subset = [r for r in triplet["records"] if cls is None or r["class"]==cls]
        for field,stats in triplet["summary"][key].items():
            for stat,saved in stats.items(): verify_summary((r["within_slot"][field][stat] for r in subset),saved)
    model = data["model_parameter_binding.json"]
    init = load(prior/"initialization_reference.json")
    require(model["stage21b_initialization_reference_sha256"]==sha(prior/"initialization_reference.json"), "initialization binding")
    require(model["pretrained_sha256"]==init["pretrained_sha256"] and model["seed"]==42 and model["full_parameter_hash_equal_frozen_reference"], "public initialization")
    for group,name in cfg["selected_parameters"].items():
        require(model["selected_parameters"][group]["name"]==name and model["selected_parameters"][group]["initialization_sha256"]==init["parameters"][name], "selected parameter binding")
    require(model["feature_layers"]==cfg["feature_layers"] and model["training_checkpoints_used"]==0, "feature/trained checkpoint binding")
    feature = data["feature_distribution_audit.json"]
    feature_bounds = {}
    for layer,classes in feature["summary"].items():
        for cls,row in classes.items():
            for bank,stats in [("real",row["real"]),*row["banks"].items()]:
                count = sum(cls in x["class_instances"] for x in source_real) if bank=="real" else sum(x["class"]==cls for x in source_slots)
                require(stats["n"]==count and stats["covariance_trace"]>=0 and 0<=stats["effective_rank"]<=count-1+1e-6, "feature sample count/rank bound")
            feature_bounds[layer+"/"+cls] = {"real_max_sample_covariance_rank":row["real"]["n"]-1,
                "synthetic_max_sample_covariance_rank":row["banks"]["A"]["n"]-1}
    candidates = data["mechanism_candidate_summary.json"]
    require(not candidates["pearson_spearman_computed"] and not candidates["utility_predictor_claim"], "three-bank interpretation")
    require(not any(row.get("all_predeclared_descriptive_gates_pass",False) for row in candidates["candidates"].values()), "candidate unexpectedly passed")
    require(all(not candidates["candidates"][name]["A_real_synthetic_direction"] for name in ("loss_distribution","gradient_norm","gradient_direction_conflict")), "A gates")
    for candidate, artifact, metric in (("loss_distribution","loss_distribution_audit.json","total_loss"),
        ("gradient_norm","gradient_norm_audit.json","gradient_norm_selected_concat")):
        gate_per_class=[]
        for cls in CLASSES:
            reference=data[artifact]["summary"]["real"][cls][metric]
            vals={bank:data[artifact]["summary"][bank][cls][metric]["median"] for bank in GROUPS[1:]}
            saved=candidates["candidates"][candidate]["by_class"][cls]
            require(near(saved["real"],reference["median"]) and all(near(saved["synthetic_median_or_fraction"][bank],v) for bank,v in vals.items()), "candidate source values")
            separated=all(v < reference["q25"] or v > reference["q75"] for v in vals.values())
            require(saved["clearly_separated_from_real_control"]==separated, "candidate IQR separation")
            gate_per_class.append(separated)
        require(candidates["candidates"][candidate]["A_real_synthetic_direction"]==all(gate_per_class), "candidate Gate A recomputation")
    for cls in CLASSES:
        mirror=data["classwise_mechanism_audit.json"][cls]
        for bank in GROUPS:
            require(mirror["loss"][bank]==data["loss_distribution_audit.json"]["summary"][bank][cls], "classwise loss")
            require(mirror["gradient_norm"][bank]==data["gradient_norm_audit.json"]["summary"][bank][cls], "classwise norm")
            require(mirror["direction"][bank]==data["gradient_direction_audit.json"]["summary"][bank][cls], "classwise direction")
            require(mirror["conflict"][bank]==data["gradient_conflict_audit.json"]["by_bank_and_class"][bank][cls], "classwise conflict")
        require(mirror["matched_triplets"]==triplet["summary"][cls], "classwise RNG triplets")
        require(mirror["feature"]=={layer:rows[cls] for layer,rows in feature["summary"].items()}, "classwise feature")
    status = data["stage22a_status.json"]
    require(status["status"]=="SYNTHETIC_OPTIMIZATION_MECHANISM_NOT_IDENTIFIED" and status["probe_forwards"]==292 and status["probe_backwards"]==292, "status")
    for field in ("generator_training_count","generation_count","detector_training_count","optimizer_step_count",
        "official_final_eval_image_read","official_final_eval_label_read","official_final_eval_forward",
        "internal_dev_content_read","BootstrapGuard","DeepPCB"):
        require(status[field]==0, "forbidden action counter: "+field)
    require(not status["stage22b_auto_start"], "next stage auto start")
    guard = data["file_access_guard.json"]
    require(guard==status["guard_counts"] and not guard["internal_dev_content_allowed"], "guard binding")
    require(all(value==0 for key,value in guard.items() if key!="internal_dev_content_allowed"), "guard nonzero access")
    for name,field in (("model_parameter_binding.json","model_parameter_binding_sha256"),
        ("diagnostic_protocol.json","diagnostic_protocol_sha256"),("diagnostic_data_manifest.json","data_manifest_sha256")):
        require(sha(root/name)==status[field], "downloaded artifact hash: "+name)
    report = {"status":"STAGE22A_DOWNLOADED_RESULT_REVIEW_PASS", "formal_status":status["status"],
        "review_scope":"CPU metadata and raw-record recomputation; no image decode or detector forward",
        "raw_records":292,"matched_triplets":61,"raw_population_and_class_bindings":"PASS",
        "loss_norm_direction_conflict_summary_recomputation":"PASS", "triplet_recomputation":"PASS",
        "protocol_and_initialization_bindings":"PASS", "forbidden_action_counters":"PASS",
        "gate_A_failed_for_all_three_numeric_candidates":True,"not_identified_even_without_annotation_count_guard":True,
        "synthetic_negative_cosine_conflicts":sum(r["global_conflict"]+sum(r["same_class_conflict"].values()) for r in directions if r["bank"]!="real"),
        "all_global_cosine_min":min(r["cosine_real_global"] for r in directions),
        "all_same_class_cosine_min":min(v for r in directions for v in r["cosine_real_same_class"].values()),
        "median_cls_fraction_of_total_loss":float(np.median([r["loss"]["cls_loss"]/r["loss"]["total_loss"] for r in losses])),
        "median_backbone_fraction_of_selected_gradient_squared_norm":float(np.median([r["by_group"]["backbone_late"]**2/r["selected_concat"]**2 for r in norms])),
        "feature_sample_covariance_rank_bounds":feature_bounds,
        "feature_raw_vectors_available":False,"feature_moments_independently_recomputed":False,
        "feature_interpretation":"Lower synthetic effective rank cannot establish lower intrinsic diversity because n differs (63/32 Flash, 46/29 Black).",
        "direction_interpretation":"No measured negative cosine at this frozen initialization; this does not establish absence of conflict at later training states or in every parameter group.",
        "executed_server_script_sha256_in_downloaded_artifacts":None,
        "model_immutability_evidence":"Completed runner code verifies parameters and restored buffers before writing final status; no separate post-probe full tensor hashes were exported.",
        "original_downloaded_report_sha256":sha(root/"STAGE22A_REPORT.md")}
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=REPO/"results_for_gpt/plastic_bomo_stage22a_failure_mechanism")
    a=p.parse_args(); result=review(a.output)
    (a.output/"review_integrity_audit.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    ledger={"scope":"downloaded formal outputs and CPU review", "files":{
        path.name:{"sha256":sha(path),"bytes":path.stat().st_size} for path in sorted(a.output.iterdir())
        if path.is_file() and path.name!="result_artifact_manifest.json"},"official_content_read":0,"model_forwards":0}
    (a.output/"result_artifact_manifest.json").write_text(json.dumps(ledger,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"review":result["status"],"formal_status":result["formal_status"],"records":292,"triplets":61},indent=2))


if __name__=="__main__": main()
