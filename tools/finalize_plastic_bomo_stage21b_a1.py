"""Read-only result calculations plus Stage21B report; no GPU/dev forwards."""
from collections import Counter
import argparse
import math
from pathlib import Path
import statistics
import sys

from stage21b_a1_common import *


def inspect_draw_logs(run):
    """Actual transformed input/target equality for every paired occurrence."""
    run = Path(run)
    base = load(run / "stage21b_base_protocol_snapshot.json")
    schedules = {m:load(run / (m + "_schedule_v2.json")) for m in ("uniform","curriculum")}
    expected = {(i,k) for i in range(170) for k in range(150)}
    banks, updates, init = {},{},set()
    for bank in BANKS:
        pair = {}
        for mode in ("uniform","curriculum"):
            arm = bank + "_" + mode; a = load(run / (arm + "_training.json")); s = schedules[mode]
            log = run / "logs" / (arm + "_draws.jsonl")
            require(a["status"] == "COMPLETE" and a["epochs"] == 150 and a["actual_draws"] == 25500
                    and a["attempted_optimizer_updates"] == 450
                    and a["successful_optimizer_updates"] + a["AMP_skipped_updates"] == 450
                    and a["schedule_sha256"] == sha(run / (mode + "_schedule_v2.json"))
                    and sha(log) == a["per_draw_log_sha256"],"STAGE21B_COMPUTE_PAIRING_INVALID")
            occurrences = {}; count = 0
            with log.open(encoding="utf-8") as stream:
                for line in stream:
                    x = json.loads(line); epoch,pos = divmod(count,170)
                    require(epoch < 150 and x["epoch"] == epoch and x["position"] == pos,"STAGE21B_COMPUTE_PAIRING_INVALID")
                    i,k = s["epoch_draws"][epoch][pos]
                    require((x["role_index"],x["occurrence_index"]) == (i,k)
                            and x["augmentation_seed"] == occurrence_seed(s["roles"][i],k,base)
                            and x["normalizer"] == (64 if pos < 128 else 42)
                            and math.isfinite(x["loss"]),"STAGE21B_COMPUTE_PAIRING_INVALID")
                    require((i,k) not in occurrences and isinstance(x["augmentation_sha256"],str)
                            and len(x["augmentation_sha256"]) == 64,"STAGE21B_AUGMENTATION_PAIRING_INVALID")
                    occurrences[(i,k)] = x["augmentation_sha256"]; count += 1
            require(count == 25500 and set(occurrences) == expected,"STAGE21B_COMPUTE_PAIRING_INVALID")
            require(len(a["epoch_records"]) == 150,"STAGE21B_COMPUTE_PAIRING_INVALID")
            actual_success = 0
            for e,row in enumerate(a["epoch_records"]):
                require(row["epoch_0based"] == e and row["draws"] == 170
                        and row["sequence_sha256"] == digest(s["epoch_draws"][e])
                        and [q["normalizer"] for q in row["attempts"]] == [64,64,42]
                        and [q["draws"] for q in row["attempts"]] == [64,64,42]
                        and [q["position"] for q in row["attempts"]] == [63,127,169],"STAGE21B_COMPUTE_PAIRING_INVALID")
                for q in row["attempts"]:
                    require(isinstance(q["optimizer_step_executed"],bool) and q["AMP_skip"] == (not q["optimizer_step_executed"]),
                            "STAGE21B_COMPUTE_PAIRING_INVALID")
                    actual_success += int(q["optimizer_step_executed"])
            require(actual_success == a["successful_optimizer_updates"],"STAGE21B_COMPUTE_PAIRING_INVALID")
            updates[arm] = a["successful_optimizer_updates"]; init.add(a["initialization_sha256"])
            pair[mode] = occurrences
        mismatches = [list(k) for k in sorted(expected) if pair["uniform"][k] != pair["curriculum"][k]]
        banks[bank] = {"occurrence_count":25500,"mismatched_occurrence_count":len(mismatches),
            "mismatched_occurrences":mismatches[:20],"augmentation_binding_sha256":digest([[*k,v] for k,v in sorted(pair["uniform"].items())])}
    require(len(init) == 1,"STAGE21B_INITIALIZATION_CHANGED")
    audit = {"status":"PASS" if all(x["mismatched_occurrence_count"] == 0 for x in banks.values()) else "FAIL",
        "banks":banks,"pairing_scope":"within each bank, every model input/target augmentation occurrence",
        "actual_occurrence_counts_equal":True,"actual_draws_each_arm":25500,"actual_optimizer_attempts_each_arm":450,
        "successful_updates":updates,"successful_updates_equal":len(set(updates.values())) == 1,
        "successful_update_variation_not_compensated":True,"same_initialization":True}
    save(run / "augmentation_pairing_audit.json",audit)
    require(audit["status"] == "PASS","STAGE21B_AUGMENTATION_PAIRING_INVALID")
    return audit


def calculate(metrics,cfg):
    check_cfg(cfg)
    require(set(metrics) == set(ARMS),"STAGE21B_METRICS_INCOMPLETE")
    for arm,x in metrics.items():
        require(x["status"] == "COMPLETE" and x["arm"] == arm and x["evaluation_image_count"] == 27
                and x["primary_checkpoint"] == "epoch150 last.pt"
                and x["dev_population"] == "DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR","STAGE21B_METRICS_INVALID")
        values = [x["map5095"]] + [x["per_class"][c]["ap5095"] for c in ("flash","black")]
        require(all(math.isfinite(v) and 0 <= v <= 1 for v in values),"STAGE21B_METRICS_INVALID")
    delta = {b:100*(metrics[b + "_curriculum"]["map5095"] - metrics[b + "_uniform"]["map5095"]) for b in BANKS}
    classes = {c:{b:100*(metrics[b + "_curriculum"]["per_class"][c]["ap5095"]
                            - metrics[b + "_uniform"]["per_class"][c]["ap5095"]) for b in BANKS} for c in ("flash","black")}
    means = {c:statistics.mean(v.values()) for c,v in classes.items()}
    values = {m:[100*metrics[b + "_" + m]["map5095"] for b in BANKS] for m in ("uniform","curriculum")}
    std = {m:statistics.stdev(v) for m,v in values.items()}
    mean = statistics.mean(delta.values()); wins = sum(v > 0 for v in delta.values())
    gates = {"Gate_A":mean >= .5,"Gate_B":wins >= 2,"Gate_C":all(v >= -2 for v in means.values()),
             "Gate_D":std["curriculum"] <= std["uniform"]}
    s = {"status":"SYNTHETIC_EXPOSURE_TIMING_PREMISE_SUPPORTED" if all(gates.values()) else "SYNTHETIC_EXPOSURE_TIMING_PREMISE_NOT_SUPPORTED",
         **gates,**{f"delta_{b}_pp":delta[b] for b in BANKS},"mean_delta_pp":mean,
         "uniform_bank_std_pp":std["uniform"],"curriculum_bank_std_pp":std["curriculum"],
         "Flash_mean_delta_pp":means["flash"],"Black_mean_delta_pp":means["black"]}
    headroom = {"comparison":"within-bank curriculum minus uniform, NOT RealRepeat headroom", "delta_pp":delta,
                "mean_delta_pp":mean,"strict_bank_wins":wins,"gates":gates,"scope":"conditional detector internal_dev premise only"}
    classwise = {"paired_deltas_pp":classes,"class_mean_deltas_pp":means,"threshold_pp":-2}
    variance = {"bank_map5095_pp":values,"sample_std_pp":std,"ddof":1,"bank_count":3,"detector_seed_count":1}
    return s,headroom,classwise,variance


def finalize(run):
    run = Path(run); verify_prepared(run)
    barrier = checkpoint_barrier(run)
    require(barrier == load(run / "detector_checkpoint_binding.json"),"STAGE21B_CHECKPOINT_CHANGED")
    pair = inspect_draw_logs(run)
    ev = load(run / "internal_dev_evaluation_audit.json")
    require(ev["status"] == "COMPLETE" and ev["rounds_started"] == ev["rounds_completed"] == 1
            and ev["internal_dev_forwards"] == 162 and ev["state_at_first_dev_content_access"] == "ALL_SIX_FROZEN"
            and ev["checkpoint_binding_sha256"] == sha(run / "detector_checkpoint_binding.json"),"STAGE21B_EVAL_ROUND_INVALID")
    for arm in ARMS:
        require(ev["arms"][arm]["image_forwards"] == 27 and len(set(ev["arms"][arm]["node_ids"])) == 27,"STAGE21B_EVAL_ROUND_INVALID")
    guards = {tag:load(run / (tag + "_file_access_guard.json")) for tag in ("parent","safety",*ARMS,"internal_dev")}
    require(all(g["official_blocked_open_attempts"] == g["internal_dev_blocked_open_attempts"] == 0
                and g["official_final_eval_image_read"] == g["official_final_eval_label_read"] == g["official_final_eval_forward"] == 0 for g in guards.values()),
            "STAGE21B_FILE_ACCESS_GUARD_VIOLATION")
    metrics = {a:load(run / (a + "_metrics.json")) for a in ARMS}
    for a,m in metrics.items(): require(m["checkpoint_sha256"] == barrier["arms"][a]["sha256"],"STAGE21B_METRIC_CHECKPOINT_CHANGED")
    cfg = load(run / "stage21b_a1_protocol_snapshot.json")
    result,headroom,classes,variance = calculate(metrics,cfg)
    for name,value in (("stage21b_headroom",headroom),("stage21b_classwise",classes),("stage21b_variance",variance)):
        save(run / (name + ".json"),value)
    save(run / "actual_compute_equality_audit.json", {"status":"EQUAL_SCHEDULE_PASS" if pair["successful_updates_equal"] else "EQUAL_SCHEDULE_PASS_WITH_AMP_SUCCESSFUL_UPDATE_VARIATION",
        "total_draws_each_arm":25500,"optimizer_attempts_each_arm":450,"successful_updates":pair["successful_updates"],
        "successful_updates_equal":pair["successful_updates_equal"],"not_compensated":True,
        "comparison_definition":cfg["compute_definition"],"actual_augmentation_pairing":True})
    update(run,**result,phase="COMPLETE",full_experiment_complete=True,internal_dev_eval_rounds=1,
        internal_dev_forward=162,successful_updates=pair["successful_updates"],detector_training_count=6,
        optimizer_step_count=sum(pair["successful_updates"].values()),STAGE21B_EXECUTION_AUTHORIZED=False,
        official_final_eval_access_count=0,final_eval_access_count=0,generation_count=0,BootstrapGuard=0,
        strict_equal_successful_updates_claim=pair["successful_updates_equal"],
        innovation1_confirmatory_protocol_design_allowed=all(result[k] for k in ("Gate_A","Gate_B","Gate_C","Gate_D")),
        confirmatory_execution_authorized=False,Stage21C_auto_start=False)
    write_report(run)
    # Metadata only. Heavy inputs/checkpoints/draw traces are separately SHA-bound, not recursively scanned.
    names = sorted(p.name for p in run.iterdir() if p.is_file() and p.suffix in (".json",".md") and p.name != "stage21b_result_manifest.json")
    save(run / "stage21b_result_manifest.json", {"sha256":{n:sha(run / n) for n in names},"full_experiment_complete":True,
        "heavy_asset_binding":"detector_checkpoint_binding.json and per-arm training log SHA256"})
    print(canonical(load(run / "stage21b_status.json")))


def write_report(run):
    run = Path(run); s = load(run / "stage21b_status.json")
    lines = ["# Stage21B Amendment A1 — Timing premise", "",f"状态：`{s['status']}`。完整 GPU 实验完成：{s['full_experiment_complete']}。", "",
        "## 冻结范围", "", "原 e100960 入口审计及 STAGE21B_CONDITIONING_BACKGROUND_DEV_OVERLAP 永久保留，不是 premise FAIL/PASS。",
        "split 原文件字节保持一致：seed2101 / Plastic_Bomo_Stage21B_TrainOnly；107 train groups、27 dev groups；109 real train images、27 dev images。",
        "仅额外排除授权的4个 conditioning-dev 同源 slot，理由 KNOWN_SOURCE_GROUP_LEAKAGE_EXCLUSION；三 bank 同步，flash32 / black29，总61。无补样本、重生成或质量筛选。", "",
        "## 生成器历史暴露限制", "", "冻结生成器曾使用当前 dev group 中33个 annotation inputs（black23 / flash10）。",
        "本实验仅 `DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR`，绝非 `PIPELINE_INDEPENDENT_HOLDOUT`；不通过 generator 重训练修复。", "",
        "## 固定训练及 compute", "", "六 arm 顺序 A_uniform→A_curriculum→B_uniform→B_curriculum→C_uniform→C_curriculum；same frozen YOLO11s init、seed42、imgsz1536、batch1、AdamW及损失/增强参数采用 Stage20B 快照。",
        "150 epochs×170 draws；每 role150次。real16350、synthetic9150、total25500；每 epoch groups=[64,64,42]，tail /42，总450 scheduled attempts。",
        "Uniform每 epoch109 real+61 synthetic；Curriculum前50 epoch170 real+0synthetic，后100 epoch偶数late91synthetic+79real、奇数92synthetic+78real。",
        "role内稳定 cyclic ordering；occurrence augmentation seed仅依赖 namespace/role/index/seed42。实际训练记录 input/target tensor SHA，所有25500 occurrences按 bank配对核验。",
        "reuse Stage20B callback controller/backward；只改变 order/occurrence RNG及必要的170-draw LR进度。保留 native EMA每attempt更新与FP16 epoch150 last.pt格式，无best/earlystop/dev选checkpoint。",
        "compute固定的是 scheduled draws及attempts；实际AMP successful/skipped计数另报，不补 steps，若不相同绝不声称成功更新数相同。", "",
        "## 访问与评估 barrier", "", "Python open audit hook + numpy.fromfile/cv2.imread wrapper +显式 native dataset image allowlist。官方 V2 final_eval 始终禁止；不是 OS 沙箱。",
        "internal_dev image/label首次统一读取，必须在六个last.pt全部SHA冻结且实际augmentation/exposure审计通过后。每checkpoint只评估一次，发生部分失败不自动重试。",
        "single-batch safety仅internal_train real，两个schedule各做forward/backward，optimizer.step=0；模型丢弃，六arm从相同初始化重新开始。", "",
        "## 结果", ""]
    if s["full_experiment_complete"]:
        lines += ["| bank | Uniform mAP50-95 | Curriculum mAP50-95 | delta pp |", "|---|---:|---:|---:|"]
        for bank in BANKS:
            u,c = (load(run / (bank + "_" + mode + "_metrics.json"))["map5095"]*100 for mode in ("uniform","curriculum"))
            lines.append(f"| {bank} | {u:.4f} | {c:.4f} | {s['delta_' + bank + '_pp']:+.4f} |")
        lines += ["",f"mean delta={s['mean_delta_pp']:+.4f} pp；sample SD Uniform={s['uniform_bank_std_pp']:.4f}、Curriculum={s['curriculum_bank_std_pp']:.4f} pp。",
            f"Flash mean delta={s['Flash_mean_delta_pp']:+.4f} pp；Black={s['Black_mean_delta_pp']:+.4f} pp。",
            "", "Gate A≥+0.50pp；B≥2 strict wins；C两类mean≥−2pp；D curriculum sampleSD≤uniform（ddof1）。",
            canonical({k:s[k] for k in ("Gate_A","Gate_B","Gate_C","Gate_D")}),"", "实际 successful updates：" + canonical(s["successful_updates"])]
    else:
        lines += ["未产生正式 metrics/headroom/classwise/variance；Gate_A/B/C/D=NOT_RUN，delta/std=null（失败时保留已产生的局部输出，不作为完整结果）。",
                  "本地无 torch/CUDA/ultralytics；CPU检查不能替代single-batch GPU safety、实际tensor配对或六arm结果。"]
    lines += ["", "## 解释边界与停止条件", "", "即使全部Gate PASS，也仅支持这一固定 late schedule 的内部 timing premise，不证明最终创新、最优schedule、bank不稳定因果解释或V2 final_eval效果。",
        "任一Gate失败，停止等待人工决策，不搜索warmup/seed/比例。PASS仅允许下一步方案设计，不能自动执行。",
        "official final_eval content/forward=0；generation=0；BootstrapGuard=0；DeepPCB=0；没有新selector/generator/TDCRG/weighting/Stage21C。", ""]
    (run / "STAGE21B_REPORT.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--run",type=Path,required=True);a=p.parse_args()
    try: finalize(a.run.resolve())
    except Exception as error:
        print(canonical({"status":"STAGE21B_FINALIZATION_FAILED","detail":str(error)}),file=sys.stderr);return 2
    return 0


if __name__ == "__main__":sys.exit(main())
