"""One-shot six-arm launcher; no native train/eval until immutable metadata and safety pass."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import traceback

from stage21b_a1_common import *
from prepare_plastic_bomo_stage21b import verify_assets


def run(a):
    prepared = a.prepared.resolve(); manifest = verify_prepared(prepared)
    cfg = load(prepared / "stage21b_a1_protocol_snapshot.json");check_cfg(cfg)
    evidence = load(prepared / "source_evidence_index_A1.json")
    for name,h in evidence["code_sha256"].items(): require(sha(REPO / name) == h,"STAGE21B_EXECUTABLE_CHANGED: " + name)
    require(not a.output.exists(),"STAGE21B_OUTPUT_EXISTS_USE_NEW_RUN; no overwrite or automatic resume")
    a.output.mkdir(parents=True); run = a.output.resolve(); (run / "logs").mkdir()
    try:
        for name,h in manifest["sha256"].items(): verified_copy(prepared / name,run / name,h)
        verified_copy(prepared / "amended_metadata_manifest.json",run / "amended_metadata_manifest.json",sha(prepared / "amended_metadata_manifest.json"))
        save(run / "stage21b_status.json",initial_status())
        # All source checks here are metadata or internal_train/synthetic only.
        md,rmd,versions = check_runtime(a.p0,a.r0,a.detector_model)
        bound_metadata(a.stage20a,"frozen_artifact_manifest.json",STAGE20A_LEDGER,
                       ("real_source_registry.json","annotation_provenance_v2.json","normal_source_registry.json","source_group_registry.json"),[])
        bound_metadata(a.stage20b,"formal_result_artifact_manifest.json",STAGE20B_LEDGER,
                       tuple(f"formal_bank_{b}_manifest.json" for b in BANKS),[])
        u,c = [load(run / (m + "_schedule_v2.json")) for m in ("uniform","curriculum")]
        base = load(run / "stage21b_base_protocol_snapshot.json");validate_schedules(u,c,base,cfg)
        binding = {"stage20a":str(a.stage20a.resolve()),"stage20b":str(a.stage20b.resolve()),"p0":str(a.p0.resolve()),
            "r0":str(a.r0.resolve()),"real_root":str(a.real_root.resolve()),"detector_model":str(a.detector_model.resolve()),
            "detector_sha256":md["detector_initialization_binding.json"]["sha256"],"device":"cuda:" + a.device,
            "code_sha256":evidence["code_sha256"],"package_versions":versions,
            "prepared_metadata_manifest_sha256":sha(run / "amended_metadata_manifest.json")}
        save(run / "execution_binding.json",binding)
        # Parent file guard protects all content checks before child training.
        with guard_for(run,binding).python_guard() as guard:
            reals = [r for r in load(run / "stage21b_split_manifest.json")["records"] if r["internal_split"] == "internal_train"]
            audit = verify_assets(load(run / "stage21b_synthetic_allowlist_v2.json"),a.stage20b,reals,a.real_root)
            audit["internal_dev_content_read"] = 0;save(run / "server_source_asset_hash_audit.json",audit)
        save(run / "parent_file_access_guard.json",guard.counts)
        env = dict(os.environ,STAGE21B_BINDING_SHA256=sha(run / "execution_binding.json"))
        update(run,status="STAGE21B_A1_RUNNING",phase="SAFETY",STAGE21B_EXECUTION_AUTHORIZED=True)
        def child(phase,arm=None):
            tag = arm or phase
            logfile = run / "logs" / (tag + ".log")
            cmd = [sys.executable,str(REPO / "tools/execute_plastic_bomo_stage21b_a1.py"),"--run",str(run),"--phase",phase]
            if arm:cmd += ["--arm",arm]
            print("Starting " + tag + " → " + str(logfile),flush=True)
            with logfile.open("x",encoding="utf-8") as log:
                result = subprocess.run(cmd,cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            require(result.returncode == 0,"STAGE21B_CHILD_FAILED: " + canonical({"phase":phase,"arm":arm,"exit_code":result.returncode,"log":str(logfile)}))
        child("safety")
        for i,arm in enumerate(ARMS):
            update(run,phase=arm,detector_training_count=i+1)
            child("train",arm)
            tr = load(run / (arm + "_training.json"))
            s = load(run / "stage21b_status.json"); successful = s["successful_updates"];successful[arm]=tr["successful_optimizer_updates"]
            update(run,completed_arms=list(ARMS[:i+1]),successful_updates=successful,
                   optimizer_step_count=sum(x or 0 for x in successful.values()))
        from finalize_plastic_bomo_stage21b_a1 import inspect_draw_logs,finalize
        pairing = inspect_draw_logs(run)  # All occurrence hashes must match before first dev content access.
        save(run / "detector_checkpoint_binding.json",checkpoint_barrier(run))
        update(run,phase="ALL_SIX_FROZEN")
        child("evaluate")
        finalize(run)
    except Exception as error:
        if (run / "stage21b_status.json").exists():
            previous = load(run / "stage21b_status.json")
            update(run,status="STAGE21B_A1_STOPPED",reason=str(error),
                child_detail=previous.get("reason"),STAGE21B_EXECUTION_AUTHORIZED=False,full_experiment_complete=False)
            from finalize_plastic_bomo_stage21b_a1 import write_report
            write_report(run)
        print(traceback.format_exc(),flush=True); return 2
    return 0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stage20a",type=Path,default=Path("results_for_gpt/plastic_bomo_stage20a_v2_protocol"))
    p.add_argument("--stage20b",type=Path,default=Path("results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543"))
    p.add_argument("--p0",type=Path,default=Path("results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115"))
    p.add_argument("--r0",type=Path,default=Path("results_for_gpt/plastic_bomo_stage20b_r0_runtime/server_runtime_20261008_121254"))
    p.add_argument("--real_root",type=Path,required=True);p.add_argument("--detector_model",type=Path,default=Path("pretrained/yolo11s.pt"))
    p.add_argument("--device",default="0")
    a=p.parse_args()
    try:return run(a)
    except Exception as error:
        print(canonical({"status":"STAGE21B_A1_LAUNCHER_REJECTED","detail":str(error)}),file=sys.stderr);return 2


if __name__ == "__main__":sys.exit(main())
