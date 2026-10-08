"""Verify V2 metadata against unchanged real assets; no model imports or runs."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8-sig"))


def save(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8", newline="\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--repo_root", type=Path, default=Path("."))
    p.add_argument("--path_map", action="append", default=[], help="old_prefix=new_prefix; paths only, never change IDs/splits")
    p.add_argument("--verify_only", action="store_true")
    a = p.parse_args(); root=a.output; repo=a.repo_root.resolve()
    mappings=sorted([v.split("=",1) for v in a.path_map],key=lambda x:len(x[0]),reverse=True)

    def resolved(raw):
        s=str(raw).replace("\\", "/")
        for old,new in mappings:
            old=old.replace("\\", "/").rstrip("/")
            if s==old or s.startswith(old+"/"):
                s=new.rstrip("/")+s[len(old):]; break
        q=Path(s)
        return q if q.is_absolute() else repo/q

    status=load(root/"stage20a_status.json")
    registry=load(root/"real_source_registry.json")
    rows=registry["records"]; by={r["node_id"]:r for r in rows}
    if len(by)!=len(rows): raise RuntimeError("V2_DUPLICATE_ASSET_NODE")
    requests={}
    for r in rows:
        requests[r["current_path"]]=r["file_sha256"]
        m=r.get("source_xml_metadata")
        if m:requests[m["path"]]=m["sha256"]
    ledger=load(root/"annotation_provenance_v2.json")["records"]
    for x in ledger:requests[x["annotation_file"]]=x["label_sha256"]
    for x in registry["derived_representations"]:requests[x["path"]]=x["sha256"]
    def check(item):
        raw,expected=item; f=resolved(raw)
        actual=sha(f) if f.is_file() else None
        return None if actual==expected else {"path":raw,"resolved":str(f),"expected":expected,"actual":actual}
    with ThreadPoolExecutor(max_workers=4) as pool: errors=[x for x in pool.map(check,requests.items()) if x]
    for e in load(root/"source_group_graph.json")["edges"]:
        if by[e["from"]]["split_v2"]!=by[e["to"]]["split_v2"]:errors.append({"cross_split_edge":e})
    folds=load(root/"v2_oof_fold_manifest.json")
    for x in folds["donor_records"]:
        if by[x["node_id"]]["split_v2"]!="train" or folds["folds"][x["source_group_id"]]!=x["held_out_fold"]:
            errors.append({"invalid_oof_donor":x["node_id"]})
    old=[x for x in ledger if x["split_membership_old"]=="train"]
    if Counter(x["source_family"] for x in old)!={"XML":118,"SUPPLEMENTAL_PRELABEL":50}:
        errors.append({"old_provenance_counts_changed":True})
    if any(x["human_verification_status"]=="CONFIRMED" for x in ledger):
        errors.append({"unsubstantiated_human_verification":True})
    normal=load(root/"normal_source_registry.json")["records"]
    if any(x["confirmatory_allowed"] and (x["split"]!="train" or x["status"]!="NORMAL_SOURCE_RESOLVED") for x in normal):
        errors.append({"invalid_normal_allowlist":True})
    if errors:
        print(json.dumps({"status":"V2_ASSET_OR_METADATA_VERIFICATION_FAILED","errors":errors},indent=2))
        raise RuntimeError("V2_ASSET_OR_METADATA_VERIFICATION_FAILED_NO_TRAINING")
    if a.verify_only:
        frozen=load(root/"frozen_artifact_manifest.json")
        mismatch=[f for f,h in frozen["sha256"].items() if not (root/f).is_file() or sha(root/f)!=h]
        if mismatch:raise RuntimeError("V2_FROZEN_METADATA_CHANGED:"+str(mismatch))
        print(json.dumps({"status":"V2_FROZEN_ASSETS_VERIFIED","files_checked":len(requests),"model_runs":0,"split_recomputed":False}))
        return
    if (root/"frozen_artifact_manifest.json").exists():raise RuntimeError("V2_ALREADY_FROZEN_USE_VERIFY_ONLY")
    evidence_dir=root/"evidence";evidence_dir.mkdir()
    protocol=load(root/"v2_protocol.json"); archived={}
    for name,ev in protocol["evidence"].items():
        if ev.get("origin")=="WORKTREE":raw=resolved(ev["path"]).read_bytes()
        else:
            raw=subprocess.check_output(["git","-C",str(repo),"show","23731281570f8bedadad50f058ec4a718fb83524:"+ev["path"]])
        if hashlib.sha256(raw).hexdigest()!=ev["sha256"]:raise RuntimeError("HISTORICAL_EVIDENCE_CHANGED:"+name)
        dest=evidence_dir/(name+".json");dest.write_bytes(raw)
        archived[name]={"archive":dest.relative_to(root).as_posix(),"sha256":sha(dest),"original":ev}
    save(root/"historical_evidence_archive.json",archived)
    test=subprocess.run(["python","-m","unittest","discover","-s","tests","-p","test_plastic_bomo_stage20a.py","-v"],cwd=repo,capture_output=True,text=True)
    if test.returncode:raise RuntimeError("V2_TESTS_FAILED:"+test.stderr)
    save(root/"verification_audit.json",{"status":"PASS","asset_file_hashes_rechecked":len(requests),"changed_asset_files":0,
        "test_returncode":test.returncode,"test_output":test.stdout+test.stderr,"review_script_sha256":sha(Path(__file__)),
        "historical_evidence_archived_byte_exact":True,"split_recomputed":False,"old_status_files_modified":False,
        "model_import_or_forward_count":0,"TDCRG_IMPLEMENTED":False,"Stage20B_STARTED":False})
    files=sorted(x for x in root.rglob("*") if x.is_file() and x.name!="frozen_artifact_manifest.json")
    save(root/"frozen_artifact_manifest.json",{"protocol":"Plastic_Bomo_Generation_V2","version":"V2.0",
        "sha256":{x.relative_to(root).as_posix():sha(x) for x in files},
        "future_asset_binding":"Map paths and verify identical bytes; NEVER rerun hashing/splitting with a different inventory under V2.0",
        "revisions_require_new_version":True})
    print(json.dumps({"status":"V2_REVIEW_AND_FREEZE_PASS","files_checked":len(requests),"frozen_artifacts":len(files),
        "manifest_sha256":sha(root/"frozen_artifact_manifest.json"),"stage20a_status":status["status"]},indent=2))


if __name__=="__main__":main()
