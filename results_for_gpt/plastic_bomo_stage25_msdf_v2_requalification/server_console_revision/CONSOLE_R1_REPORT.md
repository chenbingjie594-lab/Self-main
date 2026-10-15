# Stage25 Console R1 — execution display and packaging correction

Status: `SERVER_EXECUTION_REQUIRED`. No model execution occurred here.

The reported server exits were missing local-test/command-document files, before runtime/model loading. These were packaging prerequisites, not MSDF performance failures.

This revision changes only the execution shell and diagnostic presentation. The Stage25 frozen scientific protocol JSON, MSDF core, literal historical losses, generator/downstream implementations, data/normal/slot identities, seeds, optimizer budgets, checkpoint barriers and scientific gates are unchanged.

The runtime code manifest keeps all actual execution/source/protocol bindings mandatory. Four nonexecuted items move to `audit_only_sha256`: local unit tests, the Chinese command document, the optional shell launcher, and the historical generic `run_train_msdf.py` wrapper. Their hashes are preserved for provenance; they are not used by the formal runner. No executed algorithm/data check is bypassed. This revision creates a new code-manifest binding and requires a fresh runtime qualification, not reuse of an older qualified executable envelope.

The direct Python command now streams each child's stdout/stderr to both the terminal and its phase log. Child PID, phase start/end and exit codes are displayed. A quiet running child emits a15-second heartbeat. Missing required files are listed together, in readable Chinese; failure prints `[已退出]` and exits nonzero. Ctrl+C stops only the owned child; claims and partial results are retained, with no automatic retries.

The optional launcher is also foreground and tees output, with an exit trap. The direct command does not require uploading or invoking that launcher. Foreground operation requires keeping the terminal connection open.

Upload/overwrite these three server files together:

- `tools/run_plastic_bomo_stage25_msdf_v2.py`
- `tools/stage25_msdf_contract.py`
- `configs/plastic_bomo_stage25_server_code_manifest.json`

Additional local changes are `tests/test_plastic_bomo_stage25_p1.py`, `tools/launch_plastic_bomo_stage25_server.sh`, and `Stage25服务器命令.txt`. They are not necessary for direct Python execution.

From the actual server project directory:

```bash
env CUDA_VISIBLE_DEVICES=3 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/mnt/sda/cbj/miniconda3/envs/yolo11/bin/python -u \
tools/run_plastic_bomo_stage25_msdf_v2.py --mode all --device 0
```

Validation:42 stdlib/contract/console tests and13 original P0 read-only tests passed; both shell command files passed `bash -n`. Tests cover absent optional files, unchanged mandatory algorithm checks, aggregated missing-file messages, Chinese output, actual live child output, stderr/exit7 preservation, quiet-child heartbeat and owned-child cancellation. Pure Python/fixture subprocesses are not model forwards. Original local P0 ledger entries remain unchanged.

No SSH, package changes, Windows training, model forward, optimizer step, synthetic generation, detector training or final_eval content access was performed. No Git commit/push was performed for this code-only correction. No archive was created.
