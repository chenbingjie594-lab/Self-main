# Stage20B formal execution handoff

Formal execution is complete on the server; downloaded results are being archived.
The authoritative run is `manual_C_20261010_032543`: baseline headroom was not confirmed.
See its immutable report and the separate post-execution local review in
`results_for_gpt/plastic_bomo_stage20b_formal/`. No new experiment is authorized.
No additional preflight stage, package changes, selector, TDCRG, BootstrapGuard or DeepPCB work.

Upload these seven new files to the same repository's `tools/` on the server:

- `stage20b_formal_common.py`
- `train_plastic_bomo_stage20b_generator.py`
- `generate_plastic_bomo_stage20b_banks.py`
- `train_plastic_bomo_stage20b_detector.py`
- `evaluate_plastic_bomo_stage20b.py`
- `audit_plastic_bomo_stage20b.py`
- `run_plastic_bomo_stage20b_formal.py`

Copy/run the launch block in `Stage20B正式命令.txt`. No archive or git connectivity is required.
The existing Stage20A/P0/R0 metadata, R0 helpers and public weights must remain present and byte-identical.
Run only the orchestrator; standalone children require its immutable binding envelope and state.

The one-shot run performs Flash2000 successful updates, Black2000, full component freezes,
trained-checkpoint repeated-seed sanity (four non-training images), all80 frozen slots in banks A/B/C,
structural validation of all240 outputs, matched RR/A/B/C detection training, and one unified V2 final evaluation.
Failure ends the run without resume, rescue, reroll, changed budgets or automatic next-stage work.

Detector primary `last.pt` retains the pinned native Ultralytics EMA representation. It is saved
only at epoch150, without invoking native full `train()`, `save_model()`, best-checkpoint selection,
EMA resync, nonfinite-value repair or training-time validation. EMA updates use the pinned native
attempt schedule. Actual successful optimizer counts come from optimizer post-step hooks.
The detector tail24 divides loss by24; the separately frozen generator's legacy accumulation
divisor4 also applies to its epoch-end partial groups. These are distinct frozen rules.

The export copies original public VAE/text weights and all public tokenizer/config bytes;
it does not serialize fp16-rounded frozen-component copies. Only the trained UNet weights change.
The checkpoint and image manifests bind the large assets retained on the server.

## Local verification

Executed on2026-10-09: Python syntax/CLI checks and70 Stage20B CPU/unit/static tests passed,
including the existing P0/R0 tests, immutable ledgers, exact code hashes, state sequence,
first-error stopping, no premature final_eval, frozen role/geometry mapping, class-index metrics,
sample SD, strict wins and unchanged thresholds. This machine has no PyTorch/CUDA runtime;
these tests do not claim GPU training/inference success or runtime performance.

Test command:

```text
python -m unittest discover -s tests -p 'test_plastic_bomo_stage20b*.py' -v
```

Download the completed run's JSON files, `STAGE20B_REPORT.md` and `logs/` for independent
review and the final GitHub result submission. Keep weights, synthetic images and copied
train/final_eval inputs on the server. Do not start TDCRG/Stage20C after either outcome.

## Explicit manual C recovery

Two earlier executions stopped during Flash training with GPU OOM. The next run,
`formal_20261009_060552`, completed both generators, all three raw banks, RR/A/B,
then stopped when the C child returned exit code -9. The user reported a possible
accidental kill; its external cause is not independently verified.

On explicit user request, `recover_plastic_bomo_stage20b_arm_c.py` created a new
sibling run without changing the stopped source. It verified and reused the
completed frozen assets, then trained only C from its original public initialization,
seed42 and epoch1 to150. No epoch77 resume, new generator training, reroll or slot
selection occurred. The discarded C attempt remains separately disclosed: observed
76epochs/16416draws/304attempts/296successes/8AMP skips, potentially lower-bound counts.

This is not represented as an uninterrupted one-shot run. Its primary four-arm
budgets are matched; the extra physical cost of the interrupted attempt is not hidden.
The frozen seven formal scripts and three R0 scripts were not changed by recovery.

The separate `review_plastic_bomo_stage20b_formal.py` reads downloaded ledgers,
available weights/images and per-draw logs, independently recomputes arithmetic/gates,
and writes outside the immutable execution directory. It performs no GPU forward,
training, sampling, re-selection or authority/status modification.
