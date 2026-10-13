# Stage25-P1 — Server runtime and formal runner preparation

Status: `SERVER_EXECUTION_REQUIRED`.

This is code/protocol preparation, not a GPU qualification or performance result. The original Stage25 local P0 remains unchanged: its stop applied only to the Windows execution environment. It did not establish MSDF algorithm/performance failure.

## Implemented entry points

Shared frozen recipe: `configs/plastic_bomo_stage25_server.json`. It preserves the parent P0 training targets, method defaults, generation targets, detector targets and gates byte-for-byte at the parsed-field level. `plastic_bomo_stage25_server_code_manifest.json` binds executable/source bytes. Every child rechecks these identities.

`tools/run_plastic_bomo_stage25_msdf_v2.py` supports `runtime_check`, `train`, `generate`, `detector`, `evaluate`, and `all`. GPU phases run in separate processes. One-shot claims prohibit retry/resume/overwrite in a failed directory. Outputs are new `server_runtime_<timestamp>` and `formal_<timestamp>` children, not modifications to the local P0 root.

The three new implementation helpers cover immutable binding/path relocation, MSDF training/sampling, and frozen downstream/evaluation integration. They are execution shells, not new neural modules.

## Frozen mechanisms and declared shell differences

MSDFAdapter, both encoders, morphology alignment, dynamic gates, up-block injection and bounded residual are unchanged. Historical helper function definitions and the literal six-loss body are selected from SHA-bound `train_rda.py` by AST, then compiled unchanged. The generic training/inference mains and their optional teachers/selectors are not imported/executed.

Full UNet and MSDF use FP32 master parameters; VAE/text encoder are frozen FP16 runtime copies of exact public components. UNet LR remains 1e-6; MSDF LR remains 1e-4. Each class stops at 2000 actual optimizer-post-hook successes, recording attempts and AMP skips. The generator's historical divisor4, including partial epoch groups, is retained. No 8-bit optimizer, lower resolution or branch disabling is available.

Declared D7 adaptation: the conditional prompt uses the frozen Raw tokenizer padding=False contract; the unconditional embedding is padded to the same width to retain the doubled CFG-aware forward. This is explicitly recorded, not claimed numerically identical to the historical generic padded-token wrapper. Historical TRAIN mask dilation24, prior blur2, cross-reference0.30 and all auxiliary coefficients remain unchanged.

Declared D8 adaptation: sampling calls the exact Stage20B Standard inpainting `sample()` crop/RNG/paste implementation, with existing MSDF prepare/up hooks added. The frozen slot's TRAIN donor supplies MSDF reference pixels/VAE-mode features; the same frozen TRAIN normal supplies the inpainting canvas. No dynamic/magic pipeline, residual recomposition, scoring, normal filtering, reroll or replacement is used. All80 planned outputs per bank are retained or generation stops.

## Runtime qualification

Linux Python3.10, exact Stage20B package versions and source hashes, Torch2.5.1+cu124/CUDA12.4, native cuDNN and RTX3090 capability are checked before models load. tqdm must already be installed; its actual version is recorded and frozen. No installation or upgrade is attempted.

Original Stage20A/P0/R0 ledgers remain authoritative. A separately hashed `p0_view` changes paths only when actual data directories differ. All666 TRAIN image/label/normal assets, all394 normals, original80 slots, public SD2 components and detector initialization are hash-checked. Runtime/model code guards exact metadata-known final_eval paths, not the whole old physical val folder (some old val files are frozen V2 TRAIN). Physical source completeness or absence of unknown aliases is not claimed.

Fresh Flash and Black models each execute one TRAIN forward/backward with optimizer.step=0 and no sampling. Qualification checks full trainable gradient connectivity, finite loss/gradients, unchanged frozen VAE/text parameters and tensor shapes. Zero-but-connected gradients at zero-init projections are not confused with disconnected parameters. Allocated/reserved/peak VRAM is recorded. OOM stops as `STAGE25_SERVER_RUNTIME_OOM`, with no rescue.

## Comparators and downstream

The server reassesses historical frozen RawA/B/C and RR, including metadata, generator trees, all raw images/labels, detector checkpoints/budgets, runtime/source identity and evaluation protocol. Local Windows ultralytics8.4.120 is not used to deny Linux8.4.145 reuse. Fully qualified historical comparators are preferred. Missing or incompatible comparator assets produce `RAW_COMPARATOR_REUSE_NOT_QUALIFIED` and a separate same-runtime comparator rebuild, not a scientific MSDF failure.

MSDF detector training reuses the unchanged Stage20B train/controller implementation with a Stage25 binding shell. Each arm has136 real +80 synthetic roles,216 draws/epoch,150 epochs,32400 draws and600 scheduled attempts. Accumulation groups64/64/64/24, per-draw RNG, native EMA-on-every-attempt behavior and epoch150 EMA last.pt representation are retained exactly. AMP successes/skips are reported without compensating steps. No YOLO.train wrapper, best.pt, early stopping or training validation is used.

All three MSDF last.pt checkpoints, plus all four comparators if rebuilt, must be SHA-frozen before any new final_eval content access. Historical comparator metrics are imported only after reuse qualification; otherwise all seven arms are evaluated in one unified new round. Final_eval was already observed in Stage20B and is not an independent untouched test.

Fixed gates remain mean matched MSDF−Raw >=0.50pp, at least2/3 strict bank wins, and each class mean >=−2pp. Utility requires mean MSDF−RR >=0 and at least2/3 strict wins. No Stage25 result-dependent changes are allowed. Completion writes the original Stage25 audit filenames plus additional binding/compute/access records and stops; no next stage starts.

## Verification performed here

32 stdlib-only P1 tests passed, including literal source/loss extraction, original recipe equality, uint64 slot seeds, barriers, fixed gates, one-shot claims, original metadata path-relocation schema and comparator metadata integration. The integration fixtures deliberately mock large image/weight hashes; they are not scientific asset or GPU validations.

13 original P0 read-only tests passed. Every entry in its existing artifact ledger is unchanged. `bash -n` passed for the launcher. CLI --help and source compilation passed without importing Torch, diffusers or ultralytics.

Not performed: SSH, Windows package installation, GPU forward/backward, optimizer steps, sampling, generator/detector training, or final_eval image/label content access. Actual GPU/API/VRAM correctness remains a server runtime gate; static tests do not replace it.

## Server operation

From the actual Linux repository directory, run:

```bash
bash tools/launch_plastic_bomo_stage25_server.sh all
```

The launcher derives the repository from its own location, verifies entry files and the existing Python executable, and prints PID/log paths. Default visible physical GPU is3; logical device is0. Set STAGE25_PYTHON/STAGE25_GPU explicitly if the existing Stage20B locations differ. Explicit --real_root/--normal_root overrides are path-only and still hash-checked. No environment is changed.

To qualify only, replace all with runtime_check. Detailed/manual phase commands are in `Stage25服务器命令.txt`.

Upload the changed files preserving relative paths, or use the Git commit. Existing frozen historical metadata, source files and assets must still be present. If an old source byte hash differs, transfer the frozen source bytes; do not edit hashes or normalize historical code to bypass the check. No archive was created.

After execution, return the new runtime and formal JSON/MD/logs. Preserve all original generated images and checkpoints on the server for SHA-based review. This preparation ends at `SERVER_EXECUTION_REQUIRED`.
