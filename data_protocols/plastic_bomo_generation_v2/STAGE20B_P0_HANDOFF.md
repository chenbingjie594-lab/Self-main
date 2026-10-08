# Stage20B-P0 handoff — no formal run

Stage20A files are unchanged. New P0 files:

- `configs/plastic_bomo_stage20b_p0.json`
- `tools/preflight_plastic_bomo_stage20b_p0.py`
- `tests/test_plastic_bomo_stage20b_p0.py`
- `Stage20B-P0命令.txt`
- `results_for_gpt/plastic_bomo_stage20b_p0_preflight/`

The script uses only CPU metadata, file hashes and in-memory rectangular masks.
No PyTorch/diffusion/YOLO model import, pickle load, model forward, sampling,
image export, training or final_eval pixel/label read occurs. Structural unit
tests do not establish GPU determinism or end-to-end numerical validity.

## Local versus server

Local structural pass is not server authorization. The current local status is
`SERVER_ASSET_BINDING_PENDING`, execution authorization **false**. The server
command binds actual TRAIN/normal/weight bytes, generator code and installed
runtime source/version metadata. Only successful `SERVER_PREFLIGHT` can produce
`STAGE20B_STRUCTURAL_PREFLIGHT_PASS` and execution authorization true. Even then,
this P0 command never launches anything else.

Upload code/config and the full frozen Stage20A AND local Stage20B-P0 result
folders without newline conversion. Server P0 verifies local P0 artifact hashes,
the exact config bytes and all 80 semantic slots; only physical root paths may
relocate. Use the server command file. The old image/label layout is used only
as physical storage: membership is determined by V2 IDs/splits, not whether a
file happens to live under an old `images/train` or `images/val` directory. Some
V2 TRAIN images reside in former eval folders; they are TRAIN under the new V2
protocol. No V2 final_eval file is accessed at P0.

The normal directory must contain the original byte-identical 394 allowed
normal files. Do not substitute a filtered/resized pool or an old 174-normal
pool. If a root differs, change only the root path argument. SHA mismatches stop
the audit; they do not authorize reconstruction, slot changes or retraining.

The exact public SD2 FP32 safetensors files are pinned to community-mirror
revision `5f74973cbb64c8568780732c17f43eb269d63a0d`. Their public LFS SHA256
references and small Git blob IDs were retrieved from the Hugging Face model
metadata API, not inferred from folder names. No weights were downloaded.
The original pretrained YOLO11s file is bound to its existing official
initialization hash, not a trained Stage17 detector.

## Frozen budget and enforcement requirements

All 172 TRAIN annotations pass the fixed 512-crop geometry contract. Slots use
40 distinct Flash / 40 distinct Black annotation instances, images and groups.
Normals are stable-hash assigned from the entire unchanged TRAIN pool. These
choices do not use visual/quality/performance metrics.

Raw synthetic outputs are full normal canvases with one generated patch pasted
at a fixed position and the mapped target label. RR is the corresponding full
real image plus all its original labels, not a bbox crop. Each arm has 136 base
draws + 80 extra role draws per epoch. Extra-role class balance is 40/40; RR full
images can contain other labels, so role counts are NOT claimed to equal the
total RR box count.

Detector AdamW hyperparameters and augmentations are explicitly predeclared.
Composite augmentation is disabled to preserve single-role exposure. Fixed
accumulation64, with the 24-draw tail flushed every epoch, schedules 600 optimizer
attempts over 32400 draws/batches. AMP successes are separate. This is a V2
predeclared budget, not a post-hoc adjustment or reuse of old compute results.
The future trainer must implement exact role consumption, per-draw RNG,
accumulation normalization, no early stop and no automatic final_eval call.
There is no training entry point in P0; these runtime safeguards must not be
assumed to be provided by an unmodified Ultralytics default trainer.

Future class checkpoints are specified as new final_step2000 V2 pipelines;
their actual hashes cannot exist before training. Bind one frozen checkpoint
per class, identical across all banks, before any future sampling. P0 pairing
audits checkpoint roles, not nonexistent trained checkpoint bytes.

## Next action

Run only the P0 server command, then download its entire `server_preflight_*`
folder, including failed audits if any. Do not start generator/detector training,
sampling, final_eval, TDCRG, BootstrapGuard or DeepPCB. V2 absolute mAP must never
be ranked against historical Stage14-18 absolute mAP.
