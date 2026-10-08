# Plastic_Bomo Generation V2 — frozen Stage20A

This is a new dataset protocol, **not a Stage17/18 rescue**. Historical states,
labels, images, splits and manifests remain unchanged. Local bootstrap completed;
no model was loaded, trained or used for sampling.

## Frozen artifacts

Authoritative directory: `results_for_gpt/plastic_bomo_stage20a_v2_protocol/`.
The byte-exact artifact ledger is `frozen_artifact_manifest.json`:

`SHA256 = 2caa71e8cb3af1b6cae70b7d70974d7ccf6d70d3ea3eee37b4e22bbad1c2a079`.

Its historical evidence snapshots are included in `evidence/`; reproducing the
audit does not require restoring deleted historical working-tree files. Original
image/label assets remain external and are bound by SHA256, not uploaded here.

| Split | Detector images | Labeled source groups | Flash boxes | Black boxes |
|---|---:|---:|---:|---:|
| TRAIN | 136 | 134 | 75 | 97 |
| final_eval | 48 | 47 | 33 | 27 |

There are 392 groups across all registered source/normal/derived assets and 394
resolved TRAIN normals. The two labeled group counts exclude normal-only groups.
Known group/RGB/sibling/parent cross-split conflicts are zero. Unknown physical
relations are not proven absent: `physical_source_complete=false`.

All 168 former training boxes are ledgered (118 XML-derived, 50 supplemental
prelabels). Human verification is not inferred. The 64 former evaluation boxes
are separately registered administratively; unresolved provenance remains OTHER
or UNCONFIRMED. No annotations were corrected, added or removed. The two roughly
one-pixel Black clipping issues are retained in the ledger.

## Verification, not re-splitting

From the original local workspace:

```powershell
python tools/review_plastic_bomo_stage20a.py --output results_for_gpt/plastic_bomo_stage20a_v2_protocol --verify_only
python -m unittest discover -s tests -p test_plastic_bomo_stage20a.py -v
```

On another machine, the review supports repeated `--path_map old_prefix=new_prefix`
arguments. Prefix mapping only relocates files; all bytes, IDs and splits must
remain identical. Missing assets or changed bytes stop verification. Do not rerun
the bootstrap on a different inventory and call it V2.0. Any inventory/group/split
revision needs an explicit new protocol version, not an overwritten directory.

## Authorization boundary

`PLASTIC_BOMO_GENERATION_V2_PROTOCOL_READY` authorizes planning/execution of a
separately frozen Stage20B raw-baseline calibration, **not immediate training**.
Before that execution: bind server assets and original pretrained weights, freeze
generator hyperparameters and all actual slots, complete geometry/RNG structural
preflight, and pin all shared detector hyperparameters. Do not use final_eval to
make these decisions. 40 Flash + 40 Black remains unchanged, with no selector.

Only the SD2 inpainting raw V2 family is selected, based on pinned code and input
contract feasibility, not prior performance. No old fine-tuned weights are reused.
YOLO11s seed42 / 150 epochs / batch1 / imgsz1536 / epoch150 last.pt is specified.
Banks A/B/C vary only generation RNG. All baseline gates must pass before TDCRG
development is authorized. TDCRG is specification only; no fallback generator,
Stage20B, DeepPCB or BootstrapGuard experiment has been started.
