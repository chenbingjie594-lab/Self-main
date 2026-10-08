# Stage20B-P0 downloaded server review

Formal P0 status: `STAGE20B_STRUCTURAL_PREFLIGHT_PASS`.
Downloaded metadata review: `STAGE20B_P0_SERVER_REVIEW_PASS`.

Authoritative successful run:
`../server_preflight_20261008_084115/`.
Original local freeze and Stage20A remain unchanged; this review does not replace
their historical status files or alter any frozen selection or protocol.

## Verified evidence

- Stage20A's 24 artifact hashes, local P0's 19 hashes and successful server P0's
  21 hashes match their original manifests. The manifests themselves are also
  bound through the Stage20A config and successful server local-binding audit.
- Executed server script SHA256 matches the checked-in diagnostic revision:
  `f5fa4acbfff3a0713a120a7462a1aa386ce6a38d03f8b9cd538466c3602340e4`.
- Protocol SHA256 remains
  `3e3537cdd241b79e6711525432586d674c7fe31352cea505914987ae15d98c80`.
- 80 slots: Flash40 / Black40; 80 distinct annotation instances, donor images
  and known source groups. Source/normal physical paths alone relocate.
- The complete allowed normal pool is unchanged: 394 TRAIN, 128 excluded.
  Successful server P0 records 666 hash-verified TRAIN assets, and 172 generator
  annotation inputs (Flash75 / Black97), not 172 independent donor images.
- RR uses each slot's full donor image and all original labels: 80 replay
  exposures, 80 distinct donor images, maximum reuse1. These are not new real
  images; they already belong to the identical 136-image V2 TRAIN base.
- RNG seeds A42 / B3407 / C2026 are rederived for every slot. Each bank shares
  exactly the same non-RNG metadata. Actual future trained checkpoint hashes
  remain pending; checkpoint roles are not proof of trained weight identity.
- All 150 epoch role permutations and augmentation seeds are independently
  rederived. The future schedule is 32400 draws/batches and 600 scheduled
  optimizer attempts (64/64/64/24 draws per epoch accumulation groups).
  AMP-successful optimizer updates are not required to be numerically equal.
- Public SD2 weight hashes and configuration Git blobs, original pretrained
  YOLO11s identity, pinned generator code hashes, installed runtime versions
  and resolved detector arguments are internally consistent.

Frozen server versions include torch2.5.1, ultralytics8.4.145,
transformers4.57.6, accelerate1.15.0, safetensors0.8.0, numpy2.2.6,
PyYAML6.0.3 and Pillow12.3.0. The archived runtime audit contains source hashes;
do not upgrade packages during future Stage20B execution.

## Earlier failed run retained

`../server_preflight_20261008_082803/` is preserved byte-for-byte, including its
17-artifact hash ledger. It stopped at 394 missing-or-changed normal-file paths
under the original server `empty_images` root. That generic stop status was
`GENERATOR_TRAIN_SPLIT_CONTAMINATION`; it did **not** establish actual leakage.
The successful run uses `empty_images_v2_p0` and verifies the original bytes,
without changing the 394-file allowlist, slots, thresholds or protocol.
The evidence does not distinguish missing files from changed files at the old
root, so no stronger diagnosis is asserted.

## Tests and reproducible review

All 24 tests passed on local Python3.12.7:

```text
python -m unittest discover -s tests -p test_plastic_bomo_stage20a.py -v
python -m unittest discover -s tests -p test_plastic_bomo_stage20b_p0.py -v
python -m unittest discover -s tests -p test_plastic_bomo_stage20b_p0_review.py -v
```

The review covers missing/changed ledger artifacts, path traversal rejection,
and semantic RNG tampering even when a tampered record is rehashed. Run the
following with a NEW output directory; existing results are never overwritten:

```bash
python tools/review_plastic_bomo_stage20b_p0.py \
  --server results_for_gpt/plastic_bomo_stage20b_p0_preflight/server_preflight_20261008_084115 \
  --local results_for_gpt/plastic_bomo_stage20b_p0_preflight \
  --stage20a results_for_gpt/plastic_bomo_stage20a_v2_protocol \
  --protocol configs/plastic_bomo_stage20b_p0.json \
  --output results_for_gpt/plastic_bomo_stage20b_p0_preflight/new_review_directory
```

## Limits and next boundary

This is structural authorization, **not** a GPU forward, reproducibility,
gradient, numerical-validity, generation-quality or downstream-utility result.
The local review reads downloaded metadata only, not remote assets. Model
weights were not loaded. All training/generation/final_eval counts remain0;
TDCRG, DeepPCB and BootstrapGuard remain untouched and unauthorized here.

Only no **KNOWN** source-group leakage is supported: physical_source_complete
remains false. Labels are dataset-defined supervision, not universally
human-verified physical ground truth. Rectangular bbox masks provide weak
geometry, not segmentation ground truth. V2 absolute mAP cannot be ranked
against Stage14-18 absolute mAP from different splits/protocols.

Future formal execution still requires explicit manifest-only generator and
custom detector implementation, runtime safety/forward checks, frozen shared
class-checkpoint bytes before sampling, and all four epoch150 last.pt files
before unified final_eval. No formal run or new module is started by this review.
