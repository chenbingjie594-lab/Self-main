# Stage17A train-only training: downloaded records review

Reviewed server run `trainonly_20261006_022714`. This is an audit of downloaded
records, not a local loading test of the model weights.

- Preparation and isolation audit file hashes match their recorded frozen hashes.
- Protocol, annotation-manifest and training-entrypoint hashes match local files.
- All 168 source identities and image/mask hashes match local reproduction.
- Source parents: unchanged 138 real train images; 88 flash / 80 black instances.
- Server isolation audit covers 46 official validation parent images and passes
  exact parent filename, byte and decoded-RGB isolation. No validation labels or
  detection metrics were used. Unknown acquisition-group independence is unproven.
- Initialization file-hash list equals the original pretrained SD2 list in the
  first rebuild audit, not the contaminated fine-tuned checkpoint list.
- Both classes: 2000 attempts, 2000 successful updates, 0 AMP skipped updates,
  seed42, lr1e-6, batch1/accum4, fp16, text noise0, UNet training only.
- Both report FINAL_PIPELINE_SAVED. Embedded runtime records in the checkpoint
  audits equal their standalone runtime JSON. Each class lists 13 saved assets,
  including UNet, VAE and text-encoder safetensors.
- VAE/text/tokenizer serialized hashes agree across both classes and with the
  prior rebuild serialization. They do not need to equal original base file-byte
  hashes, because saving/precision conversion can change serialization; this
  comparison is not a parameter-level proof.
- UNet weight hashes differ between classes, as expected for separate training.

The actual remote checkpoint bytes and tensor payloads have NOT been independently
checked from this download. `probe_plastic_bomo_stage17a_trainonly.py` verifies
those before runtime probes. Thirty smoke samples are planned, not formal slots.
The old mixed-source checkpoint is never used by that probe. Formal generation,
detector training, normal-parent lineage and slot/bbox audits remain blocked.
No result, model, frozen annotation or historical status was overwritten.
