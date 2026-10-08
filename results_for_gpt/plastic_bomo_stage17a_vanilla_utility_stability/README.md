# Stage17A: Vanilla Synthetic Utility Stability Calibration

Final status: `SYNTHETIC_HEADROOM_NOT_REPRODUCIBLE`.

The authoritative final detector results are in
[`detector_repair_20261007_080108/results/`](detector_repair_20261007_080108/results/).
Read its [`STAGE17A_REPORT.md`](detector_repair_20261007_080108/results/STAGE17A_REPORT.md)
and [`stage17a_status.json`](detector_repair_20261007_080108/results/stage17a_status.json).

| Arm | mAP50-95 (%) | Delta vs RR (pp) |
| --- | ---: | ---: |
| RR | 39.749856 | 0 |
| Vanilla-A | 36.296590 | -3.453266 |
| Vanilla-B | 34.318704 | -5.431152 |
| Vanilla-C | 33.387179 | -6.362677 |

Mean delta is -5.082365 pp; wins are 0/3. Generation-bank sample SD is
1.485735 pp; range is 2.909411 pp. Flash mean AP50-95 delta is -8.417239 pp;
Black mean delta is -1.747491 pp. All three predeclared gates fail.
No Stage17B or generator-mechanism development is automatically started.

## Audit trail

- `trainonly_20261006_022714/`: separately authorized train-only generator
  preparation, source isolation, training runtime and checkpoint asset hashes.
  This is not the unavailable historical baseline checkpoint.
- `normals_isolated_v1_20261006_064503/`: the user-authorized 174-image
  normal pool, excluding six known related-validation source partitions;
  original files and historical records remain unchanged.
- `slots_frozen_20261007_020941/`: immutable 80 slots, 40/class, source
  reuse audit and frozen projected annotations.
- `pregeneration_20261007_024210/`: mask/label coverage audit and 30
  before-generation probe images (15 independently reloaded pairs).
- `banks_20261007_035832/`: 240 formal images, A/B/C bank manifests,
  environment/protocol audits and 15 after-generation probes. All 15
  before/formal/after groups have exactly identical decoded RGB.
- `detector_20261007_065502/preparation/`: the four frozen detector
  dataset manifests and full-image RealRepeat mapping. Its initial training
  attempt failed during dataset construction; it is not a completed arm.
- `detector_repair_20261007_080108/results/`: four completed 150-epoch
  arms, raw final metrics, each run's args/CSV, budget/environment/validation
  audits, and `order_records/<arm>/epoch_*.json` (600 raw epoch records).

Other directories retain earlier failed, diagnostic and local-test evidence;
they are not replacements for the final results identified above.

## Fairness and interpretation limits

All arms use YOLO11s, pretrained initialization protocol, detector seed 42,
imgsz 1536, batch 1, and 150 epochs. Each arm has 32,700 primary draws/batches
and 544 optimizer attempts. Primary order is equal across four arms; A/B/C
secondary augmentation fetch sequences match. Successful updates are
RR/A/B/C = 532/533/533/534, so strict successful-update equal compute is NOT
claimed. Final epoch `last.pt` is evaluated once per arm, only after all
training finishes; no best-checkpoint selection is used.

RealRepeat repeats complete original parent images and full labels.
Extra class-role draws match, but bbox-instance exposure is not identical.
This is a fixed-detector-seed generation-bank calibration, not a
three-detector-seed statistical confirmation or a causal explanation of the
observed loss. Existing annotation omissions, prelabel provenance and
Small/Big Black semantic limitations remain; no labels were corrected.
The conclusion is utility under the frozen annotation protocol, not proof
about true physical-defect recognition.

Model checkpoints and original dataset images are not Git payloads. Their
recorded SHA256 inventories are included; checkpoint bytes were verified
on the server, not independently rechecked from downloaded detector weights.
Generated image PNGs, audits and raw per-arm metrics ARE included.
Line-ending conversion is disabled for audit assets to preserve recorded
SHA256 values. Server paths in historical manifests are provenance, not
portable local filesystem paths.
