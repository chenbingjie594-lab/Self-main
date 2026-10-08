# Stage18A — Existing Generator Base Viability Audit

## 1. Why stop the current Vanilla base?

Stage17A clean Vanilla A/B/C lost to RealRepeat by -3.4533/-5.4312/-6.3627 pp;
mean -5.0824 pp, wins 0/3. Flash/Black mean deltas were -8.4172/-1.7475 pp.
These are three generation banks with detector seed 42, not a three-detector-seed claim.
`VANILLA_STAGE17A = NOT_VIABLE_AS_GENERATOR_BASE`. No extra bank, seed, tuning or module is authorized.
The 1.4857 pp nuisance SD / 2.9715 pp threshold applies ONLY to this Vanilla protocol, not another generator.

## 2. Is an auditable alternative already available?

`NO_AUDITABLE_ALTERNATIVE_GENERATOR_BASE` in scope `SERVER_EXISTING_EVIDENCE`. No alternative has a complete train-only lineage chain in the inspected evidence.
This is not proof that missing server assets do not exist. Server inventory pending: False.
AnomalyDiffusion code and a Plastic_Bomo launcher exist, but execution, output and checkpoint linkage are unconfirmed.
DefectFill has a prepared INPUT conversion report, not proof of generated outputs. Its local step_200 log identifies concrete/crack, not Plastic_Bomo.
Its inference code allows fallback to test masks if train masks are absent; actual historical use is unknown.
AnomalyDiffusion native generation writes image/mask outputs, which do not establish a detector bbox manifest.
Historical SD checkpoint files alone cannot establish the frozen training split. MSDF-v3 checkpoint identity is unverified.
RDA/CARF/DHFG configurations and historical notes are evidence of experiments, not verified clean output banks.

| Family | Isolation | Manifest-declared synthetic count | Verified legal Flash/Black |
|---|---|---:|---|
| ANOMALYDIFFUSION | ISOLATION_UNRESOLVED | None | {'flash': 0, 'black': 0} |
| DEFECTFILL | ISOLATION_UNRESOLVED | None | {'flash': 0, 'black': 0} |
| MSDF_V3 | ISOLATION_UNRESOLVED | None | {'flash': 0, 'black': 0} |
| MSDF_V3_DWBG_POOL | ISOLATION_UNRESOLVED | 761 | {'flash': 576, 'black': 185} |
| HISTORICAL_SD_BASELINE | ISOLATION_UNRESOLVED | None | {'flash': 0, 'black': 0} |
| RDA | ISOLATION_UNRESOLVED | None | {'flash': 0, 'black': 0} |
| CARF | ISOLATION_UNRESOLVED | None | {'flash': 0, 'black': 0} |
| DHFG | ISOLATION_UNRESOLVED | None | {'flash': 0, 'black': 0} |

## 3. Are there enough raw synthetic images and legal annotations?

The DWBG manifest declares 761 records, Flash 576, Black 185.
Declared bboxes are checked separately from actual label files. Missing remote files are not counted as verified legal samples.
Images, masks, input conversions, triplet previews and unrelated-dataset demonstrations are not interchangeable with a legal Plastic_Bomo detection bank.
No family has verified all requirements for a matched 138-real + 80-synthetic vs 138-real + 80-RealRepeat screen.
Annotation caveats remain: 88 Flash / 30 Black XML lineage; 25 supplemental Black images / 50 boxes are historical prelabels without resolved instance verification;
suspected omissions and Small/Big Black ambiguity remain unchanged. No labels or splits were edited.

## 4. Is the evidence selector independent?

DWBG is post-generation pool construction/scoring/selection here, not an additional UNet generator architecture.
Historical builder code performs residual extraction, quality/attribute rejection, train-box geometry resizing and background recomposition.
Support-based YOLO boxes are constructed after recomposition. The 761 records have matching composition/quality fields; this supports the derivation,
but exact execution/checkpoint lineage remains unresolved. They must NOT be called 761 untouched raw MSDF outputs.
Stage14 Random IDs are checked against this pool and the older Random manifest. That manifest explicitly samples `manifold_valid` only,
with parent/seed caps. Random-within-a-filtered-pool is not proof of selector-free raw utility.
Detector scores are present in the scored manifest; this alone does not prove detector-guided generation, nor absence of earlier rejection.
Actual validation involvement in historical pool construction is unresolved, not asserted zero.
HighMorph/M10/M01/BalancedMorph50 and BootstrapGuard results are subset/selection signals, not generator utility confirmation.

## 5. Authorize one Stage18B screen?

`STAGE18B_BASELINE_UTILITY_SCREEN_AUTHORIZED = false`. No candidate selected. Current evidence supports retiring the generator-first route
on this frozen dataset, subject to the stated unresolved remote-asset scope; it does not establish universal absence of an alternative generator.
The supplied server command only inventories existing assets and checks already-declared synthetic labels. It never generates or trains;
finding files cannot automatically pass isolation. A new evidence review is required before any authorization.

All activity counts (sampling, generator/detector training, optimizer steps, validation access, DeepPCB, BootstrapGuard changes) are zero.
