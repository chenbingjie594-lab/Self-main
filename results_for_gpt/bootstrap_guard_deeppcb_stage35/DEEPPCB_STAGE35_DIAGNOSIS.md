# DeepPCB Stage 3.5 Diagnosis

## Decision

Status: **`CLASS_IMAGE_MERGE_ERROR`**

Manifold assessment: **`INVALID_DUE_TO_LABEL_BUG`**. The Stage 3 selection manifests were not modified, and Stage 4 must not start from the current selections.

## Independent RAW audit

- Every frozen pair corresponds to exactly one official RAW tested image and one image SHA256.
- The six class annotations genuinely belong to that same tested image; there is no RAW image-class aggregation bug.
- RAW and converted annotations reconcile instance by instance with zero mismatches.
- Counts are open 130 and 129 for each of short, mousebite, spur, spurious_copper, and pinhole (775 total).
- End-to-end class mapping is consistent: open=0, short=1, mousebite=2, spur=3, spurious_copper=4, pinhole=5.

The 20-pair RAW pixel sample contains visible tested-vs-template evidence in every class. Minimum bbox MAE is 10.82–19.81 across classes, so the five zero-recall classes are not explained by absent RAW pixel changes.

## Proven Stage 3 pipeline error

The converted dataset intentionally creates one independent 512×512 crop per annotation. Each crop contains only its target real defect pasted over its paired normal background.

Stage 3 then incorrectly:

1. selected the first converted crop for each physical pair as the detector image;
2. attached all annotations from that pair as six-class YOLO labels; and
3. treated bbox coordinates from independently centered crops as coordinates on the first crop.

Registry ordering made that carrier the `open` instance. File hashes prove that **300/300** fold/split image copies contain only an `open` carrier while their label files contain all six classes. This directly explains why `open` learned and the other five classes had zero recall.

The label text itself is numerically valid: 300 images, 2,325 lines, zero invalid class IDs, non-finite values, invalid normalized boxes, or exact duplicate lines. The failure is the image-to-label spatial correspondence.

## Bbox overlap audit

- Cross-class IoU ≥ 0.5: 39 occurrences across the three fold copies.
- Cross-class IoU ≥ 0.8: 6 occurrences.
- Cross-class IoU ≥ 0.95: 0.
- Exact cross-class duplicate boxes: 0.

Thus duplicate class assignment is not the primary issue; the carrier/coordinate-frame mismatch is.

## Stopped diagnostics

Per the frozen protocol, raw prediction analysis and the 12-pair exact-overfit test were skipped after the class/image merge error was proven. Continuing detector experiments on known-invalid supervision would not provide meaningful evidence.

## Artifact impact

| Artifact | Assessment |
|---|---|
| Stage 0.5 | Not affected; RAW low-data statistics and frozen split remain valid. |
| Stage 1 | Not affected; per-instance MSDF training and smoke generation remain valid. |
| Stage 2 | Not affected; per-instance candidate pool generation remains valid. |
| Stage 3 OOF probe | Invalid supervision. |
| Stage 3 manifold | Invalid due to label bug. |
| Stage 3 Random | Identities are reproducible, but shared manifold eligibility is tainted. |
| Stage 3 DQ | Invalid as a quality baseline because its ranking uses the invalid manifold. |
| Stage 3 BootstrapGuard | Distribution constraints are reproducible, but shared manifold eligibility is tainted. |

The Stage 3 results must not be silently repaired or reused as validated selections. A separately authorized rebuild must reconstruct pair-level OOF images directly from the RAW tested images, transform RAW boxes in one consistent coordinate system, retrain the OOF probes, rebuild the manifold, and rerun all three selectors from the unchanged Stage 2 pool.
