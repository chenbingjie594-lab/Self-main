# Stage22A downloaded-result review

Formal status: `SYNTHETIC_OPTIMIZATION_MECHANISM_NOT_IDENTIFIED`.

The downloaded server audit completed 292 forward/backward probes: all 109 real internal_train images (63 Flash, 46 Black), and all 61 matched synthetic slots (32 Flash, 29 Black) in each of A/B/C. The protocol matches the implementation committed at `68672117672c64972d2ebc98c8720a67999588c2`. Metadata, source bindings, selected parameter hashes, record populations, per-class summaries, and matched triplet statistics pass the separate CPU review. No image was decoded and no detector was run during this review.

The server records report zero optimizer steps, generator/detector training, generation, internal_dev content access and official final_eval access. The downloaded formal report and numerical audit files are preserved. `review_integrity_audit.json` records the additional review, and `result_artifact_manifest.json` binds the result files by SHA-256.

## 1. No measured negative-cosine conflict

Every A/B/C synthetic gradient had positive cosine with both the global and its same-class real reference. Global and same-class conflict fractions, using the frozen `cosine < 0` rule, are 0 for both classes and every bank. Across all 292 real/synthetic records, the minimum global cosine is 0.090134 and the minimum same-class cosine is 0.113735.

This does not support a negative-gradient-conflict explanation at the measured initialization and selected parameters. It does not establish that conflicts cannot arise during training or in other parameter groups.

## 2. Mild loss/norm shifts do not pass Gate A

| Class | Group | Total loss median | Selected-gradient L2 median | Same-class cosine median |
|---|---|---:|---:|---:|
| Flash | Real | 136.1175 | 898.6199 | 0.397541 |
| Flash | A | 137.8763 | 973.4884 | 0.292401 |
| Flash | B | 138.8181 | 1002.7929 | 0.288351 |
| Flash | C | 138.0272 | 1018.1569 | 0.286979 |
| Black | Real | 135.8963 | 875.6373 | 0.283237 |
| Black | A | 138.2204 | 921.0210 | 0.312502 |
| Black | B | 139.4541 | 948.9389 | 0.302728 |
| Black | C | 138.2961 | 920.4459 | 0.310444 |

Synthetic loss and selected-gradient norm medians are higher in every bank and class, but they do not meet the predeclared separation requirement across banks/classes. Gate A fails for loss, norm and sign conflict. Therefore the NOT_IDENTIFIED decision remains supported even if the additional annotation-count guard is disregarded; it is not solely the consequence of requiring matching target-count distributions.

Flash synthetic gradients have lower same-class alignment than Flash real leave-one-out controls; Black synthetic gradients have higher alignment than Black controls. This class difference does not provide a uniform directional-conflict explanation for the observed downstream degradation. The same-class cosine is a descriptive statistic and has not been used to screen or weight samples.

## 3. Generation RNG changes optimization signals

| Class | Median within-slot loss range | Median within-slot gradient-norm range | Median within-slot same-class cosine range |
|---|---:|---:|---:|
| Flash | 4.491287 | 139.077904 | 0.024765 |
| Black | 4.504227 | 113.779584 | 0.028550 |

These ranges compare A/B/C pixels for the same frozen class, donor/conditioning roles and target label. They demonstrate observable variation with generation RNG. The frozen Gate D requires nonzero variation; it is not a statistical-significance test and does not establish that these differences caused bank utility variation.

Only descriptive bank ordering was checked. No Pearson/Spearman correlation across the three banks was computed, and no utility predictor was established.

## 4. Feature differences are not a diversity or causal result

Backbone and neck ROI mean distances, covariance traces and effective ranks were computed separately for each class. Their presence documents distribution differences at the public initialization. The signs and bank ordering of covariance-trace changes differ by class/layer.

Synthetic effective ranks are lower, but the empirical covariance rank has different sample-size limits: Flash real/synthetic maxima are 62/31, and Black maxima are 45/28. Therefore the reported effective ranks cannot, by themselves, establish reduced intrinsic synthetic diversity. The feature metrics were not measured as within-slot triplet changes, so feature evidence does not satisfy all A–E requirements.

## 5. Interpretation and evidence limits

- Real images include multiple same-class targets: Flash has 61 one-box and 2 two-box images; Black has 26 one-box, 12 two-box and 8 three-box images. Synthetic slots have one target. The primary references correctly retain all real images, but target multiplicity remains a confound.
- The median per-image classification-loss fraction of total loss is 98.2603%. The selected late-backbone tensor contributes a median 82.6536% of the concatenated selected-gradient squared norm. The measured total loss/direction therefore cannot be treated as a dedicated localization/head diagnostic.
- The probe uses one frozen public initialization, deterministic preprocessing and one architecture. It supplies no training-trajectory or intervention evidence.
- Raw gradient vectors and ROI feature vectors were not exported. The CPU review independently verifies the recorded norm, direction/conflict and triplet summaries; it cannot reconstruct the gradient references/cosines or feature covariance moments from vectors.
- The downloaded artifacts do not record the executed server script SHA-256 or separate post-probe full parameter/buffer hash dictionaries. The completed runner path checks parameter equality and restored buffers before writing final status; the downloaded selected hashes and initialization reference match the frozen metadata. This is the available immutability evidence, not an independently replayed GPU check.

No unique strongest mechanism is established. Stage22B has not been authorized or started. Stage22A ends with this reviewed descriptive result.
