# DWBG-v2 results package

This lightweight package contains the reports and scored candidate manifests needed to review DWBG-v2. It intentionally excludes model weights, full generated image pools, datasets, and intermediate logs.

- `dwbg_weakness_fixed_v3_s42.json`: three-fold OOF weakness profile.
- `oof_integrity_report.json`: OOF split and leakage audit.
- `candidate_coverage_report.json`: coverage before targeted Flash-large expansion.
- `candidate_coverage_report_expanded.json`: coverage after the currently completed expansion rounds.
- `scored_candidates_v2_fixed_v3_rect_s42.json`: initial scored pool.
- `scored_candidates_v2_fixed_v3_expanded.json`: merged scored pool after expansion.

Final candidate selection and downstream YOLO training have not been run from this package.
