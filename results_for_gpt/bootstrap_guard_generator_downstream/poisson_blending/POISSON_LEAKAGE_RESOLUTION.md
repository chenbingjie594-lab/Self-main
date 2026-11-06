# Poisson leakage resolution

The historical **0.4511 ± 0.0141** mAP50-95 result is invalid and remains archived under `invalid_stage20b_donor_run/`; it is not included in the clean aggregate.

## Root cause and repair

- Root cause: 23 original Stage20B donor images were byte-identical to frozen downstream validation images.
- The candidate donor pool is rebuilt from the 138 frozen Real-only training images and their 168 labels. Instances belonging to the two train/validation-overlapping source groups are excluded, leaving 166 eligible instances.
- Original Stage20B target boxes were also rejected because their coordinates came from the affected annotations. New boxes are deterministic functions of slot ID and clean donor dimensions.
- The original slot IDs are retained. Three normal backgrounds sharing validation lineage are deterministically replaced with clean protocol backgrounds; the remaining backgrounds are retained. Donor identity and bbox are recorded independently.

## Critical gates

- Donor audit: **PASS**; full-pool SHA/canonical/group/parent overlaps: 0/0/2/0; all affected instances were excluded from donor eligibility.
- Dataset audit: **PASS**.
- Generation audit: **PASS**, complete 80/80; no selection or reroll.
- Normal-background audit: three affected backgrounds replaced; effective validation overlap 0.

Other Stage20B-based methods are reported in `stage20b_downstream_split_impact_audit.json`. They are not automatically declared invalid; methods consuming affected frozen boxes/references require requalification.
