# Poisson Blending formal-result blocker

The completed run is invalid for the requested Chapter 3 comparison and has not been uploaded as a formal result.

- Stage20B marks all 80 donors as logical V2 TRAIN.
- A direct SHA256 comparison against the frozen YOLO validation split found 23 exact image overlaps.
- The user-specified protocol explicitly forbids validation donors, so Stage20B logical split metadata cannot override this overlap.
- The observed `0.4511 ± 0.0141` mAP50-95 is therefore not eligible for the main table.
- No samples were silently replaced and no alternative augmentation was used.

A valid rerun needs a new deterministic donor mapping drawn only from the 138 downstream training images. Because that would change the frozen donor `annotation_id` and `source_group_id` bindings, it is not performed without explicit protocol approval.
