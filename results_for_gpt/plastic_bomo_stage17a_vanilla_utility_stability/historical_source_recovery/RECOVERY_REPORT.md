# Historical Plastic_Bomo conversion recovery

The conversion was found in local Codex history and the old anomalydiffusion-master project. The crop source was recovered from an actual historical apply_patch call, not reconstructed by guessing.

Sources: 112; per-class binding counts: {'flash': {'UNIQUE_EXACT_IMAGE_CROP_BOUND': 62, 'NO_FROZEN_PARENT_REPRODUCTION': 14}, 'black': {'UNIQUE_EXACT_IMAGE_CROP_BOUND': 22, 'NO_FROZEN_PARENT_REPRODUCTION': 14}}.

The historical transform uses bbox-center rounding, crop size max(512,4*bbox-long-side), black padding at borders and JPEG quality95. It was reproduced in memory only, from hash-verified frozen real training images and annotations. Exact JPEG bytes bind the matched images.

The conversion appends sorted old YOLO val crops after train crops, then renumbers. That old split is not assumed to equal current official validation. Unmatched sources are unresolved, not automatically validation leaks. Masks still require transform evidence. No formal slots, generation, retraining, detector training, label modifications or historical status changes were performed.
