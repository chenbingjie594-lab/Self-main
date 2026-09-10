# Multi-Detector View Pilot

Status: **SECOND_DETECTOR_TOO_WEAK**

This is a descriptive OOF error-view analysis only. No synthetic selection, utility scoring, or downstream training was performed.

- Records: 168 (Flash=88, Black=80)
- RT-DETR OOF instance success rate: Flash=0.318, Black=0.250

## Failure overlap
- Flash: Jaccard=0.309; YOLO-only=8; RT-DETR-only=39; consensus=21
- Black: Jaccard=0.246; YOLO-only=1; RT-DETR-only=45; consensus=15

## Interpretation
RT-DETR shows many detector-specific failures, but its held-out instance success rate is low, especially for Black. Therefore detector-specific errors cannot yet be treated as reliable complementary views; the current gate is SECOND_DETECTOR_TOO_WEAK.

## Attribute comparison
Attribute-level recalls are in `attribute_weakness_comparison.json`. These descriptive comparisons are not used for candidate selection.
