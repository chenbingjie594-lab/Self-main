# Assignment sanity report

legacy v1.1 passed model-input xyxy pixels into criterion.preprocess(), which expects normalized xywh; this double-scaled/misformatted GT coordinates.

Status: **ASSIGNMENT_PIPELINE_FIXED**
