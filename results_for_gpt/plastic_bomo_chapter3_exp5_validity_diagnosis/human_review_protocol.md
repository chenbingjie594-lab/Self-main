# Exp5 Human Review Protocol

Review the fixed 24-group diagnostic subset only. First consult `real_reference_examples.png` from Exp4; those images are representative real examples and are not paired ground truth.

For every generated image, fill all fields without deleting uncertain cases:

- visibility: `VISIBLE`, `NOT_VISIBLE`, or `UNCERTAIN`
- position: `INSIDE_BBOX`, `PARTIALLY_OUTSIDE`, `OUTSIDE_BBOX`, or `UNCERTAIN`
- morphology: `PLAUSIBLE`, `SUSPICIOUS`, or `UNCERTAIN`
- artifact: `YES`, `NO`, or `UNCERTAIN`
- notes: free text

The 72 rows describe only this frozen diagnostic subset. Do not report them as a 720-image validity rate. Low-information candidates are prompts for inspection, not invalidity labels.
