# Stage25 MSDF-V2 Clean Requalification

MSDF_GENERATOR_IMPROVEMENT_NOT_CONFIRMED

Mean MSDF−Raw: -2.508203 pp; strict wins 0/3.

Mean MSDF−RR: -5.098347 pp; strict wins 0/3.

Final_eval was historically observed in Stage20B, not an independent untouched test. No Stage25 result-dependent tuning.

Core/literal historical losses frozen. Runtime/data shell differences D7/D8 are declared in the shared protocol.
Raw outputs retained240/240; no selector/filter/reroll. Epoch150 native EMA last.pt, no validation selection.

## Manual interruption recovery

Source: `formal_20261013_085920_253633`. Original run/logs/partial A remain unchanged.
Both step2000 generators and all240 images were SHA-verified and reused; no new generation.
A/B/C each restarted from the frozen public initialization, seed42, epoch1 through150. No optimizer/epoch resume, extra primary budget, or success-step compensation.
Interrupted A physical compute is recorded separately in manual_detector_recovery.json and recovery/source_snapshot/, not hidden or pooled into matched primary exposure.
SSH disconnection is user-reported, not independently established as the cause. Immutable assets may share hardlinks with the source; do not edit either copy.
