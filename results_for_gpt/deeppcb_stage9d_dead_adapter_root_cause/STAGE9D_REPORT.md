# DeepPCB Stage9D Dead Adapter Root-Cause Audit

Status: **NONFINITE_GRADIENT_AMP_SCALE_COLLAPSE**.

## Decision evidence

- formal_projection_state_exact_zero: True
- entire_formal_adapter_matches_fresh_initialization: True
- fresh_nonfinite_gradient: True
- fresh_gradient_reaches_projection: False
- fresh_one_step_makes_finite_nonzero_residual: False
- diagnostic_successful_updates: 0
- diagnostic_parameters_update: False
- diagnostic_stopped_on_zero_scale: True
- mask_support_collapse: False
- hooks_and_conditions_execute: True
- state_dict_roundtrip_preserves_values_including_nan: True
- amp_skip_status: AMP_SCALE_COLLAPSED_TO_ZERO

The Stage9A arm remains formally failed and is reinterpreted as an unadapted pretrained-backbone control because its learned residual path was inactive.
UNADAPTED_PRETRAINED_BACKBONE_INSUFFICIENT=true, but neither full-UNet fine-tuning nor LoRA is declared required: the lightweight path never received a valid empirical test.
VAE_BOTTLENECK_NOT_SUPPORTED remains frozen from Stage9C. All subsequent fidelity work must use the Stage7B/8B adaptive padded crop protocol.

No detector, formal 2,000-step model, new architecture, LoRA, or official-validation experiment was run.
