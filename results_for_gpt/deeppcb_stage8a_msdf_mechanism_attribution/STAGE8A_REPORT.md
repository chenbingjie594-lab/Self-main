# DeepPCB Stage 8A MSDF mechanism attribution

Status: **NO_CLEAR_COMPONENT_ATTRIBUTION**.
All interventions use the frozen MSDF-v3 checkpoints and generation seed 42; no detector was trained.
Branch removals are inference diagnostics with documented train-test mismatch, not standalone causal proof.
Pixel/perceptual, task-feature, and boundary axes remain separate; no weighted score was constructed.

Diagnostic trade-off candidates: none.

## Overall diagnostic deltas versus Full

| Variant | normalized pair gap median | LPIPS mean | edge cosine mean | boundary gradient cosine mean |
|---|---:|---:|---:|---:|
| no_latent | -0.004417 | +0.000226 | -0.000598 | +0.001575 |
| no_pixel | +0.001435 | +0.000060 | -0.000443 | -0.001423 |
| no_multiscale | -0.001174 | +0.000099 | -0.000214 | -0.000550 |
| residual_half | +0.001704 | +0.000032 | -0.000138 | -0.000417 |
| residual_zero | -0.001174 | +0.000099 | -0.000214 | -0.000550 |

Lower normalized pair gap and LPIPS are favorable; higher edge and boundary cosine are favorable. No intervention improved both overall LPIPS and edge cosine. Removing latent or all residual injection slightly reduced the feature gap, but worsened LPIPS and edge; therefore neither isolates the fidelity conflict.

## Edge-sensitive classes

For short, every branch removal worsened the feature-gap median and none improved LPIPS; this does not identify the Stage7C short-class loss with a single inference-time component.

For mousebite, no-pixel and half-residual variants produced tiny local LPIPS/edge improvements, but the effects did not hold overall or consistently across boundary and feature axes.

For pinhole, no-latent improved feature gap and boundary cosine while worsening LPIPS and edge, showing a trade-off rather than a dominant harmful branch.

Morphology alignment remains class B: bypass changes spatial support semantics and requires a minimal retraining ablation before a causal claim.
