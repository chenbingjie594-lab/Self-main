# DeepPCB Stage9A Frozen-Backbone Feasibility

Status: **FROZEN_ADAPTER_NOT_FEASIBLE**.

Stage9A used 258 frozen low-shot sources (129 short and 129 pinhole), three
predeclared generation seeds, and an exact 774-sample matrix per arm. No
detector was trained, official validation was not used, and no weighted quality
score was constructed.

## Gate result

| Class | Fidelity | Diversity | Identity gate | Task support | Overall |
|---|---:|---:|---:|---:|---:|
| short | False | False | False | False | False |
| pinhole | False | False | False | False | False |

The failure is large rather than threshold-sensitive. Relative to Vanilla SD2,
the FrozenResidualAdapter increased mask-crop LPIPS from 0.1870 to 0.3485 for
short and from 0.2319 to 0.4735 for pinhole. Edge cosine fell from 0.6298 to
0.4149 and from 0.4247 to 0.0975, respectively. Boundary-gradient cosine also
fell for both classes.

Task-feature support collapsed. Mean normalized pair gap changed from
0.6252 to 2.3916 for short and from 0.8406 to 3.3058 for pinhole. The fraction
above the real q95 threshold rose from 17.05% to 88.89% and from 28.68% to
98.45%. These task-support failures alone are sufficient to reject the
prototype under the preregistered gate.

The adapter reduced source retrieval top-1 (short: 34.37% to 8.27%; pinhole:
25.58% to 6.46%), but this did not represent useful variation: IC-LPIPS was
lower than Full MSDF for both classes, dramatically so for pinhole (0.1700 to
0.0526), while fidelity and task correspondence deteriorated. Source
memorization was therefore replaced by off-manifold or collapsed outputs, not
by task-useful conditional diversity.

## Training and bottleneck audit

Only 108,928 adapter parameters were trainable (0.01258% of the 865,925,124
parameter UNet). Both classes completed exactly 2,000 successful updates with
the UNet, VAE, and text encoder frozen. The frozen VAE itself retained the real
training crops reasonably well (short LPIPS 0.0490; pinhole LPIPS 0.0602), so
the observed generator failure cannot be attributed solely to catastrophic VAE
reconstruction loss.

Vanilla SD2 full-UNet relative L2 drift was small but nonzero across all six
classes (0.00475--0.00530), confirming that the existing baseline can encode
class/source information throughout the backbone. This drift analysis is
diagnostic only and was not used by the gate.

## Decision

`STAGE9B_DETECTOR_PILOT_AUTHORIZED=false`

Per the frozen protocol, Stage9A stops here. No detector pilot, adapter repair,
new loss, new selector, or automatic architecture modification is authorized.
