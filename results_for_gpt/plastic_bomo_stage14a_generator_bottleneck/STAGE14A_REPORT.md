# Plastic_Bomo Stage14A Generator Bottleneck Audit

Status: **IMAGE_PROPERTY_BOTTLENECK_NOT_IDENTIFIED**.

Recommended next direction: **generation-time task-aware optimization**.

## Equal-budget seed42 screening

| Arm | mAP50-95 | Delta vs Random (pp) |
|---|---:|---:|
| real_repeat | 0.398069 | -0.636 |
| random | 0.404425 | +0.000 |
| high_fidelity | 0.427283 | +2.286 |
| valid_novel | 0.377434 | -2.699 |
| high_context_compatibility | 0.385682 | -1.874 |
| low_fidelity_diagnostic | 0.365076 | -3.935 |
| low_context_diagnostic | 0.372125 | -3.230 |

Fidelity protocol complete: **False**. The morphology-matched high/low contrast is retained as diagnostic evidence, but it is not labeled broad fidelity evidence unless crop-KID and real-nearest LPIPS are complete.

Novelty was diagnostic only: duplicate-like and off-manifold groups could not supply 40 samples/class, so no novelty causal claim was made.

All trained arms used the same 138 real images, 80 extras (40/class), YOLO11s initialization, seed42, 150 epochs, batch 1, and final `last.pt`. Official validation was used only after training. DeepPCB training/generation count remained zero. Stage14B was not started.
