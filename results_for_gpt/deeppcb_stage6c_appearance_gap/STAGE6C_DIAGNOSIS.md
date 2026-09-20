# DeepPCB Stage 6C paired appearance diagnosis

Status: **SYNTHETIC_APPEARANCE_GAP_PARTIAL**. Descriptive consistency only; not a causal proof.
775 paired train instances, 100 physical pairs; official validation image use: 0.
Frozen Stage0.5 tau_sigma=58.7430814883903; frozen Stage3R Detect-input P3/P4/P5, k=5, fold/class q95.

| Class | Median normalized pair gap | Fraction >1 | Contrast Wasserstein | High-frequency Wasserstein | Residual MAE ratio median | Stage6B AP50-95 delta |
|---|---:|---:|---:|---:|---:|---:|
| open | 0.1774 | 0.062 | 0.0174 | 0.0049 | 0.992 | -0.0053 |
| short | 0.0978 | 0.047 | 0.0251 | 0.0053 | 1.069 | -0.0241 |
| mousebite | 0.1637 | 0.047 | 0.0442 | 0.0098 | 1.208 | -0.0072 |
| spur | 0.1936 | 0.062 | 0.0421 | 0.0107 | 1.240 | +0.0051 |
| spurious_copper | 0.1371 | 0.031 | 0.0693 | 0.0073 | 1.196 | +0.0099 |
| pinhole | 0.2226 | 0.070 | 0.0918 | 0.0131 | 1.557 | +0.0076 |

Classes with median normalized gap >1 and fraction >1 >=0.5: none.
Train-only low-level systematic flag: True; short-vs-gaining-classes alignment: False.
Stage0.5 scale/morphology bins are bbox-derived and identical by construction; they are not treated as appearance evidence.
No detector/classifier training, generation, selection, ratio tuning, or validation pixel access.
