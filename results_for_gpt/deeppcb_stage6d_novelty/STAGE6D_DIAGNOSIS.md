# DeepPCB Stage 6D frozen six-seed novelty diagnosis

Status: **SYNTHETIC_NOVELTY_LIMITED**. Train-only descriptive evidence; no causal claim.
Seeds discovered from manifest: [42, 2026, 3407, 7859, 12011, 15683]. Candidates: 4650; sources: 775.
OOF Detect-input P3/P4/P5 with same pooling, fold and real-only k=5 reference; validation image use: 0.

| Class | Diversity ratio median | Same-real nearest | Inner redundant | Realistic novel | Outside support | Real/synthetic rank | Within/total | Stage6B AP delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| open | 0.106 | 0.776 | 0.571 | 0.301 | 0.128 | 10.63/9.50 | 0.064 | -0.0053 |
| short | 0.062 | 0.811 | 0.616 | 0.304 | 0.080 | 10.95/9.76 | 0.034 | -0.0241 |
| mousebite | 0.074 | 0.776 | 0.609 | 0.324 | 0.067 | 11.20/9.21 | 0.033 | -0.0072 |
| spur | 0.099 | 0.766 | 0.561 | 0.289 | 0.150 | 11.02/9.08 | 0.037 | +0.0051 |
| spurious_copper | 0.102 | 0.779 | 0.559 | 0.375 | 0.066 | 10.76/9.13 | 0.044 | +0.0099 |
| pinhole | 0.074 | 0.778 | 0.521 | 0.346 | 0.133 | 11.12/9.08 | 0.030 | +0.0076 |

Classes meeting fixed low-novelty diagnostic conditions: open, short, mousebite, spur, spurious_copper, pinhole.
Classes meeting ample in-support novelty conditions: none.
The Stage6B class association is exploratory (n=6); no new seed/selector was chosen.
No detector/generator training or candidate regeneration was performed.
