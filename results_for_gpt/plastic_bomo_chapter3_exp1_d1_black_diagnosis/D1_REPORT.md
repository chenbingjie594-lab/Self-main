# Experiment1-D1 Black Failure Diagnosis

Status: `BLACK_GAP_ROBUST_BUT_MECHANISM_UNRESOLVED`.

## Paired size27 resampling

Seed2026,1000 rounds per class, replacement sampling, fixed balanced A/B/C schedule, shared Raw/MSDF indices. Original cached2048-D features and unchanged torch-fidelity KID. FID uses the algebraically exact low-rank cross-Gram nuclear norm with unbiased covariance (ddof1), only after every comparison with the original standard2048-D implementation passes abs_error<=1e-5+1e-6*abs(standard_FID), explicitly authorized after the archived strict-only failure. All1000 rounds per class are recomputed in a separate lowrank-only cache; legacy slow rows are audit references only. Intervals are conditional empirical bootstrap intervals, not causal tests or uncertainty across retrained models.

- flash delta_KID: mean=-0.00122939, median=-0.000527048, 95% interval=[-0.026513280711073946, 0.02127145114748058], P(delta>0)=0.4610.
- flash delta_FID: mean=-15.9874, median=-15.4064, 95% interval=[-37.57351838277354, 5.220677970805558], P(delta>0)=0.0660.
- black delta_KID: mean=0.0543689, median=0.0530213, 95% interval=[0.01845258972199987, 0.09864280957257177], P(delta>0)=0.9990.
- black delta_FID: mean=51.8643, median=50.9361, 95% interval=[18.646770419144726, 85.7309863971017], P(delta>0)=1.0000.

## Observable shifts

```json
{
  "flash": {
    "FID_KID_gap_robust": false,
    "nearest_reference_farther": false,
    "intensity_and_contrast_deviation_larger": false,
    "local_appearance_association": false,
    "background_association": false
  },
  "black": {
    "FID_KID_gap_robust": true,
    "nearest_reference_farther": true,
    "intensity_and_contrast_deviation_larger": false,
    "local_appearance_association": false,
    "background_association": false
  },
  "black_specific_local_pattern": false,
  "black_specific_background_pattern": false
}
```

All Flash diagnostics use the same protocol as Black. A shift shared by Flash is not Black-specific. Appearance statistics and nearest-reference LPIPS are descriptive diagnostics, not new realism scores. Greater cross-bank LPIPS variation must not be interpreted as better diversity or as causing degradation.

Background PSNR/MSE use all RGB pixels in P-minus-B. SSIM uses only11x11 windows fully contained in that region; canvas-minus-P copied pixels never enter the summaries. Infinity PSNR is retained and unsupported paired summaries are unavailable, not capped or filtered.

Raw and MSDF have identical condition bboxes; bbox geometry cannot explain the paired gap by itself. No direct held-out reference role exists for diagnostic grid slots; N/A panels are intentional, not missing selected images. All24 fixed-random matched groups and their residual plots are retained; residual scale is globally0..255.

No checkpoint, seed, crop, image selection, method, loss or attention module was changed. No training, generation, detector, BootstrapGuard or DWBG ran. A visual association is not proof of the MSDF causal mechanism. If evidence is not coherent, mechanism remains unresolved.

## Sources

- Experiment1 frozen input/metric protocol, runtime and feature cache (source commit 59a4451d7662d09423b85220a4cc523f76bcf360).
- https://github.com/toshas/torch-fidelity/blob/v0.3.0/torch_fidelity/metric_fid.py
- https://github.com/toshas/torch-fidelity/blob/v0.3.0/torch_fidelity/metric_kid.py
- https://github.com/richzhang/PerceptualSimilarity
- https://scikit-image.org/docs/0.25.x/api/skimage.metrics.html

Diagnosis complete. No follow-up experiment is authorized automatically. Git publication is performed separately after review.
