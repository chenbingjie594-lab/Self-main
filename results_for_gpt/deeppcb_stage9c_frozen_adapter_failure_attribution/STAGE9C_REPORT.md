# DeepPCB Stage9C Frozen Adapter Failure Attribution

Status: **SEMANTIC_ADAPTATION_DEFICIT_SUPPORTED**.

Canonical fidelity audit: **FIDELITY_FAILURE_RECONFIRMED**. Historical Stage9A absolute LPIPS used a tight bbox, while Stage7B/8B used an adaptive padded crop; historical values are not directly comparable across stages. The within-Stage9A adapter failure was re-evaluated without changing its formal gate.

Identity protocol direction correct: **false**. The Stage9A within/total direction was a protocol bug; the Stage9A status remains unchanged because fidelity and task support failed independently.

## Residual and semantic probes

- short block 2: median local residual/hidden=0.0000, source-level area Spearman rho=undefined.
- short block 3: median local residual/hidden=0.0000, source-level area Spearman rho=undefined.
- pinhole block 2: median local residual/hidden=0.0000, source-level area Spearman rho=undefined.
- pinhole block 3: median local residual/hidden=0.0000, source-level area Spearman rho=undefined.

This is an exact effective no-op, not merely a residual below the overshoot
threshold. All 12,000 instrumented residual records were exactly zero across
both injection blocks, both CFG branches, and all 50 diffusion timesteps. The
adapter-ON and adapter-OFF output hashes were identical for all 60 probe
sources. Consequently, there is no evidence that local injection actively
pushed samples away from the real manifold; the existing checkpoint produced
no effective inference-time injection at all.

Local overshoot supported: False.
Frozen-backbone semantic deficit supported: True.

The no-adapter pretrained backbone was already substantially weaker than the
class-finetuned Vanilla SD2 control. For short, normalized pair gap was 2.3451
for pretrained OFF/adapter ON versus 0.6848 for class-finetuned SD2, and the
fraction above real q95 was 90.0% versus 16.7%. For pinhole, pair gap was
3.3632 versus 0.8187 and the q95 fraction was 96.7% versus 30.0%. Canonical
LPIPS showed the same direction (short 0.0450 versus 0.0211; pinhole 0.1050
versus 0.0323). This supports insufficient semantic adaptation of a completely
frozen pretrained backbone.

Numerical audit: **NUMERICAL_INSTABILITY_UNCLEAR**. VAE claim audit: **VAE_BOTTLENECK_NOT_SUPPORTED**.

The Stage9A identity direction was also incorrect: lower within/total variance
means tighter same-source clustering and therefore stronger, not weaker,
source identity. This protocol bug does not change the Stage9A failure status.

The canonical re-audit explains the historical LPIPS discrepancy. Stage9A used
a tight mask bounding box, whereas Stage7B/8A/8B used a padded crop with a
minimum size of 64 and context scale 4. Under the canonical protocol, the
fidelity failure remains present, so its direction is reconfirmed even though
the historical absolute values should not be compared across evaluators.

No detector was trained; official validation was not used; Stage9A files and gate were not modified. Stage9D is not authorized. Future ideas, if warranted by the attribution, remain report-only and were not implemented.
