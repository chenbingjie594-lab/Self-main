# Stage20B-P0 local verification

Executed on 2026-10-08. Final local status: `SERVER_ASSET_BINDING_PENDING`.
No server verification or execution authorization is claimed.

- 11 P0 unit/integration tests passed, no skips.
- 5 Stage20A regression tests passed.
- Python compilation and Bash command-file syntax checks passed.
- 666 TRAIN-only image/label/normal files were hash-verified.
- Public SD2 UNet/VAE/text weights matched the pinned LFS SHA256 references;
  tokenizer/config/scheduler metadata matched public Git blob IDs.
- 80 legal slots, 40/class, with 80 distinct annotations/images/source groups.
- All 75 Flash / 97 Black TRAIN annotations passed the fixed crop contract.
- 394 TRAIN normals retained, no quality/brightness filtering.
- Portability/runtime-guard revisions did not change the frozen slot file:
  SHA256 `a32cfab656156a37a3263459cf48ac60ed95342bc69f9a71289ff191d91810fc`.
- Initial local audits were preserved under
  `logs/plastic_bomo_stage20b_p0_preflight/`, not deleted or overwritten.
- No tracked Stage20A frozen artifact/spec changed.
- No model import, training, synthetic image generation, final_eval content read,
  TDCRG, BootstrapGuard or DeepPCB execution occurred.

Negative tests cover authoritative-manifest changes, missing public base,
insufficient legal slots, target crop truncation and unsupported detector native
arguments. Positive tests cover source-group round-robin, edge padding, geometry
inversion, Windows-to-Linux path names, explicit default resolution, bank pairing
and the materialized 150-epoch role/optimizer-attempt schedule.

These tests do NOT establish GPU numerical reproducibility, end-to-end sampling,
runtime optimizer execution or successful AMP update equality. Those were not run.
