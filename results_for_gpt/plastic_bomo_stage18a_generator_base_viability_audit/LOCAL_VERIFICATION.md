# Local verification

The CPU-only audit completed on 2026-10-08. Six unit tests passed:
invalid/nonfinite bbox rejection; missing-file fail-closed behavior; actual image/label
validation with empty/mismatched/duplicate labels; train/test/val/DeepPCB discovery exclusion;
selection classification; frozen protocol constraints.

All output JSON files parsed successfully with strict finite-number serialization.
761 manifest records were audited: 576 Flash / 185 Black. Stage14 Random contains
80 IDs, all in this pool, with identical order and set to the historical Random manifest.
No actual 761 image/label files were available locally; all are recorded as unavailable,
not counted as verified legal examples and not claimed absent from the server.

55 stored detector metric records were retained with their source hashes and raw metrics:
33 selected-synthetic, 12 random-synthetic, 3 raw Stage17A Vanilla, 7 real controls
(UNKNOWN_SELECTION with is_real_control=true, not synthetic utility evidence).
Some records are reused across historical stages; they are not independent replications
and are not statistically pooled. Selected and filtered-pool random records do not unlock
generator development. Stage17A raw evidence is negative, not a candidate to rescue.

The inventory hashed available checkpoint files (~28.2 GB including three downloaded
historical baseline directories). Refreshed evidence interpretation reuses that completed
inventory without repeating model hashing; this is recorded in execution_audit.json.
No checkpoint was loaded. AnomalyDiffusion's sole observed weight is the LPIPS vgg.pth
utility weight, not a Plastic_Bomo generator checkpoint. DefectFill's observed example
log identifies concrete/crack; its input conversion is not a synthetic bank.

Server asset existence and train-only lineage remain unresolved. The server command is
an existing-file inventory only, not permission to run old launchers or Stage18B.
No generator, detector, synthetic selector, original label, original split, Stage17A result,
DeepPCB artifact or BootstrapGuard implementation was changed.
