# Modular experiment configuration

Use `configs/baseline.json` as the clean starting point. All MAGIC additions are
disabled there. Copy it for each ablation instead of changing Python code.

## Baseline

```powershell
python inference.py `
  --config configs/baseline.json `
  --model_ckpt_root <checkpoint-root> `
  --categories screw `
  --dataset_type mvtec `
  --base_dir <dataset-root> `
  --output_name <result-root>
```

`--defect_json`, `--match_json`, `--normal_masks`, and `--mask_dir` are not
needed by the baseline. When `--mask_dir` is omitted, masks are read from
`<dataset-root>/<category>/ground_truth/<defect-class>`. The default
DDIM configuration is read from `./scheduler`; override it with
`--ddim_scheduler_root` when needed.

For a strict baseline checkpoint, train with prompt noise disabled:

```powershell
python run_train.py --base_dir <dataset-root> --output_name baseline --text_noise_scale 0 --category screw
```

## Adding one module

Copy `configs/baseline.json`, change `experiment_name`, and enable only the
module being tested. Available independent switches are:

```json
{
  "modules": {
    "prompt_perturbation": {"enabled": true, "noise_scale": 1.0},
    "spatial_guidance": {"enabled": false},
    "cama": {"enabled": false},
    "ddim_noise": {"enabled": false}
  }
}
```

When `cama.enabled` is true, also pass `--match_json` and `--normal_masks`.
`configs/magic_all.json` is a complete example with the three paper modules
enabled.

Recommended experiment naming:

```text
B0_baseline
B1_prompt_perturbation
B2_spatial_guidance
B3_cama
B4_your_module
```

## MDAP module

`configs/mdap.json` enables the training-free Material-aware Defect Appearance
Prior described in `改动点/模块1.md`. It automatically takes deterministic
reference image/mask pairs from:

```text
<dataset-root>/<category>/test/<defect-class>
<dataset-root>/<category>/ground_truth/<defect-class>
```

Run it with the same checkpoint as the baseline:

```powershell
python inference.py `
  --config configs/mdap.json `
  --model_ckpt_root <checkpoint-root> `
  --categories Plastic_Bomo `
  --dataset_type mvtec `
  --base_dir <dataset-root> `
  --output_name <result-root>
```

MDAP diagnostic priors, soft masks, and physical descriptors are stored in
each defect result directory under `mdap/`.

## DTCG module

`configs/dtcg.json` enables training-free directional texture continuity
guidance. It uses the same checkpoint as the baseline and requires no
additional training. Full commands and evaluation steps are documented in
`改动点/DTCG运行说明.md`.
