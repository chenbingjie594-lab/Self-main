# MSDF-v3 两项训练级消融审计

## 实际入口与模块图

仓库没有 `run_train_msdf.py`。实际训练入口是 `run_train_rda.py`，它逐类别调用 `train_rda.py --enable_msdf --rda_mode none`。推理入口是 `inference.py`，使用 `pipeline_stable_diffusion_inpaint_magic.py` 并从 `msdf.pt` 严格加载适配器。

```text
real image / mask -> VAE latents + noisy inpaint input -> SD2 UNet
reference pixels -> pixel residual -> pixel_encoder ---------+
reference VAE latents -> local latent residual -> encoder ---+-> fusion
                                                            -> support heads
target coarse prior + reference support -> morphology alignment
aligned feature -> dynamic scale/time gate -> four projections
-> support mask -> tanh relative bound -> conditional CFG half of each UNet up-block
```

核心实现位于 `diffusers/pipelines/stable_diffusion/msdf_guidance.py`：`prepare()` 构建双分支特征、support 和动态 gate；`_align_reference()` 执行显式形态迁移；`_make_hook()` 在四个 up-block 中执行投影、support、gate、相对 RMS 的 tanh 限幅及 CFG 条件半分支注入。训练损失位于 `train_rda.py`，包含标准条件扩散损失以及 MSDF region、reconstruction、structure、background、scale、morphology support 六项。

## 消融旁路风险与修复

旧的诊断 `set_ablation()` 会在 latent encoder 已经计算后才把该分支置零，encoder 仍在 optimizer 中，因此不能作为 A1。现在 `training_ablation=no_latent` 在构造时冻结 encoder、optimizer 过滤 `requires_grad=False` 参数，并在 `prepare()` 中完全跳过 encoder。双模态 dropout 仅在两支都启用时执行，避免随机关闭 A1 的唯一 pixel 路径。

旧的 no-morph 路由虽跳过 `_align_reference()`，训练器仍会调用以参考形态为目标的 `morphology_support_loss()`，构成形态监督旁路。现在 `training_ablation=no_morph` 同时跳过显式 alignment，并将该专属 loss 从训练目标中移除；pixel support、latent semantic、fusion、gate、四层注入与 bounded residual 保持。

Checkpoint 写入 `training_ablation` 和 `training_contract`。加载使用 `strict=True`；旧 v3 checkpoint 缺少新字段时按历史 `morphology_alignment_enabled` 兼容推断，默认 Full。推理端不允许把训练时已删除的分支重新启用。

## Full checkpoint 审计

历史目录 `model/msdf_v3_sd2_2000_s42/Plastic_Bomo/{01_Flash_point,02_Big_black_spots}` 存在完整 pipeline、`msdf.pt` 和 `msdf_training.json`。两类记录均指向相同本地 SD2 Inpainting 基础模型，2000 successful target steps、batch 1、gradient accumulation 4、UNet LR 5e-6、MSDF LR 1e-4、seed 42、fp16、8-bit Adam、Full UNet，且 RDA/CARF 关闭。MSDF 固定参数为 hidden 64、context 1.75、max injection 0.75、max residual ratio 0.25、dropout 0.2、pixel support 0.5；损失权重与 `full_verified.json` 一致。

Flash/Black `msdf.pt` SHA256 分别为 `b4137f5986acfe08d3003d1fb1a093270a7a87d9a480f56200946121fa151f03` 和 `aba1b1a1596afde08c7e0ea46d4ce0d085fbe149d6714273cd7fefa0b48f3491`。四层 projection weight 范数均非零：Flash 0.5723/1.0011/0.7925/0.3041，Black 0.6203/0.9895/0.9969/0.5038。这排除了“训练后投影仍为 exact zero”的最直接 FrozenAdapter no-op 情形。

历史元数据没有训练文件 SHA manifest、donor 次序清单和初始化 RNG 状态，因此 A0 可作为冻结开发基准，但不能仅凭现存元数据证明严格因果可比。论文级结论应使用同一新入口重训 A0，或先补齐可验证的历史数据绑定。当前任务不启动该训练。

## 固定消融矩阵

| 机制/损失 | A0 Full | A1 no_latent | A2 no_morph |
|---|---:|---:|---:|
| Pixel residual | 开 | 开 | 开 |
| Latent semantic encoder | 开 | 关：不执行/无梯度/不入 optimizer | 开 |
| Reference morphology alignment | 开 | 开 | 关：不执行 |
| Morphology support loss | 开 | 开 | 关：权重有效值 0 |
| Fusion / four up-block injections | 开 | 开 | 开 |
| Dynamic gate / tanh bound / CFG | 开 | 开 | 开 |
| RDA / CARF | 关 | 关 | 关 |

## 验收范围

`tools/check_msdf_v3_ablation_routes.py` 是随机张量 CPU 单元测试，仅验证结构路由、optimizer、梯度、四层注入和 CFG，不冒充真实数据 smoke。机器可读结果写入 `checks/route_unit.json`。真实数据 1–2 step 验收由 `tools/smoke_msdf_v3_training_ablations.py` 执行，并必须在可访问 CUDA 驱动的环境中完成。

真实数据 smoke 已在 GPU 0、Black 训练集面积最大的真实 mask（036）上以 seed 42、fp16 GradScaler 跑两步，A1/A2 均 PASS。两项均有有效 MSDF 参数更新，四个 up-block 的条件注入均非零，无条件注入均为零，active 与真实 bypass 的条件预测不同且无条件预测逐元素相同，所有数值 finite，保存后 strict reload 逐张量一致。详细范数与 checkpoint SHA 见 `checks/no_latent/smoke.json` 和 `checks/no_morph/smoke.json`。
