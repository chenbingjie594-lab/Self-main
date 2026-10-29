# MSDF-v3 training ablations

本目录只包含 A1 w/o Latent Semantic 与 A2 w/o Reference Morphology Alignment 的实现审计、配置、训练前验收和命令。没有执行正式训练、KID/FID 或下游检测。

- `audit.md`：实际代码路径、旁路风险、Full checkpoint 审计与消融矩阵。
- `configs/`：已恢复的 Full 配置和两项唯一变化。
- `checks/route_unit.json`：CPU 结构路由单元测试。
- `checks/real_data_smoke_summary.json`：真实 Black 样本、真实 SD2 UNet 的两步 smoke 汇总。
- `checks/{no_latent,no_morph}/smoke.json`：loss、梯度/更新、四层注入、CFG、bypass 差异和 strict reload 证据。
- `commands.md`：正式训练和可视化命令。

临时 smoke checkpoint 只用于加载验收，不能用于正式生成或性能结论。
