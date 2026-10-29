# MSDF-v3 training ablations

状态更新时间：2026-10-29 UTC。

| 项目 | 状态 | 证据/限制 |
|---|---|---|
| A0 Full MSDF-v3 | 历史 checkpoint 可加载；严格可比性未证明 | 缺少训练前数据文件 SHA manifest、donor 顺序与初始 RNG ledger；未重训 |
| A1 w/o Latent Semantic | **正式训练完成** | Flash/Black 各 2000 successful steps；`checks/a1_training_audit.json` 为 PASS |
| A2 w/o Morphology Alignment | **正式训练完成** | Flash/Black 各 2000 successful steps；`checks/a2_training_audit.json` 为 PASS |
| A1/A2 route unit test | 完成 | 随机张量只用于结构路由单元测试 |
| A1/A2 real-data smoke | 完成 | 真实 Black 样本、真实 SD2 UNet、2 steps；不能作为正式训练结果 |
| A0/A1/A2 固定可视化生成 | **完成** | 每类 4 个预定槽位，覆盖 A/B/C；24 张正式 PNG，无重抽 |
| 三组可视化 | **完成，真实配对 PASS** | `visual_evaluation_20261029_v2/`；实际源文件和推理参数均核验 |
| KID/FID、SOTA、检测 | 未执行 | 本阶段明确禁止 |

关键文件：

- `audit.md`：代码、checkpoint、可比性和一次性 A0 补齐方案。
- `checks/a1_training_audit.json`：A1 steps、数据数量、跳步、严格加载、有限性与 SHA256。
- `checks/a2_training_audit.json`：A2 steps、数据数量、跳步、消融契约、严格加载、有限性与 SHA256。
- `training_no_morph_complete.json`：A2 的公开完成摘要。
- `checks/route_unit.json`：CPU 路由测试。
- `checks/real_data_smoke_summary.json`：真实数据两步 smoke 汇总。
- `scripts/render_msdf_ablation_grid.py`：严格配对审计及完整图、上下文 ROI、统一色标热图。
- `commands.md`：训练与可视化命令。
- `visual_evaluation_20261029_v2/VISUAL_EVALUATION_REPORT.md`：正式固定样本可视化结论、失败样本与限制。

临时 smoke checkpoint 不能用于正式生成或性能结论。正式模型权重与完整日志只保留在服务器。
