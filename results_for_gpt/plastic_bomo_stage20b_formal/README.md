# Stage20B — Plastic_Bomo Generation V2 正式结果

权威入口：`authoritative_formal_run.json`。
完整结果：`manual_C_20261010_032543/`。

正式状态：`GENERATOR_BASELINE_HEADROOM_NOT_CONFIRMED`。
当前冻结的 selector-free raw SD2 inpainting augmentation **未稳定优于 exposure-matched RealRepeat**。
不是“所有合成方法均无效”，也没有定位到 Black 下降的具体生成机制。

| Arm | mAP50-95 (%) | 相对 RR (pp) | Flash AP50-95 delta (pp) | Black AP50-95 delta (pp) |
|---|---:|---:|---:|---:|
| RR | 35.970193 | — | — | — |
| RawA | 29.709989 | -6.260204 | -4.855722 | -7.664686 |
| RawB | 33.081839 | -2.888354 | +1.043186 | -6.819894 |
| RawC | 37.348319 | +1.378126 | +3.919721 | -1.163469 |

Mean delta = **-2.590144 pp**，sample SD = **3.827887 pp**，strict wins = **1/3**。
逐类平均 delta：Flash **+0.035728 pp**；Black **-5.216016 pp**。
三个冻结 Gate 均失败：mean >= +0.50 pp、wins >= 2/3、每类 mean >= -2 pp。
`TDCRG_DEVELOPMENT_AUTHORIZED = false`；没有启动 Stage20C、TDCRG、BootstrapGuard 或 DeepPCB。

## 可复核的执行记录

- Flash / Black generator：各2000实际成功更新，冻结75/97个V2 TRAIN实例。
- A/B/C：同80 slots，各40 Flash+40 Black；generation bank seed为42/3407/2026。
- 同一对 generator checkpoints；全部240 raw输出使用，无 selector、reroll、质量筛选。
- 四个 detector：相同 seed42、初始参数、逐样本角色顺序和 augmentation seed。
- 各150 epochs、32400 draws、600尝试、592实际成功更新、8 AMP skips。
- 不为 AMP skips 补训，不选择 best checkpoint；primary为epoch150 last.pt的冻结native EMA表示。
- 四个checkpoint全冻结后，统一一次 final_eval：48 images、60 boxes、192 image forwards。
- 不把三个 generation banks 当作 detector 三seed确认；不与历史 Stage14–18 absolute mAP比较。

## 中断与人工恢复，不能省略

两次早期run（`formal_20261009_043250`、`formal_20261009_052723`）在Flash训练时OOM停止，没有完整结论。
`formal_20261009_060552`完成 generator / 三个raw banks / RR/A/B，C child以exit code -9停止。
用户报告可能误杀，但外部原因未被独立验证。

在用户明确要求下，新 sibling run复用SHA核验过的已完成资产，只将C从原始初始化seed42重新跑epoch1–150。
不是从第77轮续训，也没有覆盖原失败目录。
旧C观察到76epochs / 16416draws / 304attempts / 296successes / 8skips；可能是下界。
这份额外物理计算单独披露，未混入四组primary预算；**不能写成无中断的one-shot执行**。
具体来源、SHA、完整旧状态和日志在`manual_arm_C_recovery.json`及`manual_recovery/source_snapshot/`。

## 结论边界与归档

`physical_source_complete=false`：支持无KNOWN source-group泄漏，不证明未知物理关系不存在。
标签为dataset-defined supervision，不是所有微小缺陷均经人工核实的物理真值；矩形bbox mask为weak geometry，不是分割GT。
独立本地复核只读下载资产、训练日志和冻结协议，重算结果算术/Gate；不重新运行detector、不改变指标定义。
完整复核：`local_review_manual_C_20261010_032543_complete.json`，状态`LOCAL_FORMAL_REVIEW_PASS`，
核对1297个文件、788项一致性检查、无缺失资产、无失败检查。复核细节见`LOCAL_VERIFICATION.md`。

Git归档代码、JSON、报告、角色绑定和日志。多GB权重、原始/合成图片、copied train/final_eval inputs仅保留在本地和服务器；
它们的SHA由对应冻结manifest绑定。负结果完整保留，无自动挽救或下一阶段。
