# Stage21A — 证据综合与生成路线决策

正式状态：`GENERATOR_ROUTE_NOT_JUSTIFIED_CURRENTLY`。

当前不继续 Plastic_Bomo 的 generator-mechanism-first 路线。不实现 TDCRG，不立即换生成器。优先重新定义第一创新为**非 selector 的 training/data-integration 问题**；这是问题层面的选择，不是已验证的方法，也不是新实验授权。BootstrapGuard 保持独立第二创新。

本阶段训练、生成、检测训练、optimizer、final_eval forward、新模块实现、BootstrapGuard、DeepPCB 均为0。只读取 Git 中已有审计/指标 metadata；没有打开 final_eval 图像/标签，没有新模型运行，也没有修改 Stage14–20 的文件。

## 1. 三个问题的回答

A. 当前仍有充分依据继续 generator-mechanism-first 吗？**没有。** Legacy 子集的收益没有确认生成器机制；Stage17A 和新的 V2 Stage20B 各自在其协议内未确认 raw headroom。不能把历史子集、资格检查或一次 RawC 胜出升级为生成创新授权。

B. 第一创新应转向什么问题？优先问：**不改变生成器、不筛样、保留完整冻结合成集合，并固定总 real/synthetic exposure 时，训练过程中的使用时序能否降低不同生成 realization 的 utility 波动，同时避免类别 AP 受损？** 现有实验没有操纵时序，因此没有证明它是原因或有效解法。Stage21A 不指定 schedule、权重、阈值、训练配置，也不启动 Stage21B。

C. BootstrapGuard 如何独立？第一创新问“How should synthetic data be used during detector optimization?”；BootstrapGuard 问“Which synthetic samples should be retained?”。前者的干预在训练使用方式，后者在生成后的保留集合。不能用后者的历史收益解释前者的 raw baseline，也不能以近零 sample weighting 偷换为另一个 selector。

## 2. Stage20B 结论原样保留

权威运行：`manual_C_20261010_032543`；来源停止运行：`formal_20261009_060552`。

正式状态始终为 `GENERATOR_BASELINE_HEADROOM_NOT_CONFIRMED`；`TDCRG_DEVELOPMENT_AUTHORIZED=false`。

| Arm | mAP50-95 (%) | 相对 RR (pp) |
|---|---:|---:|
| RR | 35.9702 | +0.0000 |
| A | 29.7100 | -6.2602 |
| B | 33.0818 | -2.8884 |
| C | 37.3483 | +1.3781 |

平均 delta=-2.5901pp；样本 SD=3.8279pp；wins=1/3。冻结 gate A（均值≥+0.50pp）、B（严格胜出≥2/3）、C（每类均值≥−2.0pp）全部 FAIL。没有修改指标、阈值、seed 或成功定义。

RawC 只记为 `ONE_POSITIVE_GENERATION_REALIZATION`。其胜出是可核实事实，但“因此 baseline 稳定有效”不受支持。不能挑 C、增加 bank、调参或加入 selector 来挽救结论。

## 3. Manual C recovery 与计算边界

C 从 original public initialization、seed42 **epoch1 重新训练到150**，不是 epoch77 resume。两类生成器、240个 raw 输出和已完成 RR/A/B 复用；原停止运行及先前两次 Flash OOM 记录保留。

主要比较 RR/A/B/C 各150epochs、32400draws、600optimizer attempts、592successful updates、8AMP skips，init/order/augmentation metadata 匹配。旧 C 的额外物理计算单列：已观察76epochs、16416draws、304attempts、296successful updates、8AMP skips，可能只是下界；它不进入 matched primary exposure。早期 OOM 的完整物理计算总量没有重建，不能声称全部实际花费都相等。

用户报告可能误杀进程，退出码−9已记录；**外部 SIGKILL 原因没有独立证实**。旧运行文件未改。

## 4. 历史证据时间线

“确认/校准尝试”表示执行过预设验证问题，不表示结论通过。

| 阶段 | 保留的正式状态 | 证据类型 |
|---|---|---|
| Stage14A | `IMAGE_PROPERTY_BOTTLENECK_NOT_IDENTIFIED` | 筛查/资格/审计 |
| Stage14B | `UTILITY_SIGNAL_PRESENT_BUT_MECHANISM_UNRESOLVED` | 确认/校准尝试 |
| Stage15A | `CLASS_INTERACTION_DOMINATES_MORPHOLOGY_SIGNAL` | 确认/校准尝试 |
| Stage15B | `MORPHOLOGY_SIGNAL_PRESENT_DOSE_HYPOTHESIS_NOT_CONFIRMED` | 确认/校准尝试 |
| Stage16A | `OOF_TASK_TEACHER_MAPPING_UNAVAILABLE` | 筛查/资格/审计 |
| Stage16A-R0 | `ANNOTATION_PROVENANCE_INSUFFICIENT` | 筛查/资格/审计 |
| Stage17A | `SYNTHETIC_HEADROOM_NOT_REPRODUCIBLE` | 确认/校准尝试 |
| Stage18A | `NO_AUDITABLE_ALTERNATIVE_GENERATOR_BASE` | 筛查/资格/审计 |
| Stage20A | `PLASTIC_BOMO_GENERATION_V2_PROTOCOL_READY` | 筛查/资格/审计 |
| Stage20B-P0 | `STAGE20B_STRUCTURAL_PREFLIGHT_PASS` | 筛查/资格/审计 |
| Stage20B-R0 | `STAGE20B_RUNTIME_QUALIFICATION_PASS` | 筛查/资格/审计 |
| Stage20B | `GENERATOR_BASELINE_HEADROOM_NOT_CONFIRMED` | 确认/校准尝试 |

完整 question/protocol/result/support/non-support/validation/selector 字段及来源见 [generator_evidence_timeline.json](generator_evidence_timeline.json)。

- Stage14A：已纠正为 `IMAGE_PROPERTY_BOTTLENECK_NOT_IDENTIFIED`。当时 crop-KID/real-nearest LPIPS 缺失，high/low 是 morphology-matched 子集；+2.286/−3.935pp 只作诊断。
- Stage14B：High−Random +1.043±1.387pp，2/3wins；Flash +3.406pp、Black −1.320pp。High 的 set-KID×1000=44.722，Random=40.011；nearest-real LPIPS≈0.20905/0.20935。不是普遍 pixel/perceptual fidelity 改善。KID 没有用于 per-sample 排序。
- Stage15A：Flash条件效应在Random-Black下+2.387pp、在High-Black下−0.110pp；interaction−2.497pp。可支持条件子集效应，但不能锁定 morphology 特异机制。没有独立验证历史全部 optimizer/init 原始字节。
- Stage15B：M50−M00 +0.674pp；M50−mean(M10,M01)−1.095pp；M11−M50 +0.368pp；C/D gates失败。High/Random overlap 导致 nominal/physical dose不一致，浓缩指标不能证明 utility 损害。
- Stage16A/16A-R0：mapping/provenance 入口停止，**不是 task guidance 无效**的实验。原 Stage16A server status blob在当前提交中缺失：引用 R0 冻结 protocol 的明确 stop declaration，不能把 LOCAL snapshot 冒充服务器原结果。R0 accepted groups两类均0，后续GPU gates未运行；也不能由此断言标签一定错。
- Stage17A：权威 train-only rebuild 的三 bank mean−5.0824pp、0/3wins，存储状态为 `SYNTHETIC_HEADROOM_NOT_REPRODUCIBLE`。Flash/Black−8.4172/−1.7475pp；成功更新range2。先前有重叠的 rebuild 并非最终权威模型。
- Stage18A：SERVER_EXISTING_EVIDENCE 中没有合格替代，非全球不存在。AnomalyDiffusion/DefectFill/MSDF缺少完整合格链；DWBG761是重组/过滤派生池，不是761个原始MSDF输出。历史 Random是过滤池内随机，不是 raw utility。
- Stage20A/P0/R0：分别证明新协议、结构绑定、已测运行路径的资格；不是收益验证。Stage20A曾行政性读取旧eval的46图/46label以重分组，没有forward；R0实际生成过4张公开base探针，**非formal synthetic**，不能抹去物理采样事实。

Stage14–15是旧标签/来源边界下的探索性子集链；后续使用了先前validation结果形成问题。不能把这条连续探索链包装为独立 generator-mechanism confirmation。Stage20A重分组后 V2 与历史 split/protocol 不同，**没有跨协议绝对AP比较、平均效应合并或历史阈值转移**。以前报告的下一步建议不等于当前授权。

## 5. 三层证据

SUPPORTED：V2未示稳定raw正收益；三生成realizations有大幅观察波动；RawC一次胜出但未确认baseline；Black贡献主要平均损失；selector utility与raw utility是独立命题。

DIAGNOSTIC_ONLY：morphology alignment信号、class-specific子集interaction、RawC正结果的推广/机制解释、sample composition与utility可能有关。

NOT_SUPPORTED：visual fidelity必然带来utility、morphology最大化保证utility、raw SD2有稳定正headroom、立即TDCRG是合格确认步骤、所有synthetic都无用。

上述分类区分“数字事实”与“机制/泛化推断”，见 [supported_findings.json](supported_findings.json)。未支持不等于普遍被证伪。

## 6. RNG instability 与类别非对称

A/B/C共用 class checkpoints、80slots、normal/prompt/geometry、scheduler/采样参数以及 detector init/seed/order/augmentation/exposure。成对metadata再次核对；设计上只改变 generation RNG，生成像素/集合内容随之变化。

准确表述：**downstream performance varied by7.64pp across the three generation-RNG realizations under the frozen protocol.** 样本SD3.8279pp，range7.6383pp。这不是“RNG导致7.64pp的总体因果效应”，更不是三个detector seeds；三个bank、单个split/seed不足以估计总体variance或找到导致波动的图像机制。

类别状态为 `CLASS_ASYMMETRIC_UTILITY_FAILURE`。Flashmean+0.0357pp，Blackmean−5.2160pp，Black在三个bank均负。按现有两类macro AP恒等式：Flash贡献+0.0179pp、Black贡献−2.6080pp，总计−2.5901pp；这只是原指标的代数拆分，不是新加权质量分数。

只能说 **downstream degradation is concentrated in Black AP under the current protocol**。没有直接证据证明Black生成质量坏、morphology错或标注错。Flash均值近0也不证明无害（A的Flash下降）。Stage17A反而Flash下降更多，不能据此制定普遍Black-only惩罚/排除规则。

## 7. 三条路线的正式选择

| 选项 | 当前决定 | 理由与主要风险 |
|---|---|---|
| A：直接SD2+TDCRG | `OPTION_A_CURRENTLY_NOT_AUTHORIZED` | 三个前置gate全失败；高成本，事后救结果风险高，缺少合格机制起点 |
| B：换published generator | `OPTION_B_NOT_RECOMMENDED_AS_IMMEDIATE_NEXT_STEP` | 资产审计不是独立scientific-fit理由；不能逐个换到正结果，也不能由资产不齐断言family不适合 |
| C：非selector训练/数据整合 | 优先**问题重定义**，不授权实现 | 固定生成器和完整数据，可更清楚区分优化参与方式；仍须警惕软selector、exposure混杂和事后调参 |

三个选项均按 scientific justification、post-hoc risk、cost、causal interpretation、BootstrapGuard overlap、thesis coherence独立评价，详见 [innovation_route_options.json](innovation_route_options.json)。没有按“必须获得正结果”选择；没有新外部文献适配性比较或generator family推荐。

## 8. 第一创新候选仅作概念分析

优先是synthetic-real exposure **使用时序**的问题。其他候选依次作为待审假设：uncertainty-aware weighting、class-adaptive optimization、consistency regularization、distribution-aware integration。

它们均**没有有效性证据**。uncertainty weighting可能变为soft selector；class rule可能事后针对Black调参；consistency可能强化错/弱标签且改变compute；distribution integration依赖尚未证实的表征/分布假设且容易重叠BG。不存在现成的“复杂模块方案”。完整风险和非selector边界见 [innovation1_reframing_candidates.json](innovation1_reframing_candidates.json)。

未来即使考虑实验，也必须另行授权、预先固定非selector干预及预算、保留全集合，并说明与已有final_eval接触的隔离方案。当前这些最终指标已经用于选择研究问题，不能再拿同一final_eval调参后声称独立确认。Stage21A没有给出新实验协议或启动命令。

## 9. BootstrapGuard贡献边界

历史报告中的BG−Random平均+1.6172pp属于其旧派生池/协议下的conditional selector结果；不搬到V2，不用于否定Stage20B。Stage18A关于旧池lineage/validation未知的限制继续有效。当前没有独立V2 BG confirmation，也没有运行BG。

贡献1保留完整冻结集合、干预训练使用方式；贡献2干预生成后保留集合。不能把continuous weights换名当成独立贡献，也不能只报告组合pipeline收益而掩盖各自作用/预算。

## 10. 可复核文件与停止点

[source_evidence_index.json](source_evidence_index.json) 给出每份metadata的固定commit、路径、SHA256及GitHub链接；删除的历史worktree文件只读Git，不恢复、不改写。[evidence_verification.json](evidence_verification.json) 记录独立指标代数、gate、成对slot/checkpoint、计算预算与恢复边界检查。

所有来源锁定 `26d19a7906b4339c059d9897d9a29c4cbbee2762`。本地复核仅CPU/标准库，没有重算GPU/AP预测。此报告是既有final指标的路线综合，不是新确认实验、新方法有效性或论文新颖性证明。

停止于 Stage21A 文件与GitHub提交。**不训练、不生成、不实现TDCRG/新generator/curriculum/weighting/regularization，不运行BootstrapGuard/DeepPCB，不启动Stage21B。**
