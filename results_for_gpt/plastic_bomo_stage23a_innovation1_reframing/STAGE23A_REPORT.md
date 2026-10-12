# Stage23A — Innovation-1 Research Reframing

正式状态：`INNOVATION1_DIRECTION_SELECTED_FOR_PROTOCOL_DESIGN`。

选择 **C：generation robustness / controllability** 作为下一步可以设计协议的科研问题。不是选择一个新方法，不代表已发现 Plastic_Bomo 生成控制失败，也不授权实现、训练或生成。A 暂缓于任务适配证据不足；B 暂缓于没有可识别的整合机制。没有提出 attention、loss、adapter、具体 generator family 或启动命令。

第一创新的问题收窄为：**少样本局部缺陷生成，能否在同一可信条件、不同 RNG 下保持位置/范围及有效外观的条件遵从，同时保留未编辑背景和允许的外观多样性？** 科研动机来自任务和已有研究，不是由 negative mAP 反推“必须有这个机制”。现阶段只是选择问题；如果后续没有可测、区别于已有工作的缺口，应回到 `NEW_INNOVATION1_REQUIRES_LITERATURE_GROUNDED_REDESIGN`，不强行设计模块。

本阶段 training、generation、detector forward/backward、optimizer step、final_eval 内容读取/forward、新模块实现、BootstrapGuard、DeepPCB 均为 **0**。仅阅读既有 metadata、报告与一手文献，并进行 CPU/标准库文件一致性和指标代数检查。没有解码图像或加载 checkpoint。提交 GitHub 后停止，不启动 Stage23B。

## 1. 正式关闭已有路线

| 路线 | 当前状态 | 依据与边界 |
|---|---|---|
| CURRENT_SD2_GENERATOR_MECHANISM_ROUTE | CLOSED | Stage17A/20B 未确认稳定 raw headroom；旧子集收益不能授权现有 SD2 机制补丁或 TDCRG |
| CURRENT_SYNTHETIC_TIMING_ROUTE | CLOSED | Stage21B 固定 late schedule 在 3/3 bank 上不胜 Uniform；不扫描新 schedule 救援 |
| CURRENT_GRADIENT_MECHANISM_ROUTE | NOT_SUPPORTED | Stage22A 没有唯一 loss/norm/direction/feature 机制；不做 weighting、PCGrad、GradNorm 或 alignment module |

这些状态仅针对当前项目已有协议和证据，不是“所有 synthetic 无用”“所有 curriculum/gradient 方法失败”。历史数字与状态原样保留；只关闭其未来路线授权，不改写历史报告。C 是新的问题层面定位，**不是把旧 SD2/Stage16 task-guidance 改名重新开放**。历史 SD2 以后即使被另行批准作为参考，也不等于机制开发获得授权。完整记录见 [closed_route_registry.json](closed_route_registry.json)。

## 2. 证据复盘：事实不等于机制

来源统一锁定 commit `0445239155b9fc61f862e7523cd8d1aafeced53e`。[source_evidence_index.json](source_evidence_index.json) 给出 18 份来源的路径、精确 Git 字节 SHA256 和固定版本 GitHub 链接模板。工作区 S14B 只有 LF/CRLF 差异，已分别记录；不修改它。

| 阶段 | 可保留的事实 | 不能升级的解释 |
|---|---|---|
| Stage14A | corrected status 为 IMAGE_PROPERTY_BOTTLENECK_NOT_IDENTIFIED；morphology-matched High/Low 有诊断差异 | 当时 KID/nearest-LPIPS 缺失，不能称 broad fidelity bottleneck |
| Stage14B | High−Random +1.043±1.387pp，2/3 wins；Flash +3.406pp、Black −1.320pp | 非一致 pixel/perceptual 改善，不证明 fidelity 导致 utility |
| Stage15A | class-specific frozen subset 的条件效应与 material interaction | 不是 morphology-only 因果分离，不支持普遍 class-specific rescue |
| Stage15B | dose gates C/D 未通过，nominal dose 与真实 High membership 不一致 | 不能由浓缩描述量推出 dose 机制或继续 morphology selector |
| Stage16A/R0 | mapping/provenance 入口停止，后续机制测试没运行 | 不是 task-guidance 无效，也不能由入口失败断言 bbox 一定错误 |
| Stage17A | train-only 权威 rebuild mean −5.0824pp，0/3 wins | 旧协议证据不能与 V2 合并，不使用曾确认 overlap 的旧 rebuild 代替权威模型 |
| Stage18A | 当时检查的资产无可审计替代 generator base | 不代表全世界没有更合适的 generator family |
| Stage20A/P0/R0 | 新分组协议和实际受测路径的资格记录 | 不代表生成 utility；旧行政分组访问与 R0 探针采样不能被抹去 |
| Stage20B | raw SD2 V2 的稳定正 headroom 未确认；C 一次胜 RR | 不代表可以挑 C、增加 bank、解释 Black generator quality 或开发 TDCRG |
| Stage21B A1 | 一个固定 late schedule 的 conditional internal_dev premise 未支持 | 不代表所有 curriculum 无效，也不是独立 pipeline holdout |
| Stage22A | frozen-initialization 诊断未识别唯一机制 | 不是训练轨迹因果实验，也不授权梯度或特征补丁 |

Stage14–15 来自旧派生池/子集和历史标签/来源边界。历史 Random 是池内随机，不是 raw 未过滤输出。V2 的 split、数据边界和比较对象不同，**不跨协议比较绝对 AP、不合并效应、不转移历史阈值**。Stage16 入口情况在此引用 S21A 的历史综合记录，不冒充缺失的原 server status。

### 2.1 Stage20B：三次生成实现，不是三次 detector seed

| Arm | mAP50-95 (%) | 相对 RealRepeat (pp) |
|---|---:|---:|
| RR | 35.9702 | 0.0000 |
| A | 29.7100 | −6.2602 |
| B | 33.0818 | −2.8884 |
| C | 37.3483 | +1.3781 |

Mean −2.5901pp；样本 SD 3.8279pp；range 7.6383pp；wins 1/3，预声明 headroom gates 全部 FAIL。A/B/C 绑定相同 class checkpoints、conditions、normal roles、geometry、采样参数与 detector seed42/init/order/exposure，只改变 generation RNG。支持**观察到 realization 不稳定**和**一次正结果**，不提供总体方差估计或像素机制。C 的中断重跑及额外物理计算限制继续保留，不能称整个历史花费完全相同。

### 2.2 Stage21B：较窄的 spread 伴随较低的 utility

Curriculum−Uniform：A −2.5002、B −0.2377、C −4.4869pp；mean −2.4082pp，0/3 wins。Bank sample SD 从 3.0491 降至 0.9395pp；Flash/Black mean delta 为 −1.0856/−3.7309pp。只能说**这三 bank 的样本 spread 较小，但平均效用更差**，不能把变得一致地较差称为成功的稳健性。

此结果为 `DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR`：历史 generator 输入含 33 个 dev-group annotations，不是 generator+detector 的独立 holdout。每臂同 scheduled attempts，但成功更新 442–443，严格实际更新数不相等。不是 Stage20B RR headroom 的新确认，也不是 official final_eval 结果。

### 2.3 Stage22A：RNG 改变诊断信号，但未识别原因

292 个 frozen-init forward/backward 记录覆盖 109 real internal_train 图和每 bank 61 synthetic slots。所有 synthetic 的 global/same-class negative-cosine fraction 为0；全部记录 minimum cosine 分别约 0.090134/0.113735。仅支持**所测初始化、四个代表参数张量下未发现负梯度冲突**，不否定训练后可能出现的冲突。

Matched-slot loss range median：Flash/Black 4.491287/4.504227；selected-gradient norm range 139.077904/113.779584；same-class cosine range 0.024765/0.028550。支持同条件 RNG 改变 loss/gradient signals，但非零 range 不等于显著性或 utility 因果证据。

Feature 只有 class×bank×layer 汇总，没有 matched per-slot feature triplets 或原始 feature vectors。因此“RNG 改变 feature”只能保留为**bank-level 描述差异**，不能与配对 loss/gradient 证据等强。实/合 covariance rank 的样本数上界不同（Flash 62/31、Black 45/28），不能据此宣称 synthetic diversity collapse。Real 多目标/synthetic 单目标混杂、classification 主导的 total loss、所选 backbone 梯度主导也仍在。没有唯一机制候选满足全部预声明要求。

七项正式 finding 及具体限制见 [supported_findings_stage14_22.json](supported_findings_stage14_22.json)。不支持的机制逐项列于 [unsupported_mechanism_claims.json](unsupported_mechanism_claims.json)：fidelity/morphology/conflict/norm/feature 导致 utility、Black quality cause、普遍有效 late curriculum、TDCRG 与 gradient weighting 均不能继续宣称。

## 3. 文献提供独立动机，也限定新颖性

本次是截至 2026-10-12 的定向一手文献审查，不是穷尽式综述或 novelty certificate。论文的其他数据集收益不转移到 Plastic_Bomo。每篇的查阅范围、支持与不支持内容见 [literature_evidence.json](literature_evidence.json)。

- 少样本 anomaly 的 appearance/location 分离、生成区域与 mask 对齐已有研究。这说明条件控制是独立问题，也说明“能控制位置/外观”本身不新。[AnomalyDiffusion](https://arxiv.org/abs/2312.05767v2)。
- 工业 local inpainting 和背景整合已有任务适配先例，但 published pipelines 有的还带选择/过滤；不能把整体收益当 raw generator 贡献，也不能复制其 selector 冒充第一创新。[DefectFill](https://arxiv.org/abs/2503.13985v1)，[GAA](https://arxiv.org/abs/2507.09619v1)。
- 小尺度生成及背景融合已是近期研究主题；这是 A/C 的独立任务理由，但不是本项目 latent compression 造成 AP 下降的证据。本文不照搬其尺度阈值或模块。[UniScale](https://arxiv.org/abs/2608.07864v2)。
- 通用空间条件控制及显式条件一致性优化都有 prior art。因此 C 必须先直接评价 condition adherence，不能直接由 mAP 推导控制失败，也不能把通用 consistency/reward 改名当新方法。[ControlNet](https://openaccess.thecvf.com/content/ICCV2023/html/Zhang_Adding_Conditional_Control_to_Text-to-Image_Diffusion_Models_ICCV_2023_paper.html)，[ControlNet++](https://arxiv.org/abs/2404.07987v4)。
- 局部编辑与保留未编辑背景有独立研究动机；但自然图像编辑能力不是弱缺陷生成有效性证明。[Blended Diffusion](https://openaccess.thecvf.com/content/CVPR2022/html/Avrahami_Blended_Diffusion_for_Text-Driven_Editing_of_Natural_Images_CVPR_2022_paper.html)。
- Domain-specific normalization 为 B 提供独立研究先例，但 UDA 与有标签 real/synthetic joint training 的问题设置不同；这不诊断本项目 BN，也不推荐移植模块。[DSBN](https://arxiv.org/abs/1906.03950v1)。

以上研究说明可重新立题，不说明现成模块值得立即训练。缺乏适配证据时，宁可保留未定项，也不顺序试模型直到正结果。

## 4. 三个方向的十项评估

下表是定性科研判断，不生成综合分数。完整逐项内容分别见 [A](innovation1_option_A_generator_redesign.json)、[B](innovation1_option_B_training_integration.json)、[C](innovation1_option_C_generation_robustness.json)。复杂度/成本为相对估计，没有假造 GPU 时长或价格。

| 维度 | A：generator redesign | B：training integration | C：generation robustness |
|---|---|---|---|
| scientific motivation | 局部编辑、空间粒度、背景保留、few-shot task fit | 域统计/监督语义整合的独立文献问题 | 条件遵从、局部保留、跨 RNG 与允许多样性 |
| existing evidence | raw SD2 未稳胜；替代资产未合格 | 诊断差异存在，late schedule 不支持 | 同条件 bank 与配对信号变化，未直接测控制 |
| missing premise | task-fit 测量、有限候选、合格资产链 | 唯一可干预机制及域/监督混杂分离 | 可信 condition、独立 output 测量、具体 prior-art gap |
| novelty potential | 条件性；换 backbone 不算创新 | 当前低至条件性；现成 integration 已有 | 问题有潜力，方法与 novelty 未建立 |
| implementation complexity | 高 | 中高 | 协议澄清较低；实际方法未知 |
| experimental cost | 高，可能重新训练全部链 | 高，需 detector 与 bank 对照 | 前提审计相对较低；现在不运行 |
| post-hoc tuning risk | 高：model shopping | 高：补丁、class rule、weight tuning | 中高：倒推指标、挑 seed、collapse |
| BootstrapGuard overlap | raw-only 时低；论文过滤需剥离 | 容易变软 selector，风险高 | 全 raw output 的生成侧研究可明确分开 |
| thesis compatibility | 高，直接 generation-side | 中高，augmentation 使用问题 | 高，generation-side + 独立 post-generation |
| Plastic_Bomo support | 局部弱 bbox 可设计；family winner 未证 | 可审计监督接口；机制未识别 | 可设计弱位置/范围/背景问题；精确形状尚不支持 |

### A：不选某个 family，不因 SD2 fail 进行模型购物

更适配的**能力画像**应是：局部条件编辑、目标尺寸可表示、保留背景、少样本可复用先验、弱 bbox 条件语义明确。它不是某个架构已经更优的结论；已有 latent diffusion 也能针对任务适配，不能直接判定必须 pixel diffusion、GAN 或更大模型。必须先有具体 task-fit gap 与资产/许可/隔离能力，再预先决定有限比较；当前暂缓。

### B：独立文献问题存在，当前没有可实现的机制

联合训练可以研究域统计和监督语义，但 Stage22A 没有识别 BN 或其他 integration cause。不得将 gradient weighting、new curriculum、uncertainty/class-specific rescue 作为替代方向，也不得通过 ranking/drop/near-zero weight 改变累计样本参与而重叠 BootstrapGuard。当前暂缓，不指定 normalization、regularizer 或 loss。

### C：选择问题，而不是宣布创新或重启旧方法

它在三者中最适合先设计**生成器自身的直接评价问题**，也最容易与 post-generation retention 分开；选择依据不是“降低 variability 一定提升 mAP”。控制研究已经存在，必须进一步明确：在本项目可信弱条件、稀缺 source groups、真实局部缺陷与背景条件下，哪项控制能力缺失、能如何独立测量、与现有方法差在哪里。

## 5. C 的必要前提与因果比较边界

以下是后续协议设计必须回答的问题，**不是已冻结实验配置或执行授权**：

1. **条件真值。** 当前生成协议采用 hard bbox rectangle，明确声明不是 human segmentation。可讨论 class、位置、范围、normal/background；精确 defect silhouette、boundary IoU、指定 appearance parameter 不能由 rectangle 自动获得真值。没有授权补标或修改现有 labels/splits。
2. **独立输出测量。** 复制 input bbox 到 synthetic label 只证明标签流程，不能证明生成的缺陷真的出现在该范围。输出定位/外观有效性评价需有独立可靠依据；若考虑 learned evaluator，必须先解决其标注、source isolation 和测量有效性，不能绕过 Stage16 问题或直接调用未经资格确认的 teacher。
3. **不同的稳健性。** 同一个 RNG 精确复现是 reproducibility；不同 RNG 保持指定条件是 controllability robustness；保留允许的纹理/外观变化是 diversity。固定 seed 或输出都变一样不是创新。降低不希望的条件违背，不等于最大限度降低像素多样性。
4. **全集 raw 输出。** 条件、source roles、paired RNG、输出数与评价规则先固定。失败/弱缺陷必须计入，不 reroll、不 best-of-N、不挑 C、不加入 Top-K 或 weighted quality score。评价不能由已看过 bank utility 排序来定义。
5. **少样本与任务适配。** 用 unique source groups 而不是派生 crop 总数定义 few-shot；尺寸/对比度需直接统计，不能从 AP 损害推出 small/low-contrast cause。两类均报告，不能用 Black rescue 规则选择问题。
6. **比较逻辑。** 未来若直接前提审计支持可测控制缺口，才讨论一个有独立动机的最小干预及相同条件/RNG/输出预算的参考对照。现在没有选方法或新 baseline checkpoint；历史 SD2 作为参考也须重新满足新问题的资格，不能自动重开其机制路线。
7. **效能与隔离。** 控制指标改善不等于 downstream causal gain。任何后续 utility 确认必须独立冻结；Stage20B 的 final_eval 已用于问题选择，不能继续拿同一 final_eval 调参后称独立确认。Stage21B dev 也不是独立 generator holdout。当前未制定新 split 或生成新数据。

如果没有可信且独立可测的条件 outcome，或只是复述已有控制/小尺度工作，C 不进入方法设计。只有找到该缺口、完成 prior-art 差异和可行 baseline/对照设计，才有后续讨论价值。

## 6. 与 BootstrapGuard 的论文贡献边界

潜在贡献1问“raw generator 在 condition 下如何可靠地产出”；贡献2问“产出后哪些样本保留”。前者必须对全 raw outputs 评价，后者干预 retained set。它们可以形成论文结构，但**独立贡献、组合收益和 V2 有效性均未验证**。只报告组合 pipeline 赢，不证明各自作用；未来如有另行授权的组合研究，应区分 generator-only、selector-only、combined，并披露生成/保留/训练预算。

本阶段没有操作 BootstrapGuard，也没有给它新实验授权；历史 selector utility 不用于推翻 raw baseline。质量过滤、reroll 或近零权重不能换名为第一创新。详见 [bootstrapguard_boundary.json](bootstrapguard_boundary.json)。

## 7. 决定与停止点

[innovation1_decision.json](innovation1_decision.json)：C 获得**后续 protocol design only**；A/B 暂缓；方法、模块、family、训练、生成、detector 运行、官方评估与 Stage23B 执行授权全部为 false。[stage23a_status.json](stage23a_status.json) 记录本阶段零执行计数。

这个决定允许下一次继续设计问题的可测性、literature gap 与隔离边界，不允许跳过前提直接造模块。Stage23A 至此归档提交，**停止，不自动继续**。

## 8. 文件复核

CPU/标准库复核通过：10 项必需输出、3 个选项各 10 项评估、7 个 finding、8 个 primary literature 记录、18 份固定版本来源和 source/literature 引用均已检查。Stage20B mean/SD/range/wins、Stage21B mean/spread、Stage22A 配对 range 的六位小数及各 bank/class 的0 conflict 均与来源一致。JSON 不含 NaN/Infinity。全部执行授权与零计数合同通过。

这不是 GPU 复现实验、文献穷尽审查或新颖性证明。核查记录与本目录文档 SHA256 保存在 [review_integrity_audit.json](review_integrity_audit.json)。提交只包含本 Stage23A 目录；工作区已有的历史删除和其他改动不归入本次提交。
