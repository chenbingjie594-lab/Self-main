# Stage24A — Literature-Grounded Innovation-1 Gap Search

## 最终决策

`NO_CLEAR_INNOVATION1_GAP_IDENTIFIED`。

本次对 20 项方法做了一手论文的定向方法/实验章节核查，包含 AnomalyDiffusion、DefectFill、GAA、UniScale。没有足够证据锁定一个可作为 Innovation-1 的唯一主问题。保留两个 `PLAUSIBLE_GAP` 研究问题，但 **primary candidate = null**，不是把第一名自动转成下一阶段。

这不表示“工业缺陷生成已经没有问题”，而表示在本次检索与当前项目证据下，尚未建立一个区别于最接近工作、可独立验证且不重启已停止路线的新缺口。不能由 Stage20B 的负 AP 反推文献 novelty。

`METHOD_DESIGN_AUTHORIZED = false`、`TRAINING_AUTHORIZED = false`、`GENERATION_AUTHORIZED = false`。本阶段训练、生成、detector forward、backward、optimizer step、新模型模块、BootstrapGuard、DeepPCB、official final_eval 均为 0。提交 GitHub 后停止。

## 检索边界与证据规则

基于 commit `69cb5db28107acb0e33748ef4c6813aa01e03a0b`，检索时间为 2026-10-12，重点范围 2023–2026。只以论文原文、官方 proceedings、作者 arXiv 为结论依据。检索词、排除项和局限见 [literature_search_scope.json](literature_search_scope.json)。这是定向检索，不是所有指定期刊/会议的穷尽系统综述。

论文列表、版本、发表状态及章节定位见 [primary_source_registry.json](primary_source_registry.json)。20 行完整字段矩阵见 [method_comparison_matrix.json](method_comparison_matrix.json)：P01–P04 为四项必查方法，P05–P10 为 AnoGen、SeaS、DualAnoDiff、GroundingAnomaly、O2MAG、TF-IDG，P11–P20 为 DefectDiffu、DefFiller、DFMGAN、Controllable Industrial Synthesis、SynSur、Assigned MURA、AnomalyAny、MAGIC、DefectSimu、Blended Latent Diffusion。

特别区分：

- 真实训练标注、生成时的条件、独立评估真值，是三个不同角色。
- one-third 是类别样本比例，不能写成 1-shot/3-shot；论文的实验用量也不等于理论最少需求。
- 像素 AD/分割 AP 不等于 bbox detector AP50–95；IC-LPIPS 多样性不等于跨 generation seed 的控制失败率。
- 代码链接表示论文公布，不代表已下载、资产完整或复现成功。
- `NOT_REPORTED` 是所查章节没有明确报告；`NOT_VERIFIED` 是未核实。两者都不证明“不存在”。

GAA、GroundingAnomaly、SynSur 等按作者预印本登记，不擅自赋予会议录用状态。UniScale 的 ECCV 2026 状态来自作者 arXiv 说明，独立 proceedings 未核实。MAGIC 为 CVPR 2026 Findings；MURA 和 DefectSimu 为 workshop，不混写为主会。

## 六类问题的归因结论

| 问题 | 本次分级 | 最接近证据与未闭合边界 |
|---|---|---|
| A 仅 bbox / 无 pixel mask | PLAUSIBLE_GAP | bbox 生成已有先例；需要严格核对概念学习阶段是否用 real masks。1–3 shot 下的弱标注低占比概念学习仍是问题假设，不是已成立的新缺口。 |
| B 极小、弱对比、透明材料 | NOT_VERIFIED | tiny 和低对比已有专门研究；透明/半透明物理条件没有在本次比较中核实，不能从类别名或材料名推出 novelty。 |
| C 严格背景保持 | WEAK_OR_ALREADY_COVERED | 局部 latent blending、背景恢复与像素复制已有先例；“只复制背景”本身不足以成为新方法问题。 |
| D 极少样本适配 | WEAK_OR_ALREADY_COVERED | full tuning、LoRA、embedding、training-free reference 等路线均已有；选择参数化形式不是 gap。 |
| E 尺度、细长和各向异性结构 | WEAK_OR_ALREADY_COVERED | latent 下采样遗漏与局部放大已有明确分析和处理；尚未核实可区别于先例的各向异性新问题。 |
| F 合成标注可靠性 | PLAUSIBLE_GAP | 已有内容驱动 mask，也有 thin-scratch 失败与 condition-mask 回退；不能再笼统声称现有方法都复制条件。 |

### A：bbox 推理不能替代“全流程 bbox-only”证据

AnoGen 以少量异常学习 embedding，支持 bbox 生成；主设置的概念学习用 pixel masks，但论文也明确提供 **non-mask training ablation**，并讨论不使用 mask 时概念更易偏向整个物体。它是必须比较的既有弱监督 baseline，不能遗漏后再宣布“没人做”。[AnoGen §4、§6.1、§7](https://arxiv.org/html/2505.09263v1)

MURA 已进行带位置标注的低对比局部生成，但使用数百训练 patch，并非严格 1–3 shot；AnomalyAny 可在不训练异常概念的情况下用正常图与文字生成。它们削弱了广义“无 pixel mask 生成”的新颖性，但并不自动证明特定 tiny anomaly 的弱标注概念学习已解决。[MURA 方法与实验](https://openaccess.thecvf.com/content/CVPR2023W/VISION/papers/Liu_Assigned_MURA_Defect_Generation_Based_on_Diffusion_Model_CVPRW_2023_paper.pdf)，[AnomalyAny](https://arxiv.org/html/2406.01078v3)

DefFiller 的“无需额外标注”不能理解为真实训练数据不需要 masks；其训练仍用 image-mask pairs。这类表述必须按训练/生成/评估角色拆开。[DefFiller §3.1.4、§4.1.2](https://arxiv.org/html/2412.15570v1)

### B / E：小缺陷问题已被明确提出，不能从头包装

UniScale 明确分析小尺度异常在 VAE/latent 表示中受损，并采用多尺度概念训练与 crop-based 合成；TF-IDG 已处理局部放大与细小/狭窄缺陷。O2MAG 的附录仍报告小参考区域注意力不稳，这是该方法的边界，不足以证明整个领域没有解决方案。[UniScale §3–5](https://arxiv.org/html/2608.07864v2)，[TF-IDG 方法](https://openaccess.thecvf.com/content/ICCV2025/papers/Xu_Training-Free_Industrial_Defect_Generation_with_Diffusion_Models_ICCV_2025_paper.pdf)，[O2MAG Appendix I](https://arxiv.org/html/2603.18093v1)

低对比 MURA 已是专门先例；增强显著性与保留真实的弱对比幅度不是同一目标。透明材料的折射、厚度和采集因素未在本次静态审计中验证，不能用 Plastic_Bomo 名称或 bottle 类别替代物理证据。本阶段没有读图或计算新的 contrast / residual / LPIPS。

### C / D：已有能力不应再次命名为 Innovation-1

AnomalyDiffusion 已分离外观与空间信息并在局部生成；DefectFill 恢复 mask 外 normal latent。通用 Blended Latent Diffusion 也分析局部编辑、背景一致性及细窄 mask 下采样的困难。latent 保持并不等价于硬 RGB 相等，但无条件背景复制是实现/评估约束，不是独立的方法创新。[AnomalyDiffusion](https://arxiv.org/html/2312.05767v2)，[DefectFill §3.3](https://arxiv.org/html/2503.13985v1)，[Blended Latent Diffusion §4](https://arxiv.org/html/2206.02779v2)

适配方式已有 full/share fine-tuning、LoRA、embedding 与 training-free reference：例如 SeaS、DefectFill、AnoGen、O2MAG/TF-IDG。不能把“换一种 adapter/LoRA”本身当问题；training-free 也不必然意味着推理阶段完全没有梯度优化。[SeaS](https://arxiv.org/html/2410.14987v3)，[AnoGen](https://arxiv.org/html/2505.09263v1)，[O2MAG](https://arxiv.org/html/2603.18093v1)

### F：内容驱动标注已存在，但预测结果不是独立真值

SeaS 的 RMP 从生成特征预测 mask；DualAnoDiff 对生成的 anomaly-part 图提取 mask；DefectSimu 用 SAM 修正并进行接受筛选。因此“所有 label 都是 condition 复制”不成立。[SeaS](https://arxiv.org/html/2410.14987v3)，[DualAnoDiff](https://arxiv.org/html/2408.13509v3)，[DefectSimu](https://openaccess.thecvf.com/content/ICCV2025W/VISION%2725/papers/Wei_Two-stage_Text-Guided_Diffusion_Models_for_Disentangled_Industrial_Defect_Generation_ICCVW_2025_paper.pdf)

SynSur 明确报告 thin-scratch 上内容分割失败，因而回退到 condition mask，并承认标签噪声。这个窄边界值得记下，但不能据此直接提出新 mask 模块、teacher selector，或重命名为已停止的控制失败路线。[SynSur §4.4](https://arxiv.org/html/2604.26633v1)

### 不可忽略的协议差异

DefectFill 的评估生成使用剩余 two-thirds 的 masks；这不是本项目的 TRAIN-only 条件协议。其 LFS 为多次生成后选择；GAA 的 ARS 也进行筛选。它们的结果不能直接视为 raw-bank 结果，不能跨协议汇总为相同下游收益，更不能拿来允许本项目读取 final_eval 条件。[DefectFill 实验设置、§3.3](https://arxiv.org/html/2503.13985v1)，[GAA 方法与 Table 5](https://arxiv.org/html/2507.09619v1)

## 最多两个保留问题，不指定主创新

Q1：**严格 1–3 shot、仅 bbox/image-level 监督时，低占比局部缺陷概念能否被辨识，而不是学成物体/背景概念？**

最近先例为 AnoGen non-mask ablation、MURA、AnomalyAny。问题有测试价值，但还未证明比既有 weak baseline 多出什么未解决内容。Plastic_Bomo 的 TRAIN bbox 可定义信息预算，不能充当 pixel truth；合理的未来诊断可在公开 mask 数据上把 masks 对弱监督训练隐藏，只保留为独立评估。不得修改历史标签。此处仅描述比较需求，不授权实验。

Q2：**tiny / weak / elongated defect 的 synthetic label，是否在独立、盲评的内容判断下仍可靠？**

最近先例为 SeaS、DualAnoDiff、DefectSimu、SynSur。未知的是特定弱标注设置下原始未筛选标签的可靠性，不是“有没有生成 mask”。当前条件 bbox、差分图、teacher prediction 都不能自证。若未来单独立项，应先检验歧义是否可判定、已有内容标注是否足够；若只是复述 Stage23B 或转成筛选，立即否决。

完整候选字段、九项资格检查、外部数据候选、novelty risk 和否决条件见 [candidate_gap_ranking.json](candidate_gap_ranking.json)。两个问题风险均高；`STRONG_GAP = 0`，`primary_qualified_count = 0`。本次不提出任何网络、loss 或生成模块。

## Stage20B baseline claim audit

建议精确称谓：**repository-specific SD2 local inpainting implementation**，并注明 class-fine-tuned V2 checkpoint、固定 512 patch、weak-bbox condition、raw A/B/C banks。

静态追踪到的实际路径如下，详见 [stage20b_baseline_claim_audit.json](stage20b_baseline_claim_audit.json)：

1. repository pipeline 将 4-channel latent、1-channel mask、4-channel masked-image latent 拼为 9 channels。
2. 当前 active `__call__` 的 DDIM 更新覆盖完整 latent 网格，loop 内没有 mask 外 normal-latent 恢复；随后解码完整 patch。
3. `generate_plastic_bomo_stage20b_banks.sample` 把整个 **valid crop region** paste 到原画布，不是只 paste bbox mask。
4. 所以在解码 RGB 画布中，patch 外由原图保持；patch 内 bbox 外允许改变。这是代码权限边界，不是本次观测到的生成控制失败。也不承诺重新 JPEG 编码后逐像素相等。
5. `stage20b_formal_common.synthetic_label` 从冻结 condition 复制 class+bbox，不从生成内容定位；结构审计不等于独立 label validity。

9-channel 与全空间更新本身可以出现在常规 SD inpainting，**不能单凭这两点判为非标准/错误**。但“standard SD2 baseline”若不说明 repository 版本、wrapper、checkpoint 和 paste 语义，就不充分。历史协议的 “standard inpainting only” 可理解为排除 magic/dynamic/task-guidance 分支，不能据此证明与某个 upstream Diffusers 版本逐字节一致。

20 个已有 source/metadata 文件的 SHA 与基准 Git commit 核对一致；记录过的 pipeline/scheduler/wrapper runtime binding 被追踪。但历史 image processor / VAE 的完整传递版本身份未单独证明，upstream parity 标为 `NOT_VERIFIED`。不读取权重、不重跑、不改变 Stage20B 数值，也不宣称换称谓就能产生正结果。

## 交付与停止条件

14 个要求文件齐备；另附只读 schema/hash 核验及产物 hash 清单。核验只检查 JSON、源码/已有 metadata 字节和 Git，不调用模型，不检查图像/标签 payload。

已有历史结论与数据、模型、split、人工标签保持原样。只提交本 Stage24A 新目录，不包含工作区原有的其他修改。没有 Stage24B 自动启动，没有获准的下一步训练或 premise execution。
