# Stage23B-D0 — Controllability Measurement Feasibility Audit

源版本：`73c1e2a453641cdd63fbe5d2292ff1b10e64d49c`；审计日期：2026-10-12。

正式状态：`NO_MEASURABLE_NOVEL_CONTROLLABILITY_GAP`。

这里的含义是：本轮没有确立同时满足五项要求的新 gap，**不是所有像素变量都不可测，也不是证明未来不存在 gap**。当前有非机械决定的编辑变化可测，但“缺陷位置/形状/可见性控制”的独立真值与文献差异尚未成立。没有选方法、没有认定当前生成器存在控制失败、没有授权实验。

## 1. 实际生成链，而不是类名推断

Stage20B 的协议把 pipeline 描述为 standard inpainting；实际导入的是仓库内同名类，`__call__` 是当前仓库实现。它不能用上游 Diffusers 的惯常行为替代解释。C/M 证据编号及完整文件 SHA256 见 [source_evidence_index.json](source_evidence_index.json)。当前 pipeline、DDIM、wrapper/common 的文件字节与已有服务器/正式 binding 对应。

`normal → 固定 512 crop/黑 padding → weak bbox 矩形 mask → 全空间 latent 去噪 → 整个 patch decode → RGB uint8 → 整个有效 patch 粘贴回 normal → PNG；另一路 input bbox → YOLO label`。

| 问题 | 代码结论 | 固定证据 |
|---|---|---|
| bbox 外 latent 是否更新 | 是，初始化和 scheduler 更新整个空间网格 | C5:956–979、1040–1073；C7:410–448 |
| bbox 外最终像素是否复制 normal | patch 内不复制；只有有效 patch 外原样保留 | C1:30–34 |
| 每步是否恢复 unmasked latent | 当前 active loop 没有此操作 | C5:1040–1075；C7:342–351 |
| PNG 前是否再 compositing | 是整个有效 patch 的 paste，不是 bbox mask composite，也没有羽化 | C1:28–35、88–89 |
| 输出 annotation 是否复制 bbox | 是，完全不从生成像素估计 | C3:237–251；C1:94–100 |
| location accuracy 是否可能机械 100% | label 与 input 比较必然一致，但不是像素位置准确率 | C3:249–251；C9:56–57 |

9-channel UNet 的 mask 只是输入条件。`apply_overlay`、上方的 `prepare_latents`/`prepare_mask_latents` helper 没被 active `__call__` 使用。DDIM 的 active step 没有 normal/mask 恢复参数。当前代码没有 bbox 外 latent 恢复，不应从别的版本的 4/9-channel 分支推断。

另一个重要限制：同一 slot 的 RNG 被 original-normal VAE posterior sample、初始 noise、masked-normal VAE posterior sample 消耗；`eta=0` 只消除 DDIM step 的附加随机噪声，不意味着结果与 seed 无关。这是链路解释，不是代码修改建议。

定义全画布 Ω，有效粘贴区域 P，冻结 bbox 的 raster ROI B。硬保证的是 `G=N on Ω\P`，不是 `G=N on Ω\B`。`P\B` 的背景变化是非机械 observable；`Ω\P` 的保存准确性没有资格作为创新 endpoint。

元数据检查：80 slots，flash/black 各40；全部 512 crop、单位 resize scale、输入输出画布同尺寸；22 slots 有画布外 padding；全部 mask 为 weak bbox。raster bbox 最小宽/高分别14/10，最大58/53。这些是几何元数据统计，未读图片。服务器 A/B/C 的 frozen slot、normal、checkpoint、label 元数据成对一致。

完整历史依赖身份有边界：既有运行记录没有 image_processor/VAE 源文件的独立服务器 SHA；本次将它们固定到源 Git 版本，没有伪称完成历史 transitive runtime 重执行认证。

## 2. 哪些轴可测

| 轴 | 分类 | 解释 |
|---|---|---|
| REQUESTED_CENTER、REQUESTED_WIDTH_HEIGHT | HARD_CONSTRAINED | input/mask/label 几何；不等于生成缺陷的位置尺寸 |
| EDIT_SUPPORT_CENTER、EDIT_SUPPORT_EXTENT | MODEL_DEPENDENT_MEASURABLE | 连续 residual 的重心/协方差；不是缺陷 bbox 或分割 |
| LOCAL_CONTRAST | MODEL_DEPENDENT_MEASURABLE | 只称 LOCAL_EDIT_CONTRAST，不称 DEFECT_QUALITY |
| BOUNDARY_BLEND | MODEL_DEPENDENT_MEASURABLE | 环带可定义；厚度与规范未冻结，不能作为当前 primary |
| OUTSIDE_BACKGROUND_CHANGE | MODEL_DEPENDENT_MEASURABLE | 限 P\B；P 外硬复制的部分单独排除创新解释 |
| IN_BOX_APPEARANCE_DIVERSITY | MODEL_DEPENDENT_MEASURABLE | ROI perceptual variation 可行性；小 ROI 输入规范待冻结 |
| CROSS_RNG_CONDITION_VARIATION | MODEL_DEPENDENT_MEASURABLE | 同一 slot 的预定 RNG 重复测量；低方差不是成功 |
| CLASS_IDENTITY、DEFECT_VISIBILITY | MODEL_DEPENDENT_NOT_CURRENTLY_MEASURABLE | 尚无独立有效的 output class/visibility measurement |

`UNSUPPORTED_BY_CURRENT_LABELS` 适用于 defect shape IoU、真实 silhouette adherence、pixel appearance accuracy 等额外主张；这不是把 weak bbox 变成 segmentation truth。完整11轴见 [condition_axis_taxonomy.json](condition_axis_taxonomy.json)。

## 3. Residual 合法性和候选公式

静态结论：`RESIDUAL_MEASUREMENT_STRUCTURALLY_VALID`，没有进行实际像素验证或计算。G 是最终 RGB PNG；N 是准确的 hash-bound conditioning normal，经生成代码相同 RGB decode/conversion；两者都按 float64 RGB code values /255 表达。禁止把 source defect crop、VAE reconstruction 或另一个 normal 当 N。

同 crop origin、单位 resize、相同 full-canvas 坐标及保存像素尺度，允许定义：

`R(p) = mean_RGB |G(p)-N(p)|`。

基准采用原画布像素中心 `(i+0.5,j+0.5)`；只计有效原画布，padding 不进入测量；B 采用现有 floor/ceil raster rectangle，requested center/width/height 保持原连续 bbox。记录这两种几何的区别，不重新切出“看起来最好”的 ROI。

- `E_total=ΣΩR`：总编辑能量，不是 defect visibility。
- `BDR=Σ(Ω\B)R/E_total`；`ECR=ΣBR/E_total`。当前不是机械常数，但 `ECR=1-BDR`，不得当两个独立 primary。BDR 小也可能只是 bbox 内错误变化更强。
- `μ_R=ΣΩ pR/E_total`；`NCE=||μ_R-c*||/bbox_diagonal`。这是全局编辑重心，不是缺陷中心；patch 背景重新生成可主导它。
- bbox 内条件残差二阶矩 `Σ_RB=ΣB R(p)(p-μ_B)(p-μ_B)^T/E_B`。`rms_width=sqrt(Σ_xx)`、`rms_height=sqrt(Σ_yy)`，分别除以连续 requested width/height；此处 RMS 是一个空间标准差，不是满宽，也没有“ratio 应为1”的目标。anisotropy 用 `(λmax-λmin)/(λmax+λmin)`。只描述 bbox 内变化占据空间；受 ROI 截断，不能解释为真实缺陷形状。
- `LOCAL_EDIT_CONTRAST`：bbox 内 channelwise intensity spread 的 G−N 变化；不评价质量。

没有 PASS/FAIL threshold。`E_total=0` 时依赖它的比值/重心为 null，并报告 `UNDEFINED_ZERO_EDIT_ENERGY`；不能写 NCE=0 作为“完美位置”。`E_B=0` 和零空间 spread 同样独立标记，保留全部输出与计数。未来若实际文件/尺寸/坐标/色彩对应不成立，应写 `RESIDUAL_MEASUREMENT_NOT_VALID`，停止该指标路线，不能重新配准或换指标救结果。

bbox boundary residual discontinuity 只冻结为候选指标族。inner/outer ring 的厚度必须由分辨率/几何独立决定，统一角点/侧边权重与面积规范；不得现在随意指定几像素。环带需在 P 内，P-edge hard paste seam 与 bbox seam 分开。弱框边界不是缺陷实际边界。

完整公式、单位、不可识别项见 [candidate_metric_definitions.json](candidate_metric_definitions.json)。

## 4. 跨 RNG 与 diversity 必须分开

未来同 slot 的 M 值报告 median、IQR、range、sample SD；quantile 使用 type7 线性插值，SD 用 k−1。保留零能量/不可定义计数。不能根据 A/B/C mAP 排序选择定义。

permitted appearance diversity 的候选是同 slot、所有无序 RNG pair 的 bbox-only perceptual distance。优先评估 masked spatial LPIPS 的可行性，但现在未安装、未加载、未计算，backbone/版本/weights/小 ROI padding/resize 未冻结。[LPIPS 作者代码](https://raw.githubusercontent.com/richzhang/PerceptualSimilarity/master/lpips/lpips.py) 的 spatial 模式上采样 feature distance map；并不自动提供 defect-mask 隔离。

只在输出 map 上遮 bbox、但输入全 patch，会通过 receptive field 带入框外内容。输入先裁 B 可阻止真实框外像素参与，但仍有 crop padding、边界及框内正常纹理影响。当前小框不得不说明固定模型的合法最小输入尺度与统一处理；任意放大不是已验证的缺陷度量。

LPIPS 距离是 appearance variation，不是 defect validity。低方差可来自 collapse，高多样性可来自伪影。condition adherence、diversity、能量/无定义情况必须独立报告，不加权合并，不因为 collapse 或低质量把输出移除。

class identity 的独立候选：先冻结判读准则的 blinded human review；或具有 train-only source-group 隔离和域偏移证据的 classifier。当前均未执行/建立有效度量；teacher/classifier 也不能充当 geometry truth。暗度、边缘和能量不唯一对应 flash/black。

## 5. 七篇既有文献与 gap 判断

56格完整判断见 [controllability_prior_art_gap_matrix.json](controllability_prior_art_gap_matrix.json)。`NOT_VERIFIED` 不等于已有文献没覆盖；`POTENTIAL_GAP` 不等于 novelty。沿用 Stage23A 七篇文献身份，复核作者全文相关方法、评价和附录；CVF fetch 失败的两篇用相同论文的作者 arXiv 全文补充，不引入二手机制。

- [AnomalyDiffusion](https://arxiv.org/html/2312.05767v2)：位置/外观分离、normal/generated pixel difference、mask alignment 和 IC-LPIPS 都已有。
- [DefectFill](https://arxiv.org/html/2503.13985v1)：背景 latent 逐步恢复；同 normal/mask 多样本选一；有 no-LFS generation ablation，不能一概说没有 selector-free 对照。
- [GAA](https://arxiv.org/html/2507.09619v1)：区域/位置、mask alignment、多样性及过滤；有 no-LQSF ablation，main downstream 使用筛选。
- [UniScale](https://arxiv.org/html/2608.07864v2)：小尺度、背景融合、anomaly-region diversity 和 identical-mask 对照已有；不移植尺度 cutoff 或效应到 Plastic_Bomo。
- [ControlNet](https://arxiv.org/html/2302.05543v3)：多种空间条件、人评 fidelity；展示未 cherry-pick 的 random batch，但不是本项目同 slot 的完整严格 raw-output contract。
- [ControlNet++](https://arxiv.org/html/2404.07987v4)：从生成图提取条件的独立一致性评价；四组生成结果平均降低随机误差，不能称重复生成评估从未有人做。
- [Blended Diffusion](https://arxiv.org/html/2111.14818v2)：local edit/background preservation；最终 mask 外复制，且有 CLIP 排名，不能直接作为 raw selector-free robustness 证据。

“weak bbox + paired RNG + raw edit moments + diversity”是较窄的候选设置，但没有证明文献缺口；改数据集/改度量本身也不证明新生成机制。主张真实缺陷 geometry/class/visibility 又缺独立输出度量。五项 gate 逐候选记录在 [protocol_worthy_gap_summary.json](protocol_worthy_gap_summary.json)：可测、非 hard constraint、文献未直接覆盖、独立于 BootstrapGuard、独立于 mAP；unknown 不当作通过。

因此本轮 qualified gap=0，primary endpoints=0，不给方法设计或 premise experiment 绿灯，也不重开已关闭路线。不是使用某个弱代理硬凑 FEASIBLE。

## 6. 未来数据地位与边界

既有 A/B/C 只能是 `EXPLORATORY_MEASUREMENT_FEASIBILITY`，不能是独立确认。D0 没有读取/解码它们的图片。若将来另外确立 gap并获得授权，优先在采样前冻结新的 RNG realizations、slot、完整链路版本与独立度量；没有现在选择 seeds 或启动生成。

所有 predetermined raw outputs 全部计入。无 drop、Top-K、quality reroll、failure replacement、near-zero weight、仅保留通过图或 best-of-N。不可定义 metric 记录 null/原因，不删除图；结构运行失败停比较，不替换。BootstrapGuard 是另一条 post-generation retention 路线，本次没有变更。

## 7. 审计与提交范围

本轮只创建这个目录的报告、JSON、read-only verifier。训练、采样、detector/perceptual forward、backward、optimizer step、图片读取/解码、residual 计算、LPIPS 安装、新模块、BootstrapGuard、DeepPCB、official final_eval 均为0。旧代码/结果、模型、图片、标签、划分均不修改；用户其他 dirty worktree 变更不纳入提交。

复核命令（仅标准库，只读源文件/JSON/本目录 artifact，不访问图片或 weights）：

```text
python results_for_gpt/plastic_bomo_stage23b_d0_controllability_protocol/verify_d0_contract.py
```

它检查22个固定证据 SHA、active AST 的 full-latent update/no overlay/full-patch paste、80-slot A/B/C metadata pairing、11轴/56文献格、全零计数和无执行授权；这是静态复核，不是生成/数值 runtime 测试。最终字节清单为 `review_artifact_manifest.json`。提交 GitHub 后停止，不自动继续 Stage23B 实验。
