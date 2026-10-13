# Stage25 — MSDF-V2 Clean Requalification

## 本轮实际结论

状态：`MSDF_V2_NOT_QUALIFIABLE`，范围限定为 **当前本地 P0 执行环境**。

这是环境/入口资格未通过的停止记录，**不是 MSDF 性能失败或算法不可解耦的结论**。完整 Stage25 正式复现尚未完成，不能据此支持或否定 MSDF 的第一创新资格。

按用户“P0 任一硬条件失败即停止”的规则，本轮没有启动安全 forward/backward、generator training、正式生成、detector training 或统一评价。所有 Stage25 指标为 `NOT_RUN/null`，没有用历史结果伪装新 replicate。

基准 commit：`91a6fcb4882ea056d796a27367ebd20f2695c3fe`；日期：2026-10-13。第一创新路线按本次请求转为历史 MSDF 的 Plastic_Bomo V2 重新资格验证；Stage24B/AnoGen、新 controllability、gradient/curriculum/TDCRG 未启动。BootstrapGuard 保持独立并且本轮运行数为 0。DeepPCB 未重新开放。

## P0 五项硬条件

| 条件 | 本轮结果 | 证据边界 |
|---|---|---|
| 历史 MSDF generator 代码存在 | PASS | core、训练、推理入口从冻结 Git 版本读取；当前存在文件的字节也匹配。 |
| MSDF 与 selector 可分离 | PASS，静态代码 | MSDF-only 分支不调用 DWBG；原通用生成 wrapper 的 normal filtering 必须排除。尚无 Stage25 runtime forward 证明。 |
| V2 TRAIN 可绑定 | PASS，本地实际 SHA | 136 real images、172 instances、394 normals；666 个 TRAIN 资产全部匹配。 |
| 训练入口可运行 | FAIL，当前本机 | `train_rda.py --help` 实际退出 1：缺少 `tqdm`。同时缺少 transformers/accelerate/safetensors；检测环境版本也不匹配。 |
| 无已知 final_eval 训练 exposure | PASS，冻结已知 lineage 范围 | 候选训练/normal 来源都属于 V2 TRAIN；不声称未知物理关系全部排除。实际训练为 0。 |

没有把第三项的“数据身份可绑定”写成“训练 tensor/data adapter 已准备并测试”。正式 V2 runtime 适配、安全测试与 frozen checkpoint barrier 均未完成。

## 历史 MSDF 到底是什么

本轮冻结的是已有 **MSDF format version 3**。Stage25 的“MSDF-V2”指 Plastic_Bomo Generation V2 数据协议，不是新 MSDF-v4，也不是把 adapter 降成 format 2。

代码链为 `run_train_msdf.py → train_rda.py --rda_mode none --enable_msdf`；推理链为 `inference.py → pipeline_stable_diffusion_inpaint_magic.py → load_msdf_adapter / prepare / up-block hooks`。详细 tensor、参数和作用位置见 [msdf_historical_component_audit.json](msdf_historical_component_audit.json)。

核心包括：

1. pre-VAE RGB pixel residual encoder 与独立 pixel support head；
2. masked latent residual semantic encoder；
3. 两分支融合及 reference-derived support；
4. reference morphology 对齐：目标粗 mask 给中心/范围，reference 给形状/占比；
5. mask area / timestep / feature response 动态 gate；
6. 所有 UNet up blocks 的频率感知 feature residual injection；
7. 相对 hidden RMS 的 tanh bounded residual；
8. 历史 generator 训练的六项辅助目标。

MSDF-only 没有额外 text reference tokens 或 attention processor 架构替换；它通过 up-block forward hooks 改变 feature。full UNet 与 MSDF 一起训练，VAE/text encoder 冻结。latent residual 是条件来源，不是直接覆盖 diffusion latent 的独立 sampler。不能把它说成“只打开一个推理开关就能完整复现”。

六项 MSDF loss 的历史系数分别为 region 0.75、clean reconstruction 0.50、structure 0.25、background 0.20、multi-scale consistency 0.25、support 0.25；support 内 pixel-support 权重 0.50。历史 self/cross reference、branch dropout、mask dilation/soft-prior 都已列入协议差异，未删除以制造表面公平。

所有核心代码保持原样；没有新增 attention、adapter、loss、teacher、task guidance 或其他网络结构，没有算法行为 bug fix。历史 `_safe_rms` 等稳定性处理是已存在实现，不是本阶段偷偷加入。

## selector 分离与旧资产禁用

MSDF core 不依赖 DWBG；TRAIN 内 saliency 是原有训练 objective 的支持权重，不是生成后筛图。

但是历史 `configs/msdf.json` 的 generic inference 启用了 brightness/nonblack normal filtering。故 **不能原样运行旧命令，再把结果改名为 Stage25 raw**。Stage25 必须使用冻结 TRAIN normal pool，不进行该过滤。

历史 DWBG 链的 residual extraction/recomposition、pseudo-bbox、detector feature manifold、consensus、ranking、hard/anchor quota 与类别/来源/seed cap 都不属于 MSDF core。本阶段均未运行。详情见 [msdf_selector_separation_audit.json](msdf_selector_separation_audit.json)。

旧 761 张派生池未作为 Stage25 数据，未读取它的图像做质量判断；旧 MSDF checkpoint 未加载或复用。Stage14/15/DeepPCB 数值没有与 V2 合并。

## V2 数据和 frozen comparator 实际核对

只按 frozen TRAIN allowlist 打开文件做 byte SHA，不扫描 final_eval 图片/标签目录，不解码图像，不计算新的 quality 指标。

- 136 unique real TRAIN images，134 known source groups；Flash 75 instances，Black 97。
- 394 frozen TRAIN normals；136 image + 136 label + 394 normal，共 666 个资产匹配。
- 本地/服务器 P0 的训练语义、394 个 normal 的身份语义一致，仅路径迁移。
- Stage20A 24 个冻结 metadata 文件、local P0 19 个、server P0 21 个、正式 Stage20B 36 个结果/审计文件均匹配。
- Raw A/B/C 各 80 张，40/40；240 张 raw image 和 240 个 label 全部匹配。
- 两类 Raw generator 共 24 个组件文件匹配；tree SHA 与 2000 successful-update 记录对应。
- RR/A/B/C 四个 epoch150 `last.pt` 的实际 SHA 匹配。
- public SD2 base 的 12 个指定组件文件与 Stage20B 冻结 reference 相同。

正式 Stage20B authority 是 `manual_C_20261010_032543`，不是中断的原始 run。核对使用其最终 artifact manifest，不混用恢复前 C 的模型/指标。见 [p0_asset_hash_verification.json](p0_asset_hash_verification.json) 和 [baseline_reuse_audit.json](baseline_reuse_audit.json)。逐文件 expected hashes 已在冻结原 manifest 中；只读 collector 可输出完整核对行，汇总文件保存全部 checked-row digest。

这些是**本地下载镜像**的身份验证，不是当前服务器资产重检。已知 source-group isolation 成立；沿用 V2 对 physical source incomplete / unknown alias 的限制，不额外宣称全物理来源已证明。

## 公平性目标与尚未执行的适配

Baseline 称谓固定为：

`repository-specific SD2 local inpainting implementation (class-fine-tuned V2 checkpoints; fixed 512-patch; weak-bbox condition; raw-bank protocol)`。

[msdf_v2_protocol.json](msdf_v2_protocol.json) 保存请求的完整 frozen targets：同一 public base family、V2 TRAIN、prompt、初始 bbox rectangle、512 crop/paste、normal/slot roles、sample seed derivation、sampling steps、CFG、DDIM、synthetic count 与 detector protocol。

MSDF 自身的 coarse-prior dilation/blur、reference 分支、辅助 loss 和额外参数组属于已明确登记的 `MSDF_REQUIRED_METHOD_DIFFERENCE`。因此不能声称变换后的所有 training masks 都数值相同。UNet 的 legacy LR 5e-6 不是 core 结构必要条件；V2 common target 为 baseline 1e-6，MSDF 参数组保留历史 1e-4。

不能仅改旧训练脚本的目录：V2 manifest-only 输入、冻结样本顺序、epoch-tail accumulation、tokenizer padding、真实 optimizer post-hook 计数，以及相同 sampling path 的集成都尚未实现/数值验证。不存在可据此直接启动的 Stage25 formal runner。旧 generic wrapper 不能充当已通过 qualification 的替代入口。

Raw/RR 的历史文件身份已通过，但**正式复用暂不授权**：本地 Ultralytics 为 8.4.120，历史冻结为 8.4.145；还没有 Stage25 实际训练/推理同协议证明。若最终选择了不同执行协议，按请求必须重训 comparator，不能近似比较。本轮未重训 comparator。

## 没有改动的训练、生成与判定要求

Generator 目标为 Flash/Black 各 2000 successful updates，不依据效果延长；两类 checkpoint 全冻结后生成。

A/B/C bank seeds 为 42/3407/2026，复用完整 slot-derived uint64 seed：`int(SHA256('Plastic_Bomo_Generation_V2' + slot_id + str(bank_seed))[:16],16)`，不是所有 slot 同 seed。每个成功 slot 必须入 bank，structural failure 立即停止，禁止 reroll、best-of-N、筛选、补图。

Detector target 为 frozen YOLO11s/seed42/150 epochs/batch1/imgsz1536/AdamW 及完整 Stage20B 参数。每 epoch 216 draws，accumulation groups 为 64/64/64/24；总 32400 draws、600 scheduled optimizer attempts。记录 AMP skips/successful updates，不补 step。只使用 epoch150 `last.pt`，三个 MSDF detector 全部冻结后统一评价。

第一层原阈值：mean MSDF−Raw ≥ +0.50 pp、strict matched wins ≥ 2/3、Flash 与 Black mean delta 均 ≥ −2.0 pp。第二层原阈值：mean MSDF−RR ≥ 0 且 strict wins ≥ 2/3。未修改任何 threshold。

没有训练，所以以上 gates **全部 NOT_RUN，不是 FAIL**。所有结果文件中的 null 不是 0 AP，不是合成图缺失可以换 seed 补齐，也不是被筛掉的 bank。

## 为什么本轮不自动继续

实际训练入口 `--help` 失败，发生在模型构建/读图之前。可用本地 PyTorch 环境还缺 transformers、accelerate、safetensors，无法完成合法 checkpoint loading 与单 batch safety test。

本机 GPU 是 10 GiB RTX 3080；历史普通 SD2 的训练峰值记录约 17 GiB。这是资源风险，不能当作 MSDF 理论最小显存，但也不能擅自改成 8-bit optimizer、FP16 masters、降低 resolution/batch 来绕过冻结公平性。

只读 SSH 连接探测 `cbj@mirukj` 未建立会话，返回 connection refused；没有执行远程训练，也没有读取远程模型或数据。详见 [runtime_entry_audit.json](runtime_entry_audit.json) 和 [server_execution_access_audit.json](server_execution_access_audit.json)。本轮没有安装/升级任何用户环境包。

用户“P0 全通过后无需人工确认，自动继续”已记录，不被改成重新审批算法的要求。但当前条件没通过，且尚无可验证的服务器执行会话；因此必须停止，不能假装在服务器训练完成。

解除的是执行环境/访问与 V2 runtime qualification 障碍，而不是修改 MSDF 寻找正结果。后续需要实际可运行的服务器环境及访问方式；沿用原条件授权做 safety test/完整 qualification 后才可能继续。不能把本轮 local P0 failure 作为论文 MSDF negative result。

## 交付范围

23 个要求文件齐备，未运行的训练/生成/评价文件明确标记 `NOT_RUN`。另附只读 collector、runtime/access evidence、hash 核验和标准库测试。collector **不是 formal runner**。

本次只增加 Stage25 结果/审计目录，不修改原 MSDF、Stage20A/B、数据、标签、split 或其他已有工作区修改。提交 GitHub 后停止，不启动第二创新、MSDF-v4、额外 seed/bank、外部数据或新 ablation。
