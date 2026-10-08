# Stage18A 服务器结果下载复核

复核对象：`server_inventory_20261008_042413`。保留原始服务器输出，不改写历史结果。

## 已确认的新增证据

- MSDF-v3 checkpoint 目录实际存在，包含 Flash / Black 两套 UNet、VAE、text encoder 和 msdf.pt，共 8 个权重文件；服务器已记录各文件 SHA256。文件存在不等于已经证明其训练来源或与历史 pool 的实际运行绑定。
- 761 张 manifest 声明的 synthetic 图像和 761 份 YOLO 标签均通过服务器检查：Flash 576、Black 185，无缺失/空标签/非法 bbox/class mismatch/重复路径记录。
- 761 个图像路径、761 个标签路径各自唯一；图像和标签的 SHA256 也各自有 761 个不同值。上述是服务器审计记录的复核，本地未下载这批图像重新解码。
- 根 candidate_pool/images 中有 673 张，另外 88 张位于 flash_large_expansion 的各 seed 子目录。两项统计并不矛盾，合计 761 张。
- Stage14 Random 的 80 个 ID 全部属于该 pool，与历史 Random manifest 顺序和集合均一致。

## 没有解决的准入问题

1. MSDF checkpoint 的实际训练数据、frozen train membership、normal/background 来源以及 validation/test 接触情况仍为 `ISOLATION_UNRESOLVED`。不能将配置中的 train 路径当作已证明的隔离。
2. 761 张有合法 detector annotations 的图像来自过滤/重组后的 DWBG pool，不是 761 张 untouched raw MSDF output。其实际生成 checkpoint 与运行记录的哈希绑定仍缺失。
3. 历史 Random 使用 manifold_valid 限制和 parent/seed caps；它是过滤 pool 内的随机子集，不能自动充当 selector-free raw generator headroom 证据。
4. 当前指定服务器路径中的 AnomalyDiffusion / DefectFill 仓库及输出目录不存在。这只说明这些指定路径不可用，不证明其他服务器或其他目录没有资产。
5. 服务器缺少历史 DWBG builder/scorer/selector 代码；来源分析应结合 Git 中已存档的本地代码证据，不应声称服务器本次重新验证了全部历史构建过程。

## 原始报告中的模板文字限制

服务器 `STAGE18A_REPORT.md` 中“AnomalyDiffusion launcher 存在”“DefectFill 本地 concrete/crack 示例”等句子沿用了此前本地审计模板，不能作为本次服务器观察。实际服务器 JSON 显示上述代码目录和示例日志 unavailable，DefectFill input_conversion 为 null。

同样，服务器已经找到 MSDF 权重，因此“checkpoint identity unverified”应理解为**训练与生成运行的来源绑定未确认**，不能理解为模型文件不存在。此复核记录以 asset_inventory、registry 和 annotation audit 的具体字段为依据，不覆写原始模板报告。

## 可复核性检查

- 所有下载 JSON 可解析。
- 执行脚本 SHA256 与正式 commit `fc80b564ea6170e0eef9a96bbc2562805d2ac0cb` 的 Git blob 完全一致：`74f85a423c8f4f9e5c164323c9d321c8cd6379aed00fe339b2b959ca761b43db`。
- protocol SHA256 完全一致：`e6252ac9b355e4f99736b044cdfe149fa6631826e5ba6bb072735769a49db9aa`。
- candidate pool 和 Stage14A subset_definition 的输入 SHA256 与 Git blob 完全一致。Random manifest 的差异仅是 LF/CRLF；Git 内容换为 CRLF 后 SHA256 与服务器输入记录一致，并与当前本地文件一致，不能当成样本选择被修改。
- 55 条历史 metrics 仍被区分为 33 selected、12 random、3 Stage17A raw Vanilla 和 7 real controls。存在跨阶段复用，不作独立重复、不进行统计汇总，不把 selected utility 归给 generator。

## 最终解释与下一步

当前已核验范围内仍为 `NO_AUDITABLE_ALTERNATIVE_GENERATOR_BASE`，没有 Stage18B candidate，`STAGE18B_BASELINE_UTILITY_SCREEN_AUTHORIZED=false`。

相比本地报告，本次解决了“MSDF 文件是否存在”和“DWBG synthetic 图像/标签是否可用”两个问题；**没有解决 train-only 来源与 selector 独立性**。不能据此启动 detector screen，也不重训、不重新生成来补足审计。

当前 frozen dataset 的 generator-first 路线维持停止；这不是穷尽所有历史服务器后证明 alternative generator 不存在。若没有额外已有训练日志、source/split manifest 和生成运行绑定证据，下一步应将这份复核交给研究方案分析，而不是再启动实验。
