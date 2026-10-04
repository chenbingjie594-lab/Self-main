# Stage16A-R0 服务器停止结果复核

正式执行状态：`ANNOTATION_PROVENANCE_INSUFFICIENT`。
复核状态：`SERVER_STOP_REVIEW_PASS`，567/567 项检查通过。

## 实际扫描与原因

服务器 scope 为 `SERVER_PREFLIGHT`，扫描 frozen real-train 的 138 张图、168 个实例：
Flash 88、Black 80。168 个实例均为 `REJECTED_PROVENANCE_UNKNOWN`。
本次没有提供 annotation evidence manifest，Flash/Black 可接受 source groups 都为 0，
低于每类至少 5 组的冻结门槛。

这证明的是本次执行没有获得足以接受标注的逐实例来源证据。
不能解释为标注必然错误、人工确认从未发生，或其他位置不存在历史证据。
也不是 task guidance 有效或无效的实验结论。

## 下载与本地快照对照

按 donor ID 对照服务器与此前本地审计的全部 168 行：image 文件哈希、decoded RGB
哈希、图像尺寸、label 文件哈希、行号、类别、bbox、known group ID、拒绝状态均一致。
OOF integrity、protocol 和 audit script 的输入哈希也一致。这个对照不使用 validation。
它只能确认所扫描的本地与服务器训练资产一致，不能证明物理源隔离或 annotation provenance。

known-relation graph 有 137 个图像 connected components；这些是描述性已知关系，
不是 137 个被接受的 source groups。singleton 没有物理源恢复证据。
conditioning、teacher isolation、bbox 执行审计、可微路径、RNG、模型不变性、task loss、
visual 和 numerical runtime 门槛均未执行，不以 JSON 占位文件冒充完成。

## 归档与边界

本目录根下为下载的服务器原始结果，逐文件复制校验哈希，不修改原始报告措辞。
服务器生成的报告仍含通用的 local-audit 表述；以原始 JSON 的 `SERVER_PREFLIGHT`
scope 和本复核为准，不据该文字声称执行了其他阶段。
此前本地结果保存在 `local_preflight_snapshot/`，没有丢弃或改造成服务器结果。
`downloaded_server_review.json` 保存原始服务器文件哈希及逐项复核。

所有生成、训练、optimizer、official validation、DeepPCB 与 BootstrapGuard 操作计数为 0。
原 Stage16A 停止记录未修改。R0 后续 GPU 工作流仍未实现、未执行。
不授权正式 Stage16A-R，不自动启动 Stage16B。

只有既有、可检查并绑定实际图像/label/标注行的来源或人工复核证据，才能支持原协议下的
重新审计。任何新建人工审核数据方案需单独修订协议，不补造历史证据，不静默放宽门槛。
