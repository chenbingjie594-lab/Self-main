# 本地复核记录（2026-10-11）

正式入口为`authoritative_formal_run.json`，不是任意一个失败run。
正式实验目录/状态/指标/manifest保持下载时原始字节，不重新finalize或改写。
完整复核通过：1297个文件hash、788项检查；缺失资产0、失败检查0。

## 独立复核

`tools/review_plastic_bomo_stage20b_formal.py`是独立的只读复核器，不属于已冻结的七个正式执行脚本。
完整复核输出为`local_review_manual_C_20261010_032543_complete.json`；
较早的`local_review_manual_C_20261010_032543.json`是增加final_eval复制资产核验前的第一份记录。
以`_complete.json`中的reviewer SHA对应当前复核源码。

复核内容：权威pointer与ledger；Stage20A/P0/R0及正式代码hash；
两个训练完成的checkpoint树及原停止run/复用资产的逐文件hash；
240 raw PNG/labels及RGB解码SHA；
四份完整32400行draw日志与冻结role、position、augmentation seed；
四个primary last.pt字节；四组600尝试/592成功/8skips；
checkpoint barrier和一次48×4 final_eval；
本地复制的48 image/48 label的V2 hash与33 Flash/27 Black计数；
从四份metric JSON独立重算mean、sample SD、strict wins、逐类delta和三个冻结Gate。

没有从模型预测重新计算AP，没有重新运行detector，也没有进行本地GPU训练/推理。
runtime检查是服务器记录与冻结R0的对照，不宣称本地Python/CUDA与服务器相同。
验证支持已下载记录和资产的一致性；不独立证明未知物理来源完整，也不独立确定SIGKILL的外部原因。

## 测试

原有Stage20B P0/R0/正式/恢复测试：82项通过（本次284.989s）。
独立复核算术/hash边界测试：8项通过。共90项本地测试。
源码语法检查通过。测试仅为CPU/unit/static检查，不代表重新执行正式GPU实验。

复核命令（输出必须另用不存在的新文件名，不能覆盖旧复核）：

```text
python tools/review_plastic_bomo_stage20b_formal.py --run results_for_gpt/plastic_bomo_stage20b_formal/manual_C_20261010_032543 --output results_for_gpt/plastic_bomo_stage20b_formal/new_local_review.json
```

Git仅归档本Stage20B新增代码/命令、报告、JSON及日志；大权重、图片和cache不提交。
原有无关worktree修改/删除不包含在本提交。没有修改Stage20A/P0/R0权威文件。
当前负结果完整保留；不启动下一阶段。
