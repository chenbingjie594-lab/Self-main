# Stage21B A1 正式结论与下载复核

正式状态：`SYNTHETIC_EXPOSURE_TIMING_PREMISE_NOT_SUPPORTED`。

正式运行：`amended_execution_20261011_065553`；本地复核：`amended_review_20261012`。
`authoritative_amended_execution.json` 绑定这两个目录。原入口审计及其 `STAGE21B_CONDITIONING_BACKGROUND_DEV_OVERLAP` 状态不覆盖、不删除。

## 结果

单位：mAP50-95 和 AP50-95 均以百分数显示；delta 为 Curriculum − Uniform 的百分点差。

| Bank | Uniform mAP50-95 | Curriculum mAP50-95 | Delta pp | Flash delta pp | Black delta pp |
|---|---:|---:|---:|---:|---:|
| A | 38.1349 | 35.6347 | −2.5002 | −1.2946 | −3.7058 |
| B | 34.5425 | 34.3048 | −0.2377 | +1.0052 | −1.4805 |
| C | 40.6063 | 36.1195 | −4.4869 | −2.9672 | −6.0065 |

- 平均 delta：**−2.4082 pp**；Curriculum strict wins：**0/3**。
- Flash 平均 delta：−1.0856 pp；Black：**−3.7309 pp**。
- 跨 bank sample SD（ddof=1）：Uniform 3.0491 pp，Curriculum 0.9395 pp。
- Gate A（mean≥+0.50）：FAIL；B（至少2个strict wins）：FAIL；C（两类mean均≥−2）：FAIL；D（Curriculum SD不更高）：PASS。

这个固定 late schedule 降低了跨 bank 离散程度，但同时三个 bank 的总体效用都下降，Black 类下降超过保护阈值。不能用更低方差抵消效用下降，亦不能据此支持 timing premise。

## 协议与复核

原 split 文件字节未变：109 internal_train images / 27 internal_dev images，107 train groups / 27 dev groups，seed2101。三 bank 仅同步排除授权的4个 conditioning-group 冲突 slot，保留 Flash32 / Black29，共61个 synthetic roles；没有补样本、重生成、换normal或质量筛选。

六 arm 全部150 epochs；每 arm 25500 draws、450 scheduled optimizer attempts；每 role150次；accumulation groups 64/64/42，尾组按42归一化。实际 successful updates：A_uniform443、A_curriculum443、B_uniform442、B_curriculum443、C_uniform442、C_curriculum443。AMP跳过数有差异，没有补steps，**不声称实际成功更新数严格相同**。

本地下载复核通过：53个服务器manifest所绑定的元数据文件、6个last.pt原始文件字节及153000条draw日志均已核验。每bank所有25500个增强occurrence input/target tensor哈希完全配对；实际初始化、optimizer groups及450次attempt的learning-rate序列一致。未改服务器冻结文件，未加载模型或重跑forward。

服务器安全检查两个schedule均PASS，optimizer.step=0。六个checkpoint均冻结后统一评估internal_dev一次，每checkpoint27张，共162 image forwards。官方V2 final_eval图像/标签读取及forward计数均为0；这些访问声明基于固定代码、访问guard及下载的审计计数，并非OS级访问日志。

## 解释限制与停止

冻结generator曾使用当前dev group中的33个annotation inputs（Black23 / Flash10）。因此本实验只能称为 **DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR**，不能称为pipeline-independent holdout。

本结论针对当前固定的late schedule、单个detector seed42及三个frozen banks；不证明所有curriculum无效，不给出V2 official final_eval效果，也不构成创新确认。

Generation=0；BootstrapGuard=0；DeepPCB=0。没有新selector、sample weighting、TDCRG或Stage21C。归档本阶段代码、元数据、完整draw日志及本地复核后停止，等待人工决策；不再搜索warmup、比例、seed或其他schedule。

Git不上传checkpoint、训练/评估图像标签或预测产物，相关权重SHA在`detector_checkpoint_binding.json`中保留。完整draw日志上传，支持逐occurrence复核。
