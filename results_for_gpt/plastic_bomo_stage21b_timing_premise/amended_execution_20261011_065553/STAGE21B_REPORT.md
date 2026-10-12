# Stage21B Amendment A1 — Timing premise

状态：`SYNTHETIC_EXPOSURE_TIMING_PREMISE_NOT_SUPPORTED`。完整 GPU 实验完成：True。

## 冻结范围

原 e100960 入口审计及 STAGE21B_CONDITIONING_BACKGROUND_DEV_OVERLAP 永久保留，不是 premise FAIL/PASS。
split 原文件字节保持一致：seed2101 / Plastic_Bomo_Stage21B_TrainOnly；107 train groups、27 dev groups；109 real train images、27 dev images。
仅额外排除授权的4个 conditioning-dev 同源 slot，理由 KNOWN_SOURCE_GROUP_LEAKAGE_EXCLUSION；三 bank 同步，flash32 / black29，总61。无补样本、重生成或质量筛选。

## 生成器历史暴露限制

冻结生成器曾使用当前 dev group 中33个 annotation inputs（black23 / flash10）。
本实验仅 `DETECTOR_INTERNAL_DEV_CONDITIONAL_ON_FROZEN_GENERATOR`，绝非 `PIPELINE_INDEPENDENT_HOLDOUT`；不通过 generator 重训练修复。

## 固定训练及 compute

六 arm 顺序 A_uniform→A_curriculum→B_uniform→B_curriculum→C_uniform→C_curriculum；same frozen YOLO11s init、seed42、imgsz1536、batch1、AdamW及损失/增强参数采用 Stage20B 快照。
150 epochs×170 draws；每 role150次。real16350、synthetic9150、total25500；每 epoch groups=[64,64,42]，tail /42，总450 scheduled attempts。
Uniform每 epoch109 real+61 synthetic；Curriculum前50 epoch170 real+0synthetic，后100 epoch偶数late91synthetic+79real、奇数92synthetic+78real。
role内稳定 cyclic ordering；occurrence augmentation seed仅依赖 namespace/role/index/seed42。实际训练记录 input/target tensor SHA，所有25500 occurrences按 bank配对核验。
reuse Stage20B callback controller/backward；只改变 order/occurrence RNG及必要的170-draw LR进度。保留 native EMA每attempt更新与FP16 epoch150 last.pt格式，无best/earlystop/dev选checkpoint。
compute固定的是 scheduled draws及attempts；实际AMP successful/skipped计数另报，不补 steps，若不相同绝不声称成功更新数相同。

## 访问与评估 barrier

Python open audit hook + numpy.fromfile/cv2.imread wrapper +显式 native dataset image allowlist。官方 V2 final_eval 始终禁止；不是 OS 沙箱。
internal_dev image/label首次统一读取，必须在六个last.pt全部SHA冻结且实际augmentation/exposure审计通过后。每checkpoint只评估一次，发生部分失败不自动重试。
single-batch safety仅internal_train real，两个schedule各做forward/backward，optimizer.step=0；模型丢弃，六arm从相同初始化重新开始。

## 结果

| bank | Uniform mAP50-95 | Curriculum mAP50-95 | delta pp |
|---|---:|---:|---:|
| A | 38.1349 | 35.6347 | -2.5002 |
| B | 34.5425 | 34.3048 | -0.2377 |
| C | 40.6063 | 36.1195 | -4.4869 |

mean delta=-2.4082 pp；sample SD Uniform=3.0491、Curriculum=0.9395 pp。
Flash mean delta=-1.0856 pp；Black=-3.7309 pp。

Gate A≥+0.50pp；B≥2 strict wins；C两类mean≥−2pp；D curriculum sampleSD≤uniform（ddof1）。
{"Gate_A":false,"Gate_B":false,"Gate_C":false,"Gate_D":true}

实际 successful updates：{"A_curriculum":443,"A_uniform":443,"B_curriculum":443,"B_uniform":442,"C_curriculum":443,"C_uniform":442}

## 解释边界与停止条件

即使全部Gate PASS，也仅支持这一固定 late schedule 的内部 timing premise，不证明最终创新、最优schedule、bank不稳定因果解释或V2 final_eval效果。
任一Gate失败，停止等待人工决策，不搜索warmup/seed/比例。PASS仅允许下一步方案设计，不能自动执行。
official final_eval content/forward=0；generation=0；BootstrapGuard=0；DeepPCB=0；没有新selector/generator/TDCRG/weighting/Stage21C。
