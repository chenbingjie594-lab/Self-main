# Stage21B — Train-only Synthetic Exposure Timing Premise Test

状态：`STAGE21B_CONDITIONING_BACKGROUND_DEV_OVERLAP`。入口审计完成，六个 detector **均未启动**；这不是 premise PASS/FAIL。

## 固定划分及资产

仅 V2 TRAIN136。namespace `Plastic_Bomo_Stage21B_TrainOnly`、seed2101；134 个已知来源组按 SHA256 排序，前 floor(80%×134)=107 组训练，其余27组内部评估。
internal_train=109，internal_dev=27；无重平衡或改 seed。
按 donor-only 规则保留 65 个 slot：flash=34，black=31。三 bank 相同 IDs；其余15个 donor-dev slot 三 bank 一起排除。

## 需要先解决的来源冲突

4 个 donor 合格 slot 的 conditioning normal 与 internal_dev detector 图属于同一冻结 known source group。
此证据来自 Stage20A normal/source-group registry 与 Stage20B frozen_slot；未读取官方 final_eval 图、标签或指标。
不能将 donor-only 隔离写成完整的 detector train/dev 来源隔离。未擅自扩展排除、重新生成、更换背景或修改 split。

| slot_id | class | normal/dev group | internal_dev relative |
|---|---|---|---|
| `black_ann_071f815d8a58e681f5a2cae5ad368b34c8aeb19bb9966083bc83a34f3e96daa1` | black | `sg_b1ea54cd7bd2961d3b811821aa101c34f31d91b92635071e600a8fa726cbe1ee` | images/train/img_879_pre_part_1.jpg |
| `black_ann_37dc244042f3f61f90f6768ebe1ad7324feaaa2e48025b1c61f65ab2b7dfadb4` | black | `sg_9ab2e7ea3484b66031718d67d5cd7ae5164cd731cd5b47dd19574badb851c7b8` | images/train/img_9504_pre_part_1.jpg |
| `flash_ann_7d22d7b137469454a63acb1fd7a702eff9aed47b127143a422122fec43183c07` | flash | `sg_34da09f89e4b9b4fabd073f0dcb4651fea51fbaeae1d36d4d36017f64e96e571` | images/train/img_2500_pre_part_1.jpg |
| `flash_ann_a035013bf058ed34572388912740a76fcd6a63eae3cea0ed5d9bd9fa294f7bb8` | flash | `sg_195d914b24b2e69bc1a9b64a2a8fdc77a106f0cb2f485576fe9ab25ea4437ea2` | images/train/img_3598_pre_part_0.jpg |

需要新的明确授权：是否在三个 bank 同步增加 **conditioning group 不得属于 internal_dev** 的纯泄漏排除规则？
如果授权，仅这4个 slot 额外排除，预计 flash=32、black=29，仍≥20；目前**没有执行这项额外排除**。
不调整现有 split/seed，不补样本，不按质量删样本。批准后应建立新的 allowlist/schedule 版本，保留本次审计。

## 另一个解释限制

原 frozen generator 已使用其中 33 个 internal_dev 所属 TRAIN 输入（按生成器 annotation 输入计数）：{'black': 23, 'flash': 10}。
因此未来只能声称固定 generator 条件下的 detector 内部留出 timing premise；不能声称完整生成—检测管线均未见过 internal_dev。此共同历史暴露不能通过仅排除 donor 消除。

## 当前 donor-only schedule 数学审计（禁止执行）

每 epoch 174 draws，150 epochs；每个 real/synthetic role 均150次；每 arm real=16350、synthetic=9750、total=26100。
Uniform 与 Curriculum occurrence multiset 相同；前50 epochs synthetic=0，其余100 epochs分摊。每 epoch accumulation=[64,64,46]，共 450 attempts。
paired augmentation seed 仅依赖 role identity、occurrence index、seed42；不依赖 epoch/position。仅种子数学配对已审计，实际 augmentation tensor 配对与 GPU 运行均 NOT_RUN。
保持 Stage20B detector 超参数快照。随内部训练集变小，draws/epoch、tail-group 长度与原216-role协议不同；这是划分的必要算术结果，不是超参数搜索。
实际 successful AMP updates 尚未知；不通过追加 steps 强行匹配 compute。

## 未执行与输出边界

没有六 arm metrics/headroom/classwise/variance 文件：没有训练或内部评估，不能伪造结果。
Gate_A/B/C/D=NOT_RUN；所有 delta/std=null。未实现/启动服务器训练 runtime，不能用本目录 schedule 直接开训。
official final_eval content/forward=0；internal_dev forward=0；generation=0；detector training=0；optimizer steps=0；BootstrapGuard=0；DeepPCB=0。
Stage20A/Stage20B 历史文件未修改。未自动继续 Stage21C。
