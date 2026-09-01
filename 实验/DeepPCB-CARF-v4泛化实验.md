# DeepPCB 上的 CARF v4 泛化实验

## 目的

验证 CARF v4 的局部质量监督是否能够从自建塑料薄膜数据泛化到 PCB 微小缺陷。CARF v4 和 Baseline 都从同一原始 SD2 inpainting 权重出发，每个缺陷类别训练 2000 steps，不使用自建数据集权重。

## 公平设置

- 数据：DeepPCB 官方训练划分转换得到的 MVTec/RDA 格式。
- 类别：`mousebite open pinhole short spur spurious_copper`。
- 起点：相同 SD2 inpainting 权重。
- 每类训练：2000 steps，batch size 1，梯度累积 4。
- 推理：相同 normal、粗 Mask、seed、50 个去噪步骤和 guidance scale 7.5。
- 对照：DeepPCB Baseline 与 DeepPCB CARF v4；旧 CARF v2 仅作为版本演进参考。

## 评价

优先报告已有经典指标：KID、IC-LPIPS、Local-KID、BG-LPIPS。正式泛化结论至少要求 CARF v4 在六类宏平均上优于 Baseline，并说明退化类别，不能只报告最佳类别。

生成质量验证通过后，再使用固定的真实检测训练/验证划分进行下游检测增益实验。
