# 论文对比实验

> 本文档对比 `PDF/` 目录下 6 篇论文的方法在 MPDD-AVG 2026 上的表现。
> 持续更新：每完成一个方法（6 个配置）即更新本表并提交推送。

## 实验协议

- **骨干（固定）**：DepFormer 时序编码 + BCT 音视频协同（当前冠军方法）。
- **变量**：向量级融合模块（`fusion_type`）或 CVAE 数据增强。
- **任务**：binary 二分类；**5-fold CV，seed=42**；全部 `--cls_only`（关闭 PHQ 回归头，公平口径）。
- **配置**：Track1(Elder) + Track2(Young) × `A-V+P` / `A-V-G+P` / `G+P` 三个 subtrack。
- **主指标**：Macro-F1（均值±标准差）；次要：Accuracy、Kappa。
- **运行命令**：`python experiments/train_cv.py --track {T} --task binary --subtrack {S} --folds 5 --cls_only [--fusion_type {ft}|--use_cvae] --num_workers 0`

## 论文与方法对应

| 论文 | 方法 | 融合模块 (`fusion_type`) | 状态 |
|---|---|---|---|
| 3746027.3762062 — DepFormer (ACM MM'25) | BCT 跨模态 + 个性化融合 | `cross_fusion`（当前基线） | 已实现 |
| paper02 — CMG-VS (CVPR'26) | CVAE 跨模态引导视觉合成 | `--use_cvae` | 已实现 |
| 3743093.3770965 — Personality-Enhanced (MMAsia'25) | PTMFIM 人格-多模态交互 | `ptmfim` | 已实现 |
| 3746027.3762063 — HOPE (ACM MM'25) | 人格为 Query 的交叉注意力融合 | `hope` | 已实现 |
| 3746027.3762064 — MSF-ATS (ACM MM'25) | 模态可靠性权重融合 | `reliability` | 已实现 |
| paper01 — P3HF (AAAI'26) | 超图高阶跨模态交互 | `hypergraph` | 已实现 |

## 结果对比（Macro-F1，均值±std）

| 方法 | Track1 A-V+P | Track1 A-V-G+P | Track1 G+P | Track2 A-V+P | Track2 A-V-G+P | Track2 G+P |
|---|---|---|---|---|---|---|
| baseline (`cross_fusion`) | 0.6480±0.0832 | 0.5297±0.0832 | 0.4041±0.0083 | 0.5462±0.1512 | 0.5526±0.1494 | 0.6809±0.0736 |
| CMG-VS (`cvae`) | 0.6258±0.0632 | 0.6479±0.0713 | N/A | 0.4986±0.1087 | 0.4994±0.1362 | N/A |
| Personality-Enhanced (`ptmfim`) | 0.5749±0.0774 | 0.5216±0.0535 | 0.4041±0.0083 | 0.5662±0.1383 | 0.5461±0.0812 | 0.6211±0.0603 |
| HOPE (`hope`) | 0.5818±0.0854 | 0.4679±0.0697 | 0.4041±0.0083 | 0.5035±0.1351 | 0.5361±0.1236 | 0.6254±0.0578 |
| MSF-ATS (`reliability`) | 0.5239±0.0961 | 0.5991±0.1144 | 0.4441±0.0851 | 0.5634±0.1079 | 0.5572±0.1156 | 0.6659±0.0692 |
| P3HF (`hypergraph`) | 0.5717±0.0539 | 0.5108±0.0994 | 0.4041±0.0083 | 0.5092±0.1195 | 0.5156±0.1348 | 0.6203±0.0968 |

> ✅ 全部 6 个方法 × 6 个配置（Track1+Track2 × A-V+P/A-V-G+P/G+P）已跑完；数值为 5-fold Macro-F1 均值±标准差。

## 详细结果（F1 / Acc / Kappa）

### baseline (`cross_fusion`) — DepFormer/BCT
- Track1 A-V+P：F1 0.6480±0.0832 / Acc 0.7013±0.0956 / Kappa 0.3271±0.1622
- Track1 A-V-G+P：F1 0.5297±0.0832 / Acc 0.6902 / Kappa 0.1438
- Track1 G+P：F1 0.4041±0.0083 / Acc 0.6784 / Kappa 0.0000
- Track2 A-V+P：F1 0.5462±0.1512 / Acc 0.6124 / Kappa 0.2113
- Track2 A-V-G+P：F1 0.5526±0.1494 / Acc 0.6118 / Kappa 0.2137
- Track2 G+P：F1 0.6809±0.0736 / Acc 0.6824 / Kappa 0.3652

### CMG-VS (`cvae`)
- Track1 A-V+P：F1 0.6258±0.0632 / Acc 0.6895±0.0596 / Kappa 0.2786±0.0999
- Track1 A-V-G+P：F1 0.6479±0.0713 / Acc 0.7248 / Kappa 0.3194
- Track2 A-V+P：F1 0.4986±0.1087 / Acc 0.5209 / Kappa 0.0388
- Track2 A-V-G+P：F1 0.4994±0.1362 / Acc 0.5444 / Kappa 0.0779
- G+P：N/A（无可合成的视觉模态）

> 观察：CVAE 仅在 Track1 A-V-G+P 上显著提升（+11.8pp），其余 A-V 配置反而下降；小样本下波动较大。

### Personality-Enhanced (`ptmfim`)
- Track1 A-V+P：F1 0.5749±0.0774 / Acc 0.6327 / Kappa 0.1812
- Track1 A-V-G+P：F1 0.5216±0.0535 / Acc 0.6209 / Kappa 0.0791
- Track1 G+P：F1 0.4041±0.0083 / Acc 0.6784 / Kappa 0.0000
- Track2 A-V+P：F1 0.5662±0.1383 / Acc 0.6229 / Kappa 0.2384
- Track2 A-V-G+P：F1 0.5461±0.0812 / Acc 0.5784 / Kappa 0.1731
- Track2 G+P：F1 0.6211±0.0603 / Acc 0.6471 / Kappa 0.2851

### HOPE (`hope`)
- Track1 A-V+P：F1 0.5818±0.0854 / Acc 0.6680 / Kappa 0.1877
- Track1 A-V-G+P：F1 0.4679±0.0697 / Acc 0.6562 / Kappa 0.0384
- Track1 G+P：F1 0.4041±0.0083 / Acc 0.6784 / Kappa 0.0000
- Track2 A-V+P：F1 0.5035±0.1351 / Acc 0.5549 / Kappa 0.1223
- Track2 A-V-G+P：F1 0.5361±0.1236 / Acc 0.5784 / Kappa 0.1418
- Track2 G+P：F1 0.6254±0.0578 / Acc 0.6471 / Kappa 0.2909

### MSF-ATS (`reliability`)
- Track1 A-V+P：F1 0.5239±0.0961 / Acc 0.6444 / Kappa 0.1420
- Track1 A-V-G+P：F1 0.5991±0.1144 / Acc 0.7248 / Kappa 0.2671
- Track1 G+P：F1 0.4441±0.0851 / Acc 0.6667 / Kappa 0.0478
- Track2 A-V+P：F1 0.5634±0.1079 / Acc 0.6235 / Kappa 0.2473
- Track2 A-V-G+P：F1 0.5572±0.1156 / Acc 0.6007 / Kappa 0.1915
- Track2 G+P：F1 0.6659±0.0692 / Acc 0.6824 / Kappa 0.3588

### P3HF (`hypergraph`)
- Track1 A-V+P：F1 0.5717±0.0539 / Acc 0.6784 / Kappa 0.1678
- Track1 A-V-G+P：F1 0.5108±0.0994 / Acc 0.6784 / Kappa 0.1170
- Track1 G+P：F1 0.4041±0.0083 / Acc 0.6784 / Kappa 0.0000
- Track2 A-V+P：F1 0.5092±0.1195 / Acc 0.5556 / Kappa 0.1001
- Track2 A-V-G+P：F1 0.5156±0.1348 / Acc 0.5673 / Kappa 0.1168
- Track2 G+P：F1 0.6203±0.0968 / Acc 0.6242 / Kappa 0.2466

## 结论与观察

1. **baseline（DepFormer/BCT 的 cross_fusion）在大多数配置上是最好或接近最好的**：Track1 A-V+P（0.6480）与 Track2 G+P（0.6809）均为全局最优。
2. **没有任何一个新增融合模块能稳定超越 baseline**：ptmfim / hope / hypergraph 在几乎全部 6 个配置上都低于 baseline（多为 -2~-12pp）；reliability 在 4/6 配置有正增益但幅度小（+0.5~+6.9pp），且 Track1 A-V+P 上大幅下降（-12.4pp）。
3. **CMG-VS（cvae）只在 Track1 A-V-G+P 上显著提升（+11.8pp）**，其余 A-V 配置下降；与上一轮「CVAE 无增益」的结论一致，且增益高度依赖配置。
4. **G+P 在 cls_only 口径下普遍 class collapse**（Track1 G+P 的 baseline/各融合模块 Kappa=0），只有 reliability 略微打破（Kappa 0.048）。
5. **局限**：仅 seed=42 单次 5-fold，Elder/Young 各约 87/88 人，方差较大（std 常达 0.1）；结论需多 seed 复跑确认。这些论文多为 MPDD 2025（1s/5s 窗口）冠军方法，其完整贡献（HOPE 的 LSP 需 MER2025、MSF-ATS 的自适应时间窗、P3HF 的超图+域解耦）在 2026 数据上无法逐字复现，这里只迁移了各自的核心融合思想。
