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
| CMG-VS (`cvae`) | 0.6258±0.0632 | ⏳ | N/A | ⏳ | ⏳ | N/A |
| Personality-Enhanced (`ptmfim`) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| HOPE (`hope`) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| MSF-ATS (`reliability`) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| P3HF (`hypergraph`) | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

> ⏳ = 待跑；✅ 表示已完成。数值为 5-fold Macro-F1 均值±标准差。

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

（其余方法结果在实验完成后追加）
