# CreationCompetition

MPDD-AVG 2026 多模态抑郁症检测竞赛项目。

## 竞赛背景

MPDD-AVG 2026 (Multimodal Personality-aware Depression Detection — Audio, Video, Gait) 是一个多模态抑郁症检测挑战赛。本项目的目标是通过音频、视频、步态(IMU)和人格特征，对老年(Track1)和青年(Track2)群体进行抑郁症自动检测。

## 赛道与任务

| 维度 | 选项 |
|------|------|
| 赛道 (Track) | Track1 (Elder 老年) · Track2 (Young 青年) |
| 任务 (Task) | 二分类 (binary, label2) · 三分类 (ternary, label3) · PHQ-9 回归 |
| 子赛道 (Subtrack) | A-V+P · A-V-G+P · G+P |
| 编码器 (Encoder) | bilstm_mean · hybrid_attn · depformer |

二分类和三分类任务默认**联合训练** PHQ-9 回归头，所以每个实验同时输出分类指标和回归指标。

当前实验分支补充了 DepFormer、BCT 和 CVAE 数据增强。CVAE 模式下会关闭 PHQ 回归头，因此这类结果主要看 Macro-F1、Acc、Kappa，不应把日志中的分类派生 CCC/RMSE/MAE 当成 PHQ-9 回归指标。

## 模态与特征

| 模态 | 特征 | Elder 维度 | Young 维度 |
|------|------|-----------|------------|
| Audio | wav2vec | 768 | 1024 |
| Audio | MFCC | 64 | 64 |
| Audio | OpenSmile | 65 | 65 |
| Video | ResNet | 1000 | 1000 |
| Video | DenseNet | 1000 | 1000 |
| Video | OpenFace | 710 | 710 |
| Gait | IMU | 12 | 12 |
| Personality | RoBERTa embeddings | 1024 | 1024 |

## 评估指标 (6个)

| 指标 | 来源 | 用途 |
|------|------|------|
| Macro-F1 | 分类头 | **选模依据**(分类任务) |
| ACC | 分类头 | 准确率 |
| Kappa | 分类头 | Cohen's Kappa |
| CCC | PHQ-9 回归头 | 一致性相关系数(**选模依据**-回归任务) |
| RMSE | PHQ-9 回归头 | 均方根误差 |
| MAE | PHQ-9 回归头 | 平均绝对误差 |

## 项目结构

```
CreationProject/
├── docs/                    # 项目文档
├── test/
│   ├── MPDD-AVG-2026-main/  # 官方基线代码 (MPDD-AVG 2026)
│   │   ├── train.py         # 训练入口
│   │   ├── test.py          # 测试/评估入口
│   │   ├── dataset.py       # 数据加载与预处理
│   │   ├── metrics.py       # 评估指标 (F1, CCC, RMSE等)
│   │   ├── train_val_split.py # 训练/验证集划分
│   │   ├── config.json      # 默认配置
│   │   ├── models/          # 模型实现
│   │   ├── scripts/         # 训练/测试脚本 (Track1/Track2 × 子赛道 × 任务)
│   │   ├── feature_extract/ # 特征提取工具
│   │   └── make_submission_forcodabench/ # CodaBench 提交工具
│   ├── MPDD-main/           # 另一版参考实现 (MPDD 2025)
│   ├── MPDD-Elderly/        # 老年组数据和特征
│   ├── MPDD-Young/          # 青年组数据和特征
│   ├── MPDD-Test/           # 测试集
│   ├── Elder/               # 老年数据 (特征文件)
│   └── Young/               # 青年数据 (特征文件)
├── PDF/                     # 论文与报告
├── .gitignore
└── CLAUDE.md                # Claude Code 配置
```

## 快速链接

- [官方网站](https://hacilab.github.io/MPDD-AVG-2026.github.io/index.html)
- [基线代码](https://github.com/hacilab/MPDD-AVG-2026)
- [数据集 (HuggingFace)](https://huggingface.co/datasets/chasonfff/MPDD-AVG-2026/tree/main)
- [CodaBench 提交平台](https://www.codabench.org/)
- 论文: Fu, C. et al. "The First MPDD Challenge: Multimodal Personality-aware Depression Detection." ACM Multimedia 2025, pp. 13924-13929.
