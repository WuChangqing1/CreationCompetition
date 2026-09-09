# OurModel 旧版单次划分消融实验使用说明

## 1. 协议说明

消融实验不使用 5-Fold。每个变体只训练一次，并在训练结束后直接执行独立测试集评估。

- 2025：使用已经冻结的旧版 292 条训练事件、45 条验证事件；
- 2026：不存在可恢复的历史 292/45 清单，因此使用 seed 2024 生成一次受试者级分层 90/10 划分；
- 特征长度：26；
- Batch size：8；
- Epochs：300；
- Learning rate：2×10⁻⁵；
- Optimizer：AdamW；
- Scheduler：CosineAnnealing；
- 最佳模型：验证集 Macro-F1 最高的 epoch；
- 测试口径：受试者多数投票后回填事件，与旧版 2025 指标口径一致；
- 结果字段 `EnsembleFolds=0`。

## 2. 代码边界

- 原始模型 `models/our/our_model.py` 未修改；
- 消融子类位于 `models/ourablation_model.py`；
- 单次划分入口为 `experiments/run_ablation.py`；
- 命令行没有 `--folds` 参数，也没有单独的 CV/独立测试阶段；
- 每个变体训练完成后立即保存最佳 checkpoint 并执行独立测试。

## 3. 消融变体

```text
full
no_personality
no_vem
no_cfm
no_a2v
no_v2a
no_aux_focal
audio_only
visual_only
personality_only
```

`full` 直接执行原始 OurModel 的 forward。自动化测试确认相同权重和相同输入下，两者 logits 完全一致。

## 4. 2025 完整运行命令

在已经激活 `dachuangxiangmu` 环境后，使用一行命令：

```powershell
python experiments\run_ablation.py --variants "full,no_personality,no_vem,no_cfm,no_a2v,no_v2a,no_aux_focal,audio_only,visual_only,personality_only" --dataset-year 2025 --cohort Elder --track Track1 --task binary --data-root "D:\Files\Works\CreationProject\test\2025" --audio-feature mfccs --video-feature densenet --personality-id-source filename --split-window 1s --seed 2024 --epochs 300 --device cuda
```

2025 默认输出：

```text
experiments/results_ablation_legacy_single_split/2025/<时间戳>/
```

## 5. 2026 完整运行命令

```powershell
python experiments\run_ablation.py --variants "full,no_personality,no_vem,no_cfm,no_a2v,no_v2a,no_aux_focal,audio_only,visual_only,personality_only" --dataset-year 2026 --cohort Elder --track Track1 --task binary --data-root "D:\Files\Works\CreationProject\test\2026" --audio-feature mfccs --video-feature densenet --personality-id-source subject_id --split-window 1s --seed 2024 --epochs 300 --device cuda
```

2026 默认输出：

```text
experiments/results_ablation_legacy_single_split/2026/<时间戳>/
```

## 6. 输出文件

一次运行的目录结构：

```text
<时间戳>/
├── run_config.json
├── ablation_results.csv
├── predictions/
│   ├── full.csv
│   ├── no_vem.csv
│   └── ...
├── full/
│   ├── config.json
│   ├── checkpoint.pth
│   ├── training_history.csv
│   └── result.csv
├── no_vem/
│   └── ...
└── ...
```

最终对比结果位于：

```text
ablation_results.csv
```

该文件含 `Ablation` 列以及 Accuracy、Macro-F1、Weighted-F1、Sensitivity、Specificity、ROC-AUC、TN、FP、FN、TP 等指标。

每次执行默认创建新的时间戳目录，因此不会覆盖已有对比实验或先前消融结果。

## 7. 分批运行

可以先只运行核心结构：

```powershell
python experiments\run_ablation.py --variants "full,no_vem,no_cfm" --dataset-year 2025 --cohort Elder --data-root "D:\Files\Works\CreationProject\test\2025" --personality-id-source filename --seed 2024 --epochs 300 --device cuda
```

每次分批运行都会创建独立时间戳目录。不同运行目录中的结果不能自动合并为同一次实验，应在论文统计阶段按运行配置统一整理。

## 8. 历史 checkpoint 的使用边界

历史混淆矩阵 `[[187,0],[14,26]]` 来自已有完整 OurModel checkpoint，可以继续通过 `experiments/run_legacy_ourmodel.py` 独立复现。

但是历史 checkpoint 不能作为正式消融表中的 `full` 行，而其他消融变体使用新训练 checkpoint。这样会同时改变训练过程和模型结构，无法把性能差异归因于被删除模块。

因此，本入口会对 `full` 和所有消融变体在相同单次划分、相同 seed、相同超参数下分别重新训练一次。新训练的 `full` 结果不保证逐数值等于历史 checkpoint 的 0.9383；历史结果应作为“历史复现结果”单列。

## 9. 2025 与 2026 的论文表述

- 2025 可以表述为历史冻结 292/45 单次划分；
- 2026 应表述为受试者级分层单次留出验证；
- 不应声称 2026 使用了原项目历史划分，因为原项目没有提供该清单；
- 两年均不是交叉验证，不报告五折均值或标准差；
- 主表应明确报告单次运行 seed 2024，并将统计局限性写入实验限制。

## 10. 本轮范围

本轮只修改入口、测试和文档，不自动启动十个变体的 300 epoch 长时间训练。
