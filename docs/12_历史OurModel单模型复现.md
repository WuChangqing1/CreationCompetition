# 历史 OurModel 单模型精确复现

## 复现目标

本入口专门复现原负责人提供的 MPDD 2025 Track1 Elderly 二分类结果：

```text
Acc(W)=0.8250  Acc(U)=0.9383  F1(W)=0.9329  F1(U)=0.8759
Confusion Matrix = [[187, 0], [14, 26]]
```

它不是八模型 5-Fold 入口，也不会创建、读取或集成任何 Fold checkpoint。

## 固定实验口径

- 模型：旧版 OurModel / BiCFNet。
- checkpoint 数量：1。
- checkpoint：`best_model_2026-07-29-21.42.51.pth`。
- 数据集：MPDD 2025 独立测试集，Track1 Elderly，共 227 条事件。
- 输入：1s MFCC + DenseNet + Personality。
- 最大序列长度：26。
- batch size：8。
- device：CUDA；入口拒绝回退到 CPU。
- 聚合：先按受试者对事件预测执行多数投票，平票时使用平均概率决胜，再把受试者预测回填到该受试者的全部事件。

## 运行命令

先在 PowerShell 中显式激活唯一环境：

```powershell
conda activate dachuangxiangmu
```

然后在项目根目录执行一行命令：

```powershell
python experiments\run_legacy_ourmodel.py --data-root "D:\Files\Works\CreationProject\test\2025\MPDD-Test\MPDD-Elderly" --checkpoint "D:\CodingData\Competition\GuoChuang\MPDD-main\checkpoints\BiCFNet_1s_2labels_mfccs+densenet\best_model_2026-07-29-21.42.51.pth" --device cuda
```

命令结束时必须打印：

```text
[PASS] 历史 OurModel 结果已完全复现
Accuracy=0.9383
Macro-F1=0.8759
Confusion Matrix=[[187, 0], [14, 26]]
```

如果实际矩阵或指标不同，入口会抛出“历史结果未复现”，不会把不同结果标记为成功。

## 输出位置

每次运行都会创建新目录，不覆盖以前结果：

```text
experiments/results_legacy_ourmodel_2025/<运行时间>/
├── run_config.json
├── metrics.json
└── submission.csv
```

本次已验证结果位于：

```text
experiments/results_legacy_ourmodel_2025/2026-09-08-22.31.18/
```

`run_config.json` 明确记录 `cross_validation=false`、`folds=0`、`checkpoint_count=1` 和 CUDA；`metrics.json` 保存全部指标、混淆矩阵、标签文件、预测文件和 checkpoint 来源。

## 与五折实验的区别

历史 `93.83%` 是单个既有 checkpoint 在独立测试集上经过受试者投票后得到的事件级 Accuracy。八模型框架中的 5-Fold 数字是训练数据上的跨受试者验证结果。两者数据范围、checkpoint 数量及统计目标不同，论文中必须分开命名和报告。

建议论文将本结果标记为“历史 OurModel 独立测试复现”，不要把它伪装成五折均值，也不要用2025 checkpoint直接报告2026结果。
