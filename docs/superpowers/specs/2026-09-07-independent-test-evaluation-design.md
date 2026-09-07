# MPDD 八模型独立测试集评估设计

## 1. 目标

在现有 2025/2026 subject-level 5-Fold 对比完成后，为八个正式模型增加统一的独立测试集评估入口。评估必须复用训练阶段固定的特征、人格开关、模型配置和随机种子，不使用测试标签调参，不修改 `raw_results.csv`，并将结果写入同目录的新文件：

```text
experiments/results/independent_test_results.csv
```

正式模型固定为：SVM、XGBoost、MLP、BiLSTM、LightWeightTrans、LMF、MulT、OurModel。

## 2. 评估原则

- 2025 与 2026 分开运行、分开识别数据布局，但写入同一个独立测试结果 CSV。
- 每个模型使用同一年份的五个 Fold 形成概率集成，不从五个 Fold 中挑一个“看起来最好”的 checkpoint。
- 六个 PyTorch 模型加载五个已保存 checkpoint，逐模型平均五份类别概率。
- SVM 与 XGBoost 的 CV 阶段没有保存 estimator，因此按原五折训练集合重新拟合五个 estimator，再平均测试概率。
- 测试标签只在全部概率生成后用于一次性计算指标，不能影响训练、Fold 选择、阈值或权重。
- 五个 Fold 采用等权平均，避免利用测试集或单 Fold 指标确定集成权重。
- 六个 PyTorch 模型与 XGBoost 使用 CUDA；SVM 固定使用 CPU。

## 3. 新入口与调用关系

新增：

```text
experiments/run_independent_test.py
```

调用关系：

```text
命令行参数
  -> 定位 raw_results.csv 中对应年份的八个 Run_ID
  -> 解析 trainval 与独立测试集
  -> PyTorch: 加载 5 Fold checkpoint 并推理
  -> Classical: 按 5 Fold 重新拟合并推理
  -> 五折概率等权平均
  -> 事件级指标
  -> 受试者级概率平均与指标
  -> 原子更新 independent_test_results.csv
  -> 保存独立测试 predictions 与 confusion matrix
```

## 4. 数据布局

### 2025 Elder

训练数据：

```text
D:/Files/Works/CreationProject/test/2025/MPDD-Elderly/Training
```

测试特征与 manifest：

```text
D:/Files/Works/CreationProject/test/2025/MPDD-Test/MPDD-Elderly
```

测试真实标签：

```text
D:/Files/Works/CreationProject/test/2025/MPDD-Test/MM2025_Track1_Elderly.json
```

真实标签按 `test_id` 与测试 manifest 的特征文件 stem 对齐。

### 2026 Elder

训练数据：

```text
D:/Files/Works/CreationProject/test/2026/MPDD-AVG2026-trainval/Elder
```

测试特征与标签：

```text
D:/Files/Works/CreationProject/test/2026/MPDD-AVG2026-test/Elder
```

`split_labels_test.csv` 按受试者 ID 提供标签；同一受试者的事件继承该标签。音频 `mfccs` 参数继续映射到 2026 的 `mfcc` 目录，视觉使用 `densenet`。

## 5. Run 与 checkpoint 选择

`raw_results.csv` 是 CV 运行的唯一索引。脚本按以下完整条件筛选：

- DatasetYear
- Cohort
- Model
- Track
- Task
- AudioFeature
- VideoFeature
- UsePersonality
- SplitWindow
- Seed
- Status=PASS

每个模型必须得到唯一 Run_ID，并且 Fold 必须恰好为 1..5。缺 Fold、重复 Run_ID 或实验条件不一致时立即失败，不进行模糊匹配。

PyTorch checkpoint 路径由 Run_ID 精确解析：

```text
experiments/results/raw/<Run_ID>/<model>/fold_<n>/checkpoint.pth
```

加载时校验 checkpoint 的模型名、年份、cohort、特征、personality、fold 与 seed。任何不一致均停止该模型评估并报告明确错误。

## 6. 模型推理

### PyTorch 六模型

每个 Fold 使用 checkpoint 内保存的最终配置创建模型，加载 state dict，移动到 `cuda`，使用 `eval()` 与 `torch.no_grad()` 推理。每个 Fold 生成 `[N,2]` 概率，五份概率逐元素平均。

### SVM 与 XGBoost

读取 CV 阶段保存的同一组 subject folds。对每个 Fold：

1. 仅使用该 Fold 的训练 subject 构造训练数据。
2. 对 A/V 做与 CV 相同的时间均值池化并拼接 personality。
3. 创建对应 estimator 并拟合。
4. 对独立测试数据调用 `predict_proba()`。

五个 Fold 的测试概率等权平均。SVM 的 scaler 继续位于训练 Fold 的 Pipeline 内，防止泄漏。

## 7. 事件级与受试者级结果

### 事件级

每个测试事件保留一个真实标签和平均概率，预测类别为概率最大值。2026 的受试者标签扩展到该受试者的全部事件。

### 受试者级

同一受试者的事件概率先取均值，再以概率最大值确定一个受试者预测。真实标签必须在该受试者内一致，否则立即失败。

两种层级分别计算指标，不把事件行与受试者行混在一个统计量中。

## 8. 输出文件

主结果：

```text
experiments/results/independent_test_results.csv
```

每个年份、模型、层级一行。字段固定为：

```text
DatasetYear,Cohort,Model,EvaluationLevel,Samples,
Accuracy,Macro_F1,Weighted_F1,Precision,Recall,
Positive_Recall,Specificity,ROC_AUC,TN,FP,FN,TP,
EnsembleFolds,Run_ID,Device,Seed,Status
```

预测明细：

```text
experiments/results/independent_test_predictions/
  <year>_<cohort>_<model>_event.csv
  <year>_<cohort>_<model>_subject.csv
```

混淆矩阵：

```text
experiments/results/independent_test_confusion_matrix/
  <year>_<cohort>_<model>_event.csv-or-png
  <year>_<cohort>_<model>_subject.csv-or-png
```

不得写入或修改：

```text
experiments/results/raw_results.csv
experiments/results/summary.csv
```

## 9. 不覆盖与重跑规则

写入主结果时先读取已有 CSV，以 `(DatasetYear, Cohort, Model, EvaluationLevel)` 为唯一键：

- 新键追加。
- 相同键重跑时原子替换对应行。
- 其他年份、cohort、模型和层级全部保留。
- 先写 `.tmp`，成功后替换目标文件，避免中途中断破坏已有结果。

预测明细和混淆矩阵使用包含年份、cohort、模型、层级的唯一文件名，不触碰旧 CV 文件。

## 10. 错误处理

- CUDA 不可用：除 SVM 外立即报错，不退回 CPU。
- 找不到唯一 Run_ID：报出筛选条件和候选 Run_ID。
- Fold 不完整：报出缺失 Fold。
- checkpoint 元数据不匹配：拒绝加载。
- 测试特征缺失或 A/V 事件无法配对：报出具体 subject 与文件。
- 标签无法对齐或同一 subject 标签冲突：停止评估，不跳过样本。
- 单一类别导致 ROC-AUC 不可计算：记录 `N/A`，其余指标照常计算。
- 单模型失败时不伪造结果；批量入口继续后续模型，并在该模型结果中记录 FAILED 与错误摘要。

## 11. 命令接口

用户先显式执行：

```powershell
conda activate dachuangxiangmu
cd D:\CodingData\Competition\GuoChuang\MPDD-main
```

2025 与 2026 分别运行，公共参数包括：

```text
--models
--dataset-year
--cohort
--data-root
--cv-results
--output
--audio-feature
--video-feature
--use-personality
--split-window
--folds
--seed
--device
```

默认输出为 `experiments/results/independent_test_results.csv`，但正式命令仍显式传入该路径，便于审计。

## 12. 测试与验收

自动化测试至少覆盖：

- 2025 测试 manifest 与真实标签精确对齐。
- 2026 测试事件与 subject 标签精确对齐。
- CV 条件只能解析出唯一 Run_ID 和完整五折。
- 五折概率采用等权平均。
- subject 概率聚合正确。
- 独立测试结果写入不修改 `raw_results.csv`。
- 先写2025再写2026时，两届结果同时保留。
- 重跑同一键只替换对应行。
- SVM/XGBoost 使用训练 Fold 重新拟合。
- PyTorch checkpoint 元数据不匹配时拒绝评估。

Smoke Test 使用合成数据或临时目录，不读取和写入正式结果文件。除非用户明确下令，不自动执行完整八模型独立测试。

## 13. 明确不做

- 不根据独立测试指标重新选择模型、阈值、Fold 或集成权重。
- 不把独立测试结果写回 CV 的 `raw_results.csv`。
- 不覆盖旧 `answer_Track1/submission.csv`。
- 不使用旧负责人模型替代本次五折产生的模型。
- 不自动启动长时间独立测试。
