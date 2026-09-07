# MPDD 2025 多模型对比实验框架设计

> 历史设计说明：本文记录 2026-09-06 已批准的原始十模型方案。自 2026-09-07 起，正式实验范围已调整为八模型，DepMamba 与 Proposed 不再注册、运行或参与对比；当前口径以 `docs/CURRENT_STATUS.md` 为准。

## 1. 目标与范围

在不删除原有 Track2、三分类、五分类、scripts、checkpoints、logs 和既有训练/测试能力的前提下，将 MPDD-main 增量改造成统一、可复现、可扩展的多模型对比实验框架。本轮默认范围是 Track1 Elderly Binary Classification，固定比较 MFCC、DenseNet 和 Personality Enabled，不自动运行所有模型的完整 5-Fold 长时训练。

## 2. 已确认的现状

- `dataset.py` 返回 `A_feat`、`V_feat`、`personalized_feat` 和 `emo_label`。音视频单样本经补齐/截断后是 `[T, D]`，DataLoader 后是 `[B, T, D]`；个性化向量缺失时使用 1024 维零向量。
- 受试者 ID 主要从音频特征文件名第一个下划线前提取；旧 Track1 划分中的五分类特殊逻辑会把同一受试者的样本拆到训练集和验证集，存在 subject leakage 风险。
- `models/__init__.py` 已有基于命名约定的动态发现代码，但 `train.py` 和 `test.py` 都直接导入并实例化 `ourModel`，所以动态入口尚未真正启用。
- OurModel 使用音频/视觉 LSTM 编码、线性投影、视觉自注意力增强、双向跨模态反馈、掩码均值池化、1024 维个性化投影和双分类器；原损失为主分类器交叉熵加辅助分类器 Focal Loss。
- 旧 checkpoint 是直接保存的完整模型 `state_dict`，缺少模型名、特征配置、fold 和 seed 元数据。
- 当前 PowerShell 无法保持 `conda activate`，但 `conda run -n dachuangxiangmu` 可强制使用目标环境。该环境有 PyTorch、CUDA、scikit-learn、numpy、pandas 和 scipy；当前缺少 xgboost、matplotlib、mamba_ssm 与 causal_conv1d。
- 当前项目目录不是 Git 仓库，不能创建提交或使用 Git worktree。
- 当前机器未找到 MPDD 2025 数据目录，不能把真实数据读取测试伪报为通过。

## 3. 总体架构

采用兼容优先的两层 Registry：

```text
train.py / test.py / experiments/run_model_cv.py
                    ↓
       experiments/model_registry.py
              ┌─────┴─────┐
              ↓           ↓
        Classical       PyTorch
        SVM / XGB   models/__init__.py
                          ↓
                     create_model()
                          ↓
       MLP / BiLSTM / LightWeightTrans / LMF / MulT /
                OurModel / DepMamba / Proposed
```

`models.create_model()` 是 PyTorch 模型唯一创建入口。`experiments/model_registry.py` 只做类型分派；SVM 和 XGBoost 不继承 `BaseModel`。

## 4. PyTorch 模型接口

新增模型遵循现有 `BaseModel` 约定，并实现 `set_input`、`forward` 和 `optimize_parameters`。所有模型统一暴露 `emo_logits`、`emo_pred` 和 `emo_label`。为避免重复实现输入搬运、交叉熵、优化器和掩码池化，新增一个小型公共音视频分类基类或辅助模块；它只服务于新增 adapter，不改变 OurModel 核心网络。

模型接入顺序严格为 OurModel 兼容、MLP、BiLSTM、LightWeightTrans、SVM/XGBoost、LMF、MulT、统一 CV/Evaluator、run_all、DepMamba、Proposed placeholder。

## 5. 模型设计

- MLP：音频与视觉沿时间维做掩码均值池化，拼接可选个性化向量，经过两层 MLP 输出分类 logits。
- BiLSTM：复用 `models/networks/lstm.py` 的双向能力，分别编码音频和视觉，做掩码池化后与个性化向量晚融合。
- LightWeightTrans：复用 `models/networks/LightWeightTrans.py` 的 `TransEncoder`，adapter 负责输入投影、张量轴顺序、池化和分类。
- LMF：在 `models/networks/lmf.py` 实现低秩多模态融合核，音频、视觉、个性化先形成定长表示再融合。
- MulT：只实现项目需要的音视频 cross-modal attention 和 transformer block；个性化向量只在晚期融合，不复制到每个时间步。
- OurModel：只修改统一入口、设备与 checkpoint 兼容所必需的代码，保留网络主干行为。公平架构比较默认可配置统一交叉熵策略，并在结果中记录损失和类别平衡策略；原 CE+Focal 行为作为 original 策略保留。
- DepMamba：延迟导入可选依赖。缺依赖时只有该模型实例化返回明确 optional-dependency 错误，其他模型不受影响。
- Proposed：只提供合法类与明确的未实现异常，状态为 TODO/N/A，绝不生成虚假结果。

## 6. 数据与 Subject-Level 5-Fold

新增公共数据工具负责从 JSON 条目稳定提取 subject ID，并检查同一受试者是否具有一致标签。`create_splits.py` 先聚合到受试者级别，再用 seed 3407 的 StratifiedKFold 生成 5 份 JSON。每个 fold 保存 train/val ID、类别分布、seed 和 fold 编号。

已存在且完整的 fold 默认复用；只有 `--force` 才覆盖。加载 fold 时必须检查训练与验证 subject ID 交集，为非空则抛出 `Subject leakage detected.`。

当前没有真实 MPDD 2025 数据。实现和合成测试可以继续，但最终真实 Dataset smoke 必须标记 `SKIPPED` 并说明需要 `--data-root`；用户提供数据后才可验证真实维度和类别分布。

## 7. 训练、Checkpoint 与结果流

`run_model_cv.py` 加载模型 JSON 配置并叠加命令行参数，复制最终配置到该次结果目录。Classical 分支把训练集的音频/视觉分别做时间均值并拼接 personality；StandardScaler 只在训练数据上拟合。PyTorch 分支通过 `models.create_model()` 创建模型并使用统一 batch 循环。

新 checkpoint 保存模型名、模型 state_dict、配置、特征配置、fold 和 seed。`test.py` 同时识别新格式和旧纯 `state_dict`；旧格式使用命令行 `--model` 及当前特征配置。

每个 fold 保存 raw prediction、confusion matrix、效率指标和 `raw_results.csv` 行。OOF 汇总生成 overall confusion matrix。未运行不写假数字，状态使用 `FAILED`、`SKIPPED` 或 `N/A`。

## 8. 评估与效率

`evaluator.py` 接收 `y_true`、`y_pred`、`y_prob`，计算 Accuracy、Macro-F1、Weighted-F1、Precision、Recall、Positive Recall、Specificity、ROC-AUC 和 Confusion Matrix；主指标为 Macro-F1。ROC-AUC 遇到单类输入时返回 `N/A` 并带原因，不吞异常。

`efficiency.py` 对 PyTorch 统计参数量、序列化模型大小、10 次 warmup 后 50 次推理延迟和 CUDA peak VRAM。非适用项目返回 `N/A`。

## 9. run_all 容错

`run_all.py` 逐模型启动同一 Python 环境下的 CV 命令，捕获单模型退出状态并继续后续模型。`run_report.json` 记录 model、status、duration、error 和 timestamp。DepMamba 依赖缺失、Proposed 未实现分别映射为 SKIPPED 和 N/A。

## 10. 测试策略

采用阶段化测试驱动：每个阶段先增加会失败的最小测试，再实现并重新运行。测试层次包括 import/registry、合成单 batch forward、evaluator 边界、subject split、classical scaler、run_all 失败隔离，以及在提供真实数据目录后读取一个真实样本。

所有 Python 命令只通过 `conda run -n dachuangxiangmu python ...` 执行。不会自动安装依赖，不会运行完整多模型 5-Fold 长训练。

## 11. 文档与持久上下文

创建根目录 `AGENTS.md`，以及 `docs/PROJECT_CONTEXT.md`、`docs/CURRENT_STATUS.md`、00 至 07 中文文档和 `CHANGELOG_EXPERIMENTS.md`。目标读者是会基础 Python、第一次接触 MPDD 的本科生。每阶段更新 CURRENT_STATUS 和 CHANGELOG；只有稳定架构事实写入 PROJECT_CONTEXT。

OurModel 文档中的 Mermaid 图严格依据源码：音频/视觉 LSTM 编码后投影，视觉经过 VEM，自适应跨模态反馈 CFM，掩码池化后拼接个性化投影，送入主/辅助分类器。

## 12. 依赖策略与已知限制

不覆盖现有 requirements，只追加实际核心、实验与可选依赖分组。xgboost 与 matplotlib 当前缺失，仅写入 requirements 和安装说明；未经用户额外授权不安装。DepMamba 的 Windows 依赖不固定到未经验证的 wheel。

项目不在 Git 中，因此不执行 commit。真实 MPDD 2025 数据当前不可见，因此不生成 fold、不跑真实样本 smoke、不宣称 Dataset PASS。

## 13. 完成标准

- `train.py`、`test.py`、`run_model_cv.py` 对 PyTorch 模型均调用 `models.create_model()`。
- mlp、bilstm、lightweighttrans、lmf、mult、our 能被发现、实例化并完成合成单 batch forward。
- evaluator、subject split、aggregate、efficiency、run_all 的自动化测试通过。
- DepMamba 缺依赖时明确 SKIPPED，Proposed 明确 TODO/N/A。
- 所有规定中文文档与持久上下文文件完成。
- 最终输出实际环境、关键 tree、模型状态、测试证据、已知问题和完整 PowerShell 后续命令。
