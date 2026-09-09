# MPDD-main 稳定技术上下文

## 原始目录与职责

- `dataset.py`：从预提取 `.npy` 特征和个性化 embedding 构造样本。
- `feature_extraction/`：离线生成音频、视觉、个性化特征，不参与训练时 backbone 更新。
- `models/networks/`：LSTM、分类器、注意力等网络组件，多数不是可直接训练的完整模型。
- `models/our/our_model.py`：原 MPDD-specific BiCFNet 完整模型。
- `train.py` / `test.py`：原训练与测试入口；改造前均直接创建 `ourModel`。
- `scripts/`：Track1/Track2 旧 Shell 命令。
- `checkpoints/`、`logs/`：历史模型与日志，必须保留。

## Dataset 实际接口

`AudioVisualDataset.__getitem__()` 返回：

```python
{
    "A_feat": audio_feature,
    "V_feat": video_feature,
    "emo_label": label,
    "personalized_feat": personalized_feature,
}
```

音视频 `.npy` 原始形状是 `[time, feature_dim]`，经 `pad_or_truncate` 后是 `[T, Da]` / `[T, Dv]`，DataLoader 后是 `[B,T,Da]` / `[B,T,Dv]`。个性化 embedding 由 `id -> embedding` 字典读取；缺失时原代码返回 1024 维零向量。真实数据目录本轮不可见，实际 Da、Dv 和样本分布尚未验证。

受试者 ID 在 Dataset 中从 `audio_feature_path` 的文件名第一个下划线前提取；旧 split 的 Track1 五分类特殊分支可能把一个受试者拆到训练与验证两侧。

## 数据根目录

- 统一数据根：`D:\Files\Works\CreationProject\test`。
- 2025：`2025\MPDD-Elderly`、`2025\MPDD-Young`、`2025\MPDD-Test`。
- 2026：`2026\MPDD-AVG2026-trainval\{Elder,Young}` 与 `2026\MPDD-AVG2026-test\{Elder,Young}`。
- `experiments/dataset_layouts.py` 将两届目录和标签格式转换为统一的 A/V/P/label manifest。

## OurModel 真实结构

音频/视觉分别经 `LSTMEncoder`，投影到 `hidden_size`；视觉再经 `LightweightVEM` 自注意力，音视频经 `BiCFNetCrossFeedbackModule` 双向门控反馈。两路做掩码均值池化，与 1024 维 personality 的线性投影拼接，送入主分类器 EmoC 和辅助分类器 EmoCF。训练损失为主 CE 加辅助 Focal Loss；`emo_pred` 是主 logits 的 softmax。

## BaseModel

提供 setup、train/eval、test、loss 查询、网络保存/加载、学习率更新和梯度开关。完整模型需实现 set_input、forward、optimize_parameters。原实现的 device 与 nn.Module 关系较松散，新 adapter 必须通过实际参数设备搬运输入。

## 原 Registry 与调用链

`models/__init__.py` 按 `<name>_model.py` 和 `<Name>Model` 命名发现 BaseModel 子类，并提供 `create_model(opt)`。改造前 train/test 均直接 `from models.our.our_model import ourModel`，没有使用该入口。

## 原训练、测试和 checkpoint

train 读取 JSON，调用旧 subject-aware holdout，创建 DataLoader，按 Macro-F1 选择最佳模型，但只保存 `model.state_dict()`。test 为每个 checkpoint 固定创建 OurModel，支持概率加权 ensemble 和 subject majority voting。新格式需要元数据，测试端必须继续接受旧纯 state_dict。

## 新实验架构

OurModel 消融采用新增 `models/ourablation_model.py` 继承原 `ourModel`，原模型文件保持不变。`experiments/run_ablation.py` 按旧版单次划分训练并立即执行独立测试，不使用 5-Fold：2025 使用冻结 292/45，2026 使用固定 seed 的受试者级分层 90/10。每个变体使用隔离目录并生成带 `Ablation` 列的 `ablation_results.csv`。`full` 直接调用原始 forward；其余变体通过旁路 VEM/CFM、关闭单向反馈、人格槽位置零、Focal 权重置零或禁用模态编码器实现。

历史 OurModel 精确复现入口为 `experiments/run_legacy_ourmodel.py`。它只接受一个历史 checkpoint，固定 2025 Track1 Elderly 二分类、1s、MFCC、DenseNet、长度 26、batch 8，并调用旧 `test.py` 的受试者多数投票后事件回填口径；不生成或读取 Fold。每次输出使用独立时间戳目录，并在 `metrics.json` 中记录指标、混淆矩阵和输入来源。

旧版八模型单次划分入口为 `experiments/run_legacy_comparison.py`。它固定 seed 2024、292/45、长度 26、batch 8、300 epochs，不接受 `--folds`。旧 `train_val_split1` 的 `set` 遍历会导致跨进程划分漂移，因此 `experiments/legacy_comparison.py` 冻结 45 条验证事件身份；OurModel 只评估历史 checkpoint 且先校验 187/0/14/26，七个基线再在同一冻结划分上各训练一次。

独立测试评估入口：`experiments/run_independent_test.py`，使用已完成 CV 的五折概率集成，输出独立的 `experiments/results/independent_test_results.csv`。使用方法见 `docs/09_八模型独立测试集评估.md`。用户于 2026-09-08 要求取消新增哈希验证、合并为最终一次测试；独立测试保留直接来源和样本检查。

2026 personality 修复采用新增 `experiments/subject_aware_dataset.py`：显式模式从 manifest 的 `subject_id` 取人格向量，原 `dataset.py` 和默认 `filename` 模式保留。修复模式写入 Run_ID、配置及 checkpoint 元数据，并使用 `experiments/results_subject_aware_2026`，需要重新训练2026模型。

- PyTorch：`train.py/test.py/run_model_cv.py -> models.create_model()`。
- 整体：`experiments/model_registry.py` 区分 classical 与 torch。
- 完整模型：MLP、BiLSTM、LightWeightTrans、LMF、MulT、OurModel。
- Fold：`experiments/create_splits.py` 在 subject 级先聚合标签，再 StratifiedKFold；复用 JSON 并强制检查交集。
- 结果：evaluator 统一指标与 raw predictions；run_model_cv 追加 raw_results；aggregate 输出 mean±sample std；run_all 隔离单模型失败。
- 正式比较模型固定为 SVM、XGBoost、MLP、BiLSTM、LightWeightTrans、LMF、MulT、OurModel；DepMamba 与 Proposed 已退出实验范围。
- XGBoost 3.2.0 当前已安装；若其他机器缺包，只对该模型报告 SKIPPED，不影响其余七个模型。
