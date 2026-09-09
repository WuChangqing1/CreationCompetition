# 当前开发状态

最后更新时间：2026-09-09

## 已完成

- 已新增独立 OurModel 消融模型与统一运行入口；原 `models/our/our_model.py` 未修改，十个变体、五折训练、独立测试和隔离汇总 CSV 的使用方法见 `docs/15_OurModel消融实验使用说明.md`。
- 已完成环境检查、源码分析、持久上下文、设计审批、分阶段实现、代码审查和中文文档。
- 已建立精确模型 Registry、统一数据接口、subject-level 5-Fold、指标与效率统计、checkpoint 元数据、独立失败隔离和结果聚合。
- 已完成自动化测试、合成数据集成测试，以及 2025/2026 真实数据 Smoke Test。
- 正式比较范围已按用户决定收缩为 8 个模型，不再包含 DepMamba 与 Proposed。
- 正式运行入口默认强制 CUDA：六个 PyTorch 模型与 XGBoost 使用 GPU，SVM 保留 CPU；结果新增 `Device` 字段。
- 已接入统一的 2025/2026 数据布局解析，并对两年 Elder/Young 数据验证真实读取与 subject-level 5-Fold。
- 本轮未执行完整多模型 5-Fold 长时间训练，也未伪造正式结果。
- `legacy_bicfnet` 已成为默认实验协议；原 20 轮方案保留为显式 `--protocol modern`。
- 旧版 OurModel 训练恢复 seed 2024、长度 26、batch 8、300 epochs、2e-5 学习率、余弦调度，并按验证集 Macro-F1 保存最佳 epoch。
- 独立测试在旧版协议下同时输出事件级、严格受试者多数投票和历史投票回填事件三种口径。
- 已新增历史 OurModel 单 checkpoint 复现入口，不创建 Fold、不重新训练，直接执行旧版受试者多数投票与事件回填。
- 已用历史 checkpoint 在 2025 Elderly 独立测试集精确复现 `Acc(U)=0.9383`、`F1(U)=0.8759` 和混淆矩阵 `[[187, 0], [14, 26]]`。
- 已新增 2025 旧版单次划分八模型对比入口；OurModel 冻结历史权重并先做结果锁验证，其余七模型使用固定 292/45 划分训练一次。
- 已消除旧 `train_val_split1` 中 `set` 遍历导致的跨进程漂移，冻结 45 条验证事件身份；入口没有 `--folds` 参数。
- 用户已完成2025旧版单次划分八模型实验与2026 subject-aware 五折训练及独立测试；详细结果与方法学审查见 `docs/14_MPDD_2025_2026八模型对比实验报告.md`。
- 2025独立测试中OurModel与BiLSTM并列第一；2026受试者级BiLSTM第一、OurModel第三。

## 当前任务

两年度八模型实验均已完成；OurModel 消融代码已接入，等待用户显式启动正式消融训练。

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| A 环境与源码理解 | PASS | 用户显式激活 `dachuangxiangmu`；2025/2026 数据已定位 |
| B Registry / OurModel | PASS | 精确路由、兼容新旧 checkpoint、人格开关已验证 |
| C 简单 PyTorch 模型 | PASS | MLP/BiLSTM/LightWeightTrans forward 通过 |
| D Classical | PASS | SVM 与 XGBoost Registry/Smoke 均通过 |
| E LMF | PASS | forward 通过 |
| F MulT | PASS | forward 通过 |
| G CV/Evaluator | PASS | 划分防泄漏、缓存指纹、Run_ID、OOF 与指标测试通过 |
| H run_all | PASS | 独立失败隔离和直接脚本入口通过 |
| I 文档与验收 | PASS | 中文文档、变更记录、设计与计划齐全 |

## 模型状态

| 模型 | Registry | Forward / Fit Smoke | 状态 |
|---|---|---|---|
| OurModel | PASS | logits `(2, 2)` | PASS |
| MLP | PASS | logits `(2, 2)` | PASS |
| BiLSTM | PASS | logits `(2, 2)` | PASS |
| LightWeightTrans | PASS | logits `(2, 2)` | PASS |
| LMF | PASS | logits `(2, 2)` | PASS |
| MulT | PASS | logits `(2, 2)` | PASS |
| SVM | PASS | sklearn Pipeline | PASS |
| XGBoost | PASS | XGBClassifier | PASS |

## 验收证据

- `python -m unittest discover -s tests -v`：126 tests，全部通过（2026-09-08）。
- `python experiments\smoke_test.py`：六个 PyTorch 模型、SVM、Evaluator、Subject Split 均 PASS；XGBoost 按实际依赖状态报告。
- Dataset：2025/2026 的 Elder/Young 四套训练数据真实读取及 5-Fold 检查均 PASS。
- tiny MLP fold：2025 Elder 与 2026 Elder 均完成真实数据 5-Fold 单 batch 闭环。

## Known Issues

- 用户当前 PowerShell 可以显式执行 `conda activate dachuangxiangmu`；用户命令直接使用该环境中的 `python`。
- 完整多模型正式训练尚未执行；tiny 结果仅用于接口验证，不能作为正式成绩。
- 旧历史日志未保存 train/validation 逐样本清单；冻结划分依据历史 292/45 规模、28/17 验证标签分布及现有 checkpoint 诊断恢复，不能声称还原了未留档基线的历史逐样本预测。
- 2025旧版单次划分中受试者64同时出现在训练和验证集；独立测试仍无训练受试者交叉，但验证选模存在潜在乐观偏倚。
- 2026当前checkpoint策略不完全一致：OurModel按验证Macro-F1选最佳轮次，其他五个神经基线保存第300轮；当前表格应视为已完成结果，投稿前需要统一策略复验。
- GPU 强制运行及旧版单次划分入口均已完成自动化测试；本轮未自动启动七基线完整 300-epoch 长训练。
- 项目已建立 Git 仓库，后续改动统一进入 `GuoChuang` 分支。
- matplotlib、seaborn 当前缺失；XGBoost 3.2.0 已安装并通过创建 Smoke Test。
- 新多模型 CV 入口当前限定二分类；旧 `train.py` / `test.py` 仍保留 3 类、5 类兼容路径。
- `modern` 兼容协议仍保存最终轮 checkpoint；默认 `legacy_bicfnet` 的 OurModel 保存验证集 Macro-F1 最优轮次。

## 下一步

历史 OurModel 精确复现见 `docs/12_历史OurModel单模型复现.md`；旧版八模型单次划分命令与论文使用边界见 `docs/13_旧版八模型单次划分对比实验.md`。

最终验证（2026-09-08）：在 `dachuangxiangmu` 执行 `python -m unittest discover -s tests -v`，126 项通过；旧版对比入口另以 `--models our` 完成 CUDA 端到端历史结果锁验证。未执行七基线完整 300-epoch 训练。

2026-09-08：用户已报告两年八模型 CV 全部 PASS。当前新增独立测试入口，命令及输出位置见 `docs/09_八模型独立测试集评估.md`。独立测试正式运行由用户启动；早期 tiny/训练待运行状态仅是历史记录。

2026 personality ID 问题已通过独立 subject-aware Dataset 模式修复，原 Dataset 保留。修复模式需要重新运行2026五折训练，输出到 `experiments/results_subject_aware_2026`；尚未启动该长时间训练。

按 `docs/08_2025与2026数据集使用说明.md` 选择年份和人群，先跑八模型 Smoke Test，再由用户显式启动正式 5-Fold 实验。
