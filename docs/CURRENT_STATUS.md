# 当前开发状态

最后更新时间：2026-09-07

## 已完成

- 已完成环境检查、源码分析、持久上下文、设计审批、分阶段实现、代码审查和中文文档。
- 已建立精确模型 Registry、统一数据接口、subject-level 5-Fold、指标与效率统计、checkpoint 元数据、独立失败隔离和结果聚合。
- 已完成 52 项自动化测试、合成数据集成测试，以及 2025/2026 真实数据 Smoke Test。
- 已接入统一的 2025/2026 数据布局解析，并对两年 Elder/Young 数据验证真实读取与 subject-level 5-Fold。
- 本轮未执行完整多模型 5-Fold 长时间训练，也未伪造正式结果。

## 当前任务

阶段 J：已完成；真实数据已验证，等待用户决定何时启动正式实验。

## 阶段状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| A 环境与源码理解 | PASS | 使用 `conda run -n dachuangxiangmu` 强制隔离；2025/2026 数据已定位 |
| B Registry / OurModel | PASS | 精确路由、兼容新旧 checkpoint、人格开关已验证 |
| C 简单 PyTorch 模型 | PASS | MLP/BiLSTM/LightWeightTrans forward 通过 |
| D Classical | PARTIAL | SVM PASS；XGBoost 因缺依赖 SKIPPED |
| E LMF | PASS | forward 通过 |
| F MulT | PASS | forward 通过 |
| G CV/Evaluator | PASS | 划分防泄漏、缓存指纹、Run_ID、OOF 与指标测试通过 |
| H run_all | PASS | 独立失败隔离和直接脚本入口通过 |
| I DepMamba | SKIPPED | 缺少 `mamba_ssm` 与 `causal_conv1d` |
| J 文档与验收 | PASS | 00-07 中文文档、变更记录、设计与计划齐全 |

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
| XGBoost | PASS | 未运行 | SKIPPED：缺依赖 |
| DepMamba | PASS | 未运行 | SKIPPED：缺可选依赖 |
| Proposed | PASS | 未实现 | N/A：仅保留显式占位 |

## 验收证据

- `python -m unittest discover -s tests -v`：52 tests，全部通过。
- `python experiments\smoke_test.py`：环境、六个 PyTorch 模型、SVM、Evaluator、Subject Split 均 PASS。
- Dataset：2025/2026 的 Elder/Young 四套训练数据真实读取及 5-Fold 检查均 PASS。
- tiny MLP fold：2025 Elder 与 2026 Elder 均完成真实数据 5-Fold 单 batch 闭环。

## Known Issues

- 当前 PowerShell 未执行 conda init，直接 `conda activate dachuangxiangmu` 失败；Agent 必须使用 `conda run -n dachuangxiangmu`。
- 完整多模型正式训练尚未执行；tiny 结果仅用于接口验证，不能作为正式成绩。
- 项目不是 Git 仓库，无法 commit、worktree 或提供 Git diff。
- xgboost、matplotlib、seaborn、mamba_ssm、causal_conv1d 当前缺失。
- DepMamba 目前只有依赖检查与显式占位；即使补齐依赖，真实模型 adapter 仍需实现和验证。
- 新多模型 CV 入口当前限定二分类；旧 `train.py` / `test.py` 仍保留 3 类、5 类兼容路径。
- 神经网络 CV 使用固定 epochs 的最终轮 checkpoint，不进行 best-validation epoch 选择。

## 下一步

按 `docs/08_2025与2026数据集使用说明.md` 选择年份和人群，先跑 Smoke Test，再由用户显式启动正式 5-Fold 实验。
