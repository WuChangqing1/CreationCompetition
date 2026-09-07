# MPDD-main 开发规则

## 项目目标

构建 MPDD 2025 Elderly Track1 二分类多模型公平对比框架，同时保留 Track2、三分类、五分类与原训练/测试能力。

## 唯一环境

- 只允许 Conda 环境 `dachuangxiangmu`。
- Agent 命令统一使用 `conda run -n dachuangxiangmu python ...`。
- 不创建新环境，不修改 base，不使用系统 Python 或裸 `pip`。

## 核心架构

- PyTorch 模型唯一创建入口：`models.create_model(opt)`。
- SVM/XGBoost 由 `experiments/model_registry.py` 分派，不继承 `BaseModel`。
- 模型输入键为 `A_feat`、`V_feat`、`personalized_feat`、`emo_label`。
- 新 checkpoint 必须保存模型名、state_dict、配置、特征、fold 与 seed，并兼容旧 state_dict。

## 公平性与质量

- 主比较固定 MFCC + DenseNet + Personality Enabled，默认 seed 3407。
- Fold 必须按 subject 分层，训练/验证 subject 交集必须为空。
- 主指标为 Macro-F1；损失与类别平衡策略必须写入结果。
- 不写死数据路径/GPU，不在 import 时训练，不吞异常，不伪造结果。
- DepMamba 缺依赖时 SKIPPED；Proposed 未实现时 N/A。
- 不自动运行完整多模型 5-Fold 长训练。

## 文档读取顺序

1. `AGENTS.md`
2. `docs/CURRENT_STATUS.md`
3. `docs/PROJECT_CONTEXT.md`
4. 当前任务涉及的正式文档与源码

## 完成标准

Registry、单 batch forward、subject split、evaluator 与 smoke test 有实际自动化证据；全部中文文档和状态文件同步更新。

## Git 工作流

- 远端仓库：`https://github.com/WuChangqing1/CreationCompetition.git`。
- 本项目后续代码改动统一提交到 `GuoChuang` 分支。
- 数据集、checkpoint、日志和实验输出不得提交到 Git。
