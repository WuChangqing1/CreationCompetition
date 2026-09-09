# 实验框架变更记录

## 2026-09-09

- 用户已完成2025旧版单次划分八模型实验，以及2026 subject-aware 五折训练和独立测试。
- 新增 `docs/14_MPDD_2025_2026八模型对比实验报告.md`，系统整理两年度独立测试、2026五折均值与标准差、模型复杂度、类别不平衡和方法学限制。
- 数据审查确认：2025中OurModel与BiLSTM并列第一；2026受试者级BiLSTM第一、OurModel第三。
- 记录投稿级复验阻碍：2025旧版划分存在受试者64训练/验证交叉；2026 OurModel与其他神经基线checkpoint选择策略不一致；`raw_results.csv` 同时包含seed 3407和2024，汇总时必须按实验条件过滤。

## 2026-09-08

- 新增 `experiments/run_legacy_comparison.py` 与 `experiments/legacy_comparison.py`，提供无 5-Fold 的 2025 八模型单次划分对比入口。
- OurModel 在该入口中冻结历史 checkpoint，并在任何基线训练前严格校验 `Accuracy=0.9383`、`Macro-F1=0.8759`、混淆矩阵 `[[187,0],[14,26]]`。
- 冻结 45 条历史兼容验证事件身份，消除旧 `train_val_split1` 因 `set` 遍历造成的跨进程 292/45、294/43 等漂移。
- 七个基线使用固定 292/45 划分训练一次；神经模型按验证 Macro-F1 保存最佳 epoch，并在同一 227 条独立测试集执行旧版多数投票与事件回填。
- 新增 `docs/13_旧版八模型单次划分对比实验.md`，记录命令、输出、恢复证据和论文报告边界。
- 新增 `experiments/run_legacy_ourmodel.py`，提供零 Fold、单 checkpoint 的 2025 历史 OurModel 精确复现入口。
- `test.py` 新增可选独立输出目录与 `metrics.json`，默认历史行为不变；复现结果不会覆盖五折结果或旧日志。
- 使用历史 `best_model_2026-07-29-21.42.51.pth` 在 CUDA 上精确复现 `Acc(U)=0.9383`、`F1(U)=0.8759`、混淆矩阵 `[[187,0],[14,26]]`。
- 新增 `docs/12_历史OurModel单模型复现.md`，记录口径、命令、输出和论文使用边界。
- 修复旧版 seed 2024 与原 seed 3407 五折缓存冲突：`legacy_bicfnet` 自动使用独立的 `experiments/splits/legacy_bicfnet`，保留现代协议缓存且无需 `--force`。
- 新增并默认启用 `legacy_bicfnet` 协议：seed 2024、序列长度 26、batch 8、300 epochs；OurModel 恢复 2e-5 学习率、0.01 weight decay、0.1 Focal 权重、余弦调度和验证集 Macro-F1 最优 checkpoint。
- 原 20 轮实现保留为 `--protocol modern`；旧版结果独立写入 `experiments/results_legacy_bicfnet`，不覆盖已有结果。
- 独立测试的旧版协议新增受试者多数投票和平票概率决胜，同时输出严格受试者级及 `legacy_voted_event` 历史兼容口径。
- 新增 `docs/11_OurModel_forward与旧版实验协议.md`，说明 forward、损失、选模、评价口径和两年完整命令。
- 新增八模型独立测试评估，五折概率等权平均，分别输出事件级及受试者级指标。
- PyTorch 复用 checkpoint，SVM/XGBoost 按已保存的训练折重新拟合；结果写入新 CSV。
- 根据用户最新要求取消独立测试路径哈希校验，开发后统一测试。
- 新增 `docs/09_八模型独立测试集评估.md`，提供两年命令、数据与结果位置，以及历史 personality 匹配限制。
- 新增 subject-aware Dataset，修复2026 `A_1.npy` 被误识别为受试者 `A` 的人格向量匹配问题；原 Dataset 与旧结果保留，修复后的2026结果使用独立目录。

## 2026-09-07

- 用户运行命令统一改为先 `conda activate dachuangxiangmu`，随后直接使用 `python`。
- 正式实验默认设备改为 CUDA，XGBoost 使用 GPU hist，PyTorch DataLoader 启用 pinned memory，并在结果中记录 Device；SVM 明确保留 CPU。
- 正式对比范围由十模型调整为八模型，移除 DepMamba 与 Proposed 的 Registry、占位代码、配置、Smoke Test 和运行文档入口。
- 项目已建立 Git 仓库并使用 `GuoChuang` 分支维护后续改动。
- 将目标目录中的 2025 和 2026 数据分别归档到 `test\2025`、`test\2026`。
- 接入两届 Elder/Young 数据解析，并完成四套训练数据真实读取、5-Fold 检查。
- 2025 Elder 与 2026 Elder 完成真实数据 MLP tiny 5-Fold 闭环。
- 为 2026 独立测试集生成标准 `split_labels_test.csv` 并补齐人格向量文件。

## 2026-09-06

### Added

- `AGENTS.md`：稳定项目规则和文档入口。
- `docs/PROJECT_CONTEXT.md`：源码确认的稳定技术事实。
- `docs/CURRENT_STATUS.md`：阶段、模型、问题和下一步状态。
- `docs/00_项目结构说明.md`、`docs/07_环境与依赖说明.md`。
- 已批准的设计文档和详细实施计划。
- 新增 MLP、BiLSTM、LightWeightTrans、LMF、MulT 完整 PyTorch 模型 adapter。
- 新增 SVM/XGBoost classical baseline 与两层 experiment registry。
- 新增 subject-level split、evaluator、efficiency、checkpoint、run_model_cv、run_all、aggregate 和 smoke test。
- 新增 01 至 06 中文说明和自动化 tests。
- 新增 2025/2026 统一数据布局解析器和 `08_2025与2026数据集使用说明.md`。

### Changed

- `train.py` 与 `test.py` 改用 `models.create_model()`；test 同时支持新旧 checkpoint。
- `models/__init__.py` 使用精确模型名并抛明确错误，不再正常退出掩盖失败。
- OurModel 仅训练时创建保存目录，推理实例化不再产生目录副作用。
- OurModel 在 `use_personality=false` 时真正忽略输入人格向量。
- requirements 保留已有依赖并增加分组说明。
- `run_model_cv.py`、`run_all.py` 与 `smoke_test.py` 新增 `--dataset-year`、`--cohort`，支持两届 Elder/Young 数据。
- Dataset 关闭 personality 时不再要求人格特征文件，并稳定返回零向量占位。

### Fixed

- 文档明确阻止环境激活失败后误用系统 Python。
- 新 subject folds 杜绝 segment-level leakage。
- run_all 直接文件入口加入项目根路径，`python experiments\run_all.py` 可正常导入包。
- subject ID 优先使用显式 `subject_id`，否则遵循数据集文件名前缀约定，最后才回退到通用 `id`。
- fold 缓存加入 label、seed、fold 数和数据指纹校验，阻止残缺或跨条件缓存误复用。
- 新 CV 入口限定二分类；旧训练/测试入口继续保留 3 类、5 类兼容。
- raw rows 加入 Run_ID/完整实验条件并对同折重跑去重，summary 不再混合不同 seed 或特征组合。
- 测试入口校验 checkpoint 特征配置和集成权重；每折重新设 seed 并使用显式 DataLoader generator。

### Known Issues

- tiny 运行结果仅验证数据与框架闭环，正式多模型训练尚未执行。
- matplotlib 可选依赖缺失；XGBoost 3.2.0 已安装并通过创建 Smoke Test。
