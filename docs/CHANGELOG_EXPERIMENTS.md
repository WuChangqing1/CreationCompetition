# 实验框架变更记录

## 2026-09-08

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
