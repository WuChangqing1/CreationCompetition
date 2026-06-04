# 技术决策记录

> 记录所有重要的技术决策，确保项目方向一致

---

## D001 — 基线代码选型

- **日期**: 2026-06-02
- **决策**: 使用 MPDD-AVG-2026 官方基线作为起点（`test/MPDD-AVG-2026-main/`）
- **为什么**: 这是 MPDD-AVG 2026 竞赛的官方基线，包含完整的 train/val/test 流程、双编码器支持(bilstm_mean/hybrid_attn)、联合 PHQ-9 回归头，代码质量高且有配套论文
- **备选**: MPDD 2025 代码也保留在 `test/MPDD-main/` 作为参考，其模型架构更丰富（OurModel, AutoEncoder, LightWeightTrans 等）

## D002 — 仓库策略

- **日期**: 2026-06-02
- **决策**: 私有仓库，只提交代码和文档，不提交数据集和模型权重
- **为什么**: 竞赛代码闭源保护；数据文件过大且有访问授权限制；`checkpoints/`/`.npy`/`.pth` 已加入 `.gitignore`

## D003 — 分支策略

- **日期**: 2026-06-02
- **决策**: main(稳定) / develop(日常开发) / feature/* (功能分支)
- **为什么**: 标准 Git Flow，确保主分支稳定，所有实验在 feature 分支进行

## D004 — conda 环境命名

- **日期**: 2026-06-02
- **决策**: 使用 `creationcompetition` 作为 conda 环境名（不使用官方 README 中的 `mpddavg`）
- **为什么**: 与项目名 CreationCompetition 保持一致，方便识别和管理；不影响基线代码运行（代码中无环境名硬编码）

## D005 — 批量训练脚本语言

- **日期**: 2026-06-02
- **决策**: 使用 Python（run_baseline.py）替代 12 个 shell 脚本作为批量训练入口
- **为什么**: 用户环境为 Windows，无法直接执行 .sh 脚本；Python 跨平台且可直接调用 train.py，避免 WSL 和 conda 环境兼容问题

## D006 — 移除 run_baseline.py 中不受支持的 CLI 参数

- **日期**: 2026-06-03
- **决策**: 从 run_baseline.py 的 GP_CONFIGS 和 AVP_CONFIGS 中删除所有 `extra` 字段（--selection_metric, --cls_loss_weight, --reg_loss_weight, --weighted_sampler, --label_smoothing）
- **为什么**: 当前 train.py 的 argparse 仅接受 28 个参数，不包含这些高级特性。shell 脚本引用了这些参数但 train.py 未实现（代码版本不匹配）。选择指标硬编码为 f1（分类）/ ccc（回归），损失为 CrossEntropyLoss + MSELoss 等权。强行传入会导致 argparse 报错退出
- **后果**: G+P/A-V+P 训练均使用简化版损失函数，缺少 weighted sampling 和 label smoothing，可能是基线性能偏低的原因之一。后续需将这些特性补回 train.py

## D008 — CV + 联合训练架构决策

- **日期**: 2026-06-04
- **决策**: ① 创建独立的 `experiments/train_cv.py` 而非修改原 `train.py`（避免破坏基线完整性）② 联合训练使用 per-cohort label_map（Elder/Young 各自独立构建 `MPDDElderDataset`，通过 `JointDataset` 拼接）③ CV 划分使用数组索引而非 ID 做 key（解决 61 个 Elder/Young ID 重叠问题）
- **为什么**: ① 基线 train.py 应保持不变作为对照 ② ID 重叠意味着直接用 `int(ID)` 做 key 会导致标签混淆 ③ 数组索引+per-cohort 分解方案零侵入，不修改 baseline dataset.py
- **结果**: `experiments/train_cv.py` 同时支持单赛道 CV（`--track Track1`）和联合训练 CV（`--joint`），代码复用同一 `train_one_fold()` 函数

## D007 — 首轮训练策略：优先验证 pipeline，暂不铺全量

- **日期**: 2026-06-03
- **决策**: 首轮只跑 Track1 G+P 和 A-V+P 各一次（binary+ternary），不铺 44 个全量任务
- **为什么**: ① G+P 验证了 pipeline 端到端可运行 ② A-V+P 暴露了维度灾难和超参问题 ③ 全量 44 任务烧 GPU 时间但没有诊断价值 ④ 需要先理解失败模式再决定调参方向
- **后续**: 诊断完成后选择性重跑，而非无脑全量

---

（后续决策在开发过程中自动追加）
