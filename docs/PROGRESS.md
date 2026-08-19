# 项目进度

> 最后更新：2026-08-18
> 本文件记录当前研究状态、下一步任务和已完成工作。

## 进行中
- [ ] 当前没有 MPDD-AVG 2026 official test 集；这不影响训练/CV 继续推进，只影响最终 official test 评估和 CodaBench 提交。

## 待办
- [ ] 多 seed 复跑 cls_only vs CVAE（当前仅 seed=42），确认 CVAE 是否真的无增益、方差是否稳定。
- [ ] 修正或重命名 `metrics.py` / CV 日志里的分类派生 `ccc/rmse/mae`，避免被误读为 PHQ-9 回归指标。
- [ ] 跑 Track2 Young 和 A-V-G+P，对比 Track1 当前冠军方法。
- [ ] 处理三分类 class collapse：类别权重、采样、label smoothing 或暂时降优先级。
- [ ] 生成 CodaBench 提交文件：需要后续拿到 official test 集和最终模型 checkpoint。

## 当前状态快照
- 当前分支：`feature/champion-methods`，本地 HEAD 已包含 DepFormer、BCT、CVAE 数据增强实验。
- 当前 Conda 环境：`dachuangxiangmu`。
- 环境验证：Python 3.10.20 / PyTorch 2.14.0.dev20260717+cu130 / CUDA 可用 / GPU 为 NVIDIA GeForce RTX 5070 Laptop GPU。
- 完整 smoke test：2026-07-30 使用 `PROCESSOR_ARCHITECTURE=AMD64` 后通过，覆盖依赖、Elder/Young 数据、划分、G+P 3 epoch 训练链路。
- 用户确认：虽然没有 2026 official test 数据集，但当前代码是可以跑通的。
- 代码静态状态：上次扫描 61 个 Python 文件，AST 语法检查 0 错误；当前未发现已跟踪文件的未提交修改。
- 主要实验入口：`experiments/train_cv.py`。
- 主要模型：`test/MPDD-AVG-2026-main/models/torchcat_baseline.py`，已集成 DepFormer/BCT/CVAE。
- 最新详细实验文档：`docs/CVAE_INTEGRATION.md`，当前仍是未跟踪文件。
- Git 纳入范围：用户已确认当前未跟踪文件不需要纳入 Git，保持未跟踪即可。
- 数据状态：`test/Elder` 和 `test/Young` 是当前可用的 2026 trainval 特征；`test/MPDD-AVG-2026-main/MPDD-AVG2026/MPDD-AVG2026-trainval` 与 `MPDD-AVG2026-test` 当前均为空；`test/MPDD-Test` 是 MPDD 2025 参考测试数据，不等同于 2026 official test。

## 关键结果
- Elder G+P 5-fold CV binary：F1=0.6405±0.1197，Kappa=0.3076±0.2222。
- Joint Elder+Young G+P 5-fold CV binary：F1=0.6592±0.0959，Kappa=0.3231±0.1898。
- Track1 A-V+P binary，DepFormer/BCT，新配置无 CVAE（有回归头）：F1=0.609735±0.055819，Acc=0.645098±0.057561，Kappa=0.229057±0.114453。
- Track1 A-V+P binary，DepFormer/BCT + CVAE（旧，有混淆）：F1=0.636038±0.096527，Acc=0.726144±0.085028，Kappa=0.301581±0.181251。
- **Track1 A-V+P binary，cls_only（无 CVAE 无回归头，seed=42，2026-08-18）**：F1=0.6480±0.0832，Acc=0.7013±0.0956，Kappa=0.3271±0.1622。← 当前最好干净结果。
- Track1 A-V+P binary，CMG-VS 忠实版 CVAE（序列级 L_consis + detach + λ_aug=1.0，seed=42，2026-08-18）：F1=0.6258±0.0632，Acc=0.6895±0.0596，Kappa=0.2786±0.0999。

### 消融结论（2026-08-18）
- 去掉 PHQ 回归头是真正的增益来源：无 CVAE 时 F1 0.6097→0.6480（+3.8pp）。
- CMG-VS 的 CVAE 数据增强在 MPDD-AVG（87 样本）上没有带来增益（0.6258 vs 0.6480，落在噪声范围内，且方差更小）。
- 因此旧记录里“CVAE +2.6pp F1”是回归头混淆造成的假象，实际归因应改为“去掉回归头”。

### 论文对比结论（2026-08-19，见 `docs/PAPER_COMPARISON.md`）
- 6 篇论文方法（baseline/CMG-VS/PTMFIM/HOPE/MSF-ATS/P3HF）× 6 配置（Track1+Track2 × A-V+P/A-V-G+P/G+P）5-fold 全部跑完。
- baseline（DepFormer/BCT 的 cross_fusion）整体最优；新增 ptmfim/hope/reliability/hypergraph 融合模块均无法稳定超越 baseline；CMG-VS 仅 Track1 A-V-G+P +11.8pp。

## 已完成
- [x] [2026-08-19] 完成 6 篇论文对比实验（4 个新融合模块 + 全部 6 配置 5-fold），结果与结论写入 `docs/PAPER_COMPARISON.md` 并逐方法提交推送。
- [x] [2026-08-19] 实现 `models/fusion_methods.py`（PTMFIM/HOPE/Reliability/Hypergraph）+ `--fusion_type` 参数，冒烟通过。
- [x] [2026-08-18] 完成 cls_only vs CMG-VS 忠实版 CVAE 的 5-fold 消融（seed=42）：cls_only F1=0.6480 为当前最好干净结果；CVAE 无增益。
- [x] [2026-08-18] 按 CMG-VS 论文忠实化 CVAE：`torchcat_baseline.py` CVAE 输入 detach + 返回序列张量；`train_cv.py` 序列级 `L_consis`、`λ_aug=1.0`、`use_regression_head` 配置化、`--cls_only` 公平对照、`--num_workers` 参数、checkpoint variant 分目录。修复 K016/K017 两个 bug，两条路径冒烟通过。
- [x] [2026-07-30] 确认实际 Conda 环境为 `dachuangxiangmu`，并验证 PyTorch cu130 + RTX 5070 可用。
- [x] [2026-07-30] 完成接收 smoke test：环境、数据、划分、G+P 最小训练流程通过。
- [x] [2026-07-30] 确认当前未跟踪文件无需纳入 Git。
- [x] [2026-07-30] 用户确认没有 MPDD-AVG 2026 official test 数据集；当前只有 Elder/Young trainval 特征和 MPDD 2025 参考 test。
- [x] [2026-07-30] 扫描项目状态：代码、Git、实验日志、文档时效性和主要风险。
- [x] [2026-07-23] 集成 CVAE 数据增强：`models/cvae_synthesizer.py`、`models/torchcat_baseline.py`、`experiments/train_cv.py`。
- [x] [2026-07-23] 完成 Track1 A-V+P binary CVAE 5-fold 实验，当前最好 F1=0.6360±0.0965。
- [x] [2026-07-18] PyTorch nightly cu130 安装：RTX 5070 Blackwell sm_120 兼容，CUDA 可用。
- [x] [2026-07-18] 正式基线训练：Track1 G+P binary (F1≈0.65) + ternary (kappa=0)。
- [x] [2026-07-18] 5-fold CV Elder G+P binary：F1=0.6405±0.1197, Kappa=0.3076±0.2222。
- [x] [2026-07-18] 5-fold CV Joint Elder+Young G+P binary：F1=0.6592±0.0959, Kappa=0.3231±0.1898。
- [x] [2026-06-04] 实现 5-fold CV + Elder/Young 联合训练：`train_val_split.py` 新增 `create_kfold_splits()`，`experiments/train_cv.py` 新增 `JointDataset` + per-cohort ID 去重。
- [x] [2026-06-04] 数据集全面分析：17,231 个文件，9 大章节，写入 `数据集分析.md`。
- [x] [2026-06-03] 修复 `run_baseline.py`：移除 `train.py` 不接受的 extra CLI 参数。
- [x] [2026-06-03] 基线训练首轮完成：G+P binary F1=0.75/kappa=0.50；A-V+P binary F1=0.50；ternary 表现差。
- [x] [2026-06-02] 项目初始化：docs/ 目录及记忆文件、官方基线代码、GitHub 私有仓库、requirements、smoke_test、run_baseline。
