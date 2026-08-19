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
- **决策**: 实际使用 `dachuangxiangmu` 作为当前 Conda 环境名（旧记录中的 `creationcompetition` 已不作为当前环境）
- **为什么**: 用户确认当前可用环境是 `dachuangxiangmu`；2026-07-30 已验证该环境可运行 Python 3.10.20、PyTorch 2.14.0.dev20260717+cu130，并能识别 RTX 5070 CUDA。后续训练、验证和复现实验统一使用 `conda run -n dachuangxiangmu ...`。

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

## D009 — G+P 基线性能天花板确认

- **日期**: 2026-07-18
- **决策**: G+P (步态+人格) 二分类天花板在 F1≈0.66，Joint 联合训练仅比 Elder 单独高 0.02，不足以突破。三分类和 PHQ-9 回归在 175 样本下仍不可行。
- **为什么**: 
  - Elder 5-fold CV F1=0.64±0.12，Joint F1=0.66±0.10，提升有限
  - 回归头 CCC≈0，表明 PHQ-9 分数预测在当前设置下无信号
  - 三分类 kappa 全程 0.0（全猜 class 0）
  - 12 维步态特征信息量本就有限，核心信号来自 1024 维人格特征
- **后果**: 要突破天花板需要：① 换更强模型架构（如 HOPE 论文的分层融合）② 引入音频/视频特征并解决维度灾难 ③ 专注二分类放弃三分类和回归

## D010 — 当前冠军方法：DepFormer + BCT 优先服务 A-V+P

- **日期**: 2026-07-30
- **决策**: 在 `experiments/train_cv.py` 中以 Track1 A-V+P binary 为主要突破口，使用 DepFormerTemporalEncoder 保留时序序列，再接 BCT 做音视频跨模态交互。
- **为什么**: 早期 G+P 的 F1 天花板约 0.64-0.66；A-V+P 虽有维度灾难，但音频/视频含有额外抑郁信号。DepFormer/BCT 保留并利用时序结构，比简单 BiLSTM mean pooling 更适合多模态对齐。
- **后果**: 当前主实验代码已经偏离官方 baseline，架构文档和实验记录必须同步；Track2、A-V-G+P、ternary 仍需单独验证。

## D011 — CVAE 作为数据增强候选，但需要严格消融

- **日期**: 2026-07-30
- **决策**: 保留 CVAE 数据增强路径，但暂不把 F1 提升完全归因于 CVAE。
- **为什么**: 最新 Track1 A-V+P binary 结果显示 CVAE 配置 F1=0.6360±0.0965，高于非 CVAE 新配置 F1=0.6097±0.0558；但 CVAE 模式同时关闭了 PHQ 回归头，而非 CVAE 对照仍启用回归头，这是关键混淆因素。
- **后果**: 下一步必须跑 `No-CVAE + no-regression-head` 对照，并考虑 `L_consis` detach 实验，才能判断 CVAE 是否真的贡献了主要增益。

## D012 — 分类实验的主指标以 F1/Acc/Kappa 为准

- **日期**: 2026-07-30
- **决策**: 在关闭 PHQ 回归头的分类实验中，报告和选模重点看 Macro-F1、Accuracy、Kappa，不把日志中的 `ccc/rmse/mae` 当作 PHQ-9 回归指标。
- **为什么**: 当前 `metrics.py` 在分类-only 路径会基于整数类别标签计算 CCC/RMSE/MAE，这些值可作分类标签一致性的辅助信息，但不是 PHQ 分数预测质量。
- **后果**: 后续应修正指标命名或日志字段，避免论文/汇报时误用。

## D013 — 当前未跟踪资料暂不纳入 Git

- **日期**: 2026-07-30
- **决策**: `.agents/`、`StudyVault/`、PPT、PDF、`docs/CVAE_INTEGRATION.md`、`skills-lock.json` 等当前未跟踪文件暂不加入 Git。
- **为什么**: 用户明确确认这些未跟踪文件不需要纳入 Git；其中部分是个人学习资料、插件/技能缓存或大体积研究材料，不应在没有筛选前进入代码仓库。
- **后果**: 后续提交文档或代码时只 stage 明确需要的文件；不要因为 `git status` 显示这些路径就自动 `git add -A`。

## D014 — 按 CMG-VS 论文忠实化 CVAE 任务引导机制

- **日期**: 2026-08-18
- **决策**: 依据本地论文 `PDF/paper02`（CMG-VS, CVPR 2026）把现有 CVAE 数据增强路径改造成论文忠实的「任务引导视觉合成」：
  1. `L_consis` 改为**序列级** L1 重建 `‖v_seq − v_synth_seq‖₁`（论文 eq.10），不再用池化向量。
  2. CVAE 输入 `v_seq` / `cond_seq` 全部 `.detach()`，使 `L_consis`/`L_KL` 只更新 CVAE（论文中 `f_v`/`f_a` 是固定特征），任务引导梯度仅通过 `L_aug → f_v_synth_seq → decoder/encoder` 回传。
  3. `λ_aug` 由 0.5 提到论文默认 1.0。
  4. 新增 `--cls_only` 公平对照（无 CVAE 且无回归头），把「数据增强效应」和「去回归头效应」解耦。
- **为什么**: 旧实现 `L_consis` 只对池化向量做 L1 且不 detach，会反向扰动 DepFormer/BCT 主编码器（K012 的根源）；且 CVAE 组关闭回归头造成归因混淆，无法证明增益来自 CVAE。
- **后果**: `torchcat_baseline.py` 的 CVAE 分支与 `experiments/train_cv.py` 损失/配置已更新；checkpoint 目录新增 variant 区分（`cvae`/`cls_only`/`base`），避免不同消融互相覆盖。

## D015 — 论文对比实验协议（6 篇论文 × 6 配置）

- **日期**: 2026-08-19
- **决策**: 把 `PDF/` 下 6 篇论文作为对比实验：以 DepFormer/BCT 为固定骨干，只替换向量级融合模块（`fusion_type`），在 Track1+Track2 × A-V+P/A-V-G+P/G+P 的 binary 任务上跑完整 5-fold CV（seed=42，`--cls_only` 口径），结果写入 `docs/PAPER_COMPARISON.md`。
- **为什么**: 用户要求“把文件夹下论文都跑出来作为对比实验”；统一骨干+统一配置能隔离出“融合模块”这一变量的贡献；`--cls_only` 消除 PHQ 回归头的归因混淆（D014）。
- **结果**: 新增 `ptmfim`/`hope`/`reliability`/`hypergraph` 四个融合模块均无法稳定超越 baseline（cross_fusion）；baseline 在 Track1 A-V+P（0.6480）与 Track2 G+P（0.6809）仍最优；CMG-VS 仅 Track1 A-V-G+P +11.8pp。完整数字见 `docs/PAPER_COMPARISON.md`。
- **后果**: 这些论文为 MPDD 2025（1s/5s 窗口）方法，其完整贡献依赖外部数据/窗口结构，本次只迁移核心融合思想；结论需多 seed 确认。

---

（后续决策在开发过程中自动追加）
