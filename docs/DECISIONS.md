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

---

（后续决策在开发过程中自动追加）
