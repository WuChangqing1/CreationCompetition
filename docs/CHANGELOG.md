# 变更日志

> 基于 Git 提交历史自动维护

格式基于 [Keep a Changelog](https://keepachangelog.com/)

---

## [Unreleased]

### Added
- [2026-08-18] EXP: 5-fold 消融结果（seed=42）— cls_only（无回归头）F1=0.6480±0.0832 为当前最好干净结果；CMG-VS 忠实版 CVAE F1=0.6258±0.0632 无增益。结论：增益来自去掉回归头（0.610→0.648，+3.8pp），非 CVAE。
- [2026-08-18] FEAT: 按 CMG-VS 论文忠实化 CVAE 任务引导机制（序列级 L_consis + CVAE 输入 detach + λ_aug=1.0），新增 `--cls_only` 公平对照与 `--num_workers` 参数
- [2026-08-18] DOCS: 新增 D014 决策、K016-K017 问题，更新 K012（公平对照混淆）
- [2026-07-30] DOCS：同步项目当前状态、`dachuangxiangmu` 环境、DepFormer/BCT/CVAE 结果与后续消融计划
- [2026-07-30] DOCS：新增 D010-D012 决策（冠军方法、CVAE 消融要求、分类指标口径）
- [2026-07-30] DOCS：新增 D013 决策（当前未跟踪资料暂不纳入 Git）
- [2026-07-30] DOCS：新增 K011-K015 问题（旧环境名、CVAE 对照混淆、分类-only 指标命名、opensmile DLL、conda run 编码）
- [2026-07-23] FEAT: CVAE 数据增强集成，Track1 A-V+P binary 当前最好 F1=0.6360±0.0965
- [2026-07-23] FEAT: DepFormer/BCT 冠军方法集成，A-V+P 新配置非 CVAE F1=0.6097±0.0558
- [2026-07-18] CV 正式结果：Elder G+P binary F1=0.64±0.12，Joint G+P binary F1=0.66±0.10，回归头(CCC)≈0 不可行
- [2026-07-18] PyTorch nightly cu130：修复 RTX 5070 Blackwell GPU 兼容性
- [2026-06-04] FEAT: 5-fold CV + Elder/Young 联合训练（experiments/train_cv.py + train_val_split.py create_kfold_splits），含 JointDataset + per-cohort ID 去重
- [2026-06-04] DOCS：数据集全面分析报告（数据集分析.md），17,231 文件扫描，9 章节，8 条改进建议
- [2026-06-03] DOCS：新增 K008-K010 三个问题（train.py 缺失特性、维度灾难、三元分类失败）
- [2026-06-03] DOCS：新增 D006-D007 两个决策（移除无效 CLI 参数、首轮训练策略）
- [2026-06-03] baseline 训练首轮结果：G+P binary F1=0.75/kappa=0.50，A-V+P 性能不及 G+P
- [2026-06-02] 初始化项目记忆系统（docs/ 全部 6 个文件）
- [2026-06-02] 项目文档：README、ARCHITECTURE、GETTING_STARTED
- [2026-06-02] .gitignore for Python/ML project
- [2026-06-02] MPDD-AVG-2026 官方基线代码
- [2026-06-02] MPDD 2025 参考实现代码
- [2026-06-02] 竞赛提交工具 make_submission_forcodabench
- [2026-06-02] 论文 PDF（MPDD 相关论文 4 篇）

### Fixed
- [2026-06-03] run_baseline.py：移除 train.py 不接受的 extra CLI 参数（--selection_metric, --cls_loss_weight, --reg_loss_weight, --weighted_sampler, --label_smoothing），修复 binary G+P 报错 "unrecognized arguments"

### Changed
- [2026-07-30] PROGRESS/GETTING_STARTED/ISSUES/LOG：明确没有 2026 official test 集不阻塞当前代码跑通，只阻塞最终 official test/CodaBench
- [2026-07-30] GETTING_STARTED/CLAUDE：当前 Conda 环境改为 `dachuangxiangmu`
- [2026-07-30] GETTING_STARTED：补充 Windows 下直接调用环境 python 和 `PROCESSOR_ARCHITECTURE=AMD64` 的 smoke test 前置条件
- [2026-07-30] PROGRESS：从 2026-07-18 旧 HOPE 阅读状态更新为当前实验状态快照
- [2026-06-03] PROGRESS：更新首轮训练结果和待办项，明确下一步方向
- [2026-06-02] GETTING_STARTED：环境名改为 creationcompetition，补充完整依赖清单（14 个包）
- [2026-06-02] PROGRESS：conda 环境已创建，依赖已安装，进度更新
- [2026-06-02] ARCHITECTURE：补充完整数据清单（目录树 + 统计表 + 兼容性说明 + 6 个已知问题）
- [2026-06-02] GETTING_STARTED：修正训练命令使用实际数据路径（test/Elder, test/Young）
- [2026-06-02] ISSUES：新增 K003-K007 共计 5 条数据相关问题
- [2026-06-02] requirements.txt：创建，含全部 14 个 pip 依赖
- [2026-06-02] smoke_test.py：创建，4 阶段冒烟测试全部通过
- [2026-06-02] run_baseline.py：创建，统一批量训练脚本，支持 12 种赛道组合

## [1ff79da] — 2026-06-02

### Added
- 项目初始化：122 files, 26,670 lines
- GitHub 私有仓库: https://github.com/WuChangqing1/CreationCompetition
