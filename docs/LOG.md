# 会话日志

> 每次会话结束前追加一条记录

---

## 2026-08-18 会话 (第8次)

**做了什么**：
- 精读本地论文 `PDF/paper02`（CMG-VS: Cross-Modal Guided Visual Synthesis, CVPR 2026），对照现有 CVAE 实现找出 4 处与论文不符的差距。
- 按论文忠实化 CVAE：`torchcat_baseline.py` 中 CVAE 输入 `v_seq`/`cond_seq` 全部 `.detach()`（使 `L_consis`/`L_KL` 只更新 CVAE），并返回 `v_seq_real`/`v_synth_seq`；`train_cv.py` 中 `L_consis` 改为序列级 L1、`λ_aug` 提到 1.0。
- 新增公平对照：`--cls_only`（无 CVAE 且无回归头）+ `use_regression_head` 配置化 + `--num_workers` 参数 + checkpoint variant 分目录。
- 修复两个 bug：K016（沙箱下 DataLoader 多进程管道被拒）、K017（关回归头但未开 CVAE 时 `reg_out=None` 崩溃）。
- 跑 5-fold 消融（Track1 A-V+P binary，seed=42）：cls_only F1=0.6480±0.0832，CMG-VS 忠实版 CVAE F1=0.6258±0.0632。
- 更新记忆文件：PROGRESS/DECISIONS(D014)/ISSUES(K012,K016,K017)/CHANGELOG/ARCHITECTURE/GETTING_STARTED。

**结论**：真正提升精度的是去掉 PHQ 回归头（F1 0.610→0.648，+3.8pp）；CVAE 数据增强在 MPDD-AVG 87 样本上没有增益，旧记录的“CVAE +2.6pp”是回归头混淆的假象。

**下次继续**：
- 多 seed 复跑 cls_only vs CVAE，确认方差与结论稳定性。
- 修正分类-only 日志中 `ccc/rmse/mae` 的命名，避免误读为 PHQ-9 指标。
- 跑 Track2 Young 与 A-V-G+P；处理三分类 class collapse。

**提交**：待提交（推送到 feature/champion-methods）

## 2026-07-30 会话 (第7次)

**做了什么**：
- 扫描并整理项目当前状态：分支为 `feature/champion-methods`，核心进展是 DepFormer/BCT/CVAE 实验。
- 根据用户确认，将当前 Conda 环境记录为 `dachuangxiangmu`。
- 验证环境：Python 3.10.20 / PyTorch 2.14.0.dev20260717+cu130 / CUDA True / NVIDIA GeForce RTX 5070 Laptop GPU。
- 确认 `test/` 目录状态：`test/Elder` 和 `test/Young` 是 2026 trainval 特征；官方 `MPDD-AVG2026-test` 当前为空；`test/MPDD-Test` 是 MPDD 2025 参考测试数据。
- 用户补充确认：当前确实没有 2026 official test 数据集，但现有代码可以跑通；test 缺口只影响最终 official test/CodaBench。
- 用户确认当前未跟踪文件不需要纳入 Git。
- 完成 smoke test：设置 `PROCESSOR_ARCHITECTURE=AMD64` 后，环境、数据、划分、G+P 3 epoch 训练流程全部通过。
- 更新记忆文件：`CLAUDE.md`、`docs/GETTING_STARTED.md`、`docs/PROGRESS.md`、`docs/DECISIONS.md`、`docs/ISSUES.md`、`docs/CHANGELOG.md`、`docs/ARCHITECTURE.md`、`docs/README.md`。
- 记录当前最好结果：Track1 A-V+P binary，CVAE F1=0.6360±0.0965，Acc=0.7261±0.0850，Kappa=0.3016±0.1813。
- 明确关键风险：CVAE 模式关闭 PHQ 回归头，非 CVAE 对照启用回归头，因此需要严格消融。

**下次继续**：
- 跑 `No-CVAE + no-regression-head` 对照实验。
- 修正分类-only 指标命名，避免把类别派生 CCC/RMSE/MAE 误读为 PHQ-9 指标。
- 继续当前训练/CV/消融工作；只有准备 official test 评估或 CodaBench 时才需要补齐 test 集。

**提交**：尚未提交

## 2026-07-18 会话 (第6次)

**做了什么**：
- 修复 CUDA 兼容性：RTX 5070 (Blackwell sm_120) → PyTorch nightly cu130，GPU 正常
- 跑通完整基线训练：Track1 G+P binary (F1≈0.65, 151 epochs) + ternary (kappa=0, 71 epochs)
- 5-fold CV Elder G+P binary：F1=0.6405±0.1197, Kappa=0.3076±0.2222，Fold 间波动很大
- 5-fold CV Joint Elder+Young G+P binary：F1=0.6592±0.0959, Kappa=0.3231±0.1898，改善有限
- 确认回归头 CCC≈0（PHQ-9 预测不可行），三分类 kappa=0（全猜 class 0）
- 更新 PROGRESS/DECISIONS(D009)/CHANGELOG/LOG
- 发现 HOPE 论文（PDF/3746027.3762063.txt）作为模型改进参考

**下次继续**：
- 阅读 HOPE 论文，评估分层融合架构
- 引入音频/视频特征（需先解决维度灾难：PCA 降维 or 更强的正则化）
- 尝试 A-V+P + 5-fold CV，对比纯 G+P
- 探索更激进的数据增强

---

## 2026-06-04 会话 (第5次)

**做了什么**：
- 生成数据集全面分析报告：扫描 17,231 个文件，覆盖 Elder 87人 + Young 88人
- 分析报告含 9 大章节 + 8 条改进建议
- 关键发现：78 训练样本 vs 2100 维特征（27:1 维度灾难）、9 人验证集极不稳定、三分类不可行、seed 顺序 bug、Elder/Young 61 个 ID 重叠
- 实现 5-fold CV：train_val_split.py 新增 create_kfold_splits()（StratifiedKFold）
- 实现 Elder+Young 联合训练：experiments/train_cv.py（JointDataset + per-cohort label_map 解决 ID 重叠）
- 修复 train_val_split.py 的 seed 顺序 bug（seed 在 split 之前设置）
- 冒烟测试通过：3-fold CV pipeline 端到端跑通
- 更新 PROGRESS/CHANGELOG/DECISIONS(D008)/LOG

**下次继续**：
- 跑正式 5-fold CV 训练（G+P binary, 320 epochs）
- 跑 Elder+Young 联合训练（G+P binary, 175 样本）
- 对比单赛道 CV vs 联合 CV 结果

---

## 2026-06-03 会话 (第4次)

**做了什么**：
- 修复 run_baseline.py：移除 train.py 不接受的 `extra` CLI 参数（--selection_metric, --cls_loss_weight, --reg_loss_weight, --weighted_sampler, --label_smoothing），这些参数在当前 train.py 中未实现
- 确认 shell 脚本与 train.py 版本不匹配：脚本引用了未实现的特性（损失权重、加权采样、标签平滑、自定义选择指标）
- 跑通 Track1 G+P baseline：binary F1=0.75/kappa=0.50（可用但受限），ternary 完全失败（val_kappa=0.0，全猜 class 0）
- 跑通 Track1 A-V+P baseline：binary F1=0.50/kappa=0.00（不如 G+P），ternary F1=0.36/kappa=0.08（微弱信号）。发现维度灾难：更多特征 + 同样 78 样本 = 更差性能
- 诊断三个核心问题：① train.py 缺失高级特性 ② 78 样本 vs 2100 维特征的维度灾难 ③ 三元分类当前数据规模不可行
- 新增 3 个 ISSUES（K008-K010），新增 2 个决策（D006-D007），更新 PROGRESS/CHANGELOG

**环境状态**：
- Python 3.10.20 / PyTorch 2.7.1+cu128 / CUDA 12.9 GPU OK
- creationcompetition conda 环境完整可用
- baseline 训练 pipeline 已验证可端到端运行

**下次继续**：
- 试 hybrid_attn 编码器对抗高维特征
- 调优 A-V+P 超参（hidden_dim↑, lr↑, epochs↑）
- 跑 Track2 Young（数据分布不同）
- 审计音频/视频特征加载（NaN、归一化）
- 将 weighted_sampler/label_smoothing/cls_loss_weight 补回 train.py

**提交**：73e409d

---

## 2026-06-02 会话 (第3次)

**做了什么**：
- 全项目文件盘点：逐目录扫描所有子文件夹和文件
- 数据集详细盘点：Elder 87 人 / Young 88 人，核对所有特征文件
- 发现 5 个数据相关问题（K003-K007：路径不匹配、openface格式、CSV差异、无test集、无预分割val）
- 确认基线 dataset.py 已兼容目录名变体（wav2vec2→wav2vec, mfcc64→mfcc）
- 创建 requirements.txt（14 个依赖，阿里云镜像加速安装）
- 安装全部依赖到 creationcompetition 环境（PyTorch 2.7.1+cu128，GPU CUDA ready）
- 创建 smoke_test.py：4 阶段冒烟测试（环境→数据→划分→训练），全部通过
- 创建 run_baseline.py：统一 Python 批量训练脚本，替代 12 个 shell 脚本，支持 44 个训练任务
- 更新 ARCHITECTURE（数据清单+表格）、GETTING_STARTED（修正数据路径）、ISSUES（+5条）、PROGRESS、DECISIONS

**环境最终状态**：
- Python 3.10.20 / PyTorch 2.7.1+cu128 / CUDA 12.9 GPU OK
- 全部 14 个依赖已安装（numpy, sklearn, pandas, transformers, librosa, opencv, soundfile, resampy, opensmile, tqdm, Pillow）

**下次继续**：
- 运行 run_baseline.py 开始正式基线训练
- 下载 test 集数据（可选，目前只有 trainval）

---

## 2026-06-02 会话 (第2次)

**做了什么**：
- 创建 conda 环境 `creationcompetition`（Python 3.10.20, CUDA 12.9, GPU 可用）
- 环境诊断：确认环境为空壳，仅有 pip/setuptools/wheel，12 个 ML 依赖待安装
- 更新文档：GETTING_STARTED（环境名 + 完整依赖清单）、PROGRESS（标记环境已创建）、DECISIONS（D004 环境命名决策）

**环境状态**：
- Python 3.10.20 ✓
- CUDA 12.9 (NVIDIA GPU) ✓
- PyTorch / torchvision 未安装
- numpy, sklearn, pandas, tqdm 未安装
- transformers, soundfile, resampy, librosa, opensmile 未安装
- opencv-python, Pillow 未安装

**下次继续**：
- 安装全部依赖到 creationcompetition 环境
- 下载 MPDD-AVG 2026 数据集
- 跑通基线训练

---

## 2026-06-02 会话 (第1次)

**做了什么**：
- 初始化项目记忆系统：创建 docs/ 目录及全部 6 个记忆文件
- 创建 .gitignore（Python/ML 项目类型）
- 基线代码入库：MPDD-AVG-2026 官方基线 + MPDD 2025 参考实现
- 编写项目文档：README（总览）+ ARCHITECTURE（架构）+ GETTING_STARTED（入门）
- 初始化 Git 仓库，创建 GitHub 私有仓库 CreationCompetition
- 完成首次提交并推送到远程

**提交**：`1ff79da` — Initial commit: MPDD-AVG 2026 competition baseline + project docs (122 files)
