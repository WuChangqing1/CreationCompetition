# 会话日志

> 每次会话结束前追加一条记录

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
