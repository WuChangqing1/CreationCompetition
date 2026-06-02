# 会话日志

> 每次会话结束前追加一条记录

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
