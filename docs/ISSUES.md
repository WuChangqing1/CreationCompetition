# 问题记录

> 记录开发中遇到的 bug、坑、注意事项

---

## Known Issues

### K001 — Young Track2 test 视频特征为空
- **发现日期**: 2026-06-02
- **来源**: 官方基线 README
- **详情**: `MPDD-AVG2026-test/Young/Video/densenet`、`resnet`、`openface` 下文件为 0 字节空文件
- **影响**: Track2 A-V+P 和 A-V-G+P 在 test 阶段视频分支退化为零输入，评分会偏低
- **状态**: 等待组织方更新数据

### K002 — Young trainval OpenFace 部分无效
- **发现日期**: 2026-06-02
- **来源**: 官方基线 README
- **详情**: Young trainval 的 Video/openface 不是全量有效文件
- **影响**: 使用 OpenFace 特征时有效样本数减少
- **处理**: 训练时结合日志中的有效样本数判断

### K003 — 数据路径不匹配
- **发现日期**: 2026-06-02
- **来源**: 本地文件盘点
- **详情**: 基线代码期望 `MPDD-AVG2026/MPDD-AVG2026-trainval/{Elder,Young}`，实际数据在 `test/{Elder,Young}/`
- **影响**: config.json 的默认路径无效，必须通过 CLI 参数显式指定 data_root/split_csv/personality_npy
- **处理**: 已更新文档中的训练命令，使用 `../../Elder` 和 `../../Young` 等相对路径

### K004 — Young openface 目录格式
- **发现日期**: 2026-06-02
- **来源**: 本地文件盘点
- **详情**: `test/Young/Video/train/openface/{id}/` 下是子目录 `event_N/event_N_all.npy`，非 flat .npy 文件
- **影响**: 需 dataset.py 中的特殊处理逻辑（已确认基线代码兼容）

### K005 — Elder CSV 格式差异
- **发现日期**: 2026-06-02
- **来源**: 本地文件盘点
- **详情**: Elder CSV 有 BOM 头 (`﻿`)，列序为 `split,label3,label2,ID,PHQ-9`；Young CSV 无 BOM，列序为 `ID,split,label2,label3,phq9_score`
- **影响**: 需确认 dataset.py 的 CSV 解析是否兼容两种列序（已确认兼容）

### K006 — 无 2026 test 集
- **发现日期**: 2026-06-02
- **来源**: 本地文件盘点
- **详情**: 用户确认当前没有 MPDD-AVG 2026 official test 数据集；`test/MPDD-AVG-2026-main/MPDD-AVG2026/MPDD-AVG2026-test/` 目录为空，仅存在空壳。`test/MPDD-Test/` 属于 MPDD 2025 参考测试数据，不等同于 2026 official test。
- **影响**: 不影响当前训练、CV 和 smoke test 跑通；只影响 official test 评估和 CodaBench 提交。
- **处理**: 若后续要提交 CodaBench，再下载或放入 official test 集数据。
- **状态**: 已确认缺口，非当前训练阻塞项。

### K007 — 标签全为 train
- **发现日期**: 2026-06-02
- **来源**: 本地文件盘点
- **详情**: Elder 和 Young 的 split_labels_train.csv 中 split 列全为 `train`，无预分割的 val 或 test
- **影响**: 训练时 val 集完全依赖 train_val_split.py 的自动划分（val_ratio=0.1）

### K008 — train.py 缺少 shell 脚本引用的高级训练特性
- **发现日期**: 2026-06-03
- **来源**: 基线训练首轮
- **详情**: 12 个 shell 脚本使用 `--selection_metric`、`--cls_loss_weight`、`--reg_loss_weight`、`--weighted_sampler`、`--label_smoothing` 等参数，但当前 train.py argparse 完全不接受这些参数。选择指标硬编码为 f1/ccc，损失为 CE+MSE 等权，无加权采样和标签平滑
- **影响**: ① 类别不平衡无法通过 cls_loss_weight 缓解 ② 无法用 kappa/ccc_floor 选最佳模型 ③ 小数据集下 label_smoothing 的防过拟合效果缺失
- **处理**: run_baseline.py 已移除这些无效参数；后续需将这些特性补回 train.py

### K009 — A-V+P 性能反而不如 G+P（维度灾难）
- **发现日期**: 2026-06-03
- **来源**: Track1 A-V+P vs G+P 对比训练
- **详情**: G+P binary F1=0.75/kappa=0.50，A-V+P binary F1=0.50/kappa=0.00。加了音频+视频后性能下降。A-V+P 输入维度 ~2100d（opensmile~88d + resnet~1000d + personality~1024d），G+P 仅 ~1036d（gait 12d + personality 1024d），但训练样本只有 78 个
- **根因**: ① 维度灾难 — 特征维度增加但样本量不变，模型需要更多数据才能从高维特征中提取信号 ② 超参不匹配 — A-V+P 的 hidden_dim(64) 反而小于 G+P(128)，lr(3e-5) 低于 G+P(8e-5)，epochs(60) 仅为 G+P(320) 的 1/5 ③ 特征归一化未知 — opensmile/resnet 特征的 scale 差异可能干扰训练
- **状态**: 待解决

### K010 — 三元分类（ternary）在 Track1 Elder 上完全失败
- **发现日期**: 2026-06-03
- **来源**: Track1 G+P 和 A-V+P ternary 训练
- **详情**: G+P ternary val_kappa=0.0 best_epoch=1，A-V+P ternary val_kappa=0.08 best_epoch=10。两者模型均将所有样本预测为单一类别（全猜 class 0），完全无区分能力。78 训练样本 / 3 类别 / 仅 9 验证样本
- **影响**: 三元分类在当前数据规模（Elder 87 人）和特征组合下不可行。可能需要：① 更大的训练集（Elder+Young 联合？）② 更强的特征组合（A-V-G+P）③ 数据增强
- **状态**: 待解决

### K011 — 旧文档环境名错误
- **发现日期**: 2026-07-30
- **来源**: 用户确认 + 环境验证
- **详情**: 旧记忆文件记录 Conda 环境为 `creationcompetition`，但当前实际可用环境为 `dachuangxiangmu`。
- **影响**: 后续 Agent 或人工复现实验时如果使用旧环境名，会误判依赖缺失或无法运行 PyTorch。
- **处理**: 已更新 `CLAUDE.md`、`docs/GETTING_STARTED.md`、`docs/PROGRESS.md`、`docs/DECISIONS.md`。当前验证结果：Python 3.10.20 / PyTorch 2.14.0.dev20260717+cu130 / CUDA True / RTX 5070。
- **状态**: 已处理，后续命令统一使用 `conda run -n dachuangxiangmu ...`。

### K012 — CVAE 实验存在公平对照混淆
- **发现日期**: 2026-07-30
- **来源**: `experiments/train_cv.py` 和 `docs/CVAE_INTEGRATION.md`
- **详情**: CVAE 模式会关闭 PHQ 回归头，非 CVAE 对照仍使用分类+回归联合训练。因此 CVAE 的 +2.6pp F1 提升混合了“数据增强效应”和“去除回归噪声效应”。
- **影响**: 不能直接把当前最好 F1 全部归功于 CVAE；论文/汇报中需要避免过度结论。
- **处理**: 2026-08-18 已按 D014 修复：新增 `--cls_only` 公平对照（无 CVAE + 无回归头），并将 `L_consis` 改为序列级且 detach。
- **状态**: 修复已实现，等 5-fold 消融结果验证。

### K013 — 分类-only 日志中的 CCC/RMSE/MAE 容易误读
- **发现日期**: 2026-07-30
- **来源**: `metrics.py`
- **详情**: 在无 PHQ 回归头时，`classification_metrics()` 会基于整数类别标签计算 CCC/RMSE/MAE；这些不是 PHQ-9 回归指标。
- **影响**: CVAE JSON 里的 `cv_ccc/cv_rmse/cv_mae` 可能被误认为 PHQ 分数预测表现。
- **处理**: 后续应改字段名或只在有回归输出时记录 PHQ 指标。
- **状态**: 待修复。

### K014 — Windows 下 opensmile/audresample 依赖平台名识别异常
- **发现日期**: 2026-07-30
- **来源**: `smoke_test.py`
- **详情**: 直接运行 smoke test 时，`opensmile -> audresample` 会尝试加载 `audresample\core\bin\win_\audresample.dll`，但实际 DLL 位于 `win_amd64`。原因是当前进程中 `platform.machine()` 返回空字符串。
- **影响**: 不设置环境变量时，完整依赖检查会失败；已预提取特征的训练核心依赖仍正常。
- **处理**: 在当前 PowerShell 会话设置 `$env:PROCESSOR_ARCHITECTURE = 'AMD64'` 后，`opensmile` 导入正常，完整 `smoke_test.py` 已通过。
- **状态**: 有 workaround；后续可考虑在启动脚本或文档中固定该环境变量。

### K015 — Conda run 在当前 PowerShell 输出编码下可能崩溃
- **发现日期**: 2026-07-30
- **来源**: `conda run -n dachuangxiangmu python smoke_test.py`
- **详情**: `conda run` 在打印子进程输出时触发 `UnicodeEncodeError: 'gbk' codec can't encode character`。
- **影响**: 可能误判为项目测试失败；实际直接调用环境内 `python.exe` 可正常运行。
- **处理**: 当前优先使用 `D:\App\Business\Coding\Python\Miniconda\envs\dachuangxiangmu\python.exe` 直接执行项目命令。
- **状态**: 有 workaround。

### K016 — Windows 受限环境 DataLoader 多进程无法创建管道
- **发现日期**: 2026-08-18
- **来源**: `experiments/train_cv.py` 在沙箱/受限 shell 下运行
- **详情**: `DataLoader(num_workers=2)` 在 Windows 沙箱下抛 `PermissionError: [WinError 5] 拒绝访问`（`multiprocessing` 需要创建命名管道，受限环境禁止）。
- **影响**: 在受限环境里训练脚本无法启动；普通用户本机（非沙箱）不受影响。
- **处理**: 新增 `--num_workers` CLI 参数（默认 2），受限环境用 `--num_workers 0` 单进程加载。本数据集很小（87 样本），单进程加载对速度影响可忽略。
- **状态**: 已修复。

### K017 — 关闭回归头但未启用 CVAE 时训练崩溃
- **发现日期**: 2026-08-18
- **来源**: `experiments/train_cv.py` 新增 `--cls_only` 后暴露
- **详情**: 当 `use_cvae=False` 且 `use_regression_head=False` 时，模型返回 `(logits, None)`，但旧训练循环仍按「有回归头」解包并计算 `MSELoss(reg_out, phq9)`，对 `None` 调用 `.size()` 报 `AttributeError`。这正是 K012 混淆变量的代码根源。
- **处理**: 重构 `train_one_fold` 的 criterion 选择与训练循环，按 `use_regression_head` 分支：分类-only 用单一 CrossEntropy(+focal)，联合用 (CE, focal, MSE)。
- **状态**: 已修复，`--cls_only` 冒烟通过。
