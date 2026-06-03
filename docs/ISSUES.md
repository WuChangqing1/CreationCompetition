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
- **详情**: `test/MPDD-AVG-2026-main/MPDD-AVG2026/MPDD-AVG2026-test/` 目录为空，仅存在空壳
- **影响**: 当前无法运行 test.py 做测试集评估，只能训练+验证
- **处理**: 需从 HuggingFace 下载 test 集数据

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
