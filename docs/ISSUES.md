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

---

（后续问题在开发过程中自动追加）
