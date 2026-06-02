# 代码架构

## 核心数据流

```
CSV 标签文件 ──→ train_val_split.py ──→ {train_map, val_map, phq_map}
                                              │
                                              ▼
                MPDDElderDataset(dataset.py) ──→ DataLoader ──→ train.py
                        │                                          │
                        ▼                                          ▼
                  TorchcatBaseline(model) ◄── 前向传播 ── 损失计算 + 评估
                        │                                          │
                        ▼                                          ▼
                  evaluate_model(metrics.py) ◄── 验证/测试 ── best checkpoint
```

## 模块详解

### 1. dataset.py — 数据加载 (478 行)

核心类 `MPDDElderDataset`，负责从文件系统读取多模态特征并组装成 batch。

**关键设计:**

- **ID-based split**：按受试者 ID 划分 train/val，避免同一受试者跨集泄漏
- **Pair 机制**：每个受试者有最多 4 对音频/视频 (pair 1-4)，通过 `pair_mask` 处理缺失对
- **时序插值**：所有时态模态线性插值到 `target_t=128` 统一长度
- **特征归一化**：Z-score 标准化后 clip 到 [-5, 5]
- **PHQ-9 log 变换**：`normalize_phq_target()` 使用 `log1p` 处理 PHQ-9 分数

**文件发现策略:**
- Elder: 文件命名模式 `A_{pair_idx}.npy` / `V_{pair_idx}.npy`
- Young: 文件命名模式 `E{idx}.npy` / `event_{idx}.npy` 或 `event_{idx}/event_{idx}_all.npy`
- 自动处理 `trainval/test` 目录配对、大小写差异

### 2. models/torchcat_baseline.py — 模型 (189 行)

`TorchcatBaseline` 是多模态融合模型，支持三个子赛道：

| 子赛道 | 模态组合 |
|--------|---------|
| A-V+P | Audio + Video + Personality |
| A-V-G+P | Audio + Video + Gait + Personality |
| G+P | Gait + Personality |

**编码器选项:**

- **bilstm_mean**: `ModalityEncoder` — BiLSTM 后取时序平均
- **hybrid_attn**: `HybridTemporalEncoder` — Conv1d + BiLSTM + TemporalAttentionPool

**融合策略:**
1. 各模态独立编码到 `hidden_dim` 维
2. 多 pair 音频/视频通过 `pair_mask` 加权平均
3. 所有模态特征 concat
4. 分类头 + 回归头双输出

**PersonalityEncoder**: 1024 → 256 → hidden_dim 的 MLP (专门的人格特征编码器)

### 3. models/hybrid_temporal_encoder.py — 时序编码器 (83 行)

```
输入 [B, T, C] → PreProjection(可选) → Conv1d(k5) → Conv1d(k3) → BiLSTM → TemporalAttentionPool → LayerNorm
```

`TemporalAttentionPool` 使用自注意力机制做时序池化，自动学习重要时间步。

### 4. metrics.py — 评估指标 (152 行)

分类任务输出 6 个指标，通过 `joint_regression_metrics()` 联合计算。

**最佳模型选择 (selection_score):**
- 分类任务 → Macro-F1
- 回归任务 → CCC

### 5. train.py — 训练流程 (424 行)

```
加载配置 → 划分 train/val → 构建 Dataset/DataLoader → 初始化模型
  → 训练循环: forward → loss = CE(cls) + MSE(reg)
  → 验证: evaluate_model() → 选最佳 F1/CCC
  → EarlyStopping(patience=20, min_delta=1e-4)
  → 保存 checkpoint + 历史曲线 + 汇总
```

**输出文件 (每个实验):**
- `best_model_{timestamp}.pth` — 最佳 checkpoint (含 model_kwargs)
- `result_{timestamp}.log` — 训练日志
- `history_{timestamp}.csv` — 每 epoch 指标
- `train_result_{timestamp}.json` — 汇总
- `{experiment_name}.csv` — 实验表格 (追加写入)

### 6. test.py — 评估流程 (222 行)

从 checkpoint 恢复模型 → 加载 test 数据 → 评估 → 输出指标和预测。

**关键**: checkpoint 内置了 `model_kwargs`，可完全复现模型结构。支持旧绝对路径的兼容性 remap。

### 7. train_val_split.py — 数据划分 (157 行)

- 按 `ID` 级别划分 (无数据泄漏)
- 优先使用分层采样 (StratifiedShuffleSplit)，少数类不足时退化为 ShuffleSplit
- 默认 `val_ratio=0.1`
- 固定 seed 保证可复现

## 实际数据清单（2026-06-02 盘点）

### 本地数据位置

数据不在基线期望的 `MPDD-AVG2026/` 下，而在 `test/` 下：

```
test/
├── Elder/                              # 87 个受试者，全为 train 集
│   ├── Audio/train/
│   │   ├── mfcc/       87 人  文件: A_1~4.npy
│   │   ├── opensmile/  87 人  文件: A_1~4.npy
│   │   └── wav2vec2/   87 人  文件: A_1~4.npy    ← 注意: 目录名 wav2vec2 非 wav2vec
│   ├── Video/train/
│   │   ├── densenet/   87 人  文件: V_1~4.npy
│   │   ├── resnet/     87 人  文件: V_1~4.npy
│   │   └── openface/   87 人  文件: V_1~4.npy
│   ├── IMU/train/      87 人  文件: {id}.npy
│   ├── split_labels_train.csv          ← 87 行, 列: split,label3,label2,ID,PHQ-9, BOM头
│   └── descriptions_embeddings_with_ids.npy
├── Young/                              # 88 个受试者，全为 train 集
│   ├── Audio/train/
│   │   ├── mfcc64/     88 人  文件: E1~3.npy     ← 注意: 目录名 mfcc64 非 mfcc
│   │   ├── opensmile/  88 人  文件: E1~3.npy
│   │   └── wav2vec2/   88 人  文件: E1~3.npy     ← 注意: 目录名 wav2vec2 非 wav2vec
│   ├── Video/train/
│   │   ├── densenet/   88 人  文件: event_1~3.npy
│   │   ├── resnet/     88 人  文件: event_1~3.npy
│   │   └── openface/   88 人  子目录: event_N/event_N_all.npy  ← 注意: 目录格式
│   ├── IMU/train/      88 人  文件: {id}.npy
│   ├── split_labels_train.csv          ← 88 行, 列: ID,split,label2,label3,phq9_score
│   └── descriptions_embeddings_with_ids.npy
├── MPDD-AVG-2026-main/                 # 基线代码 (已完整入库)
├── MPDD-main/                          # MPDD 2025 参考代码
├── MPDD-Elderly/                       # MPDD 2025 Elder 1s/5s 分段数据
├── MPDD-Young/                         # MPDD 2025 Young 1s/5s 分段数据
└── MPDD-Test/                          # MPDD 2025 测试集数据
```

### 数据统计

| 指标 | Elder | Young |
|------|-------|-------|
| 受试者数 | 87 | 88 |
| 每人 pair 数 | 1-4 | 1-3 |
| split 分布 | 全部 train | 全部 train |
| 特征目录 | mfcc, opensmile, wav2vec2, densenet, resnet, openface, IMU | mfcc64, opensmile, wav2vec2, densenet, resnet, openface, IMU |

### baseline 代码兼容性

dataset.py 已内置目录名兼容映射：
- `mfcc` → 自动匹配 `mfcc` 或 `mfcc64`
- `wav2vec` → 自动匹配 `wav2vec` 或 `wav2vec2` 或 `wav2vec2-FRA`
- Young openface 目录格式 (`event_N/event_N_all.npy`) → 已兼容

### 已知问题

1. **Young test 视频特征为空**: `MPDD-AVG2026-test/Young/Video/densenet`、`resnet`、`openface` 目录下的文件为 0 字节，Track2 含视频的任务在 test 阶段视频分支退化为零输入
2. **Young trainval OpenFace**: 部分样本无效，需参考训练日志中的有效样本数
3. **PHQ-9 log 变换**: 训练时使用 `log1p` 处理的 PHQ-9 目标值
4. **数据路径不匹配**: 数据在 `test/Elder/` 和 `test/Young/`，基线期望在 `MPDD-AVG2026/MPDD-AVG2026-trainval/`，运行前需传参指定
5. **无 test 集数据**: `MPDD-AVG2026/MPDD-AVG2026-test/` 为空，暂时只能训练无法测试
6. **CSV 格式不一致**: Elder CSV 列序 `split,label3,label2,ID,PHQ-9` (有 BOM)，Young CSV 列序 `ID,split,label2,label3,phq9_score`
