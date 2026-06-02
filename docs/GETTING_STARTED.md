# 快速开始

## 1. 环境配置

```bash
conda create -n mpddavg python=3.10 -y
conda activate mpddavg
pip install --upgrade pip

# PyTorch (根据 CUDA 版本选择)
pip install torch torchvision torchaudio

# 其余依赖
pip install numpy scikit-learn
```

## 2. 数据集准备

从 [HuggingFace](https://huggingface.co/datasets/chasonfff/MPDD-AVG-2026/tree/main) 下载数据集，放置在 `test/MPDD-AVG-2026-main/` 下：

```
test/MPDD-AVG-2026-main/MPDD-AVG2026/
├── MPDD-AVG2026-trainval/
│   ├── Elder/
│   │   ├── Audio/    (wav2vec, mfcc, opensmile)
│   │   ├── Video/    (densenet, resnet, openface)
│   │   ├── IMU/      (gait)
│   │   ├── split_labels_train.csv
│   │   └── descriptions_embeddings_with_ids.npy
│   └── Young/
│       └── ...
└── MPDD-AVG2026-test/
    ├── Elder/
    └── Young/
```

## 3. 运行训练

### Track1 (Elder) 二分类 — A-V+P

```bash
cd test/MPDD-AVG-2026-main
python train.py \
  --track Track1 \
  --task binary \
  --subtrack A-V+P \
  --encoder_type bilstm_mean \
  --audio_feature wav2vec \
  --video_feature resnet \
  --device cuda
```

### Track2 (Young) 三分类 — A-V-G+P

```bash
python train.py \
  --track Track2 \
  --task ternary \
  --subtrack A-V-G+P \
  --encoder_type hybrid_attn \
  --audio_feature wav2vec \
  --video_feature resnet \
  --data_root MPDD-AVG2026/MPDD-AVG2026-trainval/Young \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-trainval/Young/split_labels_train.csv \
  --personality_npy MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy \
  --device cuda
```

### 使用脚本批量运行

```bash
bash scripts/Track1/A-V-P/run_binary.sh      # Elder 二分类 9种特征组合
bash scripts/Track1/A-V-G+P/run_ternary.sh   # Elder 三分类
bash scripts/Track2/G-P/run_binary.sh        # Young Gait-only
```

## 4. 测试/评估

```bash
python test.py \
  --checkpoint checkpoints/Track2/A-V-G+P/ternary/{exp_name}/best_model_*.pth \
  --data_root MPDD-AVG2026/MPDD-AVG2026-test/Young \
  --split_csv MPDD-AVG2026/MPDD-AVG2026-test/Young/split_labels_test.csv \
  --personality_npy MPDD-AVG2026/MPDD-AVG2026-trainval/Young/descriptions_embeddings_with_ids.npy
```

## 5. 生成提交文件

```bash
cd make_submission_forcodabench
python make_submission_sample.py \
  --binary_csv binary.csv \
  --ternary_csv ternary.csv \
  --binary_sample binary_sample.csv \
  --ternary_sample ternary_sample.csv \
  --output_dir submission
```

提交 `submission/submission.zip` 到 CodaBench。

## 6. 常用环境变量覆盖

```bash
DEVICE=cpu EPOCHS=10 BATCH_SIZE=16 LR=1e-4 TARGET_T=256 \
  bash scripts/Track1/A-V-P/run_binary.sh
```

| 变量 | 默认值 | 说明 |
|------|--------|------|
| DEVICE | cuda | 设备 |
| EPOCHS | 80 | 训练轮数 |
| BATCH_SIZE | 8 | 批次大小 |
| LR | 3e-4 | 学习率 |
| WEIGHT_DECAY | 1e-4 | 权重衰减 |
| HIDDEN_DIM | 64 | 隐层维度 |
| DROPOUT | 0.5 | Dropout 率 |
| PATIENCE | 20 | Early stopping 耐心 |
| TARGET_T | 128 | 时序插值长度 |
