# 快速开始

## 1. 环境配置

### 环境名称

`creationcompetition`（conda 环境）

```bash
conda create -n creationcompetition python=3.10 -y
conda activate creationcompetition
pip install --upgrade pip
```

### 安装 PyTorch（CUDA 12.9）

```bash
pip install torch torchvision torchaudio
```

### 安装其余依赖

```bash
pip install numpy scikit-learn pandas tqdm
pip install transformers soundfile resampy librosa opensmile
pip install opencv-python Pillow
```

### 依赖清单

| 包名 | 用途 | 环境验证 |
|------|------|----------|
| torch | 深度学习框架 | 待安装 |
| torchvision | CNN 模型 (ResNet, DenseNet) | 待安装 |
| numpy | 数值计算 | 待安装 |
| scikit-learn | 评估指标、数据划分 | 待安装 |
| pandas | 数据处理 | 待安装 |
| tqdm | 进度条 | 待安装 |
| transformers | Wav2Vec2, RoBERTa 特征提取 | 待安装 |
| soundfile | 音频文件读取 | 待安装 |
| resampy | 音频重采样 | 待安装 |
| librosa | MFCC 特征提取 | 待安装 |
| opensmile | OpenSmile 特征提取 | 待安装 |
| opencv-python | 视频帧读取 | 待安装 |
| Pillow | 图像处理 | 待安装 |

### 验证安装

```bash
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python -c "import numpy, sklearn, pandas, tqdm, transformers, soundfile, resampy, librosa, opensmile, cv2, PIL; print('All OK')"
```

## 2. 数据集

### 本地数据位置

数据已就位，位于 `test/Elder/` 和 `test/Young/`：

| 项目 | Elder | Young |
|------|-------|-------|
| 受试者 | 87 | 88 |
| split | 全部 train | 全部 train |
| 特征目录 | mfcc, opensmile, wav2vec2, densenet, resnet, openface, IMU | mfcc64, opensmile, wav2vec2, densenet, resnet, openface, IMU |

> 2026 test 集尚未下载，`test/MPDD-AVG-2026-main/MPDD-AVG2026/MPDD-AVG2026-test/` 为空。

若需补充下载完整数据集（含 test）：
- [HuggingFace](https://huggingface.co/datasets/chasonfff/MPDD-AVG-2026/tree/main)

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
  --data_root ../../Elder \
  --split_csv ../../Elder/split_labels_train.csv \
  --personality_npy ../../Elder/descriptions_embeddings_with_ids.npy \
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
  --data_root ../../Young \
  --split_csv ../../Young/split_labels_train.csv \
  --personality_npy ../../Young/descriptions_embeddings_with_ids.npy \
  --device cuda
```

### 使用脚本批量运行

```bash
bash scripts/Track1/A-V-P/run_binary.sh      # Elder 二分类 9种特征组合
bash scripts/Track1/A-V-G+P/run_ternary.sh   # Elder 三分类
bash scripts/Track2/G-P/run_binary.sh        # Young Gait-only
```

## 4. 测试/评估

> test 集数据未下载，以下为预期用法。

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
