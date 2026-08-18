# 快速开始

## 1. 环境配置

### 环境名称

`dachuangxiangmu`（conda 环境）

```bash
conda create -n dachuangxiangmu python=3.10 -y
conda activate dachuangxiangmu
pip install --upgrade pip
```

### 安装 PyTorch（CUDA 13 / RTX 5070）

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
| torch | 深度学习框架 | 已验证：2.14.0.dev20260717+cu130 |
| torchvision | CNN 模型 (ResNet, DenseNet) | 待复核 |
| numpy | 数值计算 | 待复核 |
| scikit-learn | 评估指标、数据划分 | 待复核 |
| pandas | 数据处理 | 待复核 |
| tqdm | 进度条 | 待复核 |
| transformers | Wav2Vec2, RoBERTa 特征提取 | 待复核 |
| soundfile | 音频文件读取 | 待复核 |
| resampy | 音频重采样 | 待复核 |
| librosa | MFCC 特征提取 | 待复核 |
| opensmile | OpenSmile 特征提取 | 待复核 |
| opencv-python | 视频帧读取 | 待复核 |
| Pillow | 图像处理 | 待复核 |

### 验证安装

```bash
python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.cuda.is_available())"
python -c "import numpy, sklearn, pandas, tqdm, transformers, soundfile, resampy, librosa, opensmile, cv2, PIL; print('All OK')"
```

当前机器可直接用：

```powershell
& 'D:\App\Business\Coding\Python\Miniconda\envs\dachuangxiangmu\python.exe' -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

已验证输出：PyTorch 2.14.0.dev20260717+cu130，CUDA=True，GPU=NVIDIA GeForce RTX 5070 Laptop GPU。

### Windows 注意事项

当前 PowerShell 里 `conda run` 可能因为 GBK 编码崩溃；优先直接调用环境解释器：

```powershell
& 'D:\App\Business\Coding\Python\Miniconda\envs\dachuangxiangmu\python.exe' .\smoke_test.py
```

如果 `opensmile/audresample` 报找不到 `bin\win_\audresample.dll`，先在当前 PowerShell 会话设置：

```powershell
$env:PROCESSOR_ARCHITECTURE = 'AMD64'
& 'D:\App\Business\Coding\Python\Miniconda\envs\dachuangxiangmu\python.exe' .\smoke_test.py
```

2026-07-30 已用该方式通过完整 smoke test。

## 2. 数据集

### 本地数据位置

数据已就位，位于 `test/Elder/` 和 `test/Young/`：

| 项目 | Elder | Young |
|------|-------|-------|
| 受试者 | 87 | 88 |
| split | 全部 train | 全部 train |
| 特征目录 | mfcc, opensmile, wav2vec2, densenet, resnet, openface, IMU | mfcc64, opensmile, wav2vec2, densenet, resnet, openface, IMU |

> 当前没有 2026 official test 集，`test/MPDD-AVG-2026-main/MPDD-AVG2026/MPDD-AVG2026-test/` 为空。`test/MPDD-Test/` 是 MPDD 2025 参考测试数据，不等同于 MPDD-AVG 2026 official test。这个缺口不影响训练/CV 跑通，只影响最终 official test 评估和 CodaBench 提交。

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

### 5-fold CV + CMG-VS 消融（当前主线）

主实验入口是 `experiments/train_cv.py`。CMG-VS 论文忠实化后的两组对照：

```powershell
$env:PROCESSOR_ARCHITECTURE = 'AMD64'
$py = 'D:\App\Business\Coding\Python\Miniconda\envs\dachuangxiangmu\python.exe'

# 公平基线：无 CVAE 且无 PHQ 回归头（与 CVAE 组同口径）
& $py experiments/train_cv.py --track Track1 --task binary --subtrack A-V+P --folds 5 --cls_only --num_workers 0

# CMG-VS：任务引导视觉合成（序列级 L_consis + detach + λ_aug=1.0）
& $py experiments/train_cv.py --track Track1 --task binary --subtrack A-V+P --folds 5 --use_cvae --num_workers 0
```

- `--num_workers 0`：受限/沙箱环境下必须单进程加载（避免 Windows 命名管道被拦截）；普通本机可省略（默认 2）。
- `--cls_only` / `--use_cvae` 会自动选择对应 `TRAIN_CFG` 变体，并把 checkpoint 写入 `experiments/cv_checkpoints/.../{variant}/`，互不覆盖。
- 分类-only 实验只看 Macro-F1 / Acc / Kappa；日志里的 `ccc/rmse/mae` 是对类别标签派生的，不是 PHQ-9 回归指标。

## 4. 测试/评估

> 当前没有 2026 official test 集，以下为拿到 official test 后的预期用法。

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
