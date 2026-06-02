"""冒烟测试 — 验证环境、数据、训练流程是否正常"""
import os, sys, io

# Windows GBK 编码兼容
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# 将 baseline 代码加入 path
PROJECT_ROOT = os.path.dirname(__file__)
BASELINE_DIR = os.path.join(PROJECT_ROOT, "test", "MPDD-AVG-2026-main")
sys.path.insert(0, BASELINE_DIR)

def check_env():
    """检查 Python 环境和依赖"""
    print("=" * 50)
    print("[1/4] 环境检查")
    print(f"  Python: {sys.version.split()[0]}")
    import torch; print(f"  PyTorch: {torch.__version__}  CUDA: {torch.cuda.is_available()}")
    import numpy; print(f"  numpy: {numpy.__version__}")
    import sklearn; print(f"  sklearn: {sklearn.__version__}")
    import pandas; print(f"  pandas: {pandas.__version__}")
    import transformers; print(f"  transformers: {transformers.__version__}")
    import librosa; print(f"  librosa: {librosa.__version__}")
    import cv2; print(f"  opencv: {cv2.__version__}")
    import soundfile, resampy, opensmile, tqdm, PIL
    print("  所有依赖 OK\n")

def check_data():
    """检查数据文件是否存在"""
    print("=" * 50)
    print("[2/4] 数据检查")
    checks = {
        "Elder CSV": "test/Elder/split_labels_train.csv",
        "Elder personality": "test/Elder/descriptions_embeddings_with_ids.npy",
        "Elder IMU": "test/Elder/IMU/train",
        "Elder Audio/mfcc": "test/Elder/Audio/train/mfcc",
        "Young CSV": "test/Young/split_labels_train.csv",
        "Young personality": "test/Young/descriptions_embeddings_with_ids.npy",
        "Young IMU": "test/Young/IMU/train",
    }
    all_ok = True
    import pandas as pd
    for name, rel_path in checks.items():
        abs_path = os.path.join(PROJECT_ROOT, rel_path)
        ok = os.path.exists(abs_path)
        flag = "OK" if ok else "MISSING"
        extra = ""
        if ok and name == "Elder CSV":
            df = pd.read_csv(abs_path)
            extra = f" ({len(df)} rows, cols: {', '.join(df.columns.tolist())})"
        if ok and name == "Young CSV":
            df = pd.read_csv(abs_path)
            extra = f" ({len(df)} rows, cols: {', '.join(df.columns.tolist())})"
        if ok and name.endswith("IMU"):
            count = len(os.listdir(abs_path))
            extra = f" ({count} subjects)"
        print(f"  [{flag}] {name}{extra}")
        if not ok:
            all_ok = False
    print()
    return all_ok

def check_split():
    """验证 train_val_split 能正常运行"""
    print("=" * 50)
    print("[3/4] 数据划分检查")
    from train_val_split import create_train_val_split

    elder_csv = os.path.join(PROJECT_ROOT, "test", "Elder", "split_labels_train.csv")

    payload = create_train_val_split(
        split_csv=elder_csv, task="binary", val_ratio=0.1
    )
    train_count = len(payload["train_map"])
    val_count = len(payload["val_map"])
    # 统计类别分布
    from collections import Counter
    train_dist = Counter(payload["train_map"].values())
    val_dist = Counter(payload["val_map"].values())
    print(f"  Train: {train_count} samples, class dist: {dict(train_dist)}")
    print(f"  Val:   {val_count} samples, class dist: {dict(val_dist)}")
    print()

def run_mini_train():
    """最小训练测试 (G+P, 3 epochs)"""
    print("=" * 50)
    print("[4/4] 训练流程测试 (G+P, 3 epochs)")
    import torch
    import numpy as np
    import random
    from torch.utils.data import DataLoader
    from dataset import MPDDElderDataset, collate_batch, infer_input_dims
    from models import TorchcatBaseline
    from metrics import evaluate_model
    from train_val_split import create_train_val_split

    # 固定随机种子
    torch.manual_seed(3407)
    np.random.seed(3407)
    random.seed(3407)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_root = os.path.join(PROJECT_ROOT, "test", "Elder")
    split_csv = os.path.join(data_root, "split_labels_train.csv")
    personality_npy = os.path.join(data_root, "descriptions_embeddings_with_ids.npy")

    # 数据划分
    payload = create_train_val_split(
        split_csv=split_csv, task="binary", val_ratio=0.1
    )
    num_classes = len(set(payload["train_map"].values()))

    # 构建 dataset — 使用正确的 API
    train_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=payload["train_map"],
        source_split_map=payload["source_split_map"],
        subtrack="G+P",
        task="binary",
        audio_feature="mfcc",
        video_feature="resnet",
        personality_npy=personality_npy,
        phq_map=payload.get("train_phq_map"),
        target_t=128,
    )
    val_dataset = MPDDElderDataset(
        data_root=data_root,
        label_map=payload["val_map"],
        source_split_map=payload["source_split_map"],
        subtrack="G+P",
        task="binary",
        audio_feature="mfcc",
        video_feature="resnet",
        personality_npy=personality_npy,
        phq_map=payload.get("val_phq_map"),
        target_t=128,
    )

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_batch)

    input_dims = infer_input_dims(train_dataset)
    print(f"  Input dims: {input_dims}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"  Num classes: {num_classes}")

    model = TorchcatBaseline(
        subtrack="G+P",
        num_classes=num_classes,
        audio_dim=input_dims.get("audio_dim", 0),
        video_dim=input_dims.get("video_dim", 0),
        gait_dim=input_dims.get("gait_dim", 0),
        encoder_type="bilstm_mean",
        hidden_dim=64,
        dropout=0.5,
        use_regression_head=True,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    cls_criterion = torch.nn.CrossEntropyLoss()
    reg_criterion = torch.nn.MSELoss()

    criterion = (cls_criterion, reg_criterion)

    best_score = 0
    for epoch in range(3):
        model.train()
        total_loss = 0
        for batch in train_loader:
            labels = batch["label"].to(device)
            outputs = model(
                audio=batch["audio"].to(device) if "audio" in batch else None,
                video=batch["video"].to(device) if "video" in batch else None,
                gait=batch["gait"].to(device) if "gait" in batch else None,
                personality=batch["personality"].to(device),
                pair_mask=batch["pair_mask"].to(device) if "pair_mask" in batch else None,
            )
            logits, reg_out = outputs
            phq9 = batch["phq9"].to(device)
            loss = cls_criterion(logits, labels) + reg_criterion(reg_out, phq9)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        metrics = evaluate_model(model, val_loader, criterion, device, "binary")
        f1 = metrics.get("macro_f1", 0)
        best_score = max(best_score, f1)
        print(f"  Epoch {epoch + 1}  loss: {total_loss / len(train_loader):.4f}  val_f1: {f1:.4f}")

    print(f"\n  Best val Macro-F1: {best_score:.4f}")
    print()

def main():
    os.chdir(BASELINE_DIR)  # 基线代码相对路径解析需要在其目录下
    check_env()
    if not check_data():
        print("Data incomplete, please check and retry")
        return
    check_split()
    try:
        run_mini_train()
        print("=" * 50)
        print("Smoke test PASSED!")
    except Exception as e:
        print(f"\nTraining error: {e}")
        import traceback; traceback.print_exc()
        print("\nSmoke test FAILED, check the error above")

if __name__ == "__main__":
    main()
