"""
批量基线训练脚本 — 替代 12 个 shell 脚本，Windows 原生可用

用法:
    # 列出所有训练任务（不执行）
    python run_baseline.py --list

    # 跑单个任务 (Track1 Elder G+P binary)
    python run_baseline.py --track Track1 --subtrack G+P --task binary

    # 跑某个赛道所有任务
    python run_baseline.py --track Track2

    # 只在某个子赛道上跑
    python run_baseline.py --track Track1 --subtrack A-V-P

    # 仅跑 G+P（无特征组合，适合快速试跑）
    python run_baseline.py --track Track1 --subtrack G+P

    # 快速模式：epochs=5 纯验证用
    python run_baseline.py --track Track1 --subtrack G+P --quick
"""
import os, sys, argparse, subprocess, time
from pathlib import Path
from itertools import product

PROJECT_ROOT = Path(__file__).resolve().parent
BASELINE_DIR = PROJECT_ROOT / "test" / "MPDD-AVG-2026-main"
TRAIN_SCRIPT = BASELINE_DIR / "train.py"

# ============================================================
# 配置定义 — 来自 12 个 shell 脚本
# ============================================================

# 数据路径（修正为实际本地路径）
DATA_PATHS = {
    "Track1": {
        "data_root": "../../test/Elder",
        "split_csv": "../../test/Elder/split_labels_train.csv",
        "personality_npy": "../../test/Elder/descriptions_embeddings_with_ids.npy",
    },
    "Track2": {
        "data_root": "../../test/Young",
        "split_csv": "../../test/Young/split_labels_train.csv",
        "personality_npy": "../../test/Young/descriptions_embeddings_with_ids.npy",
    },
}

# G+P 子赛道 — 不需要 audio/video 特征循环
# 注意: train.py 不接受 --selection_metric/--cls_loss_weight/--reg_loss_weight/
# --weighted_sampler/--label_smoothing CLI 参数（shell 脚本与当前 train.py 版本不匹配）。
# 选择指标硬编码为 f1（分类）/ ccc（回归），损失为 CrossEntropyLoss + MSELoss 等权。
GP_CONFIGS = {
    "Track1": {
        "binary": {
            "epochs": 320, "batch_size": 2, "lr": 8e-5, "weight_decay": 1e-5,
            "hidden_dim": 128, "dropout": 0.3, "patience": 90, "seed": 42,
        },
        "ternary": {
            "epochs": 300, "batch_size": 2, "lr": 5e-5, "weight_decay": 1e-5,
            "hidden_dim": 128, "dropout": 0.4, "patience": 70, "seed": 42,
        },
    },
    "Track2": {
        "binary": {
            "epochs": 100, "batch_size": 8, "lr": 3e-4, "weight_decay": 1e-4,
            "hidden_dim": 64, "dropout": 0.5, "patience": 20, "seed": 3407,
        },
        "ternary": {
            "epochs": 100, "batch_size": 8, "lr": 3e-4, "weight_decay": 1e-4,
            "hidden_dim": 64, "dropout": 0.5, "patience": 20, "seed": 3407,
        },
    },
}

# A-V+P / A-V-G+P — 特征组合循环 + 每赛道不同超参
AVP_CONFIGS = {
    ("Track1", "A-V+P", "binary"): {
        "audio_features": ["opensmile"], "video_features": ["resnet"],
        "epochs": 60, "batch_size": 8, "lr": 3e-5, "weight_decay": 1e-5,
        "hidden_dim": 64, "dropout": 0.4, "patience": 20, "seed": 42,
        "encoder_type": "bilstm_mean",
    },
    ("Track1", "A-V+P", "ternary"): {
        "audio_features": ["mfcc"], "video_features": ["resnet"],
        "epochs": 140, "batch_size": 4, "lr": 2e-4, "weight_decay": 5e-5,
        "hidden_dim": 160, "dropout": 0.5, "patience": 30, "seed": 42,
        "encoder_type": "bilstm_mean",
    },
    ("Track1", "A-V-G+P", "binary"): {
        "audio_features": ["mfcc"], "video_features": ["resnet"],
        "epochs": 140, "batch_size": 4, "lr": 8e-5, "weight_decay": 1e-5,
        "hidden_dim": 128, "dropout": 0.45, "patience": 35, "seed": 42,
        "encoder_type": "bilstm_mean",
    },
    ("Track1", "A-V-G+P", "ternary"): {
        "audio_features": ["wav2vec"], "video_features": ["openface"],
        "epochs": 160, "batch_size": 4, "lr": 8e-5, "weight_decay": 1e-5,
        "hidden_dim": 160, "dropout": 0.4, "patience": 40, "seed": 42,
        "encoder_type": "bilstm_mean",
    },
    ("Track2", "A-V+P", "binary"): {
        "audio_features": ["mfcc", "opensmile", "wav2vec"],
        "video_features": ["densenet", "resnet", "openface"],
        "epochs": 80, "batch_size": 8, "lr": 5e-4, "weight_decay": 1e-4,
        "hidden_dim": 64, "dropout": 0.4, "patience": 15, "seed": 3407,
        "encoder_type": "bilstm_mean",
    },
    ("Track2", "A-V+P", "ternary"): {
        "audio_features": ["mfcc", "opensmile", "wav2vec"],
        "video_features": ["densenet", "resnet", "openface"],
        "epochs": 80, "batch_size": 8, "lr": 5e-4, "weight_decay": 1e-4,
        "hidden_dim": 64, "dropout": 0.4, "patience": 15, "seed": 3407,
        "encoder_type": "bilstm_mean",
    },
    ("Track2", "A-V-G+P", "binary"): {
        "audio_features": ["mfcc", "opensmile", "wav2vec"],
        "video_features": ["densenet", "resnet", "openface"],
        "epochs": 100, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4,
        "hidden_dim": 64, "dropout": 0.5, "patience": 20, "seed": 3407,
        "encoder_type": "bilstm_mean",
    },
    ("Track2", "A-V-G+P", "ternary"): {
        "audio_features": ["mfcc", "opensmile", "wav2vec"],
        "video_features": ["densenet", "resnet", "openface"],
        "epochs": 100, "batch_size": 16, "lr": 1e-3, "weight_decay": 1e-4,
        "hidden_dim": 64, "dropout": 0.5, "patience": 20, "seed": 3407,
        "encoder_type": "bilstm_mean",
    },
}


def build_cmd(track, subtrack, task, cfg, audio_feature=None, video_feature=None, quick=False):
    """构建 train.py 命令行"""
    data = DATA_PATHS[track]
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--track", track,
        "--task", task,
        "--subtrack", subtrack,
        "--encoder_type", cfg.get("encoder_type", "bilstm_mean"),
        "--data_root", data["data_root"],
        "--split_csv", data["split_csv"],
        "--personality_npy", data["personality_npy"],
        "--seed", str(cfg["seed"]),
        "--epochs", "5" if quick else str(cfg["epochs"]),
        "--batch_size", str(cfg["batch_size"]),
        "--lr", str(cfg["lr"]),
        "--weight_decay", str(cfg["weight_decay"]),
        "--hidden_dim", str(cfg["hidden_dim"]),
        "--dropout", str(cfg["dropout"]),
        "--patience", str(cfg["patience"]),
        "--target_t", str(cfg.get("target_t", 128)),
        "--device", "cuda",
    ]
    if audio_feature:
        cmd += ["--audio_feature", audio_feature]
    if video_feature:
        cmd += ["--video_feature", video_feature]
    return cmd


def get_tasks(track=None, subtrack=None, task=None):
    """生成需要执行的训练任务列表"""
    tasks = []

    # G+P 子赛道
    for t in ["Track1", "Track2"]:
        if track and t != track:
            continue
        if subtrack and subtrack != "G+P":
            continue
        for tk in ["binary", "ternary"]:
            if task and tk != task:
                continue
            cfg = GP_CONFIGS[t][tk]
            tasks.append((t, "G+P", tk, cfg, None, None))

    # A-V+P / A-V-G+P 子赛道（含特征组合循环）
    for (t, st, tk), cfg in AVP_CONFIGS.items():
        if track and t != track:
            continue
        if subtrack and st != subtrack:
            continue
        if task and tk != task:
            continue
        for af, vf in product(cfg["audio_features"], cfg["video_features"]):
            tasks.append((t, st, tk, cfg, af, vf))

    return tasks


def list_tasks():
    """列出所有训练任务"""
    tasks = get_tasks()
    print(f"共 {len(tasks)} 个训练任务:\n")
    for i, (track, subtrack, task, cfg, af, vf) in enumerate(tasks):
        label = f"[{track}][{subtrack}][{task}]"
        feat = f"  audio={af} video={vf}" if af else ""
        print(f"  {i+1:3d}. {label}{feat}")


def run_task(track, subtrack, task, cfg, af, vf, quick, dry_run):
    """执行单个训练任务"""
    label = f"[{track}][{subtrack}][{task}]"
    if af:
        label += f" audio={af} video={vf}"
    cmd = build_cmd(track, subtrack, task, cfg, af, vf, quick)

    print(f"\n{'=' * 60}")
    print(f">>> {label}")
    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return True
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    os.chdir(BASELINE_DIR)
    start = time.time()
    result = subprocess.run(cmd, cwd=str(BASELINE_DIR))
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"-- DONE ({elapsed:.0f}s) --")
        return True
    else:
        print(f"-- FAILED (exit={result.returncode}) --")
        return False


def main():
    parser = argparse.ArgumentParser(description="MPDD-AVG 2026 批量基线训练")
    parser.add_argument("--track", choices=["Track1", "Track2"])
    parser.add_argument("--subtrack", choices=["A-V+P", "A-V-G+P", "G+P"])
    parser.add_argument("--task", choices=["binary", "ternary"])
    parser.add_argument("--list", action="store_true", help="列出所有任务")
    parser.add_argument("--quick", action="store_true", help="快速模式 (epochs=5)")
    parser.add_argument("--dry-run", action="store_true", help="仅打印命令不执行")
    args = parser.parse_args()

    if args.list:
        list_tasks()
        return

    tasks = get_tasks(args.track, args.subtrack, args.task)
    if not tasks:
        print("没有匹配的训练任务")
        return

    print(f"将执行 {len(tasks)} 个训练任务")
    success = 0
    for track, subtrack, task, cfg, af, vf in tasks:
        ok = run_task(track, subtrack, task, cfg, af, vf, args.quick, args.dry_run)
        if ok:
            success += 1

    print(f"\n完成: {success}/{len(tasks)} 成功")


if __name__ == "__main__":
    main()
