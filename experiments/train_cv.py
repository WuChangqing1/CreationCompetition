"""
K-Fold 交叉验证训练脚本

替代单次 9:1 随机划分，用 StratifiedKFold 做 K 折交叉验证，
得到稳定的验证指标估计（mean ± std）。

用法:
    # G+P binary 5-fold CV (Elder)
    python experiments/train_cv.py --track Track1 --task binary --subtrack G+P --folds 5

    # G+P binary 5-fold CV (Young)
    python experiments/train_cv.py --track Track2 --task binary --subtrack G+P --folds 5

    # G+P binary 5-fold CV + Elder&Young 联合训练
    python experiments/train_cv.py --joint --task binary --subtrack G+P --folds 5

    # 快速冒烟测试 (epochs=3)
    python experiments/train_cv.py --track Track1 --task binary --subtrack G+P --quick

特点:
    - seed 在 split 之前设置（修复了 train.py 的 seed 顺序 bug）
    - 每 fold 独立训练，互不干扰
    - 自动汇总 CV metrics（mean ± std per metric）
    - 保存 best model per fold + CV summary JSON
    - joint 模式正确处理 Elder/Young ID 重叠（61个重叠ID）
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

# 将 baseline 代码加入 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = PROJECT_ROOT / "test" / "MPDD-AVG-2026-main"
sys.path.insert(0, str(BASELINE_DIR))

from dataset import (  # noqa: E402
    REGRESSION_TASK, MPDDElderDataset, collate_batch,
    get_phq9_target, get_task_label, infer_input_dims,
)
from metrics import evaluate_model  # noqa: E402
from models import TorchcatBaseline, kl_divergence  # noqa: E402
from train import FocalLoss  # noqa: E402
from train_val_split import create_kfold_splits  # noqa: E402

CV_LOG_ROOT = PROJECT_ROOT / "experiments" / "cv_logs"
CV_CHECKPOINT_ROOT = PROJECT_ROOT / "experiments" / "cv_checkpoints"


# ══════════════════════════ 训练配置 ══════════════════════════

TRAIN_CFG = {
    # ---- Track1 Elder (DepFormer) ----
    ("Track1", "G+P", "binary"): {
        "epochs": 320, "batch_size": 2, "lr": 8e-5, "weight_decay": 1e-5,
        "hidden_dim": 64, "dropout": 0.5, "patience": 90,
        "encoder_type": "depformer",
    },
    ("Track1", "G+P", "ternary"): {
        "epochs": 300, "batch_size": 2, "lr": 5e-5, "weight_decay": 1e-5,
        "hidden_dim": 64, "dropout": 0.4, "patience": 70,
        "encoder_type": "bilstm_mean",
    },
    ("Track1", "A-V+P", "binary"): {
        "epochs": 200, "batch_size": 8, "lr": 1e-4, "weight_decay": 1e-5,
        "hidden_dim": 128, "dropout": 0.55, "patience": 60,
        "encoder_type": "depformer", "audio_feature": "mfcc", "video_feature": "densenet",
    },
    # ---- CVAE data augmentation (CMG-VS) ----
    ("Track1", "A-V+P", "binary", "cvae"): {
        "epochs": 200, "batch_size": 8, "lr": 1e-4, "weight_decay": 1e-5,
        "hidden_dim": 128, "dropout": 0.55, "patience": 60,
        "encoder_type": "depformer", "audio_feature": "mfcc", "video_feature": "densenet",
        "use_cvae": True, "cvae_d_z": 16, "cvae_num_layers": 1, "cvae_num_heads": 2,
        "lambda_aug": 0.5, "lambda_cvae": 0.1, "beta_kl": 0.01,
        "gradient_clip": 0.5,
    },
    ("Track1", "A-V-G+P", "binary", "cvae"): {
        "epochs": 200, "batch_size": 8, "lr": 1e-4, "weight_decay": 1e-5,
        "hidden_dim": 128, "dropout": 0.55, "patience": 60,
        "encoder_type": "depformer", "audio_feature": "mfcc", "video_feature": "densenet",
        "use_cvae": True, "cvae_d_z": 16, "cvae_num_layers": 1, "cvae_num_heads": 2,
        "lambda_aug": 0.5, "lambda_cvae": 0.1, "beta_kl": 0.01,
        "gradient_clip": 0.5,
    },
    ("Track2", "A-V+P", "binary", "cvae"): {
        "epochs": 200, "batch_size": 8, "lr": 1e-4, "weight_decay": 1e-5,
        "hidden_dim": 128, "dropout": 0.55, "patience": 60,
        "encoder_type": "depformer", "audio_feature": "mfcc", "video_feature": "densenet",
        "use_cvae": True, "cvae_d_z": 16, "cvae_num_layers": 1, "cvae_num_heads": 2,
        "lambda_aug": 0.5, "lambda_cvae": 0.1, "beta_kl": 0.01,
        "gradient_clip": 0.5,
    },
    ("Track1", "A-V+P", "ternary"): {
        "epochs": 200, "batch_size": 4, "lr": 4e-5, "weight_decay": 1e-5,
        "hidden_dim": 64, "dropout": 0.35, "patience": 40,
        "encoder_type": "bilstm_mean", "audio_feature": "mfcc", "video_feature": "densenet",
    },
    ("Track1", "A-V-G+P", "binary"): {
        "epochs": 320, "batch_size": 2, "lr": 8e-5, "weight_decay": 1e-5,
        "hidden_dim": 64, "dropout": 0.5, "patience": 90,
        "encoder_type": "depformer", "audio_feature": "mfcc", "video_feature": "densenet",
    },
    # ---- Track2 Young ----
    ("Track2", "G+P", "binary"): {
        "epochs": 320, "batch_size": 2, "lr": 8e-5, "weight_decay": 1e-5,
        "hidden_dim": 64, "dropout": 0.5, "patience": 90,
        "encoder_type": "depformer",
    },
    ("Track2", "G+P", "ternary"): {
        "epochs": 200, "batch_size": 4, "lr": 5e-5, "weight_decay": 1e-5,
        "hidden_dim": 64, "dropout": 0.4, "patience": 40,
        "encoder_type": "bilstm_mean",
    },
    ("Track2", "A-V+P", "binary"): {
        "epochs": 320, "batch_size": 2, "lr": 8e-5, "weight_decay": 1e-5,
        "hidden_dim": 64, "dropout": 0.5, "patience": 90,
        "encoder_type": "depformer", "audio_feature": "mfcc", "video_feature": "densenet",
    },
}

DATA_PATHS = {
    "Track1": {
        "data_root": PROJECT_ROOT / "test" / "Elder",
        "split_csv": PROJECT_ROOT / "test" / "Elder" / "split_labels_train.csv",
        "personality_npy": PROJECT_ROOT / "test" / "Elder" / "descriptions_embeddings_with_ids.npy",
    },
    "Track2": {
        "data_root": PROJECT_ROOT / "test" / "Young",
        "split_csv": PROJECT_ROOT / "test" / "Young" / "split_labels_train.csv",
        "personality_npy": PROJECT_ROOT / "test" / "Young" / "descriptions_embeddings_with_ids.npy",
    },
}

JOINT = {
    "elder_root": PROJECT_ROOT / "test" / "Elder",
    "young_root": PROJECT_ROOT / "test" / "Young",
    "elder_csv": PROJECT_ROOT / "test" / "Elder" / "split_labels_train.csv",
    "young_csv": PROJECT_ROOT / "test" / "Young" / "split_labels_train.csv",
    "elder_personality": PROJECT_ROOT / "test" / "Elder" / "descriptions_embeddings_with_ids.npy",
    "young_personality": PROJECT_ROOT / "test" / "Young" / "descriptions_embeddings_with_ids.npy",
}


# ══════════════════════════ 工具函数 ══════════════════════════

def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_num_classes(task: str, regression_label: str) -> int:
    if task == "binary":
        return 2
    if task == "ternary":
        return 3
    if task == REGRESSION_TASK:
        return 2 if regression_label == "label2" else 3
    raise ValueError(f"Unsupported task: {task}")


def build_class_weights(labels: list[int], num_classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=num_classes).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def summarize_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    array_keys = {"ids", "y_true", "y_pred", "class_true", "class_pred",
                   "phq_true", "phq_pred", "confusion_matrix"}
    return {k: v for k, v in metrics.items() if k not in array_keys}


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


# ══════════════════════════ 联合数据集 ══════════════════════════

class JointDataset(torch.utils.data.Dataset):
    """将 Elder + Young 的 MPDDElderDataset 拼接为一个数据集。"""

    def __init__(self, elder_ds: MPDDElderDataset, young_ds: MPDDElderDataset):
        self.elder = elder_ds
        self.young = young_ds
        self.elder_n = len(elder_ds)
        self.young_n = len(young_ds)

    def __len__(self) -> int:
        return self.elder_n + self.young_n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx < self.elder_n:
            return self.elder[idx]
        return self.young[idx - self.elder_n]

    @property
    def samples(self) -> list[dict[str, Any]]:
        return self.elder.samples + self.young.samples


# ══════════════════════════ 单 fold 训练 ══════════════════════════

def _make_dataset(data_root, label_map, source_split_map, args, cfg,
                  personality_npy, phq_map):
    return MPDDElderDataset(
        data_root=str(data_root),
        label_map=label_map,
        source_split_map=source_split_map,
        subtrack=args.subtrack,
        task=args.task,
        audio_feature=cfg.get("audio_feature", "mfcc"),
        video_feature=cfg.get("video_feature", "densenet"),
        personality_npy=str(personality_npy),
        phq_map=phq_map,
        target_t=cfg.get("target_t", 128),
    )


def train_one_fold(
    fold_idx: int,
    total_folds: int,
    split_payload: dict[str, Any],
    args: argparse.Namespace,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """训练一个 fold，返回 best_val_metrics summary。"""
    # ── 构建数据集 ──
    if args.joint:
        # 联合模式：分别构建 Elder 和 Young 的子数据集
        elder_train = _make_dataset(
            JOINT["elder_root"],
            split_payload["elder_train_map"],
            split_payload["source_split_map"],
            args, cfg, JOINT["elder_personality"],
            split_payload.get("elder_train_phq"),
        )
        young_train = _make_dataset(
            JOINT["young_root"],
            split_payload["young_train_map"],
            split_payload["source_split_map"],
            args, cfg, JOINT["young_personality"],
            split_payload.get("young_train_phq"),
        )
        elder_val = _make_dataset(
            JOINT["elder_root"],
            split_payload["elder_val_map"],
            split_payload["source_split_map"],
            args, cfg, JOINT["elder_personality"],
            split_payload.get("elder_val_phq"),
        )
        young_val = _make_dataset(
            JOINT["young_root"],
            split_payload["young_val_map"],
            split_payload["source_split_map"],
            args, cfg, JOINT["young_personality"],
            split_payload.get("young_val_phq"),
        )
        train_dataset = JointDataset(elder_train, young_train)
        val_dataset = JointDataset(elder_val, young_val)
        track_name = "Joint"
    else:
        data = DATA_PATHS[args.track]
        train_dataset = _make_dataset(
            data["data_root"], split_payload["train_map"],
            split_payload["source_split_map"], args, cfg,
            data["personality_npy"], split_payload.get("train_phq_map"),
        )
        val_dataset = _make_dataset(
            data["data_root"], split_payload["val_map"],
            split_payload["source_split_map"], args, cfg,
            data["personality_npy"], split_payload.get("val_phq_map"),
        )
        track_name = args.track

    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"],
                              shuffle=True, collate_fn=collate_batch,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.get("val_batch_size", cfg["batch_size"]),
                            shuffle=False, collate_fn=collate_batch,
                            num_workers=2, pin_memory=True)

    # ── 构建模型 ──
    input_dims = infer_input_dims(train_dataset)
    num_classes = get_num_classes(args.task, args.regression_label)
    model = TorchcatBaseline(
        subtrack=args.subtrack, num_classes=num_classes,
        is_regression=False,
        use_regression_head=False if cfg.get("use_cvae") else True,
        audio_dim=input_dims["audio_dim"], video_dim=input_dims["video_dim"],
        gait_dim=input_dims["gait_dim"], hidden_dim=cfg["hidden_dim"],
        dropout=cfg["dropout"], encoder_type=cfg["encoder_type"],
        use_asp=args.use_asp, use_cross_fusion=args.use_cross_fusion,
        use_cvae=cfg.get("use_cvae", False),
        cvae_d_z=cfg.get("cvae_d_z", 16),
        cvae_num_layers=cfg.get("cvae_num_layers", 1),
        cvae_num_heads=cfg.get("cvae_num_heads", 2),
    ).to(device)

    if device.type == "cuda":
        gpu_mem = torch.cuda.memory_allocated() / 1024**2
        print(f"  [GPU] 模型已加载到 {torch.cuda.get_device_name(0)}, 显存占用: {gpu_mem:.1f} MB")

    all_labels = [int(s["label"]) for s in train_dataset.samples]
    class_weights = build_class_weights(all_labels, num_classes, device)
    focal_criterion = FocalLoss(alpha=class_weights, gamma=2.0) if args.use_focal else None
    use_cvae = cfg.get("use_cvae", False)
    if use_cvae:
        # Single criterion (evaluate_model uses non-joint path → compatible with no reg head)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = (
            nn.CrossEntropyLoss(weight=class_weights),
            focal_criterion,
            nn.MSELoss(),
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, cfg["epochs"]))

    # ── 训练循环 ──
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_score = -1.0
    best_epoch = 0
    best_val_metrics = None
    epochs_without_improve = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda"):
                if cfg.get("use_cvae"):
                    # ── CVAE dual-stream mode ──
                    outputs = model(
                        audio=batch["audio"].to(device, non_blocking=True) if "audio" in batch else None,
                        video=batch["video"].to(device, non_blocking=True) if "video" in batch else None,
                        gait=batch["gait"].to(device, non_blocking=True) if "gait" in batch else None,
                        personality=batch["personality"].to(device, non_blocking=True),
                        pair_mask=batch["pair_mask"].to(device, non_blocking=True) if "pair_mask" in batch else None,
                        return_cvae_outputs=True,
                    )
                    cls_criterion = criterion

                    L_real = cls_criterion(outputs["logits_real"], labels)
                    if focal_criterion is not None:
                        L_real = L_real + 0.5 * focal_criterion(outputs["logits_real"], labels)

                    L_aug = cls_criterion(outputs["logits_aug"], labels)
                    if focal_criterion is not None:
                        L_aug = L_aug + 0.5 * focal_criterion(outputs["logits_aug"], labels)

                    L_consis = torch.nn.functional.l1_loss(
                        outputs["v_pooled_real"], outputs["v_pooled_synth"])
                    L_kl = kl_divergence(outputs["mu"], outputs["logvar"])

                    lmd_aug = cfg.get("lambda_aug", 0.5)
                    lmd_cvae = cfg.get("lambda_cvae", 0.1)
                    beta_kl = cfg.get("beta_kl", 0.01)

                    loss = L_real + lmd_aug * L_aug + lmd_cvae * (L_consis + beta_kl * L_kl)
                else:
                    # ── Normal mode ──
                    outputs = model(
                        audio=batch["audio"].to(device, non_blocking=True) if "audio" in batch else None,
                        video=batch["video"].to(device, non_blocking=True) if "video" in batch else None,
                        gait=batch["gait"].to(device, non_blocking=True) if "gait" in batch else None,
                        personality=batch["personality"].to(device, non_blocking=True),
                        pair_mask=batch["pair_mask"].to(device, non_blocking=True) if "pair_mask" in batch else None,
                    )
                    criterion_cls, criterion_focal, criterion_reg = criterion
                    logits, reg_out = outputs
                    cls_loss = criterion_cls(logits, labels)
                    focal_loss = criterion_focal(logits, labels) if criterion_focal is not None else 0.0
                    loss = cls_loss + 0.5 * focal_loss + criterion_reg(reg_out, batch["phq9"].to(device, non_blocking=True))

            # backward with AMP scaler
            clip_val = cfg.get("gradient_clip", 1.0)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                optimizer.step()
            running_loss += float(loss.item()) * len(labels)

        scheduler.step()
        val_metrics = evaluate_model(model, val_loader, criterion, device, args.task)

        current_score = float(val_metrics["selection_score"])
        if current_score > best_score + cfg.get("min_delta", 0.0001):
            best_score = current_score
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_without_improve = 0
            ckpt_dir = CV_CHECKPOINT_ROOT / track_name / args.subtrack / args.task
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "subtrack": args.subtrack, "task": args.task,
                "encoder_type": cfg["encoder_type"],
                "best_epoch": epoch, "fold": fold_idx,
                "best_val_metrics": summarize_metrics(val_metrics),
            }, ckpt_dir / f"fold{fold_idx}_best.pth")
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= cfg["patience"]:
                break

    if best_val_metrics is None:
        best_val_metrics = val_metrics

    s = summarize_metrics(best_val_metrics)
    print(f"  Fold {fold_idx}/{total_folds} | {track_name} | "
          f"epoch={best_epoch} | f1={s.get('f1', 0):.4f} | "
          f"kappa={s.get('kappa', 0):.4f} | ccc={s.get('ccc', 0):.4f} | "
          f"train={len(train_dataset)} val={len(val_dataset)}")
    return s


# ══════════════════════════ 联合数据 CV 划分 ══════════════════════════

def _build_joint_folds(args: argparse.Namespace) -> list[dict[str, Any]]:
    """合并 Elder + Young CSV，用数组索引做 StratifiedKFold，
    然后分解为 per-cohort 的 label_map，避免 ID 重叠问题。"""
    elder_rows = _load_csv_rows(JOINT["elder_csv"])
    young_rows = _load_csv_rows(JOINT["young_csv"])

    # 标记来源
    for r in elder_rows:
        r["_cohort"] = "elder"
    for r in young_rows:
        r["_cohort"] = "young"

    all_rows = elder_rows + young_rows  # 175 rows
    reg_label = args.regression_label
    labels = [get_task_label(r, args.task, reg_label) for r in all_rows]
    indices = list(range(len(all_rows)))

    label_counts = Counter(int(l) for l in labels)
    has_stratify = label_counts and min(label_counts.values()) >= args.folds

    folds: list[dict[str, Any]] = []
    for train_idx, val_idx in StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed,
    ).split(indices, labels if has_stratify else indices):
        train_set = set(int(i) for i in train_idx)
        val_set = set(int(i) for i in val_idx)
        source_split_map = {int(r["ID"]): "train" for r in all_rows}

        # Elder train/val maps
        elder_train_map = {}
        elder_val_map = {}
        elder_train_phq = {}
        elder_val_phq = {}
        # Young train/val maps
        young_train_map = {}
        young_val_map = {}
        young_train_phq = {}
        young_val_phq = {}

        for i, r in enumerate(all_rows):
            pid = int(r["ID"])
            label = get_task_label(r, args.task, reg_label)
            phq = get_phq9_target(r)
            if i in train_set:
                if r["_cohort"] == "elder":
                    elder_train_map[pid] = label
                    elder_train_phq[pid] = phq
                else:
                    young_train_map[pid] = label
                    young_train_phq[pid] = phq
            else:
                if r["_cohort"] == "elder":
                    elder_val_map[pid] = label
                    elder_val_phq[pid] = phq
                else:
                    young_val_map[pid] = label
                    young_val_phq[pid] = phq

        folds.append({
            "elder_train_map": elder_train_map,
            "elder_val_map": elder_val_map,
            "young_train_map": young_train_map,
            "young_val_map": young_val_map,
            "elder_train_phq": elder_train_phq,
            "elder_val_phq": elder_val_phq,
            "young_train_phq": young_train_phq,
            "young_val_phq": young_val_phq,
            "source_split_map": source_split_map,
            "rows": all_rows,
        })
    return folds


# ══════════════════════════ 主入口 ══════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-Fold CV 训练")
    p.add_argument("--track", default="Track1", choices=["Track1", "Track2"])
    p.add_argument("--task", default="binary",
                   choices=["binary", "ternary", REGRESSION_TASK])
    p.add_argument("--regression_label", default="label2", choices=["label2", "label3"])
    p.add_argument("--subtrack", default="G+P",
                   choices=["A-V+P", "A-V-G+P", "G+P"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--joint", action="store_true", help="Elder+Young 联合训练")
    p.add_argument("--quick", action="store_true", help="快速模式 (epochs=3)")
    p.add_argument("--device", default="cuda")
    # Champion method toggles
    p.add_argument("--encoder_type", default=None,
                   choices=["bilstm_mean", "hybrid_attn", "depformer"],
                   help="覆盖 TRAIN_CFG 中的 encoder_type")
    p.add_argument("--use_asp", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--use_cross_fusion", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--use_focal", type=lambda x: x.lower() != "false", default=True)
    p.add_argument("--use_cvae", action="store_true", help="启用 CVAE 数据增强 (CMG-VS)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)  # ★ seed 在 split 之前，修复原始 bug

    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        device = torch.device("cuda")
        print(f"Device: cuda (GPU: {torch.cuda.get_device_name(0)}, "
              f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB) | Seed: {args.seed}")
    else:
        device = torch.device("cpu")
        print(f"Device: cpu | Seed: {args.seed}")

    # ── 配置 ──
    track_name = "Joint" if args.joint else args.track
    if args.joint:
        cfg_key = ("Track1", args.subtrack, args.task)
    else:
        cfg_key = (args.track, args.subtrack, args.task)
    # Try CVAE variant first if requested
    if args.use_cvae:
        cvae_key = cfg_key + ("cvae",)
        if cvae_key in TRAIN_CFG:
            cfg_key = cvae_key
    cfg = TRAIN_CFG.get(cfg_key)
    if cfg is None:
        print(f"错误: 未找到配置 {cfg_key}，请在 TRAIN_CFG 中添加")
        sys.exit(1)
    cfg = dict(cfg)
    if args.encoder_type is not None:
        cfg["encoder_type"] = args.encoder_type
    if args.quick:
        cfg["epochs"] = 3
        cfg["patience"] = 1

    # ── 生成 K-Fold ──
    if args.joint:
        folds = _build_joint_folds(args)
    else:
        folds = create_kfold_splits(
            split_csv=str(DATA_PATHS[args.track]["split_csv"]),
            task=args.task, n_splits=args.folds,
            regression_label=args.regression_label, seed=args.seed,
        )

    n_folds = len(folds)
    print(f"Track: {track_name} | Subtrack: {args.subtrack} | "
          f"Task: {args.task} | Folds: {n_folds}")
    print(f"Config: {cfg}")
    print("=" * 60)

    # ── 逐 fold 训练 ──
    fold_results: list[dict[str, Any]] = []
    t0 = time.time()

    for i, payload in enumerate(folds, start=1):
        fold_results.append(
            train_one_fold(i, n_folds, payload, args, cfg, device))

    elapsed = time.time() - t0

    # ── CV 汇总 ──
    metric_names = ["f1", "acc", "kappa", "ccc", "rmse", "mae",
                    "loss", "cls_loss", "reg_loss"]
    if args.task == REGRESSION_TASK:
        metric_names.append("r2")

    summary: dict[str, Any] = {
        "track": track_name, "subtrack": args.subtrack, "task": args.task,
        "folds": n_folds, "seed": args.seed, "joint": args.joint,
        "config": cfg, "elapsed_sec": round(elapsed, 1),
    }
    print(f"\n{'=' * 60}")
    print(f"CV 汇总 ({n_folds} folds, {elapsed:.0f}s)")
    print("=" * 60)
    for m in metric_names:
        values = [r[m] for r in fold_results if m in r and r[m] is not None]
        if values:
            mv = float(np.mean(values))
            sv = float(np.std(values))
            summary[f"cv_{m}_mean"] = round(mv, 6)
            summary[f"cv_{m}_std"] = round(sv, 6)
            print(f"  {m:12s} = {mv:.4f} ± {sv:.4f}")

    # 额外：打印每 fold 的 f1 序列
    f1s = [r.get("f1", 0) for r in fold_results]
    print(f"  per-fold f1  = {[f'{v:.4f}' for v in f1s]}")

    # ── 保存 ──
    log_dir = CV_LOG_ROOT / track_name / args.subtrack / args.task
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    result_path = log_dir / f"cv_result_{n_folds}fold_{ts}.json"
    with open(result_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    print(f"\nCV 结果已保存: {result_path}")


if __name__ == "__main__":
    main()
