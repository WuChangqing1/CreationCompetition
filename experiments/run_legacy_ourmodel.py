"""Reproduce the historical MPDD 2025 OurModel single-checkpoint test."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ACCURACY = 213 / 227
EXPECTED_MACRO_F1 = 0.875898156825992
EXPECTED_CONFUSION_MATRIX = [[187, 0], [14, 26]]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the historical OurModel checkpoint without cross-validation"
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="2025 MPDD-Test/MPDD-Elderly directory",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Historical best_model_2026-07-29-21.42.51.pth",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser.parse_args(argv)


def build_test_command(args, python_executable=None):
    output_dir = args.output_dir
    if output_dir is None:
        stamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
        output_dir = ROOT / "experiments" / "results_legacy_ourmodel_2025" / stamp
    output_dir = Path(output_dir).resolve()
    metrics_path = output_dir / "metrics.json"
    executable = python_executable or sys.executable
    command = [
        str(executable),
        str(ROOT / "test.py"),
        "--labelcount", "2",
        "--track_option", "Track1",
        "--feature_max_len", "26",
        "--data_rootpath", str(Path(args.data_root).resolve()),
        "--train_model", str(Path(args.checkpoint).resolve()),
        "--audiofeature_method", "mfccs",
        "--videofeature_method", "densenet",
        "--splitwindow_time", "1s",
        "--batch_size", "8",
        "--lr", "2e-05",
        "--device", args.device,
        "--model", "our",
        "--output_dir", str(output_dir),
        "--metrics_json", str(metrics_path),
    ]
    return command, metrics_path


def validate_inputs(args):
    checkpoint = Path(args.checkpoint)
    data_root = Path(args.data_root)
    required = [
        checkpoint,
        data_root / "labels" / "Testing_files.json",
        data_root / "individualEmbedding" / "descriptions_embeddings_with_ids.npy",
        data_root / "1s" / "Audio" / "mfccs",
        data_root / "1s" / "Visual" / "densenet",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("历史复现输入缺失：\n" + "\n".join(missing))


def validate_historical_result(metrics):
    matrix = metrics.get("confusion_matrix")
    accuracy = float(metrics.get("accuracy", -1))
    macro_f1 = float(metrics.get("macro_f1", -1))
    if (
        matrix != EXPECTED_CONFUSION_MATRIX
        or abs(accuracy - EXPECTED_ACCURACY) > 1e-12
        or abs(macro_f1 - EXPECTED_MACRO_F1) > 1e-12
    ):
        raise RuntimeError(
            "历史结果未复现："
            f"accuracy={accuracy:.6f}, macro_f1={macro_f1:.6f}, confusion_matrix={matrix}"
        )


def main(argv=None):
    args = parse_args(argv)
    validate_inputs(args)

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝使用 CPU 执行历史 OurModel 复现")

    command, metrics_path = build_test_command(args)
    metrics_path.parent.mkdir(parents=True, exist_ok=False)
    run_metadata = {
        "protocol": "historical_ourmodel_single_checkpoint",
        "cross_validation": False,
        "folds": 0,
        "checkpoint_count": 1,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()),
        "device": args.device,
        "feature_max_len": 26,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "audio_feature": "mfccs",
        "video_feature": "densenet",
        "subject_aggregation": "majority_vote_then_event_backfill",
    }
    (metrics_path.parent / "run_config.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    if not metrics_path.exists():
        raise RuntimeError(f"测试完成但未生成指标文件：{metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    validate_historical_result(metrics)
    print("[PASS] 历史 OurModel 结果已完全复现")
    print(f"Accuracy={metrics['accuracy']:.4f}")
    print(f"Macro-F1={metrics['macro_f1']:.4f}")
    print(f"Confusion Matrix={metrics['confusion_matrix']}")
    print(f"Results={metrics_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
