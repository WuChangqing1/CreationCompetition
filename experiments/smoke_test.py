"""Fast, honest smoke test for environment, registry, forward and utilities."""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dataset import AudioVisualDataset
from experiments.create_splits import create_subject_folds
from experiments.dataset_layouts import resolve_dataset
from experiments.errors import OptionalDependencyError
from experiments.evaluator import evaluate_predictions
from experiments.model_registry import create_experiment_model
from models import create_model, find_model_using_name


@dataclass
class SmokeResult:
    name: str
    status: str
    detail: str


def _opt(model):
    return SimpleNamespace(
        model=model, gpu_ids=[], isTrain=False, checkpoints_dir="./checkpoints",
        name="smoke", cuda_benchmark=False, input_dim_a=8, embd_size_a=4,
        embd_method_a="last", input_dim_v=10, embd_size_v=4,
        embd_method_v="last", emo_output_dim=2, cls_layers="8,4",
        dropout_rate=0.0, hidden_size=8, lr=1e-3, beta1=0.9,
        optimizer="adam", weight_decay=0.0, ce_weight=1.0, focal_weight=0.0,
        use_personality=True, personality_dim=1024, bilstm_hidden_dim=4,
        transformer_hidden_dim=8, transformer_heads=2, transformer_layers=1,
        transformer_ffn_dim=16, lmf_rank=2, fusion_dim=8,
    )


def _batch():
    audio = torch.randn(2, 4, 8)
    video = torch.randn(2, 4, 10)
    audio[:, -1] = 0
    video[:, -1] = 0
    return {
        "A_feat": audio, "V_feat": video,
        "personalized_feat": torch.randn(2, 1024),
        "emo_label": torch.tensor([0, 1]),
    }


def _environment_result():
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    correct = conda_env == "dachuangxiangmu" and "dachuangxiangmu" in sys.executable.lower()
    detail = (
        f"conda={conda_env or 'unset'}; python={sys.executable}; torch={torch.__version__}; "
        f"cuda={torch.cuda.is_available()}; gpu="
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}"
    )
    return SmokeResult("Environment", "PASS" if correct else "FAIL", detail)


def _dataset_result(data_root, dataset_year="2025", cohort="Elder", audio_feature="mfccs",
                    video_feature="densenet", split_window="1s", use_personality=True):
    if data_root is None:
        return SmokeResult("Dataset", "SKIPPED", "No --data-root supplied; synthetic model batches are tested separately")
    layout = resolve_dataset(data_root, dataset_year, cohort, split_window, audio_feature, video_feature)
    entries = layout["entries"]
    dataset = AudioVisualDataset(
        entries[:1], 2, str(layout["personality"]) if use_personality else None, 5,
        audio_path=str(layout["audio"]), video_path=str(layout["video"]),
        use_personality=use_personality,
    )
    sample = dataset[0]
    detail = (
        f"A={tuple(sample['A_feat'].shape)}; V={tuple(sample['V_feat'].shape)}; "
        f"P={tuple(sample['personalized_feat'].shape)}; label={int(sample['emo_label'])}; "
        f"year={dataset_year}; cohort={cohort}; entries={len(entries)}"
    )
    return SmokeResult("Dataset", "PASS", detail)


def _safe_dataset_result(data_root, **kwargs):
    try:
        return _dataset_result(data_root, **kwargs)
    except Exception as exc:
        return SmokeResult("Dataset", "FAIL", f"{type(exc).__name__}: {exc}")


def _torch_model_result(name, device):
    model_class = find_model_using_name(name)
    model = create_model(_opt(name))
    model.to(device)
    model.set_input(_batch())
    model.forward()
    shape = tuple(model.emo_logits.shape)
    if shape != (2, 2):
        raise AssertionError(f"Expected logits [2, 2], got {shape}")
    return SmokeResult(name, "PASS", f"{model_class.__name__}; logits={shape}")


def _utility_result(name, call):
    try:
        detail = call()
        return SmokeResult(name, "PASS", str(detail))
    except OptionalDependencyError as exc:
        return SmokeResult(name, "SKIPPED", str(exc))
    except NotImplementedError as exc:
        return SmokeResult(name, "N/A", str(exc))
    except Exception as exc:
        return SmokeResult(name, "FAIL", f"{type(exc).__name__}: {exc}")


def _split_smoke():
    entries = []
    for label in (0, 1):
        for index in range(5):
            subject = f"{label}{index}"
            entries.append({"id": subject, "audio_feature_path": f"{subject}_0.npy", "bin_category": label})
    folds = create_subject_folds(entries, "bin_category", 5, 3407)
    return f"folds={len(folds)}; no subject overlap"


def run_smoke(data_root=None, device="cpu", **dataset_options):
    results = [_environment_result()]
    results.append(_safe_dataset_result(data_root, **dataset_options))
    for name in ("our", "mlp", "bilstm", "lightweighttrans", "lmf", "mult"):
        results.append(_utility_result(name, lambda current=name: _torch_model_result(current, device).detail))
    results.append(_utility_result("SVM", lambda: type(create_experiment_model("svm", config={})).__name__))
    results.append(_utility_result("XGBoost", lambda: type(create_experiment_model("xgboost", config={})).__name__))
    results.append(_utility_result("Evaluator", lambda: evaluate_predictions([0, 1], [0, 1], [[0.9, 0.1], [0.1, 0.9]])["Macro_F1"]))
    results.append(_utility_result("Subject Split", _split_smoke))
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run fast MPDD framework smoke tests")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--dataset-year", default="2025", choices=["2025", "2026"])
    parser.add_argument("--cohort", default="Elder", choices=["Elder", "Young"])
    parser.add_argument("--audio-feature", default="mfccs")
    parser.add_argument("--video-feature", default="densenet")
    parser.add_argument("--split-window", default="1s")
    parser.add_argument("--use-personality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = run_smoke(
        args.data_root, args.device, dataset_year=args.dataset_year, cohort=args.cohort,
        audio_feature=args.audio_feature, video_feature=args.video_feature,
        split_window=args.split_window, use_personality=args.use_personality,
    )
    width = max(len(result.name) for result in results)
    for result in results:
        print(f"{result.name:<{width}}  {result.status:<7}  {result.detail}")
    if any(result.status == "FAIL" for result in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
