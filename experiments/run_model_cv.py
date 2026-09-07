"""Run one model over reusable subject-level folds without starting on import."""

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import AudioVisualDataset
from experiments.checkpointing import build_checkpoint_payload
from experiments.classical.common import pool_multimodal_features
from experiments.create_splits import LABEL_KEYS, create_subject_folds, validate_no_subject_leakage, write_subject_folds
from experiments.data_utils import extract_subject_id, filter_entries_by_subjects
from experiments.dataset_layouts import resolve_dataset
from experiments.efficiency import measure_torch_efficiency
from experiments.evaluator import evaluate_predictions, save_confusion_matrix, save_overall_confusion, save_predictions
from experiments.model_registry import create_experiment_model, get_model_kind


RAW_COLUMNS = [
    "Run_ID", "DatasetYear", "Cohort", "Model", "Fold", "Track", "Task", "AudioFeature", "VideoFeature",
    "UsePersonality", "SplitWindow",
    "Accuracy", "Macro_F1", "Weighted_F1", "Precision", "Recall",
    "Positive_Recall", "Specificity", "ROC_AUC", "Parameters", "Model_Size_MB",
    "Inference_Latency_ms", "Peak_VRAM_MB", "Loss_Function",
    "Class_Balancing_Strategy", "Seed", "Status",
]


def set_random_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_model_config(model_name, config_path=None):
    path = Path(config_path) if config_path else ROOT / "experiments" / "configs" / f"{model_name}.json"
    return load_json(path)


def build_run_id(args, config=None):
    condition = {
        "model": args.model.lower(), "track": args.track, "task": args.task,
        "dataset_year": getattr(args, "dataset_year", "2025"),
        "cohort": getattr(args, "cohort", "Elder"),
        "audio_feature": args.audio_feature, "video_feature": args.video_feature,
        "use_personality": args.use_personality, "split_window": args.split_window,
        "seed": args.seed, "folds": getattr(args, "folds", 5),
        "tiny": getattr(args, "tiny", False),
        "epochs": getattr(args, "epochs", None),
        "batch_size": getattr(args, "batch_size", None),
        "feature_max_len": getattr(args, "feature_max_len", None),
        "data_root": str(getattr(args, "data_root", None)),
        "dataset_fingerprint": getattr(args, "dataset_fingerprint", None),
        "config": config or {},
    }
    digest = hashlib.sha256(
        json.dumps(condition, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:12]
    return f"{condition['model']}-{digest}"


def resolve_data_paths(args):
    if args.data_root is None:
        env_root = os.environ.get("MPDD_DATA_ROOT")
        if not env_root:
            raise ValueError("Provide --data-root or set MPDD_DATA_ROOT")
        args.data_root = Path(env_root)
    return resolve_dataset(
        args.data_root, args.dataset_year, args.cohort, args.split_window,
        args.audio_feature, args.video_feature,
    )


def infer_feature_dims(entries, paths):
    if not entries:
        raise ValueError("Training JSON is empty")
    audio = np.load(paths["audio"] / entries[0]["audio_feature_path"])
    video = np.load(paths["video"] / entries[0]["video_feature_path"])
    if audio.ndim != 2 or video.ndim != 2:
        raise ValueError("Audio and video feature files must be two-dimensional")
    return int(audio.shape[1]), int(video.shape[1])


def build_opt(model_name, config, input_dim_a, input_dim_v, classes, feature_max_len, run_name):
    base = load_json(ROOT / "config.json")
    base.update(config)
    base.update({
        "model": model_name,
        "input_dim_a": input_dim_a,
        "input_dim_v": input_dim_v,
        "emo_output_dim": classes,
        "feature_max_len": feature_max_len,
        "personality_dim": 1024,
        "use_personality": bool(config.get("use_personality", True)),
        "dropout_rate": float(config.get("dropout", base.get("dropout_rate", 0.2))),
        "lr": float(config.get("learning_rate", base.get("lr", 1e-3))),
        "gpu_ids": [],
        "isTrain": True,
        "cuda_benchmark": False,
        "name": run_name,
    })
    return SimpleNamespace(**base)


def make_dataset(entries, args, paths):
    return AudioVisualDataset(
        entries, args.classes, str(paths["personality"]) if args.use_personality else None, args.feature_max_len,
        batch_size=args.batch_size, audio_path=str(paths["audio"]), video_path=str(paths["video"]),
        use_personality=args.use_personality,
    )


def collect_classical_arrays(dataset):
    audio, video, personality, labels = [], [], [], []
    for index in range(len(dataset)):
        item = dataset[index]
        audio.append(item["A_feat"].numpy())
        video.append(item["V_feat"].numpy())
        personality.append(item["personalized_feat"].numpy())
        labels.append(int(item["emo_label"]))
    return np.stack(audio), np.stack(video), np.stack(personality), np.asarray(labels)


def evaluate_torch(model, loader, device):
    model.eval()
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            model.set_input(batch)
            model.forward()
            labels.extend(batch["emo_label"].cpu().numpy().tolist())
            probabilities.extend(model.emo_pred.detach().cpu().numpy().tolist())
    probabilities = np.asarray(probabilities)
    return np.asarray(labels), probabilities.argmax(1), probabilities


def append_raw_result(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows = []
    if path.exists():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            existing_rows = list(csv.DictReader(handle))
    identity = (str(row.get("Run_ID", "N/A")), str(row.get("Model", "N/A")), str(row.get("Fold", "N/A")))
    existing_rows = [
        saved for saved in existing_rows
        if (saved.get("Run_ID", "N/A"), saved.get("Model", "N/A"), saved.get("Fold", "N/A")) != identity
    ]
    existing_rows.append({column: row.get(column, "N/A") for column in RAW_COLUMNS})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
        writer.writeheader()
        writer.writerows(existing_rows)
    temporary.replace(path)


def run_fold(args, fold_data, entries, paths, config):
    set_random_seed(args.seed)
    validate_no_subject_leakage(fold_data["train_ids"], fold_data["val_ids"])
    train_entries = filter_entries_by_subjects(entries, fold_data["train_ids"])
    val_entries = filter_entries_by_subjects(entries, fold_data["val_ids"])
    train_dataset = make_dataset(train_entries, args, paths)
    val_dataset = make_dataset(val_entries, args, paths)
    fold_number = int(fold_data["fold"])
    model_name = args.model.lower()
    dataset_year = getattr(args, "dataset_year", "2025")
    cohort = getattr(args, "cohort", "Elder")
    run_id = getattr(args, "run_id", None) or build_run_id(args, config)
    resolved_config = dict(config)
    resolved_config["seed"] = args.seed
    feature_config = {
        "dataset_year": dataset_year, "cohort": cohort,
        "track": args.track, "task": args.task, "audio_feature": args.audio_feature,
        "video_feature": args.video_feature, "use_personality": args.use_personality,
        "split_window": args.split_window,
    }
    efficiency = {key: "N/A" for key in ("Parameters", "Model_Size_MB", "Inference_Latency_ms", "Peak_VRAM_MB")}

    artifact_dir = args.results_dir / "raw" / run_id / model_name / f"fold_{fold_number}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "config.json").write_text(json.dumps(resolved_config, ensure_ascii=False, indent=2), encoding="utf-8")

    if get_model_kind(model_name) == "classical":
        train_a, train_v, train_p, train_y = collect_classical_arrays(train_dataset)
        val_a, val_v, val_p, val_y = collect_classical_arrays(val_dataset)
        train_x = pool_multimodal_features(train_a, train_v, train_p, args.use_personality)
        val_x = pool_multimodal_features(val_a, val_v, val_p, args.use_personality)
        model = create_experiment_model(model_name, config=resolved_config)
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(val_x)
        predictions = probabilities.argmax(1)
        labels = val_y
    else:
        input_dim_a, input_dim_v = infer_feature_dims(entries, paths)
        opt = build_opt(model_name, resolved_config, input_dim_a, input_dim_v, args.classes, args.feature_max_len, f"{model_name}_fold{fold_number}")
        opt.use_personality = args.use_personality
        model = create_experiment_model(model_name, opt=opt)
        device = torch.device(args.device)
        model.to(device)
        generator = torch.Generator().manual_seed(args.seed + fold_number)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        epochs = 1 if args.tiny else args.epochs
        for _ in range(epochs):
            model.train(True)
            for batch_index, batch in enumerate(train_loader):
                model.set_input(batch)
                model.optimize_parameters(batch_index)
                if args.tiny:
                    break
        labels, predictions, probabilities = evaluate_torch(model, val_loader, device)
        sample_batch = next(iter(val_loader))
        model.set_input(sample_batch)
        efficiency = measure_torch_efficiency(model, model.forward, device, args.warmup, args.measure)
        checkpoint = build_checkpoint_payload(model_name, model.state_dict(), vars(opt), feature_config, fold_number, args.seed)
        torch.save(checkpoint, artifact_dir / "checkpoint.pth")
        (artifact_dir / "config.json").write_text(json.dumps(vars(opt), ensure_ascii=False, indent=2), encoding="utf-8")

    metrics = evaluate_predictions(labels, predictions, probabilities)
    subject_ids = [extract_subject_id(entry) for entry in val_entries]
    save_predictions(
        args.results_dir / "predictions" / f"{run_id}_{model_name}_fold{fold_number}.csv",
        subject_ids, labels, predictions, probabilities, fold_number, model_name,
    )
    save_confusion_matrix(
        metrics["Confusion_Matrix"],
        args.results_dir / "confusion_matrix" / f"{run_id}_{model_name}_fold{fold_number}.png",
        f"{model_name} Fold {fold_number}",
    )
    row = {
        "Run_ID": run_id, "DatasetYear": dataset_year, "Cohort": cohort,
        "Model": model_name, "Fold": fold_number,
        "Track": args.track, "Task": args.task, "AudioFeature": args.audio_feature,
        "VideoFeature": args.video_feature, "UsePersonality": args.use_personality,
        "SplitWindow": args.split_window,
        **{key: metrics[key] for key in metrics if key != "Confusion_Matrix"},
        **efficiency,
        "Loss_Function": config.get("loss_function", "N/A"),
        "Class_Balancing_Strategy": config.get("class_balancing_strategy", "None"),
        "Seed": args.seed, "Status": "PASS",
    }
    append_raw_result(args.results_dir / "raw_results.csv", row)
    return row


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run one MPDD model with reusable subject-level CV")
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset-year", default="2025", choices=["2025", "2026"])
    parser.add_argument("--cohort", choices=["Elder", "Young"])
    parser.add_argument("--track", default="Track1", choices=["Track1", "Track2"])
    parser.add_argument("--task", default="binary", choices=["binary"], help="The new comparison pipeline currently supports binary classification only")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--audio-feature", default="mfccs")
    parser.add_argument("--video-feature", default="densenet")
    parser.add_argument("--use-personality", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-window", default="1s")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feature-max-len", type=int, default=5)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "results")
    parser.add_argument("--force-splits", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measure", type=int, default=50)
    args = parser.parse_args(argv)
    args.cohort = args.cohort or ("Elder" if args.track == "Track1" else "Young")
    args.splits_dir = args.splits_dir or ROOT / "experiments" / "splits" / args.dataset_year / args.cohort
    config = load_model_config(args.model.lower(), args.config)
    args.batch_size = args.batch_size or int(config.get("batch_size", 32))
    args.epochs = args.epochs or int(config.get("epochs", 20))
    args.classes = {"binary": 2, "ternary": 3, "quinary": 5}[args.task]
    return args, config


def main(argv=None):
    args, config = parse_args(argv)
    set_random_seed(args.seed)
    paths = resolve_data_paths(args)
    entries = paths["entries"]
    folds = create_subject_folds(entries, LABEL_KEYS[args.task], args.folds, args.seed)
    args.dataset_fingerprint = folds[0]["dataset_fingerprint"]
    args.run_id = build_run_id(args, config)
    split_paths = write_subject_folds(folds, args.splits_dir, args.force_splits)
    rows = []
    for split_path in split_paths:
        rows.append(run_fold(args, load_json(split_path), entries, paths, config))
    prediction_paths = [
        args.results_dir / "predictions" / f"{args.run_id}_{args.model.lower()}_fold{fold['fold']}.csv"
        for fold in folds
    ]
    save_overall_confusion(
        prediction_paths,
        args.results_dir / "confusion_matrix" / f"{args.run_id}_{args.model.lower()}_overall.png",
        args.model.lower(),
    )
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
