"""Run the historical 2025 single-split eight-model comparison on CUDA."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.legacy_comparison import (
    APPROVED_MODELS,
    HISTORICAL_SEED,
    build_execution_plan,
    execute_plan,
    load_historical_split,
    normalize_models,
)


def _model_list(value):
    try:
        return normalize_models(value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=_model_list, default=list(APPROVED_MODELS))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    args = parser.parse_args(argv)
    args.data_root = args.data_root.expanduser().resolve()
    args.historical_checkpoint = args.historical_checkpoint.expanduser().resolve()
    if args.output_dir is not None:
        args.output_dir = args.output_dir.expanduser().resolve()
    return args


def _default_output_dir():
    stamp = time.strftime("%Y-%m-%d-%H.%M.%S", time.localtime())
    return ROOT / "experiments" / "results_legacy_comparison_2025" / stamp


def _runtime_args(cli_args, output_dir):
    return SimpleNamespace(
        data_root=cli_args.data_root,
        device=cli_args.device,
        output_dir=output_dir,
        dataset_year="2025",
        cohort="Elder",
        track="Track1",
        task="binary",
        audio_feature="mfccs",
        video_feature="densenet",
        use_personality=True,
        personality_id_source="filename",
        split_window="1s",
        seed=HISTORICAL_SEED,
        feature_max_len=26,
        batch_size=8,
        epochs=300,
        classes=2,
    )


def _result_from_historical_metrics(metrics):
    from experiments.independent_test import RESULT_COLUMNS

    tn, fp = metrics["confusion_matrix"][0]
    fn, tp = metrics["confusion_matrix"][1]
    precision_0 = tn / (tn + fn) if tn + fn else 0.0
    precision_1 = tp / (tp + fp) if tp + fp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    row = {
        "DatasetYear": "2025", "Cohort": "Elder", "Model": "our",
        "EvaluationLevel": "legacy_voted_event", "Samples": metrics["sample_count"],
        "Accuracy": metrics["accuracy"], "Macro_F1": metrics["macro_f1"],
        "Weighted_F1": metrics["weighted_f1"],
        "Precision": (precision_0 + precision_1) / 2,
        "Recall": metrics["balanced_accuracy"],
        "Positive_Recall": positive_recall, "Specificity": specificity,
        "ROC_AUC": "N/A", "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "EnsembleFolds": 0, "Run_ID": "historical-2026-07-29-21.42.51",
        "Device": "cuda", "Seed": HISTORICAL_SEED, "Status": "PASS",
    }
    return {column: row[column] for column in RESULT_COLUMNS}


def _write_rows(path, rows):
    from experiments.independent_test import RESULT_COLUMNS

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _verify_historical_our(item, args):
    from experiments.run_legacy_ourmodel import main as reproduce_our

    model_output = args.output_dir / "our_historical"
    test_root = args.data_root / "MPDD-Test" / "MPDD-Elderly"
    return_code = reproduce_our([
        "--data-root", str(test_root),
        "--checkpoint", str(item["checkpoint"]),
        "--output-dir", str(model_output),
        "--device", args.device,
    ])
    if return_code != 0:
        raise RuntimeError(f"历史 OurModel 复现失败，退出码 {return_code}")
    metrics = json.loads((model_output / "metrics.json").read_text(encoding="utf-8"))
    return _result_from_historical_metrics(metrics)


def _legacy_paths(args):
    from experiments.dataset_layouts import resolve_dataset, resolve_independent_test

    training = resolve_dataset(
        args.data_root, "2025", "Elder", "1s", "mfccs", "densenet",
    )
    testing = resolve_independent_test(
        args.data_root, "2025", "Elder", "1s", "mfccs", "densenet",
    )
    training_json = training["track_root"] / "Training" / "labels" / "Training_Validation_files.json"
    train_entries, val_entries = load_historical_split(training_json)
    return training, testing, train_entries, val_entries


def _dataset(entries, args, paths):
    from experiments.run_model_cv import make_dataset

    return make_dataset(entries, args, paths)


def _legacy_voted_result(model_name, probabilities, test_entries, args, run_id, device):
    from experiments.data_utils import extract_subject_id
    from experiments.independent_test import (
        aggregate_subject_predictions,
        broadcast_subject_predictions,
        build_result_row,
        write_prediction_csv,
    )

    labels = np.asarray([int(entry["bin_category"]) for entry in test_entries], dtype=int)
    subject_ids = [extract_subject_id(entry) for entry in test_entries]
    unique_ids, _, subject_probabilities = aggregate_subject_predictions(
        subject_ids, labels, probabilities, method="majority_vote",
    )
    voted = broadcast_subject_predictions(subject_ids, unique_ids, subject_probabilities)
    row = build_result_row(
        year="2025", cohort="Elder", model=model_name, level="legacy_voted_event",
        labels=labels, probabilities=voted, folds=0, run_id=run_id,
        device=device, seed=HISTORICAL_SEED,
    )
    write_prediction_csv(
        args.output_dir / "predictions" / f"{model_name}.csv",
        subject_ids, labels, voted, model_name, "legacy_voted_event",
    )
    return row


def _train_baseline(item, args, paths):
    import torch
    from torch.utils.data import DataLoader

    from experiments.classical.common import pool_multimodal_features
    from experiments.evaluator import evaluate_predictions
    from experiments.model_registry import create_experiment_model, get_model_kind
    from experiments.protocols import resolve_protocol
    from experiments.run_model_cv import (
        build_opt,
        collect_classical_arrays,
        evaluate_torch,
        infer_feature_dims,
        is_better_validation,
        load_model_config,
        set_random_seed,
    )

    model_name = item["model"]
    training_paths, test_paths, train_entries, val_entries = paths
    set_random_seed(HISTORICAL_SEED)
    config = load_model_config(model_name)
    _, config = resolve_protocol("legacy_bicfnet", model_name, config)
    config.update({
        "seed": HISTORICAL_SEED,
        "feature_max_len": 26,
        "batch_size": 8,
        "epochs": 300,
        "checkpoint_selection": "val_macro_f1",
        "personality_id_source": "filename",
    })
    if model_name == "xgboost":
        config.update(device="cuda", tree_method="hist")

    train_dataset = _dataset(train_entries, args, training_paths)
    val_dataset = _dataset(val_entries, args, training_paths)
    test_dataset = _dataset(test_paths["entries"], args, test_paths)
    run_id = f"legacy-single-split-{model_name}-seed{HISTORICAL_SEED}"
    artifact_dir = args.output_dir / "models" / model_name
    artifact_dir.mkdir(parents=True, exist_ok=False)
    (artifact_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    if get_model_kind(model_name) == "classical":
        import joblib

        print(f"[RUN] {model_name}: 单次旧版划分训练与独立测试")
        train_a, train_v, train_p, train_y = collect_classical_arrays(train_dataset)
        test_a, test_v, test_p, _ = collect_classical_arrays(test_dataset)
        train_x = pool_multimodal_features(train_a, train_v, train_p, True)
        test_x = pool_multimodal_features(test_a, test_v, test_p, True)
        model = create_experiment_model(model_name, config=config)
        model.fit(train_x, train_y)
        probabilities = np.asarray(model.predict_proba(test_x), dtype=float)
        joblib.dump(model, artifact_dir / "model.joblib")
        device = "cpu:svm" if model_name == "svm" else "cuda:xgboost"
    else:
        print(f"[RUN] {model_name}: 300 epochs，按验证集 Macro-F1 保存最佳轮次")
        device_object = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，拒绝用 CPU 训练正式对比模型")
        input_dim_a, input_dim_v = infer_feature_dims(training_paths["entries"], training_paths)
        opt = build_opt(model_name, config, input_dim_a, input_dim_v, 2, 26, run_id)
        opt.use_personality = True
        opt.device = "cuda"
        model = create_experiment_model(model_name, opt=opt)
        model.to(device_object)
        generator = torch.Generator().manual_seed(HISTORICAL_SEED)
        train_loader = DataLoader(
            train_dataset, batch_size=8, shuffle=True, generator=generator, pin_memory=True,
        )
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, pin_memory=True)
        scheduler = None
        if config.get("scheduler") == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                model.optimizer, T_max=300, eta_min=float(config.get("scheduler_eta_min", 1e-6)),
            )
        best_macro_f1 = None
        best_epoch = None
        best_state = None
        history_path = artifact_dir / "training_history.csv"
        history_handle = history_path.open("w", encoding="utf-8-sig", newline="")
        history_writer = csv.DictWriter(
            history_handle, fieldnames=("epoch", "validation_macro_f1", "best_epoch", "best_macro_f1"),
        )
        history_writer.writeheader()
        for epoch in range(1, 301):
            model.train(True)
            for batch in train_loader:
                model.set_input(batch)
                model.optimize_parameters(epoch - 1)
            labels, predictions, probabilities = evaluate_torch(model, val_loader, device_object)
            macro_f1 = evaluate_predictions(labels, predictions, probabilities)["Macro_F1"]
            if is_better_validation(macro_f1, best_macro_f1):
                best_macro_f1 = macro_f1
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            history_writer.writerow({
                "epoch": epoch,
                "validation_macro_f1": macro_f1,
                "best_epoch": best_epoch,
                "best_macro_f1": best_macro_f1,
            })
            history_handle.flush()
            if epoch == 1 or epoch % 10 == 0 or epoch == 300:
                print(
                    f"[{model_name}] epoch {epoch:03d}/300 "
                    f"val_macro_f1={macro_f1:.4f} best={best_macro_f1:.4f}@{best_epoch}"
                )
            if scheduler is not None:
                scheduler.step()
        history_handle.close()
        model.load_state_dict(best_state, strict=True)
        checkpoint = {
            "model_name": model_name,
            "model_state_dict": best_state,
            "config": {**vars(opt), "best_epoch": best_epoch, "best_val_macro_f1": best_macro_f1},
            "feature_config": {
                "dataset_year": "2025", "cohort": "Elder", "track": "Track1",
                "task": "binary", "audio_feature": "mfccs", "video_feature": "densenet",
                "use_personality": True, "split_window": "1s", "device": "cuda",
                "personality_id_source": "filename", "protocol": "legacy_single_split",
            },
            "fold": None, "seed": HISTORICAL_SEED,
        }
        torch.save(checkpoint, artifact_dir / "checkpoint.pth")
        _, _, probabilities = evaluate_torch(model, test_loader, device_object)
        device = "cuda"

    row = _legacy_voted_result(
        model_name, probabilities, test_paths["entries"], args, run_id, device,
    )
    print(
        f"[PASS] {model_name}: Accuracy={float(row['Accuracy']):.4f}, "
        f"Macro-F1={float(row['Macro_F1']):.4f}, "
        f"CM=[[{row['TN']},{row['FP']}],[{row['FN']},{row['TP']}]]"
    )
    return row


def main(argv=None):
    args = parse_args(argv)
    if not args.historical_checkpoint.is_file():
        raise FileNotFoundError(f"历史 checkpoint 不存在：{args.historical_checkpoint}")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，拒绝启动正式旧版对比实验")
    args.output_dir = args.output_dir or _default_output_dir()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    runtime = _runtime_args(args, args.output_dir)
    plan = build_execution_plan(args.models, args.historical_checkpoint)
    (args.output_dir / "run_config.json").write_text(
        json.dumps({
            "protocol": "historical_2025_single_split_comparison",
            "cross_validation": False,
            "folds": 0,
            "seed": HISTORICAL_SEED,
            "train_samples": 292,
            "validation_samples": 45,
            "epochs": 300,
            "feature_max_len": 26,
            "batch_size": 8,
            "device": "cuda",
            "models": args.models,
            "ourmodel_policy": "frozen_historical_checkpoint",
            "subject_aggregation": "majority_vote_then_event_backfill",
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    paths = _legacy_paths(runtime)
    result_path = args.output_dir / "comparison_results.csv"
    completed_rows = []

    def record(row):
        completed_rows.append(row)
        _write_rows(result_path, completed_rows)
        return row

    execute_plan(
        plan,
        verify_our=lambda item: record(_verify_historical_our(item, runtime)),
        train_baseline=lambda item: record(_train_baseline(item, runtime, paths)),
    )
    print(f"[PASS] 旧版单次划分对比实验完成：{result_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
