"""Evaluate completed five-fold CV runs on an independent MPDD test set."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from experiments.dataset_layouts import resolve_dataset, resolve_independent_test
from experiments.independent_test import (
    RESULT_COLUMNS,
    aggregate_subject_probabilities,
    build_result_row,
    resolve_checkpoint_paths,
    select_cv_run,
    upsert_result_rows,
    write_prediction_csv,
)
from experiments.independent_test_models import (
    classical_execution_device,
    infer_classical_fold_ensemble,
    infer_torch_fold_ensemble,
    load_saved_fold_records,
)


APPROVED_MODELS = ("svm", "xgboost", "mlp", "bilstm", "lightweighttrans", "lmf", "mult", "our")
CLASSICAL_MODELS = {"svm", "xgboost"}
DEFAULT_CV_RESULTS = ROOT / "experiments" / "results" / "raw_results.csv"
DEFAULT_OUTPUT = ROOT / "experiments" / "results" / "independent_test_results.csv"


def _model_list(value):
    models = [item.strip().lower() for item in value.split(",") if item.strip()]
    if not models:
        raise argparse.ArgumentTypeError("--models must contain at least one model")
    unknown = [model for model in models if model not in APPROVED_MODELS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown independent-test models: {unknown}")
    if len(models) != len(set(models)):
        raise argparse.ArgumentTypeError("--models must not contain duplicates")
    return models


def _resolved(path):
    return Path(path).expanduser().resolve()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=_model_list, default=list(APPROVED_MODELS))
    parser.add_argument("--dataset-year", choices=("2025", "2026"), required=True)
    parser.add_argument("--cohort", choices=("Elder", "Young"), default="Elder")
    parser.add_argument("--track", default="Track1")
    parser.add_argument("--task", choices=("binary",), default="binary")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cv-results", type=Path, default=DEFAULT_CV_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audio-feature", default="mfccs")
    parser.add_argument("--video-feature", default="densenet")
    personality = parser.add_mutually_exclusive_group()
    personality.add_argument("--use-personality", dest="use_personality", action="store_true")
    personality.add_argument("--no-use-personality", dest="use_personality", action="store_false")
    parser.set_defaults(use_personality=True)
    parser.add_argument(
        "--personality-id-source", choices=("filename", "subject_id"), default="filename",
        help="must match the mode used by the selected CV run",
    )
    parser.add_argument("--split-window", default="1s")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--feature-max-len", type=int, default=5,
        help="fallback sequence length for legacy classical CV configs (default: 5)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--splits-dir", type=Path)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments" / "results")
    args = parser.parse_args(argv)

    if args.data_root is None:
        configured_root = os.environ.get("MPDD_DATA_ROOT")
        if not configured_root:
            parser.error("provide --data-root or set MPDD_DATA_ROOT")
        args.data_root = Path(configured_root)
    if args.folds != 5:
        parser.error("independent-test evaluation requires exactly five folds")
    if any(model != "svm" for model in args.models) and not str(args.device).lower().startswith("cuda"):
        parser.error("XGBoost and PyTorch independent-test models require --device cuda")
    if args.feature_max_len <= 0 or args.batch_size <= 0:
        parser.error("--feature-max-len and --batch-size must be positive")

    args.data_root = _resolved(args.data_root)
    args.cv_results = _resolved(args.cv_results)
    args.output = _resolved(args.output)
    args.results_dir = _resolved(args.results_dir)
    args.splits_dir = _resolved(
        args.splits_dir or ROOT / "experiments" / "splits" / args.dataset_year / args.cohort
    )
    protected = {
        args.cv_results,
        _resolved(ROOT / "experiments" / "results" / "raw_results.csv"),
        _resolved(ROOT / "experiments" / "results" / "summary.csv"),
        _resolved(ROOT / "answer_Track1" / "submission.csv"),
    }
    if args.output in protected or args.output.name.lower() in {"raw_results.csv", "summary.csv", "submission.csv"}:
        parser.error("--output must not overwrite raw_results.csv, summary.csv, or legacy submission.csv")
    args.classes = 2
    args.feature_max_len_fallback = args.feature_max_len
    return args


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _artifact_dirs(output):
    parent = Path(output).parent
    return parent / "independent_test_predictions", parent / "independent_test_confusion_matrix"


def _fold_artifact_dirs(args, run_id, model_name):
    return [args.results_dir / "raw" / run_id / model_name / f"fold_{fold}" for fold in range(1, 6)]


def _load_matching_saved_config(args, run_id, model_name):
    configs = []
    for directory in _fold_artifact_dirs(args, run_id, model_name):
        path = directory / "config.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved fold config: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Saved fold config must be an object: {path}")
        configs.append(payload)
    if any(config != configs[0] for config in configs[1:]):
        raise ValueError(f"Saved fold configs disagree for {run_id}/{model_name}")
    return configs[0]


def _configure_torch_feature_length(args, checkpoint_paths):
    import torch

    lengths = []
    for path in checkpoint_paths:
        payload = torch.load(path, map_location="cpu")
        config = payload.get("config")
        if not isinstance(config, dict) or "feature_max_len" not in config:
            raise ValueError(f"Checkpoint has no saved config.feature_max_len: {path}")
        lengths.append(int(config["feature_max_len"]))
    if len(set(lengths)) != 1:
        raise ValueError(f"Saved checkpoint feature_max_len values disagree: {lengths}")
    args.feature_max_len = lengths[0]


def _condition(args, model_name):
    return {
        "DatasetYear": args.dataset_year, "Cohort": args.cohort, "Model": model_name,
        "Track": args.track, "Task": args.task, "AudioFeature": args.audio_feature,
        "VideoFeature": args.video_feature, "UsePersonality": args.use_personality,
        "SplitWindow": args.split_window, "Seed": args.seed,
    }


def evaluate_one_model(args, model_name, cv_rows, training_paths, test_paths, fold_records):
    run_id, _ = select_cv_run(cv_rows, _condition(args, model_name), folds=args.folds)
    expected_metadata = {
        "dataset_year": args.dataset_year, "cohort": args.cohort, "track": args.track,
        "task": args.task, "audio_feature": args.audio_feature, "video_feature": args.video_feature,
        "use_personality": args.use_personality, "split_window": args.split_window,
        "personality_id_source": args.personality_id_source,
        "seed": args.seed,
    }
    if model_name in CLASSICAL_MODELS:
        config = _load_matching_saved_config(args, run_id, model_name)
        saved_personality_source = str(config.get("personality_id_source", "filename")).lower()
        if saved_personality_source != args.personality_id_source:
            raise ValueError(
                "Saved CV personality ID source mismatch: "
                f"expected {args.personality_id_source}, got {saved_personality_source}"
            )
        args.feature_max_len = int(config.get("feature_max_len", args.feature_max_len_fallback))
        _, probabilities = infer_classical_fold_ensemble(
            model_name=model_name, training_entries=training_paths["entries"],
            training_paths=training_paths, test_entries=test_paths["entries"], test_paths=test_paths,
            fold_records=fold_records, config=config, args=args,
        )
        device = classical_execution_device(model_name, args)
    else:
        checkpoints = resolve_checkpoint_paths(args.results_dir, run_id, model_name, args.folds)
        _configure_torch_feature_length(args, checkpoints)
        probabilities = infer_torch_fold_ensemble(
            model_name=model_name, test_entries=test_paths["entries"], test_paths=test_paths,
            checkpoint_paths=checkpoints, args=args, expected_metadata=expected_metadata,
        )
        device = "cuda"

    # Test labels are consumed for metrics only after all five fold probabilities exist.
    entries = test_paths["entries"]
    labels = np.asarray([int(entry["bin_category"]) for entry in entries], dtype=int)
    subject_ids = [str(entry["subject_id"]) for entry in entries]
    probabilities = np.asarray(probabilities, dtype=float)
    event_row = build_result_row(
        year=args.dataset_year, cohort=args.cohort, model=model_name, level="event",
        labels=labels, probabilities=probabilities, folds=args.folds, run_id=run_id,
        device=device, seed=args.seed,
    )
    unique_subjects, subject_labels, subject_probabilities = aggregate_subject_probabilities(
        subject_ids, labels, probabilities
    )
    subject_row = build_result_row(
        year=args.dataset_year, cohort=args.cohort, model=model_name, level="subject",
        labels=subject_labels, probabilities=subject_probabilities, folds=args.folds, run_id=run_id,
        device=device, seed=args.seed,
    )
    return {
        "rows": [event_row, subject_row], "run_id": run_id,
        "event": (subject_ids, labels, probabilities),
        "subject": (unique_subjects, subject_labels, subject_probabilities),
    }


def _write_confusion_csv(path, row):
    values = ((0, 0, row["TN"]), (0, 1, row["FP"]), (1, 0, row["FN"]), (1, 1, row["TP"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("true_label", "pred_label", "count"))
        writer.writerows(values)
    temporary.replace(path)


def _failure_rows(args, model_name, run_id, reason):
    status = f"FAILED: {reason}".replace("\n", " ")[:500]
    rows = []
    for level in ("event", "subject"):
        row = {column: "N/A" for column in RESULT_COLUMNS}
        row.update({
            "DatasetYear": args.dataset_year, "Cohort": args.cohort, "Model": model_name,
            "EvaluationLevel": level, "EnsembleFolds": args.folds, "Run_ID": run_id,
            "Device": "cpu:svm" if model_name == "svm" else (
                "cuda:xgboost" if model_name == "xgboost" else "cuda"
            ),
            "Seed": args.seed, "Status": status,
        })
        rows.append(row)
    return rows


def main(argv=None):
    args = parse_args(argv)
    try:
        cv_rows = _read_csv(args.cv_results)
        training_paths = resolve_dataset(
            args.data_root, args.dataset_year, args.cohort, args.split_window,
            args.audio_feature, args.video_feature,
        )
        test_paths = resolve_independent_test(
            args.data_root, args.dataset_year, args.cohort, args.split_window,
            args.audio_feature, args.video_feature,
        )
        fold_records = load_saved_fold_records(args)
    except Exception as error:
        print(f"[FAIL] setup: {error}")
        return 1

    prediction_dir, confusion_dir = _artifact_dirs(args.output)
    failed = False
    for model_name in args.models:
        args.feature_max_len = args.feature_max_len_fallback
        run_id = "N/A"
        try:
            result = evaluate_one_model(
                args, model_name, cv_rows, training_paths, test_paths, fold_records
            )
            run_id = result["run_id"]
            for row, level in zip(result["rows"], ("event", "subject")):
                ids, labels, probabilities = result[level]
                stem = f"{args.dataset_year}_{args.cohort}_{model_name}_{level}"
                write_prediction_csv(
                    prediction_dir / f"{stem}.csv", ids, labels, probabilities, model_name, level
                )
                _write_confusion_csv(confusion_dir / f"{stem}.csv", row)
            upsert_result_rows(args.output, result["rows"])
            print(f"[PASS] {model_name}")
        except Exception as error:
            failed = True
            try:
                run_id, _ = select_cv_run(cv_rows, _condition(args, model_name), folds=args.folds)
            except Exception:
                pass
            upsert_result_rows(args.output, _failure_rows(args, model_name, run_id, str(error)))
            print(f"[FAIL] {model_name}: {error}")
    print(str(args.output))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
