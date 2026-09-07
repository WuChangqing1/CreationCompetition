"""Model inference helpers for independent-test evaluation."""

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.independent_test import (
    mean_fold_probabilities,
    validate_checkpoint_metadata,
)
from experiments.classical.common import pool_multimodal_features
from experiments.create_splits import LABEL_KEYS, validate_no_subject_leakage
from experiments.data_utils import build_subject_labels, filter_entries_by_subjects
from experiments.model_registry import create_experiment_model
from experiments.run_model_cv import collect_classical_arrays, evaluate_torch, make_dataset


_REQUIRED_FOLDS = 5
ROOT = Path(__file__).resolve().parents[1]


def _entry_labels(test_entries, classes):
    label_keys = {2: "bin_category", 3: "tri_category", 5: "pen_category"}
    try:
        label_key = label_keys[int(classes)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported label count: {classes!r}") from error
    return np.asarray([int(entry[label_key]) for entry in test_entries], dtype=int)


def _require_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Saved split {name} must be an integer")
    return value


def _classical_label_key(args):
    try:
        return LABEL_KEYS[{2: "binary", 3: "ternary", 5: "quinary"}[int(args.classes)]]
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported label count: {getattr(args, 'classes', None)!r}") from error


def load_saved_fold_records(args):
    """Load the five persisted subject-fold records selected for the CV run."""
    splits_dir = getattr(args, "splits_dir", None)
    if splits_dir is None:
        try:
            splits_dir = ROOT / "experiments" / "splits" / str(args.dataset_year) / str(args.cohort)
        except AttributeError as error:
            raise ValueError("dataset_year and cohort are required to load saved splits") from error
    splits_dir = Path(splits_dir)
    paths = [splits_dir / f"fold_{fold}.json" for fold in range(1, _REQUIRED_FOLDS + 1)]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing saved split records: " + ", ".join(map(str, missing)))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _validate_classical_fold_records(training_entries, fold_records, args):
    records = list(fold_records)
    if len(records) != _REQUIRED_FOLDS:
        raise ValueError(
            "Classical independent-test evaluation requires exactly five saved split records; "
            f"got {len(records)}"
        )
    label_key = _classical_label_key(args)
    subject_labels = build_subject_labels(training_entries, label_key)
    expected_subjects = set(subject_labels)
    records_by_fold = {}
    validation_subject_counts = Counter()

    for record_index, record in enumerate(records, 1):
        if not isinstance(record, dict):
            raise ValueError(f"Saved split record {record_index} must be an object")
        fold = _require_integer(record.get("fold"), "fold")
        folds = _require_integer(record.get("folds"), "fold count")
        seed = _require_integer(record.get("seed"), "seed")
        if fold in records_by_fold:
            raise ValueError(f"Duplicate saved split fold: {fold}")
        if folds != _REQUIRED_FOLDS:
            raise ValueError(f"Saved split fold count must be exactly five; got {folds}")
        if seed != int(args.seed):
            raise ValueError(f"Saved split seed mismatch: expected {args.seed}, got {seed}")
        if record.get("label_key") != label_key:
            raise ValueError(
                f"Saved split label key mismatch: expected {label_key!r}, got {record.get('label_key')!r}"
            )
        train_ids = record.get("train_ids")
        val_ids = record.get("val_ids")
        if not isinstance(train_ids, list) or not isinstance(val_ids, list):
            raise ValueError(f"Saved split fold {fold} requires list train_ids and val_ids")
        train_ids = [str(subject_id) for subject_id in train_ids]
        val_ids = [str(subject_id) for subject_id in val_ids]
        if len(train_ids) != len(set(train_ids)) or len(val_ids) != len(set(val_ids)):
            raise ValueError(f"Saved split fold {fold} contains duplicate subject IDs")
        validate_no_subject_leakage(train_ids, val_ids)
        if set(train_ids) | set(val_ids) != expected_subjects:
            raise ValueError(f"Saved split fold {fold} does not cover every current training subject")
        records_by_fold[fold] = {**record, "train_ids": train_ids, "val_ids": val_ids}
        validation_subject_counts.update(val_ids)

    expected_folds = set(range(1, _REQUIRED_FOLDS + 1))
    if set(records_by_fold) != expected_folds:
        raise ValueError(f"Saved split records must cover folds 1 through 5; got {sorted(records_by_fold)}")
    if set(validation_subject_counts) != expected_subjects or any(
        validation_subject_counts[subject_id] != 1 for subject_id in expected_subjects
    ):
        raise ValueError("Saved split validation IDs do not provide complete five-fold subject coverage")
    return [records_by_fold[fold] for fold in range(1, _REQUIRED_FOLDS + 1)]


def classical_execution_device(model_name, args):
    """Return the provenance device label used by the classical CV path."""
    normalized_name = str(model_name).strip().lower()
    if normalized_name == "svm":
        return "cpu:svm"
    if normalized_name == "xgboost":
        return f"{args.device}:xgboost"
    raise ValueError(f"Unsupported classical independent-test model: {model_name!r}")


def infer_classical_fold_ensemble(
    *, model_name, training_entries, training_paths, test_entries,
    test_paths, fold_records, config, args,
):
    """Refit one classical estimator per validated CV fold and average test probabilities."""
    normalized_name = str(model_name).strip().lower()
    if normalized_name not in {"svm", "xgboost"}:
        raise ValueError(f"Unsupported classical independent-test model: {model_name!r}")
    records = _validate_classical_fold_records(
        training_entries,
        load_saved_fold_records(args) if fold_records is None else fold_records,
        args,
    )

    test_dataset = make_dataset(test_entries, args, test_paths)
    test_audio, test_video, test_personality, test_labels = collect_classical_arrays(test_dataset)
    test_features = pool_multimodal_features(
        test_audio, test_video, test_personality, args.use_personality
    )
    expected_shape = (len(test_labels), 2)
    fold_probabilities = []
    for fold_record in records:
        train_entries = filter_entries_by_subjects(training_entries, fold_record["train_ids"])
        train_dataset = make_dataset(train_entries, args, training_paths)
        train_audio, train_video, train_personality, train_labels = collect_classical_arrays(train_dataset)
        train_features = pool_multimodal_features(
            train_audio, train_video, train_personality, args.use_personality
        )
        fold_config = {**config, "seed": args.seed}
        if normalized_name == "xgboost":
            fold_config.update(device=args.device, tree_method="hist")
        estimator = create_experiment_model(normalized_name, config=fold_config)
        estimator.fit(train_features, train_labels)
        probabilities = np.asarray(estimator.predict_proba(test_features))
        if probabilities.shape != expected_shape:
            raise ValueError(
                "Fold probabilities must have shape [N, 2]: "
                f"expected {expected_shape}, got {probabilities.shape}"
            )
        fold_probabilities.append(probabilities)
    return test_labels, mean_fold_probabilities(fold_probabilities)


def infer_torch_fold_ensemble(
    *, model_name, test_entries, test_paths, checkpoint_paths,
    args, expected_metadata,
):
    """Load five PyTorch CV checkpoints and equally average test probabilities."""
    paths = list(checkpoint_paths)
    if len(paths) != _REQUIRED_FOLDS:
        raise ValueError(
            "PyTorch independent-test evaluation requires exactly five checkpoint paths; "
            f"got {len(paths)}"
        )

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal independent-test evaluation")

    payloads = []
    for fold, checkpoint_path in enumerate(paths, 1):
        payload = torch.load(checkpoint_path, map_location="cpu")
        validate_checkpoint_metadata(
            payload,
            {
                **expected_metadata,
                "model_name": str(model_name).strip().lower(),
                "device": str(device),
                "fold": fold,
            },
        )
        if not isinstance(payload.get("config"), dict):
            raise ValueError(f"Checkpoint fold {fold} has no valid config")
        if "model_state_dict" not in payload:
            raise ValueError(f"Checkpoint fold {fold} has no model_state_dict")
        payloads.append(payload)

    dataset = make_dataset(test_entries, args, test_paths)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
    )
    expected_labels = _entry_labels(test_entries, args.classes)

    fold_probabilities = []
    for payload in payloads:
        opt = SimpleNamespace(**payload["config"])
        opt.isTrain = False
        opt.gpu_ids = []
        opt.device = str(device)
        model = create_experiment_model(model_name, opt=opt)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.to(device)
        labels, _, probabilities = evaluate_torch(model, loader, device)
        if not np.array_equal(np.asarray(labels), expected_labels):
            raise ValueError("Fold labels do not match the independent-test dataset order")
        probabilities = np.asarray(probabilities)
        expected_shape = (len(expected_labels), 2)
        if probabilities.shape != expected_shape:
            raise ValueError(
                "Fold probabilities must have shape [N, 2]: "
                f"expected {expected_shape}, got {probabilities.shape}"
            )
        fold_probabilities.append(mean_fold_probabilities([probabilities]))

    return mean_fold_probabilities(fold_probabilities)


__all__ = [
    "classical_execution_device",
    "infer_classical_fold_ensemble",
    "infer_torch_fold_ensemble",
    "load_saved_fold_records",
]
