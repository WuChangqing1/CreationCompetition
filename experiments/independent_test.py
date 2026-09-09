"""Strict provenance helpers for independent-test evaluation."""

import csv
from pathlib import Path

import numpy as np

from experiments.evaluator import evaluate_predictions


RESULT_COLUMNS = [
    "DatasetYear", "Cohort", "Model", "EvaluationLevel", "Samples",
    "Accuracy", "Macro_F1", "Weighted_F1", "Precision", "Recall",
    "Positive_Recall", "Specificity", "ROC_AUC", "TN", "FP", "FN", "TP",
    "EnsembleFolds", "Run_ID", "Device", "Seed", "Status",
]
APPROVED_MODELS = ("svm", "xgboost", "mlp", "bilstm", "lightweighttrans", "lmf", "mult", "our")
EXPERIMENTAL_MODELS = frozenset({"ourablation"})
CV_SELECTION_FIELDS = (
    "DatasetYear", "Cohort", "Model", "Track", "Task", "AudioFeature",
    "VideoFeature", "UsePersonality", "SplitWindow", "Seed",
)
RESULT_KEY = ("DatasetYear", "Cohort", "Model", "EvaluationLevel")


_BOOLEAN_FIELDS = {"UsePersonality", "use_personality"}
_INTEGER_FIELDS = {"Fold", "fold", "Seed", "seed"}
_METADATA_FIELDS = (
    ("model_name", "model_name", ("model_name", "Model")),
    ("fold", "fold", ("fold", "Fold")),
    ("seed", "seed", ("seed", "Seed")),
    ("dataset_year", "feature_config", ("dataset_year", "DatasetYear")),
    ("cohort", "feature_config", ("cohort", "Cohort")),
    ("track", "feature_config", ("track", "Track")),
    ("task", "feature_config", ("task", "Task")),
    ("audio_feature", "feature_config", ("audio_feature", "AudioFeature")),
    ("video_feature", "feature_config", ("video_feature", "VideoFeature")),
    ("use_personality", "feature_config", ("use_personality", "UsePersonality")),
    ("split_window", "feature_config", ("split_window", "SplitWindow")),
    ("device", "feature_config", ("device", "Device", "device_family")),
)
_MISSING = object()


def _normalize_boolean(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    return _MISSING


def _normalize_integer(value):
    if isinstance(value, bool):
        return _MISSING
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.isdecimal() or (
            normalized.startswith(("+", "-")) and normalized[1:].isdecimal()
        ):
            return int(normalized)
    return _MISSING


def _normalize_scalar(field, value):
    if field in _BOOLEAN_FIELDS:
        return _normalize_boolean(value)
    if field in _INTEGER_FIELDS:
        return _normalize_integer(value)
    if isinstance(value, str):
        return value.strip()
    return value


def _row_matches(row, condition):
    for column, expected in condition.items():
        actual_normalized = _normalize_scalar(column, row.get(column, _MISSING))
        expected_normalized = _normalize_scalar(column, expected)
        if (
            actual_normalized is _MISSING
            or expected_normalized is _MISSING
            or actual_normalized != expected_normalized
        ):
            return False
    return True


def _require_approved_model(model):
    if model not in APPROVED_MODELS and model not in EXPERIMENTAL_MODELS:
        raise ValueError(f"Unsupported independent-test model: {model!r}")


def _require_five_folds(folds):
    if folds != 5:
        raise ValueError(f"Independent-test evaluator requires exactly five folds; got {folds!r}")


def select_cv_run(rows, condition, folds=5):
    """Select one fully matching PASS CV run with each required fold exactly once."""
    _require_five_folds(folds)
    missing_condition_fields = [field for field in CV_SELECTION_FIELDS if field not in condition]
    if missing_condition_fields:
        raise ValueError(f"Missing required CV selection condition fields: {missing_condition_fields}")
    _require_approved_model(_normalize_scalar("Model", condition.get("Model", _MISSING)))
    matching = [
        row for row in rows
        if _row_matches(row, condition) and row.get("Status") == "PASS"
    ]
    run_ids = sorted({row["Run_ID"] for row in matching})
    if len(run_ids) != 1:
        raise ValueError(f"Expected one CV Run_ID for {condition}; candidates={run_ids}")

    selected = [row for row in matching if row["Run_ID"] == run_ids[0]]
    actual_folds = [int(row["Fold"]) for row in selected]
    expected_folds = list(range(1, folds + 1))
    if sorted(actual_folds) != expected_folds or len(actual_folds) != len(set(actual_folds)):
        raise ValueError(f"Incomplete or duplicate folds for {run_ids[0]}: {actual_folds}")
    return run_ids[0], sorted(selected, key=lambda row: int(row["Fold"]))


def resolve_checkpoint_paths(results_dir, run_id, model, folds=5):
    """Resolve every expected fold checkpoint without accepting partial ensembles."""
    _require_five_folds(folds)
    _require_approved_model(_normalize_scalar("Model", model))
    results_dir = Path(results_dir)
    paths = [
        results_dir / "raw" / run_id / model / f"fold_{fold}" / "checkpoint.pth"
        for fold in range(1, folds + 1)
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing checkpoint paths for Run_ID {run_id}: {missing_text}")
    return paths


def _expected_value(expected, aliases):
    for alias in aliases:
        if alias in expected:
            return expected[alias]
    return _MISSING


def _device_family(value):
    if not isinstance(value, str):
        return value
    return value.strip().casefold().split(":", 1)[0]


def validate_checkpoint_metadata(payload, expected):
    """Raise one error containing every payload/expected provenance mismatch."""
    payload = payload if isinstance(payload, dict) else {}
    feature_config = payload.get("feature_config")
    feature_config = feature_config if isinstance(feature_config, dict) else {}
    mismatches = []

    for field, source, expected_aliases in _METADATA_FIELDS:
        wanted = _expected_value(expected, expected_aliases)
        if wanted is _MISSING:
            mismatches.append(f"{field}: expected value is missing")
            continue
        actual = payload.get(field, _MISSING) if source == field else feature_config.get(field, _MISSING)
        if field == "device":
            actual_normalized = _device_family(actual)
            wanted_normalized = _device_family(wanted)
        else:
            actual_normalized = _normalize_scalar(field, actual)
            wanted_normalized = _normalize_scalar(field, wanted)
        if (
            actual_normalized is _MISSING
            or wanted_normalized is _MISSING
            or actual_normalized != wanted_normalized
        ):
            actual_display = "<missing>" if actual is _MISSING else repr(actual)
            mismatches.append(f"{field}: expected {wanted!r}, got {actual_display}")

    wanted_personality_source = _expected_value(
        expected, ("personality_id_source", "PersonalityIDSource")
    )
    if wanted_personality_source is not _MISSING:
        actual_personality_source = feature_config.get("personality_id_source", "filename")
        if str(actual_personality_source).strip().lower() != str(wanted_personality_source).strip().lower():
            mismatches.append(
                "personality_id_source: expected "
                f"{wanted_personality_source!r}, got {actual_personality_source!r}"
            )

    if mismatches:
        raise ValueError("Checkpoint metadata mismatch: " + "; ".join(mismatches))


def _validate_probabilities(probabilities, expected_rows=None):
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("Binary probabilities must have shape [N, 2]")
    if expected_rows is not None and len(values) != expected_rows:
        raise ValueError(
            f"Probability rows must match the number of labels; got {len(values)} and {expected_rows}"
        )
    if not np.isfinite(values).all():
        raise ValueError("Binary probabilities must be finite")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("Each binary probability row must sum to one")
    return values


def _validate_binary_labels(labels, expected_rows=None):
    try:
        values = np.asarray(labels, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Binary labels must contain only 0 and 1") from error
    if values.ndim != 1:
        raise ValueError("Binary labels must be one-dimensional")
    if expected_rows is not None and len(values) != expected_rows:
        raise ValueError(f"Labels must have {expected_rows} rows; got {len(values)}")
    if (
        not np.isfinite(values).all()
        or not np.equal(values, np.floor(values)).all()
        or not np.isin(values, (0, 1)).all()
    ):
        raise ValueError("Binary labels must contain only 0 and 1")
    return values.astype(int)


def mean_fold_probabilities(fold_probabilities):
    """Return the equal-weight mean of compatible binary fold probabilities."""
    folds = list(fold_probabilities)
    if not folds:
        raise ValueError("At least one fold probability array is required")
    validated = [_validate_probabilities(probabilities) for probabilities in folds]
    expected_shape = validated[0].shape
    if any(probabilities.shape != expected_shape for probabilities in validated[1:]):
        raise ValueError("All fold probability arrays must have the same shape")
    return np.mean(np.stack(validated, axis=0), axis=0)


def aggregate_subject_probabilities(subject_ids, labels, probabilities):
    """Average event probabilities per subject while enforcing one binary label."""
    subjects = [str(subject_id) for subject_id in subject_ids]
    values = _validate_probabilities(probabilities, expected_rows=len(subjects))
    binary_labels = _validate_binary_labels(labels, expected_rows=len(subjects))
    totals = {}
    counts = {}
    subject_labels = {}

    for subject_id, label, probability in zip(subjects, binary_labels, values):
        if subject_id in subject_labels and subject_labels[subject_id] != label:
            raise ValueError(f"Conflicting labels for subject {subject_id}")
        if subject_id not in totals:
            totals[subject_id] = probability.copy()
            counts[subject_id] = 1
            subject_labels[subject_id] = label
        else:
            totals[subject_id] += probability
            counts[subject_id] += 1

    unique_subject_ids = list(totals)
    aggregated_labels = np.asarray([subject_labels[subject_id] for subject_id in unique_subject_ids], dtype=int)
    aggregated_probabilities = np.asarray(
        [totals[subject_id] / counts[subject_id] for subject_id in unique_subject_ids], dtype=float
    )
    return unique_subject_ids, aggregated_labels, aggregated_probabilities


def aggregate_subject_predictions(subject_ids, labels, probabilities, method="probability_mean"):
    """Aggregate events per subject using probability mean or historical majority voting."""
    unique_ids, aggregated_labels, means = aggregate_subject_probabilities(
        subject_ids, labels, probabilities
    )
    if method == "probability_mean":
        return unique_ids, aggregated_labels, means
    if method != "majority_vote":
        raise ValueError(f"Unknown subject aggregation method: {method}")

    values = _validate_probabilities(probabilities, expected_rows=len(subject_ids))
    event_predictions = values.argmax(axis=1)
    vote_probabilities = []
    for subject_id, mean_probability in zip(unique_ids, means):
        indices = [index for index, value in enumerate(subject_ids) if str(value) == subject_id]
        counts = np.bincount(event_predictions[indices], minlength=2)
        if counts[0] == counts[1]:
            winner = int(mean_probability.argmax())
        else:
            winner = int(counts.argmax())
        vote_probabilities.append([1.0, 0.0] if winner == 0 else [0.0, 1.0])
    return unique_ids, aggregated_labels, np.asarray(vote_probabilities, dtype=float)


def broadcast_subject_predictions(event_subject_ids, subject_ids, subject_probabilities):
    """Broadcast one subject decision to its events for old test.py-compatible metrics."""
    lookup = {
        str(subject_id): probability
        for subject_id, probability in zip(subject_ids, _validate_probabilities(subject_probabilities))
    }
    missing = sorted({str(value) for value in event_subject_ids} - set(lookup))
    if missing:
        raise ValueError(f"Missing subject predictions for: {missing}")
    return np.asarray([lookup[str(subject_id)] for subject_id in event_subject_ids], dtype=float)


def build_result_row(*, year, cohort, model, level, labels, probabilities, folds, run_id, device, seed,
                     status="PASS"):
    """Build one schema-validated independent-test result row from predictions."""
    binary_labels = _validate_binary_labels(labels)
    values = _validate_probabilities(probabilities, expected_rows=len(binary_labels))
    if len(binary_labels) == 0:
        raise ValueError("Cannot build a result row from empty predictions")
    predictions = np.argmax(values, axis=1)
    metrics = evaluate_predictions(binary_labels, predictions, values)
    tn, fp, fn, tp = np.asarray(metrics["Confusion_Matrix"], dtype=int).ravel()
    result = {
        "DatasetYear": year,
        "Cohort": cohort,
        "Model": model,
        "EvaluationLevel": level,
        "Samples": len(binary_labels),
        "Accuracy": metrics["Accuracy"],
        "Macro_F1": metrics["Macro_F1"],
        "Weighted_F1": metrics["Weighted_F1"],
        "Precision": metrics["Precision"],
        "Recall": metrics["Recall"],
        "Positive_Recall": metrics["Positive_Recall"],
        "Specificity": metrics["Specificity"],
        "ROC_AUC": metrics["ROC_AUC"],
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "EnsembleFolds": folds,
        "Run_ID": run_id,
        "Device": device,
        "Seed": seed,
        "Status": status,
    }
    return {column: result[column] for column in RESULT_COLUMNS}


def _validate_result_row(row):
    if not isinstance(row, dict) or set(row) != set(RESULT_COLUMNS):
        raise ValueError("Result rows must contain exactly the approved RESULT_COLUMNS")
    return {column: row[column] for column in RESULT_COLUMNS}


def _read_result_rows(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != RESULT_COLUMNS:
            raise ValueError("Existing result CSV header does not match RESULT_COLUMNS")
        return [{column: row[column] for column in RESULT_COLUMNS} for row in reader]


def _result_key(row):
    return tuple(str(row[column]) for column in RESULT_KEY)


def upsert_result_rows(path, new_rows):
    """Atomically replace result rows with matching evaluation identity keys."""
    path = Path(path)
    rows_by_key = {_result_key(row): row for row in _read_result_rows(path)}
    for row in new_rows:
        validated = _validate_result_row(row)
        rows_by_key[_result_key(validated)] = validated
    rows = sorted(
        rows_by_key.values(),
        key=lambda row: tuple(str(row[column]) for column in RESULT_KEY),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return path


def write_prediction_csv(path, subject_ids, labels, probabilities, model, level):
    """Write validated event- or subject-level predictions with provenance columns."""
    path = Path(path)
    subjects = [str(subject_id) for subject_id in subject_ids]
    values = _validate_probabilities(probabilities, expected_rows=len(subjects))
    binary_labels = _validate_binary_labels(labels, expected_rows=len(subjects))
    predictions = np.argmax(values, axis=1)
    rows = [
        {
            "subject_id": subject_id,
            "true_label": int(label),
            "pred_label": int(prediction),
            "prob_0": probability[0],
            "prob_1": probability[1],
            "model": str(model),
            "evaluation_level": str(level),
        }
        for subject_id, label, prediction, probability in zip(subjects, binary_labels, predictions, values)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "subject_id", "true_label", "pred_label", "prob_0", "prob_1", "model", "evaluation_level",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return path
