"""Strict provenance helpers for independent-test evaluation."""

from pathlib import Path


RESULT_COLUMNS = [
    "DatasetYear", "Cohort", "Model", "EvaluationLevel", "Samples",
    "Accuracy", "Macro_F1", "Weighted_F1", "Precision", "Recall",
    "Positive_Recall", "Specificity", "ROC_AUC", "TN", "FP", "FN", "TP",
    "EnsembleFolds", "Run_ID", "Device", "Seed", "Status",
]
APPROVED_MODELS = ("svm", "xgboost", "mlp", "bilstm", "lightweighttrans", "lmf", "mult", "our")
CV_SELECTION_FIELDS = (
    "DatasetYear", "Cohort", "Model", "Track", "Task", "AudioFeature",
    "VideoFeature", "UsePersonality", "SplitWindow", "Seed",
)


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
    if model not in APPROVED_MODELS:
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

    if mismatches:
        raise ValueError("Checkpoint metadata mismatch: " + "; ".join(mismatches))
