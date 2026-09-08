"""Core helpers for the historical 2025 single-split comparison protocol."""

import json
from pathlib import Path


APPROVED_MODELS = (
    "svm", "xgboost", "mlp", "bilstm", "lightweighttrans", "lmf", "mult", "our",
)
HISTORICAL_SEED = 2024
HISTORICAL_TRAIN_SAMPLES = 292
HISTORICAL_VALIDATION_SAMPLES = 45
# Recovered from the old Track1 splitter under the process state that matches
# the historical 45-sample validation label counts (28 negative, 17 positive).
# Freezing entry identities removes the original set-iteration nondeterminism.
HISTORICAL_VALIDATION_AUDIO_PATHS = (
    "69_A_3_audio_features.npy", "69_A_1_audio_features.npy",
    "69_A_2_audio_features.npy", "69_A_4_audio_features.npy",
    "64_A_3_audio_features.npy",
    "35_A_1_audio_features.npy", "35_A_2_audio_features.npy",
    "35_A_3_audio_features.npy", "35_A_4_audio_features.npy",
    "94_A_1_audio_features.npy", "94_A_2_audio_features.npy",
    "94_A_3_audio_features.npy", "94_A_4_audio_features.npy",
    "16_A_1_audio_features.npy", "16_A_2_audio_features.npy",
    "16_A_3_audio_features.npy", "16_A_4_audio_features.npy",
    "68_A_1_audio_features.npy", "68_A_2_audio_features.npy",
    "68_A_3_audio_features.npy", "68_A_4_audio_features.npy",
    "106_A_1_audio_features.npy", "106_A_2_audio_features.npy",
    "106_A_3_audio_features.npy", "106_A_4_audio_features.npy",
    "81_A_1_audio_features.npy", "81_A_2_audio_features.npy",
    "81_A_3_audio_features.npy", "81_A_4_audio_features.npy",
    "96_A_1_audio_features.npy", "96_A_2_audio_features.npy",
    "96_A_3_audio_features.npy", "96_A_4_audio_features.npy",
    "18_A_1_audio_features.npy", "18_A_2_audio_features.npy",
    "18_A_3_audio_features.npy", "18_A_4_audio_features.npy",
    "4_A_1_audio_features.npy", "4_A_2_audio_features.npy",
    "4_A_3_audio_features.npy", "4_A_4_audio_features.npy",
    "82_A_1_audio_features.npy", "82_A_2_audio_features.npy",
    "82_A_3_audio_features.npy", "82_A_4_audio_features.npy",
)


def normalize_models(models):
    normalized = [str(model).strip().lower() for model in models]
    if not normalized:
        raise ValueError("models must not be empty")
    if len(normalized) != len(set(normalized)):
        raise ValueError("duplicate models are not allowed")
    unknown = [model for model in normalized if model not in APPROVED_MODELS]
    if unknown:
        raise ValueError(f"unknown models: {unknown}")
    return normalized


def build_execution_plan(models, historical_checkpoint):
    """Describe a comparison with one old split and no cross-validation."""
    plan = []
    for model in normalize_models(models):
        if model == "our":
            plan.append({
                "model": model,
                "action": "evaluate_frozen_checkpoint",
                "checkpoint": Path(historical_checkpoint),
            })
        else:
            plan.append({"model": model, "action": "train_single_split"})
    return plan


def split_entries_by_frozen_membership(entries):
    wanted = set(HISTORICAL_VALIDATION_AUDIO_PATHS)
    available = {entry.get("audio_feature_path") for entry in entries}
    missing = sorted(wanted - available)
    if missing:
        raise RuntimeError(f"历史验证集条目缺失：{missing}")
    train_entries = [entry for entry in entries if entry.get("audio_feature_path") not in wanted]
    val_entries = [entry for entry in entries if entry.get("audio_feature_path") in wanted]
    return train_entries, val_entries


def load_historical_split(training_json, split_fn=None):
    """Load the frozen historical Track1 membership without hash-order drift."""
    if split_fn is None:
        entries = json.loads(Path(training_json).read_text(encoding="utf-8"))
        train_entries, val_entries = split_entries_by_frozen_membership(entries)
    else:
        train_entries, val_entries, _, _ = split_fn(
            Path(training_json), val_ratio=0.1, random_seed=HISTORICAL_SEED,
        )
    actual = (len(train_entries), len(val_entries))
    expected = (HISTORICAL_TRAIN_SAMPLES, HISTORICAL_VALIDATION_SAMPLES)
    if actual != expected:
        raise RuntimeError(
            "历史单次划分必须为 292/45，"
            f"当前得到 {actual[0]}/{actual[1]}；请检查数据集和划分函数"
        )
    return train_entries, val_entries


def execute_plan(plan, *, verify_our, train_baseline):
    """Verify the historical lock before allowing any baseline training."""
    frozen = [item for item in plan if item["action"] == "evaluate_frozen_checkpoint"]
    trained = [item for item in plan if item["action"] == "train_single_split"]
    results = []
    for item in frozen:
        results.append(verify_our(item))
    for item in trained:
        results.append(train_baseline(item))
    return results


__all__ = [
    "APPROVED_MODELS",
    "HISTORICAL_SEED",
    "HISTORICAL_VALIDATION_AUDIO_PATHS",
    "build_execution_plan",
    "execute_plan",
    "load_historical_split",
    "normalize_models",
    "split_entries_by_frozen_membership",
]
