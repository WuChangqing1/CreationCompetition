"""Consistent metrics and prediction artifacts for MPDD experiments."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(y_true, y_pred, y_prob=None):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.ndim != 1 or y_pred.ndim != 1 or len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be one-dimensional arrays of equal length")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate empty predictions")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    positive_recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    auc = "N/A"
    if y_prob is not None and len(np.unique(y_true)) == 2:
        probabilities = np.asarray(y_prob, dtype=float)
        positive_prob = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
        auc = float(roc_auc_score(y_true, positive_prob))

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "Weighted_F1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "Positive_Recall": float(positive_recall),
        "Specificity": float(specificity),
        "ROC_AUC": auc,
        "Confusion_Matrix": matrix.tolist(),
    }


def save_predictions(path, subject_ids, y_true, y_pred, y_prob, fold, model):
    path = Path(path)
    probabilities = np.asarray(y_prob, dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Binary probabilities must have shape [samples, 2]")
    frame = pd.DataFrame({
        "subject_id": list(map(str, subject_ids)),
        "true_label": np.asarray(y_true, dtype=int),
        "pred_label": np.asarray(y_pred, dtype=int),
        "prob_0": probabilities[:, 0],
        "prob_1": probabilities[:, 1],
        "fold": int(fold),
        "model": str(model),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def save_confusion_matrix(matrix, path, title):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(matrix, dtype=int)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        csv_path = path.with_suffix(".csv")
        pd.DataFrame(matrix, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]).to_csv(csv_path)
        return {"status": "SKIPPED", "path": str(csv_path), "reason": "matplotlib unavailable"}

    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    axis.set_title(title)
    axis.set_xlabel("Predicted Label")
    axis.set_ylabel("True Label")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return {"status": "PASS", "path": str(path), "reason": ""}


def save_overall_confusion(prediction_paths, path, model):
    frames = [pd.read_csv(prediction_path) for prediction_path in prediction_paths]
    if not frames:
        raise ValueError("At least one prediction file is required")
    combined = pd.concat(frames, ignore_index=True)
    probabilities = combined[["prob_0", "prob_1"]].to_numpy()
    metrics = evaluate_predictions(
        combined["true_label"].to_numpy(),
        combined["pred_label"].to_numpy(),
        probabilities,
    )
    artifact = save_confusion_matrix(metrics["Confusion_Matrix"], path, f"{model} Overall OOF")
    return {"metrics": metrics, "artifact": artifact}
