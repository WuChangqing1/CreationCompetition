"""Model inference helpers for independent-test evaluation."""

from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments.independent_test import (
    mean_fold_probabilities,
    validate_checkpoint_metadata,
)
from experiments.model_registry import create_experiment_model
from experiments.run_model_cv import evaluate_torch, make_dataset


_REQUIRED_FOLDS = 5


def _entry_labels(test_entries, classes):
    label_keys = {2: "bin_category", 3: "tri_category", 5: "pen_category"}
    try:
        label_key = label_keys[int(classes)]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Unsupported label count: {classes!r}") from error
    return np.asarray([int(entry[label_key]) for entry in test_entries], dtype=int)


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
        fold_probabilities.append(mean_fold_probabilities([probabilities]))

    return mean_fold_probabilities(fold_probabilities)


__all__ = ["infer_torch_fold_ensemble"]
