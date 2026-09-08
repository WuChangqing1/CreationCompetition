"""Named experiment protocols; the historical BiCFNet setup is the default."""

from copy import deepcopy


PROTOCOLS = {
    "legacy_bicfnet": {
        "seed": 2024,
        "feature_max_len": 26,
        "batch_size": 8,
        "epochs": 300,
        "results_directory": "results_legacy_bicfnet",
        "subject_aggregation": "majority_vote",
        "model_overrides": {
            "our": {
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "focal_weight": 0.1,
                "loss_function": "CrossEntropyLoss+FocalLoss",
                "scheduler": "cosine",
                "scheduler_eta_min": 1e-6,
                "checkpoint_selection": "val_macro_f1",
            }
        },
    },
    "modern": {
        "seed": 3407,
        "feature_max_len": 5,
        "results_directory": "results",
        "subject_aggregation": "probability_mean",
        "model_overrides": {},
    },
}


def resolve_protocol(name, model_name, config):
    """Return an isolated protocol and model config with its declared overrides."""
    if name not in PROTOCOLS:
        raise ValueError(f"Unknown experiment protocol: {name}")
    protocol = deepcopy(PROTOCOLS[name])
    resolved = deepcopy(config)
    resolved.update(protocol["model_overrides"].get(model_name.lower(), {}))
    return protocol, resolved
