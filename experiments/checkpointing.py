"""New checkpoint schema helpers."""


def build_checkpoint_payload(model_name, state_dict, config, feature_config, fold, seed):
    return {
        "model_name": str(model_name),
        "model_state_dict": state_dict,
        "config": dict(config),
        "feature_config": dict(feature_config),
        "fold": int(fold),
        "seed": int(seed),
    }
