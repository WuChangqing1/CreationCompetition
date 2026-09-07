"""Optional XGBoost baseline with delayed dependency import."""

from experiments.errors import OptionalDependencyError


def create_xgboost(config=None):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise OptionalDependencyError(
            "XGBoost is unavailable. Activate dachuangxiangmu and run: "
            "python -m pip install xgboost"
        ) from exc

    config = dict(config or {})
    return XGBClassifier(
        n_estimators=int(config.get("n_estimators", 200)),
        max_depth=int(config.get("max_depth", 4)),
        learning_rate=float(config.get("learning_rate", 0.05)),
        subsample=float(config.get("subsample", 0.9)),
        colsample_bytree=float(config.get("colsample_bytree", 0.9)),
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method=str(config.get("tree_method", "hist")),
        device=str(config.get("device", "cuda")),
        random_state=int(config.get("seed", 3407)),
        n_jobs=int(config.get("n_jobs", -1)),
    )
