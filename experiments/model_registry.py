"""Top-level experiment registry for classical and PyTorch models."""

from experiments.classical.svm_baseline import create_svm
from experiments.classical.xgboost_baseline import create_xgboost
from experiments.errors import OptionalDependencyError
from models import create_model


CLASSICAL_MODELS = frozenset({"svm", "xgboost"})
TORCH_MODELS = frozenset({
    "mlp", "bilstm", "lightweighttrans", "lmf", "mult",
    "our", "depmamba", "proposed",
})


def get_model_kind(model_name):
    name = str(model_name).strip().lower()
    if name in CLASSICAL_MODELS:
        return "classical"
    if name in TORCH_MODELS:
        return "torch"
    raise ValueError(f"Unknown model: {model_name}")


def create_experiment_model(model_name, opt=None, config=None):
    name = str(model_name).strip().lower()
    kind = get_model_kind(name)
    if kind == "torch":
        if opt is None:
            raise ValueError("opt is required for PyTorch models")
        opt.model = name
        return create_model(opt)
    if name == "svm":
        return create_svm(config)
    return create_xgboost(config)


__all__ = [
    "CLASSICAL_MODELS", "TORCH_MODELS", "OptionalDependencyError",
    "get_model_kind", "create_experiment_model",
]
