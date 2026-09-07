"""Safe optional entrypoint for DepMamba."""

import importlib.util

from experiments.errors import OptionalDependencyError
from models.base_model import BaseModel


class DepMambaModel(BaseModel):
    def __init__(self, opt):
        missing = [
            package for package in ("mamba_ssm", "causal_conv1d")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            raise OptionalDependencyError(
                "DepMamba optional dependencies unavailable: " + ", ".join(missing)
            )
        raise NotImplementedError(
            "DepMamba dependencies are available, but the MPDD adapter has not been validated yet."
        )

    def set_input(self, input):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def optimize_parameters(self, epoch=None):
        raise NotImplementedError
