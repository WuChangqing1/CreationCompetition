"""Reserved interface for the team's future proposed model."""

from models.base_model import BaseModel


class ProposedModel(BaseModel):
    def __init__(self, opt):
        raise NotImplementedError("Proposed Model has not been implemented yet.")

    def set_input(self, input):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def optimize_parameters(self, epoch=None):
        raise NotImplementedError
