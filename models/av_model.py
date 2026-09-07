"""Shared BaseModel adapter for complete audio-visual classifiers."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_model import BaseModel


def valid_mask(features):
    return features.abs().sum(dim=-1) > 0


def masked_mean(features, mask):
    weights = mask.unsqueeze(-1).type_as(features)
    return (features * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class AVClassificationModel(BaseModel, nn.Module):
    """Common input, probability and optimization behavior for new models."""

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        nn.Module.__init__(self)
        self.loss_names = ["emo_CE"]
        self.model_names = []
        self.personality_dim = int(getattr(opt, "personality_dim", 1024))
        self.use_personality = bool(getattr(opt, "use_personality", True))

    def finish_initialization(self, core):
        self.netCore = core
        self.model_names = ["Core"]
        self.criterion_ce = nn.CrossEntropyLoss()
        if self.isTrain:
            optimizer_name = str(getattr(self.opt, "optimizer", "adam")).lower()
            kwargs = {
                "lr": float(getattr(self.opt, "lr", 1e-3)),
                "weight_decay": float(getattr(self.opt, "weight_decay", 0.0)),
            }
            if optimizer_name == "adamw":
                self.optimizer = torch.optim.AdamW(self.netCore.parameters(), **kwargs)
            else:
                kwargs["betas"] = (float(getattr(self.opt, "beta1", 0.9)), 0.999)
                self.optimizer = torch.optim.Adam(self.netCore.parameters(), **kwargs)
            self.optimizers = [self.optimizer]

    def set_input(self, input):
        device = next(self.parameters()).device
        self.acoustic = input["A_feat"].float().to(device)
        self.visual = input["V_feat"].float().to(device)
        self.mask_a = valid_mask(self.acoustic)
        self.mask_v = valid_mask(self.visual)
        self.emo_label = input["emo_label"].long().to(device)
        personalized = input.get("personalized_feat")
        if personalized is None or not self.use_personality:
            personalized = torch.zeros(
                self.acoustic.size(0), self.personality_dim,
                dtype=self.acoustic.dtype, device=device,
            )
        self.personalized = personalized.float().to(device)

    def forward(self):
        self.emo_logits = self.netCore(
            self.acoustic, self.visual, self.personalized, self.mask_a, self.mask_v
        )
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)
        return self.emo_logits

    def optimize_parameters(self, epoch=None):
        if not self.isTrain:
            raise RuntimeError("optimize_parameters requires isTrain=True")
        self.forward()
        self.optimizer.zero_grad()
        self.loss_emo_CE = self.criterion_ce(self.emo_logits, self.emo_label)
        self.loss_emo_CE.backward()
        torch.nn.utils.clip_grad_norm_(self.netCore.parameters(), 1.0)
        self.optimizer.step()
