"""Complete MPDD MulT adapter with late personality fusion."""

import torch
import torch.nn as nn

from models.av_model import AVClassificationModel
from models.networks.mult import MulTNetwork


class _MulTCore(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden = int(getattr(opt, "transformer_hidden_dim", 128))
        fusion = int(getattr(opt, "fusion_dim", hidden))
        self.mult = MulTNetwork(
            int(opt.input_dim_a), int(opt.input_dim_v), hidden, fusion,
            int(getattr(opt, "transformer_heads", 2)),
            int(getattr(opt, "transformer_layers", 1)),
            float(getattr(opt, "dropout_rate", 0.2)),
        )
        self.personality = nn.Linear(int(getattr(opt, "personality_dim", 1024)), fusion)
        self.classifier = nn.Sequential(
            nn.Linear(fusion * 2, fusion), nn.ReLU(),
            nn.Dropout(float(getattr(opt, "dropout_rate", 0.2))),
            nn.Linear(fusion, int(opt.emo_output_dim)),
        )

    def forward(self, audio, video, personality, mask_a, mask_v):
        multimodal = self.mult(audio, video, mask_a, mask_v)
        personality = torch.tanh(self.personality(personality))
        return self.classifier(torch.cat((multimodal, personality), dim=-1))


class MulTModel(AVClassificationModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.finish_initialization(_MulTCore(opt))
