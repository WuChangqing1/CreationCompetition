"""Simple temporal-mean MLP baseline."""

import torch
import torch.nn as nn

from models.av_model import AVClassificationModel, masked_mean


class _MLPCore(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden = int(getattr(opt, "hidden_dim", getattr(opt, "hidden_size", 128)))
        in_dim = int(opt.input_dim_a) + int(opt.input_dim_v) + int(getattr(opt, "personality_dim", 1024))
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(float(getattr(opt, "dropout_rate", 0.2))),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(float(getattr(opt, "dropout_rate", 0.2))),
            nn.Linear(hidden, int(opt.emo_output_dim)),
        )

    def forward(self, audio, video, personality, mask_a, mask_v):
        fused = torch.cat((masked_mean(audio, mask_a), masked_mean(video, mask_v), personality), dim=-1)
        return self.classifier(fused)


class MLPModel(AVClassificationModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.finish_initialization(_MLPCore(opt))
