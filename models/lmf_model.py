"""Complete LMF model adapter for MPDD."""

import torch
import torch.nn as nn

from models.av_model import AVClassificationModel, masked_mean
from models.networks.lmf import LowRankFusion


class _LMFCore(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden = int(getattr(opt, "fusion_dim", 128))
        self.audio_projection = nn.Linear(int(opt.input_dim_a), hidden)
        self.video_projection = nn.Linear(int(opt.input_dim_v), hidden)
        self.personality_projection = nn.Linear(int(getattr(opt, "personality_dim", 1024)), hidden)
        self.fusion = LowRankFusion(hidden, hidden, hidden, hidden, int(getattr(opt, "lmf_rank", 4)))
        self.classifier = nn.Sequential(
            nn.ReLU(), nn.Dropout(float(getattr(opt, "dropout_rate", 0.2))),
            nn.Linear(hidden, int(opt.emo_output_dim)),
        )

    def forward(self, audio, video, personality, mask_a, mask_v):
        audio = torch.tanh(self.audio_projection(masked_mean(audio, mask_a)))
        video = torch.tanh(self.video_projection(masked_mean(video, mask_v)))
        personality = torch.tanh(self.personality_projection(personality))
        return self.classifier(self.fusion(audio, video, personality))


class LMFModel(AVClassificationModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.finish_initialization(_LMFCore(opt))
