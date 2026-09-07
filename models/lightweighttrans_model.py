"""MPDD adapter around the existing lightweight Transformer component."""

import torch
import torch.nn as nn

from models.av_model import AVClassificationModel, masked_mean
from models.networks.LightWeightTrans import TransEncoder


class _LightWeightTransCore(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden = int(getattr(opt, "transformer_hidden_dim", 128))
        heads = int(getattr(opt, "transformer_heads", 2))
        layers = int(getattr(opt, "transformer_layers", 1))
        ffn = int(getattr(opt, "transformer_ffn_dim", hidden * 4))
        dropout = float(getattr(opt, "dropout_rate", 0.2))
        self.audio_encoder = TransEncoder((int(opt.input_dim_a), hidden), hidden, heads, layers, ffn, dropout)
        self.video_encoder = TransEncoder((int(opt.input_dim_v), hidden), hidden, heads, layers, ffn, dropout)
        self.personality = nn.Linear(int(getattr(opt, "personality_dim", 1024)), hidden)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, int(opt.emo_output_dim)),
        )

    def forward(self, audio, video, personality, mask_a, mask_v):
        encoded_a, _ = self.audio_encoder(audio.transpose(0, 1), src_key_padding_mask=~mask_a)
        encoded_v, _ = self.video_encoder(video.transpose(0, 1), src_key_padding_mask=~mask_v)
        pooled_a = masked_mean(encoded_a.transpose(0, 1), mask_a)
        pooled_v = masked_mean(encoded_v.transpose(0, 1), mask_v)
        pooled_p = torch.tanh(self.personality(personality))
        return self.classifier(torch.cat((pooled_a, pooled_v, pooled_p), dim=-1))


class LightWeightTransModel(AVClassificationModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.finish_initialization(_LightWeightTransCore(opt))
