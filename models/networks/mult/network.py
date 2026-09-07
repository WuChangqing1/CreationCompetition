"""Compact audio-video MulT representation network."""

import torch
import torch.nn as nn

from .transformer import CrossModalBlock


def _masked_mean(features, mask):
    weights = mask.unsqueeze(-1).type_as(features)
    return (features * weights).sum(1) / weights.sum(1).clamp_min(1.0)


class MulTNetwork(nn.Module):
    def __init__(self, audio_dim, video_dim, hidden_dim, output_dim, heads=2, layers=1, dropout=0.1):
        super().__init__()
        self.audio_projection = nn.Linear(audio_dim, hidden_dim)
        self.video_projection = nn.Linear(video_dim, hidden_dim)
        feedforward_dim = hidden_dim * 4
        self.audio_from_video = nn.ModuleList([
            CrossModalBlock(hidden_dim, heads, feedforward_dim, dropout) for _ in range(layers)
        ])
        self.video_from_audio = nn.ModuleList([
            CrossModalBlock(hidden_dim, heads, feedforward_dim, dropout) for _ in range(layers)
        ])
        self.output = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, audio, video, audio_mask, video_mask):
        audio_state = torch.tanh(self.audio_projection(audio))
        video_state = torch.tanh(self.video_projection(video))
        for block in self.audio_from_video:
            audio_state = block(audio_state, video_state, video_mask)
        for block in self.video_from_audio:
            video_state = block(video_state, audio_state, audio_mask)
        pooled = torch.cat(
            (_masked_mean(audio_state, audio_mask), _masked_mean(video_state, video_mask)), dim=-1
        )
        return torch.tanh(self.output(pooled))
