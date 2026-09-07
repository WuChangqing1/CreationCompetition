"""Low-rank multimodal fusion layer adapted for MPDD features."""

import torch
import torch.nn as nn


class LowRankFusion(nn.Module):
    def __init__(self, audio_dim, video_dim, personality_dim, output_dim, rank=4):
        super().__init__()
        self.audio_factor = nn.Parameter(torch.empty(rank, audio_dim + 1, output_dim))
        self.video_factor = nn.Parameter(torch.empty(rank, video_dim + 1, output_dim))
        self.personality_factor = nn.Parameter(torch.empty(rank, personality_dim + 1, output_dim))
        self.fusion_weights = nn.Parameter(torch.ones(rank))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.audio_factor)
        nn.init.xavier_normal_(self.video_factor)
        nn.init.xavier_normal_(self.personality_factor)

    def forward(self, audio, video, personality):
        ones = audio.new_ones(audio.size(0), 1)
        audio = torch.cat((ones, audio), dim=1)
        video = torch.cat((ones, video), dim=1)
        personality = torch.cat((ones, personality), dim=1)
        audio_f = torch.einsum("bi,rio->bro", audio, self.audio_factor)
        video_f = torch.einsum("bi,rio->bro", video, self.video_factor)
        personality_f = torch.einsum("bi,rio->bro", personality, self.personality_factor)
        fused = audio_f * video_f * personality_f
        return torch.einsum("bro,r->bo", fused, self.fusion_weights) + self.bias
