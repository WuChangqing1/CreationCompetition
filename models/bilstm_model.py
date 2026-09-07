"""Bidirectional LSTM audio-visual baseline."""

import torch
import torch.nn as nn

from models.av_model import AVClassificationModel, masked_mean
from models.networks.lstm import LSTMEncoder


class _BiLSTMCore(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden = int(getattr(opt, "bilstm_hidden_dim", 64))
        self.audio_encoder = LSTMEncoder(int(opt.input_dim_a), hidden, "last", bidirectional=True)
        self.video_encoder = LSTMEncoder(int(opt.input_dim_v), hidden, "last", bidirectional=True)
        self.personality = nn.Linear(int(getattr(opt, "personality_dim", 1024)), hidden * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 6, hidden * 2), nn.ReLU(),
            nn.Dropout(float(getattr(opt, "dropout_rate", 0.2))),
            nn.Linear(hidden * 2, int(opt.emo_output_dim)),
        )

    def forward(self, audio, video, personality, mask_a, mask_v):
        encoded_a = masked_mean(self.audio_encoder(audio), mask_a)
        encoded_v = masked_mean(self.video_encoder(video), mask_v)
        encoded_p = torch.tanh(self.personality(personality))
        return self.classifier(torch.cat((encoded_a, encoded_v, encoded_p), dim=-1))


class BiLSTMModel(AVClassificationModel):
    def __init__(self, opt):
        super().__init__(opt)
        self.finish_initialization(_BiLSTMCore(opt))
