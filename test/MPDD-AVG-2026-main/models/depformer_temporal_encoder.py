"""BiLSTM encoder that keeps the full temporal sequence (no pooling).

Used as the per-modality pre-encoder before BCT cross-attention.
Different from ModalityEncoder: forward() returns [B, T, H] instead of [B, H].
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepFormerTemporalEncoder(nn.Module):
    """BiLSTM encoder returning the full temporal sequence.

    Keeps temporal resolution so downstream BCT can apply cross-attention
    at every timestep before pooling.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.5,
        pre_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError(f"hidden_dim must be even for BiLSTM, got {hidden_dim}")

        if pre_dim is not None and input_dim > pre_dim:
            self.pre_proj = nn.Linear(input_dim, pre_dim)
            lstm_in = pre_dim
        else:
            self.pre_proj = None
            lstm_in = input_dim

        self.proj = nn.Linear(lstm_in, hidden_dim)
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D_in]
        if self.pre_proj is not None:
            x = F.relu(self.pre_proj(x))
        x = F.relu(self.proj(x))
        x = self.dropout(x)
        x, _ = self.lstm(x)
        x = self.dropout(x)
        return self.norm(x)  # [B, T, hidden_dim]
