"""Residual cross-modal Transformer block."""

import torch.nn as nn

from .attention import CrossModalAttention


class CrossModalBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, feedforward_dim, dropout):
        super().__init__()
        self.cross_attention = CrossModalAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
        )

    def forward(self, query, source, source_valid_mask=None):
        query = self.norm1(query + self.dropout(self.cross_attention(query, source, source_valid_mask)))
        return self.norm2(query + self.dropout(self.feedforward(query)))
