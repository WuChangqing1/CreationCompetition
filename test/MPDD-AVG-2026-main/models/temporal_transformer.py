"""Temporal Transformer Encoder for single-modality self-attention.

Used for gait temporal sequences in the G+P subtrack (no audio/video).
Single-layer self-attention + FFN, same pattern as one BCT branch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalTransformerEncoder(nn.Module):
    """Self-attention Transformer for temporal sequences.

    Single layer: Self-Attn → Add&Norm → FFN → Add&Norm.
    Input [B, T, H] → Output [B, T, H].
    """

    def __init__(
        self, hidden_dim: int, num_heads: int = 2, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN
        x_norm = self.norm1(x)
        attn_out, _ = self.self_attn(
            query=x_norm, key=x_norm, value=x_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x
