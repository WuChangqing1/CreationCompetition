"""Bimodal Collaborative Transformer (BCT) from DepFormer paper.

ACM MM 2025 — Elderly Track #1 architecture.
Two symmetric cross-attention branches: Audio→Video and Video→Audio.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _TransformerCrossAttentionLayer(nn.Module):
    """Single layer: Cross-Attn → Add&Norm → FFN → Add&Norm."""

    def __init__(
        self, hidden_dim: int, num_heads: int = 2, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
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
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-LN: norm before attention
        q_norm = self.norm1(query)
        kv_norm = self.norm1(key_value)
        attn_out, _ = self.cross_attn(
            query=q_norm, key=kv_norm, value=kv_norm,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = query + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class BimodalCollaborativeTransformer(nn.Module):
    """BCT: bidirectional cross-attention between audio and video sequences.

    N_layers x (two symmetric branches):
        - Audio→Video: Q=audio, K/V=video
        - Video→Audio: Q=video, K/V=audio
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 2,
        num_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.a2v_layers = nn.ModuleList([
            _TransformerCrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.v2a_layers = nn.ModuleList([
            _TransformerCrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        audio_seq: torch.Tensor,
        video_seq: torch.Tensor,
        audio_mask: torch.Tensor | None = None,
        video_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # audio_seq, video_seq: [B, T, H]
        a_out = audio_seq
        v_out = video_seq
        for a2v, v2a in zip(self.a2v_layers, self.v2a_layers):
            # a2v: audio queries attend to video
            a_out = a2v(a_out, v_out, key_padding_mask=video_mask)
            # v2a: video queries attend to audio
            v_out = v2a(v_out, a_out, key_padding_mask=audio_mask)
        return a_out, v_out
