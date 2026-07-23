"""CVAE Synthesizer for CMG-VS data augmentation.

Cross-Attention based CVAE adapted from CMG-VS (CVPR 2026).
Generates synthetic visual features conditioned on audio + personality sequences.

Architecture (aligned with paper):
    Encoder: Cross-Attn(v_seq, cond_seq) → Mean Pool → μ, log(σ²) → z
    Decoder: Cross-Attn(z, cond_seq) → synthetic visual sequence
"""

from __future__ import annotations

import torch
import torch.nn as nn


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(N(mu, sigma^2) || N(0, 1)), mean over batch and latent dims."""
    return -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())


class _CrossAttentionLayer(nn.Module):
    """Pre-LN Cross-Attention block: Cross-Attn → Add&Norm → FFN → Add&Norm.

    Same pattern as BCT's _TransformerCrossAttentionLayer (DepFormer paper).
    """

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
    ) -> torch.Tensor:
        q_norm = self.norm1(query)
        kv_norm = self.norm1(key_value)
        attn_out, _ = self.cross_attn(
            query=q_norm, key=kv_norm, value=kv_norm, need_weights=False,
        )
        x = query + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class CVAEEncoder(nn.Module):
    """CMG-VS Encoder: Cross-Attention over sequences → mean pool → μ, log(σ²).

    Q = target_seq (visual), K/V = cond_seq (audio + personality).
    After Cross-Attn, pools over time to get a single vector per sample,
    then projects to μ and log(σ²) for the latent distribution.
    """

    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        hidden_dim: int = 64,
        d_z: int = 16,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        # Project condition to same dim as target so Cross-Attn works
        self.cond_proj = (
            nn.Linear(cond_dim, hidden_dim)
            if cond_dim != hidden_dim
            else nn.Identity()
        )
        self.layers = nn.ModuleList([
            _CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.mu_head = nn.Linear(hidden_dim, d_z)
        self.logvar_head = nn.Linear(hidden_dim, d_z)

    def forward(
        self, target_seq: torch.Tensor, cond_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # target_seq: [B, T, target_dim], cond_seq: [B, T, cond_dim]
        cond = self.cond_proj(cond_seq)  # [B, T, hidden_dim]
        h = target_seq
        for layer in self.layers:
            h = layer(query=h, key_value=cond)
        # h: [B, T, hidden_dim]
        h_pooled = h.mean(dim=1)  # [B, hidden_dim]
        mu = self.mu_head(h_pooled)          # [B, d_z]
        logvar = self.logvar_head(h_pooled)   # [B, d_z]
        return mu, logvar


class CVAEDecoder(nn.Module):
    """CMG-VS Decoder: expand z to sequence → Cross-Attention → synthetic visual.

    Q = z (tiled to sequence length), K/V = cond_seq (audio + personality).
    The latent z carries visual semantics compressed from the Encoder;
    conditions fill in the temporal and cross-modal details.
    """

    def __init__(
        self,
        d_z: int,
        cond_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.z_proj = nn.Linear(d_z, hidden_dim)
        self.cond_proj = (
            nn.Linear(cond_dim, hidden_dim)
            if cond_dim != hidden_dim
            else nn.Identity()
        )
        self.layers = nn.ModuleList([
            _CrossAttentionLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, z: torch.Tensor, cond_seq: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        # z: [B, d_z], cond_seq: [B, T, cond_dim]
        z_up = self.z_proj(z)                        # [B, hidden_dim]
        z_seq = z_up.unsqueeze(1).expand(-1, seq_len, -1)  # [B, T, hidden_dim]
        cond = self.cond_proj(cond_seq)              # [B, T, hidden_dim]
        h = z_seq
        for layer in self.layers:
            h = layer(query=h, key_value=cond)
        return self.output_proj(h)  # [B, T, hidden_dim]


class CVAESynthesizer(nn.Module):
    """CMG-VS style CVAE for visual feature synthesis.

    Generates synthetic visual sequences conditioned on audio + personality.
    Used for task-guided data augmentation during training.

    Usage:
        # Training: encode target, reparameterize, decode
        f_v_synth, mu, logvar, z = cvae(v_seq, cond_seq)

        # Pure generation (no target): random z → decode
        f_v_synth = cvae.generate(cond_seq, T=128)
    """

    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        hidden_dim: int = 64,
        d_z: int = 16,
        num_layers: int = 1,
        num_heads: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = CVAEEncoder(
            target_dim=target_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            d_z=d_z,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.decoder = CVAEDecoder(
            d_z=d_z,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self, target_seq: torch.Tensor, cond_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full CVAE forward (training mode).

        Returns:
            f_synth:  [B, T, hidden_dim] synthetic visual sequence
            mu:       [B, d_z] posterior mean
            logvar:   [B, d_z] posterior log-variance
            z:        [B, d_z] latent vector
        """
        mu, logvar = self.encoder(target_seq, cond_seq)
        z = self.reparameterize(mu, logvar)
        T = target_seq.shape[1]
        f_synth = self.decoder(z, cond_seq, T)
        return f_synth, mu, logvar, z

    def generate(
        self, cond_seq: torch.Tensor, seq_len: int
    ) -> torch.Tensor:
        """Generate synthetic visual features from conditions only.

        Samples z from prior N(0, I) instead of posterior (for pure generation).
        """
        d_z = self.decoder.z_proj.in_features
        B = cond_seq.shape[0]
        z = torch.randn(B, d_z, device=cond_seq.device)
        return self.decoder(z, cond_seq, seq_len)
