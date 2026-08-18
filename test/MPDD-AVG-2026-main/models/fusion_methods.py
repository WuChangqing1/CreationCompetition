"""Paper-comparison fusion modules.

Each module captures the core transferable idea of one MPDD champion paper,
as a drop-in replacement for the vector-level fusion step in TorchcatBaseline
(after temporal pooling + pair aggregation).

All modules take a dict of per-modality vectors {modality: [B, H]} and return a
fused vector [B, n_mod * H] (identical output shape to CrossModalFusion), so the
backbone (DepFormer/BCT) and classifier stay fixed across comparisons.

Papers:
- PTMFIMFusion     -> Personality-Enhanced (MMAsia'25, 3743093.3770965)
- HOPEFusion       -> HOPE (ACM MM'25, 3746027.3762063)
- ReliabilityFusion-> MSF-ATS (ACM MM'25, 3746027.3762064)
- HypergraphFusion -> P3HF (AAAI'26, paper01)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _stack_others(features: dict[str, torch.Tensor],
                  modalities: list[str],
                  exclude: str = "personality") -> tuple[torch.Tensor, list[str]]:
    """Stack non-excluded modality vectors -> [B, M, H]; returns tensor + names."""
    names = [m for m in modalities if m != exclude]
    return torch.stack([features[m] for m in names], dim=1), names


class PTMFIMFusion(nn.Module):
    """PTMFIM (Personality-Enhanced): personality <-> multimodal interaction.

    Binary Correlation Attention (personality attends multimodal) ->
    Triple Interaction Attention (personality attends binary output) ->
    Gate Regulator (sigmoid gate blends binary/triple, adds to personality).
    """

    def __init__(self, hidden_dim: int, modalities: list[str], dropout: float = 0.3) -> None:
        super().__init__()
        self.modalities = modalities
        self.binary_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=2, dropout=dropout, batch_first=True)
        self.triple_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=2, dropout=dropout, batch_first=True)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        pers = features["personality"]                      # [B, H]
        multi, _ = _stack_others(features, self.modalities)  # [B, M, H]
        q = pers.unsqueeze(1)                               # [B, 1, H]
        binary_out, _ = self.binary_attn(q, multi, multi)   # [B, 1, H]
        binary_out = binary_out.squeeze(1)                  # [B, H]
        triple_out, _ = self.triple_attn(
            q, binary_out.unsqueeze(1), binary_out.unsqueeze(1))  # [B, 1, H]
        triple_out = triple_out.squeeze(1)                  # [B, H]
        gate = self.gate(torch.cat([binary_out, triple_out], dim=-1))  # [B, H]
        pers_enh = self.norm(pers + gate * self.proj(triple_out))      # [B, H]
        out = dict(features)
        out["personality"] = pers_enh
        return torch.cat([out[m] for m in self.modalities], dim=-1)


class HOPEFusion(nn.Module):
    """HOPE: personality-guided cross-attention over multimodal features.

    Personality acts as Query, multimodal features as Key/Value, producing a
    personality-conditioned context vector that is injected back into every
    non-personality modality (personality-aware enhancement).
    """

    def __init__(self, hidden_dim: int, modalities: list[str], dropout: float = 0.3) -> None:
        super().__init__()
        self.modalities = modalities
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads=2, dropout=dropout, batch_first=True)
        self.ctx_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        # per-modality injection gate
        n_other = max(1, len([m for m in modalities if m != "personality"]))
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        pers = features["personality"]                      # [B, H]
        multi, names = _stack_others(features, self.modalities)  # [B, M, H]
        q = pers.unsqueeze(1)                               # [B, 1, H]
        ctx, _ = self.cross_attn(q, multi, multi)           # [B, 1, H]
        ctx = self.ctx_proj(ctx.squeeze(1))                 # [B, H]
        out = dict(features)
        # inject personality-conditioned context into each non-personality modality
        for name in names:
            g = self.gate(torch.cat([features[name], ctx], dim=-1))  # [B, H]
            out[name] = self.norm(features[name] + g * ctx)
        out["personality"] = self.norm(pers + ctx)
        return torch.cat([out[m] for m in self.modalities], dim=-1)


class ReliabilityFusion(nn.Module):
    """MSF-ATS: non-negative per-modality reliability weights for stream fusion.

    Learns a scalar reliability weight per modality (softmax-normalized to be
    non-negative and sum to 1), scales each modality's feature, then concatenates.
    Mirrors the paper's W_{tau,m} reliability weights in eq.(7).
    """

    def __init__(self, hidden_dim: int, modalities: list[str], dropout: float = 0.3) -> None:
        super().__init__()
        self.modalities = modalities
        self.reliability = nn.Parameter(torch.zeros(len(modalities)))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        w = F.softmax(self.reliability, dim=0)              # [M], non-negative, sum=1
        out = {}
        for i, m in enumerate(self.modalities):
            out[m] = self.norm(w[i] * features[m])
        return torch.cat([out[m] for m in self.modalities], dim=-1)


class HypergraphFusion(nn.Module):
    """P3HF: high-order cross-modal interaction via hypergraph convolution.

    Modality vectors are treated as hypergraph nodes. Hyperedges = all pairwise
    modality sets + one full set, giving high-order (>=2 modality) interactions.
    A normalized clique-expansion hypergraph convolution propagates messages
    across nodes, followed by residual + projection.
    """

    def __init__(self, hidden_dim: int, modalities: list[str], dropout: float = 0.3) -> None:
        super().__init__()
        self.modalities = modalities
        n = len(modalities)
        self.incidence = self._build_incidence(n)          # [M, E]
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _build_incidence(n: int) -> torch.Tensor:
        """Incidence matrix H [M, E]: pairwise hyperedges + one full hyperedge."""
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                col = torch.zeros(n)
                col[i] = 1.0
                col[j] = 1.0
                edges.append(col)
        if n > 1:
            edges.append(torch.ones(n))                    # full hyperedge
        if not edges:
            edges = [torch.ones(n)]
        return torch.stack(edges, dim=1)                   # [M, E]

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        nodes = torch.stack([features[m] for m in self.modalities], dim=1)  # [B, M, H]
        H = self.incidence.to(nodes.device)                # [M, E]
        # clique-expansion adjacency A = H H^T, row-normalized (message passing)
        adj = H @ H.t()                                    # [M, M]
        deg = adj.sum(dim=1).clamp_min(1.0)               # [M]
        adj_norm = adj / deg.unsqueeze(1)                 # [M, M]
        propagated = torch.einsum('nm,bmh->bnh', adj_norm, nodes)  # [B, M, H]
        nodes = self.norm(nodes + self.dropout(self.proj(propagated)))  # residual
        return torch.cat([nodes[:, i, :] for i in range(len(self.modalities))], dim=-1)
