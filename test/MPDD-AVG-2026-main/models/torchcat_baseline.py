from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .bct import BimodalCollaborativeTransformer
from .cvae_synthesizer import CVAESynthesizer, kl_divergence
from .depformer_temporal_encoder import DepFormerTemporalEncoder
from .fusion_methods import (
    HOPEFusion, HypergraphFusion, PTMFIMFusion, ReliabilityFusion,
)
from .hybrid_temporal_encoder import HybridTemporalEncoder
from .temporal_transformer import TemporalTransformerEncoder


# ===========================================================================
# Attention Statistics Pooling (from MSF-ATS paper, ACM MM 2025)
# ===========================================================================
class AttentionStatisticsPooling(nn.Module):
    """Learned attention pooling over time steps, producing mean + std.

    Replaces naive mean(dim=1). The attention mechanism lets the model
    focus on depression-relevant segments while std captures variability.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim // 2)
        self.attn_score = nn.Linear(dim // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D] -> [B, 2D]
        scores = self.attn_score(torch.tanh(self.proj(x)))  # [B, T, 1]
        weights = torch.softmax(scores, dim=1)  # [B, T, 1]
        weighted_mean = (weights * x).sum(dim=1)  # [B, D]
        diff = x - weighted_mean.unsqueeze(1)
        weighted_var = (weights * diff ** 2).sum(dim=1)
        weighted_std = torch.sqrt(weighted_var + 1e-8)
        return torch.cat([weighted_mean, weighted_std], dim=-1)  # [B, 2D]


# ===========================================================================
# Cross-Modal Fusion (from DepFormer + Paper1, ACM MM 2025)
# ===========================================================================
class CrossModalFusion(nn.Module):
    """Gated cross-modal fusion replacing simple torch.cat.

    - Audio <-> Video: bidirectional gated interaction (DepFormer BCT adapted)
    - Personality -> Multimodal: gated interaction (Paper1 PTMFIM adapted)

    Uses lightweight gates instead of full multi-head attention for
    flat vectors, keeping parameter count low for small datasets.
    """

    def __init__(
        self,
        hidden_dim: int,
        modalities: list[str],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.modalities = modalities
        n_mod = len(modalities)

        # Audio-Video bidirectional gate
        if "audio" in modalities and "video" in modalities:
            self.av_gate_a = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.av_gate_v = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.av_proj_a = nn.Linear(hidden_dim, hidden_dim)
            self.av_proj_v = nn.Linear(hidden_dim, hidden_dim)

        # Personality-multimodal gate
        non_pers = [m for m in modalities if m != "personality"]
        if "personality" in modalities and len(non_pers) > 0:
            self.pers_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid(),
            )
            self.pers_proj = nn.Linear(hidden_dim, hidden_dim)

        # Residual fusion MLP (applied to concatenated features)
        concat_dim = hidden_dim * n_mod
        self.fusion_mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(concat_dim * 2, concat_dim),
        )
        self.fusion_norm = nn.LayerNorm(concat_dim)

    def forward(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        # features: {"audio": [B,D], "video": [B,D], "gait": [B,D], "personality": [B,D]}
        enhanced = dict(features)

        # --- Audio <-> Video cross-modal gating ---
        if "audio" in features and "video" in features:
            a = features["audio"]
            v = features["video"]
            joint_av = torch.cat([a, v], dim=-1)
            enhanced["audio"] = a + self.av_gate_a(joint_av) * self.av_proj_v(v)
            enhanced["video"] = v + self.av_gate_v(joint_av) * self.av_proj_a(a)

        # --- Personality <-> Multimodal gating ---
        if "personality" in features:
            other_mods = [k for k in self.modalities if k != "personality"]
            if other_mods:
                multimodal = torch.stack(
                    [enhanced[k] for k in other_mods], dim=1
                ).mean(dim=1)  # [B, D]
                pers = features["personality"]
                gate = self.pers_gate(
                    torch.cat([pers, multimodal], dim=-1)
                )
                enhanced["personality"] = pers + gate * self.pers_proj(multimodal)

        # --- Concatenate + residual MLP ---
        concat = torch.cat([enhanced[m] for m in self.modalities], dim=-1)
        return self.fusion_norm(concat + self.fusion_mlp(concat))


# ===========================================================================
# Modality Encoders
# ===========================================================================
class ModalityEncoder(nn.Module):
    """BiLSTM temporal encoder, optionally with ASP pooling."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        dropout: float = 0.5,
        pre_dim: int | None = None,
        use_asp: bool = True,
    ) -> None:
        super().__init__()
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
        self.use_asp = use_asp
        if use_asp:
            self.asp = AttentionStatisticsPooling(hidden_dim)
            # Project 2D (mean+std) back to D
            self.asp_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_proj is not None:
            x = F.relu(self.pre_proj(x))
        x = F.relu(self.proj(x))
        x = self.dropout(x)
        x, _ = self.lstm(x)
        if self.use_asp:
            x = self.asp(x)
            x = self.asp_proj(x)
        else:
            x = x.mean(dim=1)
        return self.norm(x)


class PersonalityEncoder(nn.Module):
    """2-layer MLP encoding 1024-dim RoBERTa personality embeddings."""

    def __init__(
        self, input_dim: int = 1024, hidden_dim: int = 64, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ===========================================================================
# Main Model
# ===========================================================================
class TorchcatBaseline(nn.Module):
    SUBTRACKS = {
        "A-V+P": ["audio", "video", "personality"],
        "A-V-G+P": ["audio", "video", "gait", "personality"],
        "G+P": ["gait", "personality"],
    }
    ENCODER_TYPES = {"bilstm_mean", "hybrid_attn", "depformer"}

    def __init__(
        self,
        subtrack: str = "A-V-G+P",
        num_classes: int = 3,
        is_regression: bool = False,
        use_regression_head: bool = False,
        audio_dim: int = 64,
        video_dim: int = 1000,
        gait_dim: int = 12,
        hidden_dim: int = 64,
        dropout: float = 0.3,
        encoder_type: str = "bilstm_mean",
        use_asp: bool = True,
        use_cross_fusion: bool = True,
        fusion_type: str = "cross_fusion",
        # CVAE data augmentation (CMG-VS style)
        use_cvae: bool = False,
        cvae_d_z: int = 16,
        cvae_num_layers: int = 1,
        cvae_num_heads: int = 2,
    ) -> None:
        super().__init__()
        if subtrack not in self.SUBTRACKS:
            raise ValueError(f"Unknown subtrack: {subtrack}")
        if encoder_type not in self.ENCODER_TYPES:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

        self.subtrack = subtrack
        self.modalities = self.SUBTRACKS[subtrack]
        self.encoder_type = encoder_type
        self.is_regression = is_regression
        self.use_regression_head = use_regression_head
        self.use_cross_fusion = use_cross_fusion

        # ASP is only for bilstm_mean (hybrid_attn / depformer have their own pooling)
        encoder_use_asp = use_asp and encoder_type == "bilstm_mean"
        self.is_depformer = encoder_type == "depformer"

        if self.is_depformer:
            # ── DepFormer path: BiLSTM keeps temporal dim → BCT/TemporalTransformer → pool ──
            if "audio" in self.modalities:
                pre_audio = 128 if audio_dim > 128 else None
                self.audio_enc = DepFormerTemporalEncoder(
                    audio_dim, hidden_dim, dropout, pre_dim=pre_audio,
                )
            if "video" in self.modalities:
                pre_video = 128 if video_dim > 128 else None
                self.video_enc = DepFormerTemporalEncoder(
                    video_dim, hidden_dim, dropout, pre_dim=pre_video,
                )
            if "audio" in self.modalities and "video" in self.modalities:
                self.bct = BimodalCollaborativeTransformer(
                    hidden_dim, num_heads=2, dropout=dropout,
                )
            # CVAE data augmentation (CMG-VS): audio+personality → synthetic video
            self.cvae = None
            if (use_cvae and "audio" in self.modalities
                    and "video" in self.modalities
                    and "personality" in self.modalities):
                cond_dim = hidden_dim * 2  # concat(audio_seq, pers_seq)
                self.cvae = CVAESynthesizer(
                    target_dim=hidden_dim,
                    cond_dim=cond_dim,
                    hidden_dim=hidden_dim,
                    d_z=cvae_d_z,
                    num_layers=cvae_num_layers,
                    num_heads=cvae_num_heads,
                    dropout=dropout,
                )
            if "gait" in self.modalities:
                self.gait_enc = DepFormerTemporalEncoder(
                    gait_dim, hidden_dim, dropout,
                )
                self.gait_transformer = TemporalTransformerEncoder(
                    hidden_dim, num_heads=2, dropout=dropout,
                )
        else:
            # ── Existing paths: bilstm_mean / hybrid_attn ──
            if "audio" in self.modalities:
                pre_audio = 128 if audio_dim > 128 else None
                self.audio_enc = (
                    HybridTemporalEncoder(audio_dim, hidden_dim, dropout, pre_dim=pre_audio)
                    if encoder_type == "hybrid_attn"
                    else ModalityEncoder(
                        audio_dim, hidden_dim, dropout, pre_dim=pre_audio, use_asp=encoder_use_asp
                    )
                )
            if "video" in self.modalities:
                pre_video = 128 if video_dim > 128 else None
                self.video_enc = (
                    HybridTemporalEncoder(video_dim, hidden_dim, dropout, pre_dim=pre_video)
                    if encoder_type == "hybrid_attn"
                    else ModalityEncoder(
                        video_dim, hidden_dim, dropout, pre_dim=pre_video, use_asp=encoder_use_asp
                    )
                )
            if "gait" in self.modalities:
                self.gait_enc = ModalityEncoder(
                    gait_dim, hidden_dim, dropout, use_asp=encoder_use_asp
                )

        if "personality" in self.modalities:
            self.pers_enc = PersonalityEncoder(1024, hidden_dim, dropout)

        self.fusion = self._build_fusion(fusion_type, use_cross_fusion, hidden_dim, dropout)

        fused_dim = hidden_dim * len(self.modalities)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1 if is_regression else num_classes),
        )
        if use_regression_head:
            self.regressor = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )

    def _build_fusion(self, fusion_type, use_cross_fusion, hidden_dim, dropout):
        """Build the vector-level fusion module by name (paper comparison)."""
        if not use_cross_fusion or fusion_type == "none":
            return None
        if fusion_type == "cross_fusion":
            return CrossModalFusion(hidden_dim, self.modalities, dropout)
        if fusion_type == "ptmfim":
            return PTMFIMFusion(hidden_dim, self.modalities, dropout)
        if fusion_type == "hope":
            return HOPEFusion(hidden_dim, self.modalities, dropout)
        if fusion_type == "reliability":
            return ReliabilityFusion(hidden_dim, self.modalities, dropout)
        if fusion_type == "hypergraph":
            return HypergraphFusion(hidden_dim, self.modalities, dropout)
        raise ValueError(f"Unknown fusion_type: {fusion_type}")

    @staticmethod
    def _masked_average_sequences(
        x: torch.Tensor, pair_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1).unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    @staticmethod
    def _masked_average_features(
        x: torch.Tensor, pair_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if pair_mask is None:
            return x.mean(dim=1)
        weights = pair_mask.unsqueeze(-1)
        denom = weights.sum(dim=1).clamp_min(1.0)
        return (x * weights).sum(dim=1) / denom

    @staticmethod
    def _temporal_masked_pool(x: torch.Tensor) -> torch.Tensor:
        """Masked average pooling over time dim.

        Ignores all-zero timesteps (padding) during averaging.
        x: [B, T, D] -> [B, D]
        """
        mask = (x.abs().sum(dim=-1) > 1e-8).float()  # [B, T]
        denom = mask.sum(dim=1).clamp_min(1.0)        # [B]
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom.unsqueeze(-1)

    def _encode_pairwise_sequences(
        self,
        x: torch.Tensor,
        encoder: nn.Module,
        pair_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, pair_count, seq_len, feat_dim = x.shape
        encoded = encoder(x.reshape(batch_size * pair_count, seq_len, feat_dim))
        encoded = encoded.reshape(batch_size, pair_count, -1)
        return self._masked_average_features(encoded, pair_mask)

    def forward(
        self,
        audio: torch.Tensor | None = None,
        video: torch.Tensor | None = None,
        gait: torch.Tensor | None = None,
        personality: torch.Tensor | None = None,
        pair_mask: torch.Tensor | None = None,
        return_cvae_outputs: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        features: dict[str, torch.Tensor] = {}
        v_pooled_synth = None  # only set when CVAE is active

        if self.is_depformer:
            # ── DepFormer path: encode per-pair → BCT → [CVAE] → pool → aggregate pairs ──
            # Encode personality early (needed early if CVAE uses it as condition)
            pers_enc = None
            if "personality" in self.modalities:
                pers_enc = self.pers_enc(personality)  # [B, H]

            if "audio" in self.modalities and "video" in self.modalities:
                B, P, T, Da = audio.shape
                _, _, _, Dv = video.shape
                a_seq = self.audio_enc(audio.reshape(B * P, T, Da))   # [B*P, T, H]
                v_seq = self.video_enc(video.reshape(B * P, T, Dv))   # [B*P, T, H]
                a_seq, v_seq = self.bct(a_seq, v_seq)                  # [B*P, T, H]

                # ── CVAE data augmentation (CMG-VS style) ──
                if return_cvae_outputs and self.cvae is not None:
                    pers_per_pair = pers_enc.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)  # [B*P, H]
                    pers_seq = pers_per_pair.unsqueeze(1).expand(-1, T, -1)  # [B*P, T, H]
                    cond_seq = torch.cat([a_seq, pers_seq], dim=-1)  # [B*P, T, 2H]

                    # detach: CMG-VS paper treats f_v / f_a as FIXED features, so
                    # L_consis / L_KL must only update the CVAE (not the main
                    # encoders/BCT). Task-guided feedback still reaches the CVAE
                    # through L_aug -> f_v_synth_seq -> decoder/encoder params.
                    f_v_synth_seq, mu, logvar, z = self.cvae(
                        v_seq.detach(), cond_seq.detach())  # [B*P, T, H]

                    # Pool synthetic and real video separately
                    v_pooled_real = self._temporal_masked_pool(v_seq)
                    v_pooled_synth = self._temporal_masked_pool(f_v_synth_seq)
                else:
                    v_pooled_real = self._temporal_masked_pool(v_seq)
                    v_pooled_synth = None

                a_pooled = self._temporal_masked_pool(a_seq)           # [B*P, H]
                features["audio"] = self._masked_average_features(
                    a_pooled.reshape(B, P, -1), pair_mask,
                )
                features["video"] = self._masked_average_features(
                    v_pooled_real.reshape(B, P, -1), pair_mask,
                )

            if "gait" in self.modalities:
                g_seq = self.gait_enc(gait)                      # [B, T, H]
                g_seq = self.gait_transformer(g_seq)             # [B, T, H]
                features["gait"] = self._temporal_masked_pool(g_seq)  # [B, H]

            if "personality" in self.modalities:
                features["personality"] = pers_enc
        else:
            # ── Existing paths: bilstm_mean / hybrid_attn ──
            if "audio" in self.modalities:
                if self.encoder_type == "hybrid_attn":
                    features["audio"] = self._encode_pairwise_sequences(
                        audio, self.audio_enc, pair_mask
                    )
                else:
                    features["audio"] = self.audio_enc(
                        self._masked_average_sequences(audio, pair_mask)
                    )

            if "video" in self.modalities:
                if self.encoder_type == "hybrid_attn":
                    features["video"] = self._encode_pairwise_sequences(
                        video, self.video_enc, pair_mask
                    )
                else:
                    features["video"] = self.video_enc(
                        self._masked_average_sequences(video, pair_mask)
                    )

            if "gait" in self.modalities:
                features["gait"] = self.gait_enc(gait)

            if "personality" in self.modalities:
                features["personality"] = self.pers_enc(personality)

        # ── CVAE dual-stream fusion + classify (batched) ──
        if return_cvae_outputs and v_pooled_synth is not None:
            # Build augmented features and stack with real → single forward pass
            features_aug = dict(features)
            if "audio" in self.modalities and "video" in self.modalities:
                B, P = audio.shape[0], audio.shape[1]
                features_aug["video"] = self._masked_average_features(
                    v_pooled_synth.reshape(B, P, -1), pair_mask,
                )

            # Stack real + aug along batch dim → [2*B, ...]
            keys = list(features.keys())
            stacked = {k: torch.cat([features[k], features_aug[k]], dim=0) for k in keys}

            if self.fusion is not None:
                fused_both = self.fusion(stacked)             # [2*B, fused_dim]
            else:
                fused_both = torch.cat(list(stacked.values()), dim=-1)
            logits_both = self.classifier(fused_both)          # [2*B, num_classes]
            logits_real, logits_aug = logits_both.chunk(2, dim=0)

            return {
                "logits_real": logits_real,
                "logits_aug": logits_aug,
                "v_pooled_real": v_pooled_real,
                "v_pooled_synth": v_pooled_synth,
                "v_seq_real": v_seq,          # [B*P, T, H] real video seq (BCT out)
                "v_synth_seq": f_v_synth_seq, # [B*P, T, H] synthesized video seq
                "mu": mu,
                "logvar": logvar,
            }

        # ── Normal forward (no CVAE or inference mode) ──
        if self.fusion is not None:
            fused = self.fusion(features)
        else:
            fused = torch.cat(list(features.values()), dim=-1)

        logits = self.classifier(fused)
        if self.use_regression_head:
            return logits, self.regressor(fused).squeeze(-1)
        return logits, None  # None for reg_out to keep tuple unpacking compatible
