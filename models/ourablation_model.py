"""OurModel ablation variants implemented without changing the original model."""

import torch
import torch.nn.functional as F

from models.our.our_model import ourModel


ABLATION_VARIANTS = (
    "full",
    "no_personality",
    "no_vem",
    "no_cfm",
    "no_a2v",
    "no_v2a",
    "no_aux_focal",
    "audio_only",
    "visual_only",
    "personality_only",
)


def normalize_ablation_variant(value):
    variant = str(value or "full").strip().lower()
    if variant not in ABLATION_VARIANTS:
        raise ValueError(
            f"Unknown OurModel ablation variant: {value!r}. "
            f"Choose one of: {', '.join(ABLATION_VARIANTS)}"
        )
    return variant


class OurablationModel(ourModel):
    """A thin OurModel subclass whose default ``full`` path is unchanged."""

    def __init__(self, opt):
        self.ablation_variant = normalize_ablation_variant(
            getattr(opt, "ablation_variant", "full")
        )
        super().__init__(opt)
        if self.ablation_variant == "no_aux_focal" and self.isTrain:
            self.focal_weight = 0.0

    def forward(self, acoustic_feat=None, visual_feat=None):
        if self.ablation_variant in {"full", "no_aux_focal"}:
            return super().forward(acoustic_feat, visual_feat)

        self._set_external_features(acoustic_feat, visual_feat)
        use_audio = self.ablation_variant not in {"visual_only", "personality_only"}
        use_visual = self.ablation_variant not in {"audio_only", "personality_only"}
        use_personality = self.ablation_variant not in {
            "no_personality", "audio_only", "visual_only"
        }

        audio_feat = self._encode_audio() if use_audio else None
        visual_feat = self._encode_visual() if use_visual else None

        if use_visual and self.ablation_variant != "no_vem":
            visual_feat = self.netVEM(visual_feat, key_padding_mask=~self.mask_v)

        if use_audio and use_visual:
            if self.ablation_variant == "no_cfm":
                pass
            elif self.ablation_variant in {"no_a2v", "no_v2a"}:
                audio_feat, visual_feat = self._directional_feedback(audio_feat, visual_feat)
            else:
                audio_feat, visual_feat = self.netCFM(audio_feat, visual_feat)

        batch_size = self.acoustic.size(0)
        hidden_size = self.netProjP.out_features
        reference = self.acoustic
        pooled_a = (
            self.masked_avg_pool(audio_feat, self.mask_a)
            if audio_feat is not None
            else torch.zeros(batch_size, hidden_size, device=reference.device, dtype=reference.dtype)
        )
        pooled_v = (
            self.masked_avg_pool(visual_feat, self.mask_v)
            if visual_feat is not None
            else torch.zeros(batch_size, hidden_size, device=reference.device, dtype=reference.dtype)
        )

        if use_personality and self.personalized is not None:
            personality = self.netProjP(self.personalized)
        else:
            personality = torch.zeros(
                batch_size, hidden_size, device=reference.device, dtype=reference.dtype
            )
        fused = torch.cat((pooled_a, pooled_v, personality), dim=-1)
        self.emo_logits_fusion, _ = self.netEmoCF(fused)
        self.emo_logits, _ = self.netEmoC(fused)
        self.emo_pred = F.softmax(self.emo_logits, dim=-1)

    def _set_external_features(self, acoustic_feat, visual_feat):
        if acoustic_feat is None:
            return
        device = next(self.parameters()).device
        self.acoustic = acoustic_feat.float().to(device)
        self.visual = visual_feat.float().to(device)
        self.mask_a = (acoustic_feat.abs().sum(dim=-1) > 0).to(device)
        self.mask_v = (visual_feat.abs().sum(dim=-1) > 0).to(device)

    def _encode_audio(self):
        return torch.tanh(self.netProjA(self.netEmoA(self.acoustic)))

    def _encode_visual(self):
        return torch.tanh(self.netProjV(self.netEmoV(self.visual)))

    def _directional_feedback(self, audio_feat, visual_feat):
        dim = audio_feat.size(-1)
        audio_enhanced = audio_feat
        visual_enhanced = visual_feat

        if self.ablation_variant != "no_v2a":
            visual_summary = self.netCFM.visual_adapter(
                visual_feat, target_len=audio_feat.size(1)
            )
            visual_feedback = visual_summary.unsqueeze(-1).expand(-1, -1, dim)
            audio_gate = self.netCFM.gate_a(
                torch.cat((audio_feat, visual_feedback), dim=-1)
            )
            audio_enhanced = audio_feat + audio_gate * visual_feedback

        if self.ablation_variant != "no_a2v":
            audio_summary = self.netCFM.audio_adapter(
                audio_feat, target_len=visual_feat.size(1)
            )
            audio_feedback = audio_summary.unsqueeze(-1).expand(-1, -1, dim)
            visual_gate = self.netCFM.gate_v(
                torch.cat((visual_feat, audio_feedback), dim=-1)
            )
            visual_enhanced = visual_feat + visual_gate * audio_feedback

        return audio_enhanced, visual_enhanced


__all__ = ["ABLATION_VARIANTS", "OurablationModel", "normalize_ablation_variant"]
