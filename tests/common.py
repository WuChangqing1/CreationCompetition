from types import SimpleNamespace

import torch


def make_opt(**overrides):
    values = {
        "model": "our",
        "gpu_ids": [],
        "isTrain": False,
        "checkpoints_dir": "./checkpoints",
        "name": "smoke_test",
        "cuda_benchmark": False,
        "input_dim_a": 8,
        "embd_size_a": 4,
        "embd_method_a": "last",
        "input_dim_v": 10,
        "embd_size_v": 4,
        "embd_method_v": "last",
        "emo_output_dim": 2,
        "cls_layers": "8,4",
        "dropout_rate": 0.0,
        "hidden_size": 8,
        "lr": 1e-3,
        "beta1": 0.9,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "ce_weight": 1.0,
        "focal_weight": 0.1,
        "use_personality": True,
        "personality_dim": 1024,
        "bilstm_hidden_dim": 4,
        "transformer_hidden_dim": 8,
        "transformer_heads": 2,
        "transformer_layers": 1,
        "transformer_ffn_dim": 16,
        "lmf_rank": 2,
        "fusion_dim": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def sample_batch(batch_size=2, time_steps=4):
    audio = torch.randn(batch_size, time_steps, 8)
    video = torch.randn(batch_size, time_steps, 10)
    audio[:, -1] = 0
    video[:, -1] = 0
    return {
        "A_feat": audio,
        "V_feat": video,
        "personalized_feat": torch.randn(batch_size, 1024),
        "emo_label": torch.tensor([0, 1][:batch_size], dtype=torch.long),
    }
