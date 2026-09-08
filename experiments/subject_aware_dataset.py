"""Optional subject-aware personality lookup without changing the legacy Dataset."""

import numpy as np
import torch

from dataset import AudioVisualDataset
from experiments.data_utils import extract_subject_id


PERSONALITY_ID_SOURCES = ("filename", "subject_id")


class SubjectAwareAudioVisualDataset(AudioVisualDataset):
    """Use each manifest entry's subject ID for strict personality lookup."""

    def __init__(
        self, json_data, label_count, personalized_feature_file=None, max_len=10,
        batch_size=32, audio_path="", video_path="", isTest=False,
        use_personality=True, personality_dim=1024,
    ):
        super().__init__(
            json_data, label_count, None, max_len, batch_size=batch_size,
            audio_path=audio_path, video_path=video_path, isTest=isTest,
            use_personality=False, personality_dim=personality_dim,
        )
        self.subject_personality_enabled = bool(use_personality)
        if self.subject_personality_enabled:
            if not personalized_feature_file:
                raise ValueError("personality feature file is required in subject_id mode")
            loaded = self.load_personalized_features(personalized_feature_file)
            self.personalized_features = {str(key): value for key, value in loaded.items()}

    def __getitem__(self, index):
        item = super().__getitem__(index)
        if not self.subject_personality_enabled:
            return item
        subject_id = extract_subject_id(self.data[index])
        if subject_id not in self.personalized_features:
            raise KeyError(f"Missing personality embedding for subject {subject_id}")
        item["personalized_feat"] = torch.tensor(
            np.asarray(self.personalized_features[subject_id]), dtype=torch.float32
        )
        return item


def create_audio_visual_dataset(
    json_data, label_count, personalized_feature_file=None, max_len=10,
    batch_size=32, audio_path="", video_path="", isTest=False,
    use_personality=True, personality_dim=1024, personality_id_source="filename",
):
    """Create the legacy or strict subject-aware Dataset explicitly."""
    source = str(personality_id_source).strip().lower()
    if source not in PERSONALITY_ID_SOURCES:
        raise ValueError(f"Unknown personality ID source: {personality_id_source!r}")
    dataset_type = SubjectAwareAudioVisualDataset if source == "subject_id" else AudioVisualDataset
    return dataset_type(
        json_data, label_count, personalized_feature_file, max_len,
        batch_size=batch_size, audio_path=audio_path, video_path=video_path,
        isTest=isTest, use_personality=use_personality, personality_dim=personality_dim,
    )


__all__ = [
    "PERSONALITY_ID_SOURCES",
    "SubjectAwareAudioVisualDataset",
    "create_audio_visual_dataset",
]
