"""Feature preparation shared by classical baselines."""

import numpy as np


def pool_multimodal_features(audio, video, personality=None, use_personality=True):
    audio = np.asarray(audio)
    video = np.asarray(video)
    if audio.ndim != 3 or video.ndim != 3:
        raise ValueError("audio and video must have shape [samples, time, features]")
    if audio.shape[0] != video.shape[0]:
        raise ValueError("audio and video sample counts must match")
    parts = [audio.mean(axis=1), video.mean(axis=1)]
    if use_personality:
        if personality is None:
            raise ValueError("personality is required when use_personality=True")
        personality = np.asarray(personality)
        if personality.ndim != 2 or personality.shape[0] != audio.shape[0]:
            raise ValueError("personality must have shape [samples, features]")
        parts.append(personality)
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)
