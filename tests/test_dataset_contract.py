import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset import AudioVisualDataset


class DatasetContractTest(unittest.TestCase):
    def test_personality_lookup_preserves_subject_leading_zeroes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio_dir = root / "audio"
            video_dir = root / "video"
            audio_dir.mkdir()
            video_dir.mkdir()
            np.save(audio_dir / "001_0.npy", np.ones((2, 3), dtype=np.float32))
            np.save(video_dir / "001_0.npy", np.ones((2, 4), dtype=np.float32))
            personality_path = root / "personality.npy"
            expected = np.arange(5, dtype=np.float32)
            np.save(personality_path, np.array([{"id": "001", "embedding": expected}], dtype=object))
            entries = [{
                "audio_feature_path": "001_0.npy",
                "video_feature_path": "001_0.npy",
                "bin_category": 1,
            }]
            dataset = AudioVisualDataset(entries, 2, str(personality_path), 2, audio_path=str(audio_dir), video_path=str(video_dir))
            np.testing.assert_allclose(dataset[0]["personalized_feat"].numpy(), expected)


if __name__ == "__main__":
    unittest.main()
