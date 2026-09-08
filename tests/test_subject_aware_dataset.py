import tempfile
import unittest
from pathlib import Path

import numpy as np

from dataset import AudioVisualDataset
from experiments.subject_aware_dataset import create_audio_visual_dataset
from experiments.run_model_cv import build_run_id, parse_args


class SubjectAwareDatasetTest(unittest.TestCase):
    def _fixture(self, root):
        audio_root = root / "audio"
        video_root = root / "video"
        (audio_root / "2").mkdir(parents=True)
        (video_root / "2").mkdir(parents=True)
        np.save(audio_root / "2" / "A_1.npy", np.ones((2, 3), dtype=np.float32))
        np.save(video_root / "2" / "V_1.npy", np.ones((2, 4), dtype=np.float32))
        expected = np.arange(5, dtype=np.float32)
        personality_path = root / "personality.npy"
        np.save(
            personality_path,
            np.array([{"id": "2", "embedding": expected}], dtype=object),
        )
        entries = [{
            "subject_id": "2",
            "audio_feature_path": "2/A_1.npy",
            "video_feature_path": "2/V_1.npy",
            "bin_category": 1,
        }]
        return entries, audio_root, video_root, personality_path, expected

    def test_subject_id_mode_fixes_2026_personality_lookup_without_changing_legacy_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            entries, audio_root, video_root, personality_path, expected = self._fixture(Path(directory))
            legacy = AudioVisualDataset(
                entries, 2, str(personality_path), 2,
                audio_path=str(audio_root), video_path=str(video_root),
            )
            corrected = create_audio_visual_dataset(
                entries, 2, str(personality_path), 2,
                audio_path=str(audio_root), video_path=str(video_root),
                personality_id_source="subject_id",
            )

            np.testing.assert_allclose(legacy[0]["personalized_feat"].numpy(), np.zeros(1024))
            np.testing.assert_allclose(corrected[0]["personalized_feat"].numpy(), expected)

    def test_subject_id_mode_rejects_missing_personality_instead_of_using_zeroes(self):
        with tempfile.TemporaryDirectory() as directory:
            entries, audio_root, video_root, personality_path, _ = self._fixture(Path(directory))
            entries[0]["subject_id"] = "missing"
            corrected = create_audio_visual_dataset(
                entries, 2, str(personality_path), 2,
                audio_path=str(audio_root), video_path=str(video_root),
                personality_id_source="subject_id",
            )

            with self.assertRaisesRegex(KeyError, "missing"):
                corrected[0]

    def test_personality_id_source_changes_cv_run_identity(self):
        legacy_args, config = parse_args(["--model", "mlp"])
        corrected_args, _ = parse_args([
            "--model", "mlp", "--personality-id-source", "subject_id",
        ])

        self.assertNotEqual(build_run_id(legacy_args, config), build_run_id(corrected_args, config))


if __name__ == "__main__":
    unittest.main()
