import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np


class DatasetLayoutsTest(unittest.TestCase):
    def test_2026_layout_pairs_events_and_normalizes_labels(self):
        from experiments.dataset_layouts import resolve_dataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = root / "MPDD-AVG2026-trainval" / "Elder"
            audio = track / "Audio" / "train" / "mfcc" / "7"
            video = track / "Video" / "train" / "densenet" / "7"
            audio.mkdir(parents=True)
            video.mkdir(parents=True)
            np.save(audio / "A_1.npy", np.ones((3, 64), dtype=np.float32))
            np.save(video / "V_1.npy", np.ones((4, 1000), dtype=np.float32))
            np.save(track / "descriptions_embeddings_with_ids.npy", np.array([
                {"id": "7", "embedding": np.ones(1024, dtype=np.float32)}
            ], dtype=object))
            with (track / "split_labels_train.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["ID", "split", "label2", "label3", "PHQ-9"])
                writer.writeheader()
                writer.writerow({"ID": 7, "split": "train", "label2": 1, "label3": 2, "PHQ-9": 12})

            layout = resolve_dataset(root, "2026", "Elder", "1s", "mfccs", "densenet")
            self.assertEqual(layout["entries"][0]["subject_id"], "7")
            self.assertEqual(layout["entries"][0]["audio_feature_path"], "7/A_1.npy")
            self.assertEqual(layout["entries"][0]["video_feature_path"], "7/V_1.npy")
            self.assertEqual(layout["entries"][0]["bin_category"], 1)

    def test_dataset_does_not_require_personality_file_when_disabled(self):
        from dataset import AudioVisualDataset

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / "a.npy", np.ones((2, 4), dtype=np.float32))
            np.save(root / "v.npy", np.ones((2, 5), dtype=np.float32))
            dataset = AudioVisualDataset(
                [{"audio_feature_path": "a.npy", "video_feature_path": "v.npy", "bin_category": 0}],
                2, None, audio_path=str(root), video_path=str(root), use_personality=False,
            )
            self.assertEqual(tuple(dataset[0]["personalized_feat"].shape), (1024,))


if __name__ == "__main__":
    unittest.main()
