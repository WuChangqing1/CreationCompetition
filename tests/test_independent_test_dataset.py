import csv
import json
import tempfile
import unittest
from pathlib import Path


class IndependentTestDatasetTest(unittest.TestCase):
    def _write_2025_layout(self, root, manifest_rows=None, labels=None):
        track = root / "MPDD-Test" / "MPDD-Elderly"
        audio = track / "1s" / "Audio" / "mfccs"
        video = track / "1s" / "Visual" / "densenet"
        audio.mkdir(parents=True)
        video.mkdir(parents=True)
        (track / "individualEmbedding").mkdir()
        (track / "individualEmbedding" / "descriptions_embeddings_with_ids.npy").touch()
        (track / "labels").mkdir()
        (audio / "1178_A_1.npy").touch()
        (video / "1178_V_1.npy").touch()
        rows = manifest_rows or [{
            "audio_feature_path": "1178_A_1.npy",
            "video_feature_path": "1178_V_1.npy",
        }]
        (track / "labels" / "Testing_files.json").write_text(
            json.dumps(rows), encoding="utf-8",
        )
        (root / "MPDD-Test" / "MM2025_Track1_Elderly.json").write_text(
            json.dumps(labels if labels is not None else [{"test_id": "1178", "label_bin": 1}]),
            encoding="utf-8",
        )
        return track, audio, video

    def test_2025_resolves_manifest_labels_and_feature_roots(self):
        from experiments.dataset_layouts import resolve_independent_test

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track, audio_root, video_root = self._write_2025_layout(root)

            paths = resolve_independent_test(root, "2025", "Elder", "1s", "mfccs", "densenet")

            self.assertEqual(paths["entries"][0]["subject_id"], "1178")
            self.assertEqual(paths["entries"][0]["bin_category"], 1)
            self.assertEqual(paths["audio"], audio_root)
            self.assertEqual(paths["video"], video_root)
            self.assertEqual(paths["track_root"], track)

    def test_2025_rejects_manifest_subject_without_ground_truth(self):
        from experiments.dataset_layouts import resolve_independent_test

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_2025_layout(root, labels=[])

            with self.assertRaisesRegex(KeyError, "1178"):
                resolve_independent_test(root, "2025", "Elder", "1s", "mfccs", "densenet")

    def test_2025_rejects_duplicate_ground_truth_subject(self):
        from experiments.dataset_layouts import resolve_independent_test

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_2025_layout(root, labels=[
                {"test_id": "1178", "label_bin": 1},
                {"test_id": "1178", "label_bin": 0},
            ])

            with self.assertRaisesRegex(ValueError, "1178"):
                resolve_independent_test(root, "2025", "Elder", "1s", "mfccs", "densenet")

    def test_2025_rejects_manifest_av_subject_mismatch(self):
        from experiments.dataset_layouts import resolve_independent_test

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_2025_layout(root, manifest_rows=[{
                "audio_feature_path": "1178_A_1.npy",
                "video_feature_path": "9999_V_1.npy",
            }])

            with self.assertRaisesRegex(ValueError, "1178_A_1.npy.*9999_V_1.npy"):
                resolve_independent_test(root, "2025", "Elder", "1s", "mfccs", "densenet")

    def _write_2026_layout(self, root, audio_events=(1, 2), video_events=(1, 2)):
        track = root / "MPDD-AVG2026-test" / "Elder"
        audio = track / "Audio" / "mfcc" / "7"
        video = track / "Video" / "densenet" / "7"
        audio.mkdir(parents=True)
        video.mkdir(parents=True)
        for event in audio_events:
            (audio / f"A_{event}.npy").touch()
        for event in video_events:
            (video / f"V_{event}.npy").touch()
        with (track / "split_labels_test.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["ID", "label2", "label3"])
            writer.writeheader()
            writer.writerow({"ID": 7, "label2": 1, "label3": 2})
        (track / "descriptions_embeddings_with_ids.npy").touch()
        return track, audio.parent, video.parent

    def test_2026_resolves_alias_and_inherits_subject_binary_label(self):
        from experiments.dataset_layouts import resolve_independent_test

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track, audio_root, video_root = self._write_2026_layout(root)

            paths = resolve_independent_test(root, "2026", "Elder", "1s", "mfccs", "densenet")

            self.assertEqual(paths["audio"], audio_root)
            self.assertEqual(paths["video"], video_root)
            self.assertEqual(paths["track_root"], track)
            self.assertEqual([entry["bin_category"] for entry in paths["entries"]], [1, 1])
            self.assertEqual([entry["audio_feature_path"] for entry in paths["entries"]], ["7/A_1.npy", "7/A_2.npy"])

    def test_2026_rejects_unpaired_events_instead_of_skipping_them(self):
        from experiments.dataset_layouts import resolve_independent_test

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_2026_layout(root, audio_events=(1, 2), video_events=(1,))

            with self.assertRaisesRegex(ValueError, "2"):
                resolve_independent_test(root, "2026", "Elder", "1s", "mfccs", "densenet")


if __name__ == "__main__":
    unittest.main()
