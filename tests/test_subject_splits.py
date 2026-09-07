import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys


def balanced_entries():
    rows = []
    for label in (0, 1):
        for index in range(5):
            subject = f"{label}{index}"
            for segment in range(2):
                rows.append({
                    "id": subject,
                    "audio_feature_path": f"{subject}_{segment}.npy",
                    "bin_category": label,
                })
    return rows


class SubjectSplitsTest(unittest.TestCase):
    def test_create_splits_file_entrypoint_can_import_project_package(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "experiments" / "create_splits.py"), "--help"],
            cwd=root, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_split_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.create_splits"))

    def test_leakage_is_rejected(self):
        from experiments.create_splits import validate_no_subject_leakage

        with self.assertRaisesRegex(ValueError, "Subject leakage detected"):
            validate_no_subject_leakage(["1", "2"], ["2", "3"])

    def test_each_subject_appears_in_exactly_one_validation_fold(self):
        from experiments.create_splits import create_subject_folds

        folds = create_subject_folds(balanced_entries(), "bin_category", 5, 3407)
        validation_ids = [subject for fold in folds for subject in fold["val_ids"]]
        self.assertEqual(len(validation_ids), 10)
        self.assertEqual(len(validation_ids), len(set(validation_ids)))
        for fold in folds:
            self.assertFalse(set(fold["train_ids"]) & set(fold["val_ids"]))
            self.assertEqual(fold["val_class_distribution"], {"0": 1, "1": 1})

    def test_conflicting_labels_for_one_subject_are_rejected(self):
        from experiments.data_utils import build_subject_labels

        rows = [
            {"id": "1", "audio_feature_path": "1_a.npy", "bin_category": 0},
            {"id": "1", "audio_feature_path": "1_b.npy", "bin_category": 1},
        ]
        with self.assertRaisesRegex(ValueError, "conflicting labels"):
            build_subject_labels(rows, "bin_category")

    def test_existing_splits_are_reused_without_force(self):
        from experiments.create_splits import create_subject_folds, write_subject_folds

        folds = create_subject_folds(balanced_entries(), "bin_category", 5, 3407)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            first = write_subject_folds(folds, target, force=False)
            path = target / "fold_1.json"
            saved = json.loads(path.read_text(encoding="utf-8"))
            saved["sentinel"] = "keep"
            path.write_text(json.dumps(saved), encoding="utf-8")
            second = write_subject_folds(folds, target, force=False)
            self.assertEqual(first, second)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["sentinel"], "keep")

    def test_incompatible_split_cache_is_rejected_without_force(self):
        from experiments.create_splits import create_subject_folds, write_subject_folds

        first = create_subject_folds(balanced_entries(), "bin_category", 5, 3407)
        second = create_subject_folds(balanced_entries(), "bin_category", 5, 99)
        with tempfile.TemporaryDirectory() as directory:
            write_subject_folds(first, directory)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                write_subject_folds(second, directory)

    def test_partial_split_cache_is_rejected_without_force(self):
        from experiments.create_splits import create_subject_folds, write_subject_folds

        folds = create_subject_folds(balanced_entries(), "bin_category", 5, 3407)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            target.mkdir(exist_ok=True)
            (target / "fold_1.json").write_text(json.dumps(folds[0]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partial"):
                write_subject_folds(folds, target)

    def test_audio_filename_precedes_generic_clip_id(self):
        from experiments.data_utils import extract_subject_id

        self.assertEqual(
            extract_subject_id({"id": "clip99", "audio_feature_path": "12_0.npy"}),
            "12",
        )


if __name__ == "__main__":
    unittest.main()
