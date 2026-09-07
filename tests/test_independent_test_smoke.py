import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.run_independent_test import main
from experiments.run_model_cv import RAW_COLUMNS


class IndependentTestSmoke(unittest.TestCase):
    def test_batch_failure_rerun_and_cross_year_preservation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cv_path = root / "cv.csv"
            output = root / "independent.csv"
            sentinels = [root / "raw_results.csv", root / "summary.csv", root / "submission.csv"]
            for path in sentinels:
                path.write_bytes((path.name + " unchanged").encode())
            before = [path.read_bytes() for path in sentinels]

            entries = [
                {"subject_id": "A", "bin_category": 0},
                {"subject_id": "A", "bin_category": 0},
                {"subject_id": "B", "bin_category": 1},
            ]
            paths = {"entries": entries, "audio": root, "video": root, "personality": root / "p.npy"}
            probabilities = np.array([[0.9, 0.1], [0.7, 0.3], [0.1, 0.9]])

            def write_cv(year):
                with cv_path.open("w", encoding="utf-8-sig", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
                    writer.writeheader()
                    for model in ("svm", "mlp"):
                        for fold in range(1, 6):
                            row = {column: "N/A" for column in RAW_COLUMNS}
                            row.update({
                                "Run_ID": f"{model}-{year}", "DatasetYear": year, "Cohort": "Elder",
                                "Model": model, "Fold": fold, "Track": "Track1", "Task": "binary",
                                "AudioFeature": "mfccs", "VideoFeature": "densenet", "UsePersonality": True,
                                "SplitWindow": "1s", "Seed": 3407, "Status": "PASS",
                            })
                            writer.writerow(row)

            def run(year, classical_effect=None):
                write_cv(year)
                argv = [
                    "--models", "svm,mlp", "--dataset-year", year, "--data-root", str(root),
                    "--cv-results", str(cv_path), "--output", str(output),
                    "--results-dir", str(root / "results"), "--splits-dir", str(root / "splits"),
                ]
                with patch("experiments.run_independent_test.resolve_dataset", return_value=paths), \
                     patch("experiments.run_independent_test.resolve_independent_test", return_value=paths), \
                     patch("experiments.run_independent_test.load_saved_fold_records", return_value=[{}] * 5), \
                     patch("experiments.run_independent_test._load_matching_saved_config", return_value={}), \
                     patch("experiments.run_independent_test._configure_torch_feature_length"), \
                     patch("experiments.run_independent_test.resolve_checkpoint_paths", return_value=[root / "c"] * 5), \
                     patch("experiments.run_independent_test.infer_classical_fold_ensemble", side_effect=classical_effect or [(np.array([0, 0, 1]), probabilities)]), \
                     patch("experiments.run_independent_test.infer_torch_fold_ensemble", return_value=probabilities):
                    return main(argv)

            self.assertEqual(run("2025"), 0)
            self.assertEqual(run("2025"), 0)
            self.assertEqual(run("2026"), 0)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 8)
            self.assertEqual({(row["DatasetYear"], row["Model"], row["EvaluationLevel"]) for row in rows}, {
                (year, model, level) for year in ("2025", "2026") for model in ("svm", "mlp") for level in ("event", "subject")
            })
            event_prediction = root / "independent_test_predictions" / "2025_Elder_svm_event.csv"
            subject_prediction = root / "independent_test_predictions" / "2025_Elder_svm_subject.csv"
            with event_prediction.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual([row["subject_id"] for row in csv.DictReader(handle)], ["A", "A", "B"])
            with subject_prediction.open(encoding="utf-8-sig", newline="") as handle:
                self.assertEqual([row["subject_id"] for row in csv.DictReader(handle)], ["A", "B"])
            confusion = root / "independent_test_confusion_matrix" / "2025_Elder_svm_event.csv"
            with confusion.open(encoding="utf-8-sig", newline="") as handle:
                cells = list(csv.DictReader(handle))
            self.assertEqual(len(cells), 4)
            self.assertEqual([path.read_bytes() for path in sentinels], before)

            write_cv("2025")
            argv = [
                "--models", "svm,mlp", "--dataset-year", "2025", "--data-root", str(root),
                "--cv-results", str(cv_path), "--output", str(output),
                "--results-dir", str(root / "results"), "--splits-dir", str(root / "splits"),
            ]
            with patch("experiments.run_independent_test.resolve_dataset", return_value=paths), \
                 patch("experiments.run_independent_test.resolve_independent_test", return_value=paths), \
                 patch("experiments.run_independent_test.load_saved_fold_records", return_value=[{}] * 5), \
                 patch("experiments.run_independent_test._load_matching_saved_config", return_value={}), \
                 patch("experiments.run_independent_test.infer_classical_fold_ensemble", side_effect=RuntimeError("broken estimator")), \
                 patch("experiments.run_independent_test._configure_torch_feature_length"), \
                 patch("experiments.run_independent_test.resolve_checkpoint_paths", return_value=[root / "c"] * 5), \
                 patch("experiments.run_independent_test.infer_torch_fold_ensemble", return_value=probabilities):
                self.assertEqual(main(argv), 1)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                failed_rows = [row for row in csv.DictReader(handle) if row["DatasetYear"] == "2025" and row["Model"] == "svm"]
            self.assertTrue(all(row["Status"].startswith("FAILED: broken estimator") for row in failed_rows))
            self.assertTrue(all(row["Accuracy"] == "N/A" for row in failed_rows))


if __name__ == "__main__":
    unittest.main()
