import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.independent_test import RESULT_COLUMNS
from experiments.run_independent_test import main, parse_args
from experiments.run_model_cv import RAW_COLUMNS


MODELS = "svm,xgboost,mlp,bilstm,lightweighttrans,lmf,mult,our"


def _raw_row(model, run_id):
    values = {column: "N/A" for column in RAW_COLUMNS}
    values.update({
        "Run_ID": run_id, "DatasetYear": "2025", "Cohort": "Elder",
        "Model": model, "Track": "Track1", "Task": "binary",
        "AudioFeature": "mfccs", "VideoFeature": "densenet",
        "UsePersonality": "True", "SplitWindow": "1s", "Seed": "3407",
        "Status": "PASS",
    })
    return values


class ArgumentTests(unittest.TestCase):
    def test_parses_the_approved_model_list_and_default_output(self):
        args = parse_args(["--models", MODELS, "--dataset-year", "2025", "--data-root", "data"])
        self.assertEqual(args.models, MODELS.split(","))
        self.assertEqual(args.output.name, "independent_test_results.csv")
        self.assertEqual(args.feature_max_len, 5)

    def test_rejects_unknown_duplicate_non_five_fold_and_cpu_torch_requests(self):
        cases = (
            ["--models", "unknown"], ["--models", "svm,svm"],
            ["--models", "svm", "--folds", "4"], ["--models", "mlp", "--device", "cpu"],
        )
        for extra in cases:
            with self.subTest(extra=extra), self.assertRaises(SystemExit):
                parse_args(extra + ["--dataset-year", "2025", "--data-root", "data"])

    def test_rejects_every_protected_output_alias(self):
        for name in ("raw_results.csv", "summary.csv", "submission.csv"):
            with self.subTest(name=name), self.assertRaises(SystemExit):
                parse_args([
                    "--models", "svm", "--dataset-year", "2025", "--data-root", "data",
                    "--output", str(Path("elsewhere") / ".." / name),
                ])


class OrchestrationTests(unittest.TestCase):
    def _invoke(self, root, models="svm", inference_effect=None):
        cv_path = root / "raw_results.csv"
        with cv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RAW_COLUMNS)
            writer.writeheader()
            for model in models.split(","):
                for fold in range(1, 6):
                    writer.writerow({**_raw_row(model, f"{model}-run"), "Fold": fold})
        output = root / "independent.csv"
        entries = [
            {"subject_id": "A", "bin_category": 0},
            {"subject_id": "A", "bin_category": 0},
            {"subject_id": "B", "bin_category": 1},
        ]
        paths = {"entries": entries, "audio": root, "video": root, "personality": root / "p.npy"}
        argv = [
            "--models", models, "--dataset-year", "2025", "--data-root", str(root),
            "--cv-results", str(cv_path), "--output", str(output),
            "--results-dir", str(root / "results"), "--splits-dir", str(root / "splits"),
        ]
        probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
        with patch("experiments.run_independent_test.resolve_dataset", return_value=paths), \
             patch("experiments.run_independent_test.resolve_independent_test", return_value=paths), \
             patch("experiments.run_independent_test.load_saved_fold_records", return_value=[{}] * 5), \
             patch("experiments.run_independent_test._load_matching_saved_config", return_value={}), \
             patch(
                 "experiments.run_independent_test.infer_classical_fold_ensemble",
                 side_effect=inference_effect,
                 return_value=(np.array([9, 9, 9]), probabilities),
             ):
            code = main(argv)
        return code, output

    def test_success_writes_two_levels_and_unique_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code, output = self._invoke(root)
            self.assertEqual(code, 0)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual({row["EvaluationLevel"] for row in rows}, {"event", "subject"})
            self.assertTrue(all(row["Run_ID"] == "svm-run" and row["EnsembleFolds"] == "5" for row in rows))
            self.assertTrue(all(row["Device"] == "cpu:svm" and row["Seed"] == "3407" for row in rows))
            for level in ("event", "subject"):
                self.assertTrue((root / "independent_test_predictions" / f"2025_Elder_svm_{level}.csv").is_file())
                self.assertTrue((root / "independent_test_confusion_matrix" / f"2025_Elder_svm_{level}.csv").is_file())

    def test_failure_continues_and_returns_nonzero_without_fake_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            probabilities = np.array([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9]])
            calls = [RuntimeError("broken estimator"), (np.array([0, 0, 1]), probabilities)]
            code, output = self._invoke(root, models="svm,xgboost", inference_effect=calls)
            self.assertEqual(code, 1)
            with output.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            svm_rows = [row for row in rows if row["Model"] == "svm"]
            xgb_rows = [row for row in rows if row["Model"] == "xgboost"]
            self.assertEqual(len(svm_rows), 2)
            self.assertTrue(all(row["Status"].startswith("FAILED: broken estimator") for row in svm_rows))
            self.assertTrue(all(row["Accuracy"] == "N/A" for row in svm_rows))
            self.assertEqual(len(xgb_rows), 2)
            self.assertTrue(all(row["Status"] == "PASS" for row in xgb_rows))


if __name__ == "__main__":
    unittest.main()
