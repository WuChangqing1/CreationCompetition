import importlib.util
import csv
import tempfile
import unittest
from pathlib import Path
import subprocess
import sys


class OrchestrationTest(unittest.TestCase):
    def test_run_all_file_entrypoint_can_import_project_package(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "experiments" / "run_all.py"), "--help"],
            cwd=root, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_orchestration_modules_exist(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.run_all"))
        self.assertIsNotNone(importlib.util.find_spec("experiments.aggregate_results"))

    def test_failed_model_does_not_stop_following_models(self):
        from experiments.run_all import run_models

        def runner(name):
            if name == "bad":
                raise RuntimeError("boom")
            return None

        with tempfile.TemporaryDirectory() as directory:
            report = run_models(["bad", "good"], runner, Path(directory) / "report.json")
        self.assertEqual([row["status"] for row in report], ["FAILED", "PASS"])
        self.assertIn("boom", report[0]["error"])

    def test_aggregate_formats_mean_and_sample_std_without_inventing_data(self):
        from experiments.aggregate_results import aggregate_results

        summary = aggregate_results([
            {"Model": "mlp", "Accuracy": 0.5, "Macro_F1": 0.4, "Status": "PASS"},
            {"Model": "mlp", "Accuracy": 0.7, "Macro_F1": 0.6, "Status": "PASS"},
            {"Model": "unavailable", "Accuracy": "N/A", "Macro_F1": "N/A", "Status": "SKIPPED"},
        ])
        self.assertEqual(summary[0]["Accuracy"], "0.6000 ± 0.1414")
        self.assertEqual(summary[1]["Accuracy"], "SKIPPED")

    def test_aggregate_keeps_experiment_conditions_separate(self):
        from experiments.aggregate_results import aggregate_results

        rows = [
            {"Run_ID": "a", "Model": "mlp", "Seed": 1, "AudioFeature": "mfccs", "Accuracy": 0.5, "Status": "PASS"},
            {"Run_ID": "a", "Model": "mlp", "Seed": 1, "AudioFeature": "mfccs", "Accuracy": 0.7, "Status": "PASS"},
            {"Run_ID": "b", "Model": "mlp", "Seed": 2, "AudioFeature": "wav2vec", "Accuracy": 0.9, "Status": "PASS"},
        ]
        summary = aggregate_results(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["Folds"], 2)
        self.assertEqual(summary[1]["Folds"], 1)

    def test_raw_result_retry_replaces_same_run_and_fold(self):
        from experiments.run_model_cv import append_raw_result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.csv"
            base = {"Run_ID": "abc", "Model": "mlp", "Fold": 1, "Accuracy": 0.5}
            append_raw_result(path, base)
            append_raw_result(path, {**base, "Accuracy": 0.9})
            with path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["Accuracy"], "0.9")


if __name__ == "__main__":
    unittest.main()
