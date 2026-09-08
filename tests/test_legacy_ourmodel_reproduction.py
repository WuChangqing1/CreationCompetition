import argparse
import tempfile
import unittest
from pathlib import Path


class LegacyOurModelReproductionTests(unittest.TestCase):
    def test_legacy_test_can_write_to_an_explicit_output_directory(self):
        from test import resolve_output_dir

        requested = Path("isolated-results")

        self.assertEqual(resolve_output_dir("Track1", requested), requested)

    def test_metrics_payload_preserves_the_historical_confusion_matrix(self):
        from test import build_metrics_payload

        payload = build_metrics_payload(
            y_true=[0] * 187 + [1] * 40,
            y_pred=[0] * 187 + [0] * 14 + [1] * 26,
        )

        self.assertEqual(payload["sample_count"], 227)
        self.assertEqual(payload["confusion_matrix"], [[187, 0], [14, 26]])
        self.assertAlmostEqual(payload["accuracy"], 213 / 227)
        self.assertAlmostEqual(payload["balanced_accuracy"], 0.825)
        self.assertAlmostEqual(payload["macro_f1"], 0.875898156825992)

    def test_metrics_payload_preserves_non_binary_legacy_tasks(self):
        from test import build_metrics_payload

        payload = build_metrics_payload(
            y_true=[0, 1, 2],
            y_pred=[0, 1, 2],
            label_count=3,
        )

        self.assertEqual(
            payload["confusion_matrix"],
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )

    def test_command_uses_exactly_one_historical_checkpoint_and_no_fold_arguments(self):
        from experiments.run_legacy_ourmodel import build_test_command

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = argparse.Namespace(
                data_root=root / "MPDD-Test" / "MPDD-Elderly",
                checkpoint=root / "best_model.pth",
                output_dir=root / "results",
                device="cuda",
            )

            command, metrics_path = build_test_command(args, python_executable="python")

        self.assertEqual(command.count("--train_model"), 1)
        self.assertNotIn("--folds", command)
        self.assertNotIn("--fold", command)
        self.assertEqual(command[command.index("--feature_max_len") + 1], "26")
        self.assertEqual(command[command.index("--batch_size") + 1], "8")
        self.assertEqual(command[command.index("--lr") + 1], "2e-05")
        self.assertEqual(command[command.index("--audiofeature_method") + 1], "mfccs")
        self.assertEqual(command[command.index("--videofeature_method") + 1], "densenet")
        self.assertEqual(command[command.index("--model") + 1], "our")
        self.assertEqual(metrics_path.name, "metrics.json")

    def test_historical_result_validator_accepts_expected_metrics(self):
        from experiments.run_legacy_ourmodel import validate_historical_result

        metrics = {
            "accuracy": 213 / 227,
            "macro_f1": 0.875898156825992,
            "confusion_matrix": [[187, 0], [14, 26]],
        }

        validate_historical_result(metrics)

    def test_historical_result_validator_rejects_a_different_confusion_matrix(self):
        from experiments.run_legacy_ourmodel import validate_historical_result

        metrics = {
            "accuracy": 212 / 227,
            "macro_f1": 0.86,
            "confusion_matrix": [[186, 1], [14, 26]],
        }

        with self.assertRaisesRegex(RuntimeError, "历史结果未复现"):
            validate_historical_result(metrics)


if __name__ == "__main__":
    unittest.main()
