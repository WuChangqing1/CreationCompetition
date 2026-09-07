import importlib.util
import tempfile
import unittest
from pathlib import Path

import pandas as pd


class EvaluatorTest(unittest.TestCase):
    def test_evaluator_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.evaluator"))

    def test_binary_metrics_match_hand_checked_values(self):
        from experiments.evaluator import evaluate_predictions

        result = evaluate_predictions(
            [0, 0, 1, 1], [0, 1, 1, 1],
            [[0.9, 0.1], [0.4, 0.6], [0.2, 0.8], [0.1, 0.9]],
        )
        self.assertEqual(result["Accuracy"], 0.75)
        self.assertEqual(result["Positive_Recall"], 1.0)
        self.assertEqual(result["Specificity"], 0.5)
        self.assertEqual(result["Confusion_Matrix"], [[1, 1], [0, 2]])

    def test_single_class_auc_is_na(self):
        from experiments.evaluator import evaluate_predictions

        result = evaluate_predictions([0, 0], [0, 0], [[1, 0], [1, 0]])
        self.assertEqual(result["ROC_AUC"], "N/A")

    def test_prediction_csv_contains_reproducible_columns(self):
        from experiments.evaluator import save_predictions

        with tempfile.TemporaryDirectory() as directory:
            path = save_predictions(
                Path(directory) / "mlp_fold1.csv", ["1", "2"], [0, 1], [0, 1],
                [[0.8, 0.2], [0.1, 0.9]], 1, "mlp",
            )
            frame = pd.read_csv(path)
            self.assertEqual(
                list(frame.columns),
                ["subject_id", "true_label", "pred_label", "prob_0", "prob_1", "fold", "model"],
            )

    def test_oof_predictions_are_combined_for_overall_confusion(self):
        import experiments.evaluator as evaluator

        self.assertTrue(hasattr(evaluator, "save_overall_confusion"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = evaluator.save_predictions(
                root / "mlp_fold1.csv", ["1"], [0], [0], [[0.8, 0.2]], 1, "mlp"
            )
            second = evaluator.save_predictions(
                root / "mlp_fold2.csv", ["2"], [1], [1], [[0.1, 0.9]], 2, "mlp"
            )
            result = evaluator.save_overall_confusion([first, second], root / "mlp_overall.png", "mlp")
            self.assertEqual(result["metrics"]["Confusion_Matrix"], [[1, 0], [0, 1]])


if __name__ == "__main__":
    unittest.main()
