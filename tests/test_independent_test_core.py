import tempfile
import unittest
from pathlib import Path

from experiments.run_model_cv import RAW_COLUMNS


CONDITION = {
    "DatasetYear": "2025",
    "Cohort": "Elder",
    "Model": "mlp",
    "Track": "Track1",
    "Task": "binary",
    "AudioFeature": "mfccs",
    "VideoFeature": "densenet",
    "UsePersonality": True,
    "SplitWindow": "1s",
    "Seed": 3407,
}


def make_row(*, fold, run_id="mlp-example", **overrides):
    values = {
        "Run_ID": run_id,
        "DatasetYear": "2025",
        "Cohort": "Elder",
        "Model": "mlp",
        "Fold": str(fold),
        "Track": "Track1",
        "Task": "binary",
        "AudioFeature": "mfccs",
        "VideoFeature": "densenet",
        "UsePersonality": "TRUE",
        "SplitWindow": "1s",
        "Device": "cuda",
        "Accuracy": "0.0",
        "Macro_F1": "0.0",
        "Weighted_F1": "0.0",
        "Precision": "0.0",
        "Recall": "0.0",
        "Positive_Recall": "0.0",
        "Specificity": "0.0",
        "ROC_AUC": "0.0",
        "Parameters": "0",
        "Model_Size_MB": "0.0",
        "Inference_Latency_ms": "0.0",
        "Peak_VRAM_MB": "0.0",
        "Loss_Function": "CE",
        "Class_Balancing_Strategy": "None",
        "Seed": "3407",
        "Status": "PASS",
    }
    values.update(overrides)
    return {column: values[column] for column in RAW_COLUMNS}


def metadata_payload(**overrides):
    payload = {
        "model_name": "mlp",
        "fold": 1,
        "seed": 3407,
        "feature_config": {
            "dataset_year": "2025",
            "cohort": "Elder",
            "track": "Track1",
            "task": "binary",
            "audio_feature": "mfccs",
            "video_feature": "densenet",
            "use_personality": True,
            "split_window": "1s",
            "device": "cuda:0",
        },
    }
    for key, value in overrides.items():
        if key in payload["feature_config"]:
            payload["feature_config"][key] = value
        else:
            payload[key] = value
    return payload


EXPECTED_METADATA = {
    "model_name": "mlp",
    "fold": 1,
    "seed": 3407,
    "dataset_year": "2025",
    "cohort": "Elder",
    "track": "Track1",
    "task": "binary",
    "audio_feature": "mfccs",
    "video_feature": "densenet",
    "use_personality": True,
    "split_window": "1s",
    "device": "cuda",
}


class SelectCvRunTests(unittest.TestCase):
    def test_selects_one_exact_run_with_csv_scalar_normalization(self):
        from experiments.independent_test import select_cv_run

        rows = [make_row(fold=fold) for fold in range(1, 6)]

        run_id, fold_rows = select_cv_run(rows, CONDITION, folds=5)

        self.assertEqual(run_id, "mlp-example")
        self.assertEqual([int(row["Fold"]) for row in fold_rows], [1, 2, 3, 4, 5])

    def test_rejects_multiple_matching_run_ids(self):
        from experiments.independent_test import select_cv_run

        rows = [make_row(fold=fold, run_id="run-a") for fold in range(1, 6)]
        rows.extend(make_row(fold=fold, run_id="run-b") for fold in range(1, 6))

        with self.assertRaisesRegex(ValueError, "Expected one CV Run_ID.*candidates=.*run-a.*run-b"):
            select_cv_run(rows, CONDITION)

    def test_rejects_missing_fold(self):
        from experiments.independent_test import select_cv_run

        rows = [make_row(fold=fold) for fold in range(1, 5)]

        with self.assertRaisesRegex(ValueError, "Incomplete or duplicate folds.*\[1, 2, 3, 4\]"):
            select_cv_run(rows, CONDITION)

    def test_rejects_duplicate_fold_rows(self):
        from experiments.independent_test import select_cv_run

        rows = [make_row(fold=fold) for fold in range(1, 6)]
        rows.append(make_row(fold=5))

        with self.assertRaisesRegex(ValueError, "Incomplete or duplicate folds"):
            select_cv_run(rows, CONDITION)

    def test_rejects_rows_that_did_not_pass(self):
        from experiments.independent_test import select_cv_run

        rows = [make_row(fold=fold, Status="FAILED") for fold in range(1, 6)]

        with self.assertRaisesRegex(ValueError, "Expected one CV Run_ID"):
            select_cv_run(rows, CONDITION)

    def test_rejects_unknown_boolean_spelling(self):
        from experiments.independent_test import select_cv_run

        rows = [make_row(fold=fold, UsePersonality="yes") for fold in range(1, 6)]

        with self.assertRaisesRegex(ValueError, "Expected one CV Run_ID"):
            select_cv_run(rows, CONDITION)

    def test_rejects_a_model_outside_the_approved_comparison_set(self):
        from experiments.independent_test import select_cv_run

        condition = {**CONDITION, "Model": "proposed"}
        rows = [make_row(fold=fold, Model="proposed") for fold in range(1, 6)]

        with self.assertRaisesRegex(ValueError, "Unsupported independent-test model"):
            select_cv_run(rows, condition)

    def test_rejects_an_incomplete_selection_condition(self):
        from experiments.independent_test import select_cv_run

        rows = [make_row(fold=fold) for fold in range(1, 6)]
        condition = {key: value for key, value in CONDITION.items() if key != "Seed"}

        with self.assertRaisesRegex(ValueError, "Missing required CV selection condition"):
            select_cv_run(rows, condition)

    def test_false_boolean_is_normalized_without_truthiness(self):
        from experiments.independent_test import select_cv_run

        rows = [make_row(fold=fold, UsePersonality="false") for fold in range(1, 6)]
        condition = {**CONDITION, "UsePersonality": False}

        run_id, _ = select_cv_run(rows, condition)

        self.assertEqual(run_id, "mlp-example")

    def test_rejects_each_different_provenance_condition(self):
        from experiments.independent_test import select_cv_run

        changed_columns = {
            "DatasetYear": "2026",
            "Cohort": "Young",
            "Task": "ternary",
            "AudioFeature": "wav2vec",
            "VideoFeature": "resnet",
            "Seed": "99",
        }
        for column, value in changed_columns.items():
            with self.subTest(column=column):
                rows = [make_row(fold=fold, **{column: value}) for fold in range(1, 6)]
                with self.assertRaisesRegex(ValueError, "Expected one CV Run_ID"):
                    select_cv_run(rows, CONDITION)


class CheckpointProvenanceTests(unittest.TestCase):
    def test_resolves_exactly_one_checkpoint_for_each_fold(self):
        from experiments.independent_test import resolve_checkpoint_paths

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            for fold in range(1, 6):
                checkpoint = results_dir / "raw" / "mlp-example" / "mlp" / f"fold_{fold}" / "checkpoint.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()

            paths = resolve_checkpoint_paths(results_dir, "mlp-example", "mlp", folds=5)

            self.assertEqual([path.name for path in paths], ["checkpoint.pth"] * 5)
            self.assertEqual([path.parent.name for path in paths], [f"fold_{fold}" for fold in range(1, 6)])

    def test_rejects_a_missing_checkpoint_fold(self):
        from experiments.independent_test import resolve_checkpoint_paths

        with tempfile.TemporaryDirectory() as directory:
            results_dir = Path(directory)
            for fold in range(1, 5):
                checkpoint = results_dir / "raw" / "mlp-example" / "mlp" / f"fold_{fold}" / "checkpoint.pth"
                checkpoint.parent.mkdir(parents=True)
                checkpoint.touch()

            with self.assertRaisesRegex(FileNotFoundError, "fold_5"):
                resolve_checkpoint_paths(results_dir, "mlp-example", "mlp", folds=5)

    def test_accepts_matching_metadata_and_device_family(self):
        from experiments.independent_test import validate_checkpoint_metadata

        validate_checkpoint_metadata(metadata_payload(), EXPECTED_METADATA)

    def test_reports_all_metadata_mismatches_before_inference(self):
        from experiments.independent_test import validate_checkpoint_metadata

        payload = metadata_payload(model_name="bilstm", seed=9, cohort="Young", device="cpu:svm")

        with self.assertRaises(ValueError) as context:
            validate_checkpoint_metadata(payload, EXPECTED_METADATA)

        message = str(context.exception)
        self.assertIn("model_name", message)
        self.assertIn("seed", message)
        self.assertIn("cohort", message)
        self.assertIn("device", message)


if __name__ == "__main__":
    unittest.main()
