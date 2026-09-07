import tempfile
import unittest
import csv
from pathlib import Path
from unittest.mock import patch

import numpy as np

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

    def test_rejects_malformed_boolean_values_on_either_or_both_sides(self):
        from experiments.independent_test import select_cv_run

        cases = (("yes", True), (True, "yes"), ("yes", "yes"))
        for row_value, condition_value in cases:
            with self.subTest(row_value=row_value, condition_value=condition_value):
                rows = [make_row(fold=fold, UsePersonality=row_value) for fold in range(1, 6)]
                condition = {**CONDITION, "UsePersonality": condition_value}
                with self.assertRaisesRegex(ValueError, "Expected one CV Run_ID"):
                    select_cv_run(rows, condition)

    def test_rejects_malformed_integer_values_on_either_or_both_sides(self):
        from experiments.independent_test import select_cv_run

        cases = (("not-a-seed", 3407), (3407, "not-a-seed"), ("not-a-seed", "not-a-seed"))
        for row_value, condition_value in cases:
            with self.subTest(row_value=row_value, condition_value=condition_value):
                rows = [make_row(fold=fold, Seed=row_value) for fold in range(1, 6)]
                condition = {**CONDITION, "Seed": condition_value}
                with self.assertRaisesRegex(ValueError, "Expected one CV Run_ID"):
                    select_cv_run(rows, condition)

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

    def test_rejects_non_five_fold_selection_requests(self):
        from experiments.independent_test import select_cv_run

        rows = [make_row(fold=fold) for fold in range(1, 6)]

        with self.assertRaisesRegex(ValueError, "requires exactly five folds"):
            select_cv_run(rows, CONDITION, folds=4)

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

    def test_rejects_non_five_fold_checkpoint_requests(self):
        from experiments.independent_test import resolve_checkpoint_paths

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires exactly five folds"):
                resolve_checkpoint_paths(Path(directory), "mlp-example", "mlp", folds=4)

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

    def test_rejects_malformed_metadata_scalars_on_either_or_both_sides(self):
        from experiments.independent_test import validate_checkpoint_metadata

        cases = (
            ("use_personality", "yes", True),
            ("use_personality", True, "yes"),
            ("use_personality", "yes", "yes"),
            ("seed", "not-a-seed", 3407),
            ("seed", 3407, "not-a-seed"),
            ("seed", "not-a-seed", "not-a-seed"),
        )
        for field, payload_value, expected_value in cases:
            with self.subTest(field=field, payload_value=payload_value, expected_value=expected_value):
                payload = metadata_payload(**{field: payload_value})
                expected = {**EXPECTED_METADATA, field: expected_value}
                with self.assertRaisesRegex(ValueError, field):
                    validate_checkpoint_metadata(payload, expected)


class AggregationTests(unittest.TestCase):
    def test_means_equal_sized_valid_probability_folds(self):
        from experiments.independent_test import mean_fold_probabilities

        mean = mean_fold_probabilities([
            np.array([[0.8, 0.2], [0.4, 0.6]]),
            np.array([[0.6, 0.4], [0.2, 0.8]]),
        ])

        np.testing.assert_allclose(mean, [[0.7, 0.3], [0.3, 0.7]])

    def test_rejects_malformed_probability_folds(self):
        from experiments.independent_test import mean_fold_probabilities

        cases = (
            [np.array([[0.8, 0.2, 0.0]])],
            [np.array([[np.nan, 0.2]])],
            [np.array([[0.8, 0.3]])],
            [np.array([[0.8, 0.2]]), np.array([[0.4, 0.6], [0.3, 0.7]])],
        )
        for fold_probabilities in cases:
            with self.subTest(fold_probabilities=fold_probabilities):
                with self.assertRaises(ValueError):
                    mean_fold_probabilities(fold_probabilities)

    def test_aggregates_event_probabilities_to_subjects(self):
        from experiments.independent_test import aggregate_subject_probabilities

        subject_ids, labels, probabilities = aggregate_subject_probabilities(
            ["A", "A", "B"],
            [1, 1, 0],
            np.array([[0.2, 0.8], [0.4, 0.6], [0.9, 0.1]]),
        )

        self.assertEqual(subject_ids, ["A", "B"])
        np.testing.assert_array_equal(labels, [1, 0])
        np.testing.assert_allclose(probabilities, [[0.3, 0.7], [0.9, 0.1]])

    def test_rejects_conflicting_labels_for_one_subject(self):
        from experiments.independent_test import aggregate_subject_probabilities

        with self.assertRaisesRegex(ValueError, "Conflicting labels.*A"):
            aggregate_subject_probabilities(
                ["A", "A"], [1, 0], np.array([[0.2, 0.8], [0.6, 0.4]])
            )


class ResultArtifactTests(unittest.TestCase):
    def test_build_result_row_uses_evaluator_metrics_and_confusion_order(self):
        from experiments.independent_test import RESULT_COLUMNS, build_result_row

        row = build_result_row(
            year="2025", cohort="Elder", model="mlp", level="event",
            labels=np.array([0, 0, 1, 1]),
            probabilities=np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]),
            folds=5, run_id="run-1", device="cpu", seed=3407,
        )

        self.assertEqual(list(row), RESULT_COLUMNS)
        self.assertEqual((row["TN"], row["FP"], row["FN"], row["TP"]), (1, 1, 1, 1))

    def test_upsert_preserves_other_result_keys_and_raw_results(self):
        from experiments.independent_test import RESULT_COLUMNS, upsert_result_rows

        def result_row(year, level, accuracy):
            row = {column: "" for column in RESULT_COLUMNS}
            row.update({
                "DatasetYear": str(year), "Cohort": "Elder", "Model": "mlp",
                "EvaluationLevel": level, "Accuracy": accuracy,
            })
            return row

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_path = root / "independent_results.csv"
            raw_path = root / "raw_results.csv"
            raw_path.write_bytes(b"raw results must not change\n")
            row_2025 = result_row(2025, "event", 0.50)
            row_2026 = result_row(2026, "event", 0.60)

            upsert_result_rows(result_path, [row_2025])
            upsert_result_rows(result_path, [row_2026])
            upsert_result_rows(result_path, [result_row(2025, "event", 0.75)])

            with result_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([(row["DatasetYear"], row["Accuracy"]) for row in rows], [("2025", "0.75"), ("2026", "0.6")])
            self.assertEqual(raw_path.read_bytes(), b"raw results must not change\n")
            self.assertFalse(result_path.with_suffix(".csv.tmp").exists())

    def test_upsert_keeps_original_when_tmp_write_fails(self):
        from experiments.independent_test import RESULT_COLUMNS, upsert_result_rows

        row = {column: "" for column in RESULT_COLUMNS}
        row.update({
            "DatasetYear": "2025", "Cohort": "Elder", "Model": "mlp",
            "EvaluationLevel": "event", "Accuracy": 0.50,
        })
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "independent_results.csv"
            upsert_result_rows(result_path, [row])
            original = result_path.read_bytes()

            with patch("experiments.independent_test.csv.DictWriter.writeheader", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    upsert_result_rows(result_path, [{**row, "Accuracy": 0.75}])

            self.assertEqual(result_path.read_bytes(), original)
            with result_path.open("r", encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle))[0]["Accuracy"], "0.5")

    def test_upsert_rejects_existing_csv_with_an_unapproved_header(self):
        from experiments.independent_test import RESULT_COLUMNS, upsert_result_rows

        row = {column: "" for column in RESULT_COLUMNS}
        row.update({
            "DatasetYear": "2025", "Cohort": "Elder", "Model": "mlp",
            "EvaluationLevel": "event",
        })
        with tempfile.TemporaryDirectory() as directory:
            result_path = Path(directory) / "independent_results.csv"
            result_path.write_text("wrong,column\nvalue,value\n", encoding="utf-8-sig")

            with self.assertRaisesRegex(ValueError, "header"):
                upsert_result_rows(result_path, [row])

    def test_writes_prediction_csv_with_subject_level_metadata(self):
        from experiments.independent_test import write_prediction_csv

        with tempfile.TemporaryDirectory() as directory:
            path = write_prediction_csv(
                Path(directory) / "predictions.csv", ["A", "B"], [0, 1],
                np.array([[0.8, 0.2], [0.1, 0.9]]), "mlp", "subject",
            )

            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0], {
                "subject_id": "A", "true_label": "0", "pred_label": "0",
                "prob_0": "0.8", "prob_1": "0.2", "model": "mlp",
                "evaluation_level": "subject",
            })


if __name__ == "__main__":
    unittest.main()
