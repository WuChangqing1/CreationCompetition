import unittest

import numpy as np


class LegacyProtocolDefaultsTest(unittest.TestCase):
    def test_old_bicfnet_protocol_is_the_default(self):
        from experiments.run_model_cv import parse_args

        args, config = parse_args(["--model", "our"])

        self.assertEqual(args.protocol, "legacy_bicfnet")
        self.assertEqual(args.seed, 2024)
        self.assertEqual(args.feature_max_len, 26)
        self.assertEqual(args.batch_size, 8)
        self.assertEqual(args.epochs, 300)
        self.assertEqual(config["learning_rate"], 2e-5)
        self.assertEqual(config["weight_decay"], 0.01)
        self.assertEqual(config["focal_weight"], 0.1)
        self.assertEqual(config["loss_function"], "CrossEntropyLoss+FocalLoss")
        self.assertEqual(config["scheduler"], "cosine")
        self.assertEqual(config["checkpoint_selection"], "val_macro_f1")
        self.assertEqual(args.results_dir.name, "results_legacy_bicfnet")

    def test_modern_protocol_preserves_the_current_configuration(self):
        from experiments.run_model_cv import parse_args

        args, config = parse_args(["--model", "our", "--protocol", "modern"])

        self.assertEqual(args.seed, 3407)
        self.assertEqual(args.feature_max_len, 5)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.epochs, 20)
        self.assertEqual(config["learning_rate"], 1e-4)
        self.assertEqual(config["weight_decay"], 1e-4)
        self.assertEqual(config["focal_weight"], 0.0)
        self.assertEqual(args.results_dir.name, "results")

    def test_explicit_cli_values_override_protocol_defaults(self):
        from experiments.run_model_cv import parse_args

        args, _ = parse_args([
            "--model", "our", "--seed", "7", "--feature-max-len", "9",
            "--batch-size", "4", "--epochs", "2",
        ])

        self.assertEqual((args.seed, args.feature_max_len, args.batch_size, args.epochs), (7, 9, 4, 2))

    def test_orchestration_and_independent_test_default_to_legacy_outputs(self):
        from experiments.run_all import parse_args as parse_all
        from experiments.run_independent_test import parse_args as parse_independent

        all_args = parse_all(["--models", "our"])
        independent_args = parse_independent([
            "--models", "our", "--dataset-year", "2025", "--data-root", "data"
        ])

        self.assertEqual((all_args.protocol, all_args.seed), ("legacy_bicfnet", 2024))
        self.assertEqual(all_args.results_dir.name, "results_legacy_bicfnet")
        self.assertEqual((independent_args.protocol, independent_args.seed), ("legacy_bicfnet", 2024))
        self.assertEqual(independent_args.results_dir.name, "results_legacy_bicfnet")
        self.assertEqual(independent_args.cv_results.parent.name, "results_legacy_bicfnet")
        self.assertEqual(independent_args.output.parent.name, "results_legacy_bicfnet")

    def test_independent_legacy_protocol_uses_majority_vote(self):
        from experiments.run_independent_test import parse_args

        legacy = parse_args(["--models", "our", "--dataset-year", "2025", "--data-root", "data"])
        modern = parse_args([
            "--models", "our", "--dataset-year", "2025", "--data-root", "data",
            "--protocol", "modern",
        ])

        self.assertEqual(legacy.subject_aggregation, "majority_vote")
        self.assertTrue(legacy.include_legacy_voted_event)
        self.assertEqual(modern.subject_aggregation, "probability_mean")
        self.assertFalse(modern.include_legacy_voted_event)


class LegacyCheckpointSelectionTest(unittest.TestCase):
    def test_validation_macro_f1_improvement_is_strict(self):
        from experiments.run_model_cv import is_better_validation

        self.assertTrue(is_better_validation(0.6, None))
        self.assertTrue(is_better_validation(0.6, 0.5))
        self.assertFalse(is_better_validation(0.6, 0.6))
        self.assertFalse(is_better_validation(0.5, 0.6))


class LegacyVotingTest(unittest.TestCase):
    def test_majority_vote_can_differ_from_mean_probability_and_breaks_ties_by_mean(self):
        from experiments.independent_test import aggregate_subject_predictions

        ids, labels, probabilities = aggregate_subject_predictions(
            ["A", "A", "A", "B", "B"],
            [1, 1, 1, 0, 0],
            np.array([
                [0.49, 0.51], [0.49, 0.51], [0.99, 0.01],
                [0.90, 0.10], [0.10, 0.90],
            ]),
            method="majority_vote",
        )

        self.assertEqual(ids, ["A", "B"])
        np.testing.assert_array_equal(labels, [1, 0])
        self.assertEqual(probabilities.argmax(1).tolist(), [1, 0])

    def test_old_compatible_broadcast_repeats_subject_vote_for_each_event(self):
        from experiments.independent_test import broadcast_subject_predictions

        probabilities = broadcast_subject_predictions(
            ["A", "A", "B"], ["A", "B"], np.array([[0.0, 1.0], [1.0, 0.0]])
        )

        np.testing.assert_array_equal(probabilities, [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])


if __name__ == "__main__":
    unittest.main()
