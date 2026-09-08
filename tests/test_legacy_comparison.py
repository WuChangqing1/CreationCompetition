import argparse
import tempfile
import unittest
from pathlib import Path


class LegacyComparisonContractTests(unittest.TestCase):
    def test_cli_defaults_to_all_eight_models_and_has_no_folds_option(self):
        from experiments.run_legacy_comparison import parse_args

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "historical.pth"
            args = parse_args([
                "--data-root", str(root),
                "--historical-checkpoint", str(checkpoint),
            ])

            self.assertEqual(
                args.models,
                ["svm", "xgboost", "mlp", "bilstm", "lightweighttrans", "lmf", "mult", "our"],
            )
            self.assertFalse(hasattr(args, "folds"))

            with self.assertRaises(SystemExit):
                parse_args([
                    "--data-root", str(root),
                    "--historical-checkpoint", str(checkpoint),
                    "--folds", "5",
                ])

    def test_execution_plan_freezes_ourmodel_and_trains_each_baseline_once(self):
        from experiments.legacy_comparison import build_execution_plan

        checkpoint = Path("historical.pth")
        plan = build_execution_plan(
            ["svm", "xgboost", "mlp", "bilstm", "lightweighttrans", "lmf", "mult", "our"],
            checkpoint,
        )

        self.assertEqual(len(plan), 8)
        self.assertEqual(plan[-1], {
            "model": "our",
            "action": "evaluate_frozen_checkpoint",
            "checkpoint": checkpoint,
        })
        self.assertTrue(all(item["action"] == "train_single_split" for item in plan[:-1]))
        self.assertTrue(all("fold" not in item and "folds" not in item for item in plan))

    def test_execution_plan_rejects_duplicates_and_unknown_models(self):
        from experiments.legacy_comparison import build_execution_plan

        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_execution_plan(["our", "our"], Path("historical.pth"))
        with self.assertRaisesRegex(ValueError, "unknown"):
            build_execution_plan(["our", "invented"], Path("historical.pth"))

    def test_legacy_split_uses_the_historical_split_function_and_requires_292_45(self):
        from experiments.legacy_comparison import load_historical_split

        calls = []

        def fake_split(path, val_ratio, random_seed):
            calls.append((Path(path), val_ratio, random_seed))
            return ([{"id": index} for index in range(292)], [{"id": index} for index in range(45)], {}, {})

        train, val = load_historical_split(Path("Training_Validation_files.json"), split_fn=fake_split)

        self.assertEqual((len(train), len(val)), (292, 45))
        self.assertEqual(calls, [(Path("Training_Validation_files.json"), 0.1, 2024)])

        def wrong_split(path, val_ratio, random_seed):
            return ([{}], [{}], {}, {})

        with self.assertRaisesRegex(RuntimeError, "292/45"):
            load_historical_split(Path("wrong.json"), split_fn=wrong_split)

    def test_frozen_historical_membership_does_not_depend_on_set_iteration(self):
        from experiments.legacy_comparison import (
            HISTORICAL_VALIDATION_AUDIO_PATHS,
            split_entries_by_frozen_membership,
        )

        validation = [
            {"audio_feature_path": path, "marker": index}
            for index, path in enumerate(HISTORICAL_VALIDATION_AUDIO_PATHS)
        ]
        training = [
            {"audio_feature_path": f"train_{index}.npy", "marker": index}
            for index in range(292)
        ]
        entries = list(reversed(training + validation))

        train_entries, val_entries = split_entries_by_frozen_membership(entries)

        self.assertEqual(len(train_entries), 292)
        self.assertEqual(len(val_entries), 45)
        self.assertEqual(
            {entry["audio_feature_path"] for entry in val_entries},
            set(HISTORICAL_VALIDATION_AUDIO_PATHS),
        )

    def test_ourmodel_is_validated_before_any_baseline_runs(self):
        from experiments.legacy_comparison import execute_plan

        plan = [
            {"model": "svm", "action": "train_single_split"},
            {"model": "our", "action": "evaluate_frozen_checkpoint", "checkpoint": Path("old.pth")},
        ]
        events = []

        def verify_our(item):
            events.append(("verify", item["model"]))
            raise RuntimeError("historical lock mismatch")

        def train_baseline(item):
            events.append(("train", item["model"]))

        with self.assertRaisesRegex(RuntimeError, "historical lock mismatch"):
            execute_plan(plan, verify_our=verify_our, train_baseline=train_baseline)

        self.assertEqual(events, [("verify", "our")])


if __name__ == "__main__":
    unittest.main()
