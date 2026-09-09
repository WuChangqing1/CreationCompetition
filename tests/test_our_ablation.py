import subprocess
import sys
import unittest
import csv
import tempfile
from pathlib import Path

import torch

from models.our.our_model import ourModel
from tests.common import make_opt, sample_batch


class OurAblationModelTest(unittest.TestCase):
    def _make_model(self, variant="full", *, is_train=False):
        from models.ourablation_model import OurablationModel

        model = OurablationModel(
            make_opt(model="ourablation", ablation_variant=variant, isTrain=is_train)
        )
        model.eval()
        return model

    def _forward_with_call_counts(self, variant):
        model = self._make_model(variant)
        counts = {
            "EmoA": 0,
            "EmoV": 0,
            "VEM": 0,
            "CFM": 0,
            "audio_adapter": 0,
            "visual_adapter": 0,
            "ProjP": 0,
        }
        handles = []
        for name, module in (
            ("EmoA", model.netEmoA),
            ("EmoV", model.netEmoV),
            ("VEM", model.netVEM),
            ("CFM", model.netCFM),
            ("audio_adapter", model.netCFM.audio_adapter),
            ("visual_adapter", model.netCFM.visual_adapter),
            ("ProjP", model.netProjP),
        ):
            handles.append(module.register_forward_hook(
                lambda _module, _inputs, _output, key=name: counts.__setitem__(key, counts[key] + 1)
            ))
        try:
            model.set_input(sample_batch())
            model.forward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(tuple(model.emo_pred.shape), (2, 2))
        return counts

    def test_full_variant_is_numerically_identical_to_original_model(self):
        torch.manual_seed(123)
        original = ourModel(make_opt(model="our", isTrain=False))
        ablated = self._make_model("full")
        ablated.load_state_dict(original.state_dict(), strict=True)
        original.eval()
        batch = sample_batch()

        original.set_input(batch)
        ablated.set_input(batch)
        original.forward()
        ablated.forward()

        torch.testing.assert_close(ablated.emo_logits, original.emo_logits, rtol=0, atol=0)
        torch.testing.assert_close(ablated.emo_logits_fusion, original.emo_logits_fusion, rtol=0, atol=0)

    def test_component_variants_bypass_only_the_declared_component(self):
        expected = {
            "no_personality": {"EmoA": 1, "EmoV": 1, "VEM": 1, "CFM": 1, "ProjP": 0},
            "no_vem": {"EmoA": 1, "EmoV": 1, "VEM": 0, "CFM": 1, "ProjP": 1},
            "no_cfm": {"EmoA": 1, "EmoV": 1, "VEM": 1, "CFM": 0, "ProjP": 1},
        }
        for variant, wanted in expected.items():
            with self.subTest(variant=variant):
                counts = self._forward_with_call_counts(variant)
                for module_name, call_count in wanted.items():
                    self.assertEqual(counts[module_name], call_count)

    def test_directional_variants_disable_the_correct_feedback_direction(self):
        no_a2v = self._forward_with_call_counts("no_a2v")
        self.assertEqual(no_a2v["audio_adapter"], 0)
        self.assertEqual(no_a2v["visual_adapter"], 1)

        no_v2a = self._forward_with_call_counts("no_v2a")
        self.assertEqual(no_v2a["audio_adapter"], 1)
        self.assertEqual(no_v2a["visual_adapter"], 0)

    def test_single_modality_variants_do_not_execute_disabled_encoders(self):
        expected = {
            "audio_only": {"EmoA": 1, "EmoV": 0, "VEM": 0, "CFM": 0, "ProjP": 0},
            "visual_only": {"EmoA": 0, "EmoV": 1, "VEM": 1, "CFM": 0, "ProjP": 0},
            "personality_only": {"EmoA": 0, "EmoV": 0, "VEM": 0, "CFM": 0, "ProjP": 1},
        }
        for variant, wanted in expected.items():
            with self.subTest(variant=variant):
                counts = self._forward_with_call_counts(variant)
                for module_name, call_count in wanted.items():
                    self.assertEqual(counts[module_name], call_count)

    def test_no_aux_focal_sets_auxiliary_loss_to_zero(self):
        model = self._make_model("no_aux_focal", is_train=True)
        model.set_input(sample_batch())
        model.forward()
        model.optimizer.zero_grad()
        model.backward()
        self.assertEqual(float(model.loss_EmoF_CE.detach().cpu()), 0.0)
        self.assertGreater(float(model.loss_emo_CE.detach().cpu()), 0.0)

    def test_unknown_variant_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown OurModel ablation variant"):
            self._make_model("invented")


class AblationRunnerTest(unittest.TestCase):
    def test_ablation_model_is_torch_without_expanding_formal_eight_model_set(self):
        from experiments.model_registry import TORCH_MODELS, get_model_kind

        self.assertEqual(get_model_kind("ourablation"), "torch")
        self.assertNotIn("ourablation", TORCH_MODELS)

    def test_runner_help_works_as_a_direct_file_entrypoint(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run(
            [sys.executable, str(root / "experiments" / "run_ablation.py"), "--help"],
            cwd=root, text=True, capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_variant_config_records_variant_and_legacy_training_contract(self):
        from experiments.run_ablation import build_variant_config

        config = build_variant_config("no_vem", "legacy_bicfnet")
        self.assertEqual(config["ablation_variant"], "no_vem")
        self.assertEqual(config["learning_rate"], 2e-5)
        self.assertEqual(config["focal_weight"], 0.1)
        self.assertEqual(config["checkpoint_selection"], "val_macro_f1")

        no_aux = build_variant_config("no_aux_focal", "legacy_bicfnet")
        self.assertEqual(no_aux["focal_weight"], 0.0)
        self.assertEqual(no_aux["loss_function"], "CrossEntropyLoss")

    def test_combined_csv_identifies_each_ablation_variant(self):
        from experiments.run_ablation import combine_variant_csvs

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variant, score in (("full", "0.90"), ("no_vem", "0.80")):
                path = root / variant / "raw_results.csv"
                path.parent.mkdir(parents=True)
                with path.open("w", newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("Model", "Fold", "Macro_F1"))
                    writer.writeheader()
                    writer.writerow({"Model": "ourablation", "Fold": 1, "Macro_F1": score})

            output = combine_variant_csvs(
                root, ("full", "no_vem"), "raw_results.csv", "ablation_raw_results.csv"
            )
            with output.open(newline="", encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual([row["Ablation"] for row in rows], ["full", "no_vem"])
        self.assertEqual([row["Macro_F1"] for row in rows], ["0.90", "0.80"])

    def test_runner_defaults_to_one_training_and_direct_test_pass(self):
        from experiments.run_ablation import parse_args

        args = parse_args([
            "--dataset-year", "2026",
            "--data-root", r"D:\dataset",
            "--variants", "full,no_vem",
        ])
        self.assertFalse(hasattr(args, "folds"))
        self.assertFalse(hasattr(args, "stage"))
        self.assertEqual(args.personality_id_source, "subject_id")
        self.assertEqual(args.epochs, 300)

    def test_runner_rejects_fold_and_separate_stage_arguments(self):
        from experiments.run_ablation import parse_args

        base = ["--dataset-year", "2025", "--data-root", r"D:\dataset"]
        with self.assertRaises(SystemExit):
            parse_args([*base, "--folds", "5"])
        with self.assertRaises(SystemExit):
            parse_args([*base, "--stage", "cv"])

    def test_2026_single_holdout_is_subject_disjoint_and_stratified(self):
        from experiments.run_ablation import split_2026_subject_holdout

        entries = []
        for label in (0, 1):
            for index in range(10):
                subject = f"{label}-{index}"
                entries.extend([
                    {"subject_id": subject, "bin_category": label, "audio_feature_path": f"{subject}/A_1.npy"},
                    {"subject_id": subject, "bin_category": label, "audio_feature_path": f"{subject}/A_2.npy"},
                ])
        train, validation = split_2026_subject_holdout(entries, seed=2024, validation_ratio=0.1)
        train_subjects = {row["subject_id"] for row in train}
        validation_subjects = {row["subject_id"] for row in validation}
        self.assertFalse(train_subjects & validation_subjects)
        self.assertEqual({row["bin_category"] for row in validation}, {0, 1})
        self.assertEqual(len(validation_subjects), 2)


if __name__ == "__main__":
    unittest.main()
