import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from experiments.create_splits import create_subject_folds


class _RecordingEstimator:
    def __init__(self, probabilities, audit):
        self.probabilities = np.asarray(probabilities, dtype=float)
        self.audit = audit

    def fit(self, features, labels):
        self.audit["fit_features"].append(np.asarray(features).copy())
        self.audit["fit_labels"].append(np.asarray(labels).copy())
        return self

    def predict_proba(self, features):
        self.audit["test_features"].append(np.asarray(features).copy())
        return self.probabilities


class _DeterministicModel(torch.nn.Module):
    def __init__(self, audit):
        super().__init__()
        self.audit = audit
        self.probabilities = None

    def load_state_dict(self, state_dict, strict=True):
        self.audit["loaded_folds"].append(int(state_dict["fold"]))
        self.audit["strict"].append(strict)
        self.probabilities = state_dict["probabilities"].clone()

    def to(self, device):
        self.audit["devices"].append(str(device))
        return self

    def set_input(self, batch):
        self.batch = batch

    def forward(self):
        labels = self.batch["emo_label"].cpu().numpy().tolist()
        self.audit["label_orders"].append(labels)
        self.emo_pred = self.probabilities[: len(labels)]


class IndependentTestTorchTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.audio_dir = self.root / "audio"
        self.video_dir = self.root / "video"
        self.audio_dir.mkdir()
        self.video_dir.mkdir()
        self.personality_path = self.root / "personality.npy"
        self.entries = []
        personalities = []
        for index, (subject_id, label) in enumerate((("01", 0), ("02", 1)), 1):
            filename = f"{subject_id}_event.npy"
            np.save(self.audio_dir / filename, np.full((2, 3), index, dtype=np.float32))
            np.save(self.video_dir / filename, np.full((2, 4), index, dtype=np.float32))
            self.entries.append({
                "id": subject_id,
                "audio_feature_path": filename,
                "video_feature_path": filename,
                "bin_category": label,
            })
            personalities.append({
                "id": subject_id,
                "embedding": np.full(1024, label, dtype=np.float32),
            })
        np.save(self.personality_path, np.asarray(personalities, dtype=object))
        self.paths = {
            "audio": self.audio_dir,
            "video": self.video_dir,
            "personality": self.personality_path,
        }
        self.args = SimpleNamespace(
            device="cpu",
            classes=2,
            feature_max_len=3,
            batch_size=2,
            use_personality=True,
        )
        self.expected_metadata = {
            "model_name": "mlp",
            "seed": 3407,
            "dataset_year": "2025",
            "cohort": "Elder",
            "track": "Track1",
            "task": "binary",
            "audio_feature": "mfccs",
            "video_feature": "densenet",
            "use_personality": True,
            "split_window": "1s",
            "device": "cpu",
        }

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _payload(self, fold):
        probabilities = torch.tensor([
            [0.9 - fold * 0.1, 0.1 + fold * 0.1],
            [0.2 + fold * 0.1, 0.8 - fold * 0.1],
        ], dtype=torch.float32)
        return {
            "model_name": "mlp",
            "model_state_dict": {"fold": fold, "probabilities": probabilities},
            "config": {"saved_fold": fold},
            "feature_config": {
                "dataset_year": "2025",
                "cohort": "Elder",
                "track": "Track1",
                "task": "binary",
                "audio_feature": "mfccs",
                "video_feature": "densenet",
                "use_personality": True,
                "split_window": "1s",
                "device": "cpu",
            },
            "fold": fold,
            "seed": 3407,
        }

    def _write_checkpoints(self, mutation=None, mutation_fold=1):
        checkpoint_paths = []
        for fold in range(1, 6):
            payload = self._payload(fold)
            if fold == mutation_fold and mutation is not None:
                mutation(payload)
            path = self.root / f"fold_{fold}.pth"
            torch.save(payload, path)
            checkpoint_paths.append(path)
        return checkpoint_paths

    def test_loads_five_checkpoints_strictly_and_averages_ordered_probabilities(self):
        from experiments.independent_test_models import infer_torch_fold_ensemble

        audit = {
            "loaded_folds": [],
            "strict": [],
            "devices": [],
            "label_orders": [],
        }

        def create_model(_model_name, opt=None):
            self.assertFalse(opt.isTrain)
            self.assertEqual(opt.gpu_ids, [])
            return _DeterministicModel(audit)

        with patch(
            "experiments.independent_test_models.create_experiment_model",
            side_effect=create_model,
        ) as model_factory:
            probabilities = infer_torch_fold_ensemble(
                model_name="mlp",
                test_entries=self.entries,
                test_paths=self.paths,
                checkpoint_paths=self._write_checkpoints(),
                args=self.args,
                expected_metadata=self.expected_metadata,
            )

        np.testing.assert_allclose(
            probabilities,
            [[0.6, 0.4], [0.5, 0.5]],
            rtol=0.0,
            atol=1e-6,
        )
        self.assertEqual(model_factory.call_count, 5)
        self.assertEqual(audit["loaded_folds"], [1, 2, 3, 4, 5])
        self.assertEqual(audit["strict"], [True] * 5)
        self.assertEqual(audit["devices"], ["cpu"] * 5)
        self.assertEqual(audit["label_orders"], [[0, 1]] * 5)

    def test_rejects_every_provenance_mismatch_before_model_creation(self):
        from experiments.independent_test_models import infer_torch_fold_ensemble

        mutations = {
            "fold": lambda payload: payload.update(fold=1),
            "model_name": lambda payload: payload.update(model_name="bilstm"),
            "seed": lambda payload: payload.update(seed=7),
            "audio_feature": lambda payload: payload["feature_config"].update(audio_feature="wav2vec"),
        }
        for field, mutation in mutations.items():
            with self.subTest(field=field):
                with patch(
                    "experiments.independent_test_models.create_experiment_model"
                ) as model_factory:
                    with self.assertRaisesRegex(ValueError, field):
                        infer_torch_fold_ensemble(
                            model_name="mlp",
                            test_entries=self.entries,
                            test_paths=self.paths,
                            checkpoint_paths=self._write_checkpoints(mutation, mutation_fold=5),
                            args=self.args,
                            expected_metadata=self.expected_metadata,
                        )
                model_factory.assert_not_called()

    def test_requires_exactly_five_checkpoint_paths(self):
        from experiments.independent_test_models import infer_torch_fold_ensemble

        with patch("experiments.independent_test_models.create_experiment_model") as model_factory:
            with self.assertRaisesRegex(ValueError, "exactly five"):
                infer_torch_fold_ensemble(
                    model_name="mlp",
                    test_entries=self.entries,
                    test_paths=self.paths,
                    checkpoint_paths=self._write_checkpoints()[:4],
                    args=self.args,
                    expected_metadata=self.expected_metadata,
                )
        model_factory.assert_not_called()

    def test_cuda_unavailable_raises_without_cpu_fallback(self):
        from experiments.independent_test_models import infer_torch_fold_ensemble

        self.args.device = "cuda"
        self.expected_metadata["device"] = "cuda"
        with (
            patch("experiments.independent_test_models.torch.cuda.is_available", return_value=False),
            patch("experiments.independent_test_models.create_experiment_model") as model_factory,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "CUDA is required for formal independent-test evaluation",
            ):
                infer_torch_fold_ensemble(
                    model_name="mlp",
                    test_entries=self.entries,
                    test_paths=self.paths,
                    checkpoint_paths=self._write_checkpoints(),
                    args=self.args,
                    expected_metadata=self.expected_metadata,
                )
        model_factory.assert_not_called()

    def test_rejects_non_binary_fold_probability_shape(self):
        from experiments.independent_test_models import infer_torch_fold_ensemble

        checkpoint_paths = self._write_checkpoints()
        payload = torch.load(checkpoint_paths[0], map_location="cpu")
        payload["model_state_dict"]["probabilities"] = torch.ones((2, 1))
        torch.save(payload, checkpoint_paths[0])
        audit = {
            "loaded_folds": [],
            "strict": [],
            "devices": [],
            "label_orders": [],
        }
        with patch(
            "experiments.independent_test_models.create_experiment_model",
            side_effect=lambda _model_name, opt=None: _DeterministicModel(audit),
        ):
            with self.assertRaisesRegex(ValueError, r"\[N, 2\]"):
                infer_torch_fold_ensemble(
                    model_name="mlp",
                    test_entries=self.entries,
                    test_paths=self.paths,
                    checkpoint_paths=checkpoint_paths,
                    args=self.args,
                    expected_metadata=self.expected_metadata,
                )

    def test_rejects_fold_probabilities_missing_an_event_row(self):
        from experiments.independent_test_models import infer_torch_fold_ensemble

        checkpoint_paths = self._write_checkpoints()
        for checkpoint_path in checkpoint_paths:
            payload = torch.load(checkpoint_path, map_location="cpu")
            payload["model_state_dict"]["probabilities"] = payload[
                "model_state_dict"
            ]["probabilities"][:1]
            torch.save(payload, checkpoint_path)
        audit = {
            "loaded_folds": [],
            "strict": [],
            "devices": [],
            "label_orders": [],
        }
        with patch(
            "experiments.independent_test_models.create_experiment_model",
            side_effect=lambda _model_name, opt=None: _DeterministicModel(audit),
        ):
            with self.assertRaisesRegex(ValueError, r"expected \(2, 2\).+got \(1, 2\)"):
                infer_torch_fold_ensemble(
                    model_name="mlp",
                    test_entries=self.entries,
                    test_paths=self.paths,
                    checkpoint_paths=checkpoint_paths,
                    args=self.args,
                    expected_metadata=self.expected_metadata,
                )


class IndependentTestClassicalTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.training_paths = self._make_paths("training")
        self.test_paths = self._make_paths("test")
        self.training_entries = self._make_entries(
            self.training_paths, [(f"{index:02d}", index % 2, index + 1) for index in range(10)]
        )
        self.test_entries = self._make_entries(
            self.test_paths, [("10", 0, 11), ("11", 1, 12)]
        )
        self.args = SimpleNamespace(
            classes=2,
            feature_max_len=2,
            batch_size=2,
            use_personality=True,
            seed=3407,
            device="cuda",
            dataset_year="2025",
            cohort="Elder",
        )
        self.fold_records = create_subject_folds(
            self.training_entries, "bin_category", folds=5, seed=self.args.seed
        )

    def tearDown(self):
        self.temporary_directory.cleanup()

    def _make_paths(self, name):
        root = self.root / name
        audio = root / "audio"
        video = root / "video"
        audio.mkdir(parents=True)
        video.mkdir()
        return {"audio": audio, "video": video, "personality": root / "personality.npy"}

    def _make_entries(self, paths, subjects):
        entries, personalities = [], []
        for subject_id, label, value in subjects:
            filename = f"{subject_id}_event.npy"
            np.save(
                paths["audio"] / filename,
                np.asarray([[value, value + 0.25], [value, value + 0.25]], dtype=np.float32),
            )
            np.save(
                paths["video"] / filename,
                np.asarray([[value + 10, value + 10.25], [value + 10, value + 10.25]], dtype=np.float32),
            )
            entries.append({
                "subject_id": subject_id,
                "audio_feature_path": filename,
                "video_feature_path": filename,
                "bin_category": label,
            })
            personalities.append({
                "id": subject_id,
                "embedding": np.asarray([value + 20, value + 20.25, value + 20.5], dtype=np.float32),
            })
        np.save(paths["personality"], np.asarray(personalities, dtype=object))
        return entries

    def _probabilities(self, fold):
        return np.asarray(
            [[0.9 - fold * 0.1, 0.1 + fold * 0.1], [0.2 + fold * 0.1, 0.8 - fold * 0.1]],
            dtype=float,
        )

    def test_refits_each_fold_on_only_its_training_subjects_and_equally_averages(self):
        from experiments.classical.common import pool_multimodal_features
        from experiments.independent_test_models import infer_classical_fold_ensemble

        audit = {"configs": [], "fit_features": [], "fit_labels": [], "test_features": []}

        def create_model(model_name, config):
            audit["configs"].append((model_name, dict(config)))
            return _RecordingEstimator(self._probabilities(len(audit["configs"])), audit)

        with (
            patch(
                "experiments.independent_test_models.create_experiment_model",
                side_effect=create_model,
            ) as model_factory,
            patch(
                "experiments.independent_test_models.pool_multimodal_features",
                wraps=pool_multimodal_features,
            ) as pool_features,
        ):
            labels, probabilities = infer_classical_fold_ensemble(
                model_name="svm",
                training_entries=self.training_entries,
                training_paths=self.training_paths,
                test_entries=self.test_entries,
                test_paths=self.test_paths,
                fold_records=self.fold_records,
                config={"C": 2.0, "seed": 7},
                args=self.args,
            )

        np.testing.assert_array_equal(labels, [0, 1])
        np.testing.assert_allclose(probabilities, [[0.6, 0.4], [0.5, 0.5]], rtol=0.0, atol=1e-6)
        self.assertEqual(model_factory.call_count, 5)
        self.assertEqual(pool_features.call_count, 6)
        self.assertEqual(pool_features.call_args_list[0].args[0].shape[0], 2)
        self.assertTrue(pool_features.call_args_list[0].args[3])
        self.assertEqual(len(audit["test_features"]), 5)
        self.assertEqual(audit["test_features"][0].shape, (2, 7))
        for index, record in enumerate(self.fold_records):
            trained_subject_values = set(audit["fit_features"][index][:, 0].astype(int))
            self.assertEqual(trained_subject_values, {int(subject) + 1 for subject in record["train_ids"]})
            self.assertTrue(trained_subject_values.isdisjoint({int(subject) + 1 for subject in record["val_ids"]}))
        self.assertEqual([config["seed"] for _, config in audit["configs"]], [3407] * 5)

    def test_personality_toggle_and_xgboost_cuda_configuration_match_cv(self):
        from experiments.independent_test_models import (
            classical_execution_device,
            infer_classical_fold_ensemble,
        )

        self.args.use_personality = False
        audit = {"configs": [], "fit_features": [], "fit_labels": [], "test_features": []}

        def create_model(model_name, config):
            audit["configs"].append((model_name, dict(config)))
            return _RecordingEstimator(self._probabilities(len(audit["configs"])), audit)

        with patch(
            "experiments.independent_test_models.create_experiment_model",
            side_effect=create_model,
        ):
            _, probabilities = infer_classical_fold_ensemble(
                model_name="xgboost",
                training_entries=self.training_entries,
                training_paths=self.training_paths,
                test_entries=self.test_entries,
                test_paths=self.test_paths,
                fold_records=self.fold_records,
                config={"max_depth": 2, "seed": 7},
                args=self.args,
            )

        np.testing.assert_allclose(probabilities, [[0.6, 0.4], [0.5, 0.5]], rtol=0.0, atol=1e-6)
        self.assertEqual(audit["test_features"][0].shape, (2, 4))
        self.assertEqual(classical_execution_device("svm", self.args), "cpu:svm")
        self.assertEqual(classical_execution_device("xgboost", self.args), "cuda:xgboost")
        for model_name, config in audit["configs"]:
            self.assertEqual(model_name, "xgboost")
            self.assertEqual(config["device"], "cuda")
            self.assertEqual(config["tree_method"], "hist")
            self.assertEqual(config["seed"], 3407)

    def test_rejects_invalid_saved_fold_provenance_before_creating_an_estimator(self):
        from experiments.independent_test_models import infer_classical_fold_ensemble

        mutations = {
            "fold": lambda records: records[-1].update(fold=1),
            "folds": lambda records: records[-1].update(folds=4),
            "seed": lambda records: records[-1].update(seed=7),
            "non_integer_seed": lambda records: records[-1].update(seed=3407.0),
            "label_key": lambda records: records[-1].update(label_key="tri_category"),
            "overlap": lambda records: records[-1]["train_ids"].append(records[-1]["val_ids"][0]),
            "coverage": lambda records: records[-1]["val_ids"].pop(),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                records = json.loads(json.dumps(self.fold_records))
                mutate(records)
                with patch(
                    "experiments.independent_test_models.create_experiment_model"
                ) as model_factory:
                    with self.assertRaises(ValueError):
                        infer_classical_fold_ensemble(
                            model_name="svm",
                            training_entries=self.training_entries,
                            training_paths=self.training_paths,
                            test_entries=self.test_entries,
                            test_paths=self.test_paths,
                            fold_records=records,
                            config={},
                            args=self.args,
                        )
                model_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
