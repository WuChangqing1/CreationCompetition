import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


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


if __name__ == "__main__":
    unittest.main()
