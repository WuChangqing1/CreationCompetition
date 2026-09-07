import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


class CVSupportTest(unittest.TestCase):
    def test_run_identity_separates_tiny_and_hyperparameters(self):
        from experiments.run_model_cv import parse_args, build_run_id

        args, config = parse_args(["--model", "mlp"])
        baseline = build_run_id(args, config)
        args.tiny = True
        self.assertNotEqual(baseline, build_run_id(args, config))
        args.tiny = False
        self.assertNotEqual(baseline, build_run_id(args, {**config, "learning_rate": 0.123}))

    def test_new_cv_entrypoint_rejects_non_binary_tasks(self):
        from experiments.run_model_cv import parse_args

        with self.assertRaises(SystemExit):
            parse_args(["--model", "mlp", "--task", "ternary"])

    def test_cv_modules_exist(self):
        for module in ("experiments.checkpointing", "experiments.efficiency", "experiments.run_model_cv"):
            self.assertIsNotNone(importlib.util.find_spec(module))

    def test_checkpoint_contains_reproducibility_metadata(self):
        from experiments.checkpointing import build_checkpoint_payload

        payload = build_checkpoint_payload(
            "mlp", {"w": 1}, {"lr": 0.001}, {"audio_feature": "mfccs"}, 2, 3407
        )
        self.assertEqual(
            set(payload),
            {"model_name", "model_state_dict", "config", "feature_config", "fold", "seed"},
        )

    def test_seed_resets_numpy_and_torch(self):
        from experiments.run_model_cv import set_random_seed

        set_random_seed(3407)
        first = (np.random.rand(), torch.rand(1).item())
        set_random_seed(3407)
        second = (np.random.rand(), torch.rand(1).item())
        self.assertEqual(first, second)

    def test_cpu_efficiency_has_no_fake_vram(self):
        from experiments.efficiency import measure_torch_efficiency

        model = torch.nn.Linear(3, 2)
        result = measure_torch_efficiency(model, lambda: model(torch.ones(1, 3)), "cpu", warmup=1, measure=2)
        self.assertEqual(result["Parameters"], 8)
        self.assertEqual(result["Peak_VRAM_MB"], "N/A")
        self.assertGreaterEqual(result["Inference_Latency_ms"], 0.0)

    def test_tiny_mlp_fold_runs_one_training_batch_and_saves_predictions(self):
        from experiments.run_model_cv import run_fold

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "Training"
            audio_dir = training / "1s" / "Audio" / "mfccs"
            video_dir = training / "1s" / "Visual" / "densenet"
            personality_dir = training / "individualEmbedding"
            audio_dir.mkdir(parents=True)
            video_dir.mkdir(parents=True)
            personality_dir.mkdir(parents=True)
            entries, personalities = [], []
            train_ids, val_ids = [], []
            for label in (0, 1):
                for index in range(5):
                    subject = str(label * 10 + index + 1)
                    filename = f"{subject}_0.npy"
                    np.save(audio_dir / filename, np.ones((3, 8), dtype=np.float32) * (label + 1))
                    np.save(video_dir / filename, np.ones((3, 10), dtype=np.float32) * (label + 1))
                    entries.append({"id": subject, "audio_feature_path": filename, "video_feature_path": filename, "bin_category": label})
                    personalities.append({"id": subject, "embedding": np.ones(1024, dtype=np.float32) * label})
                    (val_ids if index == 4 else train_ids).append(subject)
            np.save(personality_dir / "descriptions_embeddings_with_ids.npy", np.array(personalities, dtype=object))
            results = root / "results"
            args = SimpleNamespace(
                model="mlp", classes=2, feature_max_len=4, batch_size=8, tiny=True,
                epochs=3, device="cpu", warmup=1, measure=1, results_dir=results,
                audio_feature="mfccs", video_feature="densenet", use_personality=True,
                track="Track1", task="binary", split_window="1s", seed=3407,
            )
            paths = {
                "personality": personality_dir / "descriptions_embeddings_with_ids.npy",
                "audio": audio_dir, "video": video_dir,
            }
            fold = {"fold": 1, "train_ids": train_ids, "val_ids": val_ids}
            config = {"learning_rate": 0.001, "batch_size": 8, "epochs": 3, "hidden_dim": 8, "dropout": 0.0, "loss_function": "CrossEntropyLoss", "class_balancing_strategy": "None"}
            row = run_fold(args, fold, entries, paths, config)
            self.assertEqual(row["Status"], "PASS")
            self.assertEqual(len(list((results / "predictions").glob("*_mlp_fold1.csv"))), 1)


if __name__ == "__main__":
    unittest.main()
