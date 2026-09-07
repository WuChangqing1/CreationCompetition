import importlib.util
import unittest

import numpy as np


class ExperimentRegistryTest(unittest.TestCase):
    def test_classical_modules_exist(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.model_registry"))
        self.assertIsNotNone(importlib.util.find_spec("experiments.classical.common"))

    def test_pooling_uses_temporal_mean_and_personality(self):
        from experiments.classical.common import pool_multimodal_features

        out = pool_multimodal_features(
            np.array([[[1.0, 3.0], [3.0, 5.0]]]),
            np.array([[[2.0], [4.0]]]),
            np.array([[9.0]]),
        )
        np.testing.assert_allclose(out, [[2.0, 4.0, 3.0, 9.0]])

    def test_svm_pipeline_scales_before_classification(self):
        from experiments.classical.svm_baseline import create_svm

        self.assertEqual(list(create_svm().named_steps), ["scaler", "classifier"])

    def test_registry_separates_classical_and_torch(self):
        from experiments.model_registry import CLASSICAL_MODELS, TORCH_MODELS, get_model_kind

        self.assertEqual(get_model_kind("svm"), "classical")
        self.assertEqual(get_model_kind("xgboost"), "classical")
        self.assertEqual(get_model_kind("mlp"), "torch")
        self.assertEqual(CLASSICAL_MODELS, frozenset({"svm", "xgboost"}))
        self.assertEqual(
            TORCH_MODELS,
            frozenset({"mlp", "bilstm", "lightweighttrans", "lmf", "mult", "our"}),
        )

    def test_removed_models_are_rejected(self):
        from experiments.model_registry import get_model_kind

        for name in ("depmamba", "proposed"):
            with self.subTest(model=name):
                with self.assertRaisesRegex(ValueError, f"Unknown model: {name}"):
                    get_model_kind(name)

    def test_missing_xgboost_isolated_to_xgboost(self):
        from experiments.model_registry import OptionalDependencyError, create_experiment_model
        from tests.common import make_opt

        if importlib.util.find_spec("xgboost") is None:
            with self.assertRaisesRegex(OptionalDependencyError, "python -m pip install xgboost"):
                create_experiment_model("xgboost", make_opt())
        self.assertIsNotNone(create_experiment_model("svm", make_opt()))


if __name__ == "__main__":
    unittest.main()
