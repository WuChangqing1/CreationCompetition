import unittest
import importlib.util

import torch

from models import create_model, find_model_using_name
from tests.common import make_opt, sample_batch


class SimpleModelsTest(unittest.TestCase):
    def assert_forward(self, name):
        self.assertIsNotNone(importlib.util.find_spec(f"models.{name}_model"))
        self.assertTrue(issubclass(find_model_using_name(name), object))
        model = create_model(make_opt(model=name, isTrain=True))
        model.set_input(sample_batch())
        model.forward()
        self.assertEqual(tuple(model.emo_logits.shape), (2, 2))
        self.assertTrue(torch.allclose(model.emo_pred.sum(1), torch.ones(2), atol=1e-5))
        model.optimize_parameters(0)

    def test_mlp_forward(self):
        self.assert_forward("mlp")

    def test_bilstm_forward(self):
        self.assert_forward("bilstm")

    def test_lightweighttrans_forward(self):
        self.assert_forward("lightweighttrans")


if __name__ == "__main__":
    unittest.main()
