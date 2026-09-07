import importlib.util
import unittest

from models import create_model
from tests.common import make_opt, sample_batch


class FusionModelsTest(unittest.TestCase):
    def assert_forward(self, name):
        self.assertIsNotNone(importlib.util.find_spec(f"models.{name}_model"))
        model = create_model(make_opt(model=name, isTrain=True))
        model.set_input(sample_batch())
        model.forward()
        self.assertEqual(tuple(model.emo_logits.shape), (2, 2))
        model.optimize_parameters(0)

    def test_lmf_forward(self):
        self.assert_forward("lmf")

    def test_mult_forward(self):
        self.assert_forward("mult")


if __name__ == "__main__":
    unittest.main()
