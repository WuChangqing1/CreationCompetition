import importlib.util
import unittest

from experiments.errors import OptionalDependencyError
from models import create_model, find_model_using_name
from tests.common import make_opt


class OptionalModelsTest(unittest.TestCase):
    def test_optional_model_modules_exist(self):
        self.assertIsNotNone(importlib.util.find_spec("models.depmamba_model"))
        self.assertIsNotNone(importlib.util.find_spec("models.proposed_model"))

    def test_depmamba_discovery_does_not_import_optional_packages(self):
        self.assertEqual(find_model_using_name("depmamba").__name__, "DepMambaModel")
        self.assertEqual(find_model_using_name("mlp").__name__, "MLPModel")

    def test_missing_depmamba_dependency_is_explicitly_skipped(self):
        if importlib.util.find_spec("mamba_ssm") is None:
            with self.assertRaisesRegex(OptionalDependencyError, "mamba_ssm"):
                create_model(make_opt(model="depmamba"))

    def test_proposed_is_explicitly_unimplemented(self):
        with self.assertRaisesRegex(NotImplementedError, "has not been implemented"):
            create_model(make_opt(model="proposed"))


if __name__ == "__main__":
    unittest.main()
