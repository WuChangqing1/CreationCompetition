import importlib.util
import unittest


class SmokeContractTest(unittest.TestCase):
    def test_smoke_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("experiments.smoke_test"))

    def test_smoke_reports_required_components_and_valid_statuses(self):
        from experiments.smoke_test import run_smoke

        results = run_smoke(data_root=None, device="cpu")
        names = {result.name for result in results}
        required = {
            "Environment", "Dataset", "our", "mlp", "bilstm",
            "lightweighttrans", "lmf", "mult", "SVM", "XGBoost",
            "Evaluator", "Subject Split",
        }
        self.assertEqual(names, required)
        self.assertTrue(all(result.status in {"PASS", "FAIL", "SKIPPED", "N/A"} for result in results))
        dataset = next(result for result in results if result.name == "Dataset")
        self.assertEqual(dataset.status, "SKIPPED")


if __name__ == "__main__":
    unittest.main()
