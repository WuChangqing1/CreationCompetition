import unittest
import torch

from models import create_model, find_model_using_name
import test as test_module
from tests.common import make_opt, sample_batch


class ModelRegistryTest(unittest.TestCase):
    def test_our_model_is_created_through_registry_and_forwards(self):
        model = create_model(make_opt(model="our", isTrain=False))
        model.set_input(sample_batch())
        model.forward()
        self.assertEqual(type(model).__name__, "ourModel")
        self.assertEqual(tuple(model.emo_logits.shape), (2, 2))

    def test_missing_model_raises_import_error_instead_of_exiting(self):
        with self.assertRaises(ImportError):
            find_model_using_name("definitely_missing")

    def test_checkpoint_payload_accepts_legacy_and_metadata(self):
        self.assertTrue(hasattr(test_module, "unpack_checkpoint"))
        unpack_checkpoint = test_module.unpack_checkpoint
        legacy = {"weight": 1}
        self.assertEqual(unpack_checkpoint(legacy, "our"), ("our", legacy, {}))
        modern = {
            "model_name": "mlp",
            "model_state_dict": {"weight": 1},
            "config": {"hidden_dim": 8},
        }
        self.assertEqual(
            unpack_checkpoint(modern, "our"),
            ("mlp", {"weight": 1}, {"hidden_dim": 8}),
        )

    def test_checkpoint_feature_mismatch_is_rejected(self):
        from types import SimpleNamespace

        payload = {"feature_config": {"audio_feature": "mfccs", "video_feature": "densenet"}}
        args = SimpleNamespace(
            audiofeature_method="wav2vec", videofeature_method="densenet",
            splitwindow_time="1s", labelcount=2,
        )
        with self.assertRaisesRegex(ValueError, "audio_feature"):
            test_module.validate_checkpoint_features(payload, args)

    def test_ensemble_weights_are_validated(self):
        with self.assertRaisesRegex(ValueError, "number"):
            test_module.normalize_model_weights("0.5", 2)
        with self.assertRaisesRegex(ValueError, "positive"):
            test_module.normalize_model_weights("0,0", 2)
        self.assertEqual(test_module.normalize_model_weights("1,3", 2), [0.25, 0.75])

    def test_our_model_ignores_personality_when_disabled(self):
        model = create_model(make_opt(model="our", isTrain=False, use_personality=False))
        model.eval()
        first = sample_batch()
        second = {key: value.clone() for key, value in first.items()}
        second["personalized_feat"] = torch.randn_like(first["personalized_feat"]) * 100
        model.set_input(first)
        model.forward()
        first_logits = model.emo_logits.detach().clone()
        model.set_input(second)
        model.forward()
        self.assertTrue(torch.allclose(first_logits, model.emo_logits, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
