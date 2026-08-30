"""Regression tests for checkpoint-reproducible Q-Former training options."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "06c_train_bridge.py"
sys.path.insert(0, str(ROOT))

from src.models.connectors import build_bridge  # noqa: E402


def qformer_config(**overrides):
    """Return a small valid Q-Former configuration for unit tests."""
    config = {
        "type": "qformer",
        "num_query_tokens": 4,
        "hidden_size": 24,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "drop_cls": False,
    }
    return {**config, **overrides}


def load_trainer_module():
    """Load the numerically prefixed trainer as a module."""
    spec = importlib.util.spec_from_file_location("train_bridge_t030", TRAINER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class QFormerModuleConfigurationTests(unittest.TestCase):
    """Verify dropout and pre-LN are reconstructed from bridge_config."""

    def test_legacy_config_preserves_locked_dropout_and_post_norm(self):
        bridge = build_bridge(qformer_config(), encoder_dim=16, llm_dim=32)
        layer = bridge.decoder.layers[0]
        dropouts = [module.p for module in layer.modules()
                    if isinstance(module, torch.nn.Dropout)]
        self.assertGreaterEqual(len(dropouts), 4)
        self.assertEqual(set(dropouts), {0.1})
        self.assertFalse(layer.norm_first)
        self.assertIsNone(bridge.decoder.norm)

    def test_explicit_zero_dropout_reaches_every_decoder_dropout(self):
        bridge = build_bridge(
            qformer_config(dropout=0.0), encoder_dim=16, llm_dim=32,
        )
        dropouts = [module.p for module in bridge.decoder.layers[0].modules()
                    if isinstance(module, torch.nn.Dropout)]
        self.assertGreaterEqual(len(dropouts), 4)
        self.assertEqual(set(dropouts), {0.0})

    def test_pre_norm_has_required_final_layer_norm(self):
        bridge = build_bridge(
            qformer_config(
                hidden_size=768,
                num_hidden_layers=1,
                num_attention_heads=12,
                norm_first=True,
            ),
            encoder_dim=16,
            llm_dim=32,
        )
        self.assertTrue(bridge.decoder.layers[0].norm_first)
        self.assertIsInstance(bridge.decoder.norm, torch.nn.LayerNorm)
        self.assertEqual(
            sum(parameter.numel() for parameter in bridge.decoder.norm.parameters()),
            1536,
        )

    def test_explicit_locked_defaults_are_state_dict_identical(self):
        torch.manual_seed(12345)
        legacy = build_bridge(qformer_config(), encoder_dim=16, llm_dim=32)
        torch.manual_seed(12345)
        explicit = build_bridge(
            qformer_config(dropout=0.1, norm_first=False),
            encoder_dim=16,
            llm_dim=32,
        )
        self.assertEqual(tuple(legacy.state_dict()), tuple(explicit.state_dict()))
        for key, tensor in legacy.state_dict().items():
            self.assertTrue(torch.equal(tensor, explicit.state_dict()[key]), key)

    def test_dropout_change_does_not_change_checkpoint_tensor_schema(self):
        torch.manual_seed(7)
        locked = build_bridge(
            qformer_config(dropout=0.1), encoder_dim=16, llm_dim=32,
        )
        torch.manual_seed(7)
        no_dropout = build_bridge(
            qformer_config(dropout=0.0), encoder_dim=16, llm_dim=32,
        )
        self.assertEqual(tuple(locked.state_dict()), tuple(no_dropout.state_dict()))
        for key, tensor in locked.state_dict().items():
            self.assertTrue(torch.equal(tensor, no_dropout.state_dict()[key]), key)


class TrainerQFormerOptionTests(unittest.TestCase):
    """Verify trainer defaults, validation, provenance, and CLI surface."""

    @classmethod
    def setUpClass(cls):
        cls.trainer = load_trainer_module()

    def test_fresh_qformer_options_are_recorded_with_legacy_defaults(self):
        resolved = self.trainer.resolve_qformer_bridge_config(
            qformer_config(), dropout=None, norm_first=None,
        )
        self.assertEqual(resolved["dropout"], 0.1)
        self.assertFalse(resolved["norm_first"])
        custom = self.trainer.resolve_qformer_bridge_config(
            qformer_config(), dropout=0.0, norm_first=True,
        )
        self.assertEqual(custom["dropout"], 0.0)
        self.assertTrue(custom["norm_first"])

    def test_invalid_or_non_qformer_options_fail_closed(self):
        with self.assertRaisesRegex(SystemExit, r"\[0, 1\)"):
            self.trainer.resolve_qformer_bridge_config(
                qformer_config(), dropout=1.0, norm_first=False,
            )
        with self.assertRaisesRegex(SystemExit, "Q-Former-only"):
            self.trainer.resolve_qformer_bridge_config(
                {"type": "mlp"}, dropout=0.0, norm_first=None,
            )

    def test_resume_options_default_legacy_and_reject_conflicts(self):
        legacy = qformer_config()
        self.trainer.validate_qformer_resume_options(
            legacy, requested_dropout=None, requested_norm_first=None,
        )
        with self.assertRaisesRegex(SystemExit, "qformer-dropout"):
            self.trainer.validate_qformer_resume_options(
                {**legacy, "dropout": 0.0},
                requested_dropout=0.1,
                requested_norm_first=None,
            )
        with self.assertRaisesRegex(SystemExit, "qformer-norm-first"):
            self.trainer.validate_qformer_resume_options(
                {**legacy, "norm_first": True},
                requested_dropout=None,
                requested_norm_first=False,
            )

    def test_trainer_help_exposes_both_qformer_flags(self):
        result = subprocess.run(
            [sys.executable, str(TRAINER), "--help"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--qformer-dropout", result.stdout)
        self.assertIn("--qformer-norm-first", result.stdout)
        self.assertIn("--no-qformer-norm-first", result.stdout)


if __name__ == "__main__":
    unittest.main()
