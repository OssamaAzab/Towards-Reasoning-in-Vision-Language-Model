"""Unit tests for bridge-training architecture and continuation provenance."""
import importlib.util
import random
import tempfile
import unittest
from pathlib import Path

import torch


def load_trainer_module():
    """Load the numerically prefixed trainer as a module."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "06c_train_bridge.py"
    spec = importlib.util.spec_from_file_location("train_bridge", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeBridge(torch.nn.Module):
    """Small bridge stand-in with a predictable parameter count."""

    def __init__(self, parameter_count):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(parameter_count))


class TrainBridgeProvenanceTests(unittest.TestCase):
    """Exercise the pure pre-launch and checkpoint-provenance guards."""

    @classmethod
    def setUpClass(cls):
        cls.trainer = load_trainer_module()

    def test_mlp_width_must_be_explicit_and_match_encoder_reference(self):
        with self.assertRaisesRegex(SystemExit, "--mlp-hidden"):
            self.trainer.resolve_mlp_hidden("mlp", "clip", None)
        with self.assertRaisesRegex(SystemExit, "5888"):
            self.trainer.resolve_mlp_hidden("mlp", "clip", 4096)
        self.assertEqual(self.trainer.resolve_mlp_hidden("mlp", "clip", 5888), 5888)
        self.assertEqual(self.trainer.resolve_mlp_hidden("mlp", "ijepa", 5700), 5700)
        self.assertEqual(self.trainer.resolve_mlp_hidden("mlp", "dinov2", 5888), 5888)
        with self.assertRaisesRegex(SystemExit, "5888"):
            self.trainer.resolve_mlp_hidden("mlp", "dinov2", 4096)

    def test_strip_cls_accepts_patch_encoders_only(self):
        """CLS stripping is valid for CLIP and DINOv2 but not I-JEPA."""
        self.trainer.validate_strip_cls("clip", True)
        self.trainer.validate_strip_cls("dinov2", True)
        self.trainer.validate_strip_cls("ijepa", False)
        with self.assertRaisesRegex(SystemExit, "CLIP and DINOv2"):
            self.trainer.validate_strip_cls("ijepa", True)

    def test_non_mlp_connector_keeps_existing_default(self):
        self.assertEqual(self.trainer.resolve_mlp_hidden("pool", "clip", None), 4096)
        self.assertEqual(self.trainer.resolve_mlp_hidden("qformer", "clip", None), 4096)

    def test_reference_guard_checks_architecture_and_parameter_count(self):
        reference = {
            "bridge_config": {
                "type": "mlp",
                "drop_cls": True,
                "mlp_hidden": 5888,
                "mlp_depth": 3,
            },
            "state_dict": {"weight": torch.zeros(7)},
        }
        config = dict(reference["bridge_config"])
        self.assertEqual(
            self.trainer.assert_mlp_reference_match(FakeBridge(7), config, reference),
            7,
        )
        with self.assertRaisesRegex(SystemExit, "parameter count"):
            self.trainer.assert_mlp_reference_match(FakeBridge(8), config, reference)
        wrong_config = {**config, "mlp_hidden": 4096}
        with self.assertRaisesRegex(SystemExit, "mlp_hidden"):
            self.trainer.assert_mlp_reference_match(FakeBridge(7), wrong_config, reference)

    def test_missing_mlp_reference_requires_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge_dinov2_150k_mlp_ep3.pt"
            config = {
                "type": "mlp",
                "drop_cls": False,
                "mlp_hidden": 5888,
                "mlp_depth": 3,
            }
            bridge = FakeBridge(7)
            with self.assertRaisesRegex(SystemExit, "reference checkpoint not found"):
                self.trainer.validate_mlp_reference(
                    bridge, config, path, allow_missing=False
                )
            provenance = self.trainer.validate_mlp_reference(
                bridge, config, path, allow_missing=True
            )
        self.assertEqual(provenance["status"], "missing-opt-in")
        self.assertEqual(provenance["parameter_count"], 7)
        self.assertIsNone(provenance["sha256"])

    def test_opt_in_does_not_bypass_existing_reference_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge_dinov2_150k_mlp_ep3.pt"
            reference = {
                "bridge_config": {
                    "type": "mlp",
                    "drop_cls": False,
                    "mlp_hidden": 5888,
                    "mlp_depth": 3,
                },
                "state_dict": {"weight": torch.zeros(7)},
            }
            torch.save(reference, path)
            with self.assertRaisesRegex(SystemExit, "parameter count"):
                self.trainer.validate_mlp_reference(
                    FakeBridge(8),
                    reference["bridge_config"],
                    path,
                    allow_missing=True,
                )

    def test_trajectory_fields_are_required_and_must_match(self):
        current = {
            "lr": 1e-4,
            "weight_decay": 0.01,
            "batch_size": 8,
            "grad_accum": 1,
            "max_answer_tokens": 64,
            "warmup_frac": 0.05,
            "seed": 42,
            "loss_norm": "token",
        }
        checkpoint = dict(current)
        self.assertEqual(
            self.trainer.validate_trajectory_provenance(
                current, checkpoint, acknowledge_missing=False
            ),
            (),
        )
        checkpoint["batch_size"] = 4
        with self.assertRaisesRegex(SystemExit, "batch-size"):
            self.trainer.validate_trajectory_provenance(
                current, checkpoint, acknowledge_missing=False
            )
        del checkpoint["batch_size"]
        with self.assertRaisesRegex(SystemExit, "missing provenance"):
            self.trainer.validate_trajectory_provenance(
                current, checkpoint, acknowledge_missing=False
            )
        self.assertEqual(
            self.trainer.validate_trajectory_provenance(
                current, checkpoint, acknowledge_missing=True
            ),
            ("batch_size",),
        )

    def test_same_experiment_resume_cannot_change_target(self):
        checkpoint = {"epoch": 2, "tag": "500k_v1", "target_epochs": 6}
        result = self.trainer.validate_epoch_operation(
            checkpoint,
            target_epochs=6,
            requested_tag=None,
            continuation=False,
            lr_schedule="flat",
            acknowledge_missing=False,
        )
        self.assertEqual(result["original_target_epochs"], 6)
        with self.assertRaisesRegex(SystemExit, "target --epochs"):
            self.trainer.validate_epoch_operation(
                checkpoint,
                target_epochs=7,
                requested_tag=None,
                continuation=False,
                lr_schedule="flat",
                acknowledge_missing=False,
            )

    def test_same_experiment_resume_requires_parent_tag_even_with_ack(self):
        checkpoint = {"epoch": 2, "target_epochs": 3}
        with self.assertRaisesRegex(SystemExit, "parent tag"):
            self.trainer.validate_epoch_operation(
                checkpoint,
                target_epochs=3,
                requested_tag=None,
                continuation=False,
                lr_schedule="flat",
                acknowledge_missing=True,
            )

    def test_continuation_requires_new_tag_flat_lr_and_larger_target(self):
        checkpoint = {"epoch": 3, "tag": "150k_mlp", "target_epochs": 3}
        result = self.trainer.validate_epoch_operation(
            checkpoint,
            target_epochs=5,
            requested_tag="150k_mlp_cont5_s42_v1",
            continuation=True,
            lr_schedule="flat",
            acknowledge_missing=False,
        )
        self.assertEqual(result["original_target_epochs"], 3)
        self.assertEqual(result["extended_target_epochs"], 5)
        with self.assertRaisesRegex(SystemExit, "new --ckpt-tag"):
            self.trainer.validate_epoch_operation(
                checkpoint, 5, "150k_mlp", True, "flat", False
            )
        with self.assertRaisesRegex(SystemExit, "flat"):
            self.trainer.validate_epoch_operation(
                checkpoint, 5, "new_tag", True, "cosine", False
            )
        with self.assertRaisesRegex(SystemExit, "greater"):
            self.trainer.validate_epoch_operation(
                checkpoint, 3, "new_tag", True, "flat", False
            )

    def test_legacy_target_requires_acknowledgement(self):
        checkpoint = {"epoch": 3, "tag": "150k_mlp"}
        with self.assertRaisesRegex(SystemExit, "missing provenance"):
            self.trainer.validate_epoch_operation(
                checkpoint, 5, "new_tag", True, "flat", False
            )
        result = self.trainer.validate_epoch_operation(
            checkpoint, 5, "new_tag", True, "flat", True
        )
        self.assertEqual(result["original_target_epochs"], 3)
        self.assertEqual(result["missing_provenance"], ("target_epochs",))

    def test_rng_state_round_trip_restores_python_and_torch_cpu(self):
        random.seed(17)
        torch.manual_seed(17)
        state = self.trainer.capture_rng_state(include_cuda=False)
        expected_python = random.random()
        expected_torch = torch.rand(3)
        random.random()
        torch.rand(3)
        self.trainer.restore_rng_state(state, include_cuda=False)
        self.assertEqual(random.random(), expected_python)
        self.assertTrue(torch.equal(torch.rand(3), expected_torch))

    def test_file_sha256_uses_file_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "parent.pt"
            path.write_bytes(b"parent checkpoint")
            self.assertEqual(
                self.trainer.file_sha256(path),
                "36936fd21105c113fd28e60e907604234bb76376e28465251a724b38e947c392",
            )


if __name__ == "__main__":
    unittest.main()
