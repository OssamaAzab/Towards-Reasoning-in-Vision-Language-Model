"""Regression tests for result-stem lineage labels."""
import importlib.util
import unittest
from pathlib import Path


def load_collect_results():
    """Load the numerically prefixed collector as a module."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "10_collect_results.py"
    spec = importlib.util.spec_from_file_location("collect_results", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DescribeContinuationTests(unittest.TestCase):
    """Keep E1 continuation rows distinct from locked MLP rows."""

    @classmethod
    def setUpClass(cls):
        cls.collect_results = load_collect_results()

    def test_clip_continuation_has_distinct_encoder_label(self):
        resolved = self.collect_results.describe(
            "bridge_clip_150k_mlp_cont5_s42_v1_ep5"
        )
        self.assertEqual(resolved, ("clip_150k_mlp_cont5", 5, "none"))

    def test_ijepa_continuation_has_distinct_encoder_label(self):
        resolved = self.collect_results.describe(
            "bridge_ijepa_150k_mlp_cont5_s42_v1_ep4"
        )
        self.assertEqual(resolved, ("ijepa_150k_mlp_cont5", 4, "none"))


class DescribeCapacityVariantTests(unittest.TestCase):
    """Keep Q-Former capacity variants distinct and reject silent lineage loss."""

    @classmethod
    def setUpClass(cls):
        cls.collect_results = load_collect_results()

    def test_capacity_variants_have_distinct_encoder_labels(self):
        expected = {
            "bridge_clip_150k_q64_ep3": ("clip_150k_q64", 3, "none"),
            "bridge_clip_150k_q128_ep3": ("clip_150k_q128", 3, "none"),
            "bridge_clip_150k_l8_ep3": ("clip_150k_l8", 3, "none"),
            "bridge_clip_150k_q64l8_ep3": ("clip_150k_q64l8", 3, "none"),
            "bridge_ijepa_150k_q128_ep2": ("ijepa_150k_q128", 2, "none"),
        }
        for stem, resolved in expected.items():
            with self.subTest(stem=stem):
                self.assertEqual(self.collect_results.describe(stem), resolved)

    def test_unknown_bridge_variant_raises(self):
        with self.assertRaisesRegex(ValueError, "unrecognised bridge stem token.*mystery"):
            self.collect_results.describe("bridge_clip_150k_mystery_ep3")

    def test_known_multitoken_variants_remain_valid(self):
        expected = {
            "bridge_clip_150k_3ep_lora_attn": ("clip_150k_lora_attn", 3, "none"),
            "bridge_clip_150k_mlp_ep3_lora_attn": (
                "clip_150k_lora_attn_mlp",
                3,
                "none",
            ),
            "bridge_clip_150k_mlp_cont5_s42_v1_ep5": (
                "clip_150k_mlp_cont5",
                5,
                "none",
            ),
        }
        for stem, resolved in expected.items():
            with self.subTest(stem=stem):
                self.assertEqual(self.collect_results.describe(stem), resolved)


class DescribeScaleAndWaveTests(unittest.TestCase):
    """Keep A/T/B cells distinct by data scale and preregistered wave token."""

    @classmethod
    def setUpClass(cls):
        cls.collect_results = load_collect_results()

    def test_scale_is_part_of_identity(self):
        expected = {
            "bridge_clip_150k_mlp_tuned_ep3": ("clip_150k_mlp_tuned", 3, "none"),
            "bridge_clip_500k_mlp_tuned_ep3": ("clip_500k_mlp_tuned", 3, "none"),
        }
        for stem, resolved in expected.items():
            with self.subTest(stem=stem):
                self.assertEqual(self.collect_results.describe(stem), resolved)

    def test_anchor_and_tuned_tokens_are_registered(self):
        expected = {
            "bridge_clip_150k_mlp_anchor_ep3": ("clip_150k_mlp_anchor", 3, "none"),
            "bridge_ijepa_150k_mlp_tuned_ep3": ("ijepa_150k_mlp_tuned", 3, "none"),
        }
        for stem, resolved in expected.items():
            with self.subTest(stem=stem):
                self.assertEqual(self.collect_results.describe(stem), resolved)


class DescribeEncoderTests(unittest.TestCase):
    """Keep newly approved encoder families distinct from training variants."""

    @classmethod
    def setUpClass(cls):
        cls.collect_results = load_collect_results()

    def test_dinov2_is_a_registered_base_encoder(self):
        expected = {
            "bridge_dinov2_150k_mlp_ep3": ("dinov2_150k_mlp", 3, "none"),
            "bridge_dinov2_150k_3ep": ("dinov2_150k", 3, "none"),
        }
        for stem, resolved in expected.items():
            with self.subTest(stem=stem):
                self.assertEqual(self.collect_results.describe(stem), resolved)


if __name__ == "__main__":
    unittest.main()
