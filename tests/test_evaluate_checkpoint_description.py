"""Tests for the evaluation run's human-readable provenance line.

Job 2285362 (Wave 1, I-JEPA, MLP, bf16) logged every one of its five per-epoch evaluations
as "32 query tokens, 4-bit". The checkpoint on disk was correct throughout — type=mlp,
mlp_hidden=5700, llm_precision=bf16 — so nothing computed was wrong. Only the log was, and
the log is the artifact a reader audits when checking what a number was produced by.

Two independent causes: `num_query_tokens` is a config default that survives into MLP
checkpoints where it is meaningless, and the precision was a 4-bit/8-bit boolean that
predated bf16 and fp16 support.
"""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.report import describe_checkpoint  # noqa: E402


class DescribeCheckpointTests(unittest.TestCase):
    """The line must report the connector and precision actually loaded."""

    @staticmethod
    def describe(*args):
        """Call the shared helper (both eval scripts render this line through it)."""
        return describe_checkpoint(*args)

    def test_mlp_is_not_described_as_having_query_tokens(self):
        """The exact defect from job 2285362's five evaluation blocks."""
        cfg = {"type": "mlp", "num_query_tokens": 32, "mlp_hidden": 5700, "mlp_depth": 3}
        out = self.describe({"encoder": "ijepa"}, cfg, "bf16", False)
        self.assertNotIn("query tokens", out)
        self.assertIn("mlp", out)
        self.assertIn("all patch tokens forwarded", out)

    def test_bf16_is_not_described_as_4bit(self):
        """The precision predicate predated bf16; every bf16 run logged itself as 4-bit."""
        cfg = {"type": "mlp", "num_query_tokens": 32}
        out = self.describe({"encoder": "ijepa"}, cfg, "bf16", False)
        self.assertIn("LLM bf16", out)
        self.assertNotIn("4-bit", out)

    def test_every_supported_precision_is_reported_verbatim(self):
        cfg = {"type": "qformer", "num_query_tokens": 32}
        for precision in ("4bit", "8bit", "bf16", "fp16"):
            out = self.describe({"encoder": "clip"}, cfg, precision, False)
            self.assertIn(f"LLM {precision}", out)

    def test_qformer_still_reports_its_query_tokens(self):
        """The compressing connectors are the ones for which the count is meaningful."""
        cfg = {"type": "qformer", "num_query_tokens": 32}
        out = self.describe({"encoder": "clip"}, cfg, "bf16", False)
        self.assertIn("32 query tokens", out)

    def test_pool_reports_query_tokens_too(self):
        cfg = {"type": "pool", "num_query_tokens": 32}
        out = self.describe({"encoder": "clip"}, cfg, "bf16", False)
        self.assertIn("32 query tokens", out)

    def test_missing_type_defaults_to_qformer_for_legacy_checkpoints(self):
        """Pre-B2 checkpoints have no 'type' key and were all Q-Formers."""
        out = self.describe({"encoder": "ijepa"}, {"num_query_tokens": 32}, "4bit", False)
        self.assertIn("32 query tokens", out)

    def test_cls_stripping_is_reported_only_when_it_happened(self):
        cfg = {"type": "mlp", "num_query_tokens": 32}
        self.assertIn("[CLS] STRIPPED", self.describe({}, cfg, "bf16", True))
        self.assertNotIn("[CLS] STRIPPED", self.describe({}, cfg, "bf16", False))

    def test_the_encoder_is_always_named(self):
        cfg = {"type": "mlp", "num_query_tokens": 32}
        self.assertIn("facebook/ijepa_vith14_1k",
                      self.describe({"encoder": "facebook/ijepa_vith14_1k"}, cfg, "bf16", False))


class BothEvalScriptsUseTheHelperTests(unittest.TestCase):
    """A fix applied to one evaluator and not the other is half a fix."""

    def test_script_07_renders_the_line_through_the_helper(self):
        text = (ROOT / "scripts" / "07_evaluate.py").read_text(encoding="utf-8")
        self.assertIn("from src.eval.report import describe_checkpoint", text)
        self.assertIn("describe_checkpoint(ckpt, bcfg, llm_precision, bridge.drop_cls)", text)

    def test_script_09_renders_the_line_through_the_helper(self):
        text = (ROOT / "scripts" / "09_augment_eval.py").read_text(encoding="utf-8")
        self.assertIn("from src.eval.report import describe_checkpoint", text)
        self.assertIn("describe_checkpoint(ckpt, bcfg, llm_precision, bridge.drop_cls)", text)

    def test_neither_script_still_hardcodes_the_precision_binary(self):
        """`'8-bit' if eight_bit else '4-bit'` is what logged bf16 runs as 4-bit."""
        for name in ("07_evaluate.py", "09_augment_eval.py"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertNotIn("""'8-bit' if eight_bit else '4-bit'""", text, name)


if __name__ == "__main__":
    unittest.main()
