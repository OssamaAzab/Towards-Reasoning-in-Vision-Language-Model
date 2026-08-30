"""Tests for the central prompt builder, the corrected metric and the artifact guard.

The most important test here is test_raw_format_is_bit_identical_to_legacy: it is the
gate that proves introducing the prompt module changed nothing for the 56 existing
checkpoints. Everything else is only safe because that one passes.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import prompt as P  # noqa: E402
from src.artifact import artifact_stem, write_json, write_text  # noqa: E402
from src.eval.metrics import exact_full, normalize_answer, normalize_full  # noqa: E402
from src.utils import load_config  # noqa: E402


def _tokenizer():
    """Load the project's tokenizer once (CPU only, from the local HF cache)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(load_config()["models"]["llm"])


class PromptFormatTests(unittest.TestCase):
    """Raw parity, ChatML splice equivalence, and the supervised span."""

    @classmethod
    def setUpClass(cls):
        cls.tok = _tokenizer()

    def test_raw_format_is_bit_identical_to_legacy(self):
        # Legacy: [visual | tok(question + cue) | tok(answer, truncated)], labels
        # masking exactly n_visual + len(question). Reproduced for 12 pairs.
        pairs = [("Is the sky blue?", "yes"), ("What colour is the bus?", "red"),
                 ("Are there two cats?", "no"), ("What is the man holding?", "an umbrella")]
        spec = P.PromptSpec()  # defaults: raw, supervise_eos False
        for q, a in pairs:
            for n_vis in (32, 256, 257):
                built = P.build(spec, q, self.tok, answer=a, n_visual=n_vis)
                legacy_q = self.tok(q + P.SHORT_CUE).input_ids
                legacy_a = self.tok(a, add_special_tokens=False, truncation=True,
                                    max_length=64).input_ids
                self.assertEqual(built.body_ids, legacy_q)
                self.assertEqual(built.target_ids, legacy_a)
                self.assertEqual(built.prefix_ids, [])   # visual comes first in legacy
                self.assertEqual(built.mid_ids, [])
                self.assertEqual(built.n_prefix, n_vis + len(legacy_q))

    def test_chatml_manual_splice_equals_apply_chat_template_minus_newline(self):
        spec = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True)
        for q in ["Is the sky blue?", "What colour is the bus?", "Are there two cats?"]:
            for a in ["yes", "red", "a small brown dog"]:
                built = P.build(spec, q, self.tok, answer=a, n_visual=0)
                manual = built.prompt_ids() + built.target_ids
                template = self.tok.apply_chat_template(
                    [{"role": "system", "content": P.SYSTEM_TEXT},
                     {"role": "user", "content": q + P.SHORT_CUE},
                     {"role": "assistant", "content": a}], tokenize=True)
                self.assertEqual(manual, template[:-1])
                self.assertEqual(template[-1], P.NEWLINE)

    def test_default_system_string_never_appears(self):
        spec = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True)
        built = P.build(spec, "Is the sky blue?", self.tok, answer="yes")
        text = self.tok.decode(built.prompt_ids())
        self.assertNotIn("You are a helpful assistant.", text)
        self.assertIn(P.SYSTEM_TEXT, text)

    def test_system_turn_can_be_omitted_entirely(self):
        # apply_chat_template cannot express "no system turn"; the module can.
        spec = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True, system_text=None)
        built = P.build(spec, "Is the sky blue?", self.tok, answer="yes")
        self.assertNotIn("system", self.tok.decode(built.prompt_ids()))

    def test_text_only_prompt_is_image_prompt_without_the_visual_span(self):
        q = "Is the sky blue?"
        img = P.build(P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True),
                      q, self.tok, answer="yes", n_visual=256)
        txt = P.build(P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True,
                                   input_mode="text_only"), q, self.tok, answer="yes")
        self.assertEqual(img.prompt_ids(), txt.prompt_ids())
        self.assertEqual(img.n_prefix - 256, txt.n_prefix)

    def test_graph_only_body_is_identical_to_image_plus_graph_body(self):
        # Load-bearing: the ONLY difference between the two arms must be the visual
        # span, so image_plus_graph - graph_only is a clean image effect at fixed text.
        ctx, q = "a bus left of a tree; a person above a bench", "Is the sky blue?"
        kw = dict(prompt_format="chatml_v1", supervise_eos=True)
        g = P.build(P.PromptSpec(input_mode="graph_only", **kw), q, self.tok, context=ctx)
        ig = P.build(P.PromptSpec(input_mode="image_plus_graph", **kw), q, self.tok,
                     context=ctx, n_visual=256)
        self.assertEqual(g.body_ids, ig.body_ids)


class EosSupervisionTests(unittest.TestCase):
    """The conditional-EOS rule and the trailing-newline omission."""

    @classmethod
    def setUpClass(cls):
        cls.tok = _tokenizer()
        cls.spec = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True)

    def test_short_answer_is_supervised_with_im_end(self):
        built = P.build(self.spec, "Is the sky blue?", self.tok, answer="yes")
        self.assertEqual(built.target_ids[-1], P.IM_END)
        self.assertTrue(built.meta["eos_appended"])

    def test_truncated_answer_gets_no_eos(self):
        # Appending unconditionally would supervise "stop" at an arbitrary
        # mid-sentence cut; the guard makes truncation and EOS mutually exclusive.
        built = P.build(self.spec, "Describe it.", self.tok, answer=" ".join(["word"] * 200))
        self.assertEqual(len(built.target_ids), 64)
        self.assertNotEqual(built.target_ids[-1], P.IM_END)
        self.assertTrue(built.meta["answer_truncated"])
        self.assertFalse(built.meta["eos_appended"])

    def test_trailing_newline_is_never_supervised(self):
        built = P.build(self.spec, "Is the sky blue?", self.tok, answer="yes")
        self.assertNotIn(P.NEWLINE, built.target_ids)

    def test_supervise_eos_false_reproduces_the_legacy_target(self):
        spec = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=False)
        built = P.build(spec, "Is the sky blue?", self.tok, answer="yes")
        self.assertEqual(built.target_ids,
                         self.tok("yes", add_special_tokens=False).input_ids)

    def test_generation_stop_set_is_the_supervised_token_only(self):
        # generation_config ships [151645, 151643]; only 151645 is ever supervised,
        # so allowing 151643 would make "the bridge learned to stop" unfalsifiable.
        kw = P.generation_kwargs(self.spec, "direct")
        self.assertEqual(kw["eos_token_id"], P.IM_END)
        self.assertEqual(kw["pad_token_id"], P.ENDOFTEXT)
        self.assertEqual(kw["max_new_tokens"], 32)
        self.assertFalse(kw["do_sample"])


class GuardTests(unittest.TestCase):
    """Construction-time and body-safety guards."""

    def test_raw_format_cannot_supervise_eos(self):
        with self.assertRaisesRegex(ValueError, "chatml_v1"):
            P.PromptSpec(supervise_eos=True)

    def test_body_starting_with_any_linebreak_is_rejected(self):
        for bad in ["\nQ", "\n\nQ", " \nQ", "\t\nQ", "\r\nQ"]:
            with self.assertRaises(ValueError):
                P.assert_body_safe(bad)

    def test_safe_bodies_are_accepted(self):
        for good in [" Is the sky blue?", "Is the sky blue?", "a\n\nb"]:
            P.assert_body_safe(good)

    def test_evidence_layer_classification(self):
        self.assertEqual(P.PromptSpec().evidence_layer, P.EVIDENCE_LEGACY)
        self.assertEqual(
            P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True).evidence_layer,
            P.EVIDENCE_CORRECTED)

    def test_checkpoint_without_prompt_format_is_treated_as_legacy(self):
        self.assertEqual(P.format_of({"encoder": "x"}), P.PROMPT_FORMAT_RAW)
        self.assertEqual(P.evidence_layer_of({"encoder": "x"}), P.EVIDENCE_LEGACY)

    def test_protocol_mismatch_is_refused_but_overridable(self):
        legacy_ckpt = {"encoder": "x"}
        corrected = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True)
        with self.assertRaisesRegex(SystemExit, "protocol mismatch"):
            P.assert_compatible(legacy_ckpt, corrected)
        P.assert_compatible(legacy_ckpt, corrected, allow_mismatch=True)
        P.assert_compatible(legacy_ckpt, P.PromptSpec())     # matching pair passes


class MetricTests(unittest.TestCase):
    """The corrected primary metric, and the defect it replaces."""

    def test_exact_full_does_not_truncate_at_the_first_line(self):
        # normalize_answer keeps only line 1; normalize_full keeps everything. This
        # asymmetry is what handed the multiline text-only floor +13.6 points.
        multiline = "yes\nthere are two birds"
        self.assertEqual(normalize_answer(multiline), "yes")
        self.assertNotEqual(normalize_full(multiline), "yes")
        self.assertFalse(exact_full(multiline, "yes"))

    def test_exact_full_is_symmetric_in_normalisation(self):
        self.assertTrue(exact_full("The Yes.", "yes"))
        self.assertTrue(exact_full("yes", "  YES  "))

    def test_exact_full_ignores_articles_case_and_punctuation(self):
        self.assertTrue(exact_full("A Red Bus!", "red bus"))
        self.assertFalse(exact_full("red bus and a car", "red bus"))

    def test_metric_version_is_pinned(self):
        from src.eval.metrics import METRIC_VERSION
        self.assertEqual(METRIC_VERSION, "exact_full/1.0.0")


class ArtifactGuardTests(unittest.TestCase):
    """The no-overwrite guard and protocol-aware naming."""

    def test_stem_separates_every_condition(self):
        a = artifact_stem("ck", evidence_layer="corrected_chatml_v1_eos", split="tune500",
                          input_mode="image_only", reason_mode="direct")
        b = artifact_stem("ck", evidence_layer="corrected_chatml_v1_eos", split="tune500",
                          input_mode="image_plus_graph", reason_mode="direct")
        c = artifact_stem("ck", evidence_layer="legacy_raw_prompt_no_eos", split="tune500",
                          input_mode="image_only", reason_mode="direct")
        self.assertEqual(len({a, b, c}), 3)

    def test_write_refuses_to_overwrite_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            write_json(p, [{"qid": 1}], meta={"evidence_layer": "x"})
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                write_json(p, [{"qid": 2}], meta={"evidence_layer": "x"})
            write_json(p, [{"qid": 2}], meta={"evidence_layer": "x"}, overwrite=True)
            self.assertEqual(json.loads(p.read_text())["records"][0]["qid"], 2)

    def test_metadata_is_embedded_in_the_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            write_json(p, [{"qid": 1}], meta={"evidence_layer": "corrected_chatml_v1_eos",
                                              "input_mode": "graph_only"})
            meta = json.loads(p.read_text())["_meta"]
            self.assertEqual(meta["input_mode"], "graph_only")

    def test_markdown_writer_also_refuses_to_overwrite(self):
        """Both evaluators write their results table through write_text; it must guard too."""
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.md"
            write_text(p, "first", meta={"evidence_layer": "corrected_chatml_v1_eos"})
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                write_text(p, "second", meta={"evidence_layer": "corrected_chatml_v1_eos"})
            self.assertIn("first", p.read_text())

    def test_markdown_banner_states_which_layer_the_table_belongs_to(self):
        """A reader must not have to guess whether a table is legacy or corrected."""
        with tempfile.TemporaryDirectory() as tmp:
            corrected, legacy = Path(tmp) / "c.md", Path(tmp) / "l.md"
            write_text(corrected, "x", meta={"evidence_layer": "corrected_chatml_v1_eos"})
            write_text(legacy, "x", meta={"evidence_layer": "legacy_raw_prompt_no_eos"})
            self.assertIn("primary protocol", corrected.read_text())
            self.assertIn("exploratory only", legacy.read_text())


if __name__ == "__main__":
    unittest.main()
