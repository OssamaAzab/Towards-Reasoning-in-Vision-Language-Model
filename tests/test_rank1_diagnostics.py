"""Contract tests for the Rank 1 diagnostics (scripts 34 and 35).

Script 34 decomposes the training loss; script 35 partitions eval questions by the
floor/bridge correctness pair. Neither trains, and neither may become a selection surface.

The specific hazards:
  - script 34 reporting a quantity that does NOT decompose the trainer scalar while
    printing plausible per-source percentages (this was a real defect, caught in review)
  - either script naming a latent property from one observed outcome
  - the two data sources being swapped, which would inverT the headline conclusion
  - script 35 comparing arms whose floors differ, which makes the partition incomparable
  - either script being read as confirmatory rather than exploratory
"""
from pathlib import Path
import ast
import re
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NLL = ROOT / "scripts" / "34_nll_decomposition.py"
VIS = ROOT / "scripts" / "35_floor_partition.py"


def _strip_comments(text):
    """Source with comment lines removed, so a test cannot pass by matching prose."""
    return "\n".join(l for l in text.splitlines() if not l.strip().startswith("#"))


class NllDecompositionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.text = NLL.read_text(encoding="utf-8")
        cls.code = _strip_comments(cls.text)

    def test_parses(self):
        ast.parse(self.text)

    def test_reproduces_the_trainer_scalar_as_a_control(self):
        """Without this the decomposition could describe a quantity nobody optimised."""
        self.assertIn("trainer.eval_loss(", self.code)
        self.assertRegex(self.code, r"delta\s*>\s*1e-4")
        self.assertIn("does not reproduce eval_loss", self.text)

    def test_the_control_can_fail(self):
        """Both sides of the control must be computed, or it compares a value to itself."""
        self.assertRegex(self.code, r"delta\s*=\s*abs\(reported\s*-\s*hf_agg\)")
        # reported comes from OUR per-token CE; hf_agg from HuggingFace's own reduction.
        self.assertRegex(self.code, r"reported\s*=\s*\(sum\(mine")
        self.assertRegex(self.code, r"hf_agg\s*=\s*\(sum\(s\s*\*\s*n")
        self.assertIn("per_batch_max", self.code)

    def test_refuses_a_legacy_layer_checkpoint(self):
        self.assertIn('spec.prompt_format != "chatml_v1"', self.code)
        self.assertIn("not a corrected-layer checkpoint", self.text)

    def test_cross_checks_source_labelling(self):
        """Swapping llava/vqav2 would invert the headline; a positional guess is not enough."""
        self.assertIn("SHORT_SUFFIX", self.code)
        self.assertIn("source labelling disagrees", self.text)

    def test_takes_the_mixture_from_the_checkpoint_not_config(self):
        for key in ('from_ckpt("limit")', 'from_ckpt("short_frac")', 'from_ckpt("val_frac")'):
            self.assertIn(key, self.code)
        # from_ckpt must consult the checkpoint FIRST; config is only the fallback.
        body = self.code[self.code.index("def from_ckpt"):]
        body = body[:body.index("limit = from_ckpt")]
        self.assertLess(body.index("ckpt.get(key)"), body.index("in tcfg"))

    def test_reproduces_the_trainers_val_split_not_a_tail_slice(self):
        """A tail slice of build_examples is 100% VQAv2 — a vacuous source decomposition."""
        self.assertIn("zlib.crc32", self.code)
        self.assertIn("canonical_dir_map", self.code)
        self.assertIn("split_key", self.code)
        self.assertIn("single-source", self.text)

    def test_separates_the_three_share_quantities(self):
        """Conflating these was the review stop: only one decomposes the objective."""
        for key in ("token_share_pct", "global_token_loss_share_pct",
                    "trainer_additive_contribution"):
            self.assertIn(key, self.code)

    def test_the_additive_decomposition_is_checked_to_sum(self):
        """A 'decomposition' whose parts do not sum to the whole is not one."""
        self.assertIn("is not an additive decomposition", self.text)
        self.assertRegex(self.code, r"abs\(parts - reported\)\s*>\s*1e-6")

    def test_contribution_and_share_are_documented_as_different(self):
        """A contribution is in loss units; a share is that over the whole. Quoting a
        percentage against the contribution formula is a units error, and it shipped once."""
        flat = re.sub(r"\s+", " ", self.text)
        self.assertIn("trainer_additive_contribution = sum_b n_b*(CE_{b,s}/T_b) / sum_b n_b", flat)
        self.assertIn("100 * sum_b n_b*(CE_{b,s}/T_b) / sum_b n_b*(CE_b/T_b)", flat)
        self.assertIn("A percentage must never be quoted against the contribution formula", flat)

    def test_launcher_does_not_claim_positional_importance(self):
        """The script isolates two positions; it does not measure which positions matter."""
        sb = (ROOT / "scripts" / "cluster" / "36_nll_decomposition.sbatch")
        if not sb.is_file():
            self.skipTest("launcher not present")
        # Strip leading '#' before flattening: a phrase wrapped across two comment lines
        # otherwise reads as "... importance # was measured" and no assertion can match it.
        body = "\n".join(re.sub(r"^\s*#\s?", "", l) for l in sb.read_text(encoding="utf-8").splitlines())
        flat = re.sub(r"\s+", " ", body)
        for banned in ("positions that matter", "CE actually contributed",
                       "drives the objective", "THIS IS A GATE"):
            self.assertNotIn(banned, flat, f"launcher still claims: {banned!r}")
        self.assertIn("NOT because positional importance was measured", flat)

    def test_records_batch_identity(self):
        """Without n_b and T_b the objective-additive share cannot be reconstructed."""
        for key in ('"batch"', '"batch_n"', '"batch_tokens"'):
            self.assertIn(key, self.code)

    def test_disclaims_gradient_share_explicitly(self):
        flat = re.sub(r"\s+", " ", self.text)
        self.assertIn("None of the three is gradient-norm share", flat)
        self.assertIn("CANNOT on its own promote or reject", flat)

    def test_asserts_prompt_provenance_against_the_checkpoint(self):
        """Both loss controls share one spec, so only the checkpoint can falsify it."""
        self.assertIn("P.assert_compatible(ckpt, spec)", self.code)
        self.assertIn("short_cue_sha256", self.code)
        self.assertIn("prompt_module_version", self.code)

    def test_eos_ce_only_when_eos_was_appended(self):
        """On a truncated answer P.build appends no EOS; the last token is ordinary text."""
        self.assertIn('meta.get("eos_appended")', self.code)
        self.assertIn('meta.get("answer_truncated")', self.code)

    def test_optional_fields_are_aggregated_none_safely(self):
        """Making eos_ce nullable crashed all 16 cells: the aggregator still summed it.

        Any per-record field that can be None must be filtered before arithmetic, and the
        count of contributing records reported so the mean is interpretable.
        """
        self.assertIn('r["eos_ce"] for r in sel if r["eos_ce"] is not None', self.code)
        self.assertIn('"n_with_eos"', self.code)
        self.assertNotRegex(self.code, r'sum\(r\["eos_ce"\] for r in sel\)\s*/')

    def test_shuffles_before_filtering_like_the_trainer(self):
        """Filter-then-shuffle gives different batches, so a different reported scalar."""
        i_shuf = self.code.index("shuffle(labelled)")
        i_filt = self.code.index("if is_val(e)")
        self.assertLess(i_shuf, i_filt)
        self.assertIn("shuffle_pool", self.code)

    def test_never_trains_and_never_writes_a_checkpoint(self):
        self.assertNotIn("backward()", self.code)
        self.assertNotIn("optimizer", self.code)
        self.assertNotIn("torch.save", self.code)

    def test_every_name_in_the_output_dict_is_bound_on_all_paths(self):
        """A name bound only inside an optional branch crashes AFTER the GPU work is done.

        This happened: `truth` was assigned only under --verify-trainer, so a sweep without
        that flag ran four minutes of forward passes per checkpoint and then died before
        writing anything. Cheap to prevent, expensive to discover.
        """
        tree = ast.parse(self.text)
        main = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "main")
        conditional = set()
        for node in ast.walk(main):
            if isinstance(node, (ast.If, ast.For, ast.While)):
                for inner in ast.walk(node):
                    if isinstance(inner, ast.Assign):
                        conditional |= {t.id for t in inner.targets if isinstance(t, ast.Name)}
        unconditional = {t.id for node in main.body if isinstance(node, ast.Assign)
                         for t in node.targets if isinstance(t, ast.Name)}
        risky = conditional - unconditional
        self.assertNotIn("truth", risky,
                         "'truth' is only bound inside a branch; initialise it first")

    def test_applies_the_causal_shift(self):
        """Off-by-one here would silently score every token against the wrong target."""
        self.assertIn("logits[:, :-1, :]", self.code)
        self.assertIn("labels[:, 1:]", self.code)


class FloorPartitionTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.text = VIS.read_text(encoding="utf-8")
        cls.code = _strip_comments(cls.text)

    def test_parses(self):
        ast.parse(self.text)

    def test_asserts_the_record_schema(self):
        """The project has already shipped one comparison of None against None."""
        for f in ("bridge_exact_full", "floor_exact_full"):
            self.assertIn(f, self.code)
        self.assertIn("would silently compare None to None", self.text)

    def test_requires_the_floor_to_match_across_arms(self):
        self.assertIn("floor differs across arms", self.text)

    def test_declares_itself_exploratory(self):
        flat = re.sub(r"\s+", " ", self.text)
        self.assertIn("exploratory", flat)
        self.assertIn("never a selection surface", flat)

    def test_defaults_to_the_confirmatory_slice_and_refuses_the_locked_one(self):
        self.assertIn('default="confirm_3000_qids"', self.code)
        self.assertIn("names the locked endpoint", self.text)

    def test_refuses_ambiguous_or_mismatched_inputs(self):
        """hits[0] and a silent QID intersection can compare different questions."""
        self.assertIn("len(hits) != 1", self.code)
        self.assertIn("do not cover identical QID sets", self.text)
        self.assertIn("duplicate QIDs", self.text)
        self.assertIn("evidence_layer", self.code)

    def test_does_not_name_a_latent_property_from_an_observed_outcome(self):
        """floor_wrong is an outcome of one frozen system, not 'the question needs vision'."""
        code = self.code
        for banned in ("vision_sensitive", "vision_win", "vision_loss", "acc_vision"):
            self.assertNotIn(banned, code, f"{banned} names a latent property")
        flat = re.sub(r"\s+", " ", self.text)
        self.assertIn("is NOT a \"vision-sensitive\" set", flat)


class PartitionLogicTests(unittest.TestCase):
    """The four cells must actually mean what the table says they mean."""

    def setUp(self):
        # Deliberately asymmetric, so restricting to the floor-incorrect subset is not
        # a no-op. A symmetric fixture makes the last test below unable to fail.
        # floor:  1 1 1 0 0 0     bridge: 1 1 0 1 0 0
        self.floor = np.array([True, True, True, False, False, False])
        self.bridge = np.array([True, True, False, True, False, False])

    def test_cells_are_mutually_exclusive_and_exhaustive(self):
        f, b = self.floor, self.bridge
        cells = [(f & b), (f & ~b), (~f & b), (~f & ~b)]
        stacked = np.vstack(cells)
        self.assertTrue((stacked.sum(axis=0) == 1).all(), "cells overlap or leave a gap")
        self.assertEqual(sum(int(c.sum()) for c in cells), len(f))

    def test_displaced_is_floor_right_bridge_wrong(self):
        loss = self.floor & ~self.bridge
        self.assertEqual(list(loss), [False, False, True, False, False, False])

    def test_floor_incorrect_subset_excludes_floor_correct(self):
        vs = ~self.floor
        self.assertEqual(int(vs.sum()), 3)
        self.assertTrue(not (vs & self.floor).any())

    def test_restricting_to_floor_incorrect_changes_accuracy(self):
        """If the subset were a no-op the whole analysis would be pointless."""
        vs = ~self.floor
        self.assertNotEqual(self.bridge.mean(), self.bridge[vs].mean())


if __name__ == "__main__":
    unittest.main()
