"""Tests for scripts/33_seed_aggregate_qformer.py.

The statistical failure this script must not permit is treating three seeds answering the same
3,000 questions as 9,000 independent observations, which would shrink every interval by sqrt(3)
on a false assumption. The second is letting the per-seed sensitivity curve be read as a
replicated interaction when the MLP cells have only one seed.
"""
from importlib import import_module
from pathlib import Path
import json
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

M = import_module("33_seed_aggregate_qformer")
SCRIPT = ROOT / "scripts" / "33_seed_aggregate_qformer.py"


def artifact(tmp, name, outcomes, *, seed=42, split="confirm_3000_qids"):
    """A corrected records artifact where question i is correct iff outcomes[i]."""
    recs = [{"qid": f"q{i}", "category": "relate", "gold": "yes", "bridge": "yes",
             "bridge_stop_reason": "eos_151645", "bridge_cap_hit": False,
             "bridge_n_generated": 2,
             **{f"bridge_{m}": bool(o) for m in M.METRICS}}
            for i, o in enumerate(outcomes)]
    p = Path(tmp) / f"{name}.json"
    p.write_text(json.dumps({"_meta": {"split": split, "seed": seed}, "records": recs}))
    return str(p)


class GuardTests(unittest.TestCase):
    """Ways the inputs can be wrong."""

    def _run(self, args):
        import subprocess
        return subprocess.run([sys.executable, str(SCRIPT)] + args,
                              capture_output=True, text=True, cwd=ROOT)

    def test_duplicate_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, i = artifact(tmp, "c", [1, 0]), artifact(tmp, "i", [0, 1])
            r = self._run(["--seed", "42", c, i, "--seed", "42", c, i])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("supplied twice", r.stdout + r.stderr)

    def test_non_confirmatory_split_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c1, i1 = artifact(tmp, "c1", [1, 0]), artifact(tmp, "i1", [0, 1])
            c2 = artifact(tmp, "c2", [1, 0], seed=43, split="tune_500_qids")
            i2 = artifact(tmp, "i2", [0, 1], seed=43, split="tune_500_qids")
            r = self._run(["--seed", "42", c1, i1, "--seed", "43", c2, i2])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("must use the confirmatory slice", r.stdout + r.stderr)

    def test_mislabelled_seed_is_rejected(self):
        """A file whose checkpoint seed disagrees with its command-line label."""
        with tempfile.TemporaryDirectory() as tmp:
            c1, i1 = artifact(tmp, "c1", [1, 0]), artifact(tmp, "i1", [0, 1])
            c2 = artifact(tmp, "c2", [1, 0], seed=99)
            i2 = artifact(tmp, "i2", [0, 1], seed=99)
            r = self._run(["--seed", "42", c1, i1, "--seed", "43", c2, i2])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("mislabelled", r.stdout + r.stderr)

    def test_one_seed_says_nothing_about_seed_variance(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(["--seed", "42", artifact(tmp, "c", [1, 0]),
                           artifact(tmp, "i", [0, 1])])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("at least two seeds", r.stdout + r.stderr)

    def test_mismatched_question_sets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            c1, i1 = artifact(tmp, "c1", [1, 0, 1]), artifact(tmp, "i1", [0, 1, 0])
            c2 = artifact(tmp, "c2", [1, 0], seed=43)
            i2 = artifact(tmp, "i2", [0, 1], seed=43)
            r = self._run(["--seed", "42", c1, i1, "--seed", "43", c2, i2])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("different question set", r.stdout + r.stderr)


class StatisticalContractTests(unittest.TestCase):
    """The two claims the script must never make."""

    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_gap_direction_is_clip_minus_ijepa(self):
        with tempfile.TemporaryDirectory() as tmp:
            c, _ = M.load(artifact(tmp, "c", [1, 1, 1, 0]))    # 75%
            i, _ = M.load(artifact(tmp, "i", [1, 0, 0, 0]))    # 25%
            got = M.gap_with_ci(c, i, ["q0", "q1", "q2", "q3"], "exact_full")
            self.assertAlmostEqual(got["gap"], 50.0, places=6)

    def test_seeds_are_not_pooled_as_independent_questions(self):
        self.assertIn("NOT pooled as independent questions", self.text)
        self.assertIn("correlated models", self.text)

    def test_sensitivity_curve_is_labelled_not_an_interaction(self):
        self.assertIn("CONDITIONAL ON THE FIXED SEED-42 MLP GAP", self.text)
        self.assertIn("NOT a replicated interaction", self.text)
        self.assertIn('"is_replicated_interaction": False', self.text)

    def test_across_seed_spread_is_described_as_coarse(self):
        self.assertIn("COARSE estimate of training variance", self.text)
        self.assertIn("not a bound", self.text)

    def test_no_across_seed_confidence_interval_is_computed(self):
        """Three values cannot support an interval; only spread is reported."""
        self.assertNotIn("across_seed_ci", self.text)

    def test_sign_reversal_is_reported_not_hidden(self):
        self.assertIn("a seed reverses sign", self.text)


class EndToEndTests(unittest.TestCase):
    """The CLI path, on two well-formed seeds."""

    def test_reports_every_seed_and_the_spread(self):
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            c1, i1 = artifact(tmp, "c1", [1, 1, 1, 0]), artifact(tmp, "i1", [1, 0, 0, 0])
            c2 = artifact(tmp, "c2", [1, 1, 0, 0], seed=43)
            i2 = artifact(tmp, "i2", [1, 0, 0, 0], seed=43)
            r = subprocess.run([sys.executable, str(SCRIPT), "--seed", "42", c1, i1,
                                "--seed", "43", c2, i2],
                               capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("--- seed 42", r.stdout)
            self.assertIn("--- seed 43", r.stdout)
            self.assertIn("sign consistent", r.stdout)
            self.assertIn("sensitivity curve", r.stdout)
            self.assertRegex(r.stdout, r"seed 42: \+50\.00")
            self.assertRegex(r.stdout, r"seed 43: \+25\.00")


if __name__ == "__main__":
    unittest.main()
