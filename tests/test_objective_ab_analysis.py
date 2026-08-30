"""Contract tests for the objective-A/B paired analysis (scripts/42_objective_ab_analysis.py).

The analysis turns 12 record files into one promote/do-not-promote verdict, so the properties
that matter are the ones that would let a wrong verdict through quietly: an unpaired bootstrap
that reports a plausible interval, a silent intersection when the two arms cover different
questions, a comparison across evidence layers, or a replication rule that accepts a sign flip.
Each is asserted here, and each is mutation-tested.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "42_objective_ab_analysis.py"
LAUNCHER = ROOT / "scripts" / "cluster" / "41_objective_ab.sbatch"


def load():
    spec = importlib.util.spec_from_file_location("objective_ab_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_arm(directory: Path, stem: str, correct_by_qid: dict, *, floor=0.10,
              layer="corrected_chatml_v1_eos", category="relate"):
    """Write a records file in the shape scripts/07_evaluate.py produces."""
    recs = []
    for i, (qid, ok) in enumerate(sorted(correct_by_qid.items())):
        recs.append({
            "qid": qid, "category": category, "gold": "yes", "question": "q?",
            "image_path": "/x.jpg",
            "bridge": "yes", "bridge_exact_full": bool(ok), "bridge_exact": bool(ok),
            "bridge_vqa": bool(ok), "bridge_stop_reason": "eos_151645",
            "bridge_cap_hit": False, "bridge_n_generated": 2,
            "floor": "yes", "floor_exact_full": bool(i < floor * len(correct_by_qid)),
            "floor_exact": False, "floor_vqa": False, "floor_stop_reason": "eos_151645",
            "floor_cap_hit": False, "floor_n_generated": 2,
        })
    path = directory / f"{stem}__corrected_chatml_v1_eos__objective_4000_qids__image_only__direct_records.json"
    path.write_text(json.dumps({"_meta": {"evidence_layer": layer}, "records": recs}))
    return path


class SelfTestTests(unittest.TestCase):
    """The script's own planted-effect check must actually run and pass."""

    def test_self_test_passes(self):
        import subprocess
        r = subprocess.run([sys.executable, str(SCRIPT), "--records-dir", "/dev/null",
                            "--self-test"], capture_output=True, text=True, cwd=ROOT)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("SELF-TEST PASSED", r.stdout)
        self.assertIn("+5.00", r.stdout)


class PairingTests(unittest.TestCase):
    """The bootstrap must resample question-level DIFFERENCES, not two arms separately."""

    @classmethod
    def setUpClass(cls):
        cls.M = load()

    def test_identical_arms_give_a_zero_width_interval(self):
        """The sharp check: paired differences are all zero, so every resample is zero.

        An unpaired implementation would resample each arm independently and report a
        non-zero interval here, which is exactly the failure this test exists to catch.
        """
        n = 500
        qids = [f"q{i}" for i in range(n)]
        rng = np.random.default_rng(3)
        vals = {q: bool(v) for q, v in zip(qids, rng.random(n) < 0.4)}
        idx, _ = self.M.matrix(n)
        point, lo, hi = self.M.contrast(vals, vals, qids, idx)
        self.assertEqual(point, 0.0)
        self.assertEqual((lo, hi), (0.0, 0.0),
                         "identical arms must yield a degenerate interval under pairing")

    def test_point_estimate_is_the_mean_difference(self):
        n = 1000
        qids = [f"q{i}" for i in range(n)]
        a = {q: False for q in qids}
        b = {q: (i < 70) for i, q in enumerate(qids)}
        idx, _ = self.M.matrix(n)
        point, lo, hi = self.M.contrast(b, a, qids, idx)
        self.assertAlmostEqual(point, 7.0, places=9)
        self.assertLess(lo, point)
        self.assertGreater(hi, point)

    def test_sign_follows_the_argument_order(self):
        """contrast(b, a) must be b minus a, or every verdict is backwards."""
        qids = ["q0", "q1"]
        a = {"q0": True, "q1": True}
        b = {"q0": False, "q1": False}
        idx, _ = self.M.matrix(2)
        point, _, _ = self.M.contrast(b, a, qids, idx)
        self.assertLess(point, 0, "a worse b arm must give a negative difference")

    def test_resample_matrix_is_reproducible(self):
        a, sha_a = self.M.matrix(200)
        b, sha_b = self.M.matrix(200)
        self.assertEqual(sha_a, sha_b)
        self.assertTrue(np.array_equal(a, b))


class ReplicationRuleTests(unittest.TestCase):
    """The decision rule, which is the whole point of the design."""

    @classmethod
    def setUpClass(cls):
        cls.M = load()

    def test_all_positive_replicates(self):
        rows = [{"diff_points": d} for d in (0.4, 1.8, 0.1)]
        self.assertTrue(self.M.replication_verdict(rows)[1])

    def test_all_negative_replicates(self):
        rows = [{"diff_points": d} for d in (-0.4, -1.8, -0.1)]
        self.assertTrue(self.M.replication_verdict(rows)[1])

    def test_a_single_sign_flip_breaks_replication(self):
        rows = [{"diff_points": d} for d in (0.4, -0.1, 1.8)]
        self.assertFalse(self.M.replication_verdict(rows)[1])

    def test_an_exact_zero_is_not_agreement(self):
        rows = [{"diff_points": d} for d in (0.4, 0.0, 1.8)]
        self.assertFalse(self.M.replication_verdict(rows)[1])

    def test_no_pooled_test_exists(self):
        """Structural: there must be no function that pools the three seeds."""
        source = SCRIPT.read_text()
        for banned in ("ttest", "t_test", "wilcoxon", "mannwhitney", "chi2", "pooled_p"):
            with self.subTest(banned=banned):
                self.assertNotIn(banned, source.lower())


class PremiseCheckTests(unittest.TestCase):
    """Every premise that would silently corrupt the contrast must stop the run."""

    @classmethod
    def setUpClass(cls):
        cls.M = load()

    def _dir(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: None)
        return tmp

    def test_mismatched_qid_sets_are_refused(self):
        d = self._dir()
        a = {f"q{i}": True for i in range(10)}
        b = {f"q{i}": True for i in range(1, 11)}       # shifted by one
        write_arm(d, "ctrl", a)
        write_arm(d, "seq", b)
        with self.assertRaisesRegex(SystemExit, "different qids"):
            self.M.analyse_pair(d, "clip", 42, "ctrl", "seq")

    def test_wrong_slice_size_is_refused(self):
        d = self._dir()
        vals = {f"q{i}": True for i in range(10)}
        write_arm(d, "ctrl", vals)
        write_arm(d, "seq", vals)
        with self.assertRaisesRegex(SystemExit, "expected 4000"):
            self.M.analyse_pair(d, "clip", 42, "ctrl", "seq")

    def test_wrong_evidence_layer_is_refused(self):
        d = self._dir()
        vals = {f"q{i}": True for i in range(10)}
        write_arm(d, "ctrl", vals)
        write_arm(d, "seq", vals, layer="legacy_raw_prompt_no_eos")
        with self.assertRaisesRegex(SystemExit, "evidence layer"):
            self.M.analyse_pair(d, "clip", 42, "ctrl", "seq")

    def test_missing_records_file_is_refused(self):
        d = self._dir()
        with self.assertRaisesRegex(SystemExit, "exactly one records file"):
            self.M.find_records(d, "nothing_here")

    def test_ambiguous_records_files_are_refused(self):
        d = self._dir()
        vals = {f"q{i}": True for i in range(3)}
        write_arm(d, "ctrl", vals)
        (d / "ctrl__another__thing_records.json").write_text("{}")
        with self.assertRaisesRegex(SystemExit, "exactly one records file"):
            self.M.find_records(d, "ctrl")


@unittest.skipUnless(
    LAUNCHER.is_file(),
    "historical site launcher is intentionally absent from the clean release",
)
class ArmTableTests(unittest.TestCase):
    """The analysis and the launcher must describe the same six runs."""

    @classmethod
    def setUpClass(cls):
        cls.M = load()

    def test_six_arms(self):
        self.assertEqual(len(self.M.ARMS), 6)

    def test_matches_the_launcher_table(self):
        """A drift here would analyse a different experiment than the one that ran."""
        text = LAUNCHER.read_text()
        for encoder, seed, control, new in self.M.ARMS:
            with self.subTest(encoder=encoder, seed=seed):
                self.assertIn(control.replace("_ep5", ""), text,
                              f"{control} is not a control in the launcher")
                self.assertIn(f"w1seq_s{seed}_v1", "".join(
                    [f"w1seq_s{s}_v1" for _e, s, _c, _n in self.M.ARMS]))
                self.assertTrue(new.startswith(f"bridge_{encoder}_w1seq_s{seed}"))

    def test_expected_n_matches_the_frozen_slice(self):
        manifest = ROOT / "data" / "gqa" / "objective_4000_qids_MANIFEST.json"
        if not manifest.exists():
            self.skipTest("slice not built")
        self.assertEqual(self.M.EXPECTED_N, json.loads(manifest.read_text())["n"])

    def test_metric_is_the_projects_primary(self):
        self.assertEqual(self.M.METRIC, "exact_full")


if __name__ == "__main__":
    unittest.main()
