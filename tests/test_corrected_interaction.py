"""Contract tests for scripts/30_corrected_interaction.py.

The interaction is the project's headline quantity, so the tests below are mostly about the
ways it can be silently WRONG rather than the ways it can crash: a cell wired into the wrong
slot, two evidence layers merged, questions that do not actually align, or a cross-connector
contrast quietly spanning two GPU architectures.

The regression test at the bottom is the strongest of them: script 30 must reproduce
scripts/27_metric_ci.py's independently-computed legacy interaction to the last digit.
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

M = import_module("30_corrected_interaction")


def make_records(outcomes, *, meta=None, metrics=("exact_full", "exact", "vqa")):
    """Build a records artifact where question i is correct iff outcomes[i]."""
    recs = [{"qid": f"q{i}", "category": "relate", "gold": "yes", "question": "?",
             "bridge": "yes", "bridge_stop_reason": "eos_151645", "bridge_cap_hit": False,
             "bridge_n_generated": 2,
             **{f"bridge_{m}": bool(o) for m in metrics}}
            for i, o in enumerate(outcomes)]
    return {"_meta": meta or {"evidence_layer": "corrected_chatml_v1_eos",
                              "metric_version": "exact_full/1.0.0",
                              "split": "tune_500_qids", "checkpoint_epoch": 5,
                              "checkpoint_tag": "t"},
            "records": recs}


def write(tmp, name, payload):
    """Write a records artifact and return its path."""
    p = Path(tmp) / f"{name}.json"
    p.write_text(json.dumps(payload))
    return str(p)


class AlignmentTests(unittest.TestCase):
    """Pairing is only meaningful if every cell answers the same questions once."""

    def test_duplicate_qid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            payload = make_records([1, 0])
            payload["records"][1]["qid"] = "q0"
            p = write(tmp, "dup", payload)
            with self.assertRaises(SystemExit) as cm:
                M.load_cell(p, name="dup")
            self.assertIn("duplicate qid", str(cm.exception))

    def test_cells_on_different_question_sets_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, _ = M.load_cell(write(tmp, "a", make_records([1, 1, 1])), name="a")
            b, _ = M.load_cell(write(tmp, "b", make_records([1, 1])), name="b")
            with self.assertRaises(SystemExit) as cm:
                M.align({"a": (a, {}), "b": (b, {})})
            self.assertIn("not on the same question set", str(cm.exception))

    def test_identical_question_sets_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, _ = M.load_cell(write(tmp, "a", make_records([1, 0])), name="a")
            b, _ = M.load_cell(write(tmp, "b", make_records([0, 1])), name="b")
            self.assertEqual(M.align({"a": (a, {}), "b": (b, {})}), ["q0", "q1"])


class ProvenanceTests(unittest.TestCase):
    """Legacy and corrected results are separate evidence layers and never share a row."""

    def _cells(self, tmp, metas):
        return {n: M.load_cell(write(tmp, n, make_records([1, 0], meta=m)), name=n)
                for n, m in metas.items()}

    def test_mixed_evidence_layers_are_rejected(self):
        base = {"evidence_layer": "corrected_chatml_v1_eos",
                "metric_version": "exact_full/1.0.0", "split": "tune_500_qids"}
        legacy = {**base, "evidence_layer": "legacy_raw_prompt_no_eos"}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                M.check_provenance(self._cells(tmp, {"a": base, "b": legacy}))
            self.assertIn("evidence_layer", str(cm.exception))

    def test_mixed_splits_are_rejected(self):
        base = {"evidence_layer": "e", "metric_version": "m", "split": "tune_500_qids"}
        other = {**base, "split": "confirm_3000_qids"}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                M.check_provenance(self._cells(tmp, {"a": base, "b": other}))
            self.assertIn("split", str(cm.exception))

    def test_metric_absent_from_one_cell_is_skipped_not_faked(self):
        with tempfile.TemporaryDirectory() as tmp:
            full = M.load_cell(write(tmp, "f", make_records([1, 0])), name="f")
            thin = M.load_cell(write(tmp, "t", make_records([1, 0], metrics=("exact", "vqa"))),
                               name="t")
            got = M.available_metrics({"f": full, "t": thin}, ["q0", "q1"])
            self.assertNotIn("exact_full", got)
            self.assertEqual(set(got), {"exact", "vqa"})


class InteractionAlgebraTests(unittest.TestCase):
    """The quantity itself: direction, identity, and pairing."""

    def test_gap_is_left_minus_right(self):
        import numpy as np
        a, b = np.array([1, 1, 1, 0], bool), np.array([1, 0, 0, 0], bool)
        idx = np.zeros((2, 4), dtype=np.int32)
        point, _ = M.gap(a, b, idx)
        self.assertAlmostEqual(point, 50.0, places=6)   # 75% - 25%

    def test_interaction_identity_holds_on_real_arithmetic(self):
        """E_QF - E_MLP must equal C_IJEPA - C_CLIP, or a cell is in the wrong slot."""
        import numpy as np
        rng = np.random.default_rng(0)
        n = 200
        o = {k: rng.random(n) < p for k, p in
             (("clip-mlp", .45), ("ijepa-mlp", .40), ("clip-qf", .35), ("ijepa-qf", .25))}
        idx = rng.integers(0, n, size=(4, n), dtype=np.int32)
        e_mlp, _ = M.gap(o["clip-mlp"], o["ijepa-mlp"], idx)
        e_qf, _ = M.gap(o["clip-qf"], o["ijepa-qf"], idx)
        c_clip, _ = M.gap(o["clip-mlp"], o["clip-qf"], idx)
        c_ijepa, _ = M.gap(o["ijepa-mlp"], o["ijepa-qf"], idx)
        self.assertAlmostEqual(e_qf - e_mlp, c_ijepa - c_clip, places=9)

    def test_discordance_counts_are_directional(self):
        import numpy as np
        a, b = np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool)
        d = M.discordance(a, b)
        self.assertEqual((d["both_correct"], d["left_only"], d["right_only"], d["both_wrong"]),
                         (1, 1, 1, 1))
        self.assertEqual(d["discordant"], 2)

    def test_bootstrap_is_paired_not_independent(self):
        """Two perfectly-correlated arms must give a zero-width interval around zero."""
        import numpy as np
        n = 300
        rng = np.random.default_rng(1)
        a = rng.random(n) < 0.4
        idx = rng.integers(0, n, size=(500, n), dtype=np.int32)
        _, series = M.gap(a, a.copy(), idx)
        self.assertEqual(float(series.min()), 0.0)
        self.assertEqual(float(series.max()), 0.0)


class HardwareGuardTests(unittest.TestCase):
    """A cross-connector contrast spanning two GPU architectures is confounded by ~0.4 points."""

    SCRIPT = ROOT / "scripts" / "30_corrected_interaction.py"

    @classmethod
    def setUpClass(cls):
        cls.text = cls.SCRIPT.read_text(encoding="utf-8")

    def test_cross_connector_quantities_are_the_guarded_ones(self):
        """Encoder gaps stay within one connector, so only these three are at risk."""
        self.assertIn('CROSS = {"C_CLIP", "C_IJEPA", "INTERACTION"}', self.text)

    def test_architecture_is_declared_never_inferred(self):
        self.assertIn("will not be guessed", self.text)
        self.assertIn("--arch-mlp", self.text)
        self.assertIn("--arch-qf", self.text)

    def test_undeclared_architecture_withholds_rather_than_assumes(self):
        self.assertIn("withheld_hardware_mismatch", self.text)

    def test_override_exists_and_marks_results_exploratory(self):
        self.assertIn("--allow-hardware-mismatch", self.text)
        self.assertIn("exploratory_hardware_confounded", self.text)

    def test_identity_violation_aborts(self):
        self.assertIn("interaction identity broken", self.text)


class EndToEndTests(unittest.TestCase):
    """Exercise the actual CLI path.

    Every guard in this script lives in main(), so unit tests over the helpers cannot see a
    cell wired into the wrong slot. A mutation that computed E_QF against the MLP cell passed
    the entire helper-level suite; only running the script catches it.
    """

    SCRIPT = ROOT / "scripts" / "30_corrected_interaction.py"

    def _run(self, tmp, extra=()):
        import subprocess
        # Deliberately asymmetric cells: a symmetric fixture would satisfy the interaction
        # identity even when the cells are misassigned, and prove nothing.
        cells = {"clip-mlp": [1, 1, 1, 1, 0, 0], "ijepa-mlp": [1, 1, 0, 0, 0, 0],
                 "clip-qf": [1, 0, 1, 0, 1, 0], "ijepa-qf": [0, 0, 0, 1, 0, 0]}
        args = [sys.executable, str(self.SCRIPT)]
        for name, o in cells.items():
            args += [f"--{name}", write(tmp, name, make_records(o))]
        return subprocess.run(args + list(extra), capture_output=True, text=True, cwd=ROOT)

    def test_runs_end_to_end_and_reports_the_interaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(tmp, ["--arch-mlp", "blackwell", "--arch-qf", "blackwell"])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("INTERACTION", r.stdout)
            self.assertIn("E_MLP", r.stdout)
            # clip-mlp 4/6 - ijepa-mlp 2/6 = +33.3
            self.assertRegex(r.stdout, r"E_MLP\s+\+33\.3")

    def test_cross_connector_quantities_withheld_when_hardware_differs(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(tmp, ["--arch-mlp", "blackwell", "--arch-qf", "ampere"])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("INTERACTION  WITHHELD", r.stdout)
            self.assertNotIn("EXPLORATORY", r.stdout)

    def test_override_reports_the_interaction_but_labels_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self._run(tmp, ["--arch-mlp", "blackwell", "--arch-qf", "ampere",
                                "--allow-hardware-mismatch"])
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertNotIn("WITHHELD", r.stdout)
            self.assertIn("EXPLORATORY", r.stdout)

    def test_identity_check_aborts_when_a_cell_is_misassigned(self):
        """The guard that the surviving mutant would have tripped."""
        with tempfile.TemporaryDirectory() as tmp:
            src = self.SCRIPT.read_text(encoding="utf-8")
            broken = Path(tmp) / "broken.py"
            broken.write_text(src.replace(
                'quantities["E_QF"], series["E_QF"] = gap(o["clip-qf"], o["ijepa-qf"], idx_mat)',
                'quantities["E_QF"], series["E_QF"] = gap(o["clip-qf"], o["ijepa-mlp"], idx_mat)'))
            import subprocess
            cells = {"clip-mlp": [1, 1, 1, 1, 0, 0], "ijepa-mlp": [1, 1, 0, 0, 0, 0],
                     "clip-qf": [1, 0, 1, 0, 1, 0], "ijepa-qf": [0, 0, 0, 1, 0, 0]}
            args = [sys.executable, str(broken)]
            for name, o in cells.items():
                args += [f"--{name}", write(tmp, name, make_records(o))]
            import os
            env = {**os.environ, "PYTHONPATH": str(ROOT)}  # the copy lives outside scripts/
            r = subprocess.run(args + ["--arch-mlp", "b", "--arch-qf", "b"],
                               capture_output=True, text=True, cwd=ROOT, env=env)
            self.assertNotEqual(r.returncode, 0, "a misassigned cell must abort")
            self.assertIn("interaction identity broken", r.stdout + r.stderr)


class LegacyRegressionTests(unittest.TestCase):
    """Script 30 must reproduce scripts/27_metric_ci.py, which was verified independently.

    27 and 30 share no code: 27 hardcodes the legacy stems and its own bootstrap, 30 takes
    cells as arguments. Agreement to the last digit on all six quantities is therefore a real
    cross-implementation check, not a tautology.
    """

    CSV = ROOT / "outputs" / "eval" / "metric_ci_interaction.csv"
    EVAL = ROOT / "outputs" / "eval"
    CELLS = {"clip-mlp": "bridge_clip_150k_mlp_ep3", "ijepa-mlp": "bridge_ijepa_150k_mlp_ep3",
             "clip-qf": "bridge_clip_150k_3ep", "ijepa-qf": "bridge_ijepa_150k_3ep"}

    def setUp(self):
        if not self.CSV.is_file() or any(
                not (self.EVAL / f"{s}_records.json").is_file() for s in self.CELLS.values()):
            self.skipTest("legacy artifacts not present (outputs/ is gitignored)")

    def test_reproduces_script_27_interaction_exactly(self):
        import csv
        import numpy as np
        want = {(r["metric"], r["quantity"]): r for r in csv.DictReader(self.CSV.open())}
        cells = {n: M.load_cell(self.EVAL / f"{s}_records.json", name=n)
                 for n, s in self.CELLS.items()}
        qids = M.align(cells)
        n = len(qids)
        for metric, csv_metric in (("exact", "exact"), ("vqa", "vqa_soft")):
            rng = np.random.default_rng(M.SEED)
            idx = rng.integers(0, n, size=(M.N_RESAMPLES, n), dtype=np.int32)
            o = {k: M.outcomes(i, qids, metric) for k, (i, _) in cells.items()}
            e_mlp, s_mlp = M.gap(o["clip-mlp"], o["ijepa-mlp"], idx)
            e_qf, s_qf = M.gap(o["clip-qf"], o["ijepa-qf"], idx)
            for quantity, point, series in (
                    ("rq2_gap_mlp", e_mlp, s_mlp), ("rq2_gap_qformer", e_qf, s_qf),
                    ("interaction_qformer_minus_mlp", e_qf - e_mlp, s_qf - s_mlp)):
                got = M.ci(series, point)
                exp = want[(csv_metric, quantity)]
                self.assertEqual(
                    got, (float(exp["point"]), float(exp["ci_lo"]), float(exp["ci_hi"])),
                    f"{csv_metric}/{quantity} disagrees with script 27")


if __name__ == "__main__":
    unittest.main()
