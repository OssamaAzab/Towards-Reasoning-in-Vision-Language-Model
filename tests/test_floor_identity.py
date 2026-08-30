"""Tests for scripts/32_verify_floor_identity.py.

These exist because the equivalent logic, embedded as heredoc Python inside launcher 21,
crashed in production after all ten of that job's evaluations had succeeded. The launcher's
tests asserted the string "byte-identical" appeared in the file — they could not execute the
code. Every test here runs the real function against a real artifact.
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

M = import_module("32_verify_floor_identity")


def corrected(tmp, name, floors, *, n=None):
    """Write a CORRECTED artifact: {"_meta": ..., "records": [...]} — the form that broke it."""
    recs = [{"qid": f"q{i}", "floor": f, "bridge": "x", "floor_exact": True}
            for i, f in enumerate(floors)]
    p = Path(tmp) / f"{name}.json"
    p.write_text(json.dumps({"_meta": {"split": "tune_500_qids"}, "records": recs}))
    return str(p)


def legacy(tmp, name, floors):
    """Write a LEGACY artifact: a bare list."""
    recs = [{"qid": f"q{i}", "floor": f} for i, f in enumerate(floors)]
    p = Path(tmp) / f"{name}.json"
    p.write_text(json.dumps(recs))
    return str(p)


class ArtifactFormTests(unittest.TestCase):
    """The exact regression: corrected artifacts wrap records in a dict."""

    def test_reads_the_corrected_dict_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = M.floors(corrected(tmp, "a", ["yes", "no"]))
            self.assertEqual(got, {"q0": "yes", "q1": "no"})

    def test_reads_the_legacy_list_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = M.floors(legacy(tmp, "a", ["yes", "no"]))
            self.assertEqual(got, {"q0": "yes", "q1": "no"})

    def test_dict_without_records_key_is_a_clear_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.json"
            p.write_text(json.dumps({"_meta": {}}))
            with self.assertRaises(SystemExit) as cm:
                M.load_records(p)
            self.assertIn("without a 'records' key", str(cm.exception))


class ComparisonTests(unittest.TestCase):
    """The check itself."""

    def test_identical_floors_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [corrected(tmp, n, ["yes", "no", "red"]) for n in ("a", "b", "c")]
            _, n, report = M.compare(paths)
            self.assertEqual(n, 3)
            self.assertTrue(all(d == 0 for _, d, _ in report))

    def test_a_single_differing_generation_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = corrected(tmp, "a", ["yes", "no", "red"])
            b = corrected(tmp, "b", ["yes", "no", "blue"])
            _, _, report = M.compare([a, b])
            self.assertEqual(report[0][1], 1)
            self.assertEqual(report[0][2], ("q2", "red", "blue"))

    def test_mixed_forms_compare_correctly(self):
        """A legacy and a corrected artifact of the same run must not appear to differ."""
        with tempfile.TemporaryDirectory() as tmp:
            a = corrected(tmp, "a", ["yes", "no"])
            b = legacy(tmp, "b", ["yes", "no"])
            _, _, report = M.compare([a, b])
            self.assertEqual(report[0][1], 0)

    def test_differing_question_sets_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = corrected(tmp, "a", ["yes", "no", "red"])
            b = corrected(tmp, "b", ["yes", "no"])
            with self.assertRaises(SystemExit) as cm:
                M.compare([a, b])
            self.assertIn("not on the same question set", str(cm.exception))

    def test_expected_count_is_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = corrected(tmp, "a", ["yes", "no"])
            with self.assertRaises(SystemExit) as cm:
                M.floors(a, expect_n=3000)
            self.assertIn("expected 3000", str(cm.exception))

    def test_duplicate_qids_abort(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "dup.json"
            p.write_text(json.dumps({"_meta": {}, "records": [
                {"qid": "q0", "floor": "yes"}, {"qid": "q0", "floor": "yes"}]}))
            with self.assertRaises(SystemExit) as cm:
                M.floors(p)
            self.assertIn("duplicate qid", str(cm.exception))

    def test_one_file_is_not_a_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit) as cm:
                M.compare([corrected(tmp, "a", ["yes"])])
            self.assertIn("at least two", str(cm.exception))


class CliTests(unittest.TestCase):
    """The CLI path, since the production failure was in a script's top-level flow."""

    SCRIPT = ROOT / "scripts" / "32_verify_floor_identity.py"

    def _run(self, args):
        import subprocess
        return subprocess.run([sys.executable, str(self.SCRIPT)] + args,
                              capture_output=True, text=True, cwd=ROOT)

    def test_exits_zero_and_reports_pass_on_identical_floors(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [corrected(tmp, n, ["yes", "no"]) for n in ("a", "b")]
            r = self._run(paths)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn("byte-identical", r.stdout)

    def test_exits_nonzero_on_divergence(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = corrected(tmp, "a", ["yes", "no"])
            b = corrected(tmp, "b", ["yes", "MAYBE"])
            r = self._run([a, b])
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("divergent floor", r.stdout + r.stderr)


if __name__ == "__main__":
    unittest.main()
