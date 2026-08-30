"""Tests for the paired-analysis layer (scripts/28_paired_analysis.py).

The failure modes that matter here are silent ones: pairing on the wrong questions, letting
a duplicated qid double-weight itself, resampling the two arms independently (which breaks
the pairing that makes the interval narrow), or inferring a difference between two arms from
the fact that one interval excludes zero and the other does not.

The last is the reason arm-to-arm contrast exists at all, and it is tested with a
constructed case where both within-arm effects are identical and the arm-to-arm difference
is nonetheless large.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _analysis():
    """Load scripts/28_paired_analysis.py as a module (numbered scripts are not importable)."""
    spec = importlib.util.spec_from_file_location(
        "paired_analysis", ROOT / "scripts" / "28_paired_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["paired_analysis"] = mod
    spec.loader.exec_module(mod)
    return mod


PA = _analysis()


def rec(qid, *, a=True, b=False, cat="relate", ctx="objects: man", gen=2,
        stop="eos", cap=False, ans="yes"):
    """Build one per-example record in scripts/09's saved schema."""
    return {"qid": qid, "category": cat, "question": "q?", "gold": "yes",
            "image_path": f"/img/{qid}.jpg",
            "a_ans": ans if a else "no", "a_exact": a, "a_vqa": a, "a_exact_full": a,
            "b_ans": ans if b else "no", "b_exact": b, "b_vqa": b, "b_exact_full": b,
            "a_raw": ans, "b_raw": ans, "a_context": ctx,
            "a_stop_reason": stop, "a_cap_hit": cap, "a_n_generated": gen,
            "b_stop_reason": "eos", "b_cap_hit": False, "b_n_generated": 2}


def write_arm(tmp, name, records, meta=None):
    """Write a records artifact in the {_meta, records} form scripts/09 emits."""
    path = Path(tmp) / f"{name}.json"
    path.write_text(json.dumps({"_meta": meta or {"augmentation": name}, "records": records}))
    return path


class BootstrapTests(unittest.TestCase):
    """The interval must be deterministic, correctly signed, and honest about zero."""

    def test_is_deterministic_under_the_fixed_seed(self):
        deltas = [1.0] * 30 + [0.0] * 40 + [-1.0] * 30
        self.assertEqual(PA.bootstrap_ci(deltas), PA.bootstrap_ci(deltas))

    def test_all_positive_deltas_give_an_interval_above_zero(self):
        point, lo, hi = PA.bootstrap_ci([1.0] * 100)
        self.assertAlmostEqual(point, 100.0)
        self.assertGreater(lo, 0)

    def test_all_negative_deltas_give_an_interval_below_zero(self):
        point, lo, hi = PA.bootstrap_ci([-1.0] * 100)
        self.assertAlmostEqual(point, -100.0)
        self.assertLess(hi, 0)

    def test_symmetric_noise_gives_an_interval_containing_zero(self):
        point, lo, hi = PA.bootstrap_ci(([1.0, -1.0] * 100))
        self.assertAlmostEqual(point, 0.0)
        self.assertLess(lo, 0)
        self.assertGreater(hi, 0)

    def test_empty_input_does_not_crash(self):
        self.assertEqual(PA.bootstrap_ci([]), (0.0, 0.0, 0.0))

    def test_point_estimate_is_the_mean_delta_in_points(self):
        deltas = [1.0] * 25 + [0.0] * 75
        self.assertAlmostEqual(PA.bootstrap_ci(deltas)[0], 25.0)


class PairingTests(unittest.TestCase):
    """Pairing must be on qid, refuse duplicates, and never silently mismatch."""

    def test_duplicate_qids_are_refused(self):
        """A duplicated qid would double-weight that question in every statistic."""
        with self.assertRaises(SystemExit):
            PA.index_by_qid([rec("1"), rec("1")], source="dupe.json")

    def test_qids_are_compared_as_strings(self):
        """Records carry qids as str; an int/str mix must not create phantom non-overlap."""
        left = PA.index_by_qid([{"qid": 7, **{k: v for k, v in rec("7").items() if k != "qid"}}],
                               source="l")
        right = PA.index_by_qid([rec("7")], source="r")
        self.assertEqual(PA.paired_qids(left, right, left_name="l", right_name="r"), ["7"])

    def test_disjoint_qid_sets_are_a_hard_error(self):
        left = PA.index_by_qid([rec("1")], source="l")
        right = PA.index_by_qid([rec("2")], source="r")
        with self.assertRaises(SystemExit):
            PA.paired_qids(left, right, left_name="l", right_name="r")

    def test_partial_overlap_pairs_only_the_intersection(self):
        left = PA.index_by_qid([rec("1"), rec("2"), rec("3")], source="l")
        right = PA.index_by_qid([rec("2"), rec("3"), rec("4")], source="r")
        self.assertEqual(PA.paired_qids(left, right, left_name="l", right_name="r"), ["2", "3"])


class FlipTests(unittest.TestCase):
    """Flip counts must partition the questions exactly."""

    def test_flips_are_counted_in_the_right_direction(self):
        records = [rec("1", a=True, b=False),      # wrong -> right
                   rec("2", a=False, b=True),      # right -> wrong
                   rec("3", a=True, b=True),       # both correct
                   rec("4", a=False, b=False)]     # both wrong
        idx = PA.index_by_qid(records, source="t")
        row = PA.contrast(idx, idx, sorted(idx), left_side="a", right_side="b")["exact_full"]
        self.assertEqual(row["wrong_to_right"], 1)
        self.assertEqual(row["right_to_wrong"], 1)
        self.assertEqual(row["both_correct"], 1)
        self.assertEqual(row["both_wrong"], 1)

    def test_the_four_buckets_partition_every_question(self):
        records = [rec(str(i), a=i % 2 == 0, b=i % 3 == 0) for i in range(40)]
        idx = PA.index_by_qid(records, source="t")
        row = PA.contrast(idx, idx, sorted(idx), left_side="a", right_side="b")["exact_full"]
        total = (row["wrong_to_right"] + row["right_to_wrong"]
                 + row["both_correct"] + row["both_wrong"])
        self.assertEqual(total, 40)

    def test_effect_equals_the_flip_difference(self):
        """A sanity identity: effect = (W->R - R->W) / n, so flips and effect cannot drift."""
        records = [rec(str(i), a=i < 30, b=i >= 20) for i in range(50)]
        idx = PA.index_by_qid(records, source="t")
        row = PA.contrast(idx, idx, sorted(idx), left_side="a", right_side="b")["exact_full"]
        expected = 100.0 * (row["wrong_to_right"] - row["right_to_wrong"]) / 50
        self.assertAlmostEqual(row["effect"], expected, places=6)


class ArmToArmContrastTests(unittest.TestCase):
    """The reason this layer exists: overlapping-interval reasoning is not a contrast."""

    def test_two_arms_with_identical_within_arm_effects_can_still_differ(self):
        """Both arms beat baseline by exactly +20 pts, yet disagree on 40% of questions.

        This is the case that defeats "G2's CI excludes zero and G1's does not, therefore
        conditioning helps": the within-arm effects are IDENTICAL here, so any such
        inference would be pure noise-reading. The arm-to-arm contrast is what can tell
        them apart, and here it correctly reports a difference of zero with real churn.
        """
        # Arm X correct on 0-59; arm Y correct on 20-79. Baseline correct on 0-39.
        x = [rec(str(i), a=i < 60, b=i < 40) for i in range(100)]
        y = [rec(str(i), a=20 <= i < 80, b=i < 40) for i in range(100)]
        xi, yi = PA.index_by_qid(x, source="x"), PA.index_by_qid(y, source="y")
        qids = sorted(xi)

        x_eff = PA.contrast(xi, xi, qids, left_side="a", right_side="b")["exact_full"]
        y_eff = PA.contrast(yi, yi, qids, left_side="a", right_side="b")["exact_full"]
        self.assertAlmostEqual(x_eff["effect"], 20.0)
        self.assertAlmostEqual(y_eff["effect"], 20.0)

        direct = PA.contrast(yi, xi, qids, left_side="a", right_side="a")["exact_full"]
        self.assertAlmostEqual(direct["effect"], 0.0)
        # ... but they are not the same arm: 20 questions flip each way.
        self.assertEqual(direct["wrong_to_right"], 20)
        self.assertEqual(direct["right_to_wrong"], 20)

    def test_arm_to_arm_uses_the_augmented_side_of_both(self):
        """Comparing arm X's augmented answers to arm Y's BASELINE would be meaningless."""
        x = [rec(str(i), a=True, b=False) for i in range(20)]
        y = [rec(str(i), a=False, b=False) for i in range(20)]
        xi, yi = PA.index_by_qid(x, source="x"), PA.index_by_qid(y, source="y")
        direct = PA.contrast(xi, yi, sorted(xi), left_side="a", right_side="a")["exact_full"]
        self.assertAlmostEqual(direct["effect"], 100.0)

    def test_report_contrast_end_to_end_compares_augmented_against_augmented(self):
        """Exercises the CLI path, not just contrast().

        Added because mutation testing found the gap: flipping report_contrast's right_side
        from 'a' to 'b' — which would silently compare one arm's augmented answers against
        the other arm's BASELINE — passed the whole suite. Every assertion below is chosen
        so that mutation fails it.
        """
        # Arm X augmented: all correct. Arm Y augmented: all wrong. BOTH baselines: all wrong.
        # augmented-vs-augmented => +100. augmented-vs-baseline would also be +100 for X-vs-Y,
        # so the discriminating case is the REVERSE direction, asserted second.
        x = [rec(str(i), a=True, b=False) for i in range(20)]
        y = [rec(str(i), a=False, b=True) for i in range(20)]
        with tempfile.TemporaryDirectory() as tmp:
            px = write_arm(tmp, "armX", x, meta={"augmentation": "X"})
            py = write_arm(tmp, "armY", y, meta={"augmentation": "Y"})

            out = PA.report_contrast(px, py)
            self.assertEqual(out["left"], "X")
            self.assertEqual(out["right"], "Y")
            self.assertEqual(out["n"], 20)
            # X augmented (all right) minus Y augmented (all wrong) = +100.
            self.assertAlmostEqual(out["metrics"]["exact_full"]["effect"], 100.0)

            # Reversed: Y augmented (all wrong) minus X augmented (all right) = -100.
            # Against X's BASELINE (all wrong) it would be 0.0, so this pins the side.
            back = PA.report_contrast(py, px)
            self.assertAlmostEqual(back["metrics"]["exact_full"]["effect"], -100.0)

    def test_report_contrast_carries_per_category_rows(self):
        recs_x = [rec(str(i), a=True, b=False, cat="relate") for i in range(10)]
        recs_y = [rec(str(i), a=False, b=False, cat="relate") for i in range(10)]
        with tempfile.TemporaryDirectory() as tmp:
            px = write_arm(tmp, "x", recs_x)
            py = write_arm(tmp, "y", recs_y)
            out = PA.report_contrast(px, py)
            self.assertIn("relate", out["per_category"])
            self.assertAlmostEqual(out["per_category"]["relate"]["effect"], 100.0)


class PerCategoryTests(unittest.TestCase):
    """Category breakdowns must filter correctly and never invent categories."""

    def test_categories_are_filtered_not_pooled(self):
        records = ([rec(str(i), a=True, b=False, cat="relate") for i in range(10)]
                   + [rec(str(100 + i), a=False, b=True, cat="exist") for i in range(10)])
        idx = PA.index_by_qid(records, source="t")
        rows = PA.per_category(idx, idx, sorted(idx), left_side="a", right_side="b")
        self.assertAlmostEqual(rows["relate"]["effect"], 100.0)
        self.assertAlmostEqual(rows["exist"]["effect"], -100.0)
        self.assertEqual(rows["relate"]["n"], 10)

    def test_absent_categories_are_omitted_not_reported_as_zero(self):
        idx = PA.index_by_qid([rec("1", cat="relate")], source="t")
        rows = PA.per_category(idx, idx, ["1"], left_side="a", right_side="b")
        self.assertIn("relate", rows)
        self.assertNotIn("compare", rows)

    def test_category_counts_sum_to_the_paired_total(self):
        records = [rec(str(i), cat=["relate", "exist", "choose"][i % 3]) for i in range(30)]
        idx = PA.index_by_qid(records, source="t")
        rows = PA.per_category(idx, idx, sorted(idx), left_side="a", right_side="b")
        self.assertEqual(sum(r["n"] for r in rows.values()), 30)


class OutputHealthTests(unittest.TestCase):
    """Termination and validity diagnostics."""

    def test_eos_cap_and_empty_rates(self):
        records = [rec("1", stop="eos_151645", cap=False, ans="yes"),
                   rec("2", stop="cap", cap=True, ans="yes"),
                   rec("3", stop="eos_151645", cap=False, ans="   ")]
        h = PA.output_health(records, "a")
        self.assertAlmostEqual(h["eos_rate"], 200 / 3)
        self.assertAlmostEqual(h["cap_hit_rate"], 100 / 3)
        self.assertAlmostEqual(h["empty_rate"], 100 / 3, places=5)

    def test_eos_matches_the_real_stop_reason_vocabulary(self):
        """src/models/vlm.py writes f"eos_{token_id}", never the bare string "eos".

        Matching on equality reported 0.0% termination for job 2286093's arms, every one of
        which in fact stopped cleanly on all 500 questions. Wave 1 independently reported
        100% EOS for the same checkpoint, which is what exposed the discrepancy.
        """
        records = [rec(str(i), stop="eos_151645") for i in range(10)]
        self.assertAlmostEqual(PA.output_health(records, "a")["eos_rate"], 100.0)

    def test_stop_reasons_are_surfaced_not_just_aggregated(self):
        """Only token 151645 is ever supervised; a stop on Qwen's alternate EOS (151643)
        would make "the bridge learned to stop" unfalsifiable, so it must stay visible."""
        records = ([rec(str(i), stop="eos_151645") for i in range(8)]
                   + [rec("8", stop="eos_151643"), rec("9", stop="cap", cap=True)])
        h = PA.output_health(records, "a")
        self.assertEqual(h["stop_reasons"]["eos_151645"], 8)
        self.assertEqual(h["stop_reasons"]["eos_151643"], 1)
        self.assertEqual(h["stop_reasons"]["cap"], 1)
        self.assertAlmostEqual(h["eos_rate"], 90.0)

    def test_cap_stops_are_not_counted_as_eos(self):
        records = [rec(str(i), stop="cap", cap=True) for i in range(5)]
        self.assertAlmostEqual(PA.output_health(records, "a")["eos_rate"], 0.0)

    def test_generated_token_stats(self):
        records = [rec("1", gen=2), rec("2", gen=4), rec("3", gen=30)]
        h = PA.output_health(records, "a")
        self.assertEqual(h["median_generated"], 4)
        self.assertAlmostEqual(h["mean_generated"], 12.0)


class ContextStatsTests(unittest.TestCase):
    """Coverage and injected-context length, computed from the SAVED context."""

    def test_coverage_counts_only_non_empty_context(self):
        records = [rec("1", ctx="objects: man"), rec("2", ctx=""), rec("3", ctx="   ")]
        cs = PA.context_stats(records)
        self.assertEqual(cs["n_with_graph"], 1)
        self.assertAlmostEqual(cs["coverage_pct"], 100 / 3)

    def test_token_counts_use_the_supplied_tokenizer(self):
        class FakeTok:
            def encode(self, text, add_special_tokens=False):
                return text.split()
        cs = PA.context_stats([rec("1", ctx="a b c"), rec("2", ctx="a b c d e")], FakeTok())
        self.assertAlmostEqual(cs["mean_tokens"], 4.0)
        self.assertEqual(cs["max_tokens"], 5)

    def test_no_tokenizer_still_reports_characters(self):
        cs = PA.context_stats([rec("1", ctx="abcd")], None)
        self.assertIn("mean_chars", cs)
        self.assertNotIn("mean_tokens", cs)


class RecordLoadingTests(unittest.TestCase):
    """Both artifact shapes must load; provenance must survive."""

    def test_loads_the_meta_plus_records_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = write_arm(tmp, "arm", [rec("1")], meta={"augmentation": "scene_graph_pred",
                                                        "pred_cache": "tune500.json"})
            records, meta = PA.load_records(p)
            self.assertEqual(len(records), 1)
            self.assertEqual(meta["pred_cache"], "tune500.json")

    def test_loads_the_legacy_bare_list_form(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "legacy.json"
            p.write_text(json.dumps([rec("1")]))
            records, meta = PA.load_records(p)
            self.assertEqual(len(records), 1)
            self.assertEqual(meta, {})


class ConfigurationTests(unittest.TestCase):
    """The preregistered CI protocol must not drift."""

    def test_resamples_and_seed_match_the_project_protocol(self):
        self.assertEqual(PA.N_RESAMPLES, 10_000)
        self.assertEqual(PA.SEED, 42)

    def test_all_three_metrics_are_analysed(self):
        self.assertEqual(set(PA.METRICS), {"exact_full", "exact", "vqa"})


if __name__ == "__main__":
    unittest.main()
