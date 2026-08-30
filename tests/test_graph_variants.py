"""Contract tests for the T-049 Rank 1 graph-variant builder (script 37).

Rank 1 asks whether non-oracle graph evidence causally helps. That question is answerable only
against a corruption matched on everything except image identity, so the failure modes that
matter are the ones that would leave a *plausible-looking* control that controls nothing:

  - the qid/image key-space trap: a direct intersection of the two caches returns ZERO, so a
    naive join silently compares nothing while every downstream number still prints
  - a donor that is its own image, which makes the control identical to the treatment
  - a corruption that changes prompt length as well as content, so a length effect is read as
    a graph-quality effect
  - the oracle arm leaking into a comparison, where it is a privileged upper bound
  - the diagnostic being run against a spent evaluation slice
"""
from pathlib import Path
import ast
import json
import re
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "37_graph_variants.py"
OUT = ROOT / "outputs" / "graph_variants"
MANIFEST = OUT / "MANIFEST.json"


def _code():
    """Source with comment lines stripped, so no test passes by matching prose."""
    t = SCRIPT.read_text(encoding="utf-8")
    return "\n".join(l for l in t.splitlines() if not l.strip().startswith("#"))


# `no_graph` is deliberately absent: it is the b_* side of each run, not its own file.
RECORD_ARMS = ("pred", "pred_q", "objects_only", "objects_filtered",
               "relations_shuffled", "wrong_image", "oracle")

# Questions each arm converts from wrong to right, relative to the shared b_* baseline. Nested
# and distinct, so every printed contrast has one exact expected value.
ARM_GAINS = {"pred": 30, "wrong_image": 10, "pred_q": 20, "objects_only": 5,
             "objects_filtered": 15, "relations_shuffled": 25, "oracle": 50}


def write_fixture(d, mutate=None):
    """Write one production-shaped records file per arm, optionally mutated.

    Every field the analysis requires is present and correct by default. A fixture with partial
    metadata is worse than no fixture: it lets a validator that skips missing fields go green
    while accepting unprovenanced records, which is exactly how the previous guard passed.

    QIDs come from the REAL variant caches, so the sensitivity step (which reads the caches to
    find genuinely permuted graphs) sees the same key space it will see in production.
    """
    import random
    if not MANIFEST.is_file():
        raise unittest.SkipTest("variants not built")
    m = json.loads(MANIFEST.read_text())
    qids = sorted(json.loads((ROOT / m["arms"]["pred"]["path"]).read_text()))
    rnd = random.Random(0)
    base = {q: rnd.random() < 0.45 for q in qids}
    cats = ["relate", "compare", "exist", "choose", "query"]
    # Every arm used to carry the identical per-question outcome, so every contrast the
    # end-to-end test exercised was exactly 0.00 and an arm-lookup that read the wrong dict
    # produced the same table. Each arm now gains a DIFFERENT number of questions, nested so
    # the planted values are exact: primary (pred-wrong_image) = +4.00 and improvement
    # (pred - the common b_* side) = +6.00.
    wrong_first = [q for q in qids if not base[q]]
    for arm in RECORD_ARMS:
        gained = set(wrong_first[:ARM_GAINS[arm]])
        a_ok = {q: bool(base[q] or q in gained) for q in qids}
        recs = [{"qid": q, "category": cats[i % len(cats)], "question": "?", "gold": "x",
                 "image_path": "/img.jpg",
                 "a_ans": "x", "a_raw": "x", "a_context": "a cat on a mat",
                 "a_exact": a_ok[q], "a_vqa": a_ok[q], "a_exact_full": a_ok[q],
                 "a_stop_reason": "eos_151645", "a_cap_hit": False, "a_n_generated": 2,
                 # The b_* side is ONE shared no-graph run and must stay byte-identical across
                 # arms; the analysis refuses the set otherwise.
                 "b_ans": "y", "b_raw": "y", "b_exact": base[q], "b_vqa": base[q],
                 "b_exact_full": base[q], "b_stop_reason": "eos_151645",
                 "b_cap_hit": False, "b_n_generated": 2}
                for i, q in enumerate(qids)]
        meta = dict(m["expected_record_meta"])
        meta.update(m["expected_common_meta"])
        meta.update(m["expected_arm_meta"][arm])
        blob = {"_meta": meta, "records": recs}
        if mutate:
            mutate(blob, arm)
        (d / f"bridge_clip__{arm}__tune_500_records.json").write_text(json.dumps(blob))
    return qids


def run_analysis(mutate=None):
    """Run script 38 end to end against a fixture, returning the CompletedProcess."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        write_fixture(d, mutate)
        return subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "38_graph_trust_analysis.py"),
             "--records-dir", str(d)], capture_output=True, text=True, cwd=str(ROOT))


class BuilderContractTests(unittest.TestCase):

    def test_parses(self):
        ast.parse(SCRIPT.read_text(encoding="utf-8"))

    def test_refuses_a_spent_slice(self):
        code = _code()
        self.assertIn('for bad in ("eval_2000", "locked", "confirm_3000")', code)
        self.assertIn("names a spent slice", SCRIPT.read_text(encoding="utf-8"))

    def test_joins_through_the_question_file_and_asserts_non_empty(self):
        """The defect this guards is silent: a zero join still prints a full table."""
        code = _code()
        self.assertIn("def qid_to_image", code)
        self.assertIn("join produced ZERO graphs", SCRIPT.read_text(encoding="utf-8"))

    def test_never_lets_a_donor_be_its_own_image(self):
        code = _code()
        self.assertIn("q2img[donor_map[q]] == q2img[q]", code)
        self.assertIn("the control is not a control", SCRIPT.read_text(encoding="utf-8"))

    def test_donor_assignment_is_one_to_one(self):
        """Independent nearest-neighbour picking reused one donor six times, which makes the
        control's graph distribution unlike the treatment's."""
        code = _code()
        self.assertIn("donor set is not one-to-one", SCRIPT.read_text(encoding="utf-8"))
        self.assertIn("len(used) != len(pool)", code)

    def test_measures_with_the_evaluation_renderer(self):
        """SceneGraphStore keeps attributes; PredictedGraphStore strips them, and the latter
        is what the evaluation shows. Measuring with the wrong one inflated the length match."""
        code = _code()
        self.assertIn("PredictedGraphStore", code)
        self.assertNotIn("SceneGraphStore(str(p))", code)
        self.assertIn("i.e. the evaluated text", SCRIPT.read_text(encoding="utf-8"))

    def test_object_pruning_and_relation_removal_are_separate_arms(self):
        """Detector confidence is not confidence in deterministic geometry relations."""
        code = _code()
        self.assertIn("def strip_relations", code)
        self.assertIn("def filter_objects", code)

    def test_records_rendered_token_length_for_every_arm(self):
        self.assertIn("rendered_tokens", _code())

    def test_seed_is_fixed(self):
        self.assertRegex(_code(), r"SEED\s*=\s*42")
        self.assertIn("default_rng(SEED)", _code())


class ManifestTests(unittest.TestCase):
    """The built artifact must say what it is, and refuse to overstate itself."""

    @classmethod
    def setUpClass(cls):
        if not MANIFEST.is_file():
            raise unittest.SkipTest("variants not built")
        cls.m = json.loads(MANIFEST.read_text())

    def test_declares_itself_exploratory(self):
        self.assertIn("EXPLORATORY ONLY", self.m["status"])
        self.assertIn("no fresh endpoint exists", self.m["status"])

    def test_oracle_is_marked_ceiling_only(self):
        p = self.m["oracle_policy"]
        # "never a comparator" was false of the script's own headroom output (review record).
        self.assertIn("FORBIDDEN as a primary, promotion or inferential secondary", p)
        self.assertIn("descriptive", p)

    def test_primary_estimand_is_the_matched_contrast(self):
        self.assertIn("pred MINUS wrong_image", self.m["primary_estimand"])

    def test_development_surface_is_not_a_spent_slice(self):
        s = self.m["development_surface"]
        for bad in ("eval_2000", "locked", "confirm_3000"):
            self.assertNotIn(bad, s)

    def test_every_arm_is_hash_pinned(self):
        for name, a in self.m["arms"].items():
            self.assertRegex(a["sha256"], r"^[0-9a-f]{64}$", name)

    def test_donor_map_is_pinned_separately(self):
        w = self.m["arms"]["wrong_image"]
        self.assertIn("donor_map", w)
        self.assertRegex(w["donor_map_sha256"], r"^[0-9a-f]{64}$")

    def test_corruption_is_length_matched_to_the_treatment(self):
        """If wrong_image were much shorter/longer, a length effect would masquerade."""
        pred = self.m["arms"]["pred"]["rendered_tokens"]["mean"]
        wrong = self.m["arms"]["wrong_image"]["rendered_tokens"]["mean"]
        self.assertLess(abs(pred - wrong) / pred, 0.05,
                        f"wrong_image mean {wrong} vs pred {pred} — length is confounded")

    def test_shuffle_preserves_length_exactly(self):
        """Permuting relation NAMES must not change object count or graph size."""
        pred = self.m["arms"]["pred"]["rendered_tokens"]
        shuf = self.m["arms"]["relations_shuffled"]["rendered_tokens"]
        self.assertEqual(pred["min"], shuf["min"])
        self.assertEqual(pred["max"], shuf["max"])

    def test_shuffle_no_op_rate_is_recorded(self):
        """Graphs with <2 relations cannot be permuted and attenuate that contrast."""
        a = self.m["arms"]["relations_shuffled"]
        self.assertIn("n_actually_permuted", a)
        self.assertLess(a["n_actually_permuted"], a["n_questions"],
                        "if every graph permuted, this assertion should be revisited")

    def test_filtered_arm_states_whether_it_is_really_a_confidence_filter(self):
        a = self.m["arms"]["objects_filtered"]
        self.assertIn("confidence_scores_available", a)
        if not a["confidence_scores_available"]:
            self.assertIn("NOT a confidence filter", a["note"])


class BuiltCacheTests(unittest.TestCase):
    """Properties of the actual files, not of the code that claims to write them."""

    @classmethod
    def setUpClass(cls):
        if not MANIFEST.is_file():
            raise unittest.SkipTest("variants not built")
        cls.m = json.loads(MANIFEST.read_text())
        cls.g = {n: json.loads((ROOT / a["path"]).read_text())
                 for n, a in cls.m["arms"].items()}

    def test_all_arms_share_one_key_space(self):
        keysets = {n: frozenset(g) for n, g in self.g.items()}
        self.assertEqual(len(set(keysets.values())), 1,
                         "arms are keyed differently; contrasts would compare different questions")

    def test_objects_only_has_no_relations_left(self):
        left = sum(len(o.get("relations", []))
                   for g in self.g["objects_only"].values()
                   for o in g.get("objects", {}).values())
        self.assertEqual(left, 0)

    def test_shuffle_preserves_object_and_relation_counts(self):
        for q, g in self.g["pred"].items():
            s = self.g["relations_shuffled"][q]
            self.assertEqual(len(g.get("objects", {})), len(s.get("objects", {})), q)
            cnt = lambda x: sum(len(o.get("relations", [])) for o in x.get("objects", {}).values())
            self.assertEqual(cnt(g), cnt(s), q)

    def test_shuffle_actually_changed_relation_labels_somewhere(self):
        """A permutation that never permutes is a control that controls nothing."""
        changed = sum(1 for q, g in self.g["pred"].items()
                      if json.dumps(g, sort_keys=True)
                      != json.dumps(self.g["relations_shuffled"][q], sort_keys=True))
        self.assertGreater(changed, 0)
        self.assertEqual(changed, self.m["arms"]["relations_shuffled"]["n_actually_permuted"])

    def test_no_donor_graph_equals_the_true_graph_for_that_question(self):
        same = [q for q, g in self.g["pred"].items()
                if json.dumps(g, sort_keys=True)
                == json.dumps(self.g["wrong_image"][q], sort_keys=True)]
        self.assertEqual(same, [], f"{len(same)} donors are byte-identical to the true graph")

    def test_filtered_keeps_only_relations_among_survivors(self):
        for q, g in self.g["objects_filtered"].items():
            ids = set(g.get("objects", {}))
            for o in g.get("objects", {}).values():
                for r in o.get("relations", []):
                    self.assertIn(str(r.get("object")), ids, f"{q}: dangling relation target")


if __name__ == "__main__":
    unittest.main()


class AnalysisContractTests(unittest.TestCase):
    """The analysis path must be verifiable before any GPU time is requested."""

    SCRIPT = ROOT / "scripts" / "38_graph_trust_analysis.py"

    def test_parses(self):
        ast.parse(self.SCRIPT.read_text(encoding="utf-8"))

    def test_self_test_recovers_a_planted_effect(self):
        """Runs the real statistics path on synthetic records — no model, no GPU."""
        import subprocess
        r = subprocess.run([sys.executable, str(self.SCRIPT), "--self-test"],
                           capture_output=True, text=True, cwd=str(ROOT))
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        # Distinct planted values: a swap between the primary and improvement contrast, or an
        # arm-lookup reading the wrong dict, cannot produce both.
        self.assertIn("recovered +8.00 on the primary and +6.00 on the improvement", r.stdout)
        self.assertIn("SELF-TEST PASSED", r.stdout)

    def test_refuses_the_oracle_as_a_comparator(self):
        code = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("may not be the primary comparator", code)
        self.assertIn("forbidden as a primary, promotion or inferential secondary", code)
        # The oracle is excluded from the secondary set by construction, and appears only in
        # the headroom diagnostic where it is never differenced against another arm.
        self.assertIn('a not in (PRIMARY[0], PRIMARY[1], "no_graph", ORACLE)', code)
        self.assertIn("differenced here and NOWHERE", code)
        # Whitespace-flattened, because the assertive form and the corrective one differ by a
        # line wrap and a bare substring test would match the sentence that retracts the claim.
        flat = " ".join(code.split())
        self.assertNotIn("it is never a comparator", flat,
                         "the script differences the oracle in the headroom diagnostic")
        self.assertIn("'never a comparator' was literally false", flat,
                      "the retraction must stay next to the code it corrects")

    def test_primary_contrast_is_fixed_in_source(self):
        self.assertRegex(self.SCRIPT.read_text(encoding="utf-8"),
                         r'PRIMARY\s*=\s*\("pred",\s*"wrong_image"\)')

    def test_one_shared_matrix_with_a_printed_hash(self):
        code = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("matrix sha256", code)
        self.assertIn("hashlib.sha256(idx.tobytes())", code)

    def test_applies_no_invented_numeric_decision_threshold(self):
        """Headroom must be reported, never thresholded.

        An earlier version stopped the ladder when oracle headroom fell below five points — a
        decision rule invented after the plan was approved. Stop rules belong in the approved
        plan, not in the analysis code.
        """
        import ast as _ast
        tree = _ast.parse(self.SCRIPT.read_text(encoding="utf-8"))
        # Scope to analyse(): that is where a decision rule would act on real results. The
        # self-test's sanity assertions on synthetic fixtures are not decision thresholds.
        fn = next(n for n in _ast.walk(tree)
                  if isinstance(n, _ast.FunctionDef) and n.name == "analyse")
        for node in _ast.walk(fn):
            if isinstance(node, _ast.Compare):
                src = _ast.unparse(node)
                consts = [c.value for c in node.comparators
                          if isinstance(c, _ast.Constant) and isinstance(c.value, (int, float))]
                if "head" not in src or not consts:
                    continue
                # `head > 0` is a DOMAIN guard, not a decision rule: the "captures X% of it"
                # ratio divides by head, and with head <= 0 the two negatives cancel and a
                # predicted arm that also hurt prints a confident positive fraction. Comparing
                # against zero asks "is the denominator usable"; comparing against any other
                # number is the invented stop rule this test exists to ban.
                if all(c == 0 for c in consts):
                    continue
                self.fail(f"numeric threshold on headroom: {src!r}")
        self.assertIn("No numeric threshold is applied", self.SCRIPT.read_text(encoding="utf-8"))

    def test_reports_a_null_as_a_null(self):
        code = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("INTERVAL INCLUDES ZERO", code)
        self.assertIn("does not license Ranks 2-6", code)

    def test_asserts_the_metric_field_exists(self):
        self.assertIn("a comparison would be vacuous", self.SCRIPT.read_text(encoding="utf-8"))


class EndToEndSchemaTests(unittest.TestCase):
    """Fixture built to the evaluator's REAL record schema, which the analysis must consume.

    `scripts/09_augment_eval.py` writes a_* (augmented) and b_* (baseline) into one file per
    arm. An earlier analysis expected bridge_* and would have failed only after GPU spend.
    """

    SCRIPT = ROOT / "scripts" / "38_graph_trust_analysis.py"
    ARMS = RECORD_ARMS

    def test_analysis_reads_the_real_schema_and_anchors_the_stem(self):
        r = run_analysis()
        out = r.stdout + r.stderr
        # A clean exit is REQUIRED. Asserting only the absence of certain strings lets a
        # late failure — after loading, in a contrast or in serialisation — pass silently.
        self.assertEqual(r.returncode, 0,
                         f"analysis must run end to end on the real schema:\n{out[-2500:]}")
        self.assertNotIn("a comparison would be vacuous", out)
        self.assertNotIn("expected exactly one records file", out)
        # And it must actually have produced the estimands, not merely exited zero.
        for expect in ("PRIMARY   pred - wrong_image", "IMPROVEMENT  pred - no_graph",
                       "HEADROOM DIAGNOSTIC", "SENSITIVITY"):
            self.assertIn(expect, out, f"missing {expect!r} in a clean run")
        self.assertNotIn("wrong_image - pred", out,
                         "the reversed primary is duplicated as a secondary")
        # The planted values, recovered exactly. Previously every arm was identical, so this
        # assertion could not have existed and a swapped arm printed the same table.
        self.assertIn("PRIMARY   pred - wrong_image          +4.00", out,
                      "planted +4.00 on the primary (30 gained minus 10) was not recovered")
        self.assertIn("IMPROVEMENT  pred - no_graph              +6.00", out,
                      "planted +6.00 on the improvement contrast was not recovered")

    def test_the_printed_packet_carries_generation_and_category_evidence(self):
        """`a_mean_generated` was computed and then dropped from the table, and the category
        block held one augmented percentage with nothing to pair it against."""
        out = run_analysis().stdout
        self.assertIn("mean tok", out, "generation length is missing from the accuracy table")
        self.assertIn("no_graph (common B)", out, "the common baseline's own summary is missing")
        self.assertIn("PER-CATEGORY PAIRED EFFECTS", out)
        self.assertIn("POST-HOC", out, "category effects must be labelled post-hoc")
        for cat in ("relate", "compare", "exist"):
            self.assertIn(cat, out, f"category {cat} missing from the breakdown")

    def test_unanchored_pred_would_have_matched_pred_q(self):
        """The bug being guarded, demonstrated rather than asserted."""
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_fixture(d)
            loose = sorted(p.name for p in d.glob("*pred*records.json"))
            anchored = sorted(p.name for p in d.glob("*__pred__*records.json"))
            self.assertEqual(len(loose), 2, "fixture should expose the ambiguity")
            self.assertEqual(len(anchored), 1, "anchored match must select exactly one arm")

    def test_analysis_source_uses_a_and_b_not_bridge(self):
        code = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn('field = f"{side}_{metric}"', code)
        self.assertNotIn('f"bridge_{metric}"', code)

    def test_requires_identical_qid_sets_not_an_intersection(self):
        code = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("do not cover identical QID sets", code)
        self.assertNotIn("set.intersection", code)

    def test_reports_improvement_as_well_as_dependence(self):
        code = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("IMPROVEMENT", code)
        self.assertIn("no_graph", code)
        self.assertIn("dependence without improvement is possible", code)

    def test_sensitivity_subset_is_computed_not_read(self):
        code = self.SCRIPT.read_text(encoding="utf-8")
        self.assertIn("computed from the caches rather than read from the manifest", code)
        self.assertIn("the sensitivity subset would be wrong", code)


class MetadataGuardBidirectionalTests(unittest.TestCase):
    """The guard must reject BOTH unprovenanced and wrongly-provenanced records.

    The previous version failed both ways at once: `if expect and got and got != expect` let
    missing metadata through, and comparing checkpoint_tag ('w1_bf16_v1') against the full stem
    ('bridge_clip_w1_bf16_v1_ep5') rejected genuine production records. A correct GPU run would
    have failed analysis; an unprovenanced one would have passed.
    """

    SCRIPT = ROOT / "scripts" / "38_graph_trust_analysis.py"
    ARMS = RECORD_ARMS

    def _run(self, mutate=None):
        return run_analysis(mutate)

    def test_correct_production_metadata_is_ACCEPTED(self):
        """The direction nobody tests for: a valid run must not be rejected."""
        r = self._run()
        self.assertEqual(r.returncode, 0,
                         f"genuine production metadata was rejected:\n{(r.stdout + r.stderr)[-2000:]}")

    def test_missing_metadata_is_REJECTED(self):
        r = self._run(mutate=lambda blob, arm: blob.__setitem__("_meta", {}))
        self.assertNotEqual(r.returncode, 0, "records with no provenance were accepted")
        self.assertIn("record metadata lacks", r.stdout + r.stderr)

    def test_wrong_checkpoint_is_REJECTED(self):
        def bad(blob, arm):
            blob["_meta"]["checkpoint_tag"] = "some_other_run"
        r = self._run(mutate=bad)
        self.assertNotEqual(r.returncode, 0, "a wrong checkpoint tag was accepted")

    def test_wrong_evidence_layer_is_REJECTED(self):
        def bad(blob, arm):
            blob["_meta"]["evidence_layer"] = "legacy_raw_prompt_no_eos"
        r = self._run(mutate=bad)
        self.assertNotEqual(r.returncode, 0, "a legacy-layer record was accepted")

    def test_disagreeing_cross_arm_baseline_is_REJECTED(self):
        """b_* is one shared no-graph run; if arms disagree they share no control."""
        def bad(blob, arm):
            if arm == "oracle":
                for r_ in blob["records"][:50]:
                    r_["b_exact_full"] = not r_["b_exact_full"]
        r = self._run(mutate=bad)
        self.assertNotEqual(r.returncode, 0, "arms with different baselines were accepted")
        self.assertIn("were not scored against a common control", r.stdout + r.stderr)

    def test_context_coverage_disagreement_is_REJECTED(self):
        def bad(blob, arm):
            blob["_meta"]["n_context_injected"] = 0
        r = self._run(mutate=bad)
        self.assertNotEqual(r.returncode, 0, "a collapsed injection rate was accepted")


@unittest.skipUnless(
    MANIFEST.is_file(),
    "requires generated graph-variant records; outputs are not distributed in Git",
)
class ProductionShapedCounterexampleTests(unittest.TestCase):
    """Four invalid runs that a GPU could actually produce, each of which exited zero.

    None of them is a malformed file. Each is what the analysis would see if the launcher fed
    one cache to every arm, if the QID set drifted, if one question fell back to a plain
    prompt, or if the arms were scored against different no-graph passes. All four were found
    by audit, not by the 654 tests that were green at the time.
    """

    def _run(self, mutate):
        return run_analysis(mutate)

    def test_every_arm_claiming_the_same_predicted_cache_is_REJECTED(self):
        """Hashing local caches proves they did not change, not which one the GPU read."""
        m = json.loads(MANIFEST.read_text())
        one = m["arms"]["pred"]["path"]

        def bad(blob, arm):
            if arm != "oracle":
                blob["_meta"]["pred_cache"] = one
        r = self._run(bad)
        out = r.stdout + r.stderr
        self.assertNotEqual(r.returncode, 0, "six arms fed by one graph cache were accepted")
        self.assertIn("do not establish which graph the evaluator read", out)

    def test_the_oracle_masquerading_as_a_derived_arm_is_REJECTED(self):
        """The ceiling arm must not be able to enter under a treatment arm's name."""
        def bad(blob, arm):
            if arm == "pred":
                blob["_meta"]["augmentation"] = "scene_graph"
                blob["_meta"]["oracle"] = True
        r = self._run(bad)
        self.assertNotEqual(r.returncode, 0, "an oracle run was accepted as the pred arm")

    def test_a_foreign_qid_shared_by_every_arm_is_REJECTED(self):
        """A shared set of 500 is not the same claim as the pinned tuning 500."""
        def bad(blob, arm):
            blob["records"][7]["qid"] = "not_a_tuning_question_0001"
        r = self._run(bad)
        out = r.stdout + r.stderr
        self.assertNotEqual(r.returncode, 0, "a fake QID shared by every arm was accepted")
        self.assertIn("was not evaluated on the pinned development surface", out)

    def test_consistent_499_of_500_coverage_is_REJECTED(self):
        """_meta and the records agreeing on 99.8% is agreement, not completeness."""
        def bad(blob, arm):
            blob["_meta"]["n_context_injected"] = 499
            blob["records"][3]["a_context"] = ""
        r = self._run(bad)
        out = r.stdout + r.stderr
        self.assertNotEqual(r.returncode, 0, "a 499-of-500 run was accepted as complete")
        self.assertIn("499", out)

    def test_baseline_answer_bytes_differing_is_REJECTED(self):
        """b_exact_full equal is not b_* identical; 'byte-identical' has to mean it."""
        def bad(blob, arm):
            if arm == "oracle":
                for r_ in blob["records"][:50]:
                    r_["b_ans"] = "rewritten"
                    r_["b_raw"] = "rewritten raw generation"
        r = self._run(bad)
        out = r.stdout + r.stderr
        self.assertNotEqual(r.returncode, 0,
                            "arms whose baseline answers differ were called byte-identical")
        self.assertIn("b_ans", out)
        self.assertIn("50 of 500", out)

    def test_a_missing_baseline_field_is_REJECTED(self):
        """The comparison can only be as complete as the fields it requires."""
        def bad(blob, arm):
            for r_ in blob["records"]:
                r_.pop("b_n_generated")
        r = self._run(bad)
        self.assertNotEqual(r.returncode, 0, "records without the full B side were accepted")

    def test_a_tampered_qid_manifest_is_REJECTED(self):
        """The QID hash was stored and never verified."""
        import subprocess
        import tempfile
        m = json.loads(MANIFEST.read_text())
        surface = ROOT / m["development_surface"]
        original = surface.read_bytes()
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            write_fixture(d)
            try:
                qids = json.loads(original)
                surface.write_text(json.dumps(qids[:-1] + ["not_a_tuning_question_0001"]))
                r = subprocess.run(
                    [sys.executable, str(ROOT / "scripts" / "38_graph_trust_analysis.py"),
                     "--records-dir", str(d)],
                    capture_output=True, text=True, cwd=str(ROOT))
            finally:
                surface.write_bytes(original)
        self.assertNotEqual(r.returncode, 0, "a changed development surface was accepted")
        self.assertIn("changed since the build", r.stdout + r.stderr)
