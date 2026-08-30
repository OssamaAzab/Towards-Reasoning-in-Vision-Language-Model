"""Tests for Stage A crop construction, controls and analysis machinery.

Every test here guards a way Stage A could produce a control that is not a control, or a claim
that rests on something other than the crop's content: an irrelevant crop that still shows the
referent, a wrong crop from the question's own image, unequal token counts across the three crop
arms, or a crop chosen using the answer.

CPU-ONLY BY PROJECT RULE. Nothing here runs inference or asserts anything about answers, scores
or EOS behaviour — CPU and cluster-GPU results are not numerically interchangeable in this
project. These tests cover geometry, coverage, derangements, leakage, token shapes, frozen
parameters, bootstrap code and truncation-policy logic, which is exactly what CPU may decide.
"""
import ast
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def reads_field(source: str, field: str) -> bool:
    """True if CODE reads obj[field] or obj.get(field). Docstrings and comments cannot match.

    A plain substring scan would be satisfied — or, worse, violated — by prose: crop_augment.py's
    own docstring says it never reads question["answer"], and a text scan flags that sentence as
    the violation it describes. A guard that reads prose instead of code is not a guard.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            if node.slice.value == field:
                return True
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "get" and node.args
                and isinstance(node.args[0], ast.Constant) and node.args[0].value == field):
            return True
    return False


def code_text(source: str) -> str:
    """Executable source only: comments AND docstrings removed.

    Ordering assertions that index raw text match prose: this file's own comment about a printed
    "GROUNDED" being quoted out of context was itself matched as the GROUNDED branch. Round-
    tripping through the AST drops comments — but `ast.unparse` KEEPS docstrings, so on its own it
    leaves the same hole one layer down. Docstring statements are therefore deleted from every
    module, class and function first, and a test asserts a planted docstring cannot satisfy this.
    """
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(tree)


def test_code_text_strips_docstrings_too():
    """Guard the guard: prose in a docstring must not satisfy a containment check."""
    src = '"""module X"""\ndef f():\n    """function X"""\n    # comment X\n    return 1\n'
    assert "X" not in code_text(src)
    assert "return 1" in code_text(src)
    assert "X" in ast.unparse(ast.parse(src)), "plain unparse would have kept it"


def code_identifiers(source: str) -> set:
    """Every identifier referenced in code, excluding docstrings and comments."""
    tree = ast.parse(source)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.alias):
            names.add(node.name.split(".")[-1])
            if node.asname:
                names.add(node.asname)
    return names

from src.data.crop_augment import (AREA_TOL, CROP_MARGIN, MAX_CROP_AREA_FRAC, MIN_CROP_PX,
                                   SCALE_MATCH_FACTOR, build_wrong_crop_derangement, crop_area,
                                   crop_is_usable, crop_record, encoder_input_side,
                                   expand_and_clip, irrelevant_box, is_scale_matched, make_crop,
                                   pad_to_square, padded_geometry, relevant_box,
                                   scale_pair_diagnostics)


def obj(name, x, y, w, h):
    return {"name": name, "x": x, "y": y, "w": w, "h": h, "attributes": [], "relations": []}


def graph(objects, w=600, h=400):
    return {"width": w, "height": h, "objects": objects}


def question(ids, answer="yes"):
    return {"annotations": {"question": {str(i): oid for i, oid in enumerate(ids)}},
            "semantic": [], "answer": answer, "fullAnswer": f"The answer is {answer}."}


FIXTURE_CONDS = ["global_only", "global_plus_relevant_crop", "global_plus_irrelevant_crop",
                 "global_plus_wrong_crop", "relevant_crop_only"]


def analysis_fixture(n=40, encoders=("clip",), smoke=False, hit=None, in_scale_matched=None):
    """A records blob scripts/58 accepts, with the cohort flags it now requires.

    `hit(i, cond, in_sm) -> bool` decides exact_full per question and arm, so a test can plant any
    pattern of effect — including one that differs between the full cohort and the subgroup.
    `in_scale_matched(i) -> bool` decides subgroup membership.
    """
    hit = hit or (lambda i, c, sm: c in ("global_plus_relevant_crop", "relevant_crop_only"))
    in_scale_matched = in_scale_matched or (lambda i: True)
    recs, sm_qids = [], []
    for i in range(n):
        q = f"q{i:03d}"
        sm = bool(in_scale_matched(i))
        if sm:
            sm_qids.append(q)
        # Padded areas consistent with the flag, so a test that re-derives membership from the
        # geometry sees the same answer the flag reports.
        a_rel, a_wrong = 10_000, (15_000 if sm else 90_000)
        side = lambda a: int(a ** 0.5)  # noqa: E731
        geom = lambda a: {"crop_w": side(a), "crop_h": side(a), "crop_area": a,  # noqa: E731
                          "padded_w": side(a), "padded_h": side(a), "padded_area": a,
                          "aspect_ratio": 1.0, "crop_occupancy": 1.0, "object_occupancy": 0.8,
                          "encoder_side": None, "resize_factor": None}
        r = {"qid": q, "image_id": f"img{i}", "question": "?", "gold": "yes",
             "category": "rel", "structural": "query", "target_label": "dog",
             "wrong_donor_label": "dog", "wrong_donor_qid": f"q{(i + 1) % n:03d}",
             "wrong_donor_image_id": f"img{(i + 1) % n}", "relevant_name": "dog",
             "relevant_box": [0, 0, 9, 9], "wrong_box": [1, 1, 9, 9], "wrong_area_diff": 1,
             "in_irrelevant_subset": True, "irrelevant_name": "chair",
             "irrelevant_box": [5, 5, 9, 9], "area_match_error": 0.1,
             "in_scale_matched": sm,
             "scale": {"relevant": geom(a_rel), "wrong": geom(a_wrong),
                       "padded_side_ratio": (a_wrong / a_rel) ** 0.5,
                       "padded_area_ratio": a_wrong / a_rel,
                       "aspect_ratio_diff": 0.0, "scale_match_factor": 2.0,
                       "scale_matched": sm}}
        for e in encoders:
            r[f"{e}__tokens_global"] = 256
            r[f"{e}__tokens_crop"] = 256
            r[f"{e}__baseline_ans"] = "yes"
            r[f"{e}__baseline_stop_reason"] = "eos_151645"
            for c in FIXTURE_CONDS:
                ok = bool(hit(i, c, sm))
                r[f"{e}__{c}__ans"] = "yes" if ok else "no"
                r[f"{e}__{c}__exact_full"] = ok
                r[f"{e}__{c}__stop_reason"] = "eos_151645"
                r[f"{e}__{c}__n_generated"] = 2
        recs.append(r)
    meta = {"oracle": True, "not_deployable": "fixture", "metric_version": "v", "seed": 42,
            "n_questions": n, "smoke_test": smoke, "n_core": n, "n_irrelevant_subset": n,
            "n_scale_matched": len(sm_qids), "scale_match_factor": 2.0,
            "cohorts": {"core": [r["qid"] for r in recs],
                        "irrelevant_subset": [r["qid"] for r in recs],
                        "scale_matched": sm_qids},
            "conditions": FIXTURE_CONDS,
            "encoders": {e: {"stem": e, "sha256": "x" * 64} for e in encoders},
            "coverage": "c", "coverage_sha256": "y" * 64, "slice": "s", "slice_sha256": "z" * 64,
            "crop_rule": {}, "token_counts": {}, "view_order": "global first",
            "models_frozen": True, "baseline_parity": {"checked": n, "mismatches": 0}}
    return {"_meta": meta, "records": recs}


def run_analysis(blob):
    """Run scripts/58 on a fixture blob and return its CompletedProcess."""
    import json as _json
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rec.json"
        p.write_text(_json.dumps(blob))
        return subprocess.run([sys.executable, str(ROOT / "scripts" / "58_crop_analysis.py"),
                               "--records", str(p)], capture_output=True, text=True, cwd=str(ROOT))


class CropGeometryTests(unittest.TestCase):
    """The fixed crop rule: expand -> clip -> pad. Changing it after seeing accuracy is barred."""

    def test_expansion_uses_the_declared_margin(self):
        b = expand_and_clip((100, 100, 200, 200), 600, 400)
        dw = int(round(100 * CROP_MARGIN))
        self.assertEqual(b, (100 - dw, 100 - dw, 200 + dw, 200 + dw))

    def test_boundary_clipping_never_leaves_the_image(self):
        """An object against the frame edge must not produce negative or overflowing coordinates."""
        for box in [(0, 0, 50, 50), (560, 360, 600, 400), (2, 2, 40, 40)]:
            x1, y1, x2, y2 = expand_and_clip(box, 600, 400)
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, 600)
            self.assertLessEqual(y2, 400)

    def test_a_crop_covering_most_of_the_image_is_rejected(self):
        big = (0, 0, 599, 399)
        self.assertGreater(crop_area(big), MAX_CROP_AREA_FRAC * 600 * 400)
        self.assertFalse(crop_is_usable(big, 600, 400))

    def test_a_tiny_crop_is_rejected(self):
        self.assertFalse(crop_is_usable((10, 10, 10 + MIN_CROP_PX - 1, 60), 600, 400))
        self.assertTrue(crop_is_usable((10, 10, 10 + MIN_CROP_PX, 10 + MIN_CROP_PX), 600, 400))

    def test_padding_preserves_aspect_ratio_and_yields_a_square(self):
        from PIL import Image
        im = Image.new("RGB", (40, 100), (10, 20, 30))
        out = pad_to_square(im)
        self.assertEqual(out.size, (100, 100))
        # the original pixels survive un-stretched at the centre
        self.assertEqual(out.getpixel((50, 50)), (10, 20, 30))
        # and the pad is the declared flat fill, not black
        self.assertEqual(out.getpixel((2, 50)), (127, 127, 127))

    def test_make_crop_does_not_mutate_the_source_image(self):
        from PIL import Image
        im = Image.new("RGB", (600, 400), (10, 20, 30))
        make_crop(im, (100, 100, 200, 200))
        self.assertEqual(im.size, (600, 400))
        self.assertEqual(im.getpixel((150, 150)), (10, 20, 30))


class RelevantCropTests(unittest.TestCase):

    def test_the_relevant_crop_is_the_largest_referenced_object(self):
        g = graph({"1": obj("cat", 10, 10, 40, 40), "2": obj("dog", 100, 100, 120, 120)})
        r = relevant_box(question(["1", "2"]), g)
        self.assertEqual(r["object_id"], "2")
        self.assertEqual(r["name"], "dog")

    def test_a_question_naming_no_object_yields_no_crop(self):
        g = graph({"1": obj("cat", 10, 10, 40, 40)})
        self.assertIsNone(relevant_box({"annotations": {}, "semantic": []}, g))

    def test_referenced_object_with_only_an_unusable_box_is_rejected(self):
        g = graph({"1": obj("speck", 10, 10, 4, 4)})
        self.assertIsNone(relevant_box(question(["1"]), g))


class IrrelevantCropTests(unittest.TestCase):

    def test_the_irrelevant_crop_excludes_referenced_objects(self):
        g = graph({"1": obj("cat", 10, 10, 100, 100), "2": obj("chair", 300, 200, 100, 100)})
        irr = irrelevant_box(question(["1"]), g, crop_area(expand_and_clip((10, 10, 110, 110), 600, 400)))
        self.assertEqual(irr["object_id"], "2")

    def test_a_candidate_overlapping_a_referenced_object_is_rejected(self):
        """Non-overlap is checked, not merely 'is a different object id'."""
        g = graph({"1": obj("cat", 100, 100, 100, 100), "2": obj("collar", 110, 110, 95, 95)})
        target = crop_area(expand_and_clip((100, 100, 200, 200), 600, 400))
        self.assertIsNone(irrelevant_box(question(["1"]), g, target))

    def test_area_matching_is_enforced(self):
        g = graph({"1": obj("cat", 10, 10, 100, 100), "2": obj("tower", 300, 100, 30, 260)})
        target = crop_area(expand_and_clip((10, 10, 110, 110), 600, 400))
        irr = irrelevant_box(question(["1"]), g, target)
        if irr is not None:
            self.assertLessEqual(abs(irr["area"] - target), AREA_TOL * target)

    def test_no_matched_control_is_rejected_not_substituted(self):
        g = graph({"1": obj("cat", 10, 10, 100, 100), "2": obj("speck", 400, 300, 5, 5)})
        target = crop_area(expand_and_clip((10, 10, 110, 110), 600, 400))
        self.assertIsNone(irrelevant_box(question(["1"]), g, target))


class DerangementTests(unittest.TestCase):
    """The wrong crop must share the referenced TARGET LABEL but never the image or the answer."""

    def _profiles(self, n, label="dog"):
        return {f"q{i}": {"area": 1000 + 10 * i, "image_id": f"img{i}", "label": label}
                for i in range(n)}

    def test_is_a_true_derangement_one_to_one_and_fixed_point_free(self):
        for n in (2, 3, 8, 9, 40, 41):
            p = self._profiles(n)
            d, unusable = build_wrong_crop_derangement(list(p), p)
            with self.subTest(n=n):
                self.assertEqual(unusable, {})
                self.assertEqual(sorted(d.values()), sorted(p), "not a permutation")
                self.assertFalse([q for q in p if d[q] == q], "has a fixed point")

    def test_every_donor_shares_the_referenced_target_label(self):
        p = {}
        for i in range(6):
            p[f"dog{i}"] = {"area": 1000 + i, "image_id": f"a{i}", "label": "dog"}
            p[f"car{i}"] = {"area": 2000 + i, "image_id": f"b{i}", "label": "car"}
        d, unusable = build_wrong_crop_derangement(list(p), p)
        self.assertEqual(unusable, {})
        for q, donor in d.items():
            with self.subTest(q=q):
                self.assertEqual(p[donor]["label"], p[q]["label"],
                                 "a wrong crop must depict the same KIND of object")

    def test_no_donor_shares_the_question_image(self):
        p = {f"q{i}": {"area": 1000 + i, "image_id": f"img{i // 2}", "label": "dog"}
             for i in range(20)}
        d, _ = build_wrong_crop_derangement(list(p), p)
        same = [q for q in p if p[d[q]]["image_id"] == p[q]["image_id"]]
        self.assertEqual(same, [], f"{len(same)} donors share the question's own image")

    def test_a_singleton_class_is_reported_not_repaired(self):
        p = {"a": {"area": 1, "image_id": "i1", "label": "zebra"},
             "b": {"area": 2, "image_id": "i2", "label": "dog"},
             "c": {"area": 3, "image_id": "i3", "label": "dog"}}
        d, unusable = build_wrong_crop_derangement(list(p), p)
        self.assertIn("zebra", unusable)
        self.assertNotIn("a", d, "a question with no valid donor must be dropped, not paired")
        self.assertEqual(sorted(d), ["b", "c"])

    def test_a_class_confined_to_one_image_is_reported(self):
        p = {"a": {"area": 1, "image_id": "i1", "label": "dog"},
             "b": {"area": 2, "image_id": "i1", "label": "dog"}}
        d, unusable = build_wrong_crop_derangement(list(p), p)
        self.assertIn("dog", unusable)
        self.assertEqual(d, {})

    def test_donors_are_area_proximate_within_a_class(self):
        """The smallest valid rotation over an area-sorted class keeps donors close in size."""
        p = {f"q{i}": {"area": 100 * i, "image_id": f"i{i}", "label": "dog"} for i in range(1, 11)}
        d, _ = build_wrong_crop_derangement(list(p), p)
        diffs = [abs(p[d[q]]["area"] - p[q]["area"]) for q in p]
        self.assertLessEqual(max(diffs), 900, "a shift-1 rotation should keep donors adjacent")

    def test_matching_never_uses_the_answer(self):
        """Profiles carry no answer field at all; adding one must not change the assignment."""
        p = self._profiles(10)
        base, _ = build_wrong_crop_derangement(list(p), p)
        poisoned = {q: dict(v, answer="yes" if i % 2 else "no")
                    for i, (q, v) in enumerate(p.items())}
        alt, _ = build_wrong_crop_derangement(list(poisoned), poisoned)
        self.assertEqual(base, alt)

    def test_is_deterministic(self):
        p = self._profiles(30)
        self.assertEqual(build_wrong_crop_derangement(list(p), p)[0],
                         build_wrong_crop_derangement(list(p), p)[0])


class AnswerLeakageTests(unittest.TestCase):
    """Crop selection must never depend on the answer, or the control leaks the label."""

    def test_poisoning_the_answer_cannot_change_the_crops(self):
        g = graph({"1": obj("cat", 10, 10, 100, 100), "2": obj("chair", 300, 200, 105, 105)})
        base = crop_record(question(["1"], answer="yes"), g)
        for poisoned in ("no", "chair", "cat", "", "left"):
            with self.subTest(answer=poisoned):
                alt = crop_record(question(["1"], answer=poisoned), g)
                self.assertEqual(base, alt)

    def test_removing_the_answer_field_entirely_cannot_change_the_crops(self):
        g = graph({"1": obj("cat", 10, 10, 100, 100), "2": obj("chair", 300, 200, 105, 105)})
        q = question(["1"])
        base = crop_record(dict(q), g)
        stripped = {k: v for k, v in q.items() if k not in ("answer", "fullAnswer")}
        self.assertEqual(base, crop_record(stripped, g))

    def test_the_module_never_reads_the_answer_field(self):
        src = (ROOT / "src" / "data" / "crop_augment.py").read_text()
        for field in ("answer", "fullAnswer"):
            self.assertFalse(reads_field(src, field),
                             f"crop_augment.py reads the {field!r} field in code")

    def test_the_leakage_scanner_would_catch_a_real_read(self):
        """Guard the guard: the AST scan must fire on code that actually reads the answer."""
        self.assertTrue(reads_field('def f(q):\n    return q["answer"]\n', "answer"))
        self.assertTrue(reads_field('def f(q):\n    return q.get("answer", None)\n', "answer"))
        self.assertFalse(reads_field('"""mentions q["answer"] in prose only"""\nx = 1\n', "answer"))


class CoverageArtifactTests(unittest.TestCase):
    """The coverage file is the pre-registered cohort definition; these guard its shape."""

    COV = ROOT / "outputs" / "crop_augment" / "coverage.json"

    def setUp(self):
        if not self.COV.is_file():
            self.skipTest("coverage.json absent (gitignored output; run scripts/56)")
        self.cov = json.loads(self.COV.read_text())

    def test_the_subset_is_nested_in_core(self):
        core, sub = set(self.cov["core_qids"]), set(self.cov["irrelevant_subset_qids"])
        self.assertTrue(sub.issubset(core), "irrelevant_subset must be nested in core")
        self.assertLessEqual(len(sub), len(core))

    def test_cohort_membership_matches_the_per_record_flag(self):
        sub = set(self.cov["irrelevant_subset_qids"])
        for q in self.cov["core_qids"]:
            r = self.cov["records"][q]
            with self.subTest(q=q):
                self.assertEqual(r["has_irrelevant"], q in sub)
                if q in sub:
                    self.assertIsNotNone(r["irrelevant"])
                else:
                    self.assertIsNone(r["irrelevant"])
                    self.assertIsNotNone(r["irrelevant_reject"])

    def test_every_core_record_carries_the_required_fields(self):
        need = ("qid", "image_id", "relevant", "target_label", "has_irrelevant",
                "wrong_donor_qid", "wrong_donor_image_id", "wrong_donor_label", "wrong_box",
                "wrong_area_diff", "oracle")
        for q in self.cov["core_qids"][:60]:
            for f in need:
                self.assertIn(f, self.cov["records"][q], f"{q} lacks {f}")

    def test_every_core_donor_shares_the_target_label(self):
        for q in self.cov["core_qids"]:
            r = self.cov["records"][q]
            with self.subTest(q=q):
                self.assertEqual(r["wrong_donor_label"], r["target_label"])

    def test_it_is_marked_oracle_and_non_deployable(self):
        self.assertTrue(self.cov["oracle"])
        self.assertIn("not_deployable", self.cov)

    def test_counts_are_internally_consistent(self):
        self.assertEqual(self.cov["n_core"], len(self.cov["core_qids"]))
        self.assertEqual(self.cov["n_irrelevant_subset"],
                         len(self.cov["irrelevant_subset_qids"]))
        self.assertEqual(self.cov["unique_images"],
                         len({self.cov["records"][q]["image_id"]
                              for q in self.cov["core_qids"]}))

    def test_unusable_target_classes_are_reported(self):
        self.assertIn("unusable_target_classes", self.cov)
        self.assertIn("n_dropped_no_valid_donor", self.cov)
        for label, why in self.cov["unusable_target_classes"].items():
            self.assertTrue(why.strip(), f"{label} has no stated reason")

    def test_it_pins_dataset_and_script_hashes(self):
        for f in ("slice_sha256", "questions_sha256", "scene_graphs_sha256",
                  "module_sha256", "script_sha256"):
            self.assertRegex(self.cov[f], r"^[0-9a-f]{64}$", f"{f} is not a sha256")

    def test_no_core_question_has_a_same_image_donor(self):
        bad = [q for q in self.cov["core_qids"]
               if self.cov["records"][q]["wrong_donor_image_id"]
               == self.cov["records"][q]["image_id"]]
        self.assertEqual(bad, [])

    def test_area_match_error_is_within_tolerance_across_the_subset(self):
        for q in self.cov["irrelevant_subset_qids"]:
            self.assertLessEqual(self.cov["records"][q]["area_match_error"], AREA_TOL)


class CohortAnalysisTests(unittest.TestCase):
    """The primary contrast runs on core; the irrelevant contrast only on the nested subset."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib
        self.mod = importlib.import_module("58_crop_analysis")

    def test_primary_contrast_is_relevant_versus_wrong_on_core(self):
        self.assertEqual(self.mod.PRIMARY,
                         ("global_plus_relevant_crop", "global_plus_wrong_crop"))
        self.assertEqual(self.mod.PRIMARY_COHORT, "core")
        rows = [c for c in self.mod.CONTRASTS if (c[0], c[1]) == self.mod.PRIMARY]
        # The pair is read on exactly two cohorts: the full core (PRIMARY) and the pre-registered
        # scale-matched subgroup (ROBUSTNESS). Any third reading would be an unlabelled subgroup.
        self.assertEqual(sorted(c[2] for c in rows),
                         sorted([self.mod.PRIMARY_COHORT, self.mod.ROBUSTNESS_COHORT]))
        self.assertEqual(self.mod.ROBUSTNESS_COHORT, "scale_matched")

    def test_the_two_readings_of_the_primary_pair_are_keyed_apart(self):
        """Same treatment/control on two cohorts: the result key must carry the cohort."""
        src = code_text((ROOT / "scripts" / "58_crop_analysis.py").read_text())
        self.assertIn("[{coh}]", src,
                      "the contrast key omits the cohort, so one reading would overwrite the other")

    def test_the_irrelevant_contrast_uses_only_the_subset(self):
        row = [c for c in self.mod.CONTRASTS
               if c[1] == "global_plus_irrelevant_crop"]
        self.assertEqual(len(row), 1)
        self.assertEqual(row[0][2], "irrelevant_subset",
                         "relevant-vs-irrelevant must not be computed over core")

    def test_no_contrast_mixes_cohorts(self):
        for t, c, coh, _ in self.mod.CONTRASTS:
            with self.subTest(contrast=f"{t}-{c}"):
                self.assertIn(coh, ("core", "irrelevant_subset", "scale_matched"))
                if "irrelevant" in (t, c) or "irrelevant" in t or "irrelevant" in c:
                    self.assertEqual(coh, "irrelevant_subset")

    def test_the_eval_generates_the_irrelevant_arm_only_on_the_subset(self):
        src = (ROOT / "scripts" / "57_crop_eval.py").read_text()
        self.assertIn("if q in subset:", src)
        self.assertIn('views["global_plus_irrelevant_crop"] = [g, irr]', src)

    def test_the_eval_checks_identical_encoder_coverage_within_each_cohort(self):
        src = (ROOT / "scripts" / "57_crop_eval.py").read_text()
        self.assertIn("cohort coverage is identical across encoders", src)
        self.assertIn("encoder coverage differs or leaves its cohort", src)


class TokenShapeTests(unittest.TestCase):
    """The three crop arms must carry identical token counts, or length confounds content."""

    def test_multi_view_concatenates_token_axes(self):
        import torch
        from src.models.multi_view import visual_tokens

        class FakeBridge(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(1, 256, 8)

        out = visual_tokens(FakeBridge(), [torch.zeros(1, 257, 4)])
        self.assertEqual(tuple(out.shape), (1, 256, 8))
        out2 = visual_tokens(FakeBridge(), [torch.zeros(1, 257, 4), torch.zeros(1, 257, 4)])
        self.assertEqual(tuple(out2.shape), (1, 512, 8), "global+crop must be 2x one view")

    def test_equal_crop_token_counts_across_relevant_irrelevant_and_wrong(self):
        """All three are single views through the same connector, so all three must match."""
        import torch
        from src.models.multi_view import visual_tokens

        class FakeBridge(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(1, 256, 8)

        b = FakeBridge()
        counts = {name: visual_tokens(b, [torch.zeros(1, 257, 4)]).size(1)
                  for name in ("relevant", "irrelevant", "wrong")}
        self.assertEqual(len(set(counts.values())), 1, counts)

    def test_a_view_with_the_wrong_rank_is_rejected(self):
        import torch
        from src.models.multi_view import visual_tokens

        class FakeBridge(torch.nn.Module):
            def forward(self, x):
                return torch.zeros(1, 256, 8)

        with self.assertRaises(ValueError):
            visual_tokens(FakeBridge(), [torch.zeros(257, 4)])


class FrozenParameterTests(unittest.TestCase):

    def test_assert_frozen_rejects_a_trainable_module(self):
        import torch
        from src.models.multi_view import assert_frozen
        m = torch.nn.Linear(3, 3).eval()
        with self.assertRaises(RuntimeError):
            assert_frozen(m)
        m.requires_grad_(False)
        assert_frozen(m)

    def test_assert_frozen_rejects_training_mode(self):
        import torch
        from src.models.multi_view import assert_frozen
        m = torch.nn.Linear(3, 3)
        m.requires_grad_(False)
        m.train()
        with self.assertRaises(RuntimeError):
            assert_frozen(m)


class BootstrapTests(unittest.TestCase):
    """The clustered bootstrap must resample IMAGES, not questions."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib
        self.mod = importlib.import_module("58_crop_analysis")

    def test_cluster_draws_resample_whole_images(self):
        qids = ["a", "b", "c", "d"]
        image_of = {"a": "i1", "b": "i1", "c": "i2", "d": "i3"}
        draws, n_img = self.mod.cluster_draws(qids, image_of)
        self.assertEqual(n_img, 3)
        for ix in draws[:200]:
            counts = {}
            for j in ix:
                counts[image_of[qids[j]]] = counts.get(image_of[qids[j]], 0) + 1
            # i1 has two questions, so it can only ever appear in multiples of two
            self.assertEqual(counts.get("i1", 0) % 2, 0,
                             "a cluster was split: questions were resampled, not images")

    def test_clustering_widens_the_interval_when_questions_are_correlated(self):
        """With perfectly correlated questions per image, clustering must not be narrower."""
        n_img, per = 40, 4
        qids = [f"q{i}_{j}" for i in range(n_img) for j in range(per)]
        image_of = {f"q{i}_{j}": f"i{i}" for i in range(n_img) for j in range(per)}
        rng = np.random.default_rng(0)
        per_img = rng.integers(0, 2, size=n_img)
        v = np.array([per_img[i] for i in range(n_img) for _ in range(per)], float)
        qidx, _ = self.mod.q_matrix(len(qids))
        cdraws, _ = self.mod.cluster_draws(qids, image_of)
        _, qlo, qhi = self.mod.ci_from_draws(v, [qidx[i] for i in range(self.mod.BOOT)])
        _, clo, chi = self.mod.ci_from_draws(v, cdraws)
        self.assertGreater(chi - clo, qhi - qlo,
                           "clustered interval must be wider under within-image correlation")

    def test_question_matrix_is_seeded_and_reproducible(self):
        a, sa = self.mod.q_matrix(50)
        b, sb = self.mod.q_matrix(50)
        self.assertEqual(sa, sb)
        self.assertTrue((a == b).all())


class TruncationPolicyTests(unittest.TestCase):
    """The C41 policy: bound the effect, and fail only when it could change the conclusion."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib
        self.mod = importlib.import_module("58_crop_analysis")

    def test_treatment_arms_are_the_oracle_arms(self):
        self.assertEqual(set(self.mod.TREATMENT_ARMS),
                         {"global_plus_relevant_crop", "relevant_crop_only"})

    def test_control_truncation_lowers_the_contrast_and_treatment_truncation_raises_it(self):
        """The bias direction, asserted rather than asserted in prose.

        A truncated answer scores wrong, so a truncating arm is understated. Understating the
        CONTROL inflates treatment-minus-control; understating the TREATMENT deflates it.
        """
        n = 100
        t = {f"q{i}": True for i in range(n)}
        c = {f"q{i}": i < 50 for i in range(n)}
        qids = list(t)
        base = self.mod.paired(t, c, qids).mean() * 100
        # repair 10 truncated CONTROL answers -> control rises -> contrast falls
        c_fixed = dict(c)
        for i in range(50, 60):
            c_fixed[f"q{i}"] = True
        self.assertLess(self.mod.paired(t, c_fixed, qids).mean() * 100, base)
        # repair 10 truncated TREATMENT answers -> treatment rises -> contrast rises
        t_low = dict(t)
        for i in range(10):
            t_low[f"q{i}"] = False
        self.assertLess(self.mod.paired(t_low, c, qids).mean() * 100, base)

    def test_the_contrast_list_names_the_grounded_comparison(self):
        pairs = {(a, b) for a, b, _, _ in self.mod.CONTRASTS}
        self.assertIn(("global_plus_relevant_crop", "global_plus_wrong_crop"), pairs)
        self.assertIn(("global_plus_relevant_crop", "global_plus_irrelevant_crop"), pairs)


class SmokeVerdictSuppressionTests(unittest.TestCase):
    """A smoke test must never emit a scientific verdict, however good its numbers look."""

    SRC = (ROOT / "scripts" / "58_crop_analysis.py").read_text()

    def test_smoke_suppression_precedes_any_verdict_branch(self):
        code = code_text(self.SRC)
        smoke_at = code.index("NO SCIENTIFIC VERDICT")
        grounded_at = code.index("GROUNDED: the relevant crop beats")
        self.assertLess(smoke_at, grounded_at,
                        "the smoke check must short-circuit before GROUNDED can be assigned")

    def test_the_stripper_removes_comments_so_prose_cannot_satisfy_the_check(self):
        """Guard the guard: a comment mentioning GROUNDED must not count as the branch."""
        planted = 'x = 1  # a printed "GROUNDED" gets quoted\ny = 2\n'
        self.assertNotIn("GROUNDED", code_text(planted))
        self.assertIn("GROUNDED", planted)

    def test_the_exact_required_string_is_used(self):
        self.assertIn("NO SCIENTIFIC VERDICT — GPU smoke for parity and plumbing only.", self.SRC)

    def test_a_smoke_fixture_cannot_produce_a_verdict_even_when_every_difference_is_positive(self):
        """Mutation: an all-positive smoke fixture must still refuse to conclude."""
        out = run_analysis(analysis_fixture(smoke=True))
        self.assertIn("NO SCIENTIFIC VERDICT", out.stdout)
        self.assertNotIn("GROUNDED:", out.stdout,
                         "a smoke fixture produced a scientific verdict despite a maximal effect")

    def test_the_same_fixture_without_the_smoke_flag_does_conclude(self):
        """Guard the guard: if a clean fixture could never say GROUNDED, the test above is empty."""
        out = run_analysis(analysis_fixture(smoke=False))
        self.assertIn("GROUNDED:", out.stdout)
        self.assertNotIn("NO SCIENTIFIC VERDICT", out.stdout)


class ScaleMatchedSubgroupTests(unittest.TestCase):
    """The pre-registered factor-two subgroup: symmetric, answer-blind, fixed before analysis."""

    def test_membership_never_reads_an_answer(self):
        """The rule takes two areas. Poisoning every answer field cannot move a single member."""
        objects = {"1": obj("dog", 100, 100, 120, 90), "2": obj("cat", 300, 200, 60, 60)}
        g = graph(objects)
        clean = crop_record(question(["1"], answer="yes"), g)
        poisoned = crop_record(question(["1"], answer="absolutely not"), g)
        self.assertEqual(clean["relevant"]["box"], poisoned["relevant"]["box"])
        for a, b in ((clean, poisoned), (poisoned, clean)):
            d1 = scale_pair_diagnostics(a["relevant"]["box"], [0, 0, 200, 200])
            d2 = scale_pair_diagnostics(b["relevant"]["box"], [0, 0, 200, 200])
            self.assertEqual(d1["scale_matched"], d2["scale_matched"])
            self.assertEqual(d1["padded_area_ratio"], d2["padded_area_ratio"])

    def test_the_subgroup_rule_reads_no_answer_field_in_code(self):
        """AST, not prose: the docstrings may discuss answers, the code may not read one."""
        src = (ROOT / "src" / "data" / "crop_augment.py").read_text()
        for field in ("answer", "fullAnswer", "gold", "exact_full"):
            self.assertFalse(reads_field(src, field),
                             f"the crop module reads {field!r} in executable code")
        ids = code_identifiers(src)
        for banned in ("exact_full", "fullAnswer"):
            self.assertNotIn(banned, ids)

    def test_the_factor_two_rule_is_symmetric(self):
        """is_scale_matched(a, b) must equal is_scale_matched(b, a) for every pair."""
        rng = np.random.default_rng(7)
        areas = list(rng.integers(1, 500_000, size=400))
        for a, b in zip(areas[::2], areas[1::2]):
            self.assertEqual(is_scale_matched(int(a), int(b)), is_scale_matched(int(b), int(a)),
                             f"asymmetric verdict for areas {a} and {b}")

    def test_the_boundary_is_inclusive_and_symmetric(self):
        self.assertTrue(is_scale_matched(100, 200))     # exactly a factor of two
        self.assertTrue(is_scale_matched(200, 100))
        self.assertFalse(is_scale_matched(100, 201))    # just outside, either way round
        self.assertFalse(is_scale_matched(201, 100))

    def test_a_degenerate_area_is_never_matched(self):
        self.assertFalse(is_scale_matched(0, 100))
        self.assertFalse(is_scale_matched(100, 0))

    def test_padded_geometry_follows_the_longer_edge(self):
        """pad_to_square uses max(w, h), so every padded quantity must follow from it."""
        g = padded_geometry((0, 0, 200, 50), raw_box=(0, 0, 100, 50), encoder_side=224)
        self.assertEqual((g["padded_w"], g["padded_h"]), (200, 200))
        self.assertEqual(g["padded_area"], 200 * 200)
        self.assertAlmostEqual(g["crop_occupancy"], (200 * 50) / (200 * 200))
        self.assertAlmostEqual(g["object_occupancy"], (100 * 50) / (200 * 200))
        self.assertAlmostEqual(g["resize_factor"], 224 / 200)
        self.assertAlmostEqual(g["aspect_ratio"], 4.0)

    def test_the_analysis_reads_the_subgroup_from_a_flag_not_from_accuracy(self):
        """The cohort must be built from `in_scale_matched`, never from a score field."""
        src = code_text((ROOT / "scripts" / "58_crop_analysis.py").read_text())
        line = [ln for ln in src.splitlines() if "'scale_matched':" in ln or
                '"scale_matched":' in ln]
        self.assertTrue(line, "no scale_matched cohort is constructed")
        joined = " ".join(line)
        self.assertIn("in_scale_matched", joined)
        for banned in ("exact_full", "__ans"):
            self.assertNotIn(banned, joined,
                             "the subgroup is being derived from an accuracy field")

    def test_the_analysis_refuses_records_whose_flags_disagree_with_the_declared_cohort(self):
        """Mutation: a subgroup edited after the fact must be rejected, not silently used."""
        blob = analysis_fixture(smoke=False, in_scale_matched=lambda i: i % 2 == 0)
        blob["records"][1]["in_scale_matched"] = True      # a member added post hoc
        out = run_analysis(blob)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("not fixed before the answers", out.stdout + out.stderr)

    def test_records_predating_the_subgroup_are_refused_rather_than_analysed(self):
        blob = analysis_fixture(smoke=False)
        del blob["_meta"]["cohorts"]["scale_matched"]
        out = run_analysis(blob)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("predate", out.stdout + out.stderr)

    def test_the_evaluator_reads_the_subgroup_from_coverage_and_never_recomputes_it(self):
        """scripts/57 must copy the pre-registered cohort, not derive one beside the answers."""
        src = code_text((ROOT / "scripts" / "57_crop_eval.py").read_text())
        self.assertIn("scale_matched_qids", src)
        self.assertNotIn("is_scale_matched", code_identifiers(
            (ROOT / "scripts" / "57_crop_eval.py").read_text()),
            "the evaluator recomputes subgroup membership instead of reading it")


class ScaleConfoundVerdictTests(unittest.TestCase):
    """A positive full-cohort result with a non-positive subgroup may not be called GROUNDED."""

    def test_positive_full_cohort_with_negative_subgroup_is_scale_confounded(self):
        # Treatment wins on every scale-MISMATCHED question and loses on every matched one, so
        # the full cohort is positive while the subgroup is negative — the artefact pattern.
        def hit(i, cond, sm):
            if cond == "global_plus_relevant_crop":
                return not sm
            if cond == "global_plus_wrong_crop":
                return sm
            return False
        out = run_analysis(analysis_fixture(n=120, hit=hit,
                                            in_scale_matched=lambda i: i % 3 == 0))
        self.assertIn("SCALE-CONFOUNDED / UNRESOLVED", out.stdout)
        self.assertNotIn("GROUNDED:", out.stdout,
                         "a scale-confounded result was reported as grounded")

    def test_a_zero_subgroup_effect_also_blocks_grounded(self):
        """The rule is `subgroup point must be POSITIVE`, so exactly zero must not pass."""
        def hit(i, cond, sm):
            if sm:                                   # subgroup: treatment and control identical
                return cond in ("global_plus_relevant_crop", "global_plus_wrong_crop")
            return cond == "global_plus_relevant_crop"
        out = run_analysis(analysis_fixture(n=120, hit=hit,
                                            in_scale_matched=lambda i: i % 3 == 0))
        self.assertIn("SCALE-CONFOUNDED / UNRESOLVED", out.stdout)
        self.assertNotIn("GROUNDED:", out.stdout)

    def test_agreeing_subgroup_still_yields_grounded(self):
        """Guard the guard: the block above must not be blocking everything."""
        out = run_analysis(analysis_fixture(n=120, in_scale_matched=lambda i: i % 3 == 0))
        self.assertIn("GROUNDED:", out.stdout)
        self.assertNotIn("SCALE-CONFOUNDED", out.stdout)

    def test_the_subgroup_interval_is_never_required_to_exclude_zero(self):
        """A subgroup with a positive point but a zero-spanning interval must still be GROUNDED."""
        # Only a small positive edge inside the subgroup: the point is positive, but at n=40 the
        # interval spans zero comfortably.
        def hit(i, cond, sm):
            if cond == "global_plus_relevant_crop":
                return True
            if cond == "global_plus_wrong_crop":
                return sm and i % 12 != 0
            return False
        blob = analysis_fixture(n=120, hit=hit, in_scale_matched=lambda i: i % 3 == 0)
        out = run_analysis(blob)
        self.assertIn("GROUNDED:", out.stdout)
        self.assertNotIn("SCALE-CONFOUNDED", out.stdout)

    def test_no_equivalence_language_anywhere_in_the_verdicts(self):
        src = (ROOT / "scripts" / "58_crop_analysis.py").read_text()
        for banned in ("no effect\"", "is equivalent", "proves equivalence", "no difference between"):
            self.assertNotIn(banned, src)
        self.assertIn("never a demonstration of equivalence", src)

    def test_the_scale_branch_precedes_the_grounded_branch(self):
        code = code_text((ROOT / "scripts" / "58_crop_analysis.py").read_text())
        self.assertLess(code.index("SCALE-CONFOUNDED / UNRESOLVED"),
                        code.index("GROUNDED: the relevant crop beats"),
                        "GROUNDED would be assigned before the scale check could block it")


class TruncationBoundTests(unittest.TestCase):
    """The C41 rule bounds the POINT as well as the interval, and fails on either reversal."""

    def setUp(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        import importlib
        self.mod = importlib.import_module("58_crop_analysis")

    @staticmethod
    def bounds(point, lo, hi, n, k_treatment=0, k_control=0):
        """Replicates the coded rule, so the test asserts the arithmetic, not the prose."""
        w_lo = lo - 100 * k_control / n
        w_hi = hi + 100 * k_treatment / n
        wp_lo = point - 100 * k_control / n
        wp_hi = point + 100 * k_treatment / n
        sign_flip = (point > 0 and wp_lo <= 0) or (point < 0 and wp_hi >= 0)
        stops_excluding = (lo > 0 or hi < 0) and not (w_lo > 0 or w_hi < 0)
        return wp_lo, wp_hi, sign_flip, stops_excluding

    def test_control_truncation_that_can_reverse_the_point_sign_fails(self):
        _, _, flip, _ = self.bounds(point=+3.0, lo=+1.0, hi=+5.0, n=100, k_control=4)
        self.assertTrue(flip, "a +3.0 point with 4/100 control truncations can reach -1.0")

    def test_control_truncation_that_cannot_reverse_anything_survives(self):
        wp_lo, _, flip, stops = self.bounds(point=+20.0, lo=+15.0, hi=+25.0, n=300, k_control=6)
        self.assertAlmostEqual(wp_lo, 18.0)
        self.assertFalse(flip)
        self.assertFalse(stops, "a wide margin must not be failed by a tiny truncation")

    def test_control_truncation_that_only_breaks_the_interval_still_fails(self):
        _, _, flip, stops = self.bounds(point=+6.0, lo=+1.0, hi=+11.0, n=100, k_control=2)
        self.assertFalse(flip, "the point itself survives")
        self.assertTrue(stops, "but the interval stops excluding zero, which must fail")

    def test_a_negative_point_that_can_reach_zero_fails(self):
        _, _, flip, _ = self.bounds(point=-3.0, lo=-7.0, hi=-1.0, n=100, k_treatment=4)
        self.assertTrue(flip, "a -3.0 point with 4/100 treatment truncations can reach +1.0")

    def test_the_coded_rule_uses_both_point_bounds(self):
        src = (ROOT / "scripts" / "58_crop_analysis.py").read_text()
        self.assertIn("worst_point_lo", src)
        self.assertIn("worst_point_hi", src)
        self.assertIn("worst_case_sign_flip", src)

    def test_a_failed_validity_check_suppresses_the_verdict(self):
        """A run that failed must not print GROUNDED, whatever the numbers say."""
        code = code_text((ROOT / "scripts" / "58_crop_analysis.py").read_text())
        fails_at = code.index("NO VERDICT — ")
        smoke_at = code.index("NO SCIENTIFIC VERDICT")
        grounded_at = code.index("GROUNDED: the relevant crop beats")
        self.assertLess(fails_at, smoke_at,
                        "validity failure must short-circuit before the smoke branch")
        self.assertLess(smoke_at, grounded_at,
                        "validity failure and smoke must both short-circuit before GROUNDED")


class PromptParityTests(unittest.TestCase):
    """Stage A must send the same user body as scripts/07, including P.SHORT_CUE."""

    SRC = (ROOT / "scripts" / "57_crop_eval.py").read_text()

    def test_the_eval_composes_through_user_body(self):
        self.assertIn("P.user_body(spec, ex.question)", self.SRC)

    def test_the_eval_never_passes_the_bare_question_to_a_generator(self):
        for bad in ("generate_multi_view(bridge, llm, vs, ex.question",
                    "vlm_generate(bridge, llm, g, ex.question"):
            self.assertNotIn(bad, self.SRC)

    def test_prompt_composition_matches_scripts_07(self):
        import src.prompt as P
        spec = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True)
        q = "What colour is the dog?"
        self.assertEqual(P.user_body(spec, q), q + P.SHORT_CUE)
        self.assertTrue(P.SHORT_CUE.strip())

    def test_the_eval_asserts_parity_at_startup(self):
        self.assertIn("prompt composition disagrees with scripts/07", self.SRC)


class BaselineParityContractTests(unittest.TestCase):
    """global_only must go through the NEW path, or the parity check is vacuous."""

    SRC = (ROOT / "scripts" / "57_crop_eval.py").read_text()
    MV = (ROOT / "src" / "models" / "multi_view.py").read_text()

    def test_global_only_is_generated_by_the_multi_view_path(self):
        self.assertIn('"global_only": [g]', self.SRC)
        self.assertIn("generate_multi_view(bridge, llm, vs", self.SRC)

    def test_multi_view_does_not_delegate_to_vlm_generate(self):
        """If it delegated for one view, parity would be true by construction and untestable."""
        self.assertNotIn("vlm_generate", code_identifiers(self.MV),
                         "multi_view delegates to the established path; the parity check in "
                         "scripts/57 would then be true by construction and could never fail")

    def test_the_delegation_scanner_would_catch_a_real_call(self):
        """Guard the guard: the AST scan must fire on code that actually calls it."""
        self.assertIn("vlm_generate",
                      code_identifiers("from src.models.vlm import vlm_generate\n"
                                       "def f(*a):\n    return vlm_generate(*a)\n"))
        self.assertNotIn("vlm_generate", code_identifiers('"""mentions vlm_generate"""\nx = 1\n'))

    def test_the_eval_compares_against_the_established_path(self):
        self.assertIn("vlm_generate(bridge, llm, g, prompt", self.SRC)
        self.assertIn("parity_fail", self.SRC)

    def test_parity_failure_stops_the_run(self):
        self.assertIn("raise SystemExit(f\"FAIL baseline parity broken", self.SRC)


class CpuInferenceRuleTests(unittest.TestCase):
    """CPU may not decide anything about answers; the eval must demand a GPU."""

    def test_the_eval_refuses_to_run_without_cuda(self):
        src = (ROOT / "scripts" / "57_crop_eval.py").read_text()
        self.assertIn("torch.cuda.is_available()", src)
        self.assertIn("CPU inference is not numerically", src)


@unittest.skipUnless(
    (ROOT / "scripts/cluster/59_crop_smoke.sbatch").is_file()
    and (ROOT / "scripts/cluster/60_crop_stage_a.sbatch").is_file(),
    "historical site launchers are intentionally absent from the clean release",
)
class OracleLabellingTests(unittest.TestCase):
    """The relevant crop is ground truth; every artifact must say so."""

    def test_every_stage_a_file_declares_the_oracle_status(self):
        for rel in ("src/data/crop_augment.py", "scripts/56_crop_coverage.py",
                    "scripts/57_crop_eval.py", "scripts/58_crop_analysis.py",
                    "scripts/cluster/59_crop_smoke.sbatch",
                    "scripts/cluster/60_crop_stage_a.sbatch"):
            txt = (ROOT / rel).read_text().upper()
            self.assertIn("ORACLE", txt, f"{rel} does not declare the oracle status")

    def test_the_full_launcher_trains_nothing_and_pins_the_reviewed_cohorts(self):
        src = (ROOT / "scripts" / "cluster" / "60_crop_stage_a.sbatch").read_text()
        self.assertIn("EXPECTED_CORE=300", src)
        self.assertIn("EXPECTED_SUBSET=99", src)
        self.assertIn("EXPECTED_SCALE=134", src)
        self.assertIn("coverage_v2.json", src)
        self.assertIn("--gpus=1", src)
        for bad_part in ("a100", "3090"):
            self.assertNotIn(f"partition={bad_part}", src)

    def test_the_full_launcher_invokes_no_training(self):
        """Checked on EXECUTABLE lines only. The prose says 'no training'; that is not the check."""
        src = (ROOT / "scripts" / "cluster" / "60_crop_stage_a.sbatch").read_text()
        code = [ln for ln in src.splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
        invoked = [ln for ln in code if "python " in ln or "torchrun" in ln]
        self.assertTrue(invoked, "the launcher runs nothing at all")
        for ln in invoked:
            for banned in ("train", "finetune", "lora", "optimizer", "backward", "torchrun"):
                self.assertNotIn(banned, ln.lower(),
                                 f"an executable line invokes {banned!r}: {ln.strip()!r}")
        joined = " ".join(code)
        self.assertIn("57_crop_eval.py", joined)
        self.assertIn("58_crop_analysis.py", joined)

    def test_the_no_training_check_catches_a_planted_training_call(self):
        """Guard the guard: the check above must fail if a training line were added."""
        src = (ROOT / "scripts" / "cluster" / "60_crop_stage_a.sbatch").read_text()
        planted = src + "\npython scripts/20_train_bridge.py --epochs 3\n"
        code = [ln for ln in planted.splitlines()
                if ln.strip() and not ln.lstrip().startswith("#")]
        invoked = [ln for ln in code if "python " in ln or "torchrun" in ln]
        self.assertTrue(any("train" in ln.lower() for ln in invoked),
                        "a planted training invocation went unnoticed")

    def test_crop_record_stamps_oracle_true(self):
        g = graph({"1": obj("cat", 10, 10, 100, 100), "2": obj("chair", 300, 200, 105, 105)})
        r = crop_record(question(["1"]), g)
        self.assertTrue(r.get("oracle"))


if __name__ == "__main__":
    unittest.main()
