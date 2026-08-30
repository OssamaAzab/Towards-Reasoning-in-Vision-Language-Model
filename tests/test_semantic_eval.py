"""Guards on Level 3, the deterministic semantic cascade.

WHAT THESE PROTECT. Semantic rescue is the act of declaring an exact-wrong answer actually right.
Every failure mode here inflates a score, so each rule is tested with the case that would abuse it:

* the ADVERSARIAL case the human named — a plausible queried noun that GQA states is ABSENT must
  never be rescued, even though it appears verbatim in the question;
* "both" on a disjunction must never be rescued, because it asserts a conjunction the annotation
  contradicts;
* a synonym absent from the frozen ontology must not be rescued, or the ontology stops being the
  only synonym authority and rescues become unauditable;
* a token that appears in the question but is not annotated in `annotations.fullAnswer` must not be
  rescued, which is the mechanical form of "do not infer image truth from the question";
* `llm` must be unreachable as a judgment source — refused by the human on 2026-08-13. A value that
  merely happens to be unused is not the same as one that cannot be produced, and only the second
  is checkable, so the enum is asserted AND every code path is exercised.
"""
from __future__ import annotations

import unittest

from src.eval import semantic_eval as se
from src.eval.answer_schema import BOOLEAN, OBJECT_NAME
from src.eval.gqa_scene_inventory import build_inventory

ONTOLOGY = {"sofa": frozenset({"sofa", "couch"}), "couch": frozenset({"sofa", "couch"}),
            "tv": frozenset({"tv", "television"}), "television": frozenset({"tv", "television"})}


def _or_question(image_id="img", present="woman", absent="glasses", answer="yes",
                 question="Are there glasses or women?", full=None):
    """A real-shaped `select>exist>select>exist>or` question."""
    if full is None:
        full = f"Yes, there is a {present}." if answer == "yes" else \
               f"No, there are no {absent} or {present}."
    return {
        "imageId": image_id,
        "question": question,
        "answer": answer,
        "fullAnswer": full,
        "semantic": [
            {"operation": "select", "argument": f"{present} (7)" if answer == "yes"
             else f"{present} (-)", "dependencies": []},
            {"operation": "exist", "argument": "?", "dependencies": [0]},
            {"operation": "select", "argument": f"{absent} (-)", "dependencies": []},
            {"operation": "exist", "argument": "?", "dependencies": [2]},
            {"operation": "or", "argument": "", "dependencies": [1, 3]},
        ],
        "semanticStr": "",
        "types": {"structural": "logical", "semantic": "obj", "detailed": "existOr"},
        "annotations": {"answer": {}, "question": {}, "fullAnswer": {}},
    }


def _relate_question(image_id="img", gold="chair", present=("chair", "table")):
    return {
        "imageId": image_id,
        "question": "What is the man sitting on?",
        "answer": gold,
        "fullAnswer": f"The man is sitting on the {gold}.",
        "semantic": [
            {"operation": "select", "argument": "man (3)", "dependencies": []},
            {"operation": "relate", "argument": f"{gold},sitting on,o (5)", "dependencies": [0]},
            {"operation": "query", "argument": "name", "dependencies": [1]},
        ],
        "semanticStr": "",
        "types": {"structural": "query", "semantic": "rel", "detailed": "relO"},
        "annotations": {"answer": {}, "question": {}, "fullAnswer": {}},
    }


def _inv(image_id="img", present=(), absent=()):
    return {image_id: {"present": frozenset(present), "absent": frozenset(absent)}}


class EnumAndVersionTests(unittest.TestCase):
    def test_llm_is_not_a_permitted_judgment_source(self):
        self.assertNotIn("llm", se.JUDGMENT_SOURCES)
        self.assertEqual(se.JUDGMENT_SOURCES,
                         ("exact", "program", "full_answer", "ontology", "human"))

    def test_all_twelve_required_error_types_exist(self):
        self.assertEqual(set(se.ERROR_TYPES), {
            "correct_content_wrong_format", "synonym_or_paraphrase", "wrong_answer_type",
            "wrong_polarity", "wrong_object", "wrong_attribute", "wrong_relation",
            "reversed_relation", "wrong_referent_binding", "counting_error",
            "ambiguous_or_annotation_limited", "unresolved"})

    def test_versions_are_declared(self):
        # 1.1.0 withdrew the fullAnswer containment rule; see WithdrawnFullAnswerRuleTests.
        # 1.2.0 repaired the first review findings; 1.3.0 gated reversal on object-name queries (review defect 3).
        self.assertEqual(se.RUBRIC_VERSION, "gqa_semantic_rubric/1.3.0")

    def test_confidence_is_documented_as_ordinal_not_probability(self):
        self.assertIn("not", se.CONFIDENCE_IS_ORDINAL.lower())


class Level1Tests(unittest.TestCase):
    def test_exact_match_short_circuits_the_cascade(self):
        lab = se.evaluate(_or_question(), "yes", _inv(), ONTOLOGY)
        self.assertTrue(lab.exact_correct)
        self.assertTrue(lab.semantic_correct)
        self.assertEqual(lab.judgment_source, "exact")
        self.assertEqual(lab.error_type, "")
        self.assertFalse(lab.abstain)

    def test_exact_match_is_computed_not_supplied(self):
        # Level 1 must recompute from metrics.exact_full; punctuation and case must not matter.
        lab = se.evaluate(_or_question(), "Yes.", _inv(), ONTOLOGY)
        self.assertTrue(lab.exact_correct)


class OrExistenceAdjudicationTests(unittest.TestCase):
    """The §1 mechanism: GQA states which disjunct is present."""

    def test_naming_the_present_disjunct_is_a_format_failure_not_a_wrong_answer(self):
        lab = se.evaluate(_or_question(present="woman", absent="glasses", answer="yes"),
                          "woman", _inv(), ONTOLOGY)
        self.assertTrue(lab.semantic_correct)
        self.assertEqual(lab.error_type, "correct_content_wrong_format")
        self.assertEqual(lab.judgment_source, "program")
        self.assertFalse(lab.schema_correct)
        self.assertFalse(lab.abstain)

    def test_ADVERSARIAL_naming_the_absent_disjunct_is_never_rescued(self):
        # T4. "glasses" appears verbatim in the question and is a plausible object, and GQA says
        # it is not in the image. Rescuing it would manufacture accuracy from the question text.
        lab = se.evaluate(_or_question(present="woman", absent="glasses", answer="yes"),
                          "glasses", _inv(), ONTOLOGY)
        self.assertFalse(lab.semantic_correct)
        self.assertEqual(lab.error_type, "wrong_object")
        self.assertEqual(lab.judgment_source, "program")

    def test_both_is_never_rescued(self):
        # T6. Only one disjunct is present, so "both" asserts something GQA contradicts.
        lab = se.evaluate(_or_question(answer="yes"), "both", _inv(), ONTOLOGY)
        self.assertFalse(lab.semantic_correct)
        self.assertEqual(lab.error_type, "wrong_answer_type")

    def test_neither_with_gold_no_is_rescued(self):
        # T7. Gold `no` means neither disjunct is present, so "neither" is semantically right.
        lab = se.evaluate(_or_question(answer="no"), "neither", _inv(), ONTOLOGY)
        self.assertTrue(lab.semantic_correct)
        self.assertEqual(lab.error_type, "correct_content_wrong_format")

    def test_naming_any_disjunct_when_gold_is_no_is_wrong(self):
        lab = se.evaluate(_or_question(answer="no", present="seal", absent="bunny"),
                          "seal", _inv(), ONTOLOGY)
        self.assertFalse(lab.semantic_correct)
        self.assertEqual(lab.error_type, "wrong_object")

    def test_a_noun_from_neither_disjunct_is_not_adjudicated_by_the_or_rule(self):
        lab = se.evaluate(_or_question(), "helicopter", _inv(), ONTOLOGY)
        self.assertNotEqual(lab.error_type, "correct_content_wrong_format")

    def test_the_questions_own_plural_matches_the_programs_singular(self):
        # GQA asks "Are there any benches or books?" while the program stores "bench". Treating
        # "benches" as a different object misfiled 184 cells as wrong_answer_type.
        lab = se.evaluate(_or_question(present="bench", absent="book", answer="yes"),
                          "benches", _inv(), ONTOLOGY)
        self.assertTrue(lab.semantic_correct)
        self.assertEqual(lab.error_type, "correct_content_wrong_format")

    def test_the_plural_of_the_ABSENT_disjunct_is_still_wrong(self):
        lab = se.evaluate(_or_question(present="bench", absent="book", answer="yes"),
                          "books", _inv(), ONTOLOGY)
        self.assertFalse(lab.semantic_correct)
        self.assertEqual(lab.error_type, "wrong_object")

    def test_glass_and_glasses_are_never_folded(self):
        # In GQA "glasses" is eyewear and "glass" is a material or a vessel. Folding them would
        # credit a prediction naming a different object.
        self.assertFalse(se._matches_disjunct("glasses", "glass"))
        self.assertFalse(se._matches_disjunct("glass", "glasses"))

    def test_the_fold_is_morphological_not_semantic(self):
        self.assertTrue(se._matches_disjunct("buses", "bus"))
        self.assertTrue(se._matches_disjunct("bunnies", "bunny"))
        self.assertFalse(se._matches_disjunct("bench", "book"))
        self.assertFalse(se._matches_disjunct("cat", "cats hat"))


class PolarityTests(unittest.TestCase):
    def test_wrong_boolean_is_wrong_polarity(self):
        lab = se.evaluate(_or_question(answer="yes"), "no", _inv(), ONTOLOGY)
        self.assertFalse(lab.semantic_correct)
        self.assertEqual(lab.error_type, "wrong_polarity")
        self.assertTrue(lab.schema_correct)

    def test_a_prediction_carrying_both_polarities_is_not_resolved(self):
        lab = se.evaluate(_or_question(answer="yes"), "yes no", _inv(), ONTOLOGY)
        self.assertTrue(lab.abstain)
        self.assertIsNone(lab.semantic_correct)


class WithdrawnFullAnswerRuleTests(unittest.TestCase):
    """Rubric 1.1.0 withdrew fullAnswer containment. These pin the cases that exposed it.

    Implemented exactly as originally specified, the rule rescued 650 cells, and inspection showed
    the credit was routinely going to the question's own subject or to a partial head noun. GQA's
    annotations cannot repair it: the subject and the answer carry the same object id.
    """

    def test_the_questions_subject_is_never_rescued_from_the_answer_sentence(self):
        # gold `plastic`, prediction `coffee`, fullAnswer "The coffee is made of plastic."
        q = _relate_question(gold="plastic")
        q["question"] = "What is the coffee made of?"
        q["fullAnswer"] = "The coffee is made of plastic."
        lab = se.evaluate(q, "coffee", _inv(), ONTOLOGY)
        self.assertFalse(bool(lab.semantic_correct))
        self.assertNotEqual(lab.judgment_source, "full_answer")

    def test_a_partial_head_noun_is_not_a_synonym(self):
        # gold `street sign`, prediction `street`. A shorter phrase is a different answer.
        q = _relate_question(gold="street sign")
        q["fullAnswer"] = "The street sign is hanging above the man."
        lab = se.evaluate(q, "street", _inv(), ONTOLOGY)
        self.assertFalse(bool(lab.semantic_correct))

    def test_a_hypernym_is_not_rescued(self):
        q = _relate_question(gold="coffee table")
        q["fullAnswer"] = "The magazine is lying on top of the coffee table."
        lab = se.evaluate(q, "table", _inv(), ONTOLOGY)
        self.assertFalse(bool(lab.semantic_correct))

    def test_full_answer_is_never_emitted_as_a_judgment_source(self):
        # It stays in the enum because the required schema names it; it must not be produced.
        q = _relate_question(gold="chair")
        q["fullAnswer"] = "The man is sitting on the armchair."
        for pred in ("armchair", "man", "chair", "table", "unicorn"):
            self.assertNotEqual(se.evaluate(q, pred, _inv(), ONTOLOGY).judgment_source,
                                "full_answer")

    def test_gold_anchored_morphology_still_rescues(self):
        # The one thing the withdrawn rule got right, kept and anchored on the gold instead.
        q = _relate_question(gold="laptops")
        lab = se.evaluate(q, "laptop", _inv(), ONTOLOGY)
        self.assertTrue(lab.semantic_correct)
        self.assertEqual(lab.error_type, "correct_content_wrong_format")
        self.assertEqual(lab.judgment_source, "program")


class OntologyTests(unittest.TestCase):
    def test_a_pair_in_the_frozen_ontology_is_rescued(self):
        q = _relate_question(gold="sofa")
        q["fullAnswer"] = "The man is sitting on the sofa."
        lab = se.evaluate(q, "couch", _inv(), ONTOLOGY)
        self.assertTrue(lab.semantic_correct)
        self.assertEqual(lab.error_type, "synonym_or_paraphrase")
        self.assertEqual(lab.judgment_source, "ontology")

    def test_T8_a_pair_absent_from_the_ontology_is_not_rescued(self):
        q = _relate_question(gold="sofa")
        lab = se.evaluate(q, "settee", _inv(), ONTOLOGY)
        self.assertFalse(bool(lab.semantic_correct))
        self.assertNotEqual(lab.judgment_source, "ontology")

    def test_the_ontology_is_the_only_synonym_authority(self):
        # Same input, ontology emptied: the rescue must disappear. This is the mutation that
        # proves the rule is load-bearing rather than incidental.
        q = _relate_question(gold="sofa")
        q["fullAnswer"] = "The man is sitting on the sofa."
        lab = se.evaluate(q, "couch", _inv(), {})
        self.assertNotEqual(lab.judgment_source, "ontology")


class BindingTests(unittest.TestCase):
    def test_naming_a_present_object_with_a_DISJOINT_id_is_a_binding_failure(self):
        q = _relate_question(gold="chair")
        inv = {"img": {"present": frozenset({"chair", "table"}), "absent": frozenset(),
                       "ids": {"chair": frozenset({"3"}), "table": frozenset({"9"})}}}
        lab = se.evaluate(q, "table", inv, ONTOLOGY)
        self.assertFalse(lab.semantic_correct)
        self.assertEqual(lab.error_type, "wrong_referent_binding")
        self.assertTrue(lab.resolvable)

    def test_names_alone_can_no_longer_establish_a_binding_failure(self):
        # The invalidated rule: an inventory carrying names but no object ids used to be enough.
        # Review showed 58 such labels were annotation aliases of the SAME object.
        q = _relate_question(gold="chair")
        lab = se.evaluate(q, "table", _inv(present={"chair", "table"}), ONTOLOGY)
        self.assertNotEqual(lab.error_type, "wrong_referent_binding")
        self.assertTrue(lab.abstain)

    def test_an_object_absent_from_the_inventory_stays_unresolved(self):
        # OPEN WORLD. The inventory cannot show that "unicorn" is not in the image, so no
        # hallucination claim may be made and the case must abstain.
        q = _relate_question(gold="chair")
        lab = se.evaluate(q, "unicorn", _inv(present={"chair"}), ONTOLOGY)
        self.assertTrue(lab.abstain)
        self.assertEqual(lab.error_type, "unresolved")
        self.assertFalse(lab.resolvable)
        self.assertIsNone(lab.semantic_correct)


class AbstentionAndBlindnessTests(unittest.TestCase):
    def test_T12_abstention_carries_a_null_semantic_verdict(self):
        q = _relate_question(gold="chair")
        lab = se.evaluate(q, "unicorn", _inv(present={"chair"}), ONTOLOGY)
        self.assertTrue(lab.abstain)
        self.assertIsNone(lab.semantic_correct)

    def test_T11_evaluate_has_no_encoder_parameter(self):
        # Encoder blindness by construction: the function cannot condition on something it is
        # never given. A signature check is the strongest available form of this guarantee.
        import inspect
        params = set(inspect.signature(se.evaluate).parameters)
        self.assertEqual(params, {"question", "prediction", "inventory", "ontology"})

    def test_T13_number_normalisation_is_conservative(self):
        self.assertTrue(se.same_number("two", "2"))
        self.assertTrue(se.same_number("2", "2"))
        self.assertFalse(se.same_number("22", "2"))
        self.assertFalse(se.same_number("two", "three"))
        self.assertFalse(se.same_number("chair", "2"))


class DeterminismTests(unittest.TestCase):
    def test_repeated_evaluation_is_identical(self):
        q = _or_question()
        a = se.evaluate(q, "woman", _inv(), ONTOLOGY)
        b = se.evaluate(q, "woman", _inv(), ONTOLOGY)
        self.assertEqual(a, b)


class OntologyFileTests(unittest.TestCase):
    def test_the_shipped_ontology_loads_and_is_symmetric(self):
        onto, version = se.load_ontology()
        self.assertTrue(version)
        for term, group in onto.items():
            self.assertIn(term, group)
            for other in group:
                self.assertIn(term, onto[other], f"{term}->{other} is not symmetric")

    def test_the_shipped_ontology_declares_its_review_status(self):
        # It was seeded by an agent, not curated by a human. Any number that depends on it must be
        # discountable, so the status has to be machine-readable rather than a comment.
        self.assertIn(se.ontology_status(), ("UNREVIEWED_SEED", "HUMAN_REVIEWED"))


if __name__ == "__main__":
    unittest.main()


class BindingRequiresObjectIdsTests(unittest.TestCase):
    """Repair R1 (review finding, critical 1)."""

    def _q_with_ids(self, gold="snowboarder"):
        q = _relate_question(gold=gold)
        q["imageId"] = "img"
        return q

    def _inv_ids(self, mapping):
        return {"img": {"present": frozenset(mapping), "absent": frozenset(),
                        "ids": {k: frozenset(v) for k, v in mapping.items()}}}

    def test_an_alias_of_the_same_object_is_not_a_binding_error(self):
        # gold `snowboarder` and prediction `person` are the SAME object 0.
        lab = se.evaluate(self._q_with_ids("snowboarder"), "person",
                          self._inv_ids({"snowboarder": {"0"}, "person": {"0"}}), ONTOLOGY)
        self.assertNotEqual(lab.error_type, "wrong_referent_binding")
        self.assertEqual(lab.error_type, "ambiguous_or_annotation_limited")
        self.assertTrue(lab.abstain)
        self.assertIsNone(lab.semantic_correct)

    def test_disjoint_ids_are_a_binding_error(self):
        lab = se.evaluate(self._q_with_ids("chair"), "table",
                          self._inv_ids({"chair": {"3"}, "table": {"9"}}), ONTOLOGY)
        self.assertEqual(lab.error_type, "wrong_referent_binding")
        self.assertFalse(lab.semantic_correct)
        self.assertTrue(lab.resolvable)

    def test_unknown_gold_id_cannot_prove_a_different_referent(self):
        lab = se.evaluate(self._q_with_ids("chair"), "table",
                          self._inv_ids({"table": {"9"}}), ONTOLOGY)
        self.assertTrue(lab.abstain)
        self.assertEqual(lab.error_type, "unresolved")

    def test_overlapping_id_sets_are_not_disjoint(self):
        lab = se.evaluate(self._q_with_ids("houses"), "building",
                          self._inv_ids({"houses": {"1", "2"}, "building": {"2"}}), ONTOLOGY)
        self.assertEqual(lab.error_type, "ambiguous_or_annotation_limited")


class FilteredOrTests(unittest.TestCase):
    """Repair R2 (review finding, critical 2): base presence is not qualified presence."""

    def _filtered(self, a, a_present, b, b_present, answer="yes"):
        def sel(name, present, idx):
            return [{"operation": "select",
                     "argument": f"{name} ({idx})" if present else f"{name} (-)", "dependencies": []},
                    {"operation": "filter color", "argument": "black", "dependencies": [idx]},
                    {"operation": "exist", "argument": "?", "dependencies": [idx]}]
        return {
            "imageId": "img", "question": f"Is there either a black {a} or {b}?",
            "answer": answer, "fullAnswer": f"Yes, there is a black {a}.",
            "semantic": sel(a, a_present, 0) + sel(b, b_present, 3)
                        + [{"operation": "or", "argument": "", "dependencies": [2, 5]}],
            "semanticStr": "",
            "types": {"structural": "logical", "semantic": "obj", "detailed": "existAndOr"},
            "annotations": {"answer": {}, "question": {}, "fullAnswer": {}},
        }

    def test_one_base_present_licenses_the_rescue(self):
        # Only the calculator exists at all, so gold `yes` forces it to be the black one.
        q = self._filtered("calculator", True, "chair", False)
        lab = se.evaluate(q, "calculator", _inv(), ONTOLOGY)
        self.assertTrue(lab.semantic_correct)
        self.assertEqual(lab.error_type, "correct_content_wrong_format")

    def test_both_bases_present_cannot_license_a_rescue(self):
        # qid 201067766: calculator and chair both exist; only the calculator is black.
        q = self._filtered("calculator", True, "chair", True)
        lab = se.evaluate(q, "chair", _inv(), ONTOLOGY)
        self.assertTrue(lab.abstain)
        self.assertIsNone(lab.semantic_correct)
        self.assertEqual(lab.error_type, "unresolved")

    def test_both_prediction_is_not_rescued_when_both_bases_are_present(self):
        # qid 201861476 / 202262185: "both" cannot be established for the FILTERED objects.
        q = self._filtered("helmet", True, "mirror", True)
        lab = se.evaluate(q, "both", _inv(), ONTOLOGY)
        self.assertTrue(lab.abstain)

    def test_gold_no_still_makes_any_named_object_wrong(self):
        # If no filtered object exists, naming either base is still a false existence claim.
        q = self._filtered("hat", True, "fence", True, answer="no")
        lab = se.evaluate(q, "hat", _inv(), ONTOLOGY)
        self.assertFalse(lab.semantic_correct)
        self.assertEqual(lab.error_type, "wrong_object")

    def test_unfiltered_or_is_unaffected(self):
        lab = se.evaluate(_or_question(present="woman", absent="glasses", answer="yes"),
                          "woman", _inv(), ONTOLOGY)
        self.assertTrue(lab.semantic_correct)


class ChooseBeforeBindingTests(unittest.TestCase):
    """Repair R4 (review finding, high 4): a wrong supplied alternative is not relation binding."""

    def test_a_wrong_alternative_is_wrong_attribute_even_if_it_is_a_present_object(self):
        q = {"imageId": "img", "question": "Which is sturdy, the desk or the couch?",
             "answer": "desk", "fullAnswer": "The desk is sturdy.",
             "semantic": [{"operation": "select", "argument": "desk (1)", "dependencies": []},
                          {"operation": "choose", "argument": "desk|couch", "dependencies": [0]}],
             "semanticStr": "",
             "types": {"structural": "choose", "semantic": "obj", "detailed": "chooseObj"},
             "annotations": {"answer": {}, "question": {}, "fullAnswer": {}}}
        inv = {"img": {"present": frozenset({"desk", "couch"}), "absent": frozenset(),
                       "ids": {"desk": frozenset({"1"}), "couch": frozenset({"2"})}}}
        lab = se.evaluate(q, "couch", inv, ONTOLOGY)
        self.assertEqual(lab.error_type, "wrong_attribute")
        self.assertNotEqual(lab.error_type, "wrong_referent_binding")


class ReversedRelationTests(unittest.TestCase):
    """Repair R5.4: naming the relation's SOURCE object is a distinct, nameable error."""

    def test_naming_the_relate_source_is_reversed_relation(self):
        q = _relate_question(gold="stroller")
        q["imageId"] = "img"
        q["question"] = "What is the stuffed dog hanging from?"
        q["semantic"][0]["argument"] = "stuffed dog (3)"
        inv = {"img": {"present": frozenset({"stuffed dog", "stroller"}), "absent": frozenset(),
                       "ids": {"stuffed dog": frozenset({"3"}), "stroller": frozenset({"9"})}}}
        lab = se.evaluate(q, "stuffed dog", inv, ONTOLOGY)
        self.assertEqual(lab.error_type, "reversed_relation")
        self.assertFalse(lab.semantic_correct)

    def test_error_types_that_cannot_fire_on_this_surface_are_declared(self):
        self.assertEqual(se.NOT_APPLICABLE_ON_TESTDEV, ("counting_error", "wrong_relation"))

    def test_an_attribute_query_naming_the_relate_source_abstains(self):
        """Repair R9 (review finding, defect 3) — the real counterexample, qid 20783128.

        "Which material is the laptop near the glass made of?" selects `glass (5)`, relates to the
        laptop, then queries MATERIAL. I-JEPA answered "glass". That is a plausible material, not a
        confusion of the two relation referents, and the program cannot tell the two apart here.
        Reversal is only nameable when the question asks WHICH OBJECT.
        """
        q = {"imageId": "img",
             "question": "Which material is the laptop near the glass made of?",
             "answer": "plastic", "fullAnswer": "The laptop is made of plastic.",
             "semantic": [{"operation": "select", "argument": "glass (5)", "dependencies": []},
                          {"operation": "relate", "argument": "laptop,near,s (1)",
                           "dependencies": [0]},
                          {"operation": "query", "argument": "material", "dependencies": [1]}],
             "semanticStr": "",
             "types": {"structural": "query", "semantic": "attr", "detailed": "material"},
             "annotations": {"answer": {}, "question": {}, "fullAnswer": {}}}
        inv = {"img": {"present": frozenset({"glass", "laptop"}), "absent": frozenset(),
                       "ids": {"glass": frozenset({"5"}), "laptop": frozenset({"1"})}}}
        lab = se.evaluate(q, "glass", inv, ONTOLOGY)
        self.assertEqual(lab.schema_expected, "attribute")
        self.assertNotEqual(lab.error_type, "reversed_relation")
        self.assertEqual(lab.error_type, "unresolved")
        self.assertIsNone(lab.semantic_correct)
        self.assertTrue(lab.abstain)
