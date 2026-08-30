"""Guards on Level 2, the answer-schema classifier.

WHAT THESE PROTECT. Level 2 decides what KIND of answer a question demands, so that "couch" to a
yes/no question is recorded as a format failure rather than silently scored as a wrong object. Two
traps are pinned here. First, `choose` alternatives must come from the question PROGRAM, not from
the question text: GQA has questions whose wording says "cement" while the program says "concrete",
and reading the surface text would invent an alternative the annotation does not contain. Second,
the gold answer is not always one of the program's alternatives (915 of 1,129 on Test-Dev), so the
admissible set must include the gold or 214 correct answers would be scored as schema violations.
"""
from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path

from src.eval.answer_schema import (
    ALTERNATIVE,
    ATTRIBUTE,
    BOOLEAN,
    COMPARISON,
    NUMBER,
    OBJECT_NAME,
    SCHEMA_VERSION,
    alternatives_of,
    expected_schema,
    matches_schema,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/gqa/questions/testdev_balanced_questions.json"


def _q(structural: str, detailed: str, ops: list[tuple[str, str]], answer: str = "yes") -> dict:
    return {
        "imageId": "img",
        "question": "synthetic",
        "answer": answer,
        "fullAnswer": "Synthetic.",
        "semantic": [{"operation": o, "argument": a, "dependencies": []} for o, a in ops],
        "semanticStr": "",
        "types": {"structural": structural, "semantic": "obj", "detailed": detailed},
        "annotations": {"answer": {}, "question": {}, "fullAnswer": {}},
    }


class ExpectedSchemaTests(unittest.TestCase):
    def test_verify_is_boolean(self):
        self.assertEqual(expected_schema(_q("verify", "verifyAttr", [("select", "x (1)"),
                                                                    ("verify color", "red")])), BOOLEAN)

    def test_logical_or_existence_is_boolean(self):
        self.assertEqual(expected_schema(_q("logical", "existOr", [("select", "a (1)"),
                                                                   ("exist", "?"),
                                                                   ("or", "")])), BOOLEAN)

    def test_choose_is_alternative(self):
        self.assertEqual(expected_schema(_q("choose", "chooseAttr",
                                            [("select", "ground (10)"),
                                             ("choose color", "brown|blue")], "brown")), ALTERNATIVE)

    def test_query_name_is_object_name(self):
        self.assertEqual(expected_schema(_q("query", "relS", [("select", "a (1)"),
                                                              ("relate", "b,on,s (2)"),
                                                              ("query", "name")], "chair")), OBJECT_NAME)

    def test_query_attribute_is_attribute(self):
        self.assertEqual(expected_schema(_q("query", "material", [("select", "a (1)"),
                                                                  ("query", "material")], "wood")), ATTRIBUTE)

    def test_compare_same_is_boolean(self):
        self.assertEqual(expected_schema(_q("compare", "twoSame", [("select", "a (1)"),
                                                                   ("select", "b (2)"),
                                                                   ("same color", "")], "yes")), BOOLEAN)

    def test_compare_common_is_comparison(self):
        self.assertEqual(expected_schema(_q("compare", "common", [("select", "a (1)"),
                                                                  ("select", "b (2)"),
                                                                  ("common", "")], "color")), COMPARISON)

    def test_compare_choose_is_an_object_name_not_an_alternative(self):
        # "Which is bigger, the racket or the wristband?" uses a `choose bigger` op whose argument
        # is EMPTY. The two candidates exist only in the question wording, which this module is
        # forbidden to parse, so the schema is object_name and not alternative.
        q = _q("compare", "comparative", [("select", "racket (1)"),
                                          ("select", "wristband (2)"),
                                          ("choose bigger", "")], "racket")
        self.assertEqual(expected_schema(q), OBJECT_NAME)
        self.assertEqual(alternatives_of(q), ())


class AlternativesTests(unittest.TestCase):
    def test_alternatives_come_from_the_program(self):
        q = _q("choose", "material", [("select", "fence (8)"),
                                      ("choose material", "aluminum|concrete")], "aluminum")
        q["question"] = "Is the fence made of cement or aluminum?"
        # "cement" is in the wording and NOT in the program. It must not become an alternative.
        self.assertEqual(alternatives_of(q), ("aluminum", "concrete"))
        self.assertNotIn("cement", alternatives_of(q))

    def test_gold_is_admissible_even_when_absent_from_the_program(self):
        q = _q("choose", "chooseAttr", [("select", "a (1)"),
                                        ("choose color", "white|orange")], "cream")
        self.assertTrue(matches_schema("cream", q, ALTERNATIVE))

    def test_a_non_alternative_string_fails_the_schema(self):
        q = _q("choose", "chooseAttr", [("select", "a (1)"),
                                        ("choose color", "white|orange")], "white")
        self.assertFalse(matches_schema("banana", q, ALTERNATIVE))


class MatchesSchemaTests(unittest.TestCase):
    def test_boolean_accepts_yes_and_no_only(self):
        q = _q("verify", "verifyAttr", [("verify color", "red")])
        self.assertTrue(matches_schema("yes", q, BOOLEAN))
        self.assertTrue(matches_schema("No.", q, BOOLEAN))
        self.assertFalse(matches_schema("couch", q, BOOLEAN))
        self.assertFalse(matches_schema("both", q, BOOLEAN))

    def test_boolean_rejects_a_prediction_carrying_both_polarities(self):
        q = _q("verify", "verifyAttr", [("verify color", "red")])
        self.assertFalse(matches_schema("yes no", q, BOOLEAN))

    def test_object_name_rejects_a_boolean(self):
        q = _q("query", "relS", [("query", "name")], "chair")
        self.assertFalse(matches_schema("yes", q, OBJECT_NAME))
        self.assertTrue(matches_schema("chair", q, OBJECT_NAME))

    def test_number_type_exists_but_is_never_expected_on_this_surface(self):
        # Test-Dev balanced contains no `count` operation and no numeric gold. The type stays in
        # the enum so a future surface can use it; this pins that it is not silently repurposed.
        self.assertEqual(NUMBER, "number")

    def test_version_is_declared(self):
        self.assertEqual(SCHEMA_VERSION, "gqa_answer_schema/1.0.0")


class RealDataSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QUESTIONS.is_file():
            raise unittest.SkipTest("requires downloaded GQA Test-Dev questions; see DATA.md")
        cls.questions = json.loads(QUESTIONS.read_text())

    def test_every_question_receives_a_schema(self):
        got = Counter(expected_schema(v) for v in self.questions.values())
        self.assertEqual(sum(got.values()), 12578)
        self.assertNotIn(None, got)

    def test_all_logical_and_verify_golds_satisfy_their_own_schema(self):
        # 100% of logical/verify golds are boolean on this surface; if the classifier disagrees
        # with the annotation, the classifier is wrong.
        bad = [k for k, v in self.questions.items()
               if v["types"]["structural"] in ("logical", "verify")
               and not matches_schema(v["answer"], v, BOOLEAN)]
        self.assertEqual(bad, [])

    def test_no_question_expects_a_number(self):
        nums = [k for k, v in self.questions.items() if expected_schema(v) == NUMBER]
        self.assertEqual(nums, [])

    def test_every_choose_question_has_parseable_alternatives(self):
        bad = [k for k, v in self.questions.items()
               if v["types"]["structural"] == "choose" and len(alternatives_of(v)) != 2]
        self.assertEqual(bad, [])

    def test_gold_satisfies_its_own_schema_for_every_question(self):
        # The strongest available sanity check: the annotation must pass its own type test.
        bad = [k for k, v in self.questions.items()
               if not matches_schema(v["answer"], v, expected_schema(v))]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    unittest.main()
