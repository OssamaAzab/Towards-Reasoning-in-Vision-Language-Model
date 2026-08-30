"""Guards on the per-image object inventory built from GQA question programs.

WHAT THESE PROTECT. The inventory is the only deterministic source of image truth available on
Test-Dev, which publishes no scene graphs. It is assembled from the presence markers GQA embeds in
each question program (`select: woman (7)` is a real scene-graph object; `select: glasses (-)` is
not). Two things can silently destroy it: a parser that mistakes the marker for part of the object
name, and a contradiction between questions about the same image. The second must abort the run
rather than be averaged away, so it is tested with an injected contradiction.

The inventory is OPEN-WORLD. Absence of a name from the present set is not evidence of absence, and
these tests pin that the module exposes the absent set separately rather than inviting a negation.
"""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.eval.gqa_scene_inventory import (
    INVENTORY_VERSION,
    InventoryContradiction,
    build_inventory,
    inventory_sha256,
    parse_select_argument,
)

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "data/gqa/questions/testdev_balanced_questions.json"


def _q(image_id: str, selects: list[str]) -> dict:
    """Minimal question in the real GQA shape, carrying only the select operations."""
    return {
        "imageId": image_id,
        "question": "synthetic",
        "answer": "yes",
        "fullAnswer": "Yes.",
        "semantic": [{"operation": "select", "argument": a, "dependencies": []} for a in selects],
        "semanticStr": "",
        "types": {"structural": "verify", "semantic": "obj", "detailed": "exist"},
        "annotations": {"answer": {}, "question": {}, "fullAnswer": {}},
    }


class ParseSelectArgumentTests(unittest.TestCase):
    def test_present_object_yields_its_id(self):
        self.assertEqual(parse_select_argument("woman (7)"), ("woman", "7"))

    def test_absent_object_yields_none(self):
        self.assertEqual(parse_select_argument("glasses (-)"), ("glasses", None))

    def test_multiword_name_is_kept_whole(self):
        self.assertEqual(parse_select_argument("folding chair (-)"), ("folding chair", None))

    def test_name_is_normalised_to_lowercase(self):
        self.assertEqual(parse_select_argument("Trash Can (12)"), ("trash can", "12"))

    def test_unparseable_argument_returns_none(self):
        self.assertIsNone(parse_select_argument("scene"))

    def test_marker_is_not_absorbed_into_the_name(self):
        # The failure this exists for: "woman (7)" parsed as the object name "woman (7)".
        name, ident = parse_select_argument("woman (7)")
        self.assertNotIn("(", name)
        self.assertEqual(ident, "7")


class InventoryBuildTests(unittest.TestCase):
    def test_present_and_absent_are_separate_sets(self):
        inv = build_inventory({"1": _q("img1", ["woman (7)", "glasses (-)"])})
        self.assertEqual(inv["img1"]["present"], frozenset({"woman"}))
        self.assertEqual(inv["img1"]["absent"], frozenset({"glasses"}))

    def test_evidence_accumulates_across_questions_about_one_image(self):
        inv = build_inventory({
            "1": _q("img1", ["woman (7)"]),
            "2": _q("img1", ["umbrella (0)"]),
        })
        self.assertEqual(inv["img1"]["present"], frozenset({"woman", "umbrella"}))

    def test_contradiction_aborts(self):
        # T10. A name declared both present and absent for one image is unresolvable evidence,
        # not a tie to be broken.
        with self.assertRaises(InventoryContradiction):
            build_inventory({
                "1": _q("img1", ["woman (7)"]),
                "2": _q("img1", ["woman (-)"]),
            })

    def test_contradiction_message_names_the_image_and_the_object(self):
        try:
            build_inventory({"1": _q("imgX", ["woman (7)"]), "2": _q("imgX", ["woman (-)"])})
        except InventoryContradiction as exc:
            self.assertIn("imgX", str(exc))
            self.assertIn("woman", str(exc))
        else:
            self.fail("no contradiction raised")

    def test_hash_is_stable_across_construction_order(self):
        a = build_inventory({"1": _q("i", ["a (1)"]), "2": _q("i", ["b (2)"])})
        b = build_inventory({"2": _q("i", ["b (2)"]), "1": _q("i", ["a (1)"])})
        self.assertEqual(inventory_sha256(a), inventory_sha256(b))

    def test_version_is_declared(self):
        self.assertEqual(INVENTORY_VERSION, "gqa_scene_inventory/1.1.0")


class RealDataInventoryTests(unittest.TestCase):
    """The measured properties this design rests on, re-derived rather than quoted."""

    @classmethod
    def setUpClass(cls):
        if not QUESTIONS.is_file():
            raise unittest.SkipTest("requires downloaded GQA Test-Dev questions; see DATA.md")
        cls.questions = json.loads(QUESTIONS.read_text())
        cls.inv = build_inventory(cls.questions)

    def test_every_testdev_image_has_at_least_one_known_present_object(self):
        self.assertEqual(len(self.inv), 398)
        self.assertTrue(all(v["present"] for v in self.inv.values()))

    def test_no_contradictions_on_the_real_surface(self):
        # Building without raising IS the assertion; this pins the count at zero explicitly.
        clashes = [i for i, v in self.inv.items() if v["present"] & v["absent"]]
        self.assertEqual(clashes, [])

    def test_inventory_is_open_world_not_a_scene_graph(self):
        # Median known-absent per image is ~1: the absent set is far too sparse to support a
        # closed-world negation, and no caller may treat it as one.
        sizes = sorted(len(v["absent"]) for v in self.inv.values())
        self.assertLess(sizes[len(sizes) // 2], 5)



class ObjectIdIndexTests(unittest.TestCase):
    """Repair R1 (review finding, critical 1): names alone cannot establish a different referent.

    GQA annotates the same object under several names — `snowboarder`/`person`, `houses`/`building`,
    `street sign`/`sign` — all pointing at one object id. A name-only inventory made every such
    alias look like a wrong referent. The inventory must therefore carry object ids, and a binding
    claim needs two NON-EMPTY, DISJOINT id sets.
    """

    def test_ids_are_preserved_per_name(self):
        inv = build_inventory({"1": _q("img", ["woman (7)", "glasses (-)"])})
        self.assertEqual(inv["img"]["ids"]["woman"], frozenset({"7"}))
        self.assertNotIn("glasses", inv["img"]["ids"])

    def test_multiple_ids_for_one_name_are_unioned(self):
        inv = build_inventory({"1": _q("img", ["chair (3)"]), "2": _q("img", ["chair (9)"])})
        self.assertEqual(inv["img"]["ids"]["chair"], frozenset({"3", "9"}))

    def test_comma_separated_ids_are_split(self):
        inv = build_inventory({"1": _q("img", ["chairs (3,9)"])})
        self.assertEqual(inv["img"]["ids"]["chairs"], frozenset({"3", "9"}))

    def test_a_trailing_space_in_the_argument_is_tolerated(self):
        # GQA writes 'chair (7) ' with a trailing space in filtered-OR programs. A parser that
        # misses it silently drops a disjunct, which is how eight unsupported rescues survived.
        self.assertEqual(parse_select_argument("chair (7) "), ("chair", "7"))

    def test_present_names_still_available_for_open_world_checks(self):
        inv = build_inventory({"1": _q("img", ["woman (7)"])})
        self.assertEqual(inv["img"]["present"], frozenset({"woman"}))


class RealDataObjectIdTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not QUESTIONS.is_file():
            raise unittest.SkipTest("requires downloaded GQA Test-Dev questions; see DATA.md")
        cls.inv = build_inventory(json.loads(QUESTIONS.read_text()))

    def test_every_present_name_has_at_least_one_object_id(self):
        bad = [(img, n) for img, v in self.inv.items() for n in v["present"] if not v["ids"].get(n)]
        self.assertEqual(bad, [])

    def test_aliases_of_one_object_are_detectable(self):
        # qid 20724203: gold `snowboarder`, prediction `person`, both object 0 in that image.
        hits = [img for img, v in self.inv.items()
                if v["ids"].get("snowboarder") and v["ids"].get("person")
                and v["ids"]["snowboarder"] & v["ids"]["person"]]
        self.assertTrue(hits, "no image where snowboarder and person share an id")


if __name__ == "__main__":
    unittest.main()
