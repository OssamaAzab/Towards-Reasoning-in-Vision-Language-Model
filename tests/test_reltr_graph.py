"""Tests for the RelTR -> T-049 graph conversion.

The point of every test here is that the RelTR arm must be rendered by the SAME store, into the
SAME text shape, as the OWLv2 arm it replaces. A conversion bug that changed the text would show
up as an accuracy difference and be misread as a graph-quality effect.
"""
import json
import tempfile
import unittest
from pathlib import Path

from src.data.predicted_graph import PredictedGraphStore
from src.data.reltr_graph import (ATTRIBUTE_PREDICATE, DEFAULT_SCORE_THRESH, DEFAULT_TOPK,
                                  readable, triplets_to_graph)


def trip(sub, sbox, pred, obj, obox, ss=0.9, ps=0.8, os_=0.85):
    """One RelTR triplet in the shape decode_triplets emits."""
    return {"sub_name": sub, "sub_box": sbox, "sub_score": ss,
            "pred_name": pred, "pred_score": ps,
            "obj_name": obj, "obj_box": obox, "obj_score": os_}


class ReadableTests(unittest.TestCase):

    def test_open_images_surface_forms_are_normalised(self):
        for raw, want in (("Wine glass", "wine glass"), ("talk_on_phone", "talk on phone"),
                          ("Man", "man"), ("inside_of", "inside of"), ("Woman", "woman")):
            with self.subTest(raw=raw):
                self.assertEqual(readable(raw), want)


class EntityMergeTests(unittest.TestCase):

    def test_same_class_overlapping_boxes_merge_into_one_entity(self):
        # The same man is the subject of two triplets; the graph must show ONE man.
        g = triplets_to_graph([
            trip("Man", [0, 0, 100, 200], "wears", "Hat", [10, 0, 60, 30]),
            trip("Man", [2, 1, 101, 199], "holds", "Bottle", [80, 90, 110, 140]),
        ])
        names = sorted(o["name"] for o in g["objects"].values())
        self.assertEqual(names, ["bottle", "hat", "man"])
        man = [o for o in g["objects"].values() if o["name"] == "man"][0]
        self.assertEqual(len(man["relations"]), 2)

    def test_same_class_disjoint_boxes_stay_separate(self):
        g = triplets_to_graph([
            trip("Man", [0, 0, 50, 100], "wears", "Hat", [0, 0, 20, 20]),
            trip("Man", [400, 0, 450, 100], "holds", "Bottle", [400, 50, 420, 90]),
        ])
        self.assertEqual(sum(1 for o in g["objects"].values() if o["name"] == "man"), 2)

    def test_different_classes_never_merge_however_aligned(self):
        g = triplets_to_graph([trip("Man", [0, 0, 100, 100], "wears", "Hat", [0, 0, 100, 100])])
        self.assertEqual(len(g["objects"]), 2)


class AttributePredicateTests(unittest.TestCase):

    def test_is_becomes_an_attribute_not_a_relation(self):
        g = triplets_to_graph([trip("Table", [0, 0, 90, 90], ATTRIBUTE_PREDICATE,
                                    "Wooden", [0, 0, 90, 90])])
        self.assertEqual(len(g["objects"]), 1, "'is' must not create an object node")
        (o,) = g["objects"].values()
        self.assertEqual(o["name"], "table")
        self.assertEqual(o["attributes"], ["wooden"])
        self.assertEqual(o["relations"], [])
        self.assertEqual(g["_n_attribute_triplets"], 1)

    def test_is_attributes_never_reach_the_rendered_text(self):
        # PredictedGraphStore strips attributes, exactly as it did for T-049 colour tags.
        g = triplets_to_graph([
            trip("Table", [0, 0, 90, 90], ATTRIBUTE_PREDICATE, "Wooden", [0, 0, 90, 90]),
            trip("Bottle", [200, 0, 240, 60], "on", "Table", [0, 0, 90, 90]),
        ])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"img1": g}))
            text = PredictedGraphStore(str(p)).describe("img1")
        self.assertNotIn("wooden", text)
        self.assertIn("a bottle on a table", text)


class DegenerateTripletTests(unittest.TestCase):

    def test_self_relation_after_merge_is_dropped(self):
        g = triplets_to_graph([trip("Man", [0, 0, 100, 100], "holds", "Man", [1, 1, 99, 99])])
        (o,) = g["objects"].values()
        self.assertEqual(o["relations"], [], "an entity related to itself states nothing")

    def test_duplicate_triplets_collapse(self):
        t = trip("Man", [0, 0, 100, 200], "wears", "Hat", [10, 0, 60, 30])
        g = triplets_to_graph([t, dict(t)])
        man = [o for o in g["objects"].values() if o["name"] == "man"][0]
        self.assertEqual(len(man["relations"]), 1)

    def test_empty_input_yields_an_empty_graph_not_a_crash(self):
        g = triplets_to_graph([])
        self.assertEqual(g["objects"], {})


class RenderedFormatTests(unittest.TestCase):
    """The rendered text must be the T-049 format, character for character."""

    def _render(self, graph):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "c.json"
            p.write_text(json.dumps({"i": graph}))
            return PredictedGraphStore(str(p)).describe("i")

    def test_relation_template_matches_t049(self):
        g = triplets_to_graph([trip("Man", [0, 0, 50, 100], "wears", "Hat", [5, 0, 30, 20])])
        self.assertEqual(self._render(g), "a man wears a hat")

    def test_unrelated_objects_are_listed_after_relations(self):
        g = triplets_to_graph([
            trip("Man", [0, 0, 50, 100], "wears", "Hat", [5, 0, 30, 20]),
            trip("Dog", [300, 300, 380, 360], ATTRIBUTE_PREDICATE, "Brown", [300, 300, 380, 360]),
        ])
        self.assertEqual(self._render(g), "a man wears a hat; a dog")

    def test_facts_are_semicolon_joined(self):
        g = triplets_to_graph([
            trip("Man", [0, 0, 50, 100], "wears", "Hat", [5, 0, 30, 20]),
            trip("Woman", [200, 0, 250, 100], "holds", "Bottle", [260, 20, 280, 60]),
        ])
        self.assertEqual(self._render(g).count("; "), 1)


class UpstreamDefaultsTests(unittest.TestCase):
    """These are RelTR's published values. Changing them after seeing accuracy is forbidden."""

    def test_thresholds_are_the_upstream_inference_defaults(self):
        self.assertEqual(DEFAULT_SCORE_THRESH, 0.3)
        self.assertEqual(DEFAULT_TOPK, 10)


if __name__ == "__main__":
    unittest.main()
