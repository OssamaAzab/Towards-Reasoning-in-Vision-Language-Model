"""Tests for the visual-audit mask and probe-label construction.

Every test here guards a way the audit could produce a control that is not a control: a
"relevant" box chosen by inference rather than by GQA's annotations, an "irrelevant" control that
differs in size as well as relevance, or a probe target that a majority-class guess would win.
"""
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from src.data.visual_audit import (AREA_TOL, COLOURS, MIN_BOX_PX, SPATIAL_RELATIONS,
                                   mask_pair, probe_labels, referenced_object_ids,
                                   relative_position, usable_box)


def obj(name, x, y, w, h, attrs=(), rels=()):
    return {"name": name, "x": x, "y": y, "w": w, "h": h,
            "attributes": list(attrs), "relations": list(rels)}


def graph(objects, w=600, h=400):
    return {"width": w, "height": h, "objects": objects}


class ReferenceExtractionTests(unittest.TestCase):

    def test_ids_come_from_the_annotations_map(self):
        q = {"annotations": {"question": {"2": "413787"}, "answer": {}, "fullAnswer": {}}}
        self.assertEqual(referenced_object_ids(q), {"413787"})

    def test_ids_also_come_from_the_semantic_programme(self):
        # The surface question elides referents that relate/filter steps introduce.
        q = {"annotations": {}, "semantic": [{"argument": "birds (413787)"},
                                             {"argument": "to the left of (999,1000)"}]}
        self.assertEqual(referenced_object_ids(q), {"413787", "999", "1000"})

    def test_comma_separated_annotation_values_split(self):
        q = {"annotations": {"question": {"1": "11,22"}}}
        self.assertEqual(referenced_object_ids(q), {"11", "22"})

    def test_no_annotations_yields_no_referents_rather_than_a_guess(self):
        self.assertEqual(referenced_object_ids({"question": "is the man tall?"}), set())


class BoxUsabilityTests(unittest.TestCase):

    def test_tiny_boxes_are_unusable(self):
        self.assertFalse(usable_box(obj("x", 0, 0, MIN_BOX_PX - 1, 50), 600, 400))

    def test_off_image_boxes_are_unusable(self):
        self.assertFalse(usable_box(obj("x", 590, 0, 50, 50), 600, 400))

    def test_whole_image_boxes_are_unusable(self):
        # Masking ~everything is condition 3 by another route, not an object mask.
        self.assertFalse(usable_box(obj("x", 0, 0, 600, 400), 600, 400))

    def test_an_ordinary_box_is_usable(self):
        self.assertTrue(usable_box(obj("x", 10, 10, 100, 80), 600, 400))


class MaskPairTests(unittest.TestCase):

    def _q(self, ids):
        return {"annotations": {"question": {str(i): v for i, v in enumerate(ids)}}}

    def test_matched_control_is_within_area_tolerance(self):
        g = graph({"1": obj("dog", 10, 10, 100, 100),        # relevant, area 10000
                   "2": obj("cat", 200, 10, 98, 100),        # area 9800 -> within 25%
                   "3": obj("bus", 300, 10, 20, 20)})        # area 400 -> too small
        r = mask_pair(self._q(["1"]), g)
        self.assertNotIn("reject", r)
        self.assertEqual(r["relevant_id"], "1")
        self.assertEqual(r["irrelevant_id"], "2")
        self.assertLessEqual(abs(r["area_ratio"] - 1.0), AREA_TOL)

    def test_no_size_matched_control_is_rejected_not_substituted(self):
        g = graph({"1": obj("dog", 10, 10, 100, 100),
                   "2": obj("cat", 200, 10, 20, 20)})
        r = mask_pair(self._q(["1"]), g)
        self.assertIn("reject", r)
        self.assertIn("area", r["reject"])

    def test_the_control_is_never_a_referenced_object(self):
        g = graph({"1": obj("dog", 10, 10, 100, 100),
                   "2": obj("cat", 200, 10, 100, 100),
                   "3": obj("bus", 350, 10, 99, 100)})
        r = mask_pair(self._q(["1", "2"]), g)
        self.assertNotIn("reject", r)
        self.assertNotIn(r["irrelevant_id"], {"1", "2"})

    def test_the_largest_referent_is_the_one_masked(self):
        g = graph({"1": obj("dog", 10, 10, 40, 40),
                   "2": obj("man", 100, 10, 120, 120),
                   "9": obj("tree", 300, 10, 118, 120)})
        r = mask_pair(self._q(["1", "2"]), g)
        self.assertEqual(r["relevant_id"], "2", "masking a negligible referent would make a "
                                                "null under condition 5 uninformative")

    def test_a_question_naming_no_object_is_rejected(self):
        g = graph({"1": obj("dog", 10, 10, 100, 100), "2": obj("cat", 200, 10, 100, 100)})
        self.assertIn("reject", mask_pair({"annotations": {}}, g))

    def test_an_image_without_a_graph_is_rejected(self):
        self.assertIn("reject", mask_pair(self._q(["1"]), {"width": 0, "height": 0,
                                                           "objects": {}}))


class ProbeLabelTests(unittest.TestCase):

    def test_relative_position_uses_the_two_largest_objects(self):
        # b is to the right of a in image coordinates -> a is 'left' of b by this convention.
        g = graph({"1": obj("a", 0, 100, 100, 100), "2": obj("b", 400, 100, 100, 100),
                   "3": obj("tiny", 50, 50, 10, 10)})
        self.assertEqual(relative_position(g), "left")

    def test_relative_position_distinguishes_vertical(self):
        g = graph({"1": obj("a", 100, 0, 100, 100), "2": obj("b", 100, 250, 100, 100)})
        self.assertEqual(relative_position(g), "above")

    def test_relative_position_needs_two_objects(self):
        self.assertIsNone(relative_position(graph({"1": obj("a", 0, 0, 100, 100)})))

    def test_only_colour_attributes_become_the_colour_label(self):
        g = graph({"1": obj("car", 0, 0, 100, 100, attrs=["wooden", "large", "red"]),
                   "2": obj("car", 200, 0, 100, 100, attrs=["metal"])})
        self.assertEqual(probe_labels(g)["colour"], "red")

    def test_brightness_qualifiers_are_not_colours(self):
        for w in ("dark", "light"):
            with self.subTest(word=w):
                self.assertNotIn(w, COLOURS)

    def test_count_is_bucketed_and_capped(self):
        g = graph({str(i): obj("bird", i * 20, 0, 15, 15) for i in range(9)})
        lab = probe_labels(g)
        self.assertEqual(lab["count"], "5", "counts above the cap collapse into the top bucket")
        self.assertEqual(lab["count_raw"], 9)

    def test_only_spatial_relations_feed_the_relation_label(self):
        g = graph({"1": obj("nose", 0, 0, 20, 20, rels=[{"name": "of", "object": "2"}]),
                   "2": obj("man", 100, 0, 100, 100,
                            rels=[{"name": "to the left of", "object": "1"}])})
        self.assertEqual(probe_labels(g)["relation"], "to the left of")
        self.assertNotIn("of", SPATIAL_RELATIONS)

    def test_missing_properties_are_omitted_not_invented(self):
        g = graph({"1": obj("thing", 0, 0, 100, 100), "2": obj("other", 200, 0, 100, 100)})
        lab = probe_labels(g)
        self.assertNotIn("colour", lab, "no colour attribute must mean no colour label")
        self.assertIn("object", lab)



class InterventionTests(unittest.TestCase):
    """Conditions 4-6 alter the model's INPUT only; no model code is touched anywhere."""

    def setUp(self):
        import torch
        self.torch = torch
        # token 0 is a distinctive CLS; patches 1..8 carry their own index as content
        self.f = torch.zeros(1, 9, 4)
        self.f[0, 0, :] = 99.0
        for i in range(1, 9):
            self.f[0, i, :] = float(i)

    def _g(self, seed=0):
        return self.torch.Generator().manual_seed(seed)

    def test_cls_token_is_never_moved(self):
        from src.data.visual_audit import shuffle_patch_positions
        out = shuffle_patch_positions(self.f, self._g(), has_cls=True)
        self.assertTrue(self.torch.equal(out[0, 0], self.f[0, 0]),
                        "moving CLS would change which token drop_cls discards")

    def test_patch_contents_are_preserved_as_a_multiset(self):
        from src.data.visual_audit import shuffle_patch_positions
        out = shuffle_patch_positions(self.f, self._g(), has_cls=True)
        before = sorted(self.f[0, 1:, 0].tolist())
        after = sorted(out[0, 1:, 0].tolist())
        self.assertEqual(before, after, "shuffling must move content, never alter it")

    def test_the_permutation_actually_permutes(self):
        from src.data.visual_audit import shuffle_patch_positions
        out = shuffle_patch_positions(self.f, self._g(1), has_cls=True)
        self.assertFalse(self.torch.equal(out[0, 1:], self.f[0, 1:]))

    def test_without_cls_every_token_is_in_play(self):
        from src.data.visual_audit import shuffle_patch_positions
        out = shuffle_patch_positions(self.f, self._g(2), has_cls=False)
        self.assertEqual(sorted(out[0, :, 0].tolist()), sorted(self.f[0, :, 0].tolist()))

    def test_the_input_tensor_is_not_mutated(self):
        from src.data.visual_audit import shuffle_patch_positions
        keep = self.f.clone()
        shuffle_patch_positions(self.f, self._g(3), has_cls=True)
        self.assertTrue(self.torch.equal(self.f, keep), "in-place mutation would corrupt the "
                                                        "correct-image arm computed from the "
                                                        "same tensor")

    def test_masking_changes_only_the_named_region(self):
        from PIL import Image
        from src.data.visual_audit import mask_image_region
        im = Image.new("RGB", (100, 100), (10, 20, 30))
        out = mask_image_region(im, [10, 10, 40, 40])
        self.assertEqual(out.getpixel((25, 25)), (127, 127, 127))
        self.assertEqual(out.getpixel((80, 80)), (10, 20, 30))
        self.assertEqual(im.getpixel((25, 25)), (10, 20, 30), "the source image must not change")

class PromptParityTests(unittest.TestCase):
    """The audit must send the SAME user body as scripts/07, or it measures a different model.

    vlm_generate and text_only_generate call P.build(..., compose=False): the string they receive
    IS the finished body and nothing is appended downstream. scripts/06c trains with the
    short-answer cue and scripts/07 evaluates with it. Job 2291478 passed the bare question, so
    every arm ran out of distribution and 10/10 text-only generations truncated mid-refusal.
    """

    SRC = (ROOT / "scripts" / "51_visual_audit_eval.py").read_text()

    def test_the_audit_composes_the_same_body_as_scripts_07(self):
        import src.prompt as P
        spec = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True)
        q = "What colour is the dog?"
        self.assertEqual(P.user_body(spec, q), q + P.SHORT_CUE)

    def test_the_cue_is_not_empty(self):
        """A blank cue would make the parity test above pass while changing nothing."""
        import src.prompt as P
        self.assertTrue(P.SHORT_CUE.strip(), "SHORT_CUE must be a real instruction")

    def test_generate_calls_do_not_pass_the_bare_question(self):
        """The exact regression from job 2291478: `ex.question` handed straight to a generator."""
        for bad in ("text_only_generate(llm, ex.question", "vlm_generate(bridges[n], llm, f, ex.question"):
            self.assertNotIn(bad, self.SRC,
                             f"scripts/51 passes the bare question ({bad!r}); it must pass the "
                             f"composed body, or the cue the bridge was trained with is missing")

    def test_the_audit_routes_through_user_body(self):
        self.assertIn("P.user_body(spec, ex.question)", self.SRC,
                      "the prompt must be composed by the single prompt-composition function")

    def test_the_source_scan_would_catch_a_regression(self):
        """Guard the guard: the scan must fail on source that contains the defect."""
        regressed = self.SRC.replace("vlm_generate(bridges[n], llm, f, prompt, device",
                                     "vlm_generate(bridges[n], llm, f, ex.question, device")
        self.assertNotEqual(regressed, self.SRC, "the fixed call site was not found to mutate")
        self.assertIn("vlm_generate(bridges[n], llm, f, ex.question", regressed)


if __name__ == "__main__":
    unittest.main()
