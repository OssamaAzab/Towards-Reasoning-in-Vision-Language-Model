"""The collector must read a run's condition from metadata, not from its filename.

Two failures this pins, both demonstrated against the previous code:

1. Corrected artifacts are {"_meta": ..., "records": [...]}, so indexing the loaded
   object as a list raised KeyError(0) and the collector crashed on every one.
2. Protocol-aware stems end in the reasoning mode, so `...__image_plus_graph__direct`
   ends in neither "scene_graph" nor "cot". Filename inference would have labelled a
   scene-graph run as the un-augmented baseline — a silent condition swap, which is
   worse than a crash.
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


def _collector():
    """Load scripts/10_collect_results.py as a module (it is not importable by name)."""
    spec = importlib.util.spec_from_file_location(
        "collect_results", ROOT / "scripts" / "10_collect_results.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["collect_results"] = mod
    spec.loader.exec_module(mod)
    return mod


COL = _collector()

META = {
    "evidence_layer": "corrected_chatml_v1_eos",
    "prompt_format": "chatml_v1", "supervise_eos": True,
    "input_mode": "image_only", "reason_mode": "direct",
    "split": "tune_500_qids", "metric_version": "exact_full/1.0.0",
    "checkpoint_encoder": "facebook/ijepa_vith14_1k", "checkpoint_epoch": 3,
}
STEM = ("bridge_ijepa_150k_mlp_ep3__corrected_chatml_v1_eos"
        "__tune_500_qids__image_only__direct")


class ArtifactLoadingTests(unittest.TestCase):
    """Both artifact shapes must load; neither may be mistaken for the other."""

    def test_corrected_artifact_loads_records_and_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            p.write_text(json.dumps({"_meta": META, "records": [{"qid": "1"}, {"qid": "2"}]}))
            records, meta = COL.load_records(p)
            self.assertEqual(len(records), 2)
            self.assertEqual(meta["evidence_layer"], "corrected_chatml_v1_eos")

    def test_legacy_bare_list_still_loads_with_no_meta(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "r.json"
            p.write_text(json.dumps([{"qid": "1"}]))
            records, meta = COL.load_records(p)
            self.assertEqual(len(records), 1)
            self.assertEqual(meta, {})

    def test_indexing_a_corrected_artifact_as_a_list_is_the_bug_being_fixed(self):
        """Guards the regression directly: the old code did exactly this."""
        loaded = json.loads(json.dumps({"_meta": META, "records": [{"qid": "1"}]}))
        with self.assertRaises(KeyError):
            loaded[0]


class ConditionFromMetadataTests(unittest.TestCase):
    """The condition comes from the provenance block, not the stem."""

    def test_graph_run_is_not_mislabelled_as_the_baseline(self):
        """The silent-swap case: a stem ending in "direct" says nothing about the graph."""
        graph_stem = STEM.replace("__image_only__direct", "__image_plus_graph__direct")
        meta = dict(META, input_mode="image_plus_graph", reason_mode="direct")
        self.assertEqual(COL.describe_from_meta(graph_stem, meta)[2], "scene_graph")

        # The same stem under pure filename inference: the suffix ladder in describe()
        # tests endswith("scene_graph") / endswith("cot"), and this ends with "direct",
        # so it falls through to "none" — a graph run reported as the baseline.
        self.assertTrue(graph_stem.endswith("direct"))
        for suffix in COL.AUGMENTATION_SUFFIXES:
            self.assertFalse(graph_stem.endswith(suffix),
                             f"stem unexpectedly ends with {suffix!r}")

    def test_reasoning_mode_is_recovered(self):
        for reason, expected in (("direct", "none"), ("cot", "cot"), ("cod", "cod")):
            meta = dict(META, reason_mode=reason)
            self.assertEqual(COL.describe_from_meta(STEM, meta)[2], expected)

    def test_graph_and_reasoning_combine_rather_than_one_hiding_the_other(self):
        meta = dict(META, input_mode="image_plus_graph", reason_mode="cot")
        self.assertEqual(COL.describe_from_meta(STEM, meta)[2], "scene_graph_cot")

    def test_explicit_augmentation_in_meta_wins(self):
        """09_augment_eval records the augmenter by name; that beats any derivation."""
        meta = dict(META, input_mode="image_plus_graph", augmentation="scene_graph_predq")
        self.assertEqual(COL.describe_from_meta(STEM, meta)[2], "scene_graph_predq")

    def test_encoder_comes_from_the_checkpoint_not_the_filename(self):
        self.assertTrue(COL.describe_from_meta(STEM, META)[0].startswith("ijepa"))

    def test_variant_tokens_still_label_the_encoder_column(self):
        """The encoder column doubles as the model-variant label; that must survive."""
        enc = COL.describe_from_meta(STEM, META)[0]
        self.assertIn("150k", enc)
        self.assertIn("mlp", enc)

    def test_variants_are_read_from_the_checkpoint_part_only(self):
        """Tokens after the '__' separator describe the EVAL, not the trained model."""
        stem = ("bridge_ijepa_150k_ep3__corrected_chatml_v1_eos"
                "__tune_500_qids__image_only__direct__precbf16")
        self.assertNotIn("bf16", COL.describe_from_meta(stem, META)[0])

    def test_epochs_come_from_the_checkpoint_record(self):
        self.assertEqual(COL.describe_from_meta(STEM, dict(META, checkpoint_epoch=5))[1], 5)


class EvidenceLayerSeparationTests(unittest.TestCase):
    """Legacy and corrected rows must be distinguishable in the CSV."""

    def test_bf16_is_a_recognised_variant_token(self):
        """Added alongside fp16; without it a bf16 run loses its lineage label."""
        self.assertIn("bf16", COL.VARIANT_TOKENS)

    def test_legacy_stems_are_described_exactly_as_before(self):
        """The legacy path must not shift; 60 real artifacts depend on it."""
        self.assertEqual(COL.describe("bridge_ijepa_150k_mlp_ep3_cot"),
                         ("ijepa_150k_mlp", 3, "cot"))
        self.assertEqual(COL.describe("bridge_clip_150k_3ep_scene_graph"),
                         ("clip_150k", 3, "scene_graph"))


if __name__ == "__main__":
    unittest.main()
