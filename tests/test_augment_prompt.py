"""Pin the augmentation prompts byte-for-byte across the move into src.prompt.

The literals below are FROZEN COPIES of the strings src/augment/__init__.py built
before the port. They are duplicated here on purpose: a test that imports the
constant it is checking cannot detect the constant changing. Every existing
augmentation result was produced by these exact bytes, so any drift silently
invalidates comparisons against them.

Written and run against the UNPORTED module first, so it characterises real past
behaviour rather than the behaviour the port happens to produce.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import prompt as P  # noqa: E402
from src.augment import (CoTAugment, NoAugment, PredQAugment, PredQPlainAugment,  # noqa: E402
                         SceneGraphAugment)

# --- frozen legacy literals (do not "fix", do not import) --------------------
LEGACY_SHORT_SUFFIX = " Answer in one word or a short phrase."
LEGACY_COT_INSTRUCTION = (
    " Think step by step about the image, then give the final answer. "
    "End your response with 'Answer:' followed by one word or a short phrase."
)

QUESTIONS = [
    "Is the sky blue?",
    "What color is the bus to the left of the man?",
    "Are there both a cat and a dog in this photo?",
    "",
]
CONTEXTS = [
    "a man riding a horse; a horse on grass",
    "chair at (0.1, 0.2); table at (0.5, 0.6)",
]


class _FakeStore:
    """Minimal store stand-in: describe() returns whatever it was handed."""

    def __init__(self, desc=""):
        self.desc = desc

    def describe(self, key):
        """Return the canned description regardless of key."""
        return self.desc


def _legacy_none(q):
    """The exact prompt NoAugment.build_prompt produced before the port."""
    return q + LEGACY_SHORT_SUFFIX


def _legacy_cot(q):
    """The exact prompt CoTAugment.build_prompt produced before the port."""
    return q + LEGACY_COT_INSTRUCTION


def _legacy_graph_plain(q, ctx):
    """The exact prompt SceneGraphAugment.build_prompt produced before the port."""
    if ctx:
        return (f"Here is a structured description of the image: {ctx}.\n\n"
                f"Using this description and the image, answer the question."
                f" {q}{LEGACY_SHORT_SUFFIX}")
    return q + LEGACY_SHORT_SUFFIX


def _legacy_graph_hedged(q, ctx):
    """The exact prompt PredQAugment.build_prompt produced before the port."""
    if ctx:
        return (f"An automatic object detector found these objects in the image "
                f"(the list may be incomplete or imperfect): {ctx}.\n\n"
                f"The image itself is the final authority; use the detections only as "
                f"hints. {q}{LEGACY_SHORT_SUFFIX}")
    return q + LEGACY_SHORT_SUFFIX


# --- the literals themselves -------------------------------------------------

def test_short_cue_is_byte_identical():
    """src.prompt.SHORT_CUE must equal the augment module's old SHORT_SUFFIX."""
    assert P.SHORT_CUE == LEGACY_SHORT_SUFFIX


def test_cot_instruction_is_byte_identical():
    """src.prompt.COT_INSTRUCTION must equal the augment module's old copy."""
    assert P.COT_INSTRUCTION == LEGACY_COT_INSTRUCTION


# --- composed prompts, augmenter by augmenter --------------------------------

@pytest.mark.parametrize("q", QUESTIONS)
def test_none_prompt_unchanged(subtests, q):
    """The baseline prompt is unchanged and matches src.prompt's direct body."""
    aug = NoAugment()
    with subtests.test(where="augmenter"):
        assert aug.build_prompt(q, None) == _legacy_none(q)
    with subtests.test(where="src.prompt"):
        spec = P.PromptSpec(input_mode="image_only", reason_mode="direct")
        assert P.user_body(spec, q) == _legacy_none(q)


@pytest.mark.parametrize("q", QUESTIONS)
def test_cot_prompt_unchanged(subtests, q):
    """The CoT prompt is unchanged and matches src.prompt's cot body."""
    aug = CoTAugment()
    with subtests.test(where="augmenter"):
        assert aug.build_prompt(q, None) == _legacy_cot(q)
    with subtests.test(where="src.prompt"):
        spec = P.PromptSpec(input_mode="image_only", reason_mode="cot")
        assert P.user_body(spec, q) == _legacy_cot(q)


@pytest.mark.parametrize("q", QUESTIONS)
@pytest.mark.parametrize("ctx", CONTEXTS + ["", None])
def test_graph_plain_prompt_unchanged(subtests, q, ctx):
    """Oracle/plain graph framing is unchanged, including the no-graph fallback."""
    aug = SceneGraphAugment(_FakeStore())
    with subtests.test(where="augmenter"):
        assert aug.build_prompt(q, ctx) == _legacy_graph_plain(q, ctx)
    with subtests.test(where="src.prompt"):
        spec = P.PromptSpec(input_mode="image_plus_graph", reason_mode="direct",
                            graph_framing="plain")
        assert P.user_body(spec, q, ctx) == _legacy_graph_plain(q, ctx)


@pytest.mark.parametrize("q", QUESTIONS)
@pytest.mark.parametrize("ctx", CONTEXTS + ["", None])
def test_graph_hedged_prompt_unchanged(subtests, q, ctx):
    """Hedged graph framing is unchanged, including the no-graph fallback."""
    aug = PredQAugment(_FakeStore())
    with subtests.test(where="augmenter"):
        assert aug.build_prompt(q, ctx) == _legacy_graph_hedged(q, ctx)
    with subtests.test(where="src.prompt"):
        spec = P.PromptSpec(input_mode="image_plus_graph", reason_mode="direct",
                            graph_framing="hedged")
        assert P.user_body(spec, q, ctx) == _legacy_graph_hedged(q, ctx)


@pytest.mark.parametrize("q", QUESTIONS)
@pytest.mark.parametrize("ctx", CONTEXTS)
def test_predq_plain_inherits_plain_framing(q, ctx):
    """PredQPlainAugment keeps the ORIGINAL framing — it is the A/B control arm."""
    assert PredQPlainAugment(_FakeStore()).build_prompt(q, ctx) == _legacy_graph_plain(q, ctx)


# --- coverage counters: the silent-fallback guard ----------------------------

def test_coverage_counts_missing_graphs():
    """A missing graph must be counted, since it silently degrades to the plain prompt."""
    aug = SceneGraphAugment(_FakeStore(""))
    ex = type("Ex", (), {"image_id": "1", "qid": "q1"})()
    for _ in range(3):
        aug.context_for(ex)
    assert (aug.n_asked, aug.n_with_graph) == (3, 0)
    assert "3 silently fell back" in aug.coverage()


def test_spec_inherits_protocol_but_keeps_augmentation_identity():
    """The protocol levers come from the base spec; the framing comes from the augmenter."""
    base = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True,
                        input_mode="text_only", reason_mode="direct")
    got = PredQAugment(_FakeStore()).spec(base)
    assert (got.prompt_format, got.supervise_eos) == ("chatml_v1", True)   # from base
    assert (got.input_mode, got.graph_framing) == ("image_plus_graph", "hedged")  # from augmenter


def test_spec_defaults_to_legacy_layer():
    """With no base spec the augmenter renders under the legacy protocol."""
    assert NoAugment().spec().evidence_layer == P.EVIDENCE_LEGACY


@pytest.mark.parametrize("aug,legacy_cap,corrected_cap", [
    (NoAugment(), 10, 32),
    (CoTAugment(), 128, 128),
    (SceneGraphAugment(_FakeStore()), 10, 32),
])
def test_budget_switches_with_evidence_layer(subtests, aug, legacy_cap, corrected_cap):
    """Legacy runs keep their historical cap; corrected runs use the protocol budget."""
    corrected = P.PromptSpec(prompt_format="chatml_v1", supervise_eos=True)
    with subtests.test(layer="legacy"):
        assert aug.budget() == legacy_cap
        assert aug.budget() == aug.max_new_tokens          # attribute stays the legacy value
    with subtests.test(layer="corrected"):
        assert aug.budget(corrected) == corrected_cap


def test_cod_uses_its_own_instruction_and_cot_parsing():
    """Chain-of-draft gets the COD text but keeps CoT's 'Answer:' contract."""
    from src.augment import CoDAugment
    aug = CoDAugment()
    assert aug.build_prompt("Is it red?") == "Is it red?" + P.COD_INSTRUCTION
    assert aug.extract_answer("Sky blue. Grass green.\nAnswer: red") == "red"


@pytest.mark.parametrize("aug", [NoAugment(), CoTAugment(), SceneGraphAugment(_FakeStore()),
                                 PredQAugment(_FakeStore())])
@pytest.mark.parametrize("q", QUESTIONS[:2])
@pytest.mark.parametrize("ctx", [CONTEXTS[0], ""])
def test_threading_a_legacy_spec_changes_nothing(aug, q, ctx):
    """09_augment_eval now passes a base spec; under the legacy protocol that is a no-op.

    Guards the port itself: the evaluator went from build_prompt(q, ctx) to
    build_prompt(q, ctx, base_spec), and a legacy base_spec must reproduce the old
    string exactly or every existing augmentation result silently shifts.
    """
    legacy = P.PromptSpec()          # raw, no EOS: the protocol every past run used
    assert aug.build_prompt(q, ctx, legacy) == aug.build_prompt(q, ctx)


def test_results_table_names_the_split_it_scored():
    """The table must name the actual qid set, not assert every slice is the locked one."""
    from src.eval.report import render_results_table
    by_cat = {"exist": {"a_exact": 1, "a_vqa": 1, "b_exact": 0, "b_vqa": 0, "total": 2}}
    text = render_results_table("T", 2, "A", "B", (1, 1, 0, 0), by_cat, ["exist"],
                                "tune_500_qids")
    assert "`tune_500_qids`" in text
    assert "Locked evaluation set" not in text


def test_predq_keys_on_qid_not_image_id():
    """Question-conditioned stores are keyed by qid; using image_id would cross-wire them."""
    seen = []

    class Recording(_FakeStore):
        def describe(self, key):
            """Record the lookup key so the wiring is observable."""
            seen.append(key)
            return "obj"

    ex = type("Ex", (), {"image_id": "IMG", "qid": "QID"})()
    PredQPlainAugment(Recording()).context_for(ex)
    SceneGraphAugment(Recording()).context_for(ex)
    assert seen == ["QID", "IMG"]
