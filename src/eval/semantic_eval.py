"""Level 3 — the deterministic semantic cascade over GQA predictions.

WHAT THIS DOES. Given a question, one prediction, the scene inventory and the frozen synonym
ontology, it decides whether an answer that failed exact match was nevertheless semantically right,
and if not, what kind of error it was. It is a SECONDARY EXPLORATORY DIAGNOSTIC. `exact_full` is
the SOLE primary metric for the corrected layer and is untouched by anything here; the raw-equality
column recomputed by the runner is a compatibility regression against the stored Test-Dev flags,
not a second primary metric.

WHAT IT REFUSES TO DO. There is no image-aware judge and no LLM of any kind: the human refused
Level 4 on 2026-08-13, so `llm` is not a permitted judgment source and every unresolved case
abstains instead. Abstention is a first-class outcome — `semantic_correct` is None, never False —
because folding "we could not tell" into "wrong" is how an evaluation quietly invents evidence.

THE RULE THAT SHAPES EVERYTHING. A prediction is never credited because its words appear in the
question or anywhere in the answer sentence. Rescue requires an annotation-grounded source: the
program's own presence markers, gold-anchored morphology, or the frozen ontology. This is what
stops the evaluator from reading the question back to itself and calling it comprehension. The
`fullAnswer` containment rule that the original specification required was withdrawn in 1.1.0 for exactly that
failure; see the note above `evaluate`'s helpers.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.eval.answer_schema import (
    ALTERNATIVE,
    BOOLEAN,
    OBJECT_NAME,
    _as_number,
    alternatives_of,
    expected_schema,
    matches_schema,
)
from src.eval.gqa_scene_inventory import parse_select_argument
from src.eval.metrics import exact_full, normalize_full

RUBRIC_VERSION = "gqa_semantic_rubric/1.3.0"

# `llm` is deliberately absent. Level 4 was refused; see the module docstring.
JUDGMENT_SOURCES = ("exact", "program", "full_answer", "ontology", "human")

ERROR_TYPES = (
    "correct_content_wrong_format",
    "synonym_or_paraphrase",
    "wrong_answer_type",
    "wrong_polarity",
    "wrong_object",
    "wrong_attribute",
    "wrong_relation",
    "reversed_relation",
    "wrong_referent_binding",
    "counting_error",
    "ambiguous_or_annotation_limited",
    "unresolved",
)

# Declared, not discovered: Test-Dev balanced contains no `count` operation and no numeric gold,
# and no program field distinguishes a wrong RELATION predicate from a wrong object, so these two
# cannot fire on this surface. Naming them here keeps an empty bucket from reading as a measurement.
NOT_APPLICABLE_ON_TESTDEV = ("counting_error", "wrong_relation")

CONFIDENCE_IS_ORDINAL = (
    "Confidence ranks how directly a judgment is grounded in GQA's annotation. It is NOT a "
    "calibrated probability and no analysis may threshold on it without human review."
)
_CONFIDENCE = {"exact": 1.0, "program": 1.0, "full_answer": 0.9, "ontology": 0.8, "human": 1.0}

_ONTOLOGY_PATH = Path(__file__).resolve().parents[2] / "config/gqa_synonyms_v1.json"
_BOOLEAN_WORDS = {"yes", "no"}


@dataclass(frozen=True)
class Label:
    """One (question, prediction) verdict. `semantic_correct` is None exactly when abstaining."""
    exact_correct: bool
    schema_expected: str
    schema_correct: bool
    semantic_correct: bool | None
    judgment_source: str
    error_type: str
    confidence: float
    abstain: bool
    resolvable: bool


def load_ontology(path: Path | None = None) -> tuple[dict[str, frozenset[str]], str]:
    """Load the frozen synonym file into symmetric, transitively closed equivalence groups."""
    data = json.loads((path or _ONTOLOGY_PATH).read_text())
    groups: dict[str, set[str]] = {}
    for a, b in data["pairs"]:
        a, b = a.strip().lower(), b.strip().lower()
        merged = groups.get(a, {a}) | groups.get(b, {b}) | {a, b}
        for term in merged:
            groups[term] = merged
    return {k: frozenset(v) for k, v in groups.items()}, data["version"]


def ontology_status(path: Path | None = None) -> str:
    """Review status of the shipped ontology; UNREVIEWED_SEED until a human says otherwise."""
    return json.loads((path or _ONTOLOGY_PATH).read_text())["status"]


def same_number(a: str, b: str) -> bool:
    """True if both strings denote the same integer, in digits or in words."""
    na, nb = _as_number(a), _as_number(b)
    return na is not None and nb is not None and na == nb


# GQA question wording pluralises ("Are there any benches or books?") while the program stores the
# singular ("bench"). Matching the two is a morphology fold, not a semantic claim. It is confined to
# disjunct matching and is BLOCKED for pairs GQA treats as different objects.
_BLOCKED_FOLDS = frozenset({
    frozenset({"glass", "glasses"}),     # eyewear vs. the material/vessel
    frozenset({"pant", "pants"}),        # "pant" is not a GQA object
    frozenset({"short", "shorts"}),      # the adjective vs. the garment
})


def _matches_disjunct(prediction: str, name: str) -> bool:
    """True if a prediction names this disjunct, allowing a singular/plural difference only."""
    if prediction == name:
        return True
    if frozenset({prediction, name}) in _BLOCKED_FOLDS:
        return False
    short, long_ = sorted((prediction, name), key=len)
    if long_ in (short + "s", short + "es"):
        return True
    return short.endswith("y") and long_ == short[:-1] + "ies"


def or_disjuncts(question: dict) -> tuple[tuple[str, bool], ...]:
    """For an `or`/`and` existence question, each selected object and whether GQA marks it present."""
    ops = {s.get("operation", "") for s in question.get("semantic") or []}
    if not ({"or", "and"} & ops):
        return ()
    out = []
    for step in question.get("semantic") or []:
        if step.get("operation") != "select":
            continue
        parsed = parse_select_argument(step.get("argument", ""))
        if parsed is not None:
            out.append((parsed[0], parsed[1] is not None))
    return tuple(out)


def _label(source: str, error: str, correct: bool | None, schema_expected: str,
           schema_correct: bool, *, resolvable: bool, exact: bool = False) -> Label:
    return Label(
        exact_correct=exact,
        schema_expected=schema_expected,
        schema_correct=schema_correct,
        semantic_correct=correct,
        judgment_source=source,
        error_type=error,
        confidence=_CONFIDENCE[source],
        abstain=correct is None,
        resolvable=resolvable,
    )


def evaluate(question: dict, prediction: str, inventory: dict, ontology: dict) -> Label:
    """Run the Level 1-3 cascade. The first rule that fires wins; the source is always recorded."""
    gold = question.get("answer", "")
    schema = expected_schema(question)
    schema_ok = matches_schema(prediction, question, schema)

    # ---- Level 1: the locked metric. Never re-examined. -------------------------------------
    if exact_full(prediction, gold):
        return _label("exact", "", True, schema, schema_ok, resolvable=True, exact=True)

    p = normalize_full(prediction)
    g = normalize_full(gold)
    if not p:
        return _label("program", "wrong_answer_type", False, schema, schema_ok, resolvable=True)

    # ---- Level 3.1: numbers. Retained for other surfaces; Test-Dev balanced has none. --------
    if same_number(p, g):
        return _label("program", "correct_content_wrong_format", True, schema, schema_ok,
                      resolvable=True)

    # ---- Level 3.2: polarity, for questions whose gold is Boolean ---------------------------
    if g in _BOOLEAN_WORDS:
        polarity = [t for t in p.split() if t in _BOOLEAN_WORDS]
        if len(set(polarity)) > 1:
            # Carries both "yes" and "no": the answer's own polarity is undetermined.
            return _label("program", "unresolved", None, schema, schema_ok, resolvable=False)
        if len(polarity) == 1:
            return _label("program", "wrong_polarity", False, schema, schema_ok, resolvable=True)

        # ---- Level 3.3: `or`/`and` existence, adjudicated by GQA's own presence markers ------
        disjuncts = or_disjuncts(question)
        if disjuncts:
            n_present = sum(1 for _, present in disjuncts if present)
            filtered = _is_filtered_existence(question)
            # On a FILTERED question the marker describes the BASE object, not the qualified one:
            # "black calculator or chair" marks that a calculator and a chair exist, not that
            # either is black. Marker truth agrees with the gold on 208/208 unfiltered questions
            # and only 91/256 filtered ones. Gold `yes` with exactly one base present is still
            # decidable — the other base does not exist, so the qualified object must be the one
            # that does. Gold `yes` with several bases present is not.
            undecidable = filtered and g == "yes" and n_present != 1
            named = [(name, present) for name, present in disjuncts if _matches_disjunct(p, name)]
            if p in {"both", "neither"}:
                if undecidable or (filtered and p == "both" and g == "yes"):
                    return _label("program", "unresolved", None, schema, schema_ok,
                                  resolvable=False)
                right = (p == "neither" and n_present == 0) or \
                        (p == "both" and n_present == len(disjuncts))
                return _label("program",
                              "correct_content_wrong_format" if right else "wrong_answer_type",
                              right, schema, schema_ok, resolvable=True)
            if named:
                if undecidable:
                    return _label("program", "unresolved", None, schema, schema_ok,
                                  resolvable=False)
                present = any(present for _, present in named)
                # Naming the present disjunct answers "yes" in content; naming the absent one is a
                # claim GQA contradicts. Gold `no` means no disjunct satisfies the question, so
                # naming either base object is a false existence claim whether filtered or not.
                right = present and g == "yes"
                return _label("program",
                              "correct_content_wrong_format" if right else "wrong_object",
                              right, schema, schema_ok, resolvable=True)

    # ---- Level 3.4: morphology, anchored on the GOLD ----------------------------------------
    if _matches_disjunct(p, g):
        return _label("program", "correct_content_wrong_format", True, schema, schema_ok,
                      resolvable=True)

    # ---- Level 3.5: the frozen synonym ontology ---------------------------------------------
    if g in ontology and p in ontology.get(g, frozenset()):
        return _label("ontology", "synonym_or_paraphrase", True, schema, schema_ok, resolvable=True)

    # ---- Level 3.6: wrong type, before any content claim ------------------------------------
    if not schema_ok:
        return _label("program", "wrong_answer_type", False, schema, schema_ok, resolvable=True)

    # ---- Level 3.7: a wrong SUPPLIED alternative is a content error, not a binding failure ---
    # This must precede binding. A `choose` distractor is usually also an object in the image, so
    # a generic inventory test would relabel 66 wrong supplied choices as relation binding.
    if schema == ALTERNATIVE and p in set(alternatives_of(question)):
        return _label("program", "wrong_attribute", False, schema, schema_ok, resolvable=True)

    # ---- Level 3.8: naming the relation's own SOURCE object ---------------------------------
    # Only where the question asks WHICH OBJECT. On an attribute query the same string is not a
    # reversal: "Which material is the laptop near the glass made of?" (qid 20783128) selects
    # `glass (5)` and asks for a MATERIAL, so the answer "glass" is a wrong material, not a
    # confusion of the two referents, and nothing in the program separates the two readings.
    if schema == OBJECT_NAME:
        source = _relate_source(question)
        if source and _matches_disjunct(p, source):
            return _label("program", "reversed_relation", False, schema, schema_ok,
                          resolvable=True)

    # ---- Level 3.9: binding, and only where object identity is what the question asks for ---
    # Requires two NON-EMPTY, DISJOINT object-id sets. Names are not enough: GQA calls one object
    # `snowboarder` and `person`, so a name-only test turned annotation aliases into wrong
    # referents. Same-id cases are annotation-limited and abstain; an unknown gold id proves
    # nothing either way and abstains.
    if schema == OBJECT_NAME:
        ids = inventory.get(question.get("imageId", ""), {}).get("ids", {})
        gold_ids, pred_ids = ids.get(g, frozenset()), ids.get(p, frozenset())
        if gold_ids and pred_ids:
            if gold_ids & pred_ids:
                return _label("program", "ambiguous_or_annotation_limited", None, schema,
                              schema_ok, resolvable=False)
            return _label("program", "wrong_referent_binding", False, schema, schema_ok,
                          resolvable=True)

    # ---- Otherwise: not decidable without the image. Abstain. -------------------------------
    return _label("program", "unresolved", None, schema, schema_ok, resolvable=False)


def _is_filtered_existence(question: dict) -> bool:
    """True if the existence test is over a FILTERED object, so the marker is not its truth."""
    return any(str(s.get("operation", "")).startswith("filter")
               for s in question.get("semantic") or [])


def _relate_source(question: dict) -> str:
    """The object a `relate` step starts from — answering with it reverses the relation."""
    steps = question.get("semantic") or []
    if not any(str(s.get("operation", "")).startswith("relate") for s in steps):
        return ""
    for step in steps:
        if step.get("operation") == "select":
            parsed = parse_select_argument(step.get("argument", ""))
            if parsed is not None:
                return parsed[0]
    return ""


# WITHDRAWN IN 1.1.0 — the fullAnswer containment rule, and why it cannot be repaired.
#
# The original rule 4 credited any prediction that was a contiguous token span of `fullAnswer` and of
# the right schema type. Implemented faithfully, it rescued 650 cells, and inspection showed many
# were spurious: gold `plastic` / prediction `coffee` was credited from "The coffee is made of
# plastic.", gold `street sign` / prediction `street`, gold `material` / prediction `metal`. The
# rule credits the question's own subject and partial head nouns.
#
# It cannot be anchored on GQA's annotations either. In "Which kind of clothing is pink?" the
# `fullAnswer` annotation is {"1": "4", "4:6": "4"} — the subject "clothing" and the answer
# "tank top" carry the SAME object id, so the annotation cannot separate the answer span from the
# span the question already named.
#
# Level 3.4 is therefore gold-anchored morphology, and `full_answer` remains in JUDGMENT_SOURCES
# (the schema requires the value) while never being emitted. A hypernym or partial head noun is a
# different answer from the one GQA asked for and is left to the ontology or to `unresolved`.
