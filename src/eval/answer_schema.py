"""Level 2 — what KIND of answer each GQA question demands.

WHY THIS EXISTS. Separating "wrong type of answer" from "wrong answer" is what lets an audit say
whether a model failed to see something or failed to answer in the requested form. "couch" to
"Do you see a mirror or a couch?" is a format failure with correct content; "mirror" to the same
question, when the mirror is absent, is a content failure. Level 1 scores both simply wrong.

TWO RULES THAT LOOK LIKE DETAILS AND ARE NOT.

1. `choose` alternatives come from the question PROGRAM, never from the question text. GQA contains
   questions worded "cement or aluminum" whose program says `aluminum|concrete`; reading the
   wording would invent an alternative the annotation does not contain and would let a model be
   credited for a word GQA never offered.
2. The gold answer is admissible by definition. On Test-Dev only 915 of 1,129 `choose` golds appear
   in their own program's alternative list, so an alternatives-only test would score 214 correct
   answers as schema violations.
"""
from __future__ import annotations

from src.eval.metrics import normalize_full

SCHEMA_VERSION = "gqa_answer_schema/1.0.0"

BOOLEAN = "boolean"
NUMBER = "number"
ALTERNATIVE = "alternative"
COMPARISON = "comparison"
OBJECT_NAME = "object_name"
ATTRIBUTE = "attribute"
OTHER = "other"

_BOOLEAN_WORDS = {"yes", "no"}


def _last_operation(question: dict) -> tuple[str, str]:
    """The final program step as (operation, argument); ('', '') if the program is empty."""
    steps = question.get("semantic") or []
    if not steps:
        return "", ""
    return steps[-1].get("operation", ""), steps[-1].get("argument", "")


def alternatives_of(question: dict) -> tuple[str, ...]:
    """The alternatives the PROGRAM offers, lowercased; empty when the program supplies none."""
    op, arg = _last_operation(question)
    if not op.startswith("choose") or "|" not in arg:
        return ()
    return tuple(part.strip().lower() for part in arg.split("|") if part.strip())


def expected_schema(question: dict) -> str:
    """Classify the answer type this question demands, from its type tags and program."""
    structural = question.get("types", {}).get("structural", "")
    op, _ = _last_operation(question)

    if structural in ("verify", "logical"):
        return BOOLEAN
    if structural == "choose":
        return ALTERNATIVE
    if structural == "compare":
        if op.startswith("common"):
            return COMPARISON
        if op.startswith("choose"):
            # "Which is bigger, the racket or the wristband?" — the two candidates live only in
            # the wording, which this module may not parse, so the answer is an object name.
            return OBJECT_NAME
        return BOOLEAN
    if structural == "query":
        return OBJECT_NAME if op == "query" and _query_argument(question) == "name" else ATTRIBUTE
    return OTHER


def _query_argument(question: dict) -> str:
    for step in reversed(question.get("semantic") or []):
        if step.get("operation") == "query":
            return step.get("argument", "").strip().lower()
    return ""


def matches_schema(prediction: str, question: dict, expected: str) -> bool:
    """True if the prediction is of the demanded answer type (not whether it is correct)."""
    p = normalize_full(prediction)
    if not p:
        return False
    tokens = set(p.split())
    if expected == BOOLEAN:
        # Exactly one polarity token and nothing else: "yes no" is not a Boolean answer.
        return p in _BOOLEAN_WORDS
    if expected == ALTERNATIVE:
        admissible = set(alternatives_of(question)) | {normalize_full(question.get("answer", ""))}
        return p in admissible
    if expected == NUMBER:
        return _as_number(p) is not None
    # object_name / attribute / comparison / other: any non-Boolean content word is type-plausible.
    return not (tokens & _BOOLEAN_WORDS)


_WORD_NUMBERS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                 "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def _as_number(text: str) -> int | None:
    t = normalize_full(text)
    if t.isdigit():
        return int(t)
    return _WORD_NUMBERS.get(t)
