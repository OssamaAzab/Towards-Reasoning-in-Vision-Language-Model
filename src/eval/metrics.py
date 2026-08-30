"""Evaluation metrics for GQA short-answer VQA.

GQA scores by exact match against a short gold answer. A generative model produces
free-form text, so predictions are normalised (lowercase, take the first line,
strip punctuation and articles, collapse whitespace) before comparison. A relaxed
match (the gold answer appears as a contiguous token-substring of the prediction)
is also provided, to bracket performance and quantify how much the strict metric is
affected by verbosity. Exact match is the headline number; relaxed is a diagnostic.
"""
from __future__ import annotations

import string

_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, take the first line, strip punctuation and articles, collapse spaces."""
    text = text.strip().lower().split("\n")[0]
    text = text.translate(_PUNCT)
    tokens = [t for t in text.split() if t not in _ARTICLES]
    return " ".join(tokens)


def exact_match(pred: str, gold: str) -> bool:
    """True if the normalized prediction equals the normalized gold answer."""
    return normalize_answer(pred) == normalize_answer(gold)


def relaxed_match(pred: str, gold: str) -> bool:
    """True if the normalized gold answer appears as a contiguous run of tokens in pred."""
    p_tokens = normalize_answer(pred).split()
    g_tokens = normalize_answer(gold).split()
    if not g_tokens:
        return False
    span = len(g_tokens)
    return any(p_tokens[i:i + span] == g_tokens
               for i in range(0, max(0, len(p_tokens) - span + 1)))


METRIC_VERSION = "exact_full/1.0.0"


def normalize_full(text: str) -> str:
    """Normalize WITHOUT truncating at the first line or at punctuation.

    Identical to normalize_answer except that it keeps the whole prediction. The
    first-line rule in normalize_answer is an answer-EXTRACTION step disguised as
    normalisation: measured on the Phase-2 records it is worth +13.6 points to the
    multiline text-only floor and only +0.2 to +6.2 to the bridge, so it silently
    favours whichever arm happens to answer in multiple lines.
    """
    text = text.strip().lower().translate(_PUNCT)
    return " ".join(t for t in text.split() if t not in _ARTICLES)


def exact_full(pred: str, gold: str) -> bool:
    """Primary corrected metric: normalized exact match over the complete answer.

    The same normalisation is applied to both arms — symmetric by construction,
    because the asymmetry is exactly the defect this replaces.
    """
    return normalize_full(pred) == normalize_full(gold)


_YES_NO = {"yes", "no"}


def vqa_match(pred: str, gold: str) -> bool:
    """Fair soft match for a generative model scored against short gold answers.

    - exact after normalisation always counts;
    - for yes/no golds, the prediction's polarity is its FIRST yes/no token, so a
      verbose "Yes, there are ..." matches "yes" but a "Yes ... no door" cannot
      wrongly match "no" (avoids over-crediting);
    - otherwise the gold must appear as a contiguous token-span in the prediction,
      which credits a correct answer embedded in a longer sentence.
    """
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not g:
        return False
    if p == g:
        return True
    if g in _YES_NO:
        for tok in p.split():
            if tok in _YES_NO:
                return tok == g
        return False
    pt, gt = p.split(), g.split()
    span = len(gt)
    return any(pt[i:i + span] == gt for i in range(0, max(0, len(pt) - span + 1)))
