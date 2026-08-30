"""The canonical visible fact surface: exactly what `SceneGraphStore.describe()` renders.

WHY THIS IS ITS OWN MODULE. Every Priority-3 arm intervenes on "the facts the model actually
sees", so a structured view of those facts is the unit of the whole experiment. When that view was
inlined in a build script it silently disagreed with the renderer: it accumulated the endpoints of
every relation of the final subject and only then checked `len(facts) >= 40`, while `describe()`
returns `facts[:40]`. The IDs belonging to facts 41+ were therefore counted as visible when they
are never shown. That inflated the frozen ontology's population by 331 IDs across 206 questions
(46,835 counted versus 46,504 rendered) and every coverage number derived from it.

WHAT THE RENDERER CAN AND CANNOT CERTIFY. `assert_renderer_equivalence()` proves that the
structured facts re-render to `describe()` **byte for byte**, in order, with the same count and
within the cap. It cannot prove the object IDs are right, because the rendered text contains no
IDs: a review probe replaced fact 0's `subj_id` with a deliberately wrong value, left its text alone, and
the assertion accepted it. IDs are what G4 and G7 select on, so they are checked separately by
`assert_object_identity()`, which rebuilds each fact's `(kind, subj_id, obj_id, predicate)` tuple
straight from the raw graph. Both run over all 3,000 slice images in
`tests/test_coarse_labels.py::VisibleSurfaceTests`.
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_FACTS = 40


@dataclass(frozen=True)
class VisibleFact:
    """One rendered fact. `kind` is 'relation' or 'standalone'; `obj_id` is None for standalone."""
    index: int
    kind: str
    subj_id: str
    obj_id: str | None
    predicate: str | None
    text: str


def visible_facts(store, image_id) -> list[VisibleFact]:
    """The ordered facts `describe()` renders for this image, truncated exactly as it truncates.

    Mirrors `SceneGraphStore.describe()` step for step, then applies the same `[:MAX_FACTS]` slice.
    The slice is the part that must not be forgotten: the accumulation loop can overshoot 40 by up
    to the number of relations on the final subject.
    """
    graph = store._graphs.get(str(image_id))
    if not graph:
        return []
    objects = graph.get("objects", {})
    facts: list[VisibleFact] = []
    related: set[str] = set()

    for oid, obj in objects.items():
        subj = store._phrase(objects, oid)
        if not subj:
            continue
        for rel in obj.get("relations", []):
            tgt = store._phrase(objects, rel.get("object"))
            if tgt and rel.get("name"):
                facts.append(VisibleFact(len(facts), "relation", str(oid),
                                         str(rel.get("object")), rel["name"],
                                         f"a {subj} {rel['name']} a {tgt}"))
                related.add(str(oid))
                related.add(str(rel.get("object")))
        if len(facts) >= MAX_FACTS:
            break

    for oid, obj in objects.items():
        if str(oid) not in related and len(facts) < MAX_FACTS:
            phrase = store._phrase(objects, oid)
            if phrase:
                facts.append(VisibleFact(len(facts), "standalone", str(oid), None, None,
                                         f"a {phrase}"))

    return facts[:MAX_FACTS]


def visible_object_ids(store, image_id) -> list[str]:
    """Stable visible object IDs, in canonical first-appearance order over the rendered facts."""
    seen: set[str] = set()
    out: list[str] = []
    for fact in visible_facts(store, image_id):
        for oid in (fact.subj_id, fact.obj_id):
            if oid and oid not in seen:
                seen.add(oid)
                out.append(oid)
    return out


def render(facts: list[VisibleFact]) -> str:
    """Re-render structured facts to the string form; must equal `describe()` byte for byte."""
    return "; ".join(f.text for f in facts)


def expected_identity_sequence(store, image_id) -> list[tuple]:
    """Rebuild the whole ordered `(index, kind, subj_id, obj_id, predicate)` sequence from the graph.

    Deliberately a second implementation of the traversal, not a reuse of `visible_facts()`. A check
    that validates each emitted fact against the graph only proves each fact is *possible*; it
    cannot notice that fact 12 names a real object which is nevertheless the wrong one, or that two
    facts were transposed. Only comparing the full expected sequence positionally does that.
    """
    graph = store._graphs.get(str(image_id))
    if not graph:
        return []
    objects = graph.get("objects", {})
    expected: list[tuple] = []
    related: set[str] = set()

    for oid, obj in objects.items():
        if not store._phrase(objects, oid):
            continue
        for rel in obj.get("relations", []):
            if store._phrase(objects, rel.get("object")) and rel.get("name"):
                expected.append((len(expected), "relation", str(oid),
                                 str(rel.get("object")), rel["name"]))
                related.add(str(oid))
                related.add(str(rel.get("object")))
        if len(expected) >= MAX_FACTS:
            break

    for oid, obj in objects.items():
        if str(oid) not in related and len(expected) < MAX_FACTS:
            if store._phrase(objects, oid):
                expected.append((len(expected), "standalone", str(oid), None, None))

    return expected[:MAX_FACTS]


def assert_object_identity(store, image_id) -> None:
    """Fail if the ordered identity sequence differs from an independent reconstruction.

    This is the check `assert_renderer_equivalence()` cannot make: the rendered string carries no
    object IDs, so a fact with correct text and a wrong `subj_id` passes it.

    The v1.5 version of this function checked each fact *locally* -- `subj_id in objects`, and for
    relations that the (obj_id, predicate) pair existed somewhere on that subject. a second review probe defeated
    it by replacing a standalone fact's ID with a **different, valid ID from the same graph**: the
    ID existed, so the membership test passed. Positional comparison against the full expected
    sequence is what closes that, and it also catches transpositions and off-by-one truncation.
    """
    actual = [(f.index, f.kind, f.subj_id, f.obj_id, f.predicate)
              for f in visible_facts(store, image_id)]
    expected = expected_identity_sequence(store, image_id)
    if actual != expected:
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a != e:
                raise AssertionError(
                    f"image {image_id}: identity sequence diverges at position {i}\n"
                    f"  actual   -> {a}\n  expected -> {e}")
        raise AssertionError(
            f"image {image_id}: identity sequence has {len(actual)} facts, expected "
            f"{len(expected)}")


def assert_renderer_equivalence(store, image_id) -> None:
    """Fail if the structured facts do not re-render to `describe()` byte for byte.

    Checked both ways on purpose: a one-way check ("every structured fact appears in the text")
    would have passed the beyond-cap defect, because the extra IDs came from facts that were
    structurally present and textually absent.

    This certifies TEXT ONLY. The rendered string carries no object IDs, so a fact with the right
    text and a wrong `subj_id` passes here — see `assert_object_identity()`.
    """
    facts = visible_facts(store, image_id)
    rebuilt = render(facts)
    rendered = store.describe(image_id)
    if rebuilt != rendered:
        raise AssertionError(
            f"image {image_id}: structured facts do not re-render to describe() output\n"
            f"  structured -> {rebuilt[:200]!r}\n"
            f"  describe() -> {rendered[:200]!r}")

    text_facts = [f for f in rendered.split("; ") if f.strip()]
    if len(text_facts) != len(facts):
        raise AssertionError(
            f"image {image_id}: {len(facts)} structured facts vs {len(text_facts)} rendered")
    if len(facts) > MAX_FACTS:
        raise AssertionError(f"image {image_id}: {len(facts)} facts exceeds the cap {MAX_FACTS}")
