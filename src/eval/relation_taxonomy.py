"""Versioned mapping from GQA relation predicates to relation families.

WHY A FILE AND A VERSION. The relate taxonomy decides which questions count as "directional", and
that is the axis the encoder error analysis turns on. If it drifted between runs the comparison
would drift with it, so it lives here with a version string, and the analysis records both.

WHERE THE PREDICATES COME FROM. GQA question programs, not scene graphs. A `relate` operation
carries an argument of the form `<class>,<predicate>,<role>`, where role is `s` when the answer
plays the SUBJECT of the relation and `o` when it plays the OBJECT. That is available on Test-Dev,
which publishes no scene graphs, so the taxonomy works on the external benchmark as well as on the
annotation-rich internal splits.

THE FAMILIES, and why these cuts:

  projective      Direction-dependent. Reversing subject and object changes the truth value:
                  "left of" is not "right of". This is the family that tests whether an encoder
                  represents spatial direction rather than mere co-occurrence.
  topological     Containment, support and proximity. Largely symmetric or direction-insensitive
                  in practice: "near" reversed is still "near", and "on" reversed is rare enough
                  to be unhelpful as a directional probe.
  posture_support A deliberate third family, not folded into either of the above. "sitting on"
                  is simultaneously a body posture and a support relation, so counting it as
                  topological would dilute the topological family with action semantics, and
                  counting it as an interaction would dilute that family with spatial support.
  interaction     Human-object and object-object actions: wearing, holding, riding, watching.
                  Recognition-driven rather than geometry-driven.
  comparative     `same color`, `same material` and friends. These compare two selected objects
                  and are structurally unlike the rest.
  possessive      `of`, `with` — weak, near-syntactic links.
  other           Everything unmapped. Coverage is reported; it is never silently absorbed.

A predicate is mapped by exact match first, then by prefix rule. Unmapped predicates fall to
`other` and are counted, so a long tail can never masquerade as a resolved category.
"""
from __future__ import annotations

TAXONOMY_VERSION = "gqa_relation_taxonomy/1.0.0"

PROJECTIVE = {
    "to the left of", "to the right of", "in front of", "behind", "above", "below",
    "under", "underneath", "on top of", "beneath", "over", "on the side of",
    "standing behind", "sitting behind", "walking behind", "standing in front of",
    "sitting in front of", "walking in front of", "to the left", "to the right",
    "in the front of", "at the top of", "at the bottom of", "on the front of",
    "on the back of", "on the top of", "on the bottom of", "in the back of",
}

TOPOLOGICAL = {
    "on", "in", "inside", "near", "next to", "beside", "by", "at", "around",
    "contain", "containing", "contains", "filled with", "full of", "covered by",
    "covering", "covered in", "covered with", "touching", "against", "leaning on",
    "leaning against", "hanging from", "hanging on", "attached to", "resting on",
    "surrounding", "surrounded by", "connected to", "attached", "along", "across",
    "between", "among", "inside of", "close to", "adjacent to", "beside of",
}

POSTURE_SUPPORT = {
    "sitting on", "standing on", "lying on", "walking on", "sitting in", "standing in",
    "lying in", "walking in", "sitting at", "standing at", "sitting on top of",
    "lying on top of", "standing next to", "sitting next to", "kneeling on",
    "perched on", "parked on", "parked in", "growing on", "growing in", "floating on",
    "swimming in", "standing by", "sitting by", "laying on", "resting in", "seated on",
}

INTERACTION = {
    "wearing", "holding", "carrying", "riding", "riding on", "eating", "drinking",
    "drinking from", "using", "playing", "playing with", "looking at", "watching",
    "pulling", "pulled by", "pushing", "pushed by", "hitting", "throwing", "catching",
    "feeding", "petting", "reading", "driving", "driven by", "wears", "worn by",
    "held by", "carried by", "ridden by", "eaten by", "used by", "played by",
    "talking on", "typing on", "cutting", "cooking", "cleaning", "brushing",
    "flying", "flying in", "chasing", "following", "touching with", "grabbing",
    "reaching for", "pointing at", "waiting for", "helping", "wearing a",
}

COMPARATIVE_PREFIXES = ("same ", "different ")
COMPARATIVE_MARKERS = ("bigger than", "smaller than", "larger than", "taller than",
                       "shorter than", "wider than", "narrower than", "same", "different")
POSSESSIVE = {"of", "with", "for", "from", "'s", "has", "have"}

# Direction-carrying tokens. The exact sets above are authoritative, but the GQA predicate tail is
# long and compositional -- "hanging above", "flying above", "sitting atop", "looking down at" --
# and a predicate that carries direction belongs in `projective` regardless of the verb attached
# to it, because direction is the property the analysis is testing for.
PROJECTIVE_MARKERS = (
    "left", "right", "front", "behind", "above", "below", "under", "beneath", "over",
    "atop", "top of", "bottom of", "higher than", "lower than", "beyond", "past",
    "up on", "down at", "down on", "underneath",
)
# Body-posture verbs. Applied only after the projective check, so "standing behind" is
# projective (its direction is the informative part) while "standing beside" is posture.
POSTURE_MARKERS = (
    "sitting", "seated", "standing", "lying", "laying", "walking", "kneeling", "perched",
    "parked", "growing", "floating", "swimming", "leaning", "resting", "crouching",
    "hanging", "mounted", "sleeping", "running", "riding on",
)
INTERACTION_MARKERS = (
    "holding", "wearing", "worn", "carrying", "eating", "drinking", "using", "playing",
    "looking", "watching", "pulling", "pushing", "hitting", "throwing", "catching",
    "feeding", "petting", "reading", "driving", "grabbing", "touching", "reaching",
    "pointing", "talking", "typing", "cutting", "cooking", "cleaning", "brushing",
    "chasing", "following", "helping", "waiting",
)

_EXACT = [(PROJECTIVE, "projective"), (TOPOLOGICAL, "topological"),
          (POSTURE_SUPPORT, "posture_support"), (INTERACTION, "interaction"),
          (POSSESSIVE, "possessive")]


def relation_family(predicate: str) -> str:
    """Map one GQA relate predicate to its family; unmapped predicates become 'other'.

    Exact membership wins; then token rules, ordered so that direction dominates. A predicate
    that survives all of them is 'other' and is counted as such -- coverage is reported, never
    quietly absorbed.
    """
    p = (predicate or "").strip().lower()
    if not p:
        return "other"
    for vocab, name in _EXACT:
        if p in vocab:
            return name
    if p.startswith(COMPARATIVE_PREFIXES) or any(m in p for m in COMPARATIVE_MARKERS):
        return "comparative"
    if any(m in p for m in PROJECTIVE_MARKERS):
        return "projective"
    if any(m in p for m in POSTURE_MARKERS):
        return "posture_support"
    if any(m in p for m in INTERACTION_MARKERS):
        return "interaction"
    if any(m in p for m in ("near", "next to", "beside", "inside", "around", "against",
                            "covered", "covering", "filled", "attached", "connected",
                            "surround", "between", "close to", "adjacent", "along")):
        return "topological"
    return "other"


def parse_relate_argument(argument: str) -> tuple[str, str, str]:
    """Split a GQA relate argument `<class>,<predicate>,<role> (id)` into its three parts.

    Returns ("", "", "") when the argument does not have the expected shape, so a malformed
    entry is dropped rather than silently mapped to a family.
    """
    head = (argument or "").split(" (")[0]
    parts = head.split(",")
    if len(parts) < 3:
        return "", "", ""
    obj_class, predicate, role = parts[0], ",".join(parts[1:-1]), parts[-1]
    return obj_class.strip(), predicate.strip(), role.strip()


def question_relations(semantic: list[dict]) -> list[dict]:
    """Every relate step in a GQA program, as {class, predicate, role, family}."""
    out = []
    for op in semantic or []:
        if not str(op.get("operation", "")).startswith("relate"):
            continue
        cls, pred, role = parse_relate_argument(op.get("argument", ""))
        if not pred:
            continue
        out.append({"object_class": cls, "predicate": pred, "role": role,
                    "family": relation_family(pred)})
    return out
