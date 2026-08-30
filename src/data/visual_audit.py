"""Masks and probe labels for the visual-information audit, built from GQA's own annotations.

WHERE "RELEVANT" COMES FROM. GQA ships, per question, an `annotations` map from question-token
position to the scene-graph object id that token refers to, plus a `semantic` program whose
arguments carry object ids in `name (id)` form. Those ids are the question's OWN referents. This
module reads them and never infers relevance from word overlap, box size or salience — an
inferred referent would make condition 5 ("relevant-object masked") a guess dressed as a control.

WHAT IS EXCLUDED, AND WHY EXCLUSION BEATS GUESSING. A question is usable for the masking
conditions only when it has at least one referent that exists in the scene graph with a
non-degenerate box, AND the same image offers a non-referent object whose box area matches
within tolerance. Without the second, condition 6 has no matched control and any difference
between 5 and 6 would confound relevance with region size. Questions failing either test are
reported and dropped, never silently repaired.

PROBE LABELS ARE DECODABILITY LABELS. `object identity`, `colour`, `count` and `spatial relation`
are read from the scene graph for the image. A probe that recovers them from frozen features
shows the information is DECODABLE at that point in the pipeline. It does not show Qwen2 uses it.
"""
from __future__ import annotations

import re
from collections import Counter

# GQA writes programme arguments as "birds (413787)" or "to the left of (1234,5678)".
_ARG_IDS = re.compile(r"\((\d+(?:,\d+)*)\)")

# Colour words GQA uses as attributes. Restricted to colours so the colour probe is a colour
# probe: GQA attributes also cover material, pose, activity and size, and pooling them would
# make "colour/attribute" a label set no single probe could be said to have learned.
# "dark" and "light" are deliberately absent — they are brightness qualifiers that co-occur with
# a real colour ("dark blue"), so admitting them would put the same object in two classes.
COLOURS = {"black", "white", "gray", "grey", "red", "green", "blue", "yellow", "orange",
           "brown", "pink", "purple", "tan", "gold", "silver", "beige", "cream", "maroon",
           "navy", "teal", "turquoise", "violet"}

# Relations kept for the spatial-relation probe. GQA's relation vocabulary mixes spatial
# predicates with possessive and semantic ones ("of", "wearing"); only spatial ones bear on the
# question this audit asks.
SPATIAL_RELATIONS = {"to the left of", "to the right of", "above", "below", "on top of",
                     "under", "in front of", "behind", "near", "next to", "on", "in",
                     "inside", "outside"}

MIN_BOX_PX = 8           # a box smaller than this on a side cannot be masked meaningfully
AREA_TOL = 0.25          # matched control box area must be within +/-25% of the relevant one


def shuffle_patch_positions(features, generator, has_cls: bool = True):
    """Permute the PATCH axis of encoder features, leaving any leading CLS token in place.

    Condition 4 measures **sensitivity to the ORDERING of already position-encoded patch tokens**.
    It does NOT destroy spatial information and must never be described as doing so: CLIP's ViT
    adds positional embeddings *before* its transformer, so each output patch feature already
    carries its own position in its values. Permuting the sequence reorders vectors that remain
    individually position-tagged. A Q-Former reads them by cross-attention, which is permutation-
    invariant over its inputs, so for that connector this intervention is close to a no-op by
    construction. A null here is evidence about token ordering only.

    The CLS carve-out is not cosmetic. `drop_cls=True` makes the bridge slice index 0 *after* it
    receives this tensor (src/models/connectors.py), so permuting the whole axis would also
    change WHICH token is discarded — the arm would then confound shuffled positions with a
    dropped random patch, and a drop in accuracy could not be attributed to either.
    """
    import torch
    if features.dim() != 3:
        raise ValueError(f"expected [B, tokens, dim], got {tuple(features.shape)}")
    start = 1 if has_cls else 0
    n = features.size(1) - start
    if n < 2:
        return features.clone()
    perm = torch.randperm(n, generator=generator, device="cpu").to(features.device) + start
    out = features.clone()
    out[:, start:, :] = features[:, perm, :]
    return out


def mask_image_region(image, box, fill=(127, 127, 127)):
    """Return a copy of the PIL image with `box` filled flat, destroying that region's content.

    A flat mid-grey fill is used rather than a blur or a crop: blurring leaves recoverable
    low-frequency structure, and cropping changes the image geometry, which would alter every
    patch rather than the intended region.
    """
    from PIL import ImageDraw
    out = image.copy()
    ImageDraw.Draw(out).rectangle([int(v) for v in box], fill=tuple(fill))
    return out


def referenced_object_ids(question: dict) -> set[str]:
    """Scene-graph object ids this question refers to, from GQA's annotations and programme.

    Both sources are used because neither is complete on its own: `annotations` covers tokens
    that survived into the surface question, while the `semantic` programme carries referents
    introduced by relate/filter steps that the surface form elides.
    """
    ids: set[str] = set()
    ann = question.get("annotations") or {}
    for field in ("question", "answer", "fullAnswer"):
        for v in (ann.get(field) or {}).values():
            ids.update(str(v).split(","))
    for step in question.get("semantic") or []:
        for m in _ARG_IDS.finditer(str(step.get("argument", ""))):
            ids.update(m.group(1).split(","))
    return {i for i in ids if i.isdigit()}


def box_of(obj: dict) -> tuple[int, int, int, int]:
    """(x1, y1, x2, y2) for a GQA scene-graph object."""
    return int(obj["x"]), int(obj["y"]), int(obj["x"]) + int(obj["w"]), int(obj["y"]) + int(obj["h"])


def usable_box(obj: dict, img_w: int, img_h: int) -> bool:
    """A box worth masking: on-image, non-degenerate, and not effectively the whole picture."""
    x1, y1, x2, y2 = box_of(obj)
    if x2 - x1 < MIN_BOX_PX or y2 - y1 < MIN_BOX_PX:
        return False
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
        return False
    # Masking ~the whole image is not an object mask; it is condition 3 by another route.
    return (x2 - x1) * (y2 - y1) <= 0.5 * img_w * img_h


def mask_pair(question: dict, graph: dict) -> dict | None:
    """The (relevant, matched-irrelevant) box pair for one question, or None with a reason.

    Returns {"relevant": [...], "irrelevant": [...], "relevant_ids": [...], ...} or
    {"reject": "<reason>"} so the caller can report coverage by cause instead of a bare count.
    """
    objects = graph.get("objects") or {}
    img_w, img_h = int(graph.get("width", 0)), int(graph.get("height", 0))
    if not objects or img_w <= 0 or img_h <= 0:
        return {"reject": "image has no scene graph or no dimensions"}

    refs = referenced_object_ids(question)
    if not refs:
        return {"reject": "question annotations name no scene-graph object"}

    rel = [(i, objects[i]) for i in sorted(refs)
           if i in objects and usable_box(objects[i], img_w, img_h)]
    if not rel:
        return {"reject": "no referenced object has a usable box"}

    # The largest referent is masked: it is the one whose removal is most likely to matter, so a
    # null under condition 5 is not explained away by having masked a negligible region.
    rel_id, rel_obj = max(rel, key=lambda kv: (kv[1]["w"] * kv[1]["h"], kv[0]))
    rel_area = int(rel_obj["w"]) * int(rel_obj["h"])

    cands = []
    for oid, o in objects.items():
        if oid in refs or not usable_box(o, img_w, img_h):
            continue
        area = int(o["w"]) * int(o["h"])
        if abs(area - rel_area) <= AREA_TOL * rel_area:
            cands.append((abs(area - rel_area), oid, o))
    if not cands:
        return {"reject": f"no non-referenced object within {int(AREA_TOL * 100)}% of the "
                          f"relevant box area"}
    _, irr_id, irr_obj = min(cands, key=lambda t: (t[0], t[1]))

    return {
        "relevant_id": rel_id, "relevant_name": rel_obj.get("name"),
        "relevant_box": list(box_of(rel_obj)), "relevant_area": rel_area,
        "irrelevant_id": irr_id, "irrelevant_name": irr_obj.get("name"),
        "irrelevant_box": list(box_of(irr_obj)),
        "irrelevant_area": int(irr_obj["w"]) * int(irr_obj["h"]),
        "n_referenced": len(refs), "n_objects": len(objects),
        "area_ratio": (int(irr_obj["w"]) * int(irr_obj["h"])) / rel_area,
    }


def relative_position(graph: dict) -> str | None:
    """4-way relative position of the two largest objects, from box centroids.

    This is the PRIMARY spatial probe. GQA's own relation names collapse to near-binary on this
    slice — 'to the left of' and 'to the right of' are 94.3% of them and the majority class alone
    scores 55.2% — so a probe on them would be reporting a coin flip with extra steps. Relative
    position of the two largest objects is 4-way, is derived from boxes rather than from a
    relation vocabulary, and is precisely the information that pooling 256 patches to 32 tokens
    is most likely to destroy. Image y grows downward, so a larger y means lower in the frame.
    """
    objs = [(int(o["w"]) * int(o["h"]), oid, o) for oid, o in (graph.get("objects") or {}).items()
            if int(o.get("w", 0)) > MIN_BOX_PX and int(o.get("h", 0)) > MIN_BOX_PX]
    if len(objs) < 2:
        return None
    objs.sort(key=lambda t: (-t[0], t[1]))
    a, b = objs[0][2], objs[1][2]
    acx, acy = int(a["x"]) + int(a["w"]) / 2, int(a["y"]) + int(a["h"]) / 2
    bcx, bcy = int(b["x"]) + int(b["w"]) / 2, int(b["y"]) + int(b["h"]) / 2
    dx, dy = bcx - acx, bcy - acy
    if abs(dx) >= abs(dy):
        return "left" if dx > 0 else "right"
    return "above" if dy > 0 else "below"


def probe_labels(graph: dict) -> dict:
    """Per-image probe targets read from the scene graph. Missing targets are omitted, not faked.

    object   — the most frequent object name present (identity)
    colour   — the most frequent colour attribute present
    count    — how many instances of the most frequent object name (capped bucket)
    position — PRIMARY spatial target: 4-way relative position of the two largest objects
    relation — SECONDARY: the most frequent GQA spatial relation name. Near-binary on this
               slice and reported only alongside its majority-class baseline.
    """
    objects = graph.get("objects") or {}
    if not objects:
        return {}
    names = Counter(o.get("name") for o in objects.values() if o.get("name"))
    colours = Counter(a for o in objects.values() for a in (o.get("attributes") or [])
                      if a in COLOURS)
    rels = Counter(r.get("name") for o in objects.values() for r in (o.get("relations") or [])
                   if r.get("name") in SPATIAL_RELATIONS)

    out: dict = {}
    if names:
        top, n = names.most_common(1)[0]
        out["object"] = top
        # Counting beyond a few instances is not something a linear probe on pooled features
        # can be expected to express, so the target is a bucket, and it is labelled as one.
        out["count"] = str(min(n, 5))
        out["count_raw"] = n
    if colours:
        out["colour"] = colours.most_common(1)[0][0]
    pos = relative_position(graph)
    if pos:
        out["position"] = pos
    if rels:
        out["relation"] = rels.most_common(1)[0][0]
    return out
