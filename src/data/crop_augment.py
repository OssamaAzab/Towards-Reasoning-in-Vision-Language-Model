"""Question-conditioned visual-evidence crops, built from GQA's own annotations.

STAGE A IS AN ORACLE FEASIBILITY TEST, NOT A DEPLOYABLE METHOD. The `relevant` crop is located
using the GQA scene-graph box of an object the question itself names. Nothing at inference time
knows that box. This measures the CEILING available to question-conditioned cropping if region
selection were solved; it is not a system anyone can run on an unlabelled image, and every
artifact this module produces is stamped `oracle=True` so it cannot be mistaken for one.

THE CROP RULE IS FIXED BEFORE ANY ACCURACY EXISTS, and is the same for every arm:

    annotated box -> expand by CROP_MARGIN on each side -> clip to the image
    -> pad to square with flat mid-grey -> the encoder's own preprocessing

Padding rather than stretching matters: the processor resizes to a square input, so a non-square
crop would be anisotropically distorted and a tall thin object would be encoded differently from
a wide flat one purely because of its shape. Padding preserves aspect ratio and makes the crop
geometry a property of the box, not of the resize.

WHY THREE CROP ARMS AND NOT ONE. A `relevant` crop adds a second image's worth of tokens to the
prompt, and the frozen bridge and LLM have only ever seen one image's worth. Beating the global
baseline is therefore NOT evidence that the crop's CONTENT helped — a longer visual span, or the
mere presence of a second focused view, would move the number too. `irrelevant` (same image, no
referenced object, area-matched) and `wrong` (a crop from a deranged partner image) carry exactly
the same token count and the same "a crop was appended" structure, and differ only in whether the
region is the one the question is about. The grounded claim requires relevant to beat those.
"""
from __future__ import annotations

from src.data.visual_audit import box_of, referenced_object_ids

# ---- the fixed crop rule. Changing any of these after seeing accuracy invalidates the arm. ----
CROP_MARGIN = 0.15        # expand the annotated box by 15% of its width/height on each side
MIN_CROP_PX = 24          # a crop smaller than this on a side carries no usable detail
MAX_CROP_AREA_FRAC = 0.6  # a "crop" covering most of the image is the global view by another name
AREA_TOL = 0.25           # the irrelevant crop's area must be within +/-25% of the relevant one
PAD_FILL = (127, 127, 127)

# The relevant and wrong crops are matched on target LABEL and only then area-MINIMISED, so within
# a small label class the closest available donor can still be many times larger or smaller. Every
# crop is padded square and resized to the encoder's fixed input, so a padded-area difference is a
# difference in the RESOLUTION at which the object is presented — a scale confound sitting inside
# the primary contrast. This threshold defines the pre-registered scale-matched subgroup and is
# fixed before any accuracy exists. Changing it after seeing a result would make the robustness
# check a function of the result.
SCALE_MATCH_FACTOR = 2.0


def expand_and_clip(box, img_w: int, img_h: int, margin: float = CROP_MARGIN):
    """Expand a box by `margin` on each side, then clip it to the image. Returns (x1,y1,x2,y2).

    Clipping is what makes the rule total: an object against the frame edge would otherwise
    produce a box with negative coordinates, and PIL would silently pad it with black, adding
    content that is not in the image.
    """
    x1, y1, x2, y2 = box
    dw, dh = (x2 - x1) * margin, (y2 - y1) * margin
    return (max(0, int(round(x1 - dw))), max(0, int(round(y1 - dh))),
            min(img_w, int(round(x2 + dw))), min(img_h, int(round(y2 + dh))))


def crop_is_usable(box, img_w: int, img_h: int) -> bool:
    """A crop worth encoding: on-image, big enough to see, and not effectively the whole picture."""
    x1, y1, x2, y2 = box
    if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or x2 <= x1 or y2 <= y1:
        return False
    if (x2 - x1) < MIN_CROP_PX or (y2 - y1) < MIN_CROP_PX:
        return False
    return (x2 - x1) * (y2 - y1) <= MAX_CROP_AREA_FRAC * img_w * img_h


def crop_area(box) -> int:
    """Pixel area of a box."""
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def pad_to_square(image, fill=PAD_FILL):
    """Centre a crop on a square mid-grey canvas, so the encoder's resize cannot distort it."""
    from PIL import Image
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), tuple(fill))
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def make_crop(image, box, margin: float = CROP_MARGIN, fill=PAD_FILL):
    """The full fixed rule: expand -> clip -> crop -> pad to square. Returns a PIL image."""
    b = expand_and_clip(box, image.width, image.height, margin)
    return pad_to_square(image.crop(b), fill)


def relevant_box(question: dict, graph: dict):
    """The box of the largest object this question names, or None.

    Read from GQA's `annotations` map and `semantic` programme via referenced_object_ids — the
    question's OWN referents. Never inferred from the answer, from word overlap or from salience:
    an inferred box would make the oracle arm a guess wearing an oracle's label.
    """
    objects = graph.get("objects") or {}
    img_w, img_h = int(graph.get("width", 0)), int(graph.get("height", 0))
    if not objects or img_w <= 0 or img_h <= 0:
        return None
    refs = referenced_object_ids(question)
    cand = []
    for oid in sorted(refs):
        o = objects.get(oid)
        if not o:
            continue
        b = expand_and_clip(box_of(o), img_w, img_h)
        if crop_is_usable(b, img_w, img_h):
            cand.append((crop_area(b), oid, b))
    if not cand:
        return None
    area, oid, b = max(cand, key=lambda t: (t[0], t[1]))
    return {"object_id": oid, "name": objects[oid].get("name"), "box": list(b), "area": area,
            "raw_box": list(box_of(objects[oid]))}


def irrelevant_box(question: dict, graph: dict, target_area: int):
    """An area-matched crop from the SAME image that contains no object the question names.

    Rejecting rather than substituting is deliberate. Without an area match, a difference between
    the relevant and irrelevant arms would confound *which region* with *how much region*.
    """
    objects = graph.get("objects") or {}
    img_w, img_h = int(graph.get("width", 0)), int(graph.get("height", 0))
    refs = referenced_object_ids(question)
    ref_boxes = [box_of(objects[o]) for o in refs if o in objects]
    best = None
    for oid, o in sorted(objects.items()):
        if oid in refs:
            continue
        b = expand_and_clip(box_of(o), img_w, img_h)
        if not crop_is_usable(b, img_w, img_h):
            continue
        # The crop must not merely exclude the referent's centre — it must not OVERLAP any
        # referenced object at all, or the "irrelevant" region would still show the evidence.
        if any(_overlaps(b, rb) for rb in ref_boxes):
            continue
        a = crop_area(b)
        if abs(a - target_area) > AREA_TOL * target_area:
            continue
        err = abs(a - target_area) / target_area
        if best is None or (err, oid) < (best["area_error"], best["object_id"]):
            best = {"object_id": oid, "name": o.get("name"), "box": list(b), "area": a,
                    "raw_box": list(box_of(o)), "area_error": err}
    return best


def encoder_input_side(processor):
    """The square side an image processor resizes to, or None if it does not declare one.

    Recorded so the resize factor is the real one the encoder applied, not a nominal 224 assumed
    from the model card.
    """
    for attr in ("crop_size", "size"):
        d = getattr(processor, attr, None)
        if isinstance(d, dict):
            for k in ("height", "shortest_edge", "width"):
                if k in d:
                    return int(d[k])
        elif isinstance(d, int):
            return int(d)
    return None


def padded_geometry(box, raw_box=None, encoder_side=None) -> dict:
    """Geometry of the square canvas a clipped crop box becomes under the fixed rule.

    `box` is the expanded, clipped crop; `raw_box` is the annotated object box inside it. The
    padded side is max(w, h) because pad_to_square centres the crop on a canvas of that side, so
    everything downstream — area, occupancy, resize factor — follows from the crop's longer edge.
    """
    x1, y1, x2, y2 = box
    w, h = max(0, x2 - x1), max(0, y2 - y1)
    side = max(w, h)
    canvas = side * side
    g = {"crop_w": w, "crop_h": h, "crop_area": w * h,
         "padded_w": side, "padded_h": side, "padded_area": canvas,
         "aspect_ratio": (w / h) if h else 0.0,
         "crop_occupancy": (w * h / canvas) if canvas else 0.0,
         "object_occupancy": (crop_area(raw_box) / canvas) if (raw_box and canvas) else None,
         "encoder_side": encoder_side,
         "resize_factor": (encoder_side / side) if (encoder_side and side) else None}
    return g


def _ratio(a, b) -> float:
    """The larger of two positive quantities over the smaller: always >= 1, order-independent."""
    if a <= 0 or b <= 0:
        return float("inf")
    return max(a, b) / min(a, b)


def is_scale_matched(area_a, area_b, factor: float = SCALE_MATCH_FACTOR) -> bool:
    """True if two padded crop areas are within `factor` of each other, in either direction.

    Symmetric by construction: the comparison is max/min, so swapping the arguments cannot change
    the answer. Reads nothing but the two areas — no answer, no accuracy, no arm.
    """
    return _ratio(area_a, area_b) <= factor


def scale_pair_diagnostics(rel_box, wrong_box, rel_raw=None, wrong_raw=None,
                           encoder_side=None) -> dict:
    """Pre-registered crop-scale diagnostics for one relevant/wrong pair, plus subgroup membership.

    Computed from geometry alone, before any model runs, so subgroup membership cannot be a
    function of which arm happened to answer correctly.
    """
    r = padded_geometry(rel_box, rel_raw, encoder_side)
    w = padded_geometry(wrong_box, wrong_raw, encoder_side)
    return {
        "relevant": r, "wrong": w,
        "padded_side_ratio": _ratio(r["padded_w"], w["padded_w"]),
        "padded_area_ratio": _ratio(r["padded_area"], w["padded_area"]),
        "aspect_ratio_diff": abs(r["aspect_ratio"] - w["aspect_ratio"]),
        "scale_match_factor": SCALE_MATCH_FACTOR,
        "scale_matched": is_scale_matched(r["padded_area"], w["padded_area"]),
    }


def _overlaps(a, b) -> bool:
    """True if two boxes share any pixel."""
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def normalise_label(name) -> str:
    """Canonical form of a referenced-object label: lowercase, collapsed whitespace, no plural s.

    Used only to GROUP questions that ask about the same kind of thing. It never touches the
    answer, and it is deliberately crude — an aggressive normaliser would merge distinct classes
    and let a 'car' crop stand in for a 'cat' one.
    """
    s = " ".join(str(name or "").lower().split())
    if len(s) > 3 and s.endswith("s") and not s.endswith(("ss", "us", "is")):
        s = s[:-1]
    return s


def build_wrong_crop_derangement(qids, profiles):
    """qid -> donor qid: same referenced TARGET LABEL, different image, area-proximate.

    Returns (donor, unusable) where `unusable` maps a label class to the reason it could not form
    a valid donor group. Those questions carry no defensible wrong crop and must be excluded.

    WHY MATCH ON THE TARGET LABEL. A wrong crop drawn from an arbitrary image differs from the
    relevant crop in two ways at once: it shows a different place AND usually a different KIND of
    thing. A model could then look worse on the wrong arm simply because the crop depicts
    something the question never mentions, which is a weaker control than it appears. Requiring
    the donor to depict the same normalised target label — a 'dog' question receives a crop of a
    different image's dog — isolates the variable that matters: same kind of object, wrong
    instance, wrong image.

    NEVER THE ANSWER. Grouping uses the referenced-object label from the scene graph, which is the
    question's target. It never reads the answer, and a test asserts that poisoning the answer
    cannot change the assignment.

    THE CONSTRUCTION IS DETERMINISTIC. Within a label class, questions are sorted by crop area
    (then image id, then qid) and the class is rotated by the SMALLEST shift k >= 1 for which no
    question receives a crop from its own image. A rotation is a fixed-point-free permutation, so
    every question donates exactly once and receives exactly once; taking the smallest valid k
    over an area-sorted list keeps donors area-proximate. A class with fewer than two questions,
    or one whose images admit no valid rotation, is reported rather than repaired.
    """
    by_label: dict[str, list] = {}
    for q in qids:
        by_label.setdefault(normalise_label(profiles[q]["label"]), []).append(q)

    donor, unusable = {}, {}
    for label, members in sorted(by_label.items()):
        order = sorted(members, key=lambda q: (profiles[q]["area"], profiles[q]["image_id"], q))
        n = len(order)
        if n < 2:
            unusable[label] = f"only {n} question in this target class; no donor possible"
            continue
        if len({profiles[q]["image_id"] for q in order}) < 2:
            unusable[label] = f"all {n} questions share one image; every donor would be self-image"
            continue
        shift = None
        for k in range(1, n):
            if all(profiles[order[(i + k) % n]]["image_id"] != profiles[order[i]]["image_id"]
                   for i in range(n)):
                shift = k
                break
        if shift is None:
            unusable[label] = (f"{n} questions over "
                               f"{len({profiles[q]['image_id'] for q in order})} images admit no "
                               f"rotation without a same-image donation")
            continue
        for i, q in enumerate(order):
            donor[q] = order[(i + shift) % n]
    return donor, unusable


def crop_record(question: dict, graph: dict):
    """Per-question crop geometry, or {"reject": reason}. Answer-independent by construction.

    TWO COHORTS COME OUT OF THIS. The `relevant` crop is REQUIRED — without it there is no
    treatment arm. The `irrelevant` crop is OPTIONAL and may be None: requiring it for admission
    would discard most of the slice (271 of 500 questions have no area-matched, non-overlapping
    non-referent) and would shrink the cohort carrying the PRIMARY relevant-vs-wrong contrast for
    the sake of a secondary one. Callers therefore admit on `relevant` and treat questions that
    also carry `irrelevant` as a nested subset.

    This function never reads question["answer"] or question["fullAnswer"]; a test asserts that
    poisoning those fields cannot change its output.
    """
    objects = graph.get("objects") or {}
    img_w, img_h = int(graph.get("width", 0)), int(graph.get("height", 0))
    if not objects or img_w <= 0 or img_h <= 0:
        return {"reject": "image has no scene graph or no dimensions"}
    if not referenced_object_ids(question):
        return {"reject": "question annotations name no scene-graph object"}
    rel = relevant_box(question, graph)
    if rel is None:
        return {"reject": "no referenced object yields a usable crop under the fixed rule"}
    irr = irrelevant_box(question, graph, rel["area"])
    return {
        "oracle": True,
        "image_w": img_w, "image_h": img_h,
        "relevant": rel,
        "target_label": normalise_label(rel["name"]),
        "irrelevant": irr,
        "has_irrelevant": irr is not None,
        "area_match_error": irr["area_error"] if irr else None,
        "irrelevant_reject": None if irr else
        f"no non-referenced, non-overlapping crop within {int(AREA_TOL * 100)}% of the "
        f"relevant crop area",
        "n_referenced": len(referenced_object_ids(question)), "n_objects": len(objects),
    }
