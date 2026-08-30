"""Stage 2: PREDICTED scene graphs from a leakage-safe open-vocabulary detector.

The RQ1 scene-graph oracle (src/data/scene_graph.py) injects each eval image's OWN
ground-truth GQA graph, so it is an upper bound. Stage 2 replaces that source with a
graph PREDICTED from pixels by OWLv2 (open-vocabulary detection) plus box-geometry
spatial heuristics and a coarse colour attribute — nothing that was trained on GQA/VG
scene-graph supervision, so it passes the Section 9.4 leakage gate off the shelf.

Design (mirrors the oracle so the comparison is apples-to-apples):
  * generate_graph(image) -> a GQA-shaped graph dict
        {"objects": {id: {"name", "attributes": [...], "relations": [{"name", "object"}]}}}
    built by: OWLv2 detect -> NMS/dedup -> colour attribute -> pairwise spatial relations.
  * PredictedGraphStore reads a PRECOMPUTED file of those dicts (keyed by image id) and
    subclasses SceneGraphStore, so describe(image_id) uses the IDENTICAL serialisation.
    The heavy OWLv2 pass runs once, offline, into a cache; eval never co-loads OWLv2.

Leakage note: OWLv2 is pretrained on web image-text, never on VG scene graphs, and the
object VOCABULARY is applied uniformly to every image (it is a prompt list, not per-image
supervision). The production vocabulary is VG-train object frequencies with the 10,234
GQA-eval image ids removed (data/gqa/vg_exclude_gqa_eval.json); DEFAULT_VOCAB below is the
leakage-safe curated stand-in used for the proposal demo.
"""
from __future__ import annotations

import torch

from src.data.scene_graph import SceneGraphStore

# Curated, leakage-safe object vocabulary for the proposal demo: common GQA/VG objects,
# applied uniformly to every image. NOT derived from any eval-image graph. The production
# generator swaps in VG-train frequency ranks (eval ids excluded) via build_vocab().
DEFAULT_VOCAB = [
    "man", "woman", "person", "child", "boy", "girl", "people",
    "shirt", "jacket", "coat", "hat", "pants", "shoe", "glasses", "bag", "helmet",
    "car", "truck", "bus", "train", "bicycle", "motorcycle", "boat", "airplane",
    "dog", "cat", "horse", "cow", "sheep", "bird", "elephant", "bear", "zebra", "giraffe",
    "table", "chair", "couch", "bench", "bed", "desk", "shelf", "cabinet", "door", "window",
    "cup", "bottle", "bowl", "plate", "fork", "knife", "spoon", "glass",
    "laptop", "phone", "keyboard", "television", "clock", "book", "picture", "mirror",
    "tree", "plant", "flower", "grass", "bush", "rock", "mountain", "sky", "cloud",
    "building", "house", "wall", "fence", "roof", "sign", "pole", "street", "sidewalk", "road",
    "ball", "umbrella", "kite", "surfboard", "skateboard", "racket",
    "pizza", "sandwich", "banana", "apple", "orange", "cake", "donut",
    "lamp", "light", "vase", "pillow", "blanket", "towel", "curtain", "basket",
    "hair", "hand", "face", "head", "leg", "arm", "eye", "nose",
]

# A tiny named-colour palette (RGB anchors) for the coarse colour attribute. Kept small and
# unambiguous; a detection gets a colour only when its region maps cleanly to one of these.
_COLOURS = {
    "black": (30, 30, 30), "white": (235, 235, 235), "gray": (128, 128, 128),
    "red": (200, 40, 40), "green": (60, 160, 60), "blue": (50, 90, 200),
    "yellow": (225, 210, 60), "orange": (230, 140, 40), "brown": (120, 80, 45),
    "pink": (230, 150, 175), "purple": (135, 70, 165),
}

OWLV2_MODEL = "google/owlv2-base-patch16-ensemble"

# Words that are never object queries: question machinery, spatial/relation terms, colours,
# and photo-meta words. Everything else in a question is a candidate OWLv2 query.
_QUESTION_STOP = set(
    "what which where who whose why how does do is are was were the that this these those "
    "there their his her its any some kind type sort name color colour side part place "
    "left right front behind above below near next top bottom under over between inside "
    "outside around look looks seem appear made makes doing wearing holding sitting standing "
    "picture photo image photograph scene background foreground thing things object objects "
    "playing eating riding walking running flying holding holds hold hanging covering covered "
    "leaning lying laying watching catching throwing jumping driving pulling parked using "
    "black white gray grey red green blue yellow orange brown pink purple both either same "
    "different large small big little long short tall high low good bad new old young "
    "with from into onto about than then them they you and not but for very more most "
    "less least other another either".split())


def question_nouns(question: str, vocab: list[str], max_extra: int = 8) -> list[str]:
    """Content words from the question worth adding as OWLv2 queries (not already in vocab).

    The Stage-2a diagnosis: 80% of right->wrong flips asked about an object the graph never
    mentioned, and the model then trusted the incomplete list over the image. Querying the
    question's own nouns makes the graph speak to what is actually being asked. Uses only the
    question text (available at inference) — no eval supervision, leakage-safe.
    """
    import re
    seen, extras = set(vocab), []
    for w in re.findall(r"[a-z]+", question.lower()):
        if len(w) >= 3 and w not in _QUESTION_STOP and w not in seen:
            seen.add(w)
            extras.append(w)
            if len(extras) >= max_extra:
                break
    return extras


def load_vocab(path: str | None = None) -> list[str]:
    """The production vocabulary (VG-train frequencies, eval ids excluded) or the fallback.

    scripts/15_build_pred_vocab.py writes the file and scripts/13_vg_leakage_audit.py
    --train-ids proves its source images contain zero GQA-eval images (residual-zero gate).
    """
    import json
    from pathlib import Path
    p = Path(path or "data/gqa/pred_vocab.json")
    if p.exists():
        return json.load(open(p))["vocab"]
    return DEFAULT_VOCAB


def load_owlv2(device: str = "cuda"):
    """Load the frozen OWLv2 open-vocabulary detector (processor, model) onto device."""
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
    from src.models.revisions import revision_for

    revision = revision_for(OWLV2_MODEL)
    processor = Owlv2Processor.from_pretrained(OWLV2_MODEL, revision=revision)
    model = Owlv2ForObjectDetection.from_pretrained(
        OWLV2_MODEL, revision=revision
    ).to(device).eval()
    return processor, model


def _iou(a, b) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    inter = _intersection(a, b)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _intersection(a, b) -> float:
    """Intersection area of two [x1, y1, x2, y2] boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return iw * ih


def _containment(a, b) -> float:
    """Fraction of the SMALLER box covered by its overlap with the other (catches nesting)."""
    inter = _intersection(a, b)
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return inter / smaller if smaller > 0 else 0.0


def _nms(dets: list[dict], iou_thresh: float = 0.5, same_iou: float = 0.3,
         contain_thresh: float = 0.7) -> list[dict]:
    """Dedup detections: suppress a box if it overlaps, is largely contained in, or is a
    same-class near-duplicate of, an already-kept higher-scoring box. Stacked/nested boxes
    (low IoU but high containment) and repeated same-class boxes are the main Stage-1 noise.
    """
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: x["score"], reverse=True):
        redundant = False
        for k in kept:
            iou = _iou(d["box"], k["box"])
            same = d["name"] == k["name"]
            if iou >= iou_thresh or _containment(d["box"], k["box"]) >= contain_thresh \
                    or (same and iou >= same_iou):
                redundant = True
                break
        if not redundant:
            kept.append(d)
    return kept


def _colour_of(image, box) -> str | None:
    """Nearest named colour of the box centre, or None if the region is textured/ambiguous.

    Bails when the central region is high-variance (patterned objects like a giraffe's coat or
    a sky gradient map to a wrong flat colour), so a colour is assigned only when it is safe.
    """
    x1, y1, x2, y2 = box
    # Sample the central 50% of the box to avoid background bleeding in at the edges.
    cx1, cy1 = x1 + (x2 - x1) * 0.25, y1 + (y2 - y1) * 0.25
    cx2, cy2 = x1 + (x2 - x1) * 0.75, y1 + (y2 - y1) * 0.75
    crop = image.crop((cx1, cy1, cx2, cy2)).resize((16, 16))
    px = list(crop.getdata())
    n = len(px)
    means = [sum(p[c] for p in px) / n for c in range(3)]
    var = sum((p[c] - means[c]) ** 2 for p in px for c in range(3)) / (3 * n)
    if var > 2600:                          # textured / multi-colour region -> no confident colour
        return None
    name, dist = min(((cn, sum((means[c] - anchor[c]) ** 2 for c in range(3)))
                      for cn, anchor in _COLOURS.items()), key=lambda t: t[1])
    return name if dist < 3000 else None    # only assign a clearly-nearest colour


def _relation(a_box, b_box) -> str:
    """Spatial predicate for object A relative to object B, from box-centre geometry."""
    acx, acy = (a_box[0] + a_box[2]) / 2, (a_box[1] + a_box[3]) / 2
    bcx, bcy = (b_box[0] + b_box[2]) / 2, (b_box[1] + b_box[3]) / 2
    dx, dy = bcx - acx, bcy - acy
    if abs(dx) >= abs(dy):
        return "to the left of" if dx > 0 else "to the right of"
    return "above" if dy > 0 else "below"      # image y grows downward: smaller y = higher


@torch.no_grad()
def generate_graph(image, processor, model, device: str = "cuda", vocab: list[str] | None = None,
                   score_thresh: float = 0.10, max_objects: int = 8,
                   max_related: int = 6, max_relations: int = 12,
                   extra_queries: list[str] | None = None) -> dict:
    """Predict a GQA-shaped scene graph for one PIL image (OWLv2 + geometry + colour).

    Detections are thresholded, de-duplicated (NMS), colour-tagged, and the most salient
    object PAIRS are joined by a spatial relation. Returns {"objects": {...}} in the oracle
    schema. Relations skip indistinguishable (same name+colour) pairs and near-coincident
    boxes, and are capped so the injected prompt stays a short list of confident facts.
    """
    vocab = vocab or DEFAULT_VOCAB
    if extra_queries:                            # question-conditioned: append the question's
        vocab = vocab + [w for w in extra_queries if w not in vocab]   # nouns as extra queries
    queries = [[f"a photo of a {w}" for w in vocab]]
    inputs = processor(text=queries, images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    target = torch.tensor([[image.height, image.width]], device=device)
    try:                                        # method name varies across transformers versions
        post = processor.post_process_grounded_object_detection
    except AttributeError:
        post = processor.post_process_object_detection
    res = post(outputs=outputs, target_sizes=target, threshold=score_thresh)[0]

    dets = [{"box": [float(v) for v in box], "score": float(s), "name": vocab[int(l)]}
            for box, s, l in zip(res["boxes"], res["scores"], res["labels"])]
    dets = _nms(dets)[:max_objects]

    objects: dict[str, dict] = {}
    for i, d in enumerate(dets):
        colour = _colour_of(image, d["box"])
        objects[str(i)] = {"name": d["name"], "score": round(d["score"], 3),
                           "box": [round(v, 1) for v in d["box"]],
                           "attributes": [colour] if colour else [], "relations": []}

    # Candidate spatial relations among the highest-scoring objects, ranked by salience
    # (combined detection confidence) and kept only when the two objects are (a) genuinely
    # distinguishable and (b) far enough apart for the spatial claim to mean something.
    diag = (image.width ** 2 + image.height ** 2) ** 0.5
    top = list(objects.keys())[:max_related]
    candidates = []
    for a in range(len(top)):
        for b in range(a + 1, len(top)):
            oa, ob = objects[top[a]], objects[top[b]]
            identity = (oa["name"], tuple(oa["attributes"]))
            if identity == (ob["name"], tuple(ob["attributes"])):
                continue                        # skip 'a black bowl above a black bowl'
            acx, acy = (oa["box"][0] + oa["box"][2]) / 2, (oa["box"][1] + oa["box"][3]) / 2
            bcx, bcy = (ob["box"][0] + ob["box"][2]) / 2, (ob["box"][1] + ob["box"][3]) / 2
            sep = ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5
            if sep < 0.1 * diag:
                continue                        # boxes too coincident for a spatial relation
            candidates.append((oa["score"] + ob["score"], top[a], top[b]))
    for _, a_id, b_id in sorted(candidates, reverse=True)[:max_relations]:
        objects[a_id]["relations"].append(
            {"name": _relation(objects[a_id]["box"], objects[b_id]["box"]), "object": b_id})
    return {"objects": objects}


class PredictedGraphStore(SceneGraphStore):
    """Oracle-identical serialisation over PRECOMPUTED predicted graphs (OWLv2 + geometry).

    Reads a JSON of generate_graph() outputs keyed by image id and inherits describe() from
    SceneGraphStore unchanged, so SceneGraphAugment consumes it with no modification and the
    injected text is formatted exactly like the oracle's — only the graph SOURCE differs.

    strip_attributes=True (the primary Stage-2 condition) drops the colour attribute at load
    time: the demo A/B showed colour tags are wrong often enough ('pink sidewalk') to risk
    actively misleading the LLM, so the injected facts are objects + spatial relations only.
    """

    label = "predicted (OWLv2 + box-geometry)"

    def __init__(self, path: str, max_facts: int = 40, max_attrs: int = 3,
                 strip_attributes: bool = True):
        """Load the precomputed graphs; optionally drop attributes (colour off)."""
        super().__init__(path, max_facts=max_facts, max_attrs=max_attrs)
        if strip_attributes:
            for g in self._graphs.values():
                for o in g.get("objects", {}).values():
                    o["attributes"] = []
