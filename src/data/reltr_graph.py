"""RelTR (Open Images V6) triplets -> the T-049 graph schema, rendered by the existing store.

WHY A SEPARATE MODULE. The experiment's whole point is to change the graph SOURCE and nothing
else. So the output here is the same dict shape `src/data/predicted_graph.py` produces —
`{"objects": {id: {"name", "score", "box", "attributes": [...], "relations": [{"name", "object"}]}}}`
— and it is rendered at evaluation time by the SAME `PredictedGraphStore`. No bespoke renderer
exists for this arm, so a formatting difference cannot masquerade as an effect.

WHAT RelTR GIVES AND WHAT THIS DOES WITH IT. RelTR emits ranked (subject, predicate, object)
triplets, each with its own subject and object box. The same physical entity appears in several
triplets, so entities are merged by (class name, box IoU) into one node before relations are
attached. Without that merge a graph with 10 triplets renders 20 objects and reads as though the
detector saw two of everything.

THE `is` PREDICATE. Open Images V6 uses `is` for attribute relations (`Table is Wooden`), not
spatial ones. Rendered through the T-049 relation template it would produce "a table is a
wooden". So an `is` triplet becomes an ATTRIBUTE on its subject instead. `PredictedGraphStore`
loads with `strip_attributes=True`, exactly as it did for T-049's colour tags, so those facts
are dropped at render time by the same rule that dropped colour — not by a new one invented here.

THRESHOLDS ARE UPSTREAM DEFAULTS, NOT TUNED. `score_thresh=0.3` and `topk=10` are the values in
RelTR's own `inference.py`. They are fixed here before any accuracy exists and must not be
changed afterwards.
"""
from __future__ import annotations

# RelTR inference.py defaults — the author's published values, not chosen by this project.
DEFAULT_SCORE_THRESH = 0.3
DEFAULT_TOPK = 10

# Two detections of the same class merge into one entity above this box IoU.
ENTITY_MERGE_IOU = 0.7

ATTRIBUTE_PREDICATE = "is"


def _iou(a, b) -> float:
    """Intersection-over-union of two [x1, y1, x2, y2] boxes."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def readable(name: str) -> str:
    """Open Images label -> the lower-case surface form the T-049 template expects.

    OI names are capitalised ('Wine glass') and its predicates are snake_case
    ('talk_on_phone'); the T-049 renderer emits 'a <phrase> <relation> a <phrase>', so both are
    normalised here rather than in the renderer, which stays untouched.
    """
    return name.replace("_", " ").strip().lower()


def triplets_to_graph(triplets: list[dict], *, merge_iou: float = ENTITY_MERGE_IOU) -> dict:
    """Merge RelTR triplets into one T-049-shaped graph dict.

    `triplets` are already thresholded and ranked by the caller, best first. Each carries
    sub_name/sub_box/sub_score, pred_name/pred_score and obj_name/obj_box/obj_score.
    Entities are merged by class name and box overlap; `is` triplets become attributes.
    """
    objects: dict[str, dict] = {}
    order: list[str] = []

    def node_for(name: str, box: list[float], score: float) -> str:
        """Existing node id for this detection, or a new one."""
        nm = readable(name)
        for oid in order:
            o = objects[oid]
            if o["name"] == nm and _iou(o["box"], box) >= merge_iou:
                if score > o["score"]:            # keep the most confident box for the entity
                    o["score"], o["box"] = round(float(score), 3), [round(float(v), 1) for v in box]
                return oid
        oid = str(len(order))
        objects[oid] = {"name": nm, "score": round(float(score), 3),
                        "box": [round(float(v), 1) for v in box],
                        "attributes": [], "relations": []}
        order.append(oid)
        return oid

    n_attr = 0
    for t in triplets:
        s_id = node_for(t["sub_name"], t["sub_box"], t["sub_score"])
        pred = readable(t["pred_name"])
        if pred == ATTRIBUTE_PREDICATE:
            # 'Table is Wooden' -> attribute 'wooden' on the table; no object node is created,
            # because 'a wooden' is not a thing the renderer should ever emit as an entity.
            attr = readable(t["obj_name"])
            if attr not in objects[s_id]["attributes"]:
                objects[s_id]["attributes"].append(attr)
                n_attr += 1
            continue
        o_id = node_for(t["obj_name"], t["obj_box"], t["obj_score"])
        if o_id == s_id:
            continue                              # a merged self-relation says nothing
        if any(r["name"] == pred and r["object"] == o_id for r in objects[s_id]["relations"]):
            continue                              # exact duplicate triplet after merging
        objects[s_id]["relations"].append({"name": pred, "object": o_id})

    return {"objects": objects, "_n_attribute_triplets": n_attr}


def decode_triplets(sub_logits, obj_logits, rel_logits, sub_boxes, obj_boxes,
                    classes: list[str], predicates: list[str], size: tuple[int, int],
                    score_thresh: float = DEFAULT_SCORE_THRESH, topk: int = DEFAULT_TOPK):
    """RelTR raw outputs -> ranked triplet dicts, mirroring the upstream inference.py decode.

    Kept in one place so the cache builder and its tests agree on the decode. `size` is the
    ORIGINAL (width, height); RelTR predicts normalised cxcywh, which is converted to absolute
    xyxy here exactly as upstream `rescale_bboxes` does.
    """
    import torch

    probas = rel_logits.softmax(-1)[0, :, :-1]
    probas_sub = sub_logits.softmax(-1)[0, :, :-1]
    probas_obj = obj_logits.softmax(-1)[0, :, :-1]
    keep = torch.logical_and(
        probas.max(-1).values > score_thresh,
        torch.logical_and(probas_sub.max(-1).values > score_thresh,
                          probas_obj.max(-1).values > score_thresh))

    idx = torch.nonzero(keep, as_tuple=True)[0]
    if idx.numel() == 0:
        return []
    rank = (probas[idx].max(-1)[0] * probas_sub[idx].max(-1)[0] * probas_obj[idx].max(-1)[0])
    idx = idx[torch.argsort(-rank)[:topk]]

    def rescale(box):
        cx, cy, w, h = box.unbind(-1)
        b = torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)
        return b * torch.tensor([size[0], size[1], size[0], size[1]], dtype=torch.float32)

    sb = rescale(sub_boxes[0, idx])
    ob = rescale(obj_boxes[0, idx])

    out = []
    for j, q in enumerate(idx.tolist()):
        out.append({
            "sub_name": classes[int(probas_sub[q].argmax())],
            "sub_score": float(probas_sub[q].max()),
            "sub_box": [float(v) for v in sb[j]],
            "pred_name": predicates[int(probas[q].argmax())],
            "pred_score": float(probas[q].max()),
            "obj_name": classes[int(probas_obj[q].argmax())],
            "obj_score": float(probas_obj[q].max()),
            "obj_box": [float(v) for v in ob[j]],
        })
    return out
