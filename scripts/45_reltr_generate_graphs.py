"""Generate and cache learned scene graphs for the tuning images with RelTR (Open Images V6).

Run under the ISOLATED environment, never the production VLM venv:

    .venv_reltr/bin/python scripts/45_reltr_generate_graphs.py

WHY AN ISOLATED ENVIRONMENT. RelTR needs torchvision; the production VLM venv has torch but no
torchvision, and installing one there risks resolving a different torch under the frozen
Qwen2/CLIP stack that every reported number depends on. `.venv_reltr` is a separate interpreter
with its own torch+torchvision. Nothing in this script writes to the production environment.

INFERENCE ONLY. RelTR is frozen and loaded from the official Open Images V6 checkpoint
(`checkpoint0149_oi.pth`). Nothing is trained, fine-tuned or adapted here.

WHY THE OPEN IMAGES CHECKPOINT AND NOT THE VISUAL GENOME ONE. GQA is built ON Visual Genome
images and VG scene graphs, so a VG-trained generator would be trained on the evaluation
distribution and, for some images, the evaluation images themselves. The OI checkpoint is
verified here by its head shapes: 289 entity classes and 31 predicates is Open Images; the VG
release is 151 and 51. A checkpoint with VG geometry is rejected rather than relabelled.

LEAKAGE. scripts/43_reltr_overlap_audit.py found 6 tuning images inside the Open Images VRD
train split and 9 unresolvable; scripts/44_reltr_clean_slice.py drops all 15. This script
defaults to that clean slice. Graphs are still generated for whatever slice it is given, but
the slice is recorded in the manifest so an analysis cannot silently mix them.

NO THRESHOLD TUNING. `--score-thresh` and `--topk` default to RelTR's own published inference
values (0.3 / 10). They were fixed before any accuracy existed and must not be revisited after.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Everything on /scratch, nothing in home — the project's golden rule. This must run BEFORE the
# first torch import: building the RelTR backbone instantiates an ImageNet ResNet-50, whose
# weights torch fetches into TORCH_HOME. `.venv_reltr` is a bare interpreter that never sources
# env.sh, so without this the 102 MB download lands in ~/.cache on a small home filesystem.
# (Those weights are then entirely overwritten by the Open Images checkpoint.)
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

from src.data.reltr_graph import (DEFAULT_SCORE_THRESH, DEFAULT_TOPK,  # noqa: E402
                                  decode_triplets, triplets_to_graph)
from src.utils import load_config  # noqa: E402

RELTR_DIR = ROOT / "third_party" / "RelTR"
# Open Images V6 geometry. The VG release is (151, 51); loading that here would be a silent
# switch to a generator trained on the evaluation distribution.
OI_NUM_CLASSES, OI_NUM_REL_CLASSES = 289, 31


def sha256_file(p: Path) -> str:
    """Hash a file so every input and output is pinned in the manifest."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_reltr(ckpt_path: Path, device: str):
    """Load the frozen RelTR Open Images model, asserting the checkpoint really is Open Images."""
    import torch
    sys.path.insert(0, str(RELTR_DIR))
    from models import build_model  # noqa: E402  (vendored third-party)

    ns = argparse.Namespace(
        dataset="oi", backbone="resnet50", dilation=False, position_embedding="sine",
        hidden_dim=256, lr_backbone=1e-5, masks=False, enc_layers=6, dec_layers=6,
        dim_feedforward=2048, dropout=0.1, nheads=8, num_entities=100, num_triplets=200,
        pre_norm=False, aux_loss=False, set_cost_class=1, set_cost_bbox=5, set_cost_giou=2,
        set_iou_threshold=0.7, bbox_loss_coef=5, giou_loss_coef=2, rel_loss_coef=1,
        eos_coef=0.1, device=device, return_interm_layers=False)
    model, _, _ = build_model(ns)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    ent = tuple(sd["entity_class_embed.weight"].shape)
    rel = tuple(sd["rel_class_embed.layers.1.weight"].shape)
    if ent[0] != OI_NUM_CLASSES + 1 or rel[0] != OI_NUM_REL_CLASSES + 1:
        raise SystemExit(
            f"FAIL checkpoint head is {ent[0]} entity / {rel[0]} relation logits; Open Images "
            f"V6 is {OI_NUM_CLASSES + 1}/{OI_NUM_REL_CLASSES + 1}. A Visual Genome checkpoint "
            f"(152/52) is forbidden: GQA is built on Visual Genome.")
    model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ck.get("epoch")


def load_vocabularies(ann_dir: Path) -> tuple[list[str], list[str]]:
    """The AUTHORITATIVE Open Images class and predicate names, in training index order.

    Derived vocabularies are not acceptable here: a wrong ordering silently relabels every
    prediction and the graphs would look plausible while being nonsense.
    """
    cats = json.loads((ann_dir / "train.json").read_text())["categories"]
    classes = [c["name"] for c in sorted(cats, key=lambda c: c["id"])]
    predicates = json.loads((ann_dir / "rel.json").read_text())["rel_categories"]
    if len(classes) != OI_NUM_CLASSES or len(predicates) != OI_NUM_REL_CLASSES:
        raise SystemExit(f"FAIL vocabularies are {len(classes)}/{len(predicates)}, expected "
                         f"{OI_NUM_CLASSES}/{OI_NUM_REL_CLASSES}")
    if classes[0] != "__background__":
        raise SystemExit(f"FAIL class index 0 is {classes[0]!r}, expected '__background__'")
    return classes, predicates


def find_image(image_id: str, dirs: list[str]) -> Path | None:
    """The image file for a GQA image id, or None."""
    for d in dirs:
        p = Path(d) / f"{image_id}.jpg"
        if p.is_file():
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qids", default="data/gqa/tune_500_reltr_clean_qids.json",
                    help="the leakage-clean subset of the tuning 500, by default")
    ap.add_argument("--ckpt", default="outputs/reltr_sgg/ckpt/checkpoint0149_oi.pth")
    ap.add_argument("--ann-dir", default="outputs/reltr_sgg/oi_ann/oi")
    ap.add_argument("--out", default="outputs/reltr_sgg/graphs_clean.json")
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--score-thresh", type=float, default=DEFAULT_SCORE_THRESH)
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N images")
    args = ap.parse_args()

    for bad in ("eval_2000", "locked", "confirm_3000", "objective_4000"):
        if bad in args.qids:
            raise SystemExit(f"FAIL {args.qids!r} names a slice this experiment must not touch")

    import torch
    import torchvision.transforms as T
    from PIL import Image

    out_p = ROOT / args.out
    if out_p.exists():
        raise SystemExit(f"FAIL refusing to overwrite {out_p}; a graph cache is frozen once "
                         f"written, and outputs/ is gitignored so an overwrite is unrecoverable")
    ckpt_p, ann_d = ROOT / args.ckpt, ROOT / args.ann_dir
    for p in (ckpt_p, ann_d / "train.json", ann_d / "rel.json"):
        if not p.exists():
            raise SystemExit(f"FAIL required input absent: {p}")

    cfg = load_config()
    qids = [str(q) for q in json.loads((ROOT / args.qids).read_text())]
    allq = json.loads(Path(cfg["gqa"]["questions"]["val_balanced"]).read_text())
    q2img = {q: str(allq[q]["imageId"]) for q in qids}
    del allq
    images = sorted(set(q2img.values()))
    if args.limit:
        images = images[: args.limit]

    print("=== RelTR (Open Images V6) scene-graph generation — INFERENCE ONLY ===")
    print(f"slice   : {args.qids}  ({len(qids)} questions, {len(images)} unique images)")
    print(f"ckpt    : {args.ckpt}")
    print(f"device  : {args.device}   score_thresh={args.score_thresh} topk={args.topk} "
          f"(RelTR's published defaults)")

    classes, predicates = load_vocabularies(ann_d)
    print(f"vocab   : {len(classes)} entity classes, {len(predicates)} predicates "
          f"(authoritative, training index order)")
    model, epoch = build_reltr(ckpt_p, args.device)
    print(f"model   : loaded, checkpoint epoch {epoch}; head verified as Open Images geometry")

    transform = T.Compose([T.Resize(800), T.ToTensor(),
                           T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    graphs, failures, stats = {}, {}, {"triplets": [], "objects": [], "relations": [], "attrs": 0}
    t0 = time.time()
    for i, img_id in enumerate(images, 1):
        path = find_image(img_id, cfg["gqa"]["image_dirs"])
        if path is None:
            failures[img_id] = "image file not found"
            continue
        try:
            im = Image.open(path).convert("RGB")
            with torch.no_grad():
                o = model(transform(im).unsqueeze(0).to(args.device))
            trips = decode_triplets(o["sub_logits"], o["obj_logits"], o["rel_logits"],
                                    o["sub_boxes"], o["obj_boxes"], classes, predicates,
                                    im.size, args.score_thresh, args.topk)
            g = triplets_to_graph(trips)
        except Exception as e:                     # one bad image must not lose the whole run
            failures[img_id] = f"{type(e).__name__}: {e}"
            continue
        stats["attrs"] += g.pop("_n_attribute_triplets", 0)
        stats["triplets"].append(len(trips))
        stats["objects"].append(len(g["objects"]))
        stats["relations"].append(sum(len(x["relations"]) for x in g["objects"].values()))
        graphs[img_id] = g
        if i % 50 == 0 or i == len(images):
            print(f"  {i}/{len(images)}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    n = len(graphs)
    mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0  # noqa: E731
    empty = sum(1 for g in graphs.values() if not g["objects"])
    print(f"\ncoverage: {n}/{len(images)} images produced a graph; {len(failures)} failed")
    print(f"  empty graphs (no triplet cleared threshold): {empty}")
    print(f"  mean triplets/image : {mean(stats['triplets']):.2f}")
    print(f"  mean objects/image  : {mean(stats['objects']):.2f}")
    print(f"  mean relations/image: {mean(stats['relations']):.2f}")
    print(f"  'is' attribute triplets (stripped at render): {stats['attrs']}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(graphs))
    manifest = {
        "generator": "RelTR (Cong et al., TPAMI 2023), official Open Images V6 checkpoint",
        "mode": "inference only; frozen; nothing trained or adapted",
        "checkpoint": args.ckpt, "checkpoint_sha256": sha256_file(ckpt_p),
        "checkpoint_epoch": epoch,
        "reltr_commit": "fca7397e9aaeccd95541e83afa4b971f3fa89014",
        "entity_classes": OI_NUM_CLASSES, "predicates": OI_NUM_REL_CLASSES,
        "training_data": "Open Images V6 VRD train split — NOT Visual Genome, NOT GQA",
        "slice": args.qids, "slice_sha256": sha256_file(ROOT / args.qids),
        "n_questions": len(qids), "n_images": len(images),
        "score_thresh": args.score_thresh, "topk": args.topk,
        "thresholds_note": "RelTR's published inference defaults; fixed before any accuracy "
                           "existed and never revisited",
        "coverage": {"images_with_graph": n, "images_failed": len(failures),
                     "empty_graphs": empty, "failures": failures},
        "graph_stats": {"mean_triplets": mean(stats["triplets"]),
                        "mean_objects": mean(stats["objects"]),
                        "mean_relations": mean(stats["relations"]),
                        "attribute_triplets_stripped": stats["attrs"]},
        "cache": args.out, "cache_sha256": sha256_file(out_p),
        "renderer": "src.data.predicted_graph.PredictedGraphStore — the SAME store T-049 used",
        "wall_seconds": round(time.time() - t0, 1),
    }
    mp = out_p.with_name(out_p.stem + "_MANIFEST.json")
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out_p.relative_to(ROOT)}  sha256 {manifest['cache_sha256'][:16]}…")
    print(f"wrote {mp.relative_to(ROOT)}")
    print("NO GPU training. NO production environment was modified.")


if __name__ == "__main__":
    main()
