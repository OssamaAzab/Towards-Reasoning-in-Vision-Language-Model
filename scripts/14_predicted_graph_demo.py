"""Stage 2 PROPOSAL demo: show predicted scene graphs on a few locked-set eval images.

Runs the leakage-safe generator (src/data/predicted_graph.py: OWLv2 + box-geometry + colour)
on a small sample of the LOCKED 2,000-qid eval set and prints, per image:
  * the GQA question + gold answer,
  * the raw OWLv2 detections (name, score, box),
  * the PREDICTED graph serialised through the real PredictedGraphStore, and
  * the ORACLE graph (ground-truth GQA) for the same image, serialised identically.

This is the gate deliverable: it does NOT run any bridge/LLM eval and writes nothing to the
eval outputs. It exists so the predicted-graph APPROACH can be judged before the full run.

    python scripts/14_predicted_graph_demo.py --n 3 --category relate
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset  # noqa: E402
from src.data.predicted_graph import PredictedGraphStore, generate_graph, load_owlv2  # noqa: E402
from src.data.scene_graph import SceneGraphStore  # noqa: E402
from src.utils import load_config  # noqa: E402

DEMO_DIR = Path(__file__).resolve().parent.parent / "outputs" / "predicted_graph_demo"


def pick_qids(ds, oracle, locked, category, n):
    """First n locked qids in `category` that have a resolvable image AND an oracle graph."""
    out = []
    for qid in locked:
        ex = ds.get(qid)
        if ex.image_path is None or str(ex.image_id) not in oracle:
            continue
        if category and ds.category_of(ex) != category:
            continue
        out.append(qid)
        if len(out) >= n:
            break
    return out


def main() -> None:
    """Generate and print predicted-vs-oracle graphs for a few locked-set eval images."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=3, help="number of eval images to show")
    ap.add_argument("--category", default="relate", help="GQA category to sample (relate|exist|...|'' for any)")
    ap.add_argument("--score-thresh", type=float, default=0.10, help="OWLv2 detection threshold")
    args = ap.parse_args()

    cfg = load_config()
    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
    oracle = SceneGraphStore(cfg["gqa"]["scene_graphs"])
    locked = json.load(open("data/gqa/eval_2000_qids.json"))

    qids = pick_qids(ds, oracle, locked, args.category, args.n)
    print(f"Selected {len(qids)} locked-set '{args.category}' qids with image + oracle graph: {qids}\n")

    print("loading OWLv2 (google/owlv2-base-patch16-ensemble, frozen, off-the-shelf) ...")
    processor, model = load_owlv2("cuda")

    predicted_dicts = {}
    for qid in qids:
        ex = ds.get(qid)
        image = Image.open(ex.image_path).convert("RGB")
        graph = generate_graph(image, processor, model, "cuda", score_thresh=args.score_thresh)
        predicted_dicts[str(ex.image_id)] = graph

        print("=" * 90)
        print(f"qid {qid}  image {ex.image_id}  category '{ds.category_of(ex)}'")
        print(f"  Q: {ex.question}")
        print(f"  gold answer: {ex.answer}")
        print(f"  raw OWLv2 detections ({len(graph['objects'])} kept, thresh {args.score_thresh}):")
        for o in graph["objects"].values():
            attr = f" [{','.join(o['attributes'])}]" if o["attributes"] else ""
            print(f"      {o['name']:12}{attr:12} score={o['score']:.3f}  box={o['box']}")

    # Serialise the predicted graphs through the REAL store (write cache -> load -> describe),
    # proving the full generate->cache->store->describe path the production run will use.
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    cache = DEMO_DIR / "predicted_graphs_demo.json"
    cache.write_text(json.dumps(predicted_dicts))
    pred_store = PredictedGraphStore(str(cache))

    # Colour-off A/B: same graphs with attributes stripped, to judge if colour earns its place.
    nocolour = {iid: {"objects": {oid: {**o, "attributes": []} for oid, o in g["objects"].items()}}
                for iid, g in predicted_dicts.items()}
    nocolour_cache = DEMO_DIR / "predicted_graphs_demo_nocolour.json"
    nocolour_cache.write_text(json.dumps(nocolour))
    nocolour_store = PredictedGraphStore(str(nocolour_cache))

    print("\n" + "#" * 90)
    print("# PREDICTED vs ORACLE — identical serialisation, only the graph SOURCE differs")
    print("#" * 90)
    for qid in qids:
        ex = ds.get(qid)
        print("=" * 90)
        print(f"qid {qid}  image {ex.image_id}  | Q: {ex.question}  (gold: {ex.answer})")
        print(f"\n  PREDICTED (with colour):")
        print(f"    {pred_store.describe(ex.image_id) or '(empty)'}")
        print(f"\n  PREDICTED (colour off, A/B):")
        print(f"    {nocolour_store.describe(ex.image_id) or '(empty)'}")
        print(f"\n  ORACLE (ground-truth GQA graph — the upper bound):")
        print(f"    {oracle.describe(ex.image_id) or '(empty)'}")
        print()

    _save_overlays(ds, qids, predicted_dicts)

    print("#" * 90)
    print("GATE: proposal demo only — no bridge/LLM eval was run, nothing written to outputs/eval.")
    print("Awaiting your sign-off before the full precompute + predicted-vs-oracle evaluation.")


def _save_overlays(ds, qids, predicted_dicts):
    """Draw the kept detections (label, score, colour) on each image -> artifacts/ for review."""
    from PIL import ImageDraw
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    for qid in qids:
        ex = ds.get(qid)
        img = Image.open(ex.image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        for o in predicted_dicts[str(ex.image_id)]["objects"].values():
            x1, y1, x2, y2 = o["box"]
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
            tag = f"{' '.join(o['attributes'])} {o['name']} {o['score']:.2f}".strip()
            draw.text((x1 + 2, max(0, y1 - 11)), tag, fill=(255, 255, 0))
        path = out_dir / f"predicted_graph_demo_{ex.image_id}.png"
        img.save(path)
        print(f"  saved overlay -> {path}")


if __name__ == "__main__":
    main()
