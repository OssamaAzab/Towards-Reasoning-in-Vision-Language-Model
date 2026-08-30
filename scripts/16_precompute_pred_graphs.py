"""Stage 2: precompute PREDICTED scene graphs for every locked-set eval image.

Runs the leakage-safe generator (OWLv2 + box-geometry; vocab from scripts/15, gate from
scripts/13 --train-ids) ONCE over the unique images behind the locked 2,000-qid set and
writes the graph cache that PredictedGraphStore / the 'scene_graph_pred' augmentation
reads. Keeping this offline means the eval never co-loads OWLv2 with the LLM+encoder.

Resumable: reruns skip images already in the cache (the file is rewritten atomically
after every --save-every images, so a kill loses at most that much work).

    python scripts/16_precompute_pred_graphs.py            # all locked-set images
    python scripts/16_precompute_pred_graphs.py --limit 20 # smoke test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset  # noqa: E402
from src.data.predicted_graph import generate_graph, load_owlv2, load_vocab  # noqa: E402
from src.utils import load_config  # noqa: E402

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    """Generate predicted graphs for all locked-set images into the configured cache."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="cap images (smoke test)")
    ap.add_argument("--save-every", type=int, default=50, help="atomic cache rewrite interval")
    ap.add_argument("--score-thresh", type=float, default=0.10)
    ap.add_argument("--qids", default="data/gqa/eval_2000_qids.json",
                    help="qid list whose unique images to cover (e.g. the tuning slice)")
    ap.add_argument("--out", default=None, help="cache path (default: config pred_graphs)")
    # The vocabulary was previously an invisible default here while scripts/18 took it as
    # a flag, so the per-image cache was built from 120 terms and the per-question cache
    # from 400. Any scene_graph_pred vs scene_graph_predq comparison then conflated
    # question-conditioning with a 3.3x larger detector vocabulary. Making it explicit
    # lets the two be built from the same vocabulary when the comparison is the point.
    # Default is None -> load_vocab()'s own default, so legacy runs reproduce unchanged.
    ap.add_argument("--vocab", default=None,
                    help="leakage-safe vocabulary JSON (default: data/gqa/pred_vocab.json, "
                         "120 terms). Pass data/gqa/pred_vocab_400.json to match the "
                         "vocabulary scripts/18 uses for the per-question cache")
    args = ap.parse_args()

    cfg = load_config()
    out_path = Path(args.out or cfg["augmentation"]["pred_graphs"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
    locked = json.load(open(args.qids))
    image_ids, image_paths = [], {}
    for qid in locked:                              # unique images, locked-set order
        ex = ds.get(qid)
        if ex.image_id not in image_paths and ex.image_path is not None:
            image_ids.append(ex.image_id)
            image_paths[ex.image_id] = ex.image_path
    if args.limit:
        image_ids = image_ids[: args.limit]

    done: dict = {}
    if out_path.exists():
        done = json.load(open(out_path))
        log(f"resume: cache already holds {len(done):,} graphs")
    todo = [i for i in image_ids if str(i) not in done]
    log(f"locked set: {len(locked)} qids -> {len(image_ids):,} unique images; "
        f"{len(todo):,} to generate")
    if not todo:
        log("nothing to do; cache complete")
        return

    vocab = load_vocab(args.vocab)
    vocab_path = Path(args.vocab or "data/gqa/pred_vocab.json")
    log(f"vocabulary: {len(vocab)} terms from {vocab_path}")

    # The leakage gate is named after the vocabulary it gated, so it must be DERIVED, not
    # hardcoded: this line previously always claimed 'pred_vocab_image_ids.json' even when
    # the 400-term vocabulary was in use, whose gate is pred_vocab_400_image_ids.json. A
    # provenance line that names the wrong gate file is worse than none, because the leakage
    # exclusion is exactly the claim a reader would want to audit.
    gate = vocab_path.with_name(f"{vocab_path.stem}_image_ids.json")
    meta = json.load(open(vocab_path))
    log(f"vocab provenance: built_from={meta.get('built_from')}, "
        f"n_source_images={meta.get('n_source_images')}")
    if gate.is_file():
        log(f"leakage gate: scripts/13 --train-ids {gate} -> residual 0")
    else:
        log(f"leakage gate: WARNING no gate file at {gate}; residual-zero NOT asserted here")
    log("loading OWLv2 (frozen, off-the-shelf) ...")
    processor, model = load_owlv2("cuda")

    def flush():
        """Atomic rewrite of the cache (temp sibling + os.replace)."""
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text(json.dumps(done))
        os.replace(tmp, out_path)

    n_empty = 0
    for k, iid in enumerate(tqdm(todo, desc="graphs", unit="img", dynamic_ncols=True), 1):
        image = Image.open(image_paths[iid]).convert("RGB")
        graph = generate_graph(image, processor, model, "cuda", vocab=vocab,
                               score_thresh=args.score_thresh)
        done[str(iid)] = graph
        n_empty += not graph["objects"]
        if k % args.save_every == 0:
            flush()
    flush()

    log(f"wrote {out_path}: {len(done):,} graphs "
        f"({n_empty} with zero detections this run)")
    log("NEXT: scripts/09_augment_eval.py --augment scene_graph_pred --qids "
        "data/gqa/eval_2000_qids.json --checkpoint <bridge_..._500k_ep3.pt>")


if __name__ == "__main__":
    main()
