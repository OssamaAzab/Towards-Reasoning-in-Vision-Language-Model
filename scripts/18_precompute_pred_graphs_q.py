"""Stage 2b: precompute QUESTION-CONDITIONED predicted graphs for a qid list.

Like scripts/16 but per-question: each qid gets its own OWLv2 pass whose query list is the
leakage-safe vocabulary PLUS the question's own nouns (src.data.predicted_graph.question_nouns),
and the cache is keyed by qid. This attacks the Stage-2a failure mode measured in the run-1
records (285/356 right->wrong flips asked about an object the graph never mentioned): the
graph can no longer be silent about the thing being asked. Uses only the question text —
available at inference, no eval supervision, leakage-safe.

Resumable; atomic cache rewrites every --save-every questions.

    python scripts/18_precompute_pred_graphs_q.py --qids data/gqa/tune_500_qids.json \
        --out outputs/pred_graphs/tune500_q.json
    python scripts/18_precompute_pred_graphs_q.py            # locked set -> config path
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
from src.data.predicted_graph import generate_graph, load_owlv2, load_vocab, question_nouns  # noqa: E402
from src.utils import load_config  # noqa: E402

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    """Generate question-conditioned predicted graphs for every qid in the list."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qids", default="data/gqa/eval_2000_qids.json")
    ap.add_argument("--out", default=None, help="cache path (default: config pred_graphs_q)")
    ap.add_argument("--vocab", default="data/gqa/pred_vocab_400.json",
                    help="leakage-gated vocabulary (scripts/15 --tag; gate via scripts/13)")
    ap.add_argument("--save-every", type=int, default=50)
    ap.add_argument("--score-thresh", type=float, default=0.10)
    args = ap.parse_args()

    cfg = load_config()
    out_path = Path(args.out or cfg["augmentation"]["pred_graphs_q"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
    qids = [q for q in json.load(open(args.qids)) if ds.get(q).image_path is not None]

    done: dict = {}
    if out_path.exists():
        done = json.load(open(out_path))
        log(f"resume: cache already holds {len(done):,} graphs")
    todo = [q for q in qids if q not in done]
    log(f"{args.qids}: {len(qids)} usable qids; {len(todo):,} to generate -> {out_path}")
    if not todo:
        log("nothing to do; cache complete")
        return

    vocab = load_vocab(args.vocab)
    log(f"vocab: {len(vocab)} names ({args.vocab}; residual-zero gate: scripts/13 --train-ids "
        f"{args.vocab.replace('.json', '_image_ids.json')})")
    log("loading OWLv2 (frozen, off-the-shelf) ...")
    processor, model = load_owlv2("cuda")

    def flush():
        """Atomic rewrite of the cache (temp sibling + os.replace)."""
        tmp = out_path.with_name(out_path.name + ".tmp")
        tmp.write_text(json.dumps(done))
        os.replace(tmp, out_path)

    n_extra = 0
    for k, qid in enumerate(tqdm(todo, desc="graphs", unit="q", dynamic_ncols=True), 1):
        ex = ds.get(qid)
        extras = question_nouns(ex.question, vocab)
        n_extra += len(extras)
        image = Image.open(ex.image_path).convert("RGB")
        done[qid] = generate_graph(image, processor, model, "cuda", vocab=vocab,
                                   score_thresh=args.score_thresh, extra_queries=extras)
        if k % args.save_every == 0:
            flush()
    flush()

    log(f"wrote {out_path}: {len(done):,} graphs "
        f"(mean {n_extra / len(todo):.1f} question nouns added per query list)")


if __name__ == "__main__":
    main()
