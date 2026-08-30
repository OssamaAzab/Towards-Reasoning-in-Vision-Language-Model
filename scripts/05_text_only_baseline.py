"""Step 5: text-only GQA baseline — the frozen LLM answers without the image.

Establishes a floor: the accuracy reachable from the question text and language
priors alone (no image, no bridge). Any vision-grounded model must beat this; the
gap between this floor and a full model is the value added by actually seeing the
image. Reports exact-match accuracy (the headline), a relaxed match (diagnostic),
a majority-class floor, and a per-reasoning-category breakdown.

    python scripts/05_text_only_baseline.py --limit 200      # quick sample
    python scripts/05_text_only_baseline.py --limit 1000     # firmer estimate
"""
import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Make the repo root importable when run as `python scripts/05_text_only_baseline.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset, REASONING_CATEGORIES  # noqa: E402
from src.eval.metrics import exact_match, normalize_answer, relaxed_match  # noqa: E402
from src.models.llm import generate, load_llm  # noqa: E402
from src.utils import ensure_dir, load_config, set_seed  # noqa: E402

# Force a short answer and discourage refusals, so the score reflects language
# priors rather than verbosity. The image is deliberately not provided.
SYSTEM = ("Answer the question with a single word or a very short phrase, in "
          "lowercase, with no punctuation and no explanation. If you are unsure, "
          "give your single best guess.")

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    """Run the frozen LLM on GQA questions without images and score the answers."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=200,
                    help="number of questions to sample (default 200; None-like 0 = all)")
    ap.add_argument("--split", default="val_balanced",
                    help="which questions split key from config.gqa.questions")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.get("seed", 42))

    ds = GQADataset(cfg["gqa"]["questions"][args.split], cfg["gqa"]["image_dirs"])
    # Reproducible random sample across the split (more representative than head).
    qids = ds.qids if args.limit in (None, 0) else random.sample(
        ds.qids, min(args.limit, len(ds.qids)))
    n = len(qids)
    log(f"text-only baseline on {n} GQA questions from '{args.split}'")

    llm = load_llm(cfg, log=log)
    log("model loaded; answering (no image provided) ...")

    exact = relaxed = 0
    by_cat = defaultdict(lambda: [0, 0, 0])  # category -> [exact, relaxed, total]
    gold_counter: Counter = Counter()
    samples = []

    t0 = time.perf_counter()
    for i, qid in enumerate(qids, 1):
        ex = ds.get(qid)
        pred = generate(llm, ex.question, max_new_tokens=10, system=SYSTEM)
        cat = ds.category_of(ex)
        em, rm = exact_match(pred, ex.answer), relaxed_match(pred, ex.answer)
        exact += em
        relaxed += rm
        by_cat[cat][0] += em
        by_cat[cat][1] += rm
        by_cat[cat][2] += 1
        gold_counter[normalize_answer(ex.answer)] += 1
        if len(samples) < 8:
            samples.append((ex.question, ex.answer, pred.strip(), bool(em)))
        if i % 50 == 0:
            log(f"  {i}/{n}  running exact={100 * exact / i:.1f}%")
    dt = time.perf_counter() - t0

    # Majority-class floor: always predict the single most frequent gold answer.
    mc_answer, mc_count = gold_counter.most_common(1)[0]

    print("\n=== TEXT-ONLY BASELINE (no image) ===")
    print(f"Questions             : {n}")
    print(f"Exact-match accuracy  : {100 * exact / n:5.1f}%   ({exact}/{n})   <-- headline")
    print(f"Relaxed accuracy      : {100 * relaxed / n:5.1f}%   (gold appears in answer)")
    print(f"Majority-class floor  : {100 * mc_count / n:5.1f}%   (always answer '{mc_answer}')")
    print(f"Throughput            : {n / dt:.1f} questions/s")

    print("\nPer-category exact accuracy:")
    for c in REASONING_CATEGORIES + ["other"]:
        e, r, tot = by_cat.get(c, [0, 0, 0])
        if tot:
            print(f"  {c:8}: {100 * e / tot:5.1f}%  ({e}/{tot})")

    print("\nSample predictions:")
    for q, gold, pred, ok in samples:
        print(f"  [{'OK' if ok else '  '}] gold={gold!r:18} pred={pred!r}")
        print(f"       Q: {q}")

    # Save a summary for the record / report.
    out = ensure_dir(Path(cfg["paths"]["outputs"]) / "baselines") / f"text_only_{args.split}.json"
    summary = {
        "split": args.split, "n": n,
        "exact_accuracy": exact / n, "relaxed_accuracy": relaxed / n,
        "majority_class": {"answer": mc_answer, "accuracy": mc_count / n},
        "per_category_exact": {c: {"correct": by_cat[c][0], "total": by_cat[c][2]}
                               for c in REASONING_CATEGORIES + ["other"] if by_cat[c][2]},
    }
    out.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved summary -> {out}")


if __name__ == "__main__":
    main()
