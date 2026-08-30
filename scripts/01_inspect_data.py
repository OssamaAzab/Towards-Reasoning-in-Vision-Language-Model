"""Step 1: inspect the GQA balanced split — counts, type mix, answers, images.

The VQA equivalent of EDA. Run after downloading questions and `source env.sh`:
    python scripts/01_inspect_data.py --limit 200     # quick sample (proves it runs)
    python scripts/01_inspect_data.py                 # full split
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

# Make the repo root importable when run as `python scripts/01_inspect_data.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset, REASONING_CATEGORIES  # noqa: E402
from src.utils import load_config  # noqa: E402

BINARY_ANSWERS = {"yes", "no"}


def main() -> None:
    """Load the split, compute summary stats, and print a few sample triples."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="inspect only the first N questions (default: all)")
    ap.add_argument("--split", default="val_balanced",
                    help="which questions split key from config.gqa.questions")
    args = ap.parse_args()

    cfg = load_config()
    qpath = cfg["gqa"]["questions"][args.split]
    image_dirs = cfg["gqa"]["image_dirs"]

    ds = GQADataset(qpath, image_dirs)
    scope = "all" if args.limit is None else f"first {args.limit}"
    print(f"Loaded {len(ds):,} questions from {qpath}")
    print(f"Inspecting {scope}\n")

    cats, structural, answers = Counter(), Counter(), Counter()
    qlens: list[int] = []
    n_total = n_binary = missing_images = 0
    samples: list = []

    for ex in ds.examples(args.limit):
        n_total += 1
        cats[ds.category_of(ex)] += 1
        structural[ex.structural] += 1
        answers[ex.answer] += 1
        qlens.append(len(ex.question.split()))
        if ex.answer.lower() in BINARY_ANSWERS:
            n_binary += 1
        if ex.image_path is None:
            missing_images += 1
        if len(samples) < 5:
            samples.append(ex)

    pct = lambda x: f"{100 * x / n_total:.1f}%" if n_total else "n/a"

    print(f"Total inspected       : {n_total:,}")
    print(f"Unique answers        : {len(answers):,}")
    print(f"Binary (yes/no)       : {n_binary:,} ({pct(n_binary)})")
    print(f"Open-ended            : {n_total - n_binary:,} ({pct(n_total - n_binary)})")
    print(f"Avg question length   : {sum(qlens) / len(qlens):.1f} words")
    print(f"Images missing on disk: {missing_images:,} ({pct(missing_images)})")

    print("\nReasoning categories (RQ1):")
    for c in REASONING_CATEGORIES + ["other"]:
        print(f"  {c:8}: {cats.get(c, 0):6,} ({pct(cats.get(c, 0))})")

    print("\nRaw GQA structural types:")
    for t, c in structural.most_common():
        print(f"  {t or '<none>':10}: {c:6,} ({pct(c)})")

    print("\nTop 10 answers:")
    for a, c in answers.most_common(10):
        print(f"  {a:14}: {c:6,}")

    print("\n5 sample (question, answer, image) triples:")
    for ex in samples:
        status = "OK" if (ex.image_path and ex.image_path.exists()) else "MISSING"
        print(f"  [{ex.qid}] cat={ds.category_of(ex)}  structural={ex.structural}")
        print(f"      Q: {ex.question}")
        print(f"      A: {ex.answer}")
        print(f"      img[{status}]: {ex.image_path}")


if __name__ == "__main__":
    main()
