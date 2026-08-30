"""Step 2: visualize the GQA balanced split — saves charts + a sample image grid.

The picture half of the EDA (numbers live in 01_inspect_data.py). Writes four
PNGs to outputs/figures/. Run after `source env.sh`:
    python scripts/02_visualize_data.py --limit 500    # quick sample (proves it runs)
    python scripts/02_visualize_data.py                # full split

Figures produced (all under <outputs>/figures/):
    gqa_sample_grid.png      3x3 images captioned with their question + answer
    gqa_question_types.png   reasoning-category bar chart (RQ1 buckets)
    gqa_top_answers.png      top-20 answers
    gqa_question_length.png  question-length (in words) histogram
"""
import argparse
import random
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless cluster: render to file, never to a display.
import matplotlib.pyplot as plt
from PIL import Image

# Make the repo root importable when run as `python scripts/02_visualize_data.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset, REASONING_CATEGORIES  # noqa: E402
from src.utils import ensure_dir, load_config, set_seed  # noqa: E402

GRID_SIDE = 3  # 3x3 sample grid


def collect(ds: GQADataset, limit: int | None):
    """One pass over the split: gather category/answer/length stats + image pool."""
    cats, answers = Counter(), Counter()
    qlens: list[int] = []
    with_image: list = []  # examples whose image resolved on disk (grid candidates)
    for ex in ds.examples(limit):
        cats[ds.category_of(ex)] += 1
        answers[ex.answer] += 1
        qlens.append(len(ex.question.split()))
        if ex.image_path is not None:
            with_image.append(ex)
    return cats, answers, qlens, with_image


def plot_sample_grid(examples: list, out: Path) -> None:
    """Save a 3x3 grid of images captioned with their question and answer."""
    fig, axes = plt.subplots(GRID_SIDE, GRID_SIDE, figsize=(12, 12))
    for ax, ex in zip(axes.ravel(), examples):
        try:
            img = Image.open(ex.image_path).convert("RGB")
            ax.imshow(img)
        except Exception as e:  # unreadable image: show the reason, keep going
            ax.text(0.5, 0.5, f"[image error]\n{e}", ha="center", va="center",
                    fontsize=8, wrap=True)
        q = textwrap.fill(ex.question, width=42)
        ax.set_title(f"Q: {q}\nA: {ex.answer}", fontsize=8, loc="left")
        ax.axis("off")
    # Blank any unused cells (small samples).
    for ax in axes.ravel()[len(examples):]:
        ax.axis("off")
    fig.suptitle("GQA balanced — sample (image, question, answer) triples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_question_types(cats: Counter, n_total: int, out: Path) -> None:
    """Bar chart of the five RQ1 reasoning buckets plus 'other'."""
    labels = REASONING_CATEGORIES + ["other"]
    values = [cats.get(c, 0) for c in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel("questions")
    ax.set_title(f"GQA reasoning categories (n={n_total:,})")
    for b, v in zip(bars, values):
        pct = f"{100 * v / n_total:.1f}%" if n_total else "0%"
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n{pct}",
                ha="center", va="bottom", fontsize=8)
    ax.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_top_answers(answers: Counter, out: Path, top: int = 20) -> None:
    """Horizontal bar chart of the most common answers."""
    common = answers.most_common(top)
    labels = [a if a else "<blank>" for a, _ in common][::-1]
    values = [c for _, c in common][::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(labels, values, color="#55A868")
    ax.set_xlabel("count")
    ax.set_title(f"Top {len(common)} GQA answers")
    for i, v in enumerate(values):
        ax.text(v, i, f" {v:,}", va="center", fontsize=8)
    ax.margins(x=0.12)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def plot_question_length(qlens: list[int], out: Path) -> None:
    """Histogram of question length in words."""
    fig, ax = plt.subplots(figsize=(8, 5))
    hi = max(qlens) if qlens else 1
    ax.hist(qlens, bins=range(1, hi + 2), color="#C44E52", edgecolor="white")
    mean = sum(qlens) / len(qlens) if qlens else 0
    ax.axvline(mean, color="black", linestyle="--", linewidth=1,
               label=f"mean = {mean:.1f} words")
    ax.set_xlabel("question length (words)")
    ax.set_ylabel("questions")
    ax.set_title("GQA question-length distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    """Build the stats, then save the four figures to outputs/figures/."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N questions (default: all)")
    ap.add_argument("--split", default="val_balanced",
                    help="which questions split key from config.gqa.questions")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.get("seed", 42))  # reproducible grid sampling
    ds = GQADataset(cfg["gqa"]["questions"][args.split], cfg["gqa"]["image_dirs"])
    fig_dir = ensure_dir(Path(cfg["paths"]["outputs"]) / "figures")

    scope = "all" if args.limit is None else f"first {args.limit}"
    print(f"Loaded {len(ds):,} questions; visualizing {scope}.")

    cats, answers, qlens, with_image = collect(ds, args.limit)
    n_total = sum(cats.values())
    print(f"Inspected {n_total:,} questions; {len(with_image):,} have an image on disk.")

    k = min(GRID_SIDE * GRID_SIDE, len(with_image))
    sample = random.sample(with_image, k) if with_image else []
    if k < GRID_SIDE * GRID_SIDE:
        print(f"  note: only {k} images available for the {GRID_SIDE}x{GRID_SIDE} grid.")

    saved = []
    if sample:
        plot_sample_grid(sample, fig_dir / "gqa_sample_grid.png")
        saved.append("gqa_sample_grid.png")
    plot_question_types(cats, n_total, fig_dir / "gqa_question_types.png")
    plot_top_answers(answers, fig_dir / "gqa_top_answers.png")
    plot_question_length(qlens, fig_dir / "gqa_question_length.png")
    saved += ["gqa_question_types.png", "gqa_top_answers.png", "gqa_question_length.png"]

    print(f"\nSaved {len(saved)} figure(s) to {fig_dir}:")
    for name in saved:
        print(f"  {name}")


if __name__ == "__main__":
    main()
