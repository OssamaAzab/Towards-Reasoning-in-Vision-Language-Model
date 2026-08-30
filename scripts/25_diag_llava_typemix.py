"""D3 diagnostic: LLaVA task-type composition of the DRAWN training slice (READ-ONLY).

LLaVA-Instruct-150K is a single JSON that concatenates its task types in blocks:
conversation (multi-turn) first, then detail + complex-reasoning (both single-turn).
The locked bridge run draws its LLaVA examples file-order-first-n (no shuffle), so a
draw of n_llava < the conversation block size yields an almost purely conversational
slice and never sees the longer detail/complex-reasoning answers.

This script quantifies that against the EXACT slice the trainer consumes: it loads the
same LLaVADataset (same GQA-leakage exclusion the trainer uses), labels each kept
conversation as `conversation` (multi-turn) vs `single_turn` (detail+complex, which are
2-turn), then splits the kept pool into the DRAWN first-n and the EXCLUDED remainder and
reports counts, %conversation, and answer word-length per group. Trains nothing.

    python scripts/25_diag_llava_typemix.py                 # locked 150K draw (n_llava=45000)
    python scripts/25_diag_llava_typemix.py --n-llava 150000  # the 500K draw (whole pool)
"""
import argparse
import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.llava import LLaVADataset  # noqa: E402
from src.utils import load_config  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics"


def group_stats(name, words, n_conversation, n_total_pool):
    """Assemble one group's row: size, %of-pool, %conversation, answer word-length stats."""
    n = len(words)
    return {
        "group": name,
        "n": n,
        "pct_of_kept_pool": round(100 * n / n_total_pool, 2) if n_total_pool else 0.0,
        "n_conversation": n_conversation,
        "pct_conversation": round(100 * n_conversation / n, 2) if n else 0.0,
        "mean_answer_words": round(st.mean(words), 1) if words else 0.0,
        "median_answer_words": round(st.median(words), 1) if words else 0.0,
        "pct_ge_50_words": round(100 * sum(w >= 50 for w in words) / n, 2) if n else 0.0,
    }


def main():
    """Recompute the drawn-vs-excluded LLaVA task-type composition; write llava_typemix.csv."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-llava", type=int, default=45000,
                    help="LLaVA examples the run draws (locked 150K run = 45000; 500K run = 150000)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    # Same construction the trainer uses -> same GQA-leakage exclusion, same file order.
    llava = LLaVADataset(cfg["llava"]["annotations"], cfg["llava"]["image_dir"],
                         cfg["llava"].get("exclude_ids"))
    n_pool = len(llava)
    n_draw = min(args.n_llava, n_pool)
    print(f"kept LLaVA pool: {n_pool:,} (excluded {llava.n_excluded:,} GQA-eval images); "
          f"drawing first {n_draw:,}")

    # Per kept conversation: is it multi-turn (conversation type) and how long is its answer?
    is_conv, ans_words = [], []
    for i in range(n_pool):
        item = llava._raw[i]                       # read-only access to the raw record
        is_conv.append(len(item["conversations"]) > 2)
        ans_words.append(len(llava.get(i).answer.split()))

    drawn = range(0, n_draw)
    excl = range(n_draw, n_pool)
    rows = [
        group_stats("DRAWN_first_n", [ans_words[i] for i in drawn],
                    sum(is_conv[i] for i in drawn), n_pool),
        group_stats("EXCLUDED_remainder", [ans_words[i] for i in excl],
                    sum(is_conv[i] for i in excl), n_pool),
        group_stats("WHOLE_kept_pool", ans_words, sum(is_conv), n_pool),
    ]

    cols = ["group", "n", "pct_of_kept_pool", "n_conversation", "pct_conversation",
            "mean_answer_words", "median_answer_words", "pct_ge_50_words"]
    out_csv = (OUT_DIR / "llava_typemix.csv" if args.n_llava == 45000
               else OUT_DIR / f"llava_typemix_{args.n_llava}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n  {'group':20s} {'n':>8} {'%pool':>7} {'%conv':>7} {'mean_w':>7} {'med_w':>6} {'%>=50w':>7}")
    for r in rows:
        print(f"  {r['group']:20s} {r['n']:>8} {r['pct_of_kept_pool']:>6}% "
              f"{r['pct_conversation']:>6}% {r['mean_answer_words']:>7} "
              f"{r['median_answer_words']:>6} {r['pct_ge_50_words']:>6}%")
    d = rows[0]
    print(f"\nSUMMARY (n_llava={n_draw}): DRAWN slice is {d['pct_conversation']}% conversation, "
          f"mean {d['mean_answer_words']} words; EXCLUDED remainder n={rows[1]['n']:,}, "
          f"mean {rows[1]['mean_answer_words']} words.")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
