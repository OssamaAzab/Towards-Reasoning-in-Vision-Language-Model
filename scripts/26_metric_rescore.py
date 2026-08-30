"""Re-score every saved evaluation under four answer-matching metrics.

Reads the per-question records written by scripts/07_evaluate.py and recomputes the
bridge and floor scores under four progressively more permissive matchers. Nothing
existing is modified: this writes one new CSV and never touches results_ci.csv, any
results.md, or the locked question manifests.

Why this exists. The headline metrics disagree about whether vision helps at all. A
generative bridge answers verbosely ("yes, 1 bird is in this image.") while the
text-only floor answers tersely and wrongly ("No birds."), so strict exact match
scores terseness rather than correctness, and containment matching over-credits
list-dumps and question-echoes. Reporting all four side by side makes the size of
that measurement artifact visible instead of leaving it inside one chosen number.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import exact_match, normalize_answer, vqa_match  # noqa: E402
from src.utils import load_config  # noqa: E402

# Clause boundary: standard sentence/list punctuation only. Deliberately does not
# special-case the missing-space artifact ("silverThe bench ...") — that is a
# decoding defect and hiding it inside the metric would flatter the result.
_CLAUSE_SPLIT = re.compile(r"[,.;:!?\n]")


def first_clause(text: str) -> str:
    """Return the text before the first sentence or list boundary."""
    return _CLAUSE_SPLIT.split(text.strip(), maxsplit=1)[0].strip()


def clause_match(pred: str, gold: str) -> bool:
    """True if the prediction's first clause exactly equals the gold answer."""
    return exact_match(first_clause(pred), gold)


def prefix_match(pred: str, gold: str) -> bool:
    """True if the normalized prediction begins with the normalized gold answer."""
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    return bool(gold_tokens) and pred_tokens[: len(gold_tokens)] == gold_tokens


METRICS = {
    "exact": exact_match,
    "clause": clause_match,
    "prefix": prefix_match,
    "soft": vqa_match,
}


def score_file(path: Path) -> dict | None:
    """Score one records.json under every metric, for both bridge and floor."""
    records = json.loads(path.read_text(encoding="utf-8"))
    if not records:
        return None
    if any(k not in records[0] for k in ("gold", "bridge", "floor")):
        return None

    n = len(records)
    row = {"stem": path.name.replace("_records.json", ""), "n": n}
    for side in ("bridge", "floor"):
        for name, fn in METRICS.items():
            hits = sum(fn(r[side], r["gold"]) for r in records)
            row[f"{side}_{name}"] = round(100.0 * hits / n, 2)
    for name in METRICS:
        row[f"effect_{name}"] = round(row[f"bridge_{name}"] - row[f"floor_{name}"], 2)
    signs = {name: (row[f"effect_{name}"] > 0) for name in METRICS}
    row["sign_flips"] = len(set(signs.values())) > 1
    return row


def main() -> None:
    """Re-score every records.json under all four metrics and write one CSV."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", default=None, help="directory of *_records.json")
    ap.add_argument("--out", default=None, help="output CSV path")
    args = ap.parse_args()

    cfg = load_config()
    eval_dir = Path(args.eval_dir or Path(cfg["paths"]["outputs"]) / "eval")
    out_path = Path(args.out or eval_dir / "metric_rescore.csv")

    paths = sorted(eval_dir.glob("*_records.json"))
    if not paths:
        raise SystemExit(f"no *_records.json under {eval_dir}")

    rows, skipped = [], []
    for path in paths:
        row = score_file(path)
        (rows.append(row) if row else skipped.append(path.name))

    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {out_path}")
    fields = list(rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    flips = [r for r in rows if r["sign_flips"]]
    print(f"scored {len(rows)} record files -> {out_path}")
    if skipped:
        print(f"skipped {len(skipped)} (missing gold/bridge/floor): {skipped[:5]}")
    print(f"vision-effect SIGN FLIPS across metrics: {len(flips)}/{len(rows)}")
    print()
    hdr = f"{'stem':52} {'exact':>7} {'clause':>7} {'prefix':>7} {'soft':>7}"
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["effect_soft"]):
        print(f"{r['stem'][:52]:52} {r['effect_exact']:+7.1f} {r['effect_clause']:+7.1f} "
              f"{r['effect_prefix']:+7.1f} {r['effect_soft']:+7.1f}")


if __name__ == "__main__":
    main()
