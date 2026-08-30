"""Step 11: build the accuracy-vs-epoch and val-loss-vs-epoch curves for a training run.

For every per-epoch checkpoint bridge_<enc>_<tag>_ep<N>.pt, this reads the validation loss
stored in the checkpoint and evaluates that checkpoint on the LOCKED eval set (via
scripts/07_evaluate.py, reusing its records), then assembles one CSV row per epoch:
epoch, val_loss, exact, VQA-soft, and per-category VQA-soft. It prints both curves and the
best epoch = the validation-loss MINIMUM (the selection rule for underfit/overfit).

    python scripts/11_epoch_curve.py --encoder ijepa --tag 500k --qids data/gqa/eval_2000_qids.json
"""
import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import REASONING_CATEGORIES  # noqa: E402
from src.utils import load_config  # noqa: E402

CATS = REASONING_CATEGORIES + ["other"]


def metrics_from_records(records):
    """Overall + per-category exact/VQA-soft (percentages) from a base-eval records list."""
    n = len(records)
    overall = {
        "exact": round(100 * sum(r["bridge_exact"] for r in records) / n, 1),
        "vqa": round(100 * sum(r["bridge_vqa"] for r in records) / n, 1),
    }
    percat = {}
    for cat in CATS:
        rows = [r for r in records if r["category"] == cat]
        percat[cat] = round(100 * sum(r["bridge_vqa"] for r in rows) / len(rows), 1) if rows else ""
    return overall, percat


def main():
    """Evaluate every epoch checkpoint, assemble the curves, and report the best epoch."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoder", default="ijepa")
    ap.add_argument("--tag", default="500k")
    ap.add_argument("--qids", default="data/gqa/eval_2000_qids.json")
    ap.add_argument("--eval-dir", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    eval_dir = Path(args.eval_dir or Path(cfg["paths"]["outputs"]) / "eval")

    # Collect per-epoch checkpoints, sorted by epoch number.
    pat = re.compile(rf"bridge_{args.encoder}_{args.tag}_ep(\d+)\.pt$")
    ckpts = sorted(((int(pat.search(p.name).group(1)), p) for p in ckpt_dir.glob(
        f"bridge_{args.encoder}_{args.tag}_ep*.pt") if pat.search(p.name)))
    if not ckpts:
        raise SystemExit(f"no checkpoints matching bridge_{args.encoder}_{args.tag}_ep*.pt in {ckpt_dir}")
    print(f"found {len(ckpts)} epoch checkpoints: {[e for e, _ in ckpts]}")

    rows = []
    for epoch, ckpt in ckpts:
        val_loss = torch.load(ckpt, map_location="cpu", weights_only=False).get("val_loss", float("nan"))
        records_path = eval_dir / f"{ckpt.stem}_records.json"
        if not records_path.exists():                 # evaluate this checkpoint on the locked set
            print(f"  eval epoch {epoch}: {ckpt.name} ...", flush=True)
            subprocess.run([sys.executable, "scripts/07_evaluate.py", "--checkpoint", str(ckpt),
                            "--qids", args.qids, "--num-examples", "0"], check=True)
        records = json.load(open(records_path))
        overall, percat = metrics_from_records(records)
        rows.append({"epoch": epoch, "val_loss": round(float(val_loss), 4),
                     "exact_%": overall["exact"], "vqa_soft_%": overall["vqa"],
                     **{f"{c}_vqa_%": percat[c] for c in CATS}})

    out = eval_dir / f"epoch_curve_{args.encoder}_{args.tag}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Report both curves and the selection rule (best = val-loss minimum).
    print(f"\n=== {args.encoder} {args.tag}: epoch curves (locked 2000-qid set) ===")
    print(f"{'epoch':>5}{'val_loss':>10}{'exact%':>9}{'VQA-soft%':>11}")
    for r in rows:
        print(f"{r['epoch']:>5}{r['val_loss']:>10.4f}{r['exact_%']:>9}{r['vqa_soft_%']:>11}")
    best = min(rows, key=lambda r: r["val_loss"])
    trend = "still dropping (UNDERFIT — train another epoch)" if best["epoch"] == rows[-1]["epoch"] \
        else "turned up after the minimum (best epoch found)"
    print(f"\nbest epoch by val-loss minimum: epoch {best['epoch']} "
          f"(val_loss {best['val_loss']:.4f}, VQA-soft {best['vqa_soft_%']}%)")
    print(f"val-loss trend: {trend}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
