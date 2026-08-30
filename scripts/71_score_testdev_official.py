#!/usr/bin/env python3
"""Score Test-Dev records with the OFFICIAL GQA evaluation script, alongside our own metric.

WHY BOTH. `exact_full` and the official Accuracy agree exactly on every corrected cell measured so
far, but "they agreed last time" is not a guarantee — it is a property of the model's output being
clean, which a new checkpoint could break. So this script computes both on every run and FAILS
LOUDLY if they diverge, instead of quietly reporting whichever was asked for.

WHAT IS SENT TO THE OFFICIAL SCRIPT. The RAW model output, untouched. The official scorer does
plain string equality with no normalisation, so passing it our normalised text would be scoring
our own preprocessing and calling it official.

WHAT IS NOT REPORTED. Validity and Plausibility need `{tier}_choices.json`, which GQA does not
publish for Test-Dev; without it the script scores every question False and prints `0.00%`. That
is an absence, not a result, so this script refuses to emit them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.eval.metrics import METRIC_VERSION, exact_full   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
VENDOR = ROOT / "scripts/vendor/gqa_eval.py"
VENDOR_SHA = "de6426214a886d6baf98b797ae1dbfd1ecdfed4a66fd647104f8febbc70caf9b"
QUESTIONS = ROOT / "data/gqa/questions/testdev_balanced_questions.json"

# Printed by the official script but meaningless without files GQA withholds for Test-Dev.
UNAVAILABLE = ("validity", "plausibility", "grounding")


def sha256(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def official_scores(records: list[dict], pred_field: str) -> dict:
    """Run the vendored official scorer on the raw predictions and parse its report."""
    if not VENDOR.is_file():
        sys.exit(
            "FAIL official GQA evaluator is not installed. Follow scripts/vendor/PROVENANCE.md "
            "to download and hash-verify scripts/vendor/gqa_eval.py"
        )
    if sha256(VENDOR) != VENDOR_SHA:
        sys.exit(f"FAIL vendored gqa_eval.py has been modified — refusing to call it 'official'\n"
                 f"  expected {VENDOR_SHA}\n  got      {sha256(VENDOR)}")
    allq = json.loads(QUESTIONS.read_text())
    missing = [r["qid"] for r in records if r["qid"] not in allq]
    if missing:
        sys.exit(f"FAIL {len(missing)} record qids are not Test-Dev questions, e.g. {missing[:5]}")

    with tempfile.TemporaryDirectory() as td:
        d = pathlib.Path(td)
        (d / "run_all_questions.json").write_text(
            json.dumps({r["qid"]: allq[r["qid"]] for r in records}))
        (d / "run_predictions.json").write_text(json.dumps(
            [{"questionId": r["qid"], "prediction": r[pred_field]} for r in records]))
        (d / "gqa_eval.py").write_bytes(VENDOR.read_bytes())
        proc = subprocess.run([sys.executable, "gqa_eval.py", "--tier", "run",
                               "--questions", "{tier}_all_questions.json",
                               "--predictions", "{tier}_predictions.json"],
                              cwd=d, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.exit(f"FAIL official scorer errored:\n{proc.stderr[-2000:]}")
    out = proc.stdout

    scores, per_type = {}, {}
    for key in ("Accuracy", "Binary", "Open", "Distribution"):
        m = re.search(rf"^{key}: ([\d.]+)", out, re.M)
        if not m:
            sys.exit(f"FAIL could not parse '{key}' from the official report")
        scores[key.lower()] = float(m.group(1))
    block = re.search(r"^Accuracy / structural type:\n((?:  .*\n)+)", out, re.M)
    if block:
        for line in block.group(1).strip().splitlines():
            m = re.match(r"\s*(\w+): ([\d.]+)% \((\d+) questions\)", line)
            if m:
                per_type[m.group(1)] = {"accuracy": float(m.group(2)), "n": int(m.group(3))}
    return {"scores": scores, "accuracy_per_structural_type": per_type,
            "not_computed": {k: "requires files GQA does not publish for Test-Dev" for k in UNAVAILABLE},
            "scorer": "scripts/vendor/gqa_eval.py", "scorer_sha256": VENDOR_SHA,
            "scorer_note": "raw string equality, no normalisation; raw model output was passed in",
            "report": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="a Test-Dev *_records.json produced by scripts/07_evaluate.py")
    ap.add_argument("--pred-field", default="bridge")
    ap.add_argument("--out", default=None, help="where to write the JSON summary")
    a = ap.parse_args()

    p = pathlib.Path(a.records)
    blob = json.loads(p.read_text())
    records = blob["records"] if isinstance(blob, dict) and "records" in blob else blob
    meta = blob.get("_meta", {}) if isinstance(blob, dict) else {}

    ours = 100.0 * sum(1 for r in records
                       if exact_full(r[a.pred_field], r["gold"])) / len(records)
    off = official_scores(records, a.pred_field)

    # The whole point of running both. A divergence means our normalisation started doing work,
    # and the number can no longer be called an official GQA accuracy without a caveat.
    delta = ours - off["scores"]["accuracy"]
    agree = abs(delta) < 5e-3

    summary = {
        "records": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
        "records_sha256": sha256(p), "n": len(records),
        "checkpoint": meta.get("checkpoint_tag"), "epoch": meta.get("checkpoint_epoch"),
        "encoder": meta.get("checkpoint_encoder"), "split": meta.get("split"),
        "evidence_layer": meta.get("evidence_layer"),
        "ours": {"metric": METRIC_VERSION, "exact_full": round(ours, 4)},
        "official": off,
        "agreement": {"delta_points": round(delta, 6), "identical": agree},
    }
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(summary, indent=1) + "\n")

    print(f"{p.name}")
    print(f"  n                     : {len(records):,}")
    print(f"  ours  exact_full      : {ours:.4f}%")
    print(f"  official Accuracy     : {off['scores']['accuracy']:.4f}%")
    print(f"  official Binary/Open  : {off['scores']['binary']:.2f}% / {off['scores']['open']:.2f}%")
    print(f"  official Distribution : {off['scores']['distribution']} (lower is better)")
    print(f"  agreement             : {'IDENTICAL' if agree else f'DIVERGED by {delta:+.4f} pts'}")
    if not agree:
        sys.exit("FAIL our metric and the official scorer disagree — do not report this as an "
                 "official GQA accuracy until the difference is explained")


if __name__ == "__main__":
    main()
