"""Metric-sensitivity analysis with paired bootstrap CIs, from saved prediction records.

Re-scores completed evaluations under a ladder of answer-matching rules and reports which
conclusions survive the choice of rule. CPU only: it reads *_records.json and never loads a
model, touches a checkpoint, or regenerates a prediction.

WHAT THIS DOES AND DOES NOT FIX
    Fixes (scoring): the choice of matcher, and the asymmetric first-line truncation in
    src/eval/metrics.normalize_answer, which silently gives whichever side answers in
    multiple lines a free answer-extraction step the other side never gets.
    Does NOT fix (generation): the bridge is never trained to emit EOS, so predictions run
    to max_new_tokens and are verbose. No matcher removes that. Every number here still
    describes a model with that defect; a fairer matcher only stops the defect from being
    silently charged to one arm.

THE METRIC LADDER, strict to permissive. Every rule is applied identically to the bridge
and to the text-only floor.
    exact_first_line  the shipped exact_match: normalize_answer() keeps only the first
                      line, so it is an extraction rule as well as a matcher.
    exact_full        GQA-official-style: identical normalisation MINUS the first-line
                      truncation. Comparing it against exact_first_line isolates exactly
                      how much that truncation is worth to each arm.
    clause            first clause (split on , . ; : ! ? newline) then exact.
    prefix            normalised prediction starts with the normalised gold.
    soft              legacy vqa_match: exact, or yes/no polarity from the first polarity
                      token, or gold appearing as a contiguous token span (containment).

NOT IMPLEMENTED, deliberately: the official VQAv2 accuracy min(#humans/3, 1). It needs ten
annotator answers per question; GQA ships one gold answer, so the statistic is undefined
here. It is reported as unsupported rather than approximated.

Bootstrap: 10,000 resamples, seed 42, percentile intervals — the convention every reported
CI in this project already uses. Within one eval set and one metric, ALL quantities share a
single resample matrix, so run-vs-run contrasts and vision effects are paired throughout.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from importlib import import_module  # noqa: E402

from src.eval.metrics import _ARTICLES, _PUNCT, exact_match, vqa_match  # noqa: E402

_rescore = import_module("26_metric_rescore")
clause_match, prefix_match = _rescore.clause_match, _rescore.prefix_match

N_RESAMPLES = 10_000
SEED = 42


def normalize_full(text: str) -> str:
    """Normalise WITHOUT the first-line truncation (GQA-official-style)."""
    text = text.strip().lower().translate(_PUNCT)
    return " ".join(t for t in text.split() if t not in _ARTICLES)


def exact_full_match(pred: str, gold: str) -> bool:
    """Exact match on the whole prediction, not just its first line."""
    return normalize_full(pred) == normalize_full(gold)


METRICS = {
    "exact_first_line": exact_match,
    "exact_full": exact_full_match,
    "clause": clause_match,
    "prefix": prefix_match,
    "soft": vqa_match,
}

# Phase-2 cluster cells only. Never mixed with Phase-1 local runs (the F-20 offset stands).
CONTRASTS = [
    ("rq2_clip_minus_ijepa__mlp_150k", "bridge_clip_150k_mlp_anchor_ep3",
     "bridge_ijepa_150k_mlp_anchor_ep3"),
    ("rq2_clip_minus_dinov2__mlp_150k", "bridge_clip_150k_mlp_anchor_ep3",
     "bridge_dinov2_150k_mlp_ep3"),
    ("rq2_dinov2_minus_ijepa__mlp_150k", "bridge_dinov2_150k_mlp_ep3",
     "bridge_ijepa_150k_mlp_anchor_ep3"),
    ("connector_mlp_minus_qformer__dinov2_150k", "bridge_dinov2_150k_mlp_ep3",
     "bridge_dinov2_150k_ep3"),
    ("data_500k_minus_150k__ijepa_mlp", "bridge_ijepa_500k_mlp_ep3",
     "bridge_ijepa_150k_mlp_anchor_ep3"),
    ("schedule_tuned_minus_anchor__clip_mlp", "bridge_clip_150k_mlp_tuned_ep3",
     "bridge_clip_150k_mlp_anchor_ep3"),
    ("schedule_tuned_minus_anchor__ijepa_mlp", "bridge_ijepa_150k_mlp_tuned_ep3",
     "bridge_ijepa_150k_mlp_anchor_ep3"),
]


def load_aligned(eval_dir: Path, suffix: str) -> tuple[list, dict]:
    """Load every records file in a directory, aligned on one shared qid ordering."""
    tables = {}
    for path in sorted(eval_dir.glob("*_records.json")):
        stem = path.name.replace(f"{suffix}_records.json", "").replace("_records.json", "")
        tables[stem.rstrip("_")] = {r["qid"]: r for r in json.loads(path.read_text())}
    if not tables:
        raise SystemExit(f"no *_records.json under {eval_dir}")
    key_sets = [set(t) for t in tables.values()]
    if any(k != key_sets[0] for k in key_sets):
        raise SystemExit(f"runs in {eval_dir} are not on one shared qid set")
    return sorted(key_sets[0]), tables


def outcomes(table: dict, qids: list, field: str, fn) -> np.ndarray:
    """Per-question boolean outcome array for one run under one matcher."""
    return np.array([bool(fn(table[q][field], table[q]["gold"])) for q in qids])


def interval(series: np.ndarray, point: float) -> tuple[float, float, float]:
    """Percentile interval around a point estimate computed from the raw counts."""
    lo, hi = np.percentile(series, [2.5, 97.5])
    return round(point, 1), round(lo, 1), round(hi, 1)


def analyse(eval_dir: Path, suffix: str, label: str, out_dir: Path) -> dict:
    """Score every run and contrast under every metric; write two CSVs; return a summary."""
    qids, tables = load_aligned(eval_dir, suffix)
    n = len(qids)
    runs_csv = out_dir / f"metric_sensitivity_runs_{label}.csv"
    contrasts_csv = out_dir / f"metric_sensitivity_contrasts_{label}.csv"
    for path in (runs_csv, contrasts_csv):
        if path.exists():
            raise SystemExit(f"refusing to overwrite existing {path}")

    run_rows, contrast_rows = [], []
    print(f"\n{'=' * 100}\n{label}: n = {n} shared qids, {len(tables)} runs, "
          f"{N_RESAMPLES:,} resamples, seed {SEED}\n{'=' * 100}")

    for metric, fn in METRICS.items():
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(N_RESAMPLES, n), dtype=np.int32)

        bridge_out, floor_out = {}, {}
        for stem, table in tables.items():
            bridge_out[stem] = outcomes(table, qids, "bridge", fn)
            floor_out[stem] = outcomes(table, qids, "floor", fn)

        for stem in sorted(tables):
            b, f = bridge_out[stem], floor_out[stem]
            eff_series = (b[idx].mean(axis=1) - f[idx].mean(axis=1)) * 100
            point = 100.0 * (int(b.sum()) - int(f.sum())) / n
            p, lo, hi = interval(eff_series, point)
            run_rows.append({
                "eval_set": label, "metric": metric, "stem": stem, "n": n,
                "bridge_pct": round(100.0 * int(b.sum()) / n, 2),
                "floor_pct": round(100.0 * int(f.sum()) / n, 2),
                "vision_effect": p, "ci_lo": lo, "ci_hi": hi,
                "excludes_zero": bool(lo > 0 or hi < 0),
            })

        for name, a_stem, b_stem in CONTRASTS:
            if a_stem not in tables or b_stem not in tables:
                continue
            a, b = bridge_out[a_stem], bridge_out[b_stem]
            series = (a[idx].mean(axis=1) - b[idx].mean(axis=1)) * 100
            point = 100.0 * (int(a.sum()) - int(b.sum())) / n
            p, lo, hi = interval(series, point)
            contrast_rows.append({
                "eval_set": label, "metric": metric, "contrast": name, "n": n,
                "point": p, "ci_lo": lo, "ci_hi": hi,
                "excludes_zero": bool(lo > 0 or hi < 0),
            })

    for path, rows in ((runs_csv, run_rows), (contrasts_csv, contrast_rows)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {runs_csv} ({len(run_rows)} rows)")
    print(f"wrote {contrasts_csv} ({len(contrast_rows)} rows)")
    return {"runs": run_rows, "contrasts": contrast_rows, "n": n, "label": label}


def report_stability(summary: dict) -> None:
    """Print which contrasts change sign or significance across the metric ladder."""
    label, rows = summary["label"], summary["contrasts"]
    by_contrast: dict[str, list] = {}
    for r in rows:
        by_contrast.setdefault(r["contrast"], []).append(r)

    print(f"\n--- {label}: stability across the metric ladder ---")
    hdr = f"{'contrast':44}" + "".join(f"{m:>18}" for m in METRICS) + "   verdict"
    print(hdr)
    print("-" * len(hdr))
    for name, group in by_contrast.items():
        ordered = [next(g for g in group if g["metric"] == m) for m in METRICS]
        cells = "".join(f"{g['point']:+8.1f}{'*' if g['excludes_zero'] else ' ':1}"
                        f"{'':9}" for g in ordered)
        signs = {g["point"] > 0 for g in ordered if g["point"] != 0}
        sigs = {g["excludes_zero"] for g in ordered}
        if len(signs) > 1:
            verdict = "SIGN FLIPS"
        elif len(sigs) > 1:
            verdict = "significance changes"
        else:
            verdict = "stable"
        print(f"{name:44}{cells}   {verdict}")
    print("  * = 95% CI excludes zero")


def main() -> None:
    """Run the metric-sensitivity analysis on each eval set, separately."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="outputs/eval_phase2_rescore")
    args = ap.parse_args()

    root = Path(args.root)
    print(__doc__.split("WHAT THIS DOES")[0].strip())
    print("\nUNSUPPORTED METRIC: official VQAv2 accuracy min(#humans/3, 1) — needs ten "
          "annotator answers per question; GQA provides one gold. Not approximated.")

    summaries = []
    # tune-500 FIRST and reported first: it is the screening set. locked-2000 is reported
    # for trajectory only and must never be used to select a metric or a configuration.
    for sub, suffix, label in (("tune500", "_tune500", "tune500"),
                               ("locked2000", "", "locked2000")):
        d = root / sub
        if not d.is_dir():
            print(f"skipping missing {d}")
            continue
        summaries.append(analyse(d, suffix, label, root))

    for s in summaries:
        report_stability(s)

    print("\nSELECTION RULE: tune-500 is the screening set. locked-2000 is reported for "
          "trajectory only and was NOT used to choose a metric or a configuration.")


if __name__ == "__main__":
    main()
