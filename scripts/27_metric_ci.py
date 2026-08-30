"""Bootstrap CIs for the RQ2 gaps and the connector x encoder interaction, per metric.

Answers one question: does the reported connector-dependent RQ2 gap survive a change
of answer-matching metric? Section 8.26 reports the CLIP-minus-I-JEPA gap as large at
the Q-Former (+6.9) and small at the MLP (+2.0) on VQA-soft. scripts/26_metric_rescore.py
showed that ordering inverts under stricter matching, so the interaction needs the same
uncertainty treatment the headline numbers already have.

All four runs answer the identical locked 2,000 questions, so every contrast here is
computed on ONE shared set of resample indices — the encoder gaps and their difference
are paired all the way through, exactly as scripts/12_bootstrap_ci.py pairs a model
against its comparator. 10,000 resamples, seed 42, percentile intervals, matching the
convention every reported CI already uses.

Writes a new CSV. Never touches results_ci.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.metrics import exact_match, vqa_match  # noqa: E402
from src.utils import load_config  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_rescore = import_module("26_metric_rescore")
clause_match, prefix_match = _rescore.clause_match, _rescore.prefix_match

N_RESAMPLES = 10_000
SEED = 42

METRICS = {
    "exact": exact_match,
    "clause": clause_match,
    "prefix": prefix_match,
    "vqa_soft": vqa_match,
}

# (label, CLIP stem, I-JEPA stem) — the two connector arms whose gap difference is
# the reported publication angle.
ARMS = [
    ("qformer", "bridge_clip_150k_3ep", "bridge_ijepa_150k_3ep"),
    ("mlp", "bridge_clip_150k_mlp_ep3", "bridge_ijepa_150k_mlp_ep3"),
]


def load_aligned(eval_dir: Path, stems: list[str]) -> tuple[list, dict]:
    """Load several runs and align them on a common, identically-ordered qid list."""
    tables = {}
    for stem in stems:
        path = eval_dir / f"{stem}_records.json"
        if not path.is_file():
            raise SystemExit(f"missing records: {path}")
        tables[stem] = {r["qid"]: r for r in json.loads(path.read_text())}
    key_sets = [set(t) for t in tables.values()]
    if any(k != key_sets[0] for k in key_sets):
        raise SystemExit("runs are not on the same locked qid set")
    qids = sorted(key_sets[0])
    return qids, tables


def outcomes(table: dict, qids: list, field: str, fn) -> np.ndarray:
    """Per-question boolean outcome array for one run under one matcher."""
    return np.array([bool(fn(table[q][field], table[q]["gold"])) for q in qids])


def ci(series: np.ndarray, point: float) -> tuple[float, float, float]:
    """Percentile interval around a point estimate computed from raw counts."""
    lo, hi = np.percentile(series, [2.5, 97.5])
    return round(point, 1), round(lo, 1), round(hi, 1)


def main() -> None:
    """Bootstrap the two RQ2 gaps and their difference under every metric."""
    eval_dir = Path(load_config()["paths"]["outputs"]) / "eval"
    stems = [s for _, a, b in ARMS for s in (a, b)]
    qids, tables = load_aligned(eval_dir, stems)
    n = len(qids)

    out_path = eval_dir / "metric_ci_interaction.csv"
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite existing {out_path}")

    rows = []
    print(f"n = {n} shared locked qids | {N_RESAMPLES:,} resamples | seed {SEED}\n")
    for metric, fn in METRICS.items():
        # ONE shared resample matrix per metric: every arm and both encoders are
        # paired on the same questions, so the interaction inherits that pairing.
        rng = np.random.default_rng(SEED)
        idx = rng.integers(0, n, size=(N_RESAMPLES, n), dtype=np.int32)

        gap_series, gap_point = {}, {}
        for arm, clip_stem, ijepa_stem in ARMS:
            c = outcomes(tables[clip_stem], qids, "bridge", fn)
            i = outcomes(tables[ijepa_stem], qids, "bridge", fn)
            gap_series[arm] = (c[idx].mean(axis=1) - i[idx].mean(axis=1)) * 100
            gap_point[arm] = 100.0 * (int(c.sum()) - int(i.sum())) / n
            p, lo, hi = ci(gap_series[arm], gap_point[arm])
            rows.append({"metric": metric, "quantity": f"rq2_gap_{arm}", "n": n,
                         "point": p, "ci_lo": lo, "ci_hi": hi,
                         "excludes_zero": lo > 0 or hi < 0})
            print(f"  {metric:9} rq2_gap_{arm:8} {p:+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]"
                  f"{'  *' if (lo > 0 or hi < 0) else ''}")

        inter = gap_series["qformer"] - gap_series["mlp"]
        p, lo, hi = ci(inter, gap_point["qformer"] - gap_point["mlp"])
        rows.append({"metric": metric, "quantity": "interaction_qformer_minus_mlp", "n": n,
                     "point": p, "ci_lo": lo, "ci_hi": hi,
                     "excludes_zero": lo > 0 or hi < 0})
        print(f"  {metric:9} {'INTERACTION':17} {p:+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]"
              f"{'  * EXCLUDES ZERO' if (lo > 0 or hi < 0) else '  (includes zero)'}\n")

    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
