"""Aggregate the Q-Former encoder gap across training seeds.

Answers exactly one question: how much of `CLIP_QF - IJEPA_QF` is training randomness?

WHAT THIS IS NOT. It does not replicate the interaction. The MLP cells have only seed 42, so the
per-seed quantity `E_QF(seed) - E_MLP(seed 42)` is a SENSITIVITY CURVE conditional on a fixed MLP
gap, and is labelled as such wherever it is printed. "Multi-seed replicated interaction" is not
available language until the MLP cells are replicated too (REPLICATION_MANIFEST.md section 2).

WHY SEEDS ARE NOT POOLED. Three seeds answering the same 3,000 questions are three correlated
models, not 9,000 independent observations. Pooling their predictions would shrink intervals by
sqrt(3) on the strength of an independence assumption that is plainly false. Question bootstraps
are therefore reported PER SEED, and across-seed spread is reported as a separate, explicitly
coarse descriptive statistic.

WHY THE ACROSS-SEED SPREAD IS COARSE. The standard deviation of three numbers is itself very
uncertain — its own sampling distribution is wide enough that the value should be read as an
order of magnitude, not a bound. No across-seed interval is computed for that reason.

    python scripts/33_seed_aggregate_qformer.py \\
        --seed 42 outputs/eval/confirm/bridge_{clip,ijepa}_w1qf_bf16_v1_ep5__*_records.json \\
        --seed 43 outputs/eval/confirm/bridge_{clip,ijepa}_w1qf_s43_v1_ep5__*_records.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

METRICS = ("exact_full", "exact", "vqa")
PRIMARY = "exact_full"
N_RESAMPLES = 10_000
SEED = 42
SIDE = "bridge"
# The seed-42 MLP encoder gap the sensitivity curve is conditional on (C24.1, exact_full).
MLP_GAP_SEED42 = 6.0


def load(path):
    """Records + meta from a corrected artifact, rejecting duplicate qids."""
    payload = json.loads(Path(path).read_text())
    recs, meta = ((payload["records"], payload.get("_meta", {}))
                  if isinstance(payload, dict) else (payload, {}))
    index = {}
    for r in recs:
        qid = str(r["qid"])
        if qid in index:
            raise SystemExit(f"FAIL duplicate qid {qid!r} in {path}")
        index[qid] = r
    return index, meta


def health(index, qids):
    """EOS, cap-hit and length: a cell that does not terminate is not comparable."""
    reasons = Counter(str(index[q].get(f"{SIDE}_stop_reason", "")) for q in qids)
    eos = sum(c for r, c in reasons.items() if r.startswith("eos"))
    cap = sum(bool(index[q].get(f"{SIDE}_cap_hit")) for q in qids)
    gen = [index[q].get(f"{SIDE}_n_generated") for q in qids]
    gen = [g for g in gen if g is not None]
    return (100.0 * eos / len(qids), 100.0 * cap / len(qids),
            sum(gen) / len(gen) if gen else float("nan"))


def gap_with_ci(clip_idx, ijepa_idx, qids, metric):
    """Paired gap, percentile interval and discordance for one seed under one metric."""
    a = np.array([bool(clip_idx[q][f"{SIDE}_{metric}"]) for q in qids])
    b = np.array([bool(ijepa_idx[q][f"{SIDE}_{metric}"]) for q in qids])
    n = len(qids)
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_RESAMPLES, n), dtype=np.int32)
    point = 100.0 * (int(a.sum()) - int(b.sum())) / n
    series = (a[idx].mean(axis=1) - b[idx].mean(axis=1)) * 100
    lo, hi = np.percentile(series, [2.5, 97.5])
    return {"clip_%": round(100.0 * float(a.mean()), 2),
            "ijepa_%": round(100.0 * float(b.mean()), 2),
            "gap": round(point, 2), "ci_lo": round(float(lo), 2), "ci_hi": round(float(hi), 2),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "both_correct": int((a & b).sum()), "clip_only": int((a & ~b).sum()),
            "ijepa_only": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum()),
            "discordant": int((a ^ b).sum()), "n": n}


def main() -> None:
    """Report each seed, then the across-seed spread."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", action="append", nargs=3, required=True,
                    metavar=("SEED", "CLIP_RECORDS", "IJEPA_RECORDS"),
                    help="repeatable: --seed 43 <clip records> <ijepa records>")
    ap.add_argument("--out", default=None, help="write the aggregate as JSON")
    args = ap.parse_args()

    seeds, qid_ref = {}, None
    for raw_seed, clip_path, ijepa_path in args.seed:
        s = int(raw_seed)
        if s in seeds:
            raise SystemExit(f"FAIL seed {s} supplied twice")
        c_idx, c_meta = load(clip_path)
        i_idx, i_meta = load(ijepa_path)
        if set(c_idx) != set(i_idx):
            raise SystemExit(f"FAIL seed {s}: the two arms are not on the same question set")
        for meta, name in ((c_meta, "clip"), (i_meta, "ijepa")):
            if meta.get("split") != "confirm_3000_qids":
                raise SystemExit(f"FAIL seed {s} {name}: split is {meta.get('split')!r}, "
                                 "the seed comparison must use the confirmatory slice")
            if meta.get("seed") not in (None, s):
                raise SystemExit(f"FAIL seed {s} {name}: checkpoint records seed "
                                 f"{meta.get('seed')} — a file is mislabelled")
        qids = sorted(c_idx)
        if qid_ref is None:
            qid_ref = qids
        elif qids != qid_ref:
            raise SystemExit(f"FAIL seed {s} answers a different question set than the others")
        seeds[s] = (c_idx, i_idx, qids)

    if len(seeds) < 2:
        raise SystemExit("FAIL need at least two seeds to say anything about seed variance")

    print(f"Q-Former encoder gap (CLIP - I-JEPA) across {len(seeds)} training seeds")
    print(f"n = {len(qid_ref)} confirmatory questions | {N_RESAMPLES:,} resamples | "
          f"bootstrap seed {SEED} | primary {PRIMARY}\n")

    per_seed = {}
    for s in sorted(seeds):
        c_idx, i_idx, qids = seeds[s]
        per_seed[s] = {m: gap_with_ci(c_idx, i_idx, qids, m) for m in METRICS}
        p = per_seed[s][PRIMARY]
        eos_c, cap_c, tok_c = health(c_idx, qids)
        eos_i, cap_i, tok_i = health(i_idx, qids)
        print(f"--- seed {s}")
        print(f"  CLIP {p['clip_%']:.2f}%   I-JEPA {p['ijepa_%']:.2f}%   "
              f"gap {p['gap']:+.2f} [{p['ci_lo']:+.2f}, {p['ci_hi']:+.2f}]"
              f"{'  *' if p['excludes_zero'] else ''}")
        print(f"  discordance: both {p['both_correct']}  clip-only {p['clip_only']}  "
              f"ijepa-only {p['ijepa_only']}  neither {p['both_wrong']}  "
              f"discordant {p['discordant']}")
        print(f"  validity: CLIP EOS {eos_c:.1f}% cap {cap_c:.1f}% {tok_c:.2f} tok | "
              f"I-JEPA EOS {eos_i:.1f}% cap {cap_i:.1f}% {tok_i:.2f} tok")
        for m in METRICS:
            if m != PRIMARY:
                q = per_seed[s][m]
                print(f"    {m:11} {q['gap']:+.2f} [{q['ci_lo']:+.2f}, {q['ci_hi']:+.2f}]"
                      f"{'  *' if q['excludes_zero'] else ''}")
        print()

    print(f"--- across seeds ({PRIMARY})")
    gaps = [per_seed[s][PRIMARY]["gap"] for s in sorted(seeds)]
    mean = statistics.mean(gaps)
    sd = statistics.stdev(gaps) if len(gaps) > 1 else float("nan")
    for s, g in zip(sorted(seeds), gaps):
        print(f"  seed {s}: {g:+.2f}"
              f"{'  *' if per_seed[s][PRIMARY]['excludes_zero'] else '   (CI includes 0)'}")
    print(f"  mean {mean:+.2f}   sd {sd:.2f}   min {min(gaps):+.2f}   max {max(gaps):+.2f}   "
          f"range {max(gaps) - min(gaps):.2f}")
    all_pos = all(g > 0 for g in gaps)
    all_neg = all(g < 0 for g in gaps)
    print(f"  sign consistent: {'yes, all positive' if all_pos else 'yes, all negative' if all_neg else 'NO — a seed reverses sign'}")
    n_excl = sum(per_seed[s][PRIMARY]["excludes_zero"] for s in seeds)
    print(f"  intervals excluding zero: {n_excl}/{len(seeds)}")
    stable = all(all(per_seed[s][m]["gap"] > 0 for m in METRICS) for s in seeds) or \
             all(all(per_seed[s][m]["gap"] < 0 for m in METRICS) for s in seeds)
    print(f"  metric direction stable within every seed: {'yes' if stable else 'NO'}")

    print(f"\n  {len(gaps)} seeds give only a COARSE estimate of training variance: the standard")
    print("  deviation of a handful of values is itself very uncertain, so read the spread as an")
    print("  order of magnitude, not a bound. Seeds are NOT pooled as independent questions —")
    print(f"  the same {len(qid_ref)} questions are answered {len(gaps)} times by correlated models.")

    print(f"\n--- sensitivity curve: E_QF(seed) - E_MLP(seed 42 = {MLP_GAP_SEED42:+.1f})")
    print("  CONDITIONAL ON THE FIXED SEED-42 MLP GAP — this is NOT a replicated interaction,")
    print("  because the MLP cells have only one training seed.")
    for s, g in zip(sorted(seeds), gaps):
        print(f"    seed {s}: {g - MLP_GAP_SEED42:+.2f}")

    if args.out:
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.write_text(json.dumps({
            "n_questions": len(qid_ref), "n_resamples": N_RESAMPLES, "bootstrap_seed": SEED,
            "primary_metric": PRIMARY, "per_seed": {str(k): v for k, v in per_seed.items()},
            "across_seed": {"gaps": gaps, "mean": mean, "sd": sd,
                            "min": min(gaps), "max": max(gaps),
                            "sign_consistent": all_pos or all_neg,
                            "n_intervals_excluding_zero": n_excl},
            "sensitivity_curve_conditional_on_seed42_mlp_gap": {
                str(s): g - MLP_GAP_SEED42 for s, g in zip(sorted(seeds), gaps)},
            "is_replicated_interaction": False,
        }, indent=2))
        print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
