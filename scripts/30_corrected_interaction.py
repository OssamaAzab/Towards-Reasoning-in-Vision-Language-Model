"""The corrected connector x encoder matrix: paired gaps and the direct interaction.

Generalises scripts/27_metric_ci.py, which computed the same quantities but was hardcoded to
the legacy stems, the legacy metric family and the locked 2,000. This version takes the four
cells as arguments, defaults to the corrected metric, and adds the guards the legacy version
had no reason to carry.

THE FOUR CELLS

    encoder gaps      E_MLP = CLIP_MLP - IJEPA_MLP        (within the MLP connector)
                      E_QF  = CLIP_QF  - IJEPA_QF         (within the Q-Former connector)
    connector gaps    C_CLIP  = MLP_CLIP  - QF_CLIP       (within the CLIP encoder)
                      C_IJEPA = MLP_IJEPA - QF_IJEPA      (within the I-JEPA encoder)
    interaction       E_QF - E_MLP   ==   C_IJEPA - C_CLIP

The two forms of the interaction are algebraically identical; the script computes both and
refuses to report if they disagree, which catches a cell wired into the wrong slot.

WHY THE INTERACTION IS COMPUTED DIRECTLY. An interaction is NOT established by observing that
one marginal gap excludes zero and the other does not — two intervals can overlap heavily while
their difference is nowhere near significant, and vice versa. The difference of differences is
formed per question and bootstrapped as its own quantity.

ONE SHARED RESAMPLE MATRIX. All four cells answer the same questions, so a single index matrix
per metric is applied to every cell. Each resample keeps all four outcomes for a question
together, so question difficulty cancels out of every contrast instead of inflating it. This is
what makes the interaction's interval paired all the way through.

THE HARDWARE GUARD. The frozen text-only floor differs between RTX PRO 6000 (Blackwell) and
RTX A6000 (Ampere) on 24 of 500 greedy generations, worth ~0.4 points. Encoder gaps compare two
cells within one connector and are safe if that connector was scored on one architecture.
Connector gaps and the interaction cross the connectors, so an architecture split between them
injects a hardware offset directly into the headline quantity. Architecture is NOT recorded in
the records metadata (src/artifact.py:build_meta has no hardware field), so it cannot be
inferred here: it must be declared, and an undeclared or mismatched pairing refuses to emit the
cross-connector quantities unless --allow-hardware-mismatch marks the run exploratory.

    python scripts/30_corrected_interaction.py \\
        --clip-mlp  outputs/eval/tuning/<clip mlp>_records.json  \\
        --ijepa-mlp outputs/eval/tuning/<ijepa mlp>_records.json \\
        --clip-qf   outputs/eval/tuning/<clip qf>_records.json   \\
        --ijepa-qf  outputs/eval/tuning/<ijepa qf>_records.json  \\
        --arch-mlp blackwell --arch-qf blackwell --out outputs/eval/corrected_interaction.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import REASONING_CATEGORIES  # noqa: E402

# exact_full is primary: normalised exact match over the WHOLE generation. `exact` keeps the
# legacy first-line extraction and `vqa` is soft matching; both are diagnostics only, retained
# because the legacy interaction changed sign between the available metrics.
PRIMARY_METRIC = "exact_full"
METRICS = (PRIMARY_METRIC, "exact", "vqa")
N_RESAMPLES = 10_000
SEED = 42
SIDE = "bridge"
ANALYSIS_VERSION = "corrected_interaction/1.0.0"


def load_cell(path, *, name):
    """Load one cell's records + meta, rejecting duplicate qids."""
    payload = json.load(open(path))
    records, meta = ((payload["records"], payload.get("_meta", {}))
                     if isinstance(payload, dict) else (payload, {}))
    index = {}
    for rec in records:
        qid = str(rec["qid"])
        if qid in index:
            raise SystemExit(f"FAIL duplicate qid {qid!r} in {name} ({path}): "
                             "pairing would double-weight it")
        index[qid] = rec
    return index, meta


def align(cells):
    """Require all four cells on the IDENTICAL qid set, not merely an overlapping one."""
    sets = {n: set(idx) for n, (idx, _) in cells.items()}
    first = next(iter(sets))
    for name, s in sets.items():
        if s != sets[first]:
            only_a, only_b = len(sets[first] - s), len(s - sets[first])
            raise SystemExit(
                f"FAIL {name} and {first} are not on the same question set "
                f"({only_a} only in {first}, {only_b} only in {name}). The primary matrix "
                "requires identical questions in every cell.")
    return sorted(sets[first])


def check_provenance(cells):
    """Refuse to mix evidence layers, metric versions or splits inside one matrix."""
    fields = ("evidence_layer", "metric_version", "split")
    seen = {f: {} for f in fields}
    for name, (_, meta) in cells.items():
        for f in fields:
            seen[f].setdefault(meta.get(f, "UNKNOWN"), []).append(name)
    for f, groups in seen.items():
        if len(groups) > 1:
            detail = "; ".join(f"{v!r}: {', '.join(n)}" for v, n in groups.items())
            raise SystemExit(f"FAIL cells disagree on {f} — {detail}. Legacy and corrected "
                             "results are separate evidence layers and are never merged.")
    return {f: next(iter(g)) for f, g in seen.items()}


def available_metrics(cells, qids):
    """Metrics scored in EVERY cell. Legacy artifacts predate exact_full and only carry two."""
    probe = qids[0]
    present = [m for m in METRICS
               if all(f"{SIDE}_{m}" in idx[probe] for idx, _ in cells.values())]
    missing = [m for m in METRICS if m not in present]
    if not present:
        raise SystemExit(f"FAIL none of {METRICS} are scored in all cells")
    if missing:
        print(f"NOTE metrics absent from at least one cell, skipped: {', '.join(missing)}")
    if PRIMARY_METRIC not in present:
        print(f"WARNING the primary metric {PRIMARY_METRIC!r} is unavailable — every number "
              "below is a diagnostic, and the legacy interaction is known to change SIGN "
              "between `exact` and `vqa`, so no single one of them is the answer.")
    return present


def outcomes(index, qids, metric):
    """Per-question boolean outcome array for one cell under one metric."""
    return np.array([bool(index[q][f"{SIDE}_{metric}"]) for q in qids])


def ci(series, point):
    """Percentile interval around a point estimate computed from the raw counts."""
    lo, hi = np.percentile(series, [2.5, 97.5])
    return round(float(point), 1), round(float(lo), 1), round(float(hi), 1)


def gap(a, b, idx):
    """Paired point estimate and bootstrap series for (a - b), in percentage points."""
    point = 100.0 * (int(a.sum()) - int(b.sum())) / len(a)
    series = (a[idx].mean(axis=1) - b[idx].mean(axis=1)) * 100
    return point, series


def discordance(a, b):
    """Counts behind a paired contrast: agreement hides how much the arms actually differ."""
    return {"both_correct": int((a & b).sum()), "left_only": int((a & ~b).sum()),
            "right_only": int((~a & b).sum()), "both_wrong": int((~a & ~b).sum()),
            "discordant": int((a ^ b).sum())}


def union_analysis(clip, ijepa, qids, records_clip):
    """Complementarity: do the encoders fail on the same questions or different ones?

    Reported as a descriptive ceiling only. Choosing per question would need a selector, and
    a GQA-supervised selector would break the zero-shot design, so this is not a method.
    """
    n = len(qids)
    d = discordance(clip, ijepa)
    by_cat = Counter(records_clip[q]["category"] for i, q in enumerate(qids)
                     if clip[i] ^ ijepa[i])
    return {**d,
            "n": n,
            "clip_%": round(100.0 * int(clip.sum()) / n, 1),
            "ijepa_%": round(100.0 * int(ijepa.sum()) / n, 1),
            "union_%": round(100.0 * int((clip | ijepa).sum()) / n, 1),
            "disagreement_%": round(100.0 * d["discordant"] / n, 1),
            "discordant_by_category": dict(by_cat.most_common())}


def output_health(index, qids):
    """EOS termination and cap-hit rate: a cell that never stops is not comparable."""
    reasons = Counter(str(index[q].get(f"{SIDE}_stop_reason", "")) for q in qids)
    eos = sum(c for r, c in reasons.items() if r.startswith("eos"))
    cap = sum(bool(index[q].get(f"{SIDE}_cap_hit")) for q in qids)
    gen = [index[q].get(f"{SIDE}_n_generated") for q in qids]
    gen = [g for g in gen if g is not None]
    return {"eos_%": round(100.0 * eos / len(qids), 1),
            "cap_hit_%": round(100.0 * cap / len(qids), 1),
            "mean_tokens": round(sum(gen) / len(gen), 2) if gen else None}


def main() -> None:
    """Compute the corrected matrix, the four marginal gaps and the direct interaction."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for cell in ("clip-mlp", "ijepa-mlp", "clip-qf", "ijepa-qf"):
        ap.add_argument(f"--{cell}", required=True, help=f"{cell} records JSON")
    ap.add_argument("--arch-mlp", default=None,
                    help="GPU architecture the MLP cells were scored on (e.g. blackwell)")
    ap.add_argument("--arch-qf", default=None,
                    help="GPU architecture the Q-Former cells were scored on")
    ap.add_argument("--allow-hardware-mismatch", action="store_true",
                    help="EXPLORATORY: emit cross-connector quantities despite an "
                         "architecture split. They carry a hardware offset; label them.")
    ap.add_argument("--out", default=None, help="write all quantities as CSV")
    args = ap.parse_args()

    cells = {n: load_cell(getattr(args, n.replace("-", "_")), name=n)
             for n in ("clip-mlp", "ijepa-mlp", "clip-qf", "ijepa-qf")}
    qids = align(cells)
    prov = check_provenance(cells)
    n = len(qids)

    # ---- hardware gate: declared, never inferred ----
    cross_ok, arch_note = True, ""
    if args.arch_mlp is None or args.arch_qf is None:
        cross_ok = False
        arch_note = ("architecture not declared for both connectors (--arch-mlp/--arch-qf); "
                     "it is not recorded in the artifacts and will not be guessed")
    elif args.arch_mlp != args.arch_qf:
        cross_ok = False
        arch_note = (f"MLP cells scored on {args.arch_mlp!r} but Q-Former cells on "
                     f"{args.arch_qf!r}; the cross-connector quantities would carry a "
                     "hardware offset (~0.4 points on the frozen floor alone)")
    if not cross_ok and args.allow_hardware_mismatch:
        arch_note += "  [OVERRIDDEN: reported as EXPLORATORY, hardware-confounded]"
        cross_ok = True
        exploratory = True
    else:
        exploratory = False

    print(f"n = {n} shared qids | {N_RESAMPLES:,} resamples | seed {SEED}")
    print(f"evidence layer : {prov['evidence_layer']}")
    print(f"metric version : {prov['metric_version']}   primary = {PRIMARY_METRIC}")
    print(f"split          : {prov['split']}")
    print(f"hardware       : mlp={args.arch_mlp} qformer={args.arch_qf}"
          f"{'' if cross_ok and not exploratory else '  <-- ' + arch_note}")
    print()
    for name, (idx, meta) in cells.items():
        h = output_health(idx, qids)
        print(f"  {name:10} ep{meta.get('checkpoint_epoch')} {meta.get('checkpoint_tag')}"
              f"  EOS {h['eos_%']}%  cap-hit {h['cap_hit_%']}%  {h['mean_tokens']} tok")
    print()

    metrics = available_metrics(cells, qids)
    primary = PRIMARY_METRIC if PRIMARY_METRIC in metrics else None
    rows, union_out = [], {}
    for metric in metrics:
        rng = np.random.default_rng(SEED)
        idx_mat = rng.integers(0, n, size=(N_RESAMPLES, n), dtype=np.int32)
        o = {name: outcomes(index, qids, metric) for name, (index, _) in cells.items()}

        quantities, series = {}, {}
        quantities["E_MLP"], series["E_MLP"] = gap(o["clip-mlp"], o["ijepa-mlp"], idx_mat)
        quantities["E_QF"], series["E_QF"] = gap(o["clip-qf"], o["ijepa-qf"], idx_mat)
        quantities["C_CLIP"], series["C_CLIP"] = gap(o["clip-mlp"], o["clip-qf"], idx_mat)
        quantities["C_IJEPA"], series["C_IJEPA"] = gap(o["ijepa-mlp"], o["ijepa-qf"], idx_mat)

        inter_point = quantities["E_QF"] - quantities["E_MLP"]
        inter_series = series["E_QF"] - series["E_MLP"]
        # The identity E_QF - E_MLP == C_IJEPA - C_CLIP holds by construction; if it does not,
        # a cell is wired into the wrong slot and every number here is meaningless.
        alt_point = quantities["C_IJEPA"] - quantities["C_CLIP"]
        if abs(inter_point - alt_point) > 1e-9:
            raise SystemExit(f"FAIL interaction identity broken for {metric}: "
                             f"E_QF-E_MLP={inter_point:.6f} but "
                             f"C_IJEPA-C_CLIP={alt_point:.6f} — a cell is misassigned")
        quantities["INTERACTION"], series["INTERACTION"] = inter_point, inter_series

        CROSS = {"C_CLIP", "C_IJEPA", "INTERACTION"}
        star = lambda lo, hi: "  *" if (lo > 0 or hi < 0) else ""
        print(f"--- {metric}{'  (PRIMARY)' if metric == primary else '  (diagnostic)'}")
        for q in ("E_MLP", "E_QF", "C_CLIP", "C_IJEPA", "INTERACTION"):
            if q in CROSS and not cross_ok:
                print(f"  {q:12} WITHHELD — {arch_note}")
                rows.append({"metric": metric, "quantity": q, "n": n, "point": "",
                             "ci_lo": "", "ci_hi": "", "excludes_zero": "",
                             "status": "withheld_hardware_mismatch"})
                continue
            p, lo, hi = ci(series[q], quantities[q])
            status = ("exploratory_hardware_confounded"
                      if (q in CROSS and exploratory) else "ok")
            print(f"  {q:12} {p:+6.1f}  [{lo:+6.1f}, {hi:+6.1f}]{star(lo, hi)}"
                  f"{'   EXPLORATORY' if status != 'ok' else ''}")
            rows.append({"metric": metric, "quantity": q, "n": n, "point": p,
                         "ci_lo": lo, "ci_hi": hi,
                         "excludes_zero": lo > 0 or hi < 0, "status": status})

        # Discordance behind each encoder gap — a near-zero gap can still hide heavy disagreement.
        for conn, a, b in (("mlp", "clip-mlp", "ijepa-mlp"), ("qformer", "clip-qf", "ijepa-qf")):
            d = discordance(o[a], o[b])
            print(f"  {'discordance ' + conn:20} both {d['both_correct']}  "
                  f"clip-only {d['left_only']}  ijepa-only {d['right_only']}  "
                  f"neither {d['both_wrong']}  discordant {d['discordant']}")
            if metric == (primary or metrics[0]):
                union_out[conn] = union_analysis(o[a], o[b], qids, cells[a][0])
        print()

    if union_out:
        print(f"--- encoder complementarity ({primary or metrics[0]}, descriptive ceiling only)")
        for conn, u in union_out.items():
            print(f"  {conn:8} clip {u['clip_%']}%  ijepa {u['ijepa_%']}%  "
                  f"union {u['union_%']}%  disagreement {u['disagreement_%']}%")
            print(f"           discordant by category: {u['discordant_by_category']}")
        print("  Union is NOT achievable without a selector; a GQA-supervised selector would")
        print("  break the zero-shot design, so this is a ceiling, not a method.")

    if args.out:
        out = Path(args.out)
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        with out.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {out}")
        (out.with_suffix(".meta.json")).write_text(json.dumps({
            "analysis_version": ANALYSIS_VERSION, "n": n, "seed": SEED,
            "n_resamples": N_RESAMPLES, "primary_metric": primary, "metrics_reported": metrics,
            "arch_mlp": args.arch_mlp, "arch_qf": args.arch_qf,
            "hardware_matched": bool(cross_ok and not exploratory),
            "exploratory_override": exploratory,
            "union_analysis": union_out, **prov,
            "cells": {k: str(getattr(args, k.replace("-", "_"))) for k in cells},
        }, indent=2))


if __name__ == "__main__":
    main()
