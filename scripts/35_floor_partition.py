"""Partition eval questions by what the text-only floor and the bridge each got right.

Aggregate accuracy hides two populations. On the confirmatory slice the frozen text-only
floor already answers ~29% of questions, so a connector contrast measured over the whole
slice mixes in questions this floor could already do. The floor and the bridge also
disagree in *both* directions: there is a stable subset the floor gets right and the
bridge gets wrong.

Four cells per arm, from the floor/bridge correctness pair already stored per record:

    floor  bridge
      0      1     BRIDGE-ONLY   - bridge right where this floor was wrong
      1      0     DISPLACED     - floor right, bridge wrong
      1      1     BOTH RIGHT
      0      0     BOTH WRONG

NAMING IS DELIBERATELY LITERAL. `floor_wrong` means *this particular frozen text-only
system* was wrong; it does not establish that the question requires vision, so the
floor-incorrect subset is NOT a "vision-sensitive" set and is not named one. Likewise
`floor_right, bridge_wrong` is an answer displaced by the bridge system under the primary
metric; it is not evidence that "the image destroyed" anything. Both would be latent
properties inferred from one observed outcome.

Contrasts restricted to the floor-incorrect subset are a conditional analysis on an
outcome-defined subset, not a cleaner estimate of the same estimand.

Read-only over existing eval records. No model, no GPU, no new evaluation.

Status: exploratory/descriptive. This partition was defined after the primary endpoint
was seen, so it is a secondary analysis and never a confirmatory test or a selection
surface (REPLICATION_MANIFEST section 11.5).
"""
from collections import defaultdict
from pathlib import Path
import argparse
import glob
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BOOT, SEED = 10000, 42


def load(stem, slice_tag):
    """QID -> (bridge_correct, floor_correct, category) for exactly one evaluated cell."""
    if "locked" in slice_tag or "eval_2000" in slice_tag:
        raise SystemExit(f"FAIL {slice_tag!r} names the locked endpoint; it is spent and this "
                         f"exploratory partition may not be run against it")
    hits = sorted(glob.glob(str(ROOT / "outputs" / "eval" / "**" /
                                f"{stem}*{slice_tag}*records.json"), recursive=True))
    if len(hits) != 1:
        raise SystemExit(f"FAIL expected exactly one record file for {stem!r} on {slice_tag!r}, "
                         f"found {len(hits)}: taking the first would silently pick an arm")
    blob = json.load(open(hits[0]))
    meta, recs = blob.get("_meta", {}), blob["records"]
    layer = meta.get("evidence_layer")
    if layer != "corrected_chatml_v1_eos":
        raise SystemExit(f"FAIL {stem} is evidence layer {layer!r}; layers are never merged")
    for field in ("bridge_exact_full", "floor_exact_full", "category"):
        if field not in recs[0]:
            raise SystemExit(f"FAIL record schema lacks {field!r} — a partition built on a "
                             f"missing field would silently compare None to None")
    qids = [r["qid"] for r in recs]
    if len(set(qids)) != len(qids):
        raise SystemExit(f"FAIL {stem} has duplicate QIDs; a dict build would silently drop rows")
    return {r["qid"]: (bool(r["bridge_exact_full"]), bool(r["floor_exact_full"]),
                       r["category"]) for r in recs}


def boot_diff(a, b, idx):
    """Paired point estimate and 95% interval for a-b, in points, on one resample matrix."""
    v = a.astype(float) - b.astype(float)
    draws = v[idx].mean(1) * 100
    return v.mean() * 100, np.percentile(draws, 2.5), np.percentile(draws, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slice", default="confirm_3000_qids",
                    help="eval slice tag; the locked set is spent and stays exploratory")
    ap.add_argument("--stems", nargs="+", required=True, help="checkpoint stems to partition")
    ap.add_argument("--contrasts", nargs="*", default=[],
                    help="pairs 'stemA:stemB' to re-estimate on the floor-incorrect subset")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cells = {s: load(s, args.slice) for s in args.stems}
    keysets = {s: frozenset(c) for s, c in cells.items()}
    if len(set(keysets.values())) != 1:
        raise SystemExit("FAIL the arms do not cover identical QID sets; a silent intersection "
                         "would compare arms on different questions")
    qids = sorted(next(iter(keysets.values())))
    if not qids:
        raise SystemExit("FAIL the named cells share no QIDs")
    cat_labels = {s: tuple(c[q][2] for q in qids) for s, c in cells.items()}
    if len(set(cat_labels.values())) != 1:
        raise SystemExit("FAIL category labels differ across arms for the same QIDs")

    # The floor is the same text-only run in every arm; if it is not, the partition is
    # not comparable across arms and the whole table is meaningless.
    floors = {s: tuple(c[q][1] for q in qids) for s, c in cells.items()}
    if len({tuple(f) for f in floors.values()}) != 1:
        raise SystemExit("FAIL floor differs across arms — the partition is not comparable")
    floor = np.array(floors[args.stems[0]])

    rng = np.random.default_rng(SEED)
    idx_all = rng.integers(0, len(qids), size=(BOOT, len(qids)))
    fi = ~floor                                  # floor-incorrect subset (outcome-defined)
    rng_fi = np.random.default_rng(SEED)
    idx_fi = rng_fi.integers(0, int(fi.sum()), size=(BOOT, int(fi.sum())))

    print(f"slice={args.slice}  n={len(qids)}  floor-incorrect={int(fi.sum())} "
          f"({100 * fi.mean():.1f}%)  floor-correct={int(floor.sum())} "
          f"({100 * floor.mean():.1f}%)\n")

    table = {}
    print(f"{'arm':<28}{'BOTHRIGHT':>11}{'BRIDGEONLY':>12}{'DISPLACED':>11}{'BOTHWRONG':>11}{'acc|floor-inc':>15}")
    for s in args.stems:
        b = np.array([cells[s][q][0] for q in qids])
        cell = {"both_right": float((floor & b).mean() * 100),
                "bridge_only": float(((~floor) & b).mean() * 100),
                "displaced": float((floor & (~b)).mean() * 100),
                "both_wrong": float(((~floor) & (~b)).mean() * 100),
                "acc_all": float(b.mean() * 100),
                "acc_floor_incorrect": float(b[fi].mean() * 100)}
        table[s] = cell
        print(f"{s[:27]:<28}{cell['both_right']:>10.1f}%{cell['bridge_only']:>11.1f}%"
              f"{cell['displaced']:>10.1f}%{cell['both_wrong']:>10.1f}%"
              f"{cell['acc_floor_incorrect']:>14.2f}%")

    contrasts = {}
    if args.contrasts:
        print(f"\n{'contrast':<34}{'all questions':>26}{'floor-incorrect only':>28}")
        for pair in args.contrasts:
            x, y = pair.split(":")
            bx = np.array([cells[x][q][0] for q in qids])
            by = np.array([cells[y][q][0] for q in qids])
            a = boot_diff(bx, by, idx_all)
            v = boot_diff(bx[fi], by[fi], idx_fi)
            contrasts[pair] = {"all": a, "floor_incorrect": v}
            print(f"{pair[:33]:<34}{a[0]:>+8.2f} [{a[1]:+.2f},{a[2]:+.2f}]"
                  f"{v[0]:>+11.2f} [{v[1]:+.2f},{v[2]:+.2f}]")

    per_cat = defaultdict(dict)
    cats = sorted({cells[args.stems[0]][q][2] for q in qids})
    for s in args.stems:
        b = np.array([cells[s][q][0] for q in qids])
        for c in cats:
            m = np.array([cells[s][q][2] == c for q in qids])
            per_cat[c][s] = {"n": int(m.sum()),
                             "displaced_pct": float((floor & (~b) & m).sum() / m.sum() * 100)}

    out = {"slice": args.slice, "n": len(qids), "seed": SEED, "bootstrap": BOOT,
           "n_floor_incorrect": int(fi.sum()), "arms": table,
           "contrasts": {k: {kk: list(vv) for kk, vv in v.items()}
                         for k, v in contrasts.items()},
           "displaced_by_category": {c: per_cat[c] for c in cats},
           "status": "exploratory/descriptive; defined post-hoc; never a selection surface"}
    dest = Path(args.out or ROOT / "outputs" / "analysis" /
                f"floor_partition__{args.slice}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
