"""Data-coverage audit for the visual-information audit. CPU only. THE GATE BEFORE ANY GPU RUN.

    python scripts/50_visual_audit_coverage.py

WHY THIS RUNS FIRST. Conditions 5 and 6 ("relevant-object masked", "matched irrelevant-object
masked") only mean anything where GQA's own annotations name a referent that exists in the scene
graph with a usable box, AND the same image offers a size-matched non-referent. Without the
second, the two conditions differ in region size as well as relevance, and any gap between them
is uninterpretable. This script measures how many tuning questions clear that bar, reports the
reason for every rejection, and writes the usable subset. It never repairs a question.

WHAT IT DOES NOT DO. It does not touch the confirmatory 3,000, the locked 2,000 or the objective
4,000. It loads no model and requests no GPU.

A LOW COVERAGE NUMBER IS A RESULT, NOT A BLOCKER TO BE TUNED AWAY. If too few questions survive,
the honest response is to report that the masking conditions cannot be run defensibly on this
slice — not to widen AREA_TOL until the number looks acceptable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.visual_audit import AREA_TOL, mask_pair, probe_labels  # noqa: E402
from src.utils import load_config  # noqa: E402

SPENT = ("eval_2000", "locked", "confirm_3000", "objective_4000")
MIN_USABLE = 150      # below this the masking conditions are not worth a card; stated up front


def sha256_file(p: Path) -> str:
    """Hash an input or output so the audit pins exactly what it read and wrote."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qids", default="data/gqa/tune_500_qids.json")
    ap.add_argument("--out", default="outputs/visual_audit/coverage.json")
    args = ap.parse_args()

    for bad in SPENT:
        if bad in args.qids:
            raise SystemExit(f"FAIL {args.qids!r} names a spent endpoint; this audit is "
                             f"development-only")

    out_p = ROOT / args.out
    if out_p.exists():
        raise SystemExit(f"FAIL refusing to overwrite {out_p}")

    cfg = load_config()
    qids = [str(q) for q in json.loads((ROOT / args.qids).read_text())]
    allq = json.loads(Path(cfg["gqa"]["questions"]["val_balanced"]).read_text())
    missing = [q for q in qids if q not in allq]
    if missing:
        raise SystemExit(f"FAIL {len(missing)} qids absent from the GQA question file")
    graphs = json.loads(Path(cfg["gqa"]["scene_graphs"]).read_text())

    print("=== visual-information audit — data coverage ===")
    print(f"slice     : {args.qids} ({len(qids)} questions)")
    print(f"scene graphs available for {len(graphs)} images\n")

    usable, rejects, pairs, plabels = [], Counter(), {}, {}
    no_graph = 0
    cats = Counter()
    for q in qids:
        e = allq[q]
        img = str(e["imageId"])
        g = graphs.get(img)
        if not g:
            no_graph += 1
            rejects["image has no scene graph at all"] += 1
            continue
        mp = mask_pair(e, g)
        if "reject" in mp:
            rejects[mp["reject"]] += 1
            continue
        usable.append(q)
        pairs[q] = {**mp, "image_id": img}
        cats[e.get("types", {}).get("structural", "?")] += 1
        pl = probe_labels(g)
        if pl:
            plabels[img] = pl

    print(f"USABLE for the masking conditions : {len(usable)}/{len(qids)} "
          f"({100 * len(usable) / len(qids):.1f}%)")
    print(f"images with no scene graph        : {no_graph}")
    print("\nrejections by cause (each question counted once):")
    for reason, n in rejects.most_common():
        print(f"  {n:4d}  {reason}")

    print(f"\nusable questions by GQA structural category:")
    for c, n in cats.most_common():
        print(f"  {c:10s} {n:4d}")

    if pairs:
        ratios = [p["area_ratio"] for p in pairs.values()]
        print(f"\nmatched-control box area ratio (irrelevant / relevant): "
              f"mean {sum(ratios) / len(ratios):.3f}, "
              f"min {min(ratios):.3f}, max {max(ratios):.3f}  (tolerance +/-{AREA_TOL:.0%})")
        same = sum(1 for p in pairs.values() if p["relevant_name"] == p["irrelevant_name"])
        print(f"  control shares the relevant object's CLASS on {same}/{len(pairs)} questions "
              f"(reported, not excluded: class identity is not what condition 6 controls)")

    # ---- probe-label coverage, per property, over the usable images ----
    imgs = sorted({p["image_id"] for p in pairs.values()})
    print(f"\nprobe-label coverage over the {len(imgs)} usable images:")
    prop_cov = {}
    for prop in ("object", "colour", "count", "position", "relation"):
        have = [i for i in imgs if prop in plabels.get(i, {})]
        classes = Counter(plabels[i][prop] for i in have)
        prop_cov[prop] = {"images": len(have), "classes": len(classes),
                          "majority_class_share": (classes.most_common(1)[0][1] / len(have))
                          if have else 0.0,
                          "top": classes.most_common(5)}
        print(f"  {prop:9s} {len(have):4d}/{len(imgs)} images, {len(classes):3d} classes, "
              f"majority-class baseline {100 * prop_cov[prop]['majority_class_share']:.1f}%  "
              f"top: {[c for c, _ in classes.most_common(3)]}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "audit": "visual-information audit — data coverage",
        "slice": args.qids, "slice_sha256": sha256_file(ROOT / args.qids),
        "n_questions": len(qids), "n_usable": len(usable),
        "usable_fraction": len(usable) / len(qids),
        "rejections": dict(rejects), "images_without_scene_graph": no_graph,
        "usable_by_category": dict(cats),
        "area_tolerance": AREA_TOL,
        "probe_label_coverage": prop_cov,
        "usable_qids": usable,
        "mask_pairs": pairs,
        "probe_labels": plabels,
        "note": "probe labels are DECODABILITY targets; recovering them shows information is "
                "present in the representation, never that Qwen2 uses it",
    }
    out_p.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {out_p.relative_to(ROOT)}  sha256 {sha256_file(out_p)[:16]}…")

    print("\n=== GATE ===")
    if len(usable) < MIN_USABLE:
        print(f"  STOP only {len(usable)} usable questions (< {MIN_USABLE}). The masking "
              f"conditions cannot be run defensibly on this slice. Report this rather than "
              f"widening the tolerance.")
        raise SystemExit(3)
    print(f"  PASS {len(usable)} usable questions carry a relevant box AND a size-matched "
          f"irrelevant control.")
    print("  Conditions 1-4 do not depend on masks and can run on the full slice; conditions")
    print("  5-6 run on this subset. The two groups are never pooled into one contrast.")


if __name__ == "__main__":
    main()
