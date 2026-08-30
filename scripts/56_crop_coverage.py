"""Coverage audit for Stage A crop augmentation. CPU only. THE GATE BEFORE ANY GPU RUN.

    python scripts/56_crop_coverage.py

WHY THIS RUNS FIRST, AND WHY IT RUNS TO COMPLETION BEFORE ANY ACCURACY EXISTS. It defines TWO
NESTED COHORTS, and which contrast each may carry:

  core              a `relevant` crop from an object the question names, AND a same-target
                    `wrong` crop donated by a different image of the same normalised object
                    label. Carries global_only, global_plus_relevant_crop,
                    global_plus_wrong_crop and relevant_crop_only — and the PRIMARY
                    relevant-vs-wrong contrast.

  irrelevant_subset core questions that ADDITIONALLY have a valid `irrelevant` crop: same image,
                    overlapping no referenced object, area-matched to the relevant crop. This is
                    the only population on which the irrelevant arm is generated, and it carries
                    the SECONDARY relevant-vs-irrelevant contrast alone.

  scale_matched     core questions whose relevant and wrong PADDED crop areas are within a factor
                    of SCALE_MATCH_FACTOR of each other, in either direction. Carries the
                    ROBUSTNESS reading of the primary contrast only. Nested in core.

Both subsets are nested in core and none of the three is ever differenced against another as
though paired. Splitting core from irrelevant_subset matters: requiring an irrelevant crop for
admission would cut the PRIMARY cohort roughly in half for the sake of a secondary contrast.

WHY A SCALE-MATCHED SUBGROUP EXISTS. The wrong crop is matched on target LABEL first and only then
area-minimised, so inside a small label class the closest donor can still be many times larger or
smaller than the relevant crop. Every crop is padded square and resized to the encoder's fixed
input, so a padded-area difference is a difference in the RESOLUTION at which the object reaches
the encoder — a scale confound living inside the primary contrast. The subgroup is defined here,
from geometry alone, on a threshold fixed in src/data/crop_augment.py before any accuracy exists.

A question missing a requirement is EXCLUDED and its reason recorded — never repaired. Target
classes that cannot form a valid donor group (a single question, or all questions on one image)
are reported by name. The rules in src/data/crop_augment.py are fixed now; changing them after
seeing accuracy would make the cohorts a function of the result.

STAGE A IS AN ORACLE CEILING. The relevant crop is located from GQA's ground-truth scene graph.
Nothing at inference time knows that box. Every record here is stamped `oracle: true`.

CPU ONLY, BY PROJECT RULE. This script computes geometry, coverage and the derangement. It loads
no model, runs no inference, and produces no accuracy — CPU and cluster-GPU numbers are not
interchangeable in this project, so nothing here may inform a stop/go decision about answers.
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

from src.data.crop_augment import (AREA_TOL, CROP_MARGIN, MAX_CROP_AREA_FRAC,  # noqa: E402
                                   MIN_CROP_PX, SCALE_MATCH_FACTOR,
                                   build_wrong_crop_derangement, crop_record,
                                   scale_pair_diagnostics)
from src.utils import load_config  # noqa: E402

SPENT = ("eval_2000", "locked", "confirm_3000", "objective_4000")
MIN_USABLE = 120      # stated up front: below this, Stage A is not worth a card


def sha256_file(p: Path) -> str:
    """Hash an input or output so the audit pins exactly what it read and wrote."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qids", default="data/gqa/tune_500_qids.json")
    ap.add_argument("--out", default="outputs/crop_augment/coverage.json")
    args = ap.parse_args()

    for bad in SPENT:
        if bad in args.qids:
            raise SystemExit(f"FAIL {args.qids!r} names a SPENT endpoint. Development data only.")

    cfg = load_config()
    qids = [str(q) for q in json.loads((ROOT / args.qids).read_text())]
    allq = json.loads(Path(cfg["gqa"]["questions"]["val_balanced"]).read_text())
    graphs = json.loads(Path(cfg["gqa"]["scene_graphs"]).read_text())
    out_p = ROOT / args.out

    print("=== Stage A crop coverage — ORACLE feasibility ===")
    print(f"slice      : {args.qids}  ({len(qids)} questions)")
    print(f"crop rule  : expand {CROP_MARGIN:.0%} -> clip -> pad to square; "
          f"min side {MIN_CROP_PX}px, max {MAX_CROP_AREA_FRAC:.0%} of image, "
          f"area tolerance +/-{AREA_TOL:.0%}")
    print(f"scene graphs available for {len(graphs)} images\n")

    candidates, rejects, records = [], Counter(), {}
    for q in qids:
        ex = allq.get(q)
        if ex is None:
            rejects["question id not in the val_balanced file"] += 1
            continue
        img = str(ex.get("imageId"))
        g = graphs.get(img)
        if g is None:
            rejects["image has no scene graph"] += 1
            continue
        r = crop_record(ex, g)
        if "reject" in r:
            rejects[r["reject"]] += 1
            continue
        r["qid"], r["image_id"] = q, img
        records[q] = r
        candidates.append(q)

    print(f"candidates : {len(candidates)}/{len(qids)} carry a relevant crop "
          f"({len(candidates) / len(qids):.1%})")
    print("rejected before donor matching:")
    for why, n in rejects.most_common():
        print(f"   {n:4d}  {why}")
    if not candidates:
        raise SystemExit("FAIL no question yields a relevant crop")

    # ---- the wrong-crop derangement: same target label, different image ----
    profiles = {q: {"area": records[q]["relevant"]["area"], "image_id": records[q]["image_id"],
                    "label": records[q]["target_label"]} for q in candidates}
    donor, unusable = build_wrong_crop_derangement(candidates, profiles)

    core = sorted(donor)
    dropped_no_donor = [q for q in candidates if q not in donor]
    print(f"\ndonor matching (same normalised target label, different image, area-proximate):")
    print(f"   target classes total      : {len({profiles[q]['label'] for q in candidates})}")
    print(f"   classes forming donors    : {len({profiles[q]['label'] for q in core})}")
    print(f"   classes UNUSABLE          : {len(unusable)}")
    for label, why in sorted(unusable.items())[:12]:
        print(f"      {label!r:24s} {why}")
    if len(unusable) > 12:
        print(f"      ... and {len(unusable) - 12} more")
    print(f"   questions dropped for lack of a valid donor: {len(dropped_no_donor)}")

    for q in core:
        records[q]["wrong_donor_qid"] = donor[q]
        records[q]["wrong_donor_image_id"] = records[donor[q]]["image_id"]
        records[q]["wrong_donor_label"] = records[donor[q]]["target_label"]
        records[q]["wrong_box"] = list(records[donor[q]]["relevant"]["box"])
        records[q]["wrong_area_diff"] = abs(records[donor[q]]["relevant"]["area"]
                                            - records[q]["relevant"]["area"])

    self_don = [q for q in core if donor[q] == q]
    same_img = [q for q in core if records[donor[q]]["image_id"] == records[q]["image_id"]]
    diff_lab = [q for q in core if records[donor[q]]["target_label"] != records[q]["target_label"]]
    one_to_one = sorted(donor.values()) == sorted(core)
    print(f"\nderangement: one-to-one={one_to_one}  self-donations={len(self_don)}  "
          f"same-image={len(same_img)}  label-mismatched={len(diff_lab)}")
    if self_don or not one_to_one:
        raise SystemExit("FAIL the wrong-crop map is not a derangement")
    if same_img:
        raise SystemExit(f"FAIL {len(same_img)} donors share the question's own image")
    if diff_lab:
        raise SystemExit(f"FAIL {len(diff_lab)} donors do not share the target label")

    # ---- pre-registered crop-scale diagnostics, and the scale-matched subgroup ----
    # Computed from geometry only, with no model loaded and no accuracy in existence. Membership
    # therefore cannot be a function of which arm happened to answer correctly.
    for q in core:
        records[q]["scale"] = scale_pair_diagnostics(
            records[q]["relevant"]["box"], records[q]["wrong_box"],
            rel_raw=records[q]["relevant"].get("raw_box"),
            wrong_raw=records[donor[q]]["relevant"].get("raw_box"))
        records[q]["in_scale_matched"] = records[q]["scale"]["scale_matched"]

    scale_matched = sorted(q for q in core if records[q]["in_scale_matched"])
    ratios = sorted(records[q]["scale"]["padded_area_ratio"] for q in core)
    pct = lambda f: ratios[min(len(ratios) - 1, int(f * len(ratios)))]  # noqa: E731
    print(f"\ncrop-scale diagnostics (padded square canvas; relevant vs wrong):")
    print(f"   padded-area ratio (larger/smaller): median {pct(0.5):.2f}  p75 {pct(0.75):.2f}  "
          f"p90 {pct(0.90):.2f}  max {ratios[-1]:.2f}")
    print(f"   scale-matched at factor {SCALE_MATCH_FACTOR:g}: {len(scale_matched)}/{len(core)} "
          f"({len(scale_matched) / len(core):.1%})")

    # ---- the three cohorts ----
    irr_subset = sorted(q for q in core if records[q]["has_irrelevant"])
    print(f"\ncohorts:")
    print(f"   core              : {len(core)}  (global, relevant, same-target wrong, crop-only)")
    print(f"   irrelevant_subset : {len(irr_subset)}  (core questions that ALSO have a valid "
          f"non-overlapping area-matched irrelevant crop)")
    print(f"   scale_matched     : {len(scale_matched)}  (core questions whose relevant and wrong "
          f"padded areas are within a factor of {SCALE_MATCH_FACTOR:g})")
    print(f"   both subsets are NESTED in core; none is ever compared as though paired.")

    imgs = {records[q]["image_id"] for q in core}
    imgs_sub = {records[q]["image_id"] for q in irr_subset}
    errs = [records[q]["area_match_error"] for q in irr_subset]
    rel_a = [records[q]["relevant"]["area"] for q in core]
    wdiff = [records[q]["wrong_area_diff"] for q in core]
    cats = Counter(allq[q].get("types", {}).get("structural", "?") for q in core)

    print(f"\nunique images: core {len(imgs)}, subset {len(imgs_sub)}")
    print(f"relevant crop area (core): min {min(rel_a):,} "
          f"median {sorted(rel_a)[len(rel_a) // 2]:,} max {max(rel_a):,} px")
    print(f"wrong-donor area difference: mean {sum(wdiff) / len(wdiff):,.0f} px, "
          f"median {sorted(wdiff)[len(wdiff) // 2]:,} px")
    if errs:
        print(f"irrelevant area-match error (subset): mean {sum(errs) / len(errs):.3f} "
              f"max {max(errs):.3f}")
    print(f"structural categories (core): {dict(cats)}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": "Stage A — question-conditioned visual-evidence crops (ORACLE feasibility)",
        "oracle": True,
        "not_deployable": "the relevant crop is located from GQA ground truth; no inference-time "
                          "system knows that box. This measures a ceiling, not a method.",
        "slice": args.qids, "slice_sha256": sha256_file(ROOT / args.qids),
        "questions_file": cfg["gqa"]["questions"]["val_balanced"],
        "questions_sha256": sha256_file(Path(cfg["gqa"]["questions"]["val_balanced"])),
        "scene_graphs_file": cfg["gqa"]["scene_graphs"],
        "scene_graphs_sha256": sha256_file(Path(cfg["gqa"]["scene_graphs"])),
        "module_sha256": sha256_file(ROOT / "src" / "data" / "crop_augment.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "crop_rule": {"margin": CROP_MARGIN, "min_crop_px": MIN_CROP_PX,
                      "max_crop_area_frac": MAX_CROP_AREA_FRAC, "area_tol": AREA_TOL,
                      "pad": "square, flat mid-grey, aspect ratio preserved"},
        "n_questions": len(qids),
        "n_candidates": len(candidates),
        "n_core": len(core), "n_irrelevant_subset": len(irr_subset),
        "n_scale_matched": len(scale_matched),
        "n_rejected": len(qids) - len(core),
        "unique_images": len(imgs), "unique_images_subset": len(imgs_sub),
        "rejections": dict(rejects),
        "unusable_target_classes": unusable,
        "n_dropped_no_valid_donor": len(dropped_no_donor),
        "dropped_no_valid_donor": dropped_no_donor,
        "core_by_category": dict(cats),
        "wrong_donor_area_diff": {"mean": sum(wdiff) / len(wdiff),
                                  "median": sorted(wdiff)[len(wdiff) // 2]},
        "area_match_error": ({"mean": sum(errs) / len(errs), "max": max(errs)} if errs else None),
        "derangement": {"one_to_one": one_to_one, "self_donations": len(self_don),
                        "same_image_donations": len(same_img),
                        "label_mismatched": len(diff_lab),
                        "matched_on": "normalised referenced-object target label (never the "
                                      "answer), then a smallest-shift rotation of an area-sorted "
                                      "class that avoids same-image donation"},
        "cohorts": {
            "core": {"n": len(core),
                     "arms": ["global_only", "global_plus_relevant_crop",
                              "global_plus_wrong_crop", "relevant_crop_only"],
                     "carries": "the PRIMARY relevant - wrong contrast"},
            "irrelevant_subset": {"n": len(irr_subset),
                                  "arms": ["global_plus_irrelevant_crop"],
                                  "nested_in": "core",
                                  "carries": "the SECONDARY relevant - irrelevant contrast only"},
            "scale_matched": {"n": len(scale_matched),
                              "arms": ["global_plus_relevant_crop", "global_plus_wrong_crop"],
                              "nested_in": "core",
                              "rule": f"padded crop areas within a factor of "
                                      f"{SCALE_MATCH_FACTOR:g}, either direction",
                              "carries": "the ROBUSTNESS reading of the PRIMARY contrast only; "
                                         "its interval is never required to exclude zero"},
        },
        "scale_match_factor": SCALE_MATCH_FACTOR,
        "padded_area_ratio": {"median": pct(0.5), "p75": pct(0.75), "p90": pct(0.90),
                              "max": ratios[-1]},
        "core_qids": core,
        "irrelevant_subset_qids": irr_subset,
        "scale_matched_qids": scale_matched,
        "records": records,
    }
    if out_p.exists():
        raise SystemExit(f"FAIL refusing to overwrite {out_p}")
    out_p.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"\nwrote {out_p.relative_to(ROOT)}  sha256 {sha256_file(out_p)[:16]}…")

    print("\n=== GATE ===")
    if len(core) < MIN_USABLE:
        print(f"  STOP only {len(core)} in the core cohort (< {MIN_USABLE}). Report that Stage A "
              f"cannot be run defensibly on this slice rather than widening the rule.")
        raise SystemExit(3)
    print(f"  PASS core cohort = {len(core)} questions with a relevant crop and a same-target, "
          f"different-image wrong donor.")
    print(f"       irrelevant_subset = {len(irr_subset)}, used ONLY for relevant - irrelevant.")
    print(f"       scale_matched = {len(scale_matched)}, used ONLY as the ROBUSTNESS reading of "
          f"the primary contrast.")
    print("  ORACLE CEILING ONLY. Nothing here is deployable.")


if __name__ == "__main__":
    main()
