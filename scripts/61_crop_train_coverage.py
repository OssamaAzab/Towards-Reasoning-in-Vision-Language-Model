"""Stage B training population: oracle crops on the GQA TRAIN split. CPU only. THE GATE.

    python scripts/61_crop_train_coverage.py

EXPLORATORY — SUPERVISED GQA ORACLE-REGION ADAPTER — NOT ZERO-SHOT — NOT DEPLOYABLE.

WHY THE GQA TRAIN SPLIT, AND WHAT IT COSTS. Stage B trains a module, so it needs training data
carrying the one thing Stage A's arms need: a question, a ground-truth box for an object that
question names, and a scene graph rich enough to build the matched controls. Only GQA has that.
LLaVA and VQAv2 have no question-to-object annotation, so no oracle crop can be built from them.

THE COST IS EXPLICIT AND IS NOT HIDDEN BY THIS SCRIPT. The project's standing framing is that GQA
is evaluation data, never trained on, and that every reported GQA number is zero-shot. **An
adapter trained here is no longer zero-shot with respect to GQA**, and any comparison against the
frozen baseline must say so. The evaluation population remains the untouched development slice
and no spent endpoint is read.

WHAT IS VERIFIED RATHER THAN ASSUMED. GQA's train and val splits are image-disjoint by
construction, but "by construction" is how leakage gets missed. This script recomputes the
intersection of the training images against the val split AND against every slice this project
has spent (locked 2,000, confirmatory 3,000, objective 4,000, tuning 500) and refuses to write a
population if any of them is non-empty. That is the Section 9.4 VG/GQA exclusion applied to a new
data source, as the project rule requires.

THE CROP RULE IS THE STAGE A RULE, UNCHANGED. Same module, same margin, same square padding, same
area tolerance, same same-label different-image derangement. Using a different rule for training
than for evaluation would make the adapter's training distribution and its test distribution
differ in geometry rather than in content.

TRAIN AND VALIDATION ARE BOTH DRAWN FROM GQA TRAIN and are image-disjoint from each other, so the
validation loss reported during training is not a memorised image.

CPU ONLY, BY PROJECT RULE. Geometry, coverage and derangement. No model, no inference, no
accuracy, and nothing here may inform a stop/go decision about answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
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

SEED = 42
SPENT_SLICES = ("eval_2000", "confirm_3000", "objective_4000", "tune_500")
LABEL = "EXPLORATORY — SUPERVISED GQA ORACLE-REGION ADAPTER — NOT ZERO-SHOT — NOT DEPLOYABLE"


def sha256_file(p: Path) -> str:
    """Hash an input or output so the audit pins exactly what it read and wrote."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--questions", default="data/gqa/questions/train_balanced_questions.json")
    ap.add_argument("--graphs", default="data/gqa/train_sceneGraphs.json")
    ap.add_argument("--sample", type=int, default=40000,
                    help="questions to consider before crop/donor filtering")
    ap.add_argument("--val-images", type=int, default=800,
                    help="images held out of training for the validation loss")
    ap.add_argument("--out", default="outputs/crop_adapter/train_coverage.json")
    args = ap.parse_args()

    cfg = load_config()
    qp, gp, out_p = ROOT / args.questions, ROOT / args.graphs, ROOT / args.out
    for p in (qp, gp):
        if not p.is_file():
            raise SystemExit(f"FAIL {p} absent")

    print(f"=== Stage B training population — {LABEL} ===")
    print(f"questions : {args.questions}")
    print(f"graphs    : {args.graphs}")
    print(f"crop rule : expand {CROP_MARGIN:.0%} -> clip -> pad to square; min side {MIN_CROP_PX}px, "
          f"max {MAX_CROP_AREA_FRAC:.0%} of image, area tolerance +/-{AREA_TOL:.0%}")

    allq = json.loads(qp.read_text())
    graphs = json.loads(gp.read_text())
    print(f"loaded {len(allq):,} train questions over {len(graphs):,} scene graphs")

    # ---- leakage gate: nothing here may share an image with anything already spent ----
    val_q = json.loads(Path(cfg["gqa"]["questions"]["val_balanced"]).read_text())
    val_images = {str(q["imageId"]) for q in val_q.values()}
    train_images = {str(q["imageId"]) for q in allq.values()}
    shared = train_images & val_images
    if shared:
        raise SystemExit(f"FAIL {len(shared)} images appear in BOTH the GQA train split and the "
                         f"GQA val split this project evaluates on; refusing to build a "
                         f"training population")
    print(f"PASS train images ({len(train_images):,}) disjoint from val images "
          f"({len(val_images):,})")
    for name in SPENT_SLICES:
        p = ROOT / "data" / "gqa" / f"{name}_qids.json"
        if not p.is_file():
            raise SystemExit(f"FAIL cannot verify the {name} slice: {p} absent")
        imgs = {str(val_q[q]["imageId"]) for q in map(str, json.loads(p.read_text()))
                if q in val_q}
        bad = imgs & train_images
        if bad:
            raise SystemExit(f"FAIL {len(bad)} images of the spent slice {name} appear in the "
                             f"training population")
        print(f"PASS {name}: none of its {len(imgs):,} images is in the training split")

    # ---- deterministic sample, then the Stage A crop rule ----
    qids = sorted(allq)
    random.Random(SEED).shuffle(qids)
    qids = qids[: args.sample]
    candidates, rejects, records = [], Counter(), {}
    for q in qids:
        ex = allq[q]
        g = graphs.get(str(ex.get("imageId")))
        if g is None:
            rejects["image has no scene graph"] += 1
            continue
        r = crop_record(ex, g)
        if "reject" in r:
            rejects[r["reject"]] += 1
            continue
        r["qid"], r["image_id"] = q, str(ex["imageId"])
        records[q] = r
        candidates.append(q)
    print(f"\ncandidates : {len(candidates):,}/{len(qids):,} carry a relevant crop "
          f"({len(candidates) / len(qids):.1%})")
    for why, n in rejects.most_common(6):
        print(f"   {n:6d}  {why}")
    if not candidates:
        raise SystemExit("FAIL no training question yields a relevant crop")

    # ---- the same same-label, different-image derangement Stage A evaluates under ----
    profiles = {q: {"area": records[q]["relevant"]["area"], "image_id": records[q]["image_id"],
                    "label": records[q]["target_label"]} for q in candidates}
    donor, unusable = build_wrong_crop_derangement(candidates, profiles)
    pop = sorted(donor)
    print(f"\ndonor matching: {len(pop):,} questions keep a same-label different-image wrong crop; "
          f"{len(unusable):,} target classes unusable")
    for q in pop:
        d = donor[q]
        records[q]["wrong_donor_qid"] = d
        records[q]["wrong_donor_image_id"] = records[d]["image_id"]
        records[q]["wrong_donor_label"] = records[d]["target_label"]
        records[q]["wrong_box"] = list(records[d]["relevant"]["box"])
        records[q]["wrong_area_diff"] = abs(records[d]["relevant"]["area"]
                                            - records[q]["relevant"]["area"])
        records[q]["scale"] = scale_pair_diagnostics(
            records[q]["relevant"]["box"], records[q]["wrong_box"],
            rel_raw=records[q]["relevant"].get("raw_box"),
            wrong_raw=records[d]["relevant"].get("raw_box"))
        records[q]["in_scale_matched"] = records[q]["scale"]["scale_matched"]

    if [q for q in pop if donor[q] == q]:
        raise SystemExit("FAIL the wrong-crop map is not a derangement")
    if [q for q in pop if records[donor[q]]["image_id"] == records[q]["image_id"]]:
        raise SystemExit("FAIL a donor shares the question's own image")
    if [q for q in pop if records[donor[q]]["target_label"] != records[q]["target_label"]]:
        raise SystemExit("FAIL a donor does not share the target label")

    # ---- train / validation split, by IMAGE so no image straddles the two ----
    images = sorted({records[q]["image_id"] for q in pop})
    random.Random(SEED + 1).shuffle(images)
    val_imgs = set(images[: args.val_images])
    train_qids = [q for q in pop if records[q]["image_id"] not in val_imgs]
    val_qids = [q for q in pop if records[q]["image_id"] in val_imgs]
    if set(train_qids) & set(val_qids):
        raise SystemExit("FAIL the train and validation question sets overlap")
    straddle = ({records[q]["image_id"] for q in train_qids}
                & {records[q]["image_id"] for q in val_qids})
    if straddle:
        raise SystemExit(f"FAIL {len(straddle)} images appear in both train and validation")

    n_irr = sum(1 for q in pop if records[q]["has_irrelevant"])
    print(f"\npopulation:")
    print(f"   train      : {len(train_qids):,} questions over "
          f"{len({records[q]['image_id'] for q in train_qids}):,} images")
    print(f"   validation : {len(val_qids):,} questions over {len(val_imgs):,} images "
          f"(image-disjoint from train)")
    print(f"   carry an irrelevant crop too: {n_irr:,} ({n_irr / len(pop):.1%})")
    print(f"   scale-matched at factor {SCALE_MATCH_FACTOR:g}: "
          f"{sum(records[q]['in_scale_matched'] for q in pop):,}")

    payload = {
        "experiment": "Stage B — trainable crop-evidence adapter (ORACLE region)",
        "label": LABEL, "oracle": True,
        "not_deployable": "the relevant crop is located from GQA ground-truth scene graphs; no "
                          "inference-time system knows that box",
        "zero_shot_caveat": "this population is the GQA TRAIN split. An adapter trained on it is "
                            "NOT zero-shot with respect to GQA, unlike every frozen baseline it "
                            "is compared against. Any comparison must state this.",
        "seed": SEED, "sample_considered": len(qids),
        "questions_file": args.questions, "questions_sha256": sha256_file(qp),
        "graphs_file": args.graphs, "graphs_sha256": sha256_file(gp),
        "module_sha256": sha256_file(ROOT / "src" / "data" / "crop_augment.py"),
        "script_sha256": sha256_file(Path(__file__)),
        "crop_rule": {"margin": CROP_MARGIN, "min_crop_px": MIN_CROP_PX,
                      "max_crop_area_frac": MAX_CROP_AREA_FRAC, "area_tol": AREA_TOL,
                      "pad": "square, flat mid-grey, aspect ratio preserved"},
        "leakage_checks": {
            "train_val_image_overlap": 0,
            "spent_slice_image_overlap": {n: 0 for n in SPENT_SLICES},
            "verified_by": "recomputed in this script; a non-empty intersection raises"},
        "n_candidates": len(candidates), "n_population": len(pop),
        "n_train": len(train_qids), "n_val": len(val_qids), "n_with_irrelevant": n_irr,
        "rejections": dict(rejects),
        "unusable_target_classes": len(unusable),
        "train_qids": train_qids, "val_qids": val_qids,
        "records": {q: records[q] for q in pop},
    }
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if out_p.exists():
        raise SystemExit(f"FAIL refusing to overwrite {out_p}")
    out_p.write_text(json.dumps(payload) + "\n")
    print(f"\nwrote {out_p.relative_to(ROOT)}  sha256 {sha256_file(out_p)[:16]}…")
    print(f"\n{LABEL}")
    print("Training on GQA train means Stage B results are NOT zero-shot on GQA. State it.")


if __name__ == "__main__":
    main()
