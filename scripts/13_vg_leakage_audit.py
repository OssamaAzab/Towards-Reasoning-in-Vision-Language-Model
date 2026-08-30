"""Step 13: Visual Genome leakage gate for scene-graph Stage 2 (and retrieval).

GQA was generated from Visual Genome, so GQA-eval images ARE VG images: any scene-graph
generator trained on VG — or knowledge base built from it — may have seen the very images
(and their answer-bearing graphs) we evaluate on. This audit quantifies that overlap
BEFORE any Stage-2 generator is chosen, mirroring the VQAv2/LLaVA exclusion discipline:
the residual for whatever a generator was actually trained on must be ZERO.

Checks (all static, no GPU):
  1. eval-image universe: all GQA balanced-val images, plus the locked 2,000-qid subset;
  2. overlap with the VG corpus (image_data.json) and with VG's scene-graph annotations
     (scene_graphs.json — the supervision an SGG model trains on);
  3. COCO linkage: eval images with a VG coco_id (exposure for COCO-pretrained detectors),
     cross-checked against the existing 4,921-image exclusion list;
  4. verifies the exclusion artifact data/gqa/vg_exclude_gqa_eval.json, creating it only
     when absent — the VG image ids that must be excluded from any generator or knowledge base;
  5. --train-ids FILE: audit a candidate generator's actual training-image list (JSON list
     of VG image ids); residual must print 0 to pass the gate.

    python scripts/13_vg_leakage_audit.py
    python scripts/13_vg_leakage_audit.py --train-ids path/to/sgg_train_image_ids.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config  # noqa: E402


def verify_or_write_exclusion(out: Path, expected: list[int]) -> str:
    """Verify a frozen exclusion, or create it once when it is absent."""
    if out.exists():
        existing = [int(value) for value in json.loads(out.read_text())]
        if existing != expected:
            raise SystemExit(
                f"FAIL existing exclusion does not match the audited VG overlap: {out}"
            )
        return "verified"

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(expected))
    return "created"


def main():
    """Run the gate and print residuals; write the VG exclusion artifact."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--train-ids", type=Path, default=None,
                    help="JSON list of VG image ids a candidate generator was trained on")
    args = ap.parse_args()
    cfg = load_config()
    vg_root = Path(cfg["shared"]["vg_root"])

    # 1. The eval-image universe (GQA imageIds live in the VG image_id space).
    questions = json.load(open(cfg["gqa"]["questions"]["val_balanced"]))
    eval_imgs = {int(q["imageId"]) for q in questions.values()}
    locked_qids = json.load(open(Path(cfg["paths"]["data"]) / "gqa" / "eval_2000_qids.json"))
    locked_imgs = {int(questions[q]["imageId"]) for q in locked_qids}
    print(f"GQA balanced-val: {len(eval_imgs):,} unique eval images "
          f"({len(questions):,} questions); locked 2,000-qid set: {len(locked_imgs):,} images")
    assert locked_imgs <= eval_imgs, "locked set must be a subset of balanced-val"

    # 2. Overlap with the VG corpus and with VG's scene-graph annotations.
    image_data = json.load(open(vg_root / "image_data.json"))
    vg_ids = {rec["image_id"] for rec in image_data}
    coco_of = {rec["image_id"]: rec["coco_id"] for rec in image_data if rec["coco_id"]}
    in_vg = eval_imgs & vg_ids
    print(f"\nVG corpus (image_data.json): {len(vg_ids):,} images")
    print(f"  eval images present in VG:      {len(in_vg):,}/{len(eval_imgs):,} "
          f"({100 * len(in_vg) / len(eval_imgs):.1f}%)")
    print(f"  locked-set images present in VG: {len(locked_imgs & vg_ids):,}/{len(locked_imgs):,}")

    sg_ids = {rec["image_id"] for rec in json.load(open(vg_root / "scene_graphs.json"))}
    in_sg = eval_imgs & sg_ids
    print(f"VG scene-graph annotations (scene_graphs.json): {len(sg_ids):,} images")
    print(f"  eval images WITH a VG training graph: {len(in_sg):,} "
          f"(locked set: {len(locked_imgs & sg_ids):,})")
    print("  => an SGG model trained on unfiltered VG has seen the eval images AND their")
    print("     answer-bearing graphs; residual-zero requires excluding these ids first.")

    # 3. COCO linkage (exposure for COCO-pretrained detectors), vs the known 4,921 list.
    eval_coco = {coco_of[i] for i in eval_imgs if i in coco_of}
    locked_coco = {coco_of[i] for i in locked_imgs if i in coco_of}
    known = set(json.load(open(cfg["vqa"]["exclude_ids"])))
    print(f"\nCOCO linkage: {len(eval_coco):,} eval images have a coco_id "
          f"(locked set: {len(locked_coco):,})")
    print(f"  existing exclusion list: {len(known):,} ids; "
          f"agreement: {len(eval_coco & known):,} shared, "
          f"{len(eval_coco - known):,} new, {len(known - eval_coco):,} only-in-list")

    # 4. The exclusion artifact for Stage 2 (generator training sets / knowledge bases).
    out = Path(cfg["paths"]["data"]) / "gqa" / "vg_exclude_gqa_eval.json"
    expected = sorted(in_vg)
    exclusion_status = verify_or_write_exclusion(out, expected)
    if exclusion_status == "verified":
        print(f"\nverified existing {out} ({len(expected):,} VG image ids; unchanged)")
    else:
        print(f"\nwrote new {out} ({len(expected):,} VG image ids to exclude)")

    # 5. The gate itself, for a concrete candidate training list.
    if args.train_ids:
        train = {int(i) for i in json.load(open(args.train_ids))}
        residual = train & eval_imgs
        print(f"\nGATE — candidate training set {args.train_ids} ({len(train):,} images): "
              f"residual GQA-eval images = {len(residual)}")
        print("PASS (residual zero)" if not residual
              else f"FAIL — first offenders: {sorted(residual)[:10]}")
        sys.exit(0 if not residual else 1)
    else:
        print("\nGATE: no --train-ids supplied — rerun with the chosen generator's actual")
        print("training-image list before any predicted-graph eval; residual must be 0.")


if __name__ == "__main__":
    main()
