"""Audit image overlap between RelTR's Open Images V6 VRD training set and the GQA tuning 500.

    python scripts/43_reltr_overlap_audit.py

WHY THIS GATE EXISTS. The experiment's whole claim is that the scene-graph generator never saw
the evaluation images. RelTR's Open Images checkpoint is trained on `oidv6-train-annotations-vrd.csv`
(the split its own data/README names). Open Images and Visual Genome are both drawn from Flickr,
so "different dataset" is an assumption, not a fact, until it is measured. GQA's images ARE
Visual Genome images and GQA image ids ARE VG image ids, so the join below is exact.

THE JOIN, AND WHY IT IS EXACT RATHER THAN PERCEPTUAL.
  GQA tuning qid -> VG image id      (GQA question file)
  VG image id    -> Flickr photo id  (VG image_data.json, field `flickr_id`)
  OI VRD train ImageID -> Flickr photo id (OI image metadata `OriginalLandingURL`)
Two images with the same Flickr photo id are the same photograph. Pixel or perceptual hashing
would be weaker here, not stronger: Open Images and Visual Genome re-encode and resize
independently, so a byte hash cannot match and a perceptual hash would need a threshold that
this audit would then have to defend.

WHAT IT CANNOT DO. An image with no `flickr_id` in VG metadata cannot be joined at all. Those
are reported as UNRESOLVED — never silently counted as clean. The same applies on the Open
Images side: any VRD train image whose metadata row is missing or whose landing URL does not
parse is counted, and a large unresolved fraction there weakens every "clean" verdict, so it
is printed next to the verdict rather than buried.

NO GPU. NO NETWORK. Reads only files already on disk; the fetch is a separate, logged step.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402

# https://www.flickr.com/photos/<user>/<photoid>[/...]  and
# https://farmN.staticflickr.com/<server>/<photoid>_<secret>[_o].jpg
_LANDING = re.compile(r"flickr\.com/photos/[^/]+/(\d+)")
_STATIC = re.compile(r"staticflickr\.com/\d+/(\d+)_")


def sha256_file(p: Path) -> str:
    """Hash an input file so the audit pins exactly what it read."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def flickr_id_from_row(row: dict) -> str | None:
    """Flickr photo id from an Open Images metadata row, or None if neither URL parses."""
    for field, pat in (("OriginalLandingURL", _LANDING), ("OriginalURL", _STATIC)):
        v = row.get(field) or ""
        m = pat.search(v)
        if m:
            return m.group(1)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qids", default="data/gqa/tune_500_qids.json")
    ap.add_argument("--vrd", default="data/external/openimages/oidv6-train-annotations-vrd.csv")
    ap.add_argument("--oi-meta",
                    default="data/external/openimages/train-images-boxable-with-rotation.csv")
    ap.add_argument("--vg-meta", default="data/external/vg/image_data.json")
    ap.add_argument("--out", default="outputs/reltr_sgg/overlap_audit.json")
    args = ap.parse_args()

    for bad in ("eval_2000", "locked", "confirm_3000", "objective_4000"):
        if bad in args.qids:
            raise SystemExit(f"FAIL {args.qids!r} names a slice this experiment must not touch")

    qids_p, vrd_p = ROOT / args.qids, ROOT / args.vrd
    oim_p, vgm_p = ROOT / args.oi_meta, ROOT / args.vg_meta
    for p in (qids_p, vrd_p, oim_p, vgm_p):
        if not p.is_file():
            raise SystemExit(f"FAIL required input absent: {p}")

    print("=== RelTR Open Images V6 VRD train  vs  GQA tuning 500 — image overlap audit ===\n")

    # ---- the evaluation side: tuning qids -> VG image ids -> Flickr photo ids ----
    cfg = load_config()
    qids = [str(q) for q in json.loads(qids_p.read_text())]
    allq = json.loads(Path(cfg["gqa"]["questions"]["val_balanced"]).read_text())
    missing_q = [q for q in qids if q not in allq]
    if missing_q:
        raise SystemExit(f"FAIL {len(missing_q)} tuning qids absent from the GQA question file")
    q2img = {q: str(allq[q]["imageId"]) for q in qids}
    del allq
    eval_images = sorted(set(q2img.values()))

    vg = json.loads(vgm_p.read_text())
    vg_by_id = {str(x["image_id"]): x for x in vg}
    del vg
    absent_from_vg = [i for i in eval_images if i not in vg_by_id]

    eval_flickr: dict[str, str] = {}       # vg image id -> flickr photo id
    unresolved: list[str] = []
    for i in eval_images:
        fid = vg_by_id.get(i, {}).get("flickr_id")
        if fid:
            eval_flickr[i] = str(fid)
        else:
            unresolved.append(i)
    print(f"evaluation side: {len(qids)} tuning questions -> {len(eval_images)} unique images")
    print(f"  resolved to a Flickr photo id : {len(eval_flickr)}")
    print(f"  UNRESOLVED (no flickr_id)     : {len(unresolved)}")
    if absent_from_vg:
        print(f"  absent from VG image_data     : {len(absent_from_vg)}")

    # ---- the training side: distinct VRD train ImageIDs ----
    print("\nreading the RelTR training split (this is the file its data/README names)...")
    vrd_ids: set[str] = set()
    with open(vrd_p, newline="") as f:
        r = csv.DictReader(f)
        if "ImageID" not in (r.fieldnames or []):
            raise SystemExit(f"FAIL {vrd_p.name} has no ImageID column: {r.fieldnames}")
        for row in r:
            vrd_ids.add(row["ImageID"])
    print(f"  oidv6-train-annotations-vrd.csv -> {len(vrd_ids):,} distinct training images")

    # ---- map those training images to Flickr photo ids ----
    print("\nresolving Open Images training images to Flickr photo ids...")
    train_flickr: set[str] = set()
    seen_meta = 0
    oi_unresolved = 0
    with open(oim_p, newline="") as f:
        for row in csv.DictReader(f):
            if row["ImageID"] not in vrd_ids:
                continue
            seen_meta += 1
            fid = flickr_id_from_row(row)
            if fid:
                train_flickr.add(fid)
            else:
                oi_unresolved += 1
    no_meta_row = len(vrd_ids) - seen_meta
    print(f"  metadata rows found      : {seen_meta:,}/{len(vrd_ids):,}")
    print(f"  distinct Flickr photo ids: {len(train_flickr):,}")
    print(f"  no metadata row          : {no_meta_row:,}")
    print(f"  metadata row unparseable : {oi_unresolved:,}")

    train_side_resolved = seen_meta - oi_unresolved
    train_cov = train_side_resolved / len(vrd_ids) if vrd_ids else 0.0
    print(f"  TRAINING-SIDE COVERAGE   : {100 * train_cov:.2f}% "
          f"(every 'clean' verdict below is only as strong as this)")

    # ---- the comparison ----
    matches = {img: fid for img, fid in eval_flickr.items() if fid in train_flickr}
    clean = [img for img, fid in eval_flickr.items() if fid not in train_flickr]

    print("\n=== RESULT ===")
    print(f"  EXACT MATCHES (same Flickr photo in both) : {len(matches)}")
    print(f"  VERIFIED CLEAN (resolved, not in training): {len(clean)}")
    print(f"  UNRESOLVED (no Flickr id on the GQA side) : {len(unresolved)}")
    print(f"  ---------------------------------------------------")
    print(f"  TOTAL unique evaluation images            : {len(eval_images)}")
    if matches:
        print("\n  matching image ids (VG id -> Flickr id):")
        for img, fid in sorted(matches.items())[:20]:
            print(f"    {img} -> {fid}")
    if unresolved:
        print(f"\n  unresolved VG image ids: {', '.join(unresolved[:20])}"
              + (" ..." if len(unresolved) > 20 else ""))

    frac_match = len(matches) / len(eval_images) if eval_images else 0.0
    frac_unres = len(unresolved) / len(eval_images) if eval_images else 0.0

    payload = {
        "audit": "RelTR Open Images V6 VRD train vs GQA tuning 500",
        "join": "exact Flickr photo id (VG image_data.flickr_id vs OI OriginalLandingURL/OriginalURL)",
        "evaluation_side": {
            "qids": len(qids), "unique_images": len(eval_images),
            "resolved": len(eval_flickr), "unresolved": len(unresolved),
            "unresolved_image_ids": unresolved,
            "absent_from_vg_metadata": absent_from_vg,
        },
        "training_side": {
            "vrd_train_images": len(vrd_ids),
            "metadata_rows_found": seen_meta,
            "no_metadata_row": no_meta_row,
            "metadata_unparseable": oi_unresolved,
            "distinct_flickr_ids": len(train_flickr),
            "coverage": train_cov,
        },
        "result": {
            "exact_matches": len(matches),
            "exact_match_image_ids": sorted(matches),
            "exact_match_flickr_ids": sorted(matches.values()),
            "verified_clean": len(clean),
            "unresolved": len(unresolved),
            "fraction_matched": frac_match,
            "fraction_unresolved": frac_unres,
        },
        "inputs": {
            "qids": str(qids_p.relative_to(ROOT)), "qids_sha256": sha256_file(qids_p),
            "vrd": str(vrd_p.relative_to(ROOT)), "vrd_sha256": sha256_file(vrd_p),
            "oi_meta": str(oim_p.relative_to(ROOT)), "oi_meta_sha256": sha256_file(oim_p),
            "vg_meta": str(vgm_p.relative_to(ROOT)), "vg_meta_sha256": sha256_file(vgm_p),
        },
        "interpretation": (
            "Exact matches are images RelTR was trained on and would then be evaluated on. "
            "Unresolved images carry unmeasured risk and bound the audit: the true overlap is "
            "at least `exact_matches` and at most `exact_matches + unresolved`."
        ),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise SystemExit(f"FAIL refusing to overwrite {out}; outputs/ is gitignored so an "
                         f"overwrite is unrecoverable")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")

    print("\n=== VERDICT ===")
    print(f"  overlap is at least {len(matches)} and at most {len(matches) + len(unresolved)} "
          f"of {len(eval_images)} images "
          f"({100 * frac_match:.2f}% .. {100 * (frac_match + frac_unres):.2f}%)")
    if train_cov < 0.90:
        print("  STOP training-side coverage below 90%: a 'clean' verdict cannot be defended")
        raise SystemExit(2)
    if len(matches) > 0:
        print("  STOP the generator was trained on evaluation images. Report, do not proceed "
              "without a decision on how to handle them.")
        raise SystemExit(3)
    print("  PASS no evaluation image is in the generator's training split by exact join.")
    print("  The unresolved residual above is the honest upper bound and must be reported.")


if __name__ == "__main__":
    main()
