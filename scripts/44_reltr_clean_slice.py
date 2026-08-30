"""Derive the leakage-clean evaluation slice for the RelTR experiment, from the audit alone.

    python scripts/44_reltr_clean_slice.py

WHY THIS EXISTS. scripts/43_reltr_overlap_audit.py found that 6 of the 481 GQA tuning images are
in RelTR's Open Images V6 VRD *training* split by exact Flickr photo id, with 1-3 relationship
annotations each. A further 9 images carry no `flickr_id` in Visual Genome metadata and therefore
cannot be shown clean either way. The project's leakage rule (REPRODUCIBILITY.md) holds the
exclusion at **residual = 0**, and any new scene-graph generator inherits it unchanged, so the
evaluation runs on images the generator provably never saw.

WHAT IS EXCLUDED, AND WHY BOTH GROUPS.
  * the 6 exact matches   — proven contaminated; the generator was trained on these photographs
  * the 9 unresolved      — cannot be proven clean, and "not disproven" is not "clean"
Excluding only the first group would leave a residual that is unmeasured rather than zero.

THIS IS PRE-REGISTERED. It is derived mechanically from the audit JSON, with no accuracy in
existence anywhere: no model has been run on any of these questions under this experiment. The
exclusion therefore cannot have been chosen to move a result, and the ordering is provable from
git history and the audit file's own hash.

NOT A NEW ENDPOINT. The output is a strict SUBSET of the tuning 500, which is already the
project's development surface where selection is permitted. The tuning file itself is not
modified, and the confirmatory 3,000, locked 2,000 and objective 4,000 are untouched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config  # noqa: E402


def sha256_file(p: Path) -> str:
    """Hash a file so the derived slice pins exactly what produced it."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", default="outputs/reltr_sgg/overlap_audit.json")
    ap.add_argument("--qids", default="data/gqa/tune_500_qids.json")
    ap.add_argument("--out", default="data/gqa/tune_500_reltr_clean_qids.json")
    args = ap.parse_args()

    audit_p, qids_p, out_p = ROOT / args.audit, ROOT / args.qids, ROOT / args.out
    if not audit_p.is_file():
        raise SystemExit(f"FAIL audit {audit_p} absent; run scripts/43_reltr_overlap_audit.py")
    if out_p.exists():
        raise SystemExit(f"FAIL refusing to overwrite {out_p}; a slice is frozen once written")

    audit = json.loads(audit_p.read_text())
    if audit["inputs"]["qids_sha256"] != sha256_file(qids_p):
        raise SystemExit("FAIL the audit was run against a different tuning file than this one")

    matched = set(audit["result"]["exact_match_image_ids"])
    unresolved = set(audit["evaluation_side"]["unresolved_image_ids"])
    drop = matched | unresolved
    if len(drop) != len(matched) + len(unresolved):
        raise SystemExit("FAIL matched and unresolved sets overlap; the audit is inconsistent")

    cfg = load_config()
    qids = [str(q) for q in json.loads(qids_p.read_text())]
    allq = json.loads(Path(cfg["gqa"]["questions"]["val_balanced"]).read_text())
    q2img = {q: str(allq[q]["imageId"]) for q in qids}
    del allq

    keep = [q for q in qids if q2img[q] not in drop]
    dropped = [q for q in qids if q2img[q] in drop]
    kept_images = sorted({q2img[q] for q in keep})

    if set(kept_images) & drop:
        raise SystemExit("FAIL a dropped image survived into the clean slice")
    if len(keep) + len(dropped) != len(qids):
        raise SystemExit("FAIL kept + dropped != total; the partition is not a partition")

    print("=== leakage-clean slice for the RelTR experiment ===")
    print(f"  source slice          : {args.qids} ({len(qids)} questions, "
          f"{len(set(q2img.values()))} images)")
    print(f"  excluded images       : {len(drop)}  "
          f"({len(matched)} exact matches + {len(unresolved)} unresolved)")
    print(f"  excluded questions    : {len(dropped)}")
    print(f"  KEPT questions        : {len(keep)}")
    print(f"  KEPT images           : {len(kept_images)}")

    out_p.write_text(json.dumps(keep, indent=1) + "\n")
    manifest = {
        "slice": "tune_500_reltr_clean",
        "derivation": "tune_500_qids.json minus every question whose image is an exact Open "
                      "Images VRD-train match or is unresolvable on the Flickr join",
        "status": "PRE-REGISTERED — derived from the audit before any accuracy existed",
        "parent_slice": str(qids_p.relative_to(ROOT)),
        "parent_sha256": sha256_file(qids_p),
        "audit": str(audit_p.relative_to(ROOT)), "audit_sha256": sha256_file(audit_p),
        "excluded_images": {"exact_matches": sorted(matched), "unresolved": sorted(unresolved)},
        "excluded_question_count": len(dropped),
        "n_questions": len(keep), "n_images": len(kept_images),
        "residual_leakage": 0,
        "note": "a strict subset of the tuning 500, which is the development surface; this is "
                "NOT a new endpoint and creates no new selection budget",
    }
    mp = out_p.with_name(out_p.stem + "_MANIFEST.json")
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out_p.relative_to(ROOT)}  sha256 {sha256_file(out_p)}")
    print(f"wrote {mp.relative_to(ROOT)}")
    print("\nRESIDUAL LEAKAGE = 0 by construction. The full-481 result remains computable as a "
          "clearly-labelled sensitivity, never as the primary.")


if __name__ == "__main__":
    main()
