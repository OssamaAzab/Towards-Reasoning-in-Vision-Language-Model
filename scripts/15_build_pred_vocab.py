"""Stage 2: build the leakage-safe OWLv2 vocabulary from VG-train object frequencies.

The predicted-graph generator prompts OWLv2 with a fixed object vocabulary. To make that
vocabulary leakage-safe AND well-matched to GQA imagery, it is the top-N most frequent
object names in Visual Genome counted ONLY over images that are NOT GQA-eval images
(data/gqa/vg_exclude_gqa_eval.json, the Section 9.4 audit artifact). The set of images the
counts came from is saved alongside, so `13_vg_leakage_audit.py --train-ids` can prove
residual = 0 over exactly the data this vocabulary was derived from.

Writes:
  data/gqa/pred_vocab.json         {"vocab": [...], "n_source_images": ..., "built_from": ...}
  data/gqa/pred_vocab_image_ids.json   the contributing VG image ids (for the leakage gate)

    python scripts/15_build_pred_vocab.py --top 120
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config  # noqa: E402


def main() -> None:
    """Count VG object names over non-eval images; write the top-N vocab + source-image ids."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=120, help="vocabulary size (OWLv2 query count)")
    ap.add_argument("--min-len", type=int, default=3, help="drop names shorter than this")
    ap.add_argument("--tag", default="", help="output-file suffix (e.g. '400' -> pred_vocab_400.json) "
                    "so a rebuild never overwrites the vocab an earlier run used")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""

    cfg = load_config()
    vg_objects = Path(cfg["shared"]["vg_root"]) / "objects.json"
    exclude = {int(i) for i in json.load(open("data/gqa/vg_exclude_gqa_eval.json"))}
    print(f"exclusion list: {len(exclude):,} GQA-eval VG image ids (Section 9.4 artifact)")

    print(f"loading {vg_objects} ...")
    records = json.load(open(vg_objects))
    print(f"VG images with object annotations: {len(records):,}")

    counts: Counter = Counter()
    source_ids, n_skipped = [], 0
    for rec in records:
        iid = int(rec["image_id"])
        if iid in exclude:
            n_skipped += 1
            continue                             # NEVER count objects from a GQA-eval image
        source_ids.append(iid)
        for obj in rec.get("objects", []):
            names = obj.get("names") or []
            if names:
                name = names[0].strip().lower()
                if len(name) >= args.min_len and name.isascii():
                    counts[name] += 1
    print(f"counted {sum(counts.values()):,} object instances over {len(source_ids):,} images "
          f"({n_skipped:,} eval images skipped)")

    # Top-N with simple plural folding: skip a plural whose singular is already selected
    # (e.g. keep 'tree', drop 'trees'), so the N slots hold N distinct concepts.
    vocab = []
    for name, _ in counts.most_common():
        if name.endswith("s") and name[:-1] in vocab:
            continue
        vocab.append(name)
        if len(vocab) >= args.top:
            break

    out_vocab = Path(f"data/gqa/pred_vocab{suffix}.json")
    out_vocab.write_text(json.dumps({
        "vocab": vocab, "n_source_images": len(source_ids),
        "built_from": "VG objects.json top frequencies, GQA-eval images excluded",
        "top": args.top}, indent=2))
    out_ids = Path(f"data/gqa/pred_vocab{suffix}_image_ids.json")
    out_ids.write_text(json.dumps(source_ids))
    print(f"\nwrote {out_vocab} ({len(vocab)} names) and {out_ids} "
          f"({len(source_ids):,} source image ids for the leakage gate)")
    print(f"top 30: {vocab[:30]}")
    print(f"\nNEXT (mandatory): python scripts/13_vg_leakage_audit.py --train-ids {out_ids}")


if __name__ == "__main__":
    main()
