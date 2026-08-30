"""Stage 2b: build a TUNING slice disjoint (by image) from the locked evaluation set.

Iterating the predicted-graph generator against the locked 2,000-qid set would be tuning on
the test set. This script samples N val_balanced questions whose images do NOT appear behind
any locked qid (image-level disjointness, the same standard as the train/eval wall), giving
a slice where generator variants can be compared freely. The locked set is then run ONCE,
with the frozen winner.

    python scripts/17_build_tuning_slice.py --n 500
"""
import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset  # noqa: E402
from src.utils import load_config  # noqa: E402


def main() -> None:
    """Sample a locked-set-image-disjoint tuning slice and write its qid list."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=500, help="tuning-slice size (questions)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="data/gqa/tune_500_qids.json")
    args = ap.parse_args()

    cfg = load_config()
    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
    locked = json.load(open("data/gqa/eval_2000_qids.json"))
    locked_images = {ds.get(q).image_id for q in locked}
    locked_qids = set(locked)
    print(f"locked set: {len(locked)} qids over {len(locked_images)} images")

    # Candidates: image not behind ANY locked qid (stricter than qid-disjoint), image usable.
    rng = random.Random(args.seed)
    pool = [q for q in ds.qids
            if q not in locked_qids and ds.get(q).image_id not in locked_images
            and ds.get(q).image_path is not None]
    print(f"candidate pool (image-disjoint from locked set): {len(pool):,} questions")
    sample = rng.sample(pool, args.n)

    # Sanity: zero image overlap with the locked set, by construction — verify anyway.
    overlap = {ds.get(q).image_id for q in sample} & locked_images
    assert not overlap, f"image overlap with locked set: {sorted(overlap)[:5]}"

    cats = Counter(ds.category_of(ds.get(q)) for q in sample)
    Path(args.out).write_text(json.dumps(sample))
    print(f"wrote {args.out}: {len(sample)} qids, "
          f"{len({ds.get(q).image_id for q in sample})} unique images, 0 locked-image overlap")
    print(f"category mix: {dict(cats.most_common())}")


if __name__ == "__main__":
    main()
