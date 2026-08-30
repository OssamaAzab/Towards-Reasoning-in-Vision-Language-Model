"""Build the frozen CONFIRMATORY slice: ~3,000 questions, disjoint from tune-500 and locked-2000.

WHY A THIRD SLICE. The tuning 500 has now been used for many exploratory decisions (epoch
curves, connector choice, predicted-graph renderers, hedging), so a hypothesis confirmed on it
is confirmed on the data that suggested it. The locked 2,000 is the untouched final endpoint
and must stay that way. This slice sits between them: frozen before use, large enough to test
the corrected connector x encoder interaction, and never used for selection.

WHY ~3,000 AND NOT 500. The interaction is a difference of differences, so it carries roughly
twice the variance of either marginal contrast. Measured on the legacy layer at n=2,000
(outputs/eval/metric_ci_interaction.csv) its 95% half-widths were 1.85 (exact) to 2.95
(VQA-soft) points. Those scale as 1/sqrt(n), so at n=500 they become ~3.7 to ~5.9 — as wide as
the effect being hunted, which is why the tuning-slice interaction is expected to be
inconclusive. At n=3,000 they fall back to ~1.5 to ~2.4, which can resolve an effect of the
legacy magnitude (~5 points).

ONE QUESTION PER IMAGE. The candidate pool holds 88,251 questions over only 8,028 images —
about 11 questions per image. Sampling questions freely would put many questions behind the
same image, and the paired bootstrap resamples questions as if they were independent units.
Restricting to one question per image makes that assumption hold rather than merely assuming
it, at no cost: 3,000 of 8,028 images is comfortably feasible.

PROPORTIONAL STRATIFICATION. Category shares follow the candidate pool, so the aggregate
estimate stays unbiased for the population and comparable to the other slices. The cost is
that `compare` lands near 96 questions, which is too few for a per-category claim; per-category
`compare` results on this slice are descriptive only. A confirmatory compare result needs a
separate, clearly-labelled top-up set, never a reweighting of this one.

    python scripts/29_build_confirmatory_slice.py --n 3000

Writes data/gqa/confirm_3000_qids.json. Refuses to overwrite: the set is frozen once written.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset  # noqa: E402
from src.utils import load_config  # noqa: E402

EXISTING_SLICES = ("data/gqa/eval_2000_qids.json", "data/gqa/tune_500_qids.json")


def allocate(counts: dict, total: int) -> dict:
    """Largest-remainder proportional allocation, so the parts sum to exactly `total`."""
    pool = sum(counts.values())
    exact = {k: total * v / pool for k, v in counts.items()}
    base = {k: int(v) for k, v in exact.items()}
    for k, _ in sorted(exact.items(), key=lambda kv: kv[1] - int(kv[1]), reverse=True):
        if sum(base.values()) >= total:
            break
        base[k] += 1
    return base


def sample_one_per_image(by_cat_img: dict, targets: dict, rng: random.Random) -> list:
    """Pick `targets[cat]` questions per category, never reusing an image.

    Scarce categories are filled first: `compare` can draw from far fewer images than
    `relate`, so allocating the plentiful categories first would starve it.
    """
    used_images, chosen = set(), []
    for cat in sorted(targets, key=lambda c: targets[c]):
        images = sorted(by_cat_img[cat])
        rng.shuffle(images)
        taken = 0
        for img in images:
            if taken >= targets[cat]:
                break
            if img in used_images:
                continue
            used_images.add(img)
            chosen.append(rng.choice(sorted(by_cat_img[cat][img])))
            taken += 1
        if taken < targets[cat]:
            raise SystemExit(
                f"FAIL only {taken} images available for category '{cat}', needed "
                f"{targets[cat]} — the one-question-per-image constraint cannot be met")
    return chosen


def main() -> None:
    """Sample, audit and freeze the confirmatory slice."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=3000, help="slice size (questions)")
    ap.add_argument("--seed", type=int, default=20260731,
                    help="construction seed, recorded in the manifest and never re-rolled")
    ap.add_argument("--out", default="data/gqa/confirm_3000_qids.json")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite the frozen slice at {out} — "
                         "re-rolling it after seeing any result would destroy its purpose")

    cfg = load_config()
    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])

    used_qids, used_images = set(), set()
    for path in EXISTING_SLICES:
        qids = json.load(open(path))
        used_qids |= set(qids)
        used_images |= {ds.get(q).image_id for q in qids}
        print(f"excluding {path}: {len(qids)} qids")
    print(f"excluded totals: {len(used_qids)} qids over {len(used_images)} images")

    # Candidates are disjoint from the existing slices at the IMAGE level, which is stricter
    # than qid-disjoint and is the same standard the train/eval wall uses.
    by_cat_img = defaultdict(lambda: defaultdict(list))
    pool_n = 0
    for q in ds.qids:
        if q in used_qids:
            continue
        ex = ds.get(q)
        if ex.image_id in used_images or ex.image_path is None:
            continue
        by_cat_img[ds.category_of(ex)][ex.image_id].append(q)
        pool_n += 1
    per_cat = {c: sum(len(v) for v in imgs.values()) for c, imgs in by_cat_img.items()}
    print(f"candidate pool: {pool_n:,} questions over "
          f"{len({i for m in by_cat_img.values() for i in m}):,} images")

    targets = allocate(per_cat, args.n)
    print(f"\nproportional targets (seed {args.seed}):")
    for c, t in sorted(targets.items(), key=lambda kv: -kv[1]):
        print(f"  {c:10} {t:5}  ({100 * per_cat[c] / pool_n:4.1f}% of pool, "
              f"{len(by_cat_img[c]):,} images available)")

    rng = random.Random(args.seed)
    sample = sorted(sample_one_per_image(by_cat_img, targets, rng))

    # ---- audit: every property this slice claims, verified rather than assumed ----
    images = [ds.get(q).image_id for q in sample]
    cats = Counter(ds.category_of(ds.get(q)) for q in sample)
    problems = []
    if len(sample) != args.n:
        problems.append(f"size {len(sample)} != {args.n}")
    if len(set(sample)) != len(sample):
        problems.append(f"duplicate qids: {len(sample) - len(set(sample))}")
    if set(sample) & used_qids:
        problems.append(f"qid overlap with existing slices: {len(set(sample) & used_qids)}")
    if set(images) & used_images:
        problems.append(f"image overlap with existing slices: {len(set(images) & used_images)}")
    if len(set(images)) != len(images):
        problems.append(f"repeated images within slice: {len(images) - len(set(images))}")
    if problems:
        raise SystemExit("FAIL slice audit:\n  " + "\n  ".join(problems))

    print(f"\nAUDIT PASS")
    print(f"  questions            {len(sample)}")
    print(f"  unique images        {len(set(images))}  (1.00 questions/image)")
    print(f"  qid overlap          0 with tune-500, 0 with locked-2000")
    print(f"  image overlap        0 with tune-500, 0 with locked-2000")
    print(f"  category mix         {dict(cats.most_common())}")

    out.write_text(json.dumps(sample))
    print(f"\nwrote {out}")
    print("FROZEN. Commit this file and this script BEFORE any model is scored on it.")


if __name__ == "__main__":
    main()
