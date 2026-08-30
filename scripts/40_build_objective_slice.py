"""Build a FOURTH frozen slice: 4,000 questions, disjoint from all three existing slices.

WHY A FOURTH SLICE. The tuning 500 has been used for selection many times over, and both
the confirmatory 3,000 and the locked 2,000 are spent — the locked set is the reported final
endpoint and must never be touched again. The training-objective A/B (token-normalised vs
sequence-normalised loss) therefore has no valid endpoint left. This slice supplies one. It
is drawn from GQA val_balanced questions that no slice has ever used, at the image level.

WHY 4,000. Measured, not guessed. The corrected layer's confirmatory interaction at n=3,000
came in at -2.87 [-5.13, -0.47] on the confirmatory surface, a half-width of 2.33 points. An
interaction is a difference of differences and carries roughly twice the variance of a
marginal paired contrast, so the comparable paired half-width at n=3,000 is near 1.65 points.
Scaling as 1/sqrt(n) gives about 1.43 at n=4,000 and 1.28 at n=5,000: the last thousand
questions buy 0.15 points. They also cost all the slack — only 5,028 unused images exist, so
n=5,000 would consume 99.4% of them and leave the one-question-per-image sampler nowhere to
go when categories compete for the same image. 4,000 keeps roughly a thousand images spare.

WHAT THIS SLICE CANNOT FIX. A larger slice shrinks question-sampling error only. It does
nothing to seed-to-seed variance, which on this project runs at a mean within-recipe
discordance of 19.417%. With three seeds per arm the conclusion is limited by seed spread,
not by n, which is why the decision rule for the objective A/B is replication (consistent
sign across all three seed pairs) plus per-seed paired intervals — never a pooled p-value
over the three runs, which would treat seeds as independent questions.

ONE QUESTION PER IMAGE, as in the confirmatory slice: the paired bootstrap resamples
questions as independent units, and ~10 questions per image would break that.

    python scripts/40_build_objective_slice.py --n 4000

Writes data/gqa/objective_4000_qids.json plus a manifest. Refuses to overwrite either.
Once a model has been scored on this slice for selection, it is spent for selection, exactly
as the tuning 500 is; a later experiment needing a clean endpoint draws a new slice from the
~46,000 questions this one leaves behind.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.gqa import GQADataset  # noqa: E402
from src.utils import load_config  # noqa: E402

# Every slice already drawn. Image-level exclusion against all three is the point of the file.
EXISTING_SLICES = (
    "data/gqa/eval_2000_qids.json",
    "data/gqa/tune_500_qids.json",
    "data/gqa/confirm_3000_qids.json",
)


def confirmatory_builder():
    """Load the confirmatory slice builder to reuse its sampler, rather than copy it.

    Importing keeps one implementation of the allocation and one-per-image draw. A copy
    could drift from the algorithm that produced the confirmatory slice, and then two
    slices claiming the same construction would not actually share one.
    """
    path = ROOT / "scripts" / "29_build_confirmatory_slice.py"
    spec = importlib.util.spec_from_file_location("confirmatory_slice", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    """Sample, audit and freeze the objective-experiment slice."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=4000, help="slice size (questions)")
    ap.add_argument("--seed", type=int, default=20260803,
                    help="construction seed, recorded in the manifest and never re-rolled")
    ap.add_argument("--out", default="data/gqa/objective_4000_qids.json")
    args = ap.parse_args()

    builder = confirmatory_builder()
    out = Path(args.out)
    manifest_path = out.with_name(out.stem + "_MANIFEST.json")
    for path in (out, manifest_path):
        if path.exists():
            raise SystemExit(f"refusing to overwrite the frozen slice at {path} — "
                             "re-rolling it after seeing any result would destroy its purpose")

    cfg = load_config()
    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])

    used_qids, used_images, excluded = set(), set(), {}
    for path in EXISTING_SLICES:
        qids = json.load(open(ROOT / path))
        used_qids |= set(qids)
        used_images |= {ds.get(q).image_id for q in qids}
        excluded[path] = len(qids)
        print(f"excluding {path}: {len(qids)} qids")
    print(f"excluded totals: {len(used_qids)} qids over {len(used_images)} images")

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
    pool_images = {i for m in by_cat_img.values() for i in m}
    per_cat = {c: sum(len(v) for v in imgs.values()) for c, imgs in by_cat_img.items()}
    print(f"candidate pool: {pool_n:,} questions over {len(pool_images):,} images")

    if args.n > len(pool_images):
        raise SystemExit(f"n={args.n} exceeds the {len(pool_images):,} available images; "
                         "one question per image cannot be satisfied")

    targets = builder.allocate(per_cat, args.n)
    print(f"\nproportional targets (seed {args.seed}):")
    for c, t in sorted(targets.items(), key=lambda kv: -kv[1]):
        print(f"  {c:10} {t:5}  ({100 * per_cat[c] / pool_n:4.1f}% of pool, "
              f"{len(by_cat_img[c]):,} images available)")

    rng = random.Random(args.seed)
    sample = sorted(builder.sample_one_per_image(by_cat_img, targets, rng))

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
    # Per-slice overlap too, so a failure names the slice it collided with.
    for path in EXISTING_SLICES:
        other = set(json.load(open(ROOT / path)))
        if set(sample) & other:
            problems.append(f"qid overlap with {path}: {len(set(sample) & other)}")
    if problems:
        raise SystemExit("FAIL slice audit:\n  " + "\n  ".join(problems))

    payload = json.dumps(sample)
    print("\nAUDIT PASS")
    print(f"  questions            {len(sample)}")
    print(f"  unique images        {len(set(images))}  (1.00 questions/image)")
    print(f"  qid overlap          0 with each of {len(EXISTING_SLICES)} existing slices")
    print(f"  image overlap        0 with all existing slices")
    print(f"  category mix         {dict(cats.most_common())}")

    out.write_text(payload)
    manifest = {
        "slice": out.name,
        "purpose": "held-out endpoint for the training-objective A/B (token vs sequence "
                   "normalised loss); spent for selection once scored",
        "n": len(sample),
        "seed": args.seed,
        "builder": "scripts/40_build_objective_slice.py",
        "sampler": "scripts/29_build_confirmatory_slice.py:sample_one_per_image",
        "source": cfg["gqa"]["questions"]["val_balanced"],
        "excluded_slices": excluded,
        "excluded_qids": len(used_qids),
        "excluded_images": len(used_images),
        "candidate_pool_questions": pool_n,
        "candidate_pool_images": len(pool_images),
        "unique_images": len(set(images)),
        "category_mix": dict(cats.most_common()),
        "sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {out}  (sha256 {manifest['sha256'][:16]}...)")
    print(f"wrote {manifest_path}")
    print("FROZEN. Commit both files BEFORE any model is scored on this slice.")


if __name__ == "__main__":
    main()
