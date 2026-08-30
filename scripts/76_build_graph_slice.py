#!/usr/bin/env python3
"""Build the frozen GRAPH-MECHANISM slice: 3,000 questions with scene graphs, never yet used.

WHY A FIFTH SLICE, WHEN THE PROJECT ALREADY HAS FOUR. The oracle-degradation study asks which
controlled graph errors destroy the oracle advantage. That needs two things at once, and no
existing surface has both:

  - SCENE GRAPHS. Test-Dev has none (0 of 398 images), upstream — GQA does not publish them for
    that tier, which is why the official eval.py sets `scenes = None` for it. So the external
    benchmark cannot host this experiment at all.
  - AN UNSPENT SELECTION BUDGET. The tuning 500 has been used for epoch curves, connector choice,
    graph renderers and hedging; the confirmatory 3,000, locked 2,000, objective 4,000 and
    Test-Dev 12,578 are all spent endpoints. A mechanism result measured on any of them would be
    exploratory at best and circular at worst.

Leaving the tuning 500 as the only option would make the whole study exploratory AND underpowered:
3 encoders x 8 arms on 500 questions cannot resolve a retained-headroom difference. A new,
separately-named endpoint is the sanctioned move — the project rules forbid resampling or extending a
LOCKED artifact, not the existence of a new one.

THE EXCLUSION IS BY QUESTION, NOT BY IMAGE, AND THAT IS A DECLARED LIMITATION. Measured before
building: GQA balanced val holds 10,234 images, and the four spent slices already touch 9,206 of
them. Excluding spent images as well as spent qids leaves 1,028 images and supports at most ~950
questions, of which `relate` — the primary mechanism category — could contribute only 156. That
cannot resolve a retained-headroom difference across three encoders and eight arms, so the strict
version of this slice would buy cleanliness by guaranteeing an uninformative null.

Excluding spent QIDS only leaves 122,562 questions over 10,032 images. Those questions are new;
many of their images are not. What that costs, precisely:

  - NOT VALID as a fresh absolute-accuracy endpoint. Accuracy here may be optimistic relative to a
    truly unseen cohort, and must never be quoted as one or compared to Test-Dev.
  - VALID for what this study actually estimates. Every arm answers the same questions with the
    same frozen checkpoint, and the estimand is a WITHIN-QUESTION contrast between graph-corruption
    conditions. No corruption policy was ever tuned on any image, so prior exposure shifts all arms
    together and cancels in the difference.

Questions whose image was never touched by a spent slice are tagged in the manifest, so a strictly
clean subset analysis stays available afterwards without a second generation pass.

RELATE IS ENRICHED ON PURPOSE. Proportional stratification serves population-representative
accuracy; this slice is not for that. `relate` is where graph structure should matter most, so it
takes half the budget. The remaining half is proportional across the other categories. The mix is
declared here, before any arm is run, and the slice is not to be reweighted afterwards.

WHAT THIS SCRIPT SELECTS: nothing about any model. It samples questions, checks disjointness, and
freezes. No accuracy is read, and no checkpoint is loaded.

ONE QUESTION PER IMAGE, for the same reason as the confirmatory slice: the paired bootstrap treats
questions as independent units, and GQA val holds ~11 questions per image.

SCENE GRAPHS REQUIRED, RELATION COUNT NOT FILTERED. Every selected image must have a val scene
graph, because the oracle arms render one. Questions whose graph happens to hold few relations are
deliberately KEPT: filtering on relation count would bias the slice toward rich scenes and inflate
every relation-deletion effect. The relation distribution is recorded in the manifest instead.

    python scripts/76_build_graph_slice.py --n 3000

Writes data/gqa/graph_3000_qids.json. Refuses to overwrite: frozen once written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent


def _confirmatory_sampler():
    """Reuse the confirmatory slice's sampler verbatim — same allocation, same one-per-image rule.

    Its module name starts with a digit, so it cannot be imported by name.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_slice29", ROOT / "scripts/29_build_confirmatory_slice.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.allocate, m.sample_one_per_image


allocate, sample_one_per_image = _confirmatory_sampler()
OUT = ROOT / "data/gqa/graph_3000_qids.json"
MANIFEST = ROOT / "data/gqa/graph_3000_qids_MANIFEST.json"
QUESTIONS = ROOT / "data/gqa/questions/val_balanced_questions.json"
SCENES = ROOT / "data/gqa/val_sceneGraphs.json"
SEED = 20260808

SPENT = ("data/gqa/eval_2000_qids.json", "data/gqa/tune_500_qids.json",
         "data/gqa/confirm_3000_qids.json", "data/gqa/objective_4000_qids.json")


from src.data.gqa import category_of  # noqa: E402

# Half the budget to `relate`, the primary mechanism category; the rest proportional to the pool.
RELATE_SHARE = 0.5


def category(q: dict) -> str:
    """The project's reasoning category — the same function every other slice and table uses."""
    t = q.get("types", {})
    return category_of(q["question"], t.get("structural", ""), t.get("semantic", ""),
                       t.get("detailed", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()

    if OUT.exists():
        sys.exit(f"FAIL {OUT.name} already exists. This slice is frozen once written; a different "
                 "sample needs a different name, not an overwrite.")

    allq = json.loads(QUESTIONS.read_text())
    scenes = json.loads(SCENES.read_text())

    # Exclude spent QUESTIONS. Spent IMAGES are recorded but not excluded — see the header: doing
    # both leaves 1,028 images and 156 relate questions, which cannot answer the question this
    # slice exists for. The images that were previously touched are tagged instead.
    spent_qids, spent_imgs = set(), set()
    per_slice = {}
    for rel in SPENT:
        s = set(json.loads((ROOT / rel).read_text()))
        per_slice[rel] = len(s)
        spent_qids |= s
        spent_imgs |= {allq[q]["imageId"] for q in s if q in allq}

    by_cat_img, pool_imgs, no_graph = defaultdict(lambda: defaultdict(list)), set(), 0
    for qid, q in allq.items():
        img = q["imageId"]
        if qid in spent_qids:
            continue
        if img not in scenes:
            no_graph += 1
            continue
        by_cat_img[category(q)][img].append(qid)
        pool_imgs.add(img)

    # Declared stratification: relate takes RELATE_SHARE, the remainder is proportional.
    counts = {c: sum(len(v) for v in by_cat_img[c].values()) for c in by_cat_img}
    if "relate" not in counts:
        sys.exit("FAIL no `relate` questions in the pool; this slice exists to study relations")
    n_relate = round(a.n * RELATE_SHARE)
    rest = {c: v for c, v in counts.items() if c != "relate"}
    targets = allocate(rest, a.n - n_relate)
    targets["relate"] = n_relate
    for c, t in targets.items():
        if t > len(by_cat_img[c]):
            sys.exit(f"FAIL category '{c}' needs {t} images but only {len(by_cat_img[c])} are "
                     "available under the one-question-per-image rule")

    rng = random.Random(SEED)
    chosen = sorted(sample_one_per_image(by_cat_img, targets, rng))

    if len(chosen) != a.n or len(set(chosen)) != a.n:
        sys.exit(f"FAIL sampled {len(chosen)} ({len(set(chosen))} unique), wanted {a.n}")
    imgs = {allq[q]["imageId"] for q in chosen}
    if len(imgs) != a.n:
        sys.exit(f"FAIL {a.n} questions span only {len(imgs)} images; one-per-image was violated")

    # Question-level disjointness is enforced; image-level overlap is measured and declared.
    disjoint = {}
    for rel in SPENT:
        other = set(json.loads((ROOT / rel).read_text()))
        other_imgs = {allq[q]["imageId"] for q in other if q in allq}
        disjoint[rel] = {"shared_qids": len(set(chosen) & other),
                         "shared_images": len(imgs & other_imgs)}
        if disjoint[rel]["shared_qids"]:
            sys.exit(f"FAIL question overlap with {rel}: {disjoint[rel]['shared_qids']} qids")

    # Which of the chosen questions sit on images no spent slice ever touched. Recorded so a
    # strictly clean subset analysis is possible later without generating anything again.
    clean = sorted(q for q in chosen if allq[q]["imageId"] not in spent_imgs)
    clean_mix = dict(Counter(category(allq[q]) for q in clean).most_common())

    missing = [i for i in imgs if not (ROOT / "data/gqa/images" / f"{i}.jpg").is_file()]
    if missing:
        sys.exit(f"FAIL {len(missing)} images absent, e.g. {missing[:5]}")
    ungraphed = [i for i in imgs if i not in scenes]
    if ungraphed:
        sys.exit(f"FAIL {len(ungraphed)} selected images have no scene graph")

    rels = [len(o.get("relations", []))
            for i in imgs for o in scenes[i].get("objects", {}).values()]
    per_img_rel = sorted(sum(len(o.get("relations", []))
                             for o in scenes[i].get("objects", {}).values()) for i in imgs)
    per_img_obj = sorted(len(scenes[i].get("objects", {})) for i in imgs)

    def pct(xs, p):
        return xs[min(len(xs) - 1, int(p * len(xs)))]

    OUT.write_text(json.dumps(chosen, indent=1) + "\n")
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    MANIFEST.write_text(json.dumps({
        "slice": OUT.name,
        "purpose": "mechanism surface for the oracle-to-degraded scene-graph study; requires "
                   "scene graphs, which Test-Dev does not publish and which the spent internal "
                   "slices cannot supply without circularity",
        "status": "FROZEN AND PRE-REGISTERED — no model has been run on it",
        "selection_permitted": False,
        "exclusion": "spent QIDS only; spent IMAGES are not excluded",
        "limitation": "Questions are new but many images were seen by a spent slice. Valid for "
                      "within-question graph-corruption contrasts on frozen checkpoints, where "
                      "prior exposure shifts every arm together and cancels. NOT valid as a fresh "
                      "absolute-accuracy endpoint; never quote its accuracy as one, and never "
                      "compare it to Test-Dev.",
        "why_not_strict": "Excluding spent images too leaves 1,028 images and at most ~950 "
                          "questions, of which relate could supply only 156 — too few to resolve "
                          "a retained-headroom difference across 3 encoders and 8 arms.",
        "spent_surface_image_disjoint_subset": {
            "n": len(clean),
            "category_mix": clean_mix,
            "definition": "questions whose image appears in no spent slice",
            "use": "a direction check, not a sensitivity analysis. It is too small to carry a "
                   "per-category claim — agreement in sign is mildly reassuring, disagreement "
                   "would be worth investigating, and a null here means nothing.",
            "qids": clean,
        },
        "stratification": {"relate_share": RELATE_SHARE,
                           "why": "relate is the primary mechanism category; the slice is not "
                                  "population-representative and must not be reweighted to look "
                                  "like one",
                           "targets": targets},
        "n": len(chosen), "unique_images": len(imgs), "seed": SEED,
        "builder": "scripts/76_build_graph_slice.py",
        "sampler": "scripts/29_build_confirmatory_slice.py:sample_one_per_image",
        "source": str(QUESTIONS.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(QUESTIONS.read_bytes()).hexdigest(),
        "scene_graphs": str(SCENES.relative_to(ROOT)),
        "scene_graphs_sha256": hashlib.sha256(SCENES.read_bytes()).hexdigest(),
        "excluded_slices": per_slice,
        "excluded_qids": len(spent_qids), "excluded_images": len(spent_imgs),
        "questions_dropped_for_missing_scene_graph": no_graph,
        "candidate_pool_questions": sum(counts.values()),
        "candidate_pool_images": len(pool_imgs),
        "category_mix": dict(Counter(category(allq[q]) for q in chosen).most_common()),
        "disjoint_from": disjoint,
        "scene_graph_stats": {
            "relations_per_image": {"min": per_img_rel[0], "p25": pct(per_img_rel, .25),
                                    "median": pct(per_img_rel, .5), "p75": pct(per_img_rel, .75),
                                    "max": per_img_rel[-1],
                                    "mean": round(sum(per_img_rel) / len(per_img_rel), 2),
                                    "images_with_zero": sum(1 for r in per_img_rel if r == 0)},
            "objects_per_image": {"min": per_img_obj[0], "median": pct(per_img_obj, .5),
                                  "max": per_img_obj[-1],
                                  "mean": round(sum(per_img_obj) / len(per_img_obj), 2)},
            "total_relation_edges": sum(rels),
            "note": "relation count was NOT used as a selection criterion; filtering on it would "
                    "bias the slice toward rich scenes and inflate every relation-deletion effect",
        },
        "sha256": sha,
    }, indent=1) + "\n")

    print(f"wrote {OUT.relative_to(ROOT)}  n={len(chosen)}  images={len(imgs)}  sha256 {sha[:16]}...")
    print(f"  candidate pool     : {sum(counts.values()):,} questions over {len(pool_imgs):,} images")
    print(f"  excluded           : {len(spent_qids):,} spent qids, {len(spent_imgs):,} spent images")
    print(f"  dropped, no graph  : {no_graph:,} questions")
    print(f"  category mix       : {dict(Counter(category(allq[q]) for q in chosen).most_common())}")
    print(f"  relations/image    : median {pct(per_img_rel, .5)}, mean "
          f"{sum(per_img_rel)/len(per_img_rel):.2f}, {sum(1 for r in per_img_rel if r == 0)} with zero")
    shared_img = len(imgs & spent_imgs)
    print(f"  question overlap   : 0 with every spent slice (enforced)")
    print(f"  image overlap      : {shared_img:,} of {len(imgs):,} images were touched by a spent "
          f"slice — DECLARED, not an error")
    print(f"  strictly clean     : {len(clean)} questions on never-seen images "
          f"({clean_mix.get('relate', 0)} relate) — sensitivity subset only")
    print("  NOT a fresh absolute-accuracy endpoint; valid for within-question corruption contrasts")


if __name__ == "__main__":
    main()
