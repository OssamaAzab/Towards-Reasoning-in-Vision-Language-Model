#!/usr/bin/env python3
"""Freeze GQA balanced Test-Dev as an immutable evaluation endpoint.

WHY THIS IS NOT A SAMPLER. Every other slice in this project is drawn from GQA's validation
split, so it needs a seed, an exclusion list and a sampling rule. Test-Dev is different: it is an
EXTERNAL benchmark that ships as a fixed set, and the whole point of using it is that we did not
choose its membership. So this script selects nothing — it records the full 12,578 questions in a
deterministic order and hashes them, exactly so that any later run can prove it evaluated the
benchmark rather than a convenient subset of it.

WHAT IT CHECKS BEFORE WRITING.
  - every question resolves to an image that exists on disk;
  - the cohort is disjoint from every spent validation slice (it is drawn from a different split,
    so this must hold trivially — a non-zero intersection would mean the files are not what their
    names say);
  - no Test-Dev image appears in any validation-derived slice, which would be leakage across the
    train/eval wall in the other direction.

The `sha256` is over the written qid list. Launchers re-check it, so a silently edited endpoint
fails the run instead of quietly producing a different number.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset, category_of          # noqa: E402
from src.utils import load_config                          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_QIDS = ROOT / "data/gqa/testdev_12578_qids.json"
OUT_MANIFEST = ROOT / "data/gqa/testdev_12578_qids_MANIFEST.json"

# Validation-derived slices. Test-Dev must share no question and no IMAGE with any of them.
SPENT = ["data/gqa/eval_2000_qids.json", "data/gqa/tune_500_qids.json",
         "data/gqa/confirm_3000_qids.json", "data/gqa/objective_4000_qids.json"]


def main() -> None:
    cfg = load_config()
    qpath = cfg["gqa"]["questions"]["testdev_balanced"]
    ds = GQADataset(qpath, cfg["gqa"]["image_dirs"])
    raw = json.loads(pathlib.Path(qpath).read_text())

    # Sorted, so the file is a property of the benchmark and not of dict iteration order.
    qids = sorted(ds.qids)
    if len(qids) != len(set(qids)):
        sys.exit("FAIL duplicate qids in the Test-Dev questions file")

    unresolved, images = [], set()
    for q in qids:
        e = ds.get(q)
        if e.image_path is None:
            unresolved.append(q)
        else:
            images.add(e.image_id)
    if unresolved:
        sys.exit(f"FAIL {len(unresolved)} Test-Dev questions have no image on disk, "
                 f"e.g. {unresolved[:5]}")

    # Disjointness. Both directions matter: a shared qid would mean the files are mislabelled,
    # a shared image would mean this "external" benchmark overlaps a slice we already spent.
    overlaps = {}
    for rel in SPENT:
        p = ROOT / rel
        if not p.is_file():
            sys.exit(f"FAIL spent slice missing, cannot prove disjointness: {p}")
        other = set(json.loads(p.read_text()))
        val = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
        other_imgs = {val._raw[q]["imageId"] for q in other if q in val._raw}
        overlaps[rel] = {"shared_qids": len(set(qids) & other),
                         "shared_images": len(images & other_imgs)}
    bad = {k: v for k, v in overlaps.items() if v["shared_qids"] or v["shared_images"]}
    if bad:
        sys.exit(f"FAIL Test-Dev overlaps a spent validation slice: {bad}")

    cats = Counter(category_of(raw[q]["question"], raw[q]["types"]["structural"],
                               raw[q]["types"]["semantic"], raw[q]["types"]["detailed"])
                   for q in qids)
    # GQA's own binary/open split, taken from eval.py: open iff structural == "query".
    answer_type = Counter("open" if raw[q]["types"]["structural"] == "query" else "binary"
                          for q in qids)

    body = json.dumps(qids)
    OUT_QIDS.write_text(body)
    digest = hashlib.sha256(OUT_QIDS.read_bytes()).hexdigest()

    manifest = {
        "slice": OUT_QIDS.name,
        "purpose": "external benchmark endpoint — official GQA balanced Test-Dev, evaluated whole",
        "selection": "NONE — the full published split is used; no sampling, no seed, no exclusion",
        "n": len(qids),
        "unique_images": len(images),
        "builder": "scripts/70_build_testdev_slice.py",
        "source": str(pathlib.Path(qpath).relative_to(ROOT)),
        "source_sha256": hashlib.sha256(pathlib.Path(qpath).read_bytes()).hexdigest(),
        "evidence_layer": "corrected_chatml_v1_eos",
        "scored_by": "src/eval/metrics.py exact_full + the official GQA eval.py "
                     "(scripts/vendor/gqa_eval.py)",
        "disjoint_from": overlaps,
        "category_mix": dict(cats.most_common()),
        "answer_type_mix_gqa_official": dict(answer_type),
        "gold_answers_public": True,
        "scene_graphs_available": False,
        "scene_graph_note": "GQA ships no scene graphs for Test-Dev images, so oracle-graph and "
                            "oracle-crop arms are impossible here; the official eval.py sets "
                            "scenes/choices to None for this tier for the same reason.",
        "sha256": digest,
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=1) + "\n")

    print(f"wrote {OUT_QIDS.relative_to(ROOT)}  n={len(qids)}  sha256={digest}")
    print(f"wrote {OUT_MANIFEST.relative_to(ROOT)}")
    print(f"  images            : {len(images)}")
    print(f"  category mix      : {dict(cats.most_common())}")
    print(f"  binary/open (GQA) : {dict(answer_type)}")
    print(f"  disjointness      : {overlaps}")


if __name__ == "__main__":
    main()
