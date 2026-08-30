"""Build the RelTR experiment's condition caches. CPU only, no model is loaded.

    python scripts/46_reltr_conditions.py

FOUR CONDITIONS, THREE EVALUATOR CALLS.
  1. no_graph            — the B side of every paired call, so it needs no cache
  2. reltr               — the learned graph for the question's OWN image
  3. reltr_wrong_image   — a distribution-matched graph from a DIFFERENT image
  4. oracle              — the image's ground-truth GQA graph, DESCRIPTIVE HEADROOM ONLY

THE PRIMARY IS 2 MINUS 3, AND WHY. A graph that helps as much when it describes a different
picture is being used as generic text, not as evidence about this one. T-049 asked exactly this
of an OWLv2 + box-geometry graph and got +0.00 [-3.00, +3.00]; the ceiling in the same run was
+17.20, which localised the bottleneck to graph quality rather than to the injection mechanism.
This experiment changes only the graph source.

THE CONTROL IS A DERANGEMENT, NOT NEAREST-NEIGHBOUR PICKING. Choosing the closest donor per
question independently collapses onto popular donors and makes the control's graph distribution
unlike the treatment's. Sorting by profile and swapping ADJACENT pairs is a perfect matching:
every graph is used exactly once, the map is an involution, and partners are by construction the
most similar available. Matched on object count, relation count and rendered token length —
never on the answer.

THE ORACLE IS A PRIVILEGED CEILING. It injects the structure GQA's questions were generated
from. It exists here to prove the injection channel is open and worth something; it is forbidden
as a primary, promotion or inferential secondary comparator, and the analysis enforces that.

RENDERED BY THE SAME STORE AS T-049. Every arm is read at evaluation time by
`PredictedGraphStore`, so the arms differ in graph CONTENT and in nothing else.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact import ARTIFACT_VERSION  # noqa: E402
from src.augment import PredQPlainAugment, SceneGraphAugment  # noqa: E402
from src.data.predicted_graph import PredictedGraphStore  # noqa: E402
from src.eval.metrics import METRIC_VERSION  # noqa: E402
from src.models.revisions import revision_for  # noqa: E402
from src.utils import load_config  # noqa: E402

SEED = 42
LEN_TOL = 0.10
TREATMENT, CONTROL, ORACLE_ARM = "reltr", "reltr_wrong_image", "oracle"


def sha256_file(p: Path) -> str:
    """Hash a file so every cache is pinned in the manifest."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def counts(graph: dict) -> tuple[int, int]:
    """(object count, relation count) — the quantities the donor match is made on."""
    objs = graph.get("objects", {})
    return len(objs), sum(len(o.get("relations", [])) for o in objs.values())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qids", default="data/gqa/tune_500_reltr_clean_qids.json")
    ap.add_argument("--graphs", default="outputs/reltr_sgg/graphs_clean.json")
    ap.add_argument("--out", default="outputs/reltr_conditions")
    ap.add_argument("--checkpoint", default="bridge_clip_w1_bf16_v1_ep5")
    ap.add_argument("--tokenizer", default=None)
    args = ap.parse_args()

    for bad in ("eval_2000", "locked", "confirm_3000", "objective_4000"):
        if bad in args.qids:
            raise SystemExit(f"FAIL {args.qids!r} names a slice this experiment must not touch")

    out = ROOT / args.out
    if out.exists() and any(out.glob("*.json")):
        raise SystemExit(f"FAIL {out} already holds caches; refusing to overwrite a built "
                         f"condition set (outputs/ is gitignored, so it is unrecoverable)")

    cfg = load_config()
    qids = [str(q) for q in json.loads((ROOT / args.qids).read_text())]
    allq = json.loads(Path(cfg["gqa"]["questions"]["val_balanced"]).read_text())
    missing = [q for q in qids if q not in allq]
    if missing:
        raise SystemExit(f"FAIL {len(missing)} qids absent from the GQA question file")
    q2img = {q: str(allq[q]["imageId"]) for q in qids}
    del allq

    graphs_img = json.loads((ROOT / args.graphs).read_text())

    # The join T-049 documented: the graph cache is keyed by IMAGE, the arms by QUESTION. A
    # direct key intersection returns zero and would silently compare nothing.
    joined = {q: graphs_img[q2img[q]] for q in qids if q2img[q] in graphs_img}
    if not joined:
        raise SystemExit("FAIL qid->image join produced ZERO graphs; the caches use different "
                         "key spaces and a direct intersection compares nothing")
    direct = set(qids) & set(graphs_img)
    print(f"join: {len(joined)}/{len(qids)} questions have an image-keyed RelTR graph "
          f"({len(set(q2img.values()))} unique images; cache holds {len(graphs_img)})")
    print(f"  sanity: direct qid-vs-image-cache intersection = {len(direct)} (expected 0)")
    if len(joined) != len(qids):
        raise SystemExit(f"FAIL only {len(joined)}/{len(qids)} questions joined; a partial "
                         f"treatment arm would not be comparable to a complete control")

    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    from transformers import AutoTokenizer
    tok_id = args.tokenizer or cfg["models"]["llm"]
    tok = AutoTokenizer.from_pretrained(tok_id, revision=revision_for(tok_id))

    variants, manifest_arms = {}, {}

    def emit(name: str, graphs: dict, **extra):
        """Write one condition cache keyed by QID and record its hash and token stats."""
        p = out / f"{name}.json"
        p.write_text(json.dumps(graphs))
        # The store the EVALUATION actually uses: it strips attributes at load time, so measuring
        # with anything else reports a graph nobody will be shown.
        s = PredictedGraphStore(str(p))
        lens = [len(tok(s.describe(q), add_special_tokens=False)["input_ids"]) for q in graphs]
        variants[name] = graphs
        manifest_arms[name] = {
            "path": str(p.relative_to(ROOT)), "sha256": sha256_file(p),
            "n_questions": len(graphs),
            # The evaluator counts a question as context-injected when describe() is truthy.
            # Measuring the same predicate here makes the analysis check an equality.
            "n_context_available": sum(1 for q in graphs if s.describe(q)),
            "rendered_tokens": {"mean": float(np.mean(lens)), "median": float(np.median(lens)),
                                "min": int(min(lens)), "max": int(max(lens))},
            **extra}
        inj = manifest_arms[name]["n_context_available"]
        print(f"  {name:20s} n={len(graphs):4d}  injected={inj:4d} "
              f"({100 * inj / len(graphs):5.1f}%)  tokens mean={np.mean(lens):6.1f}")
        return lens

    print("\nbuilding condition caches (all keyed by QID so every arm shares one key space):")
    base_lens = emit(TREATMENT, dict(joined),
                     source="RelTR Open Images V6, correct image",
                     graphs_manifest=str(Path(args.graphs).with_name(
                         Path(args.graphs).stem + "_MANIFEST.json")))

    # ---- the distribution-matched wrong-image donor, as a DERANGEMENT ----
    lens_by_q = {q: base_lens[i] for i, q in enumerate(joined)}
    prof = {q: (*counts(g), lens_by_q[q]) for q, g in joined.items()}
    pool = sorted(joined, key=lambda q: (prof[q][0], prof[q][1], prof[q][2], q))
    donor = {}
    for i in range(0, len(pool) - 1, 2):
        a, b = pool[i], pool[i + 1]
        donor[a], donor[b] = b, a
    if len(pool) % 2:                       # odd tail: 3-cycle the last three
        a, b, c = pool[-3], pool[-2], pool[-1]
        donor[a], donor[b], donor[c] = b, c, a

    # Repair any pair that happens to share an image by crossing with another pair. 485 questions
    # sit on 466 images, so some questions genuinely share one and this is not hypothetical.
    def shares(q):
        return q2img[donor[q]] == q2img[q]
    for q in pool:
        if not shares(q):
            continue
        for r in pool:
            if r == q or shares(r):
                continue
            dq, dr = donor[q], donor[r]
            if q2img[dr] != q2img[q] and q2img[dq] != q2img[r] and dr != q and dq != r:
                donor[q], donor[dr] = dr, q
                donor[r], donor[dq] = dq, r
                break
    bad = [q for q in pool if q2img[donor[q]] == q2img[q]]
    if bad:
        raise SystemExit(f"FAIL the control is not a control: {len(bad)} donors share their own "
                         f"image. First: {bad[:3]}")
    if len(set(donor.values())) != len(pool):
        raise SystemExit(f"FAIL donor set is not one-to-one: {len(set(donor.values()))} distinct "
                         f"donors for {len(pool)} questions")
    if set(donor.values()) != set(pool):
        raise SystemExit("FAIL the donor map is not a permutation of the question set")
    if any(donor[q] == q for q in pool):
        raise SystemExit("FAIL a question donated its own graph; that is not a control")

    exact_ct = sum(1 for q in pool
                   if prof[donor[q]][0] == prof[q][0] and prof[donor[q]][1] == prof[q][1])
    within = sum(1 for q in pool
                 if abs(prof[donor[q]][2] - prof[q][2]) <= max(1.0, LEN_TOL * prof[q][2]))
    emit(CONTROL, {q: joined[donor[q]] for q in pool},
         donors_exact_count_match=exact_ct, donors_within_length_tolerance=within,
         length_tolerance=LEN_TOL, unique_donors=len(set(donor.values())),
         donor_assignment="derangement (sorted-profile adjacent swap, same-image repaired)",
         note="donor graph substituted wholesale; matched on object count, relation count, then "
              "rendered token length. Never matched on the ANSWER.")
    dm = out / "wrong_image_donors.json"
    dm.write_text(json.dumps(donor, indent=1))
    manifest_arms[CONTROL]["donor_map"] = str(dm.relative_to(ROOT))
    manifest_arms[CONTROL]["donor_map_sha256"] = sha256_file(dm)
    print(f"  donors: exact count-match {exact_ct}/{len(pool)}, "
          f"within {int(LEN_TOL * 100)}% token length {within}/{len(pool)}")

    # ---- the checkpoint's own metadata, so the analysis compares against the artifact ----
    ck = ROOT / "outputs" / "checkpoints" / "corrected" / f"{args.checkpoint}.pt"
    expect_meta = None
    if ck.is_file():
        import torch
        c = torch.load(ck, map_location="cpu", weights_only=False)
        expect_meta = {"checkpoint_tag": c.get("tag"), "checkpoint_encoder": c.get("encoder"),
                       "checkpoint_epoch": c.get("epoch"),
                       "checkpoint_prompt_format": c.get("prompt_format"),
                       "checkpoint_supervise_eos": bool(c.get("supervise_eos", False)),
                       "llm_precision": c.get("llm_precision"), "seed": c.get("seed"),
                       "evidence_layer": "corrected_chatml_v1_eos"}
        print(f"\ncheckpoint metadata pinned from {ck.name}: {expect_meta}")
    else:
        print(f"\nWARNING {ck} absent; expected record metadata NOT pinned")

    if (PredQPlainAugment.input_mode != SceneGraphAugment.input_mode
            or PredQPlainAugment.reason_mode != SceneGraphAugment.reason_mode
            or PredQPlainAugment.graph_framing != SceneGraphAugment.graph_framing):
        raise SystemExit("FAIL the derived arms and the oracle no longer share prompt levers")
    expected_common_meta = {
        "artifact_version": ARTIFACT_VERSION, "metric_version": METRIC_VERSION,
        "split": Path(args.qids).stem,
        "input_mode": PredQPlainAugment.input_mode,
        "reason_mode": PredQPlainAugment.reason_mode,
        "graph_framing": PredQPlainAugment.graph_framing,
        "baseline": "none", "n_questions": len(qids),
    }
    expected_arm_meta = {
        name: {"augmentation": PredQPlainAugment.name, "oracle": False,
               "pred_cache": spec["path"], "n_context_injected": spec["n_context_available"]}
        for name, spec in manifest_arms.items()}

    from src.data.scene_graph import SceneGraphStore
    gt = SceneGraphStore(cfg["gqa"]["scene_graphs"])
    oracle_cov = sum(1 for q in qids if gt.describe(q2img[q]))
    expected_arm_meta[ORACLE_ARM] = {
        "augmentation": SceneGraphAugment.name, "oracle": True, "pred_cache": None,
        "n_context_injected": oracle_cov}
    print(f"oracle coverage measured from the ground-truth store: {oracle_cov}/{len(qids)}")

    manifest = {
        "task": "External learned SGG (RelTR / Open Images V6) graph-trust experiment",
        "status": "EXPLORATORY ONLY — tuning surface; no confirmatory claim",
        "development_surface": args.qids,
        "development_surface_sha256": sha256_file(ROOT / args.qids),
        "leakage": "residual 0 by construction: 6 Open-Images-VRD-train matches and 9 "
                   "unresolvable images excluded before any accuracy existed "
                   "(scripts/43, scripts/44)",
        "checkpoint_for_evaluation": args.checkpoint,
        "checkpoint_sha256": sha256_file(ck) if ck.is_file() else None,
        "expected_record_meta": expect_meta,
        "expected_common_meta": expected_common_meta,
        "expected_arm_meta": expected_arm_meta,
        "evaluator_calls": "3 paired calls, not 4: no_graph is the B side of every run",
        "seed": SEED, "tokenizer": tok_id,
        "primary_estimand": f"exact_full, paired: {TREATMENT} MINUS {CONTROL}",
        "secondary_estimand": f"exact_full, paired: {TREATMENT} MINUS no_graph",
        "bootstrap": {"resamples": 10000, "seed": 42, "shared_matrix": True},
        "oracle_policy": "descriptive headroom only — FORBIDDEN as a primary, promotion or "
                         "inferential secondary comparator; carries no interval",
        "sources": {"graphs": args.graphs, "graphs_sha256": sha256_file(ROOT / args.graphs)},
        "arms": manifest_arms,
        "not_built": {"no_graph": "baseline arm needs no cache",
                      "oracle": "built from the GQA ground-truth store at evaluation time"},
    }
    mp = out / "MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nwrote {len(manifest_arms)} condition caches + {mp.relative_to(ROOT)}")
    print("NO GPU work is performed or approved by this script.")


if __name__ == "__main__":
    main()
