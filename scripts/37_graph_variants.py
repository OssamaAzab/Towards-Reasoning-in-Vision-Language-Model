"""T-049 Rank 1: build the eight graph-variant caches for the trust diagnostic. CPU only.

The question this serves is whether any NON-ORACLE graph evidence causally helps, which can
only be answered against a corruption that is matched on everything except image identity.
So the primary contrast is arm 2 minus arm 6 — predicted graph versus a distribution-matched
wrong-image graph — and every other contrast is secondary.

DESIGN: this script writes derived cache files ONLY. Each arm is then evaluated by pointing the
existing, already-audited `PredictedGraphStore` at one of them, so all arms are rendered by the
same code path and differ solely in graph CONTENT. No module under src/ is modified, and no
arm gets a bespoke renderer that could introduce a formatting difference masquerading as an
effect.

The two source caches use DIFFERENT KEY SPACES and neither payload carries a cross-reference:
`tune500.json` is keyed by image id (481 entries), `tune500_q.json` by question id (500). A
direct key intersection returns zero and would silently compare nothing, so the join runs
through the GQA question file and is asserted non-empty with the expected cardinality.

NOT APPROVED and NOT built here: any confirmatory claim. Rank 1 is exploratory — the
confirmatory 3,000 and locked 2,000 are spent, and no fresh image-disjoint endpoint exists.
The oracle arm is a privileged CEILING: it may be differenced only in the descriptive
headroom diagnostic, and is forbidden as a primary, promotion or inferential secondary
comparator.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.artifact import ARTIFACT_VERSION                  # noqa: E402
from src.augment import PredQPlainAugment, SceneGraphAugment   # noqa: E402
from src.data.predicted_graph import PredictedGraphStore   # noqa: E402
from src.eval.metrics import METRIC_VERSION                # noqa: E402
from src.models.revisions import revision_for              # noqa: E402
from src.utils import load_config                          # noqa: E402

SEED = 42
LEN_TOL = 0.10          # donor rendered-graph token length must be within +/-10% of the true one

# arm -> (source cache, transform). "oracle" is built separately and is a privileged ceiling;
# the manifest records the policy so an analysis script can refuse to use it inferentially.
ARMS = ("no_graph", "pred", "pred_q", "objects_only", "objects_filtered",
        "wrong_image", "relations_shuffled", "oracle")
ORACLE_ARM = "oracle"


def sha256_file(p: Path) -> str:
    """Hash a file so every derived cache is pinned in the manifest."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def qid_to_image(cfg, qids):
    """Map question id -> image id from the GQA question file, asserting full coverage."""
    qfile = cfg["gqa"]["questions"]["val_balanced"]
    with open(qfile) as f:
        allq = json.load(f)
    missing = [q for q in qids if q not in allq]
    if missing:
        raise SystemExit(f"FAIL {len(missing)} tuning QIDs absent from {qfile}: {missing[:5]}")
    return {q: str(allq[q]["imageId"]) for q in qids}


def strip_relations(graph):
    """Objects and attributes retained; every relation removed."""
    g = copy.deepcopy(graph)
    for o in g.get("objects", {}).values():
        o["relations"] = []
    return g


def filter_objects(graph, keep_frac=0.5):
    """Drop the lowest-scoring half of objects, keeping relations among survivors.

    Object detector confidence is NOT calibrated confidence in the deterministic geometry
    relations, so object pruning and relation removal are separate arms and are never
    combined into one 'filtered graph' condition.
    """
    g = copy.deepcopy(graph)
    objs = g.get("objects", {})
    scored = [(oid, o.get("score", o.get("confidence"))) for oid, o in objs.items()]
    have_scores = [s for _, s in scored if s is not None]
    if len(have_scores) == len(scored) and scored:
        order = sorted(scored, key=lambda kv: kv[1], reverse=True)
    else:
        # No scores stored: fall back to cache order, which is detector order. Recorded in the
        # manifest so the arm is never described as confidence-filtered when it was not.
        order = scored
    keep = {oid for oid, _ in order[: max(1, int(len(order) * keep_frac))]}
    g["objects"] = {oid: o for oid, o in objs.items() if oid in keep}
    for o in g["objects"].values():
        o["relations"] = [r for r in o.get("relations", []) if str(r.get("object")) in keep]
    return g, (len(have_scores) == len(scored) and bool(scored))


def shuffle_relation_labels(graph, rng):
    """Permute relation NAMES within the graph, preserving structure and every object.

    Object identities, boxes and counts are untouched, so a difference against the true graph
    isolates relation CONTENT rather than graph presence, length or object vocabulary.
    """
    g = copy.deepcopy(graph)
    names = [r["name"] for o in g.get("objects", {}).values()
             for r in o.get("relations", []) if r.get("name")]
    if len(names) < 2:
        return g, False
    perm = list(names)
    for _ in range(16):                       # avoid the identity permutation where possible
        rng.shuffle(perm)
        if perm != names:
            break
    it = iter(perm)
    for o in g.get("objects", {}).values():
        for r in o.get("relations", []):
            if r.get("name"):
                r["name"] = next(it)
    return g, perm != names


def counts(graph):
    """(object count, relation count) — the quantities the donor match is made on."""
    objs = graph.get("objects", {})
    return len(objs), sum(len(o.get("relations", [])) for o in objs.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qids", default="data/gqa/tune_500_qids.json",
                    help="development surface; the locked and confirmatory slices are spent")
    ap.add_argument("--pred", default="outputs/pred_graphs/tune500.json")
    ap.add_argument("--pred-q", default="outputs/pred_graphs/tune500_q.json")
    ap.add_argument("--out", default="outputs/graph_variants")
    ap.add_argument("--tokenizer", default=None, help="HF id; default from config")
    ap.add_argument("--checkpoint", default="bridge_clip_w1_bf16_v1_ep5",
                    help="stem under outputs/checkpoints/corrected; its metadata is pinned "
                         "into the manifest so the analysis can verify record provenance")
    args = ap.parse_args()

    for bad in ("eval_2000", "locked", "confirm_3000"):
        if bad in args.qids:
            raise SystemExit(f"FAIL {args.qids!r} names a spent slice; Rank 1 runs on tuning-500")

    cfg = load_config()
    qids = [str(q) for q in json.load(open(ROOT / args.qids))]
    q2img = qid_to_image(cfg, qids)
    pred = json.load(open(ROOT / args.pred))
    pred_q = json.load(open(ROOT / args.pred_q))

    # The join that a naive key intersection would get wrong.
    joined = {q: pred[q2img[q]] for q in qids if q2img[q] in pred}
    if not joined:
        raise SystemExit("FAIL qid->image join produced ZERO graphs; the two caches use "
                         "different key spaces and a direct intersection compares nothing")
    print(f"join: {len(joined)}/{len(qids)} questions have an image-keyed predicted graph "
          f"({len(set(q2img.values()))} unique images; cache holds {len(pred)})")
    direct = set(qids) & set(pred)
    print(f"  sanity: direct qid-vs-image-cache intersection = {len(direct)} "
          f"(expected 0 — this is why the join is required)")

    q_direct = {q: pred_q[q] for q in qids if q in pred_q}
    print(f"question-conditioned cache covers {len(q_direct)}/{len(qids)} questions")

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- token lengths of the RENDERED graph text, per question, for every arm ----
    from transformers import AutoTokenizer
    tok_id = args.tokenizer or cfg["models"]["llm"]
    tok = AutoTokenizer.from_pretrained(tok_id, revision=revision_for(tok_id))

    def render_len(store, key):
        return len(tok(store.describe(key), add_special_tokens=False)["input_ids"])

    def eval_store(path):
        """The store the EVALUATION actually uses.

        `PredictedGraphStore` strips the colour attribute at load time; `SceneGraphStore` does
        not. Measuring or matching with the latter reports a graph nobody will ever be shown —
        an earlier version of this script did exactly that and reported 96.6/96.7 mean tokens
        where the evaluated text is 83.0/82.8, inflating the apparent length match.
        """
        return PredictedGraphStore(str(path))

    variants, manifest_arms = {}, {}

    def emit(name, graphs, **extra):
        """Write one variant cache keyed by QID and record its hash and token stats."""
        p = out / f"{name}.json"
        p.write_text(json.dumps(graphs))
        s = eval_store(p)
        lens = [render_len(s, q) for q in graphs]
        variants[name] = graphs
        manifest_arms[name] = {
            "path": str(p.relative_to(ROOT)), "sha256": sha256_file(p),
            "n_questions": len(graphs),
            # The evaluator counts a question as "context injected" when describe() is truthy
            # (scripts/09_augment_eval.py:164). Measuring the same predicate here turns the
            # analysis's coverage check into an equality against a derived number, so a
            # 499-of-500 run fails instead of being reported as near-complete.
            "n_context_available": sum(1 for q in graphs if s.describe(q)),
            "rendered_tokens": {"mean": float(np.mean(lens)) if lens else 0.0,
                                "median": float(np.median(lens)) if lens else 0.0,
                                "min": int(min(lens)) if lens else 0,
                                "max": int(max(lens)) if lens else 0},
            **extra}
        print(f"  {name:20s} n={len(graphs):4d}  tokens mean={np.mean(lens) if lens else 0:6.1f} "
              f"median={np.median(lens) if lens else 0:5.1f}")
        return lens

    print("\nbuilding variants (all keyed by QID so every arm shares one key space):")
    base_lens = emit("pred", dict(joined))
    emit("pred_q", dict(q_direct))
    emit("objects_only", {q: strip_relations(g) for q, g in joined.items()})

    filt, scored_ok = {}, None
    for q, g in joined.items():
        fg, ok = filter_objects(g)
        filt[q] = fg
        scored_ok = ok if scored_ok is None else (scored_ok and ok)
    emit("objects_filtered", filt, confidence_scores_available=bool(scored_ok),
         note=("objects ranked by stored detector score" if scored_ok else
               "NO per-object score in cache; ranking fell back to cache order — this arm is "
               "an object-count reduction, NOT a confidence filter, and must be labelled so"))

    shuffled, n_shuf = {}, 0
    for q, g in joined.items():
        sg, changed = shuffle_relation_labels(g, rng)
        shuffled[q] = sg
        n_shuf += changed
    emit("relations_shuffled", shuffled, n_actually_permuted=n_shuf,
         note="object identities, boxes and counts untouched; only relation NAMES permuted")

    # ---- arm 6: distribution-matched wrong-image donor, as a DERANGEMENT ----
    # Nearest-neighbour picking independently per question collapses onto popular donors: an
    # earlier version used 337 distinct donors for 500 questions, reusing one six times, which
    # makes the control's graph distribution unlike the treatment's. Instead sort by profile and
    # swap ADJACENT pairs. That is a perfect matching (every graph used exactly once, an
    # involution) whose partners are by construction the most similar available.
    lens_by_q = {q: base_lens[i] for i, q in enumerate(joined)}
    prof = {q: (*counts(g), lens_by_q[q]) for q, g in joined.items()}
    pool = sorted(joined, key=lambda q: (prof[q][0], prof[q][1], prof[q][2], q))
    donor_map = {}
    for i in range(0, len(pool) - 1, 2):
        a, b = pool[i], pool[i + 1]
        donor_map[a], donor_map[b] = b, a
    if len(pool) % 2:                      # odd tail: 3-cycle the last three
        a, b, c = pool[-3], pool[-2], pool[-1]
        donor_map[a], donor_map[b], donor_map[c] = b, c, a

    # Repair any pair that happens to share an image by swapping partners with another pair.
    def shares(q):
        return q2img[donor_map[q]] == q2img[q]
    for q in pool:
        if not shares(q):
            continue
        for r in pool:
            if r == q or shares(r):
                continue
            dq, dr = donor_map[q], donor_map[r]
            if q2img[dr] != q2img[q] and q2img[dq] != q2img[r] and dr != q and dq != r:
                # Cross the two pairs and restore the involution: q<->dr and r<->dq.
                donor_map[q], donor_map[dr] = dr, q
                donor_map[r], donor_map[dq] = dq, r
                break
    bad = [q for q in pool if q2img[donor_map[q]] == q2img[q]]
    if bad:
        raise SystemExit(f"FAIL the control is not a control: {len(bad)} donors share their "
                         f"own image. First: {bad[:3]}")
    used = set(donor_map.values())
    if len(used) != len(pool):
        raise SystemExit(f"FAIL donor set is not one-to-one: {len(used)} distinct donors for "
                         f"{len(pool)} questions; the control's distribution would differ")

    exact_ct = sum(1 for q in pool
                   if prof[donor_map[q]][0] == prof[q][0] and prof[donor_map[q]][1] == prof[q][1])
    within_tol = sum(1 for q in pool
                     if abs(prof[donor_map[q]][2] - prof[q][2]) <= max(1.0, LEN_TOL * prof[q][2]))
    emit("wrong_image", {q: joined[donor_map[q]] for q in pool},
         donors_exact_count_match=exact_ct, donors_within_length_tolerance=within_tol,
         length_tolerance=LEN_TOL, unique_donors=len(used), donor_assignment="derangement",
         renderer="PredictedGraphStore (attributes stripped), i.e. the evaluated text",
         note="donor graph substituted wholesale; matched on object count, relation count, "
              "then rendered token length. Never matched on the ANSWER.")
    dm = out / "wrong_image_donors.json"
    dm.write_text(json.dumps(donor_map, indent=1))
    manifest_arms["wrong_image"]["donor_map"] = str(dm.relative_to(ROOT))
    manifest_arms["wrong_image"]["donor_map_sha256"] = sha256_file(dm)
    print(f"  donors: exact count-match {exact_ct}/{len(pool)}, "
          f"within {int(LEN_TOL*100)}% token length {within_tol}/{len(pool)}")

    # Record the checkpoint's OWN metadata, so the analysis compares record provenance against
    # values derived from the artifact rather than parsed out of a filename. `checkpoint_tag`
    # is "w1_bf16_v1", not the stem "bridge_clip_w1_bf16_v1_ep5" — comparing against the stem
    # rejects genuine production records, which is how an earlier guard failed in the direction
    # nobody tests for.
    ck_path = ROOT / "outputs" / "checkpoints" / "corrected" / f"{args.checkpoint}.pt"
    expect_meta = None
    if ck_path.is_file():
        import torch
        c = torch.load(ck_path, map_location="cpu", weights_only=False)
        expect_meta = {"checkpoint_tag": c.get("tag"), "checkpoint_encoder": c.get("encoder"),
                       "checkpoint_epoch": c.get("epoch"),
                       "checkpoint_prompt_format": c.get("prompt_format"),
                       "checkpoint_supervise_eos": bool(c.get("supervise_eos", False)),
                       "llm_precision": c.get("llm_precision"), "seed": c.get("seed"),
                       "evidence_layer": "corrected_chatml_v1_eos"}
        print(f"\ncheckpoint metadata pinned from {ck_path.name}: {expect_meta}")
    else:
        print(f"\nWARNING {ck_path} absent; expected record metadata NOT pinned")

    # ---- what each arm's records must SAY they are ----
    # Hashing the local cache files proves the caches did not change; it cannot prove which
    # cache the evaluator consumed. Without the block below, six records renamed into six
    # arm-shaped filenames pass every check even if one graph fed all of them.
    #
    # Every value is derived from production code or from a measurement, never from a literal
    # typed here: the augmenter classes carry the names and prompt levers the evaluator will
    # write, and the coverage numbers were counted from the caches just written.
    if (PredQPlainAugment.input_mode != SceneGraphAugment.input_mode
            or PredQPlainAugment.reason_mode != SceneGraphAugment.reason_mode
            or PredQPlainAugment.graph_framing != SceneGraphAugment.graph_framing):
        raise SystemExit("FAIL the derived arms and the oracle no longer share prompt levers; "
                         "a single common expectation would be wrong for one of them")
    expected_common_meta = {
        "artifact_version": ARTIFACT_VERSION,
        "metric_version": METRIC_VERSION,
        "split": Path(args.qids).stem,
        "input_mode": PredQPlainAugment.input_mode,
        "reason_mode": PredQPlainAugment.reason_mode,
        "graph_framing": PredQPlainAugment.graph_framing,
        "baseline": "none",
        "n_questions": len(qids),
    }
    expected_arm_meta = {
        name: {"augmentation": PredQPlainAugment.name, "oracle": False,
               "pred_cache": spec["path"], "n_context_injected": spec["n_context_available"]}
        for name, spec in manifest_arms.items()}

    # The oracle arm has no derived cache, so its coverage is measured from the ground-truth
    # store the evaluator will actually query.
    from src.data.scene_graph import SceneGraphStore
    gt = SceneGraphStore(cfg["gqa"]["scene_graphs"])
    oracle_cov = sum(1 for q in qids if gt.describe(q2img[q]))
    expected_arm_meta[ORACLE_ARM] = {
        "augmentation": SceneGraphAugment.name, "oracle": True, "pred_cache": None,
        "n_context_injected": oracle_cov}
    print(f"\noracle coverage measured from the ground-truth store: {oracle_cov}/{len(qids)}")

    manifest = {
        "task": "T-049 Rank 1 graph-quality/trust diagnostic",
        "status": "EXPLORATORY ONLY — no confirmatory claim; no fresh endpoint exists",
        "development_surface": args.qids,
        "development_surface_sha256": sha256_file(ROOT / args.qids),
        "checkpoint_for_evaluation": args.checkpoint,
        "checkpoint_sha256": sha256_file(ck_path) if ck_path.is_file() else None,
        "expected_record_meta": expect_meta,
        "expected_common_meta": expected_common_meta,
        "expected_arm_meta": expected_arm_meta,
        "evaluator_calls": ("7 paired calls, not 8: no_graph is the B side of every run, not an "
                            "independent graph arm"),
        "seed": SEED, "tokenizer": tok_id,
        "primary_estimand": "exact_full, paired: pred MINUS wrong_image",
        "oracle_policy": "descriptive headroom only — the oracle is FORBIDDEN as a primary, promotion or inferential secondary comparator, and appears solely in the privileged headroom diagnostic, which is descriptive and carries no interval",
        "secondary_note": "all other contrasts are secondary; multiplicity must be stated",
        "sources": {"pred": args.pred, "pred_q": args.pred_q,
                    "pred_sha256": sha256_file(ROOT / args.pred),
                    "pred_q_sha256": sha256_file(ROOT / args.pred_q)},
        "arms": manifest_arms,
        "not_built": {"no_graph": "baseline arm needs no cache",
                      "oracle": "built from the GQA ground-truth store at evaluation time"},
        "leakage": ("no evaluation program, gold answer or oracle graph enters arms 1-7; the "
                    "VG/GQA exclusion (Section 9.4) is inherited unchanged"),
    }
    mp = out / "MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=2))
    print(f"\nwrote {len(manifest_arms)} variant caches + {mp.relative_to(ROOT)}")
    print("NO GPU work is performed or approved by this script.")


if __name__ == "__main__":
    main()
