"""T-049 Rank 1 analysis: paired contrasts over the graph-variant arms. CPU only.

Primary estimand, fixed before any record exists:

    exact_full, paired per question:   pred  MINUS  wrong_image

That is the only contrast this design can use to support image-specific dependence. Every other
contrast is SECONDARY and is printed with its multiplicity stated.

The oracle arm is a privileged ceiling. It IS differenced once — `oracle_minus_no_graph` in the
descriptive headroom diagnostic, reported without an interval — and nowhere else. Saying it is
'never a comparator' was literally false of this script's own output (review record); the enforced
rule is that it may not be a primary, promotion or inferential secondary comparator.

One shared bootstrap resample matrix per metric, canonical sorted-QID order, seed 42, and the
matrix sha256 printed so a third party can confirm the same matrix was used throughout.

`--self-test` runs the whole path on synthetic records with a planted effect and no model, which
is what makes the statistics CPU-verifiable before any GPU time is requested.
"""
from pathlib import Path
import argparse
import glob
import hashlib
import json
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEED, BOOT = 42, 10000
PRIMARY = ("pred", "wrong_image")
ORACLE = "oracle"


N_EXPECT = 500

# The whole no-graph run, not just its score. `_baseline` used to hold `qid -> bool(b_exact_full)`
# while main printed "byte-identical": rewriting 50 baseline answers and raw generations without
# changing whether they were correct passed. Comparing these eight fields makes the phrase true.
BASELINE_FIELDS = ("b_ans", "b_raw", "b_exact_full", "b_exact", "b_vqa",
                   "b_stop_reason", "b_cap_hit", "b_n_generated")


def arm_pattern(arm):
    """The anchored filename rule for one arm, as a compiled regex.

    The evaluator names an artifact `<stem>__<suffix>_records.json`, so an arm token can be
    followed either by another `__` separator or directly by `_records.json`. Requiring `__` on
    both sides — which an earlier version did — matches nothing the launcher actually writes,
    and would have failed only after all seven GPU calls had been spent. Requiring nothing on
    the right would make "pred" select "pred_q" and silently analyse the wrong arm.
    """
    return re.compile(rf"__{re.escape(arm)}(?:__|_records\.json$)")


def find_arm_file(records_dir, arm):
    """The single records file belonging to one arm, or a hard failure naming the candidates."""
    pat = arm_pattern(arm)
    hits = sorted(p for p in glob.glob(str(Path(records_dir) / "*records.json"))
                  if pat.search(Path(p).name))
    if len(hits) != 1:
        loose = sorted(Path(p).name for p in
                       glob.glob(str(Path(records_dir) / f"*{arm}*records.json")))
        raise SystemExit(f"FAIL expected exactly one records file for arm {arm!r} in "
                         f"{records_dir}, got {len(hits)}. Files mentioning the arm at all: "
                         f"{loose}")
    return hits[0]


def load_arm(records_dir, arm, metric, side="a"):
    """QID -> bool for one arm.

    `scripts/09_augment_eval.py` writes BOTH conditions into one records file: `a_*` is the
    augmented arm and `b_*` the no-graph baseline it was run against. There is no `bridge_*`
    field — an earlier version of this script looked for one and would have failed only after
    the GPU had been spent.
    """
    hits = [find_arm_file(records_dir, arm)]
    recs = json.load(open(hits[0]))["records"]
    field = f"{side}_{metric}"
    if field not in recs[0]:
        raise SystemExit(f"FAIL records lack {field!r} (have: {sorted(recs[0])[:10]}); "
                         f"a comparison would be vacuous")
    qids = [r["qid"] for r in recs]
    if len(set(qids)) != len(qids):
        raise SystemExit(f"FAIL arm {arm!r} has duplicate QIDs")
    if len(qids) != N_EXPECT:
        raise SystemExit(f"FAIL arm {arm!r} has {len(qids)} questions, expected {N_EXPECT}")
    return {r["qid"]: bool(r[field]) for r in recs}, json.load(open(hits[0])), hits[0]


def validate(arm, blob, path, expect_meta, expect_common, expect_arm):
    """Refuse to analyse records whose provenance or output validity is unverified.

    None of this was checked before: a run against the wrong checkpoint, a different slice, or
    with a collapsed context-injection rate would have produced a full table of numbers.

    `expect_arm` binds the file to the arm its NAME claims. Hashing the local cache files proves
    only that the caches did not change; it cannot prove which cache the evaluator read. Without
    the per-arm block, six records renamed into six arm-shaped filenames pass every check even
    when a single graph fed all of them.
    """
    meta, recs = blob.get("_meta", {}), blob["records"]
    problems = []

    # ABSENCE IS FAILURE, for all three expectation blocks. An earlier version wrote
    # `if expect and got and got != expect`, so a record file with no provenance at all passed
    # every check — the "a check that cannot fail is not a check" defect, committed inside the
    # guard meant to prevent it.
    if expect_meta is None:
        problems.append("the build manifest pinned no expected record metadata; provenance "
                        "cannot be verified and the analysis will not proceed")
    if not expect_common:
        problems.append("the build manifest pinned no expected common protocol metadata; the "
                        "records cannot be shown to come from the intended evaluation protocol")
    if not expect_arm:
        problems.append(f"the build manifest pinned no expectation for arm {arm!r}; the records "
                        f"cannot be bound to the arm their filename claims")

    for key, want in list((expect_meta or {}).items()) + list((expect_common or {}).items()):
        if key not in meta:
            problems.append(f"record metadata lacks {key!r} (present: {sorted(meta)[:8]})")
        elif meta[key] != want:
            problems.append(f"{key}={meta[key]!r} != expected {want!r}")

    for key, want in (expect_arm or {}).items():
        if key == "n_context_injected":
            continue                      # checked below against the records as well
        if key not in meta:
            problems.append(f"record metadata lacks {key!r}; arm identity is unverifiable")
        elif key == "pred_cache":
            got = meta[key]
            same = ((got is None and want is None) or
                    (got is not None and want is not None
                     and Path(got).expanduser().resolve() == (ROOT / want).resolve()))
            if not same:
                problems.append(f"pred_cache={got!r} is not arm {arm!r}'s cache {want!r}; these "
                                f"records do not establish which graph the evaluator read")
        elif meta[key] != want:
            problems.append(f"{key}={meta[key]!r} != expected {want!r} for arm {arm!r}")

    # Field names are the evaluator's, verified against scripts/09_augment_eval.py: it writes
    # `a_context` per record and `n_context_injected` in _meta. An earlier version checked a
    # `context` field that production never writes, so coverage was silently unmeasured.
    for field in (("a_exact_full", "a_exact", "a_vqa", "a_context",
                   "a_stop_reason", "a_cap_hit", "a_n_generated") + BASELINE_FIELDS):
        if field not in recs[0]:
            problems.append(f"records lack {field!r}; validity cannot be established")
    if "n_context_injected" not in meta:
        problems.append("_meta lacks 'n_context_injected'; injection coverage is unverifiable")

    # COMPLETE coverage, not merely self-consistent coverage. If _meta said 499 and exactly one
    # record carried an empty context, both figures read 99.8% and the run passed. The expected
    # count is measured by the builder with the evaluator's own predicate (truthy describe()),
    # so this is an equality against a derived number rather than a tolerance.
    want_ctx = (expect_arm or {}).get("n_context_injected")
    got_meta = meta.get("n_context_injected")
    got_recs = sum(1 for r in recs if r.get("a_context"))
    if want_ctx is None:
        problems.append(f"no expected injection coverage is pinned for arm {arm!r}; incomplete "
                        f"injection would be indistinguishable from a null effect")
    else:
        if got_meta != want_ctx:
            problems.append(f"_meta reports {got_meta} questions injected, the build measured "
                            f"{want_ctx} available; a silent plain-prompt fallback is a null "
                            f"effect wearing the arm's name")
        if got_recs != want_ctx:
            problems.append(f"{got_recs} records carry a non-empty a_context, expected "
                            f"{want_ctx}")
    if problems:
        raise SystemExit(f"FAIL arm {arm!r} ({Path(path).name}):\n  - " + "\n  - ".join(problems))

    n = len(recs)
    pct = lambda f: 100 * sum(bool(f(r)) for r in recs) / n
    stats = {
        "n": n,
        "n_context_injected": meta["n_context_injected"],
        "context_coverage_pct": 100 * meta["n_context_injected"] / n,
        "a_eos_pct": pct(lambda r: str(r["a_stop_reason"]).startswith("eos")),
        "a_cap_hit_pct": pct(lambda r: r["a_cap_hit"]),
        "a_mean_generated": sum(r["a_n_generated"] for r in recs) / n,
        "b_eos_pct": pct(lambda r: str(r["b_stop_reason"]).startswith("eos")),
        "b_cap_hit_pct": pct(lambda r: r["b_cap_hit"]),
        "b_mean_generated": sum(r["b_n_generated"] for r in recs) / n,
        "a_exact_full_pct": pct(lambda r: r["a_exact_full"]),
        "a_exact_pct": pct(lambda r: r["a_exact"]),
        "a_vqa_pct": pct(lambda r: r["a_vqa"]),
        "b_exact_full_pct": pct(lambda r: r["b_exact_full"]),
    }
    # Per category, BOTH sides and all three metrics. The augmented exact_full percentage alone
    # cannot say whether a category moved, because it has nothing to be paired against.
    stats["by_category"] = {}
    for c in sorted({r.get("category") for r in recs if r.get("category")}):
        sel = [r for r in recs if r.get("category") == c]
        m = len(sel)
        stats["by_category"][c] = {"n": m, **{
            f"{k}_pct": 100 * sum(bool(r[k]) for r in sel) / m
            for k in ("a_exact_full", "b_exact_full", "a_exact", "b_exact", "a_vqa", "b_vqa")}}
    stats["_baseline"] = {r["qid"]: tuple(r[f] for f in BASELINE_FIELDS) for r in recs}
    stats["_category"] = {r["qid"]: r.get("category") for r in recs}
    return stats


def matrix(n, seed=SEED):
    """One resample matrix, and its hash, so every contrast provably shares it."""
    idx = np.random.default_rng(seed).integers(0, n, size=(BOOT, n))
    return idx, hashlib.sha256(idx.tobytes()).hexdigest()


def contrast(a, b, qids, idx):
    """Paired point estimate and 95% interval for a-b in points, on the shared matrix."""
    v = np.array([a[q] for q in qids], float) - np.array([b[q] for q in qids], float)
    d = v[idx].mean(1) * 100
    return v.mean() * 100, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def analyse(arms, metric="exact_full", restrict=None, label=""):
    """Print the primary contrast, then secondaries with multiplicity stated."""
    if ORACLE in (PRIMARY[0], PRIMARY[1]):
        raise SystemExit("FAIL the oracle is a privileged ceiling and may not be the primary comparator")
    # Require identical QID sets. A silent intersection would quietly analyse whichever
    # questions happened to be common and still print a full table.
    keysets = {n: frozenset(a) for n, a in arms.items()}
    if len(set(keysets.values())) != 1:
        sizes = {n: len(k) for n, k in keysets.items()}
        raise SystemExit(f"FAIL arms do not cover identical QID sets: {sizes}")
    qids = sorted(next(iter(keysets.values())))
    if restrict is not None:
        qids = [q for q in qids if q in restrict]
    if not qids:
        raise SystemExit("FAIL no QIDs to analyse")
    idx, sha = matrix(len(qids))
    print(f"\n=== {metric}{' — ' + label if label else ''} | n={len(qids)} | "
          f"matrix sha256={sha[:16]} ===")

    p, lo, hi = contrast(arms[PRIMARY[0]], arms[PRIMARY[1]], qids, idx)
    star = "*" if (lo > 0 or hi < 0) else " "
    print(f"PRIMARY   {PRIMARY[0]} - {PRIMARY[1]:<18} {p:+7.2f} [{lo:+.2f}, {hi:+.2f}] {star}")
    print(f"          half-width = {(hi - lo) / 2:.2f} points  <- what this slice can resolve")
    if lo <= 0 <= hi:
        print("          INTERVAL INCLUDES ZERO: this is a null. It does not license Ranks 2-6.")

    out = {"n": len(qids), "matrix_sha256": sha, "secondary": {},
           "primary": {"contrast": f"{PRIMARY[0]}-{PRIMARY[1]}", "point": p, "lo": lo, "hi": hi}}

    # The primary establishes image-specific DEPENDENCE. It cannot establish IMPROVEMENT —
    # a graph could beat a wrong graph while both are worse than no graph at all. That
    # requires pred - no_graph, which is reported as a co-equal secondary, not omitted.
    if "no_graph" in arms:
        pi, li, hi_ = contrast(arms[PRIMARY[0]], arms["no_graph"], qids, idx)
        star = "*" if (li > 0 or hi_ < 0) else " "
        print(f"IMPROVEMENT  {PRIMARY[0]} - no_graph{'':<11} {pi:+7.2f} [{li:+.2f}, {hi_:+.2f}] {star}")
        print("          dependence without improvement is possible; both are needed.")
        out["improvement"] = {"contrast": f"{PRIMARY[0]}-no_graph",
                              "point": pi, "lo": li, "hi": hi_}

    # Every graph variant is contrasted against pred, i.e. what removing or corrupting that
    # channel COSTS relative to the full predicted graph.
    # PRIMARY[1] is excluded as well: contrasting it against pred just restates the primary
    # with the sign flipped, and would inflate the multiplicity count with a duplicate.
    variants = [a for a in arms if a not in (PRIMARY[0], PRIMARY[1], "no_graph", ORACLE)]
    if variants:
        print(f"\nSECONDARY vs {PRIMARY[0]} (exploratory; {len(variants)} contrasts, "
              f"multiplicity NOT corrected):")
        for a in sorted(variants):
            p2, l2, h2 = contrast(arms[a], arms[PRIMARY[0]], qids, idx)
            print(f"          {a} - {PRIMARY[0]:<20} {p2:+7.2f} [{l2:+.2f}, {h2:+.2f}]")
            out["secondary"][f"{a}-{PRIMARY[0]}"] = {"point": p2, "lo": l2, "hi": h2}

    if ORACLE in arms and "no_graph" in arms:
        og = 100 * np.mean([arms[ORACLE][q] for q in qids])
        ng = 100 * np.mean([arms["no_graph"][q] for q in qids])
        pr = 100 * np.mean([arms[PRIMARY[0]][q] for q in qids])
        head = og - ng
        print(f"\nHEADROOM DIAGNOSTIC — DESCRIPTIVE. The oracle is differenced here and NOWHERE\n          else: it is forbidden as a primary, promotion or inferential secondary\n          comparator. No interval is computed for it and none may be inferred.")
        print(f"          no_graph {ng:6.2f}%   pred {pr:6.2f}%   oracle {og:6.2f}%")
        # The "captures X% of it" ratio is only meaningful when the ceiling is ABOVE the
        # baseline. With head < 0 — the oracle graph hurting, which this design permits — the
        # two negatives cancel and a predicted arm that also hurt prints a confident positive
        # "captured 50% of the headroom". That is a fabricated number, not a small one.
        captured = None
        if head > 0:
            captured = 100 * (pr - ng) / head
            print(f"          oracle headroom over no_graph = {head:+.2f} points; predicted "
                  f"captures {captured:.1f}% of it")
        else:
            print(f"          oracle headroom over no_graph = {head:+.2f} points: the privileged "
                  f"ceiling is NOT above the no-graph baseline, so there is no headroom to "
                  f"capture and no fraction is reported. Read this as evidence about the "
                  f"injection itself, not about the predicted graph.")
        # No numeric threshold is applied. An earlier version stopped the ladder below five
        # points, which was a decision rule invented after the plan was approved. Headroom is
        # reported so the human can judge it against the pre-registered stop rule.
        print("          Reported descriptively. Interpreting a null requires this number: a "
              "small ceiling makes a null informative about the bridge, not about graphs.")
        out["headroom"] = {"no_graph": ng, "pred": pr, "oracle": og,
                           "oracle_minus_no_graph": head,
                           "predicted_captured_pct": captured}
    return out


def category_effects(arms, qids, categories, contrasts):
    """Paired effects within each reasoning category. POST-HOC, exploratory, uncorrected.

    A category subset has a different n, so it cannot use the whole-slice resample matrix; each
    subset gets its own seeded matrix and prints its own hash. These are NOT the pre-registered
    estimand and no category result licenses a claim on its own.
    """
    cats = sorted({categories.get(q) for q in qids if categories.get(q)})
    if not cats:
        print("\nPER-CATEGORY: records carry no category field; no breakdown is possible")
        return {}
    print("\nPER-CATEGORY PAIRED EFFECTS — POST-HOC and exploratory, multiplicity NOT corrected;")
    print("  each category uses its own seeded matrix because n differs from the full slice.")
    print(f"{'category':<10}{'n':>5}" + "".join(f"{lab:>28}" for lab, _, _ in contrasts))
    out = {}
    for c in cats:
        sub = [q for q in qids if categories.get(q) == c]
        idx, sha = matrix(len(sub))
        row, cells = {"n": len(sub), "matrix_sha256": sha}, []
        for lab, x, y in contrasts:
            p, lo, hi = contrast(arms[x], arms[y], sub, idx)
            row[lab] = {"contrast": f"{x}-{y}", "point": p, "lo": lo, "hi": hi}
            cells.append(f"{p:+6.2f} [{lo:+.2f},{hi:+.2f}]")
        print(f"{c:<10}{len(sub):>5}" + "".join(f"{x:>28}" for x in cells))
        out[c] = row
    return out


def self_test():
    """Synthetic records with a planted effect; proves the path without a model."""
    rng = np.random.default_rng(0)
    q = [f"q{i:04d}" for i in range(500)]
    base = rng.random(500) < 0.45
    # Plant exactly +6.00 points: flip 30 questions the baseline gets WRONG. Flipping the
    # first 30 positions regardless would only move the ones already wrong, which is how an
    # earlier version of this fixture planted 19 and "recovered" 3.80 — the self-test caught
    # its own fixture, which is the point of having one.
    wrong_idx = [i for i, v in enumerate(base) if not v][:30]
    assert len(wrong_idx) == 30
    flip = set(wrong_idx)
    # wrong_image must NOT equal no_graph. When every control arm carries the same numbers, the
    # primary and the improvement contrast recover the same value and a swap between them — or
    # an arm-lookup that silently reads the wrong dict — passes unnoticed. Degrading 10
    # questions that the baseline answers correctly separates them: the primary becomes +8.00
    # (30 gained plus 10 the control lost) and the improvement stays +6.00.
    degrade = {i for i, v in enumerate(base) if v}
    degrade = set(sorted(degrade)[:10])
    assert len(degrade) == 10 and not (degrade & flip)
    arms = {
        "wrong_image": {k: bool(v and i not in degrade)
                        for i, (k, v) in enumerate(zip(q, base))},
        "no_graph": {k: bool(v) for k, v in zip(q, base)},
        "pred": {k: bool(v or i in flip) for i, (k, v) in enumerate(zip(q, base))},
        "objects_only": {k: bool(v) for k, v in zip(q, base)},
        "oracle": {k: True for k in q},
    }
    # Categories are planted too, so the subsetting path is verified rather than assumed: every
    # flipped question plus 70 unflipped ones is 'relate', giving that category exactly +30.00
    # and every other category exactly 0.00. A subset that selected the wrong questions could
    # not produce those two numbers.
    others = [i for i in range(len(q)) if i not in flip]
    relate = flip | set(others[:70])
    cats = {k: ("relate" if i in relate else ("exist" if i % 2 else "compare"))
            for i, k in enumerate(q)}

    print("SELF-TEST: planted +8.00 pred-vs-wrong_image and +6.00 pred-vs-no_graph, which are "
          "DIFFERENT so a swap between the two contrasts cannot pass")
    out = analyse(arms, label="synthetic")
    got = out["primary"]["point"]
    assert abs(got - 8.0) < 1e-9, f"planted +8.00 on the primary but recovered {got}"
    assert abs(out["improvement"]["point"] - 6.0) < 1e-9, \
        f"planted +6.00 on the improvement contrast but recovered {out['improvement']['point']}"
    assert "oracle" not in out["primary"]["contrast"]
    assert "oracle" not in " ".join(out["secondary"]), "oracle leaked into a contrast"
    assert out["headroom"]["oracle"] == 100.0

    # The interval is checked as an interval, not just for sign. lo > 0 alone is satisfied by a
    # bootstrap that resampled the wrong axis, returned the wrong percentiles, or produced a
    # degenerate width — all of which would still be positive here.
    for name, blk, want in (("primary", out["primary"], 8.0),
                            ("improvement", out["improvement"], 6.0)):
        lo, hi = blk["lo"], blk["hi"]
        assert lo < want < hi, f"{name} point {want} lies outside its own interval [{lo}, {hi}]"
        assert lo > 0, f"{name}: a planted positive effect should clear zero at n=500"
        assert 0.5 < hi - lo < 10.0, \
            f"{name} interval width {hi - lo:.3f} is degenerate or implausible at n=500"

    # The same contrast on the same matrix must be reproducible to the bit.
    again = analyse(arms, label="synthetic-repeat")
    assert again["primary"] == out["primary"], "the seeded bootstrap is not reproducible"
    assert again["matrix_sha256"] == out["matrix_sha256"], "the resample matrix is not stable"

    ce = category_effects(arms, sorted(q), cats,
                          [("improvement pred-no_graph", "pred", "no_graph")])
    assert sum(v["n"] for v in ce.values()) == len(q), "categories must partition the slice"
    assert ce["relate"]["n"] == 100, f"planted 100 relate questions, got {ce['relate']['n']}"
    rel = ce["relate"]["improvement pred-no_graph"]["point"]
    assert abs(rel - 30.0) < 1e-9, f"planted +30.00 within 'relate' but recovered {rel}"
    for c in ("exist", "compare"):
        z = ce[c]["improvement pred-no_graph"]["point"]
        assert z == 0.0, f"category {c} holds no flipped question but recovered {z}"

    print(f"\nrecovered {got:+.2f} on the primary and "
          f"{out['improvement']['point']:+.2f} on the improvement contrast — distinct, so the "
          f"two are separately constrained; both points lie inside their own intervals; the "
          f"seeded matrix reproduces exactly; 'relate' recovered {rel:+.2f} with every other "
          f"category at 0.00; oracle absent from the primary, improvement and secondary "
          f"contrasts (it appears only in the descriptive headroom line). SELF-TEST PASSED.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--records-dir", default="outputs/eval/graph_trust")
    # The manifest is a launch parameter, not a fixed location. The launcher freezes one
    # verified snapshot into the job directory and passes the SAME path to preflight, arm
    # extraction and this script; a hard-coded path here would let the evaluator and the
    # analysis read two different protocols if the mutable file changed between them.
    ap.add_argument("--manifest", default="outputs/graph_variants/MANIFEST.json")
    ap.add_argument("--metric", default="exact_full")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.self_test:
        return self_test()


    mpath = Path(args.manifest)
    if not mpath.is_absolute():
        mpath = ROOT / mpath
    if not mpath.is_file():
        raise SystemExit(f"FAIL manifest {args.manifest} is absent; the arms cannot be validated")
    man = json.loads(mpath.read_text())
    print(f"manifest: {mpath} sha256={hashlib.sha256(mpath.read_bytes()).hexdigest()[:16]}")
    expect_meta = man.get("expected_record_meta")
    expect_common = man.get("expected_common_meta")
    expect_arms = man.get("expected_arm_meta") or {}

    # The development surface, by content rather than by filename. The manifest stored this hash
    # and nothing ever checked it, so a shared set of 500 QIDs passed even when one of them was
    # not a tuning question at all.
    surface = ROOT / man["development_surface"]
    if not surface.is_file():
        raise SystemExit(f"FAIL development surface {man['development_surface']} is missing; the "
                         f"analysed population cannot be established")
    got = hashlib.sha256(surface.read_bytes()).hexdigest()
    if got != man["development_surface_sha256"]:
        raise SystemExit(f"FAIL {man['development_surface']} changed since the build "
                         f"({got[:12]} != {man['development_surface_sha256'][:12]}); the arms "
                         f"were built against a different question set")
    expected_qids = frozenset(str(q) for q in json.loads(surface.read_text()))
    if len(expected_qids) != N_EXPECT:
        raise SystemExit(f"FAIL the development surface holds {len(expected_qids)} questions, "
                         f"expected {N_EXPECT}")
    print(f"development surface: {man['development_surface']} sha256={got[:16]} "
          f"({len(expected_qids)} questions, verified by content)")

    # Same for the checkpoint. The launcher separately proves the cluster copy matches; this
    # proves the artifact the manifest describes is the one still on disk here.
    ck = ROOT / "outputs" / "checkpoints" / "corrected" / f"{man['checkpoint_for_evaluation']}.pt"
    if man.get("checkpoint_sha256") is None:
        raise SystemExit("FAIL the manifest pinned no checkpoint hash; record provenance rests "
                         "on metadata alone")
    if not ck.is_file():
        raise SystemExit(f"FAIL checkpoint {ck.name} is missing; its hash cannot be verified")
    ck_sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    if ck_sha != man["checkpoint_sha256"]:
        raise SystemExit(f"FAIL checkpoint {ck.name} changed since the build "
                         f"({ck_sha[:12]} != {man['checkpoint_sha256'][:12]})")
    print(f"checkpoint: {ck.name} sha256={ck_sha[:16]} matches the manifest")

    names = [a for a in man["arms"]] + [ORACLE]
    arms, validity = {}, {}
    for a in names:
        arms[a], blob, path = load_arm(args.records_dir, a, args.metric)
        validity[a] = validate(a, blob, path, expect_meta, expect_common, expect_arms.get(a))
        if frozenset(arms[a]) != expected_qids:
            extra = sorted(set(arms[a]) - expected_qids)[:5]
            missing = sorted(expected_qids - set(arms[a]))[:5]
            raise SystemExit(f"FAIL arm {a!r} was not evaluated on the pinned development "
                             f"surface: {len(set(arms[a]) - expected_qids)} foreign QIDs "
                             f"(e.g. {extra}), {len(expected_qids - set(arms[a]))} missing "
                             f"(e.g. {missing})")
    # `no_graph` is not a separate run: the evaluator scores the augmented arm (a_*) against
    # its own no-graph baseline (b_*) in one pass, so the baseline comes from the pred file's
    # b side. Taking it from a separate run would compare across two decoding passes.
    arms["no_graph"], _, _ = load_arm(args.records_dir, PRIMARY[0], args.metric, side="b")

    # The B side is the same no-graph run in every arm's file. If those disagree, the arms were
    # not scored against a common baseline and no contrast between them is interpretable. All
    # eight recorded baseline fields are compared: an earlier version kept only
    # `bool(b_exact_full)` and still called the result byte-identical, so rewritten answers and
    # raw generations passed as long as they stayed equally correct.
    baselines = {a: s.pop("_baseline") for a, s in validity.items()}
    categories = validity[PRIMARY[0]].pop("_category")
    for s in validity.values():
        s.pop("_category", None)
    ref_arm, ref = next(iter(baselines.items()))
    for a, b in baselines.items():
        if b == ref:
            continue
        bad_q = [q for q in ref if ref[q] != b.get(q)]
        fields = sorted({BASELINE_FIELDS[i] for q in bad_q[:200]
                         for i in range(len(BASELINE_FIELDS))
                         if q in b and ref[q][i] != b[q][i]})
        raise SystemExit(f"FAIL the no-graph baseline in arm {a!r} differs from arm {ref_arm!r} "
                         f"on {len(bad_q)} of {len(ref)} questions (fields: {fields}); the arms "
                         f"were not scored against a common control and no contrast between "
                         f"them is interpretable")
    print(f"baseline identity: all {len(BASELINE_FIELDS)} recorded B-side fields "
          f"({', '.join(BASELINE_FIELDS)}) are identical across all {len(baselines)} arms")

    # The caches the run consumed must be the ones this manifest describes.
    for a, spec in man["arms"].items():
        p = ROOT / spec["path"]
        if not p.is_file():
            raise SystemExit(f"FAIL variant cache {spec['path']} is missing; cache identity "
                             f"cannot be established")
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != spec["sha256"]:
            raise SystemExit(f"FAIL variant cache {a} changed since the manifest was written "
                             f"({got[:12]} != {spec['sha256'][:12]}); the arms were built from "
                             f"different graphs than the ones described")
    print(f"cache identity: all {len(man['arms'])} variant caches match their manifest hashes")

    print(f"\n{'arm':<22}{'ctx%':>7}{'EOS%':>7}{'cap%':>7}{'mean tok':>10}"
          f"{'exact_full':>12}{'exact':>8}{'vqa':>8}")
    for a, s in validity.items():
        f = lambda v: f"{v:6.1f}" if v is not None else "     -"
        print(f"{a:<22}{f(s['context_coverage_pct']):>7}{f(s['a_eos_pct']):>7}"
              f"{f(s['a_cap_hit_pct']):>7}{s['a_mean_generated']:>10.2f}"
              f"{s['a_exact_full_pct']:>12.2f}{s['a_exact_pct']:>8.2f}{s['a_vqa_pct']:>8.2f}")
    # The common B side, printed once because it is one run. Mean generated length is the
    # diagnostic that says whether an arm answered or merely stopped: it was computed and
    # silently dropped from the table in an earlier version.
    bs = validity[PRIMARY[0]]
    print(f"{'no_graph (common B)':<22}{'      -':>7}{bs['b_eos_pct']:>7.1f}"
          f"{bs['b_cap_hit_pct']:>7.1f}{bs['b_mean_generated']:>10.2f}"
          f"{bs['b_exact_full_pct']:>12.2f}{'':>8}{'':>8}")
    print("  ctx% = questions that actually received an injected graph; a silent fallback to "
          "the plain prompt would show here rather than as a null effect.")
    print("  mean tok = mean generated tokens; an arm that stops at ~2 tokens is not reasoning "
          "over the injected graph regardless of its accuracy.")
    res = analyse(arms, metric=args.metric)
    res["by_category_paired"] = category_effects(
        arms, sorted(arms[PRIMARY[0]]), categories,
        [("primary pred-wrong_image", PRIMARY[0], PRIMARY[1]),
         ("improvement pred-no_graph", PRIMARY[0], "no_graph")])
    print("  Categories are POST-HOC: they were not part of the pre-registered estimand, they "
          "are not multiplicity-corrected, and no single category licenses a claim.")

    # The shuffle arm is a no-op wherever a graph had fewer than two relations. Compute which
    # questions those are FROM THE CACHES rather than trusting the manifest's counter, then run
    # the restricted contrast as a pre-outcome sensitivity analysis. Full n stays primary for
    # this arm; the subset is a sensitivity check, not a replacement.
    gp = json.loads((ROOT / man["arms"]["pred"]["path"]).read_text())
    gs = json.loads((ROOT / man["arms"]["relations_shuffled"]["path"]).read_text())
    permuted = {q for q in gp
                if json.dumps(gp[q], sort_keys=True) != json.dumps(gs[q], sort_keys=True)}
    claimed = man["arms"]["relations_shuffled"]["n_actually_permuted"]
    if len(permuted) != claimed:
        raise SystemExit(f"FAIL manifest claims {claimed} permuted graphs but the caches show "
                         f"{len(permuted)}; the sensitivity subset would be wrong")
    # Only the relation-shuffle contrast is affected by the no-op graphs. Rerunning the primary,
    # the improvement contrast and the headroom on this subset would be a second pass over
    # unrelated estimands on a different n, which is a multiplicity problem, not a sensitivity
    # analysis. Compute the one contrast the subset bears on.
    sub = sorted(permuted)
    idx_s, sha_s = matrix(len(sub))
    ps, ls, hs = contrast(arms["relations_shuffled"], arms[PRIMARY[0]], sub, idx_s)
    print(f"\nSENSITIVITY (relation-shuffle contrast only; full n={len(gp)} remains primary "
          f"for this arm):")
    print(f"          relations_shuffled genuinely permuted {len(permuted)}/{len(gp)} graphs, "
          f"computed from the caches rather than read from the manifest")
    print(f"          relations_shuffled - pred | n={len(sub)} | matrix sha256={sha_s[:16]}"
          f"   {ps:+7.2f} [{ls:+.2f}, {hs:+.2f}]")
    res["sensitivity_relation_shuffle_permuted_only"] = {
        "n": len(sub), "matrix_sha256": sha_s, "contrast": f"relations_shuffled-{PRIMARY[0]}",
        "point": ps, "lo": ls, "hi": hs,
        "note": "pre-outcome sensitivity on the permuted subset only; full n stays primary"}

    res["validity"] = validity
    res["arm_roles"] = {
        "pred": "primary treatment",
        "wrong_image": "primary control (image-specific dependence)",
        "no_graph": "improvement baseline",
        "relations_shuffled": "CLEAN relation-content control — length-identical to pred",
        "objects_only": "DESCRIPTIVE ONLY — severely length-confounded (20.1 vs 83.0 tokens); "
                        "may not carry a relation-channel conclusion",
        "objects_filtered": "DESCRIPTIVE ONLY — a bundled practical ablation (count and "
                            "confidence change together), not a clean single-factor control",
        "oracle": "privileged ceiling — descriptive headroom ONLY; forbidden as a primary, promotion or inferential secondary comparator"}
    res["status"] = "EXPLORATORY — no confirmatory claim; the locked and confirmatory slices are spent"
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
