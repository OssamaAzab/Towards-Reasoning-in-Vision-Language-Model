"""Paired analysis for the external learned-SGG (RelTR / Open Images V6) experiment.

    python scripts/47_reltr_analysis.py --records-dir outputs/eval/reltr_<jobid> \
        --manifest outputs/eval/reltr_<jobid>/MANIFEST.snapshot.json

WHY A SEPARATE SCRIPT FROM scripts/38. T-049's analysis is bound to its eight-arm set — it
indexes `relations_shuffled` and `objects_filtered` directly — so bending it to four conditions
would mean editing a script that produced a reported number (C38). The statistical core here
(`matrix`, `contrast`, SEED, BOOT) is copied from it verbatim so the two experiments' intervals
are computed identically and are directly comparable.

THE ESTIMANDS, FIXED BEFORE ANY RECORD EXISTED.
  PRIMARY    reltr MINUS reltr_wrong_image  — is the graph read as evidence about THIS image?
  SECONDARY  reltr MINUS no_graph           — does it help at all?
Both are needed. Dependence without improvement is possible: a graph can beat a wrong-image
graph while both are worse than injecting nothing.

THE ORACLE IS A PRIVILEGED CEILING. It injects the ground-truth GQA graph, which is the
structure the questions were generated from. It appears only in the descriptive headroom
diagnostic, carries no interval, and is refused as a primary or inferential comparator.

A NULL IS NOT EQUIVALENCE. An interval including zero means no detectable difference at the
available precision. This script prints the half-width next to every interval so the precision
is never left implicit.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED, BOOT = 42, 10000
TREATMENT, CONTROL, ORACLE = "reltr", "reltr_wrong_image", "oracle"
BASELINE = "no_graph"

# The whole no-graph run, not just its score: rewriting baseline answers without changing whether
# they were correct must not pass a "byte-identical" claim.
BASELINE_FIELDS = ("b_ans", "b_raw", "b_exact_full", "b_exact", "b_vqa",
                   "b_stop_reason", "b_cap_hit", "b_n_generated")


def sha256_file(p: Path) -> str:
    """Hash a file so the analysis verifies what it read."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def matrix(n, seed=SEED):
    """One resample matrix, and its hash, so every contrast provably shares it."""
    idx = np.random.default_rng(seed).integers(0, n, size=(BOOT, n))
    return idx, hashlib.sha256(idx.tobytes()).hexdigest()


def contrast(a, b, qids, idx):
    """Paired point estimate and 95% interval for a-b in points, on the shared matrix."""
    v = np.array([a[q] for q in qids], float) - np.array([b[q] for q in qids], float)
    d = v[idx].mean(1) * 100
    return v.mean() * 100, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def arm_pattern(arm: str) -> re.Pattern:
    """`<stem>__<arm>_records.json`: the arm token ends at `__` or at `_records.json`."""
    return re.compile(rf"__{re.escape(arm)}(?:__|_records\.json$)")


def find_arm_file(records_dir: str, arm: str) -> str:
    """The one records file for this arm; ambiguity is an error, never a pick."""
    hits = sorted(p for p in glob.glob(str(Path(records_dir) / "*records.json"))
                  if arm_pattern(arm).search(Path(p).name))
    if len(hits) != 1:
        raise SystemExit(f"FAIL expected exactly one records file for arm {arm!r} in "
                         f"{records_dir}, found {len(hits)}: {[Path(h).name for h in hits]}")
    return hits[0]


def load_arm(records_dir: str, arm: str, metric: str, n_expect: int, side: str = "a"):
    """qid -> metric for one arm ('a' = the injected side, 'b' = the shared no-graph side)."""
    path = find_arm_file(records_dir, arm)
    blob = json.loads(Path(path).read_text())
    recs = blob["records"]
    field = f"{side}_{metric}"
    if field not in recs[0]:
        raise SystemExit(f"FAIL records lack {field!r} (have: {sorted(recs[0])[:10]})")
    out = {r["qid"]: bool(r[field]) for r in recs}
    if len(out) != len(recs):
        raise SystemExit(f"FAIL arm {arm!r} has duplicate QIDs")
    if len(out) != n_expect:
        raise SystemExit(f"FAIL arm {arm!r} has {len(out)} questions, expected {n_expect}")
    return out, blob, path


def validate(arm, blob, path, man):
    """Every premise checked before any number: protocol, provenance, cache identity, coverage."""
    meta = blob.get("_meta", {})
    problems = []
    for k, v in (man.get("expected_common_meta") or {}).items():
        if meta.get(k) != v:
            problems.append(f"{k}={meta.get(k)!r}, manifest expects {v!r}")
    for k, v in (man.get("expected_record_meta") or {}).items():
        if meta.get(k) != v:
            problems.append(f"{k}={meta.get(k)!r}, checkpoint says {v!r}")
    exp = (man.get("expected_arm_meta") or {}).get(arm)
    if exp is None:
        problems.append(f"the manifest defines no expectations for arm {arm!r}")
    else:
        for k, v in exp.items():
            got = meta.get(k)
            if k == "pred_cache" and got is not None and v is not None:
                got = str(got).replace("\\", "/")
                v = str(v).replace("\\", "/")
            if got != v:
                problems.append(f"{k}={got!r}, manifest expects {v!r}")
    if problems:
        raise SystemExit(f"FAIL arm {arm!r} ({Path(path).name}):\n  - " + "\n  - ".join(problems))
    return meta


def self_test() -> None:
    """Plant a known effect and confirm the machinery recovers it and the guards fire."""
    rng = np.random.default_rng(7)
    n = 485
    qids = [f"q{i}" for i in range(n)]
    base = rng.random(n) < 0.51
    ctrl = {q: bool(v) for q, v in zip(qids, base)}
    treat = base.copy()
    flip = [i for i, v in enumerate(base) if not v][:24]      # +24/485 = +4.95 points
    for i in flip:
        treat[i] = True
    tr = {q: bool(v) for q, v in zip(qids, treat)}
    idx, sha = matrix(n)
    p, lo, hi = contrast(tr, ctrl, qids, idx)
    expected = 100 * len(flip) / n
    print(f"planted   = {expected:+.2f} points")
    print(f"recovered = {p:+.2f} [{lo:+.2f}, {hi:+.2f}]  (matrix {sha[:16]}…)")
    assert abs(p - expected) < 1e-9, f"point {p} != planted {expected}"
    assert lo < p < hi, "the interval does not contain its own point estimate"
    assert lo > 0, "a ~5-point effect at n=485 should exclude zero"

    # A true null must NOT be reported as an effect.
    p0, lo0, hi0 = contrast(ctrl, ctrl, qids, idx)
    assert p0 == 0.0 and lo0 == 0.0 and hi0 == 0.0, "an arm against itself must be exactly zero"

    # The oracle must be refusable as a primary comparator.
    try:
        analyse({ORACLE: ctrl, CONTROL: ctrl, BASELINE: ctrl}, primary=(ORACLE, CONTROL))
    except SystemExit:
        pass
    else:
        raise AssertionError("the oracle was accepted as a primary comparator")
    print("\nSELF-TEST PASSED")


def analyse(arms: dict, primary=(TREATMENT, CONTROL), metric="exact_full"):
    """Print the primary and secondary contrasts, then the descriptive headroom."""
    if ORACLE in primary:
        raise SystemExit("FAIL the oracle is a privileged ceiling and may not be a primary "
                         "or inferential comparator")
    keysets = {n: frozenset(a) for n, a in arms.items()}
    if len(set(keysets.values())) != 1:
        raise SystemExit(f"FAIL arms do not cover identical QID sets: "
                         f"{ {n: len(k) for n, k in keysets.items()} }")
    qids = sorted(next(iter(keysets.values())))
    idx, sha = matrix(len(qids))
    print(f"\n=== {metric} | n={len(qids)} | 10,000 resamples, seed {SEED} | "
          f"matrix sha256={sha[:16]}… ===")

    out = {"n": len(qids), "matrix_sha256": sha, "boot": BOOT, "seed": SEED}

    p, lo, hi = contrast(arms[primary[0]], arms[primary[1]], qids, idx)
    star = "*" if (lo > 0 or hi < 0) else " "
    print(f"PRIMARY      {primary[0]} - {primary[1]:<20} {p:+7.2f} [{lo:+.2f}, {hi:+.2f}] {star}")
    print(f"             half-width = {(hi - lo) / 2:.2f} points  <- what this slice resolves")
    if lo <= 0 <= hi:
        print("             INTERVAL INCLUDES ZERO — no detectable difference at the available")
        print("             precision. This is NOT evidence that the two arms are equivalent.")
    out["primary"] = {"contrast": f"{primary[0]}-{primary[1]}", "point": p, "lo": lo, "hi": hi,
                      "half_width": (hi - lo) / 2, "excludes_zero": bool(lo > 0 or hi < 0)}

    if BASELINE in arms:
        p2, l2, h2 = contrast(arms[primary[0]], arms[BASELINE], qids, idx)
        star = "*" if (l2 > 0 or h2 < 0) else " "
        print(f"SECONDARY    {primary[0]} - {BASELINE:<20} {p2:+7.2f} [{l2:+.2f}, {h2:+.2f}] {star}")
        print(f"             half-width = {(h2 - l2) / 2:.2f} points")
        if l2 <= 0 <= h2:
            print("             INTERVAL INCLUDES ZERO — no detectable improvement over no graph")
            print("             at the available precision; not a demonstration of equivalence.")
        out["secondary"] = {"contrast": f"{primary[0]}-{BASELINE}", "point": p2, "lo": l2,
                            "hi": h2, "half_width": (h2 - l2) / 2,
                            "excludes_zero": bool(l2 > 0 or h2 < 0)}

    for name in (primary[0], primary[1], BASELINE, ORACLE):
        if name in arms:
            out.setdefault("accuracy", {})[name] = 100 * float(
                np.mean([arms[name][q] for q in qids]))

    if ORACLE in arms and BASELINE in arms:
        og, ng = out["accuracy"][ORACLE], out["accuracy"][BASELINE]
        pr = out["accuracy"][primary[0]]
        head = og - ng
        print("\nHEADROOM DIAGNOSTIC — DESCRIPTIVE. The oracle is differenced here and NOWHERE")
        print("             else. No interval is computed for it and none may be inferred.")
        print(f"             {BASELINE} {ng:6.2f}%   {primary[0]} {pr:6.2f}%   oracle {og:6.2f}%")
        captured = None
        if head > 0:
            captured = 100 * (pr - ng) / head
            print(f"             oracle headroom over {BASELINE} = {head:+.2f} points; the "
                  f"learned graph captures {captured:.1f}% of it")
        else:
            print(f"             oracle headroom = {head:+.2f} points: the ceiling is NOT above "
                  f"the baseline, so no fraction is reported.")
        out["headroom"] = {"no_graph": ng, "treatment": pr, "oracle": og,
                           "oracle_minus_no_graph": head, "captured_pct": captured}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records-dir")
    ap.add_argument("--manifest")
    ap.add_argument("--metric", default="exact_full")
    ap.add_argument("--out", default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.records_dir or not args.manifest:
        raise SystemExit("FAIL --records-dir and --manifest are required")

    man = json.loads(Path(args.manifest).read_text())

    # The surface and the checkpoint must be the ones the caches were built against.
    surf = ROOT / man["development_surface"]
    if not surf.is_file():
        raise SystemExit(f"FAIL development surface {surf} is missing")
    if sha256_file(surf) != man["development_surface_sha256"]:
        raise SystemExit(f"FAIL {surf.name} changed since the caches were built")
    expected_qids = json.loads(surf.read_text())
    n_expect = len(expected_qids)

    ck = ROOT / "outputs" / "checkpoints" / "corrected" / f"{man['checkpoint_for_evaluation']}.pt"
    if not man.get("checkpoint_sha256"):
        raise SystemExit("FAIL the manifest pinned no checkpoint hash")
    if not ck.is_file():
        raise SystemExit(f"FAIL checkpoint {ck.name} is missing; its hash cannot be verified")
    if sha256_file(ck) != man["checkpoint_sha256"]:
        raise SystemExit(f"FAIL checkpoint {ck.name} changed since the caches were built")

    print("=== external learned SGG (RelTR / Open Images V6) — paired analysis ===")
    print(f"surface   : {man['development_surface']} ({n_expect} questions)")
    print(f"checkpoint: {man['checkpoint_for_evaluation']} sha256 {man['checkpoint_sha256'][:16]}…")
    print(f"leakage   : {man['leakage']}")

    # Each condition cache must still be the file whose hash the manifest recorded.
    for a, spec in man["arms"].items():
        p = ROOT / spec["path"]
        if not p.is_file():
            raise SystemExit(f"FAIL condition cache {spec['path']} is missing")
        if sha256_file(p) != spec["sha256"]:
            raise SystemExit(f"FAIL condition cache {a} changed since the manifest was written")

    arms, blobs = {}, {}
    for a in [TREATMENT, CONTROL, ORACLE]:
        arms[a], blobs[a], path = load_arm(args.records_dir, a, args.metric, n_expect)
        # validate() already compares every field of expected_arm_meta, n_context_injected
        # included, so a separate coverage check here could never fire. It is reported rather
        # than re-checked: an unreachable guard reads as protection that is not there.
        meta = validate(a, blobs[a], path, man)
        print(f"  arm {a:<20} n={len(arms[a])}  injected={meta.get('n_context_injected')}")

    # The no-graph baseline is the B side of the treatment run, and must be identical in every run.
    arms[BASELINE], _, _ = load_arm(args.records_dir, TREATMENT, args.metric, n_expect, side="b")
    ref = {r["qid"]: tuple(r.get(f) for f in BASELINE_FIELDS)
           for r in blobs[TREATMENT]["records"]}
    for a in (CONTROL, ORACLE):
        got = {r["qid"]: tuple(r.get(f) for f in BASELINE_FIELDS) for r in blobs[a]["records"]}
        if got != ref:
            diff = sum(1 for q in ref if ref[q] != got.get(q))
            raise SystemExit(f"FAIL the no-graph baseline in arm {a!r} differs from {TREATMENT!r} "
                             f"on {diff} questions; the contrasts share no common control")
    print(f"  baseline  {BASELINE:<20} byte-identical across all three runs")

    res = analyse(arms, metric=args.metric)
    res["provenance"] = {
        "surface": man["development_surface"],
        "surface_sha256": man["development_surface_sha256"],
        "checkpoint": man["checkpoint_for_evaluation"],
        "checkpoint_sha256": man["checkpoint_sha256"],
        "leakage": man["leakage"],
        "arms": {a: man["arms"][a]["sha256"] for a in man["arms"]},
        "generator": man.get("sources", {}),
    }
    res["interpretation_rule"] = ("an interval including zero is 'no detectable difference at "
                                  "the available precision', never a demonstration of equivalence")
    print("\nEXPLORATORY. The tuning surface is a development slice; nothing here is confirmatory.")

    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
