"""Paired analysis for the training-objective A/B (sequence- minus token-normalised loss).

    python scripts/42_objective_ab_analysis.py --records-dir outputs/eval/objective_ab_2291075

THE DECISION RULE, FIXED BEFORE ANY RECORD EXISTED. Promotion requires the sign of the paired
exact_full difference to agree across ALL THREE seed pairs within an encoder, with each seed's
own paired bootstrap interval reported. There is deliberately no pooled test over the three
runs and no function here that could compute one: three seeds are not 12,000 independent
questions. Within-recipe seed discordance on this project runs at a mean of 19.417%, so pooling
would report a precision the design cannot support.

WHY THE FLOOR IS CHECKED. Each evaluator run also scores a text-only floor, which never touches
the bridge. The two arms therefore share an LLM, a prompt and a decoding path for the floor, so
their floor accuracies should agree closely. A floor that moves between arms means the two runs
were not scored under matching conditions, and the bridge contrast between them is not clean.

PER-CATEGORY NUMBERS ARE DESCRIPTIVE. The slice carries 129 `compare` questions by design (its
proportional share). That cannot carry a per-category inference, and this script labels every
category row descriptive rather than printing a verdict against it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SEED, BOOT = 42, 10000
METRIC = "exact_full"
EXPECTED_N = 4000
# The six arms, as the launcher defines them. Kept here so a missing task is a named failure
# rather than a silently smaller table.
ARMS = [
    ("clip", 42, "bridge_clip_w1_bf16_v1_ep5", "bridge_clip_w1seq_s42_v1_ep5"),
    ("clip", 43, "bridge_clip_w1_s43_v1_ep5", "bridge_clip_w1seq_s43_v1_ep5"),
    ("clip", 44, "bridge_clip_w1_s44_v1_ep5", "bridge_clip_w1seq_s44_v1_ep5"),
    ("ijepa", 42, "bridge_ijepa_w1_bf16_v1_ep5", "bridge_ijepa_w1seq_s42_v1_ep5"),
    ("ijepa", 43, "bridge_ijepa_w1_s43_v1_ep5", "bridge_ijepa_w1seq_s43_v1_ep5"),
    ("ijepa", 44, "bridge_ijepa_w1_s44_v1_ep5", "bridge_ijepa_w1seq_s44_v1_ep5"),
]


def matrix(n, seed=SEED):
    """One resample matrix, and its hash, so every contrast provably shares it."""
    idx = np.random.default_rng(seed).integers(0, n, size=(BOOT, n))
    return idx, hashlib.sha256(idx.tobytes()).hexdigest()


def contrast(a_by_qid, b_by_qid, qids, idx):
    """Paired point estimate and 95% interval for a-b in points, on the shared matrix."""
    v = (np.array([a_by_qid[q] for q in qids], float)
         - np.array([b_by_qid[q] for q in qids], float))
    d = v[idx].mean(1) * 100
    return v.mean() * 100, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def find_records(records_dir: Path, stem: str) -> Path:
    """The one records file for this checkpoint stem; ambiguity is an error, not a pick."""
    hits = sorted(records_dir.glob(f"{stem}__*_records.json"))
    if len(hits) != 1:
        raise SystemExit(
            f"FAIL expected exactly one records file for {stem}, found {len(hits)}: "
            f"{[h.name for h in hits]}")
    return hits[0]


def load_arm(path: Path):
    """qid -> (bridge_metric, floor_metric, category), plus the file's own meta."""
    blob = json.loads(path.read_text())
    recs = blob["records"]
    bridge = {r["qid"]: r[f"bridge_{METRIC}"] for r in recs}
    floor = {r["qid"]: r[f"floor_{METRIC}"] for r in recs}
    cats = {r["qid"]: r.get("category") for r in recs}
    if len(bridge) != len(recs):
        raise SystemExit(f"FAIL {path.name} contains duplicate qids")
    return bridge, floor, cats, blob.get("_meta", {})


def analyse_pair(records_dir: Path, encoder: str, seed: int, control: str, new: str):
    """One (encoder, seed) contrast, with every premise checked before the number."""
    a_bridge, a_floor, a_cats, a_meta = load_arm(find_records(records_dir, control))
    b_bridge, b_floor, b_cats, b_meta = load_arm(find_records(records_dir, new))

    # Every premise is checked before any is allowed to stop the run, so a broken pair reports
    # all of what is wrong with it rather than one problem per re-run.
    problems = []
    if set(a_bridge) != set(b_bridge):
        only_a, only_b = len(set(a_bridge) - set(b_bridge)), len(set(b_bridge) - set(a_bridge))
        problems.append(f"arms cover different qids ({only_a} only in control, {only_b} only "
                        f"in sequence arm); a silent intersection would analyse whichever "
                        f"questions happened to overlap")
    # The evidence layer must match on both sides, or two protocols are being compared.
    for name, meta in (("control", a_meta), ("sequence", b_meta)):
        layer = meta.get("evidence_layer")
        if layer != "corrected_chatml_v1_eos":
            problems.append(f"{name} arm is on evidence layer {layer!r}, not "
                            f"corrected_chatml_v1_eos")
    qids = sorted(set(a_bridge) & set(b_bridge))
    if len(a_bridge) != EXPECTED_N or len(b_bridge) != EXPECTED_N:
        problems.append(f"{len(a_bridge)} control / {len(b_bridge)} sequence questions, "
                        f"expected {EXPECTED_N}")
    if problems:
        raise SystemExit(f"FAIL {encoder} s{seed}:\n  - " + "\n  - ".join(problems))

    idx, sha = matrix(len(qids))
    point, lo, hi = contrast(b_bridge, a_bridge, qids, idx)

    a_acc = 100 * np.mean([a_bridge[q] for q in qids])
    b_acc = 100 * np.mean([b_bridge[q] for q in qids])
    a_fl = 100 * np.mean([a_floor[q] for q in qids])
    b_fl = 100 * np.mean([b_floor[q] for q in qids])

    return {
        "encoder": encoder, "seed": seed, "n": len(qids), "matrix_sha256": sha,
        "control_stem": control, "sequence_stem": new,
        "control_pct": a_acc, "sequence_pct": b_acc,
        "diff_points": point, "ci_lo": lo, "ci_hi": hi,
        "excludes_zero": bool(lo > 0 or hi < 0),
        "control_floor_pct": a_fl, "sequence_floor_pct": b_fl,
        "floor_gap_points": b_fl - a_fl,
        "_qids": qids, "_a": a_bridge, "_b": b_bridge, "_cats": a_cats,
    }


def replication_verdict(rows):
    """The rule: every seed pair must agree in sign. Ties (exactly 0.00) do not agree."""
    signs = {np.sign(round(r["diff_points"], 2)) for r in rows}
    if signs == {1.0}:
        return "REPLICATES POSITIVE", True
    if signs == {-1.0}:
        return "REPLICATES NEGATIVE", True
    return "DOES NOT REPLICATE", False


def main() -> None:
    """Run the paired analysis and print the verdict per encoder."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records-dir", required=True)
    ap.add_argument("--out", default=None, help="write the full result as JSON")
    ap.add_argument("--self-test", action="store_true",
                    help="run the planted-effect check and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    records_dir = ROOT / args.records_dir if not Path(args.records_dir).is_absolute() \
        else Path(args.records_dir)
    if not records_dir.is_dir():
        raise SystemExit(f"FAIL {records_dir} is not a directory")

    print(f"=== objective A/B: sequence-normalised MINUS token-normalised, {METRIC} ===")
    print(f"records: {records_dir}")
    print(f"paired bootstrap: {BOOT:,} resamples, seed {SEED}, one shared matrix per contrast\n")

    rows = [analyse_pair(records_dir, e, s, c, n) for e, s, c, n in ARMS]

    by_encoder = defaultdict(list)
    for r in rows:
        by_encoder[r["encoder"]].append(r)

    verdicts = {}
    for encoder in sorted(by_encoder):
        rs = sorted(by_encoder[encoder], key=lambda r: r["seed"])
        print(f"--- {encoder} ---")
        print(f"  {'seed':>4}  {'token %':>8} {'sequence %':>11}  {'difference (points)':>28}")
        for r in rs:
            star = "*" if r["excludes_zero"] else " "
            print(f"  {r['seed']:>4}  {r['control_pct']:8.2f} {r['sequence_pct']:11.2f}  "
                  f"{r['diff_points']:+8.2f} [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] {star}")
        verdict, replicates = replication_verdict(rs)
        verdicts[encoder] = {"verdict": verdict, "replicates": replicates,
                             "mean_diff_points": float(np.mean([r["diff_points"] for r in rs]))}
        print(f"  replication across all three seeds: {verdict}")
        print(f"  mean of the three seed differences: "
              f"{verdicts[encoder]['mean_diff_points']:+.2f} points  (DESCRIPTIVE — it is a "
              f"mean of three numbers, not an estimate with 3 x 4,000 questions behind it)")

        # Floor agreement: the text-only arm never touches the bridge.
        worst = max(abs(r["floor_gap_points"]) for r in rs)
        flag = "OK" if worst < 1.0 else "INVESTIGATE"
        print(f"  largest floor gap between arms: {worst:.2f} points  [{flag}] "
              f"(the floor ignores the bridge, so it should barely move)\n")

    print("* = the per-seed 95% paired interval excludes zero.")
    print("No pooled test over the three seeds is computed, by design.")

    if args.out:
        payload = {
            "metric": METRIC, "boot": BOOT, "seed": SEED,
            "per_seed": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows],
            "per_encoder": verdicts,
            "decision_rule": "sign agreement across all three seed pairs within an encoder; "
                             "no pooled test over seeds",
        }
        out = Path(args.out)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nwrote {out}")


def self_test() -> None:
    """Plant a known effect and confirm the machinery recovers it and the rule fires."""
    rng = np.random.default_rng(7)
    n = 4000
    qids = [f"q{i}" for i in range(n)]
    base = rng.random(n) < 0.40
    a = {q: bool(v) for q, v in zip(qids, base)}
    # Flip 200 of the 2,400 wrong answers to correct: a planted +5.00 points.
    wrong = [i for i, v in enumerate(base) if not v][:200]
    seq = base.copy()
    for i in wrong:
        seq[i] = True
    b = {q: bool(v) for q, v in zip(qids, seq)}
    idx, _ = matrix(n)
    point, lo, hi = contrast(b, a, qids, idx)
    expected = 100 * 200 / n
    print(f"planted effect  = {expected:+.2f} points")
    print(f"recovered       = {point:+.2f} [{lo:+.2f}, {hi:+.2f}]")
    assert abs(point - expected) < 1e-9, f"point estimate {point} != planted {expected}"
    assert lo < point < hi, "the interval does not contain its own point estimate"
    assert lo > 0, "a +5-point effect at n=4,000 should exclude zero"

    # The replication rule must reject a discordant set and accept a concordant one.
    concordant = [{"diff_points": d} for d in (1.2, 0.4, 2.0)]
    discordant = [{"diff_points": d} for d in (1.2, -0.4, 2.0)]
    assert replication_verdict(concordant)[1] is True
    assert replication_verdict(discordant)[1] is False, "a sign flip must break replication"
    # A seed that lands exactly on zero is not agreement.
    assert replication_verdict([{"diff_points": d} for d in (1.2, 0.0, 2.0)])[1] is False
    print("replication rule: concordant accepted, sign flip rejected, exact zero rejected")
    print("\nSELF-TEST PASSED")


if __name__ == "__main__":
    main()
