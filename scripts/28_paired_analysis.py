"""Paired analysis for augmentation experiments: effects, bootstrap CIs, flips, coverage.

scripts/09_augment_eval.py generates and scores; it does not produce reportable statistics.
This script is the analysis layer over its saved `*_records.json`, kept separate so a
re-analysis never re-runs generation (a 500-question arm costs ~10 GPU-minutes, and
re-generating to change a confidence interval would be indefensible).

Two comparison modes:

  WITHIN-ARM   augmented vs its own baseline, paired on qid inside one records file.
  ARM-TO-ARM   arm X's augmented answers vs arm Y's augmented answers, paired on qid
               across two records files. This is the ONLY correct way to ask whether two
               augmentations differ. Comparing "G2's CI excludes zero but G1's does not"
               is not a contrast — two intervals overlapping zero differently says nothing
               about their difference, which has its own, usually narrower, interval.

    python scripts/28_paired_analysis.py --records A.json                    # within-arm
    python scripts/28_paired_analysis.py --records B.json --versus A.json    # arm-to-arm
    python scripts/28_paired_analysis.py --records A.json B.json C.json --all-contrasts
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import REASONING_CATEGORIES  # noqa: E402
from src.models.revisions import revision_for  # noqa: E402

METRICS = ("exact_full", "exact", "vqa")
N_RESAMPLES = 10_000
SEED = 42
ANALYSIS_VERSION = "paired_analysis/1.0.0"


def load_records(path):
    """Return (records, meta) from a records artifact, tolerating the legacy bare-list form."""
    payload = json.load(open(path))
    if isinstance(payload, dict):
        return payload["records"], payload.get("_meta", {})
    return payload, {}


def index_by_qid(records, *, source):
    """Map qid -> record, refusing duplicates (a duplicated qid silently double-weights it)."""
    out = {}
    for rec in records:
        qid = str(rec["qid"])
        if qid in out:
            raise SystemExit(f"duplicate qid {qid!r} in {source}: pairing would be ambiguous")
        out[qid] = rec
    return out


def paired_qids(left, right, *, left_name, right_name):
    """Intersect two qid indexes, reporting any asymmetry rather than silently dropping it."""
    common = sorted(set(left) & set(right))
    if not common:
        raise SystemExit(f"no shared qids between {left_name} and {right_name}")
    only_left, only_right = len(left) - len(common), len(right) - len(common)
    if only_left or only_right:
        print(f"WARNING pairing on {len(common)} shared qids "
              f"({only_left} only in {left_name}, {only_right} only in {right_name})")
    return common


def bootstrap_ci(deltas, *, n_resamples=N_RESAMPLES, seed=SEED):
    """Percentile bootstrap 95% CI of the mean of per-question paired deltas.

    Resampling QUESTIONS (not the two arms independently) is what makes this paired: each
    resample keeps both arms' outcomes for the same question together, so the shared
    difficulty of a question cancels instead of inflating the interval.
    """
    if not deltas:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    n = len(deltas)
    point = 100.0 * sum(deltas) / n
    means = []
    for _ in range(n_resamples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = 100.0 * means[int(0.025 * n_resamples)]
    hi = 100.0 * means[int(0.975 * n_resamples) - 1]
    return point, lo, hi


def _correct(rec, side, metric):
    """Correctness flag for one side ('a' augmented / 'b' baseline) under one metric."""
    return bool(rec[f"{side}_{metric}"])


def contrast(left_idx, right_idx, qids, *, left_side="a", right_side="a"):
    """Paired effect + CI + flips for every metric, over `qids`, left MINUS right."""
    out = {}
    for metric in METRICS:
        deltas, flips = [], Counter()
        for qid in qids:
            lo_ = _correct(left_idx[qid], left_side, metric)
            ro_ = _correct(right_idx[qid], right_side, metric)
            deltas.append(float(lo_) - float(ro_))
            if ro_ and not lo_:
                flips["right_to_wrong"] += 1
            elif lo_ and not ro_:
                flips["wrong_to_right"] += 1
            elif lo_ and ro_:
                flips["both_correct"] += 1
            else:
                flips["both_wrong"] += 1
        point, lo_ci, hi_ci = bootstrap_ci(deltas)
        n = len(qids)
        out[metric] = {
            "left_pct": 100.0 * sum(_correct(left_idx[q], left_side, metric) for q in qids) / n,
            "right_pct": 100.0 * sum(_correct(right_idx[q], right_side, metric) for q in qids) / n,
            "effect": point, "ci_lo": lo_ci, "ci_hi": hi_ci,
            "excludes_zero": (lo_ci > 0) or (hi_ci < 0),
            "wrong_to_right": flips["wrong_to_right"],
            "right_to_wrong": flips["right_to_wrong"],
            "both_correct": flips["both_correct"],
            "both_wrong": flips["both_wrong"],
        }
    return out


def per_category(left_idx, right_idx, qids, *, left_side="a", right_side="a",
                 metric="exact_full"):
    """Paired effect + CI per GQA reasoning category (categories with no questions omitted)."""
    buckets = {}
    for qid in qids:
        buckets.setdefault(left_idx[qid].get("category", "other"), []).append(qid)
    rows = {}
    for cat in REASONING_CATEGORIES + ["other"]:
        cat_qids = buckets.get(cat)
        if not cat_qids:
            continue
        rows[cat] = {"n": len(cat_qids),
                     **contrast(left_idx, right_idx, cat_qids,
                                left_side=left_side, right_side=right_side)[metric]}
    return rows


def output_health(records, side):
    """Termination and output-validity diagnostics for one side of an arm.

    src/models/vlm.py:165 writes the stop reason as f"eos_{token_id}" (e.g. "eos_151645")
    or "cap" — never the bare string "eos". Matching on equality with "eos" reported 0.0%
    termination for arms that in fact stopped cleanly on every question, which is a
    reporting defect of exactly the kind the corrected protocol exists to catch. The
    distinct reasons are surfaced too, so an unexpected stop token shows up as itself
    rather than being folded into the EOS rate: the project only ever supervises 151645,
    and a stop on Qwen's alternate EOS would make "the bridge learned to stop" unfalsifiable.
    """
    n = len(records)
    if not n:
        return {}
    gen = [r.get(f"{side}_n_generated", 0) for r in records]
    empty = sum(1 for r in records if not str(r.get(f"{side}_ans", "")).strip())
    caps = sum(1 for r in records if r.get(f"{side}_cap_hit"))
    reasons = Counter(str(r.get(f"{side}_stop_reason", "")) for r in records)
    eos = sum(c for reason, c in reasons.items() if reason.startswith("eos"))
    return {"n": n,
            "eos_rate": 100.0 * eos / n, "cap_hit_rate": 100.0 * caps / n,
            "empty_rate": 100.0 * empty / n,
            "stop_reasons": dict(reasons),
            "mean_generated": statistics.mean(gen), "median_generated": statistics.median(gen)}


def context_stats(records, tokenizer=None):
    """Graph coverage and injected-context length, in characters and (if available) tokens.

    Token counts come from the real Qwen tokenizer over the SAVED context string, so no
    generation pass is repeated. Without a tokenizer the character stats still bound the
    'did this arm simply inject more tokens' question.
    """
    ctxs = [r.get("a_context") or "" for r in records]
    present = [c for c in ctxs if c.strip()]
    stats = {"n": len(ctxs), "n_with_graph": len(present),
             "coverage_pct": 100.0 * len(present) / len(ctxs) if ctxs else 0.0}
    if present:
        chars = [len(c) for c in present]
        stats.update(mean_chars=statistics.mean(chars), median_chars=statistics.median(chars))
        if tokenizer is not None:
            toks = [len(tokenizer.encode(c, add_special_tokens=False)) for c in present]
            toks.sort()
            stats.update(mean_tokens=statistics.mean(toks),
                         median_tokens=statistics.median(toks),
                         p10_tokens=toks[int(0.10 * len(toks))],
                         p90_tokens=toks[int(0.90 * len(toks)) - 1],
                         max_tokens=toks[-1])
    return stats


def load_tokenizer(model_id="Qwen/Qwen2-7B-Instruct"):
    """Load the Qwen tokenizer for context-length stats, or None if unavailable offline."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_id, revision=revision_for(model_id))
    except Exception as exc:                       # noqa: BLE001 — diagnostics must not abort
        print(f"NOTE tokenizer unavailable ({exc.__class__.__name__}); "
              f"reporting context length in characters only")
        return None


def _fmt(row):
    """One-line 'left vs right: effect [lo, hi]' with a significance marker."""
    mark = "  *" if row["excludes_zero"] else ""
    return (f"{row['left_pct']:5.1f}% vs {row['right_pct']:5.1f}%   "
            f"{row['effect']:+6.1f} [{row['ci_lo']:+.1f}, {row['ci_hi']:+.1f}]{mark}")


def report_arm(path, tokenizer):
    """Within-arm analysis: augmented vs its own paired baseline."""
    records, meta = load_records(path)
    idx = index_by_qid(records, source=Path(path).name)
    qids = sorted(idx)
    print(f"\n{'=' * 78}\nWITHIN-ARM  {Path(path).name}")
    print(f"  augmentation : {meta.get('augmentation')}   framing: {meta.get('graph_framing')}")
    print(f"  evidence     : {meta.get('evidence_layer')}   metric: {meta.get('metric_version')}")
    print(f"  checkpoint   : {meta.get('checkpoint_tag')} ep{meta.get('checkpoint_epoch')}  "
          f"{meta.get('checkpoint_encoder')}  {meta.get('llm_precision')}")
    print(f"  split        : {meta.get('split')}   cache: {meta.get('pred_cache')}")
    print(f"  n questions  : {len(qids)}")

    res = contrast(idx, idx, qids, left_side="a", right_side="b")
    print(f"\n  {'metric':<12}{'augmented vs baseline':<38}{'flips (W->R / R->W)'}")
    for metric in METRICS:
        row = res[metric]
        print(f"  {metric:<12}{_fmt(row):<38}{row['wrong_to_right']:>4} / {row['right_to_wrong']:<4}"
              f"  (both ok {row['both_correct']}, both wrong {row['both_wrong']})")

    print(f"\n  per-category (exact_full):")
    for cat, row in per_category(idx, idx, qids, left_side="a", right_side="b").items():
        print(f"    {cat:<10} n={row['n']:<5}{_fmt(row)}")

    print(f"\n  output health:")
    for side, label in (("a", "augmented"), ("b", "baseline ")):
        h = output_health(records, side)
        print(f"    {label}  EOS {h['eos_rate']:5.1f}%  cap-hit {h['cap_hit_rate']:4.1f}%  "
              f"empty {h['empty_rate']:4.1f}%  mean {h['mean_generated']:.2f} tok  "
              f"median {h['median_generated']:.0f}   stop={h['stop_reasons']}")

    cs = context_stats(records, tokenizer)
    line = (f"    coverage {cs['n_with_graph']}/{cs['n']} ({cs['coverage_pct']:.1f}%)")
    if "mean_tokens" in cs:
        line += (f"   context tokens mean {cs['mean_tokens']:.0f} median {cs['median_tokens']:.0f}"
                 f" p10 {cs['p10_tokens']} p90 {cs['p90_tokens']} max {cs['max_tokens']}")
    elif "mean_chars" in cs:
        line += f"   context chars mean {cs['mean_chars']:.0f} median {cs['median_chars']:.0f}"
    print(f"\n  injected context:\n{line}")
    return {"path": str(path), "meta": meta, "n": len(qids), "within_arm": res,
            "per_category": per_category(idx, idx, qids, left_side="a", right_side="b"),
            "output_health": {s: output_health(records, s) for s in ("a", "b")},
            "context": cs}


def report_contrast(left_path, right_path):
    """Arm-to-arm analysis: two augmentations' answers paired on shared qids."""
    l_recs, l_meta = load_records(left_path)
    r_recs, r_meta = load_records(right_path)
    l_idx = index_by_qid(l_recs, source=Path(left_path).name)
    r_idx = index_by_qid(r_recs, source=Path(right_path).name)
    qids = paired_qids(l_idx, r_idx,
                       left_name=Path(left_path).name, right_name=Path(right_path).name)
    l_name = l_meta.get("augmentation", Path(left_path).stem)
    r_name = r_meta.get("augmentation", Path(right_path).stem)
    print(f"\n{'=' * 78}\nARM-TO-ARM  {l_name}  MINUS  {r_name}   (n={len(qids)})")
    res = contrast(l_idx, r_idx, qids, left_side="a", right_side="a")
    for metric in METRICS:
        row = res[metric]
        print(f"  {metric:<12}{_fmt(row):<38}"
              f"{row['wrong_to_right']:>4} / {row['right_to_wrong']:<4}")
    print(f"  per-category (exact_full):")
    for cat, row in per_category(l_idx, r_idx, qids, left_side="a", right_side="a").items():
        print(f"    {cat:<10} n={row['n']:<5}{_fmt(row)}")
    return {"left": l_name, "right": r_name, "n": len(qids), "metrics": res,
            "per_category": per_category(l_idx, r_idx, qids, left_side="a", right_side="a")}


def main() -> None:
    """Run within-arm and arm-to-arm paired analyses over augmentation records."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", nargs="+", required=True, help="records JSON artifact(s)")
    ap.add_argument("--versus", default=None, help="compare --records[0] against this arm")
    ap.add_argument("--all-contrasts", action="store_true",
                    help="every ordered pair among --records, in the order given")
    ap.add_argument("--out", default=None, help="write the full analysis as JSON")
    ap.add_argument("--no-tokenizer", action="store_true", help="skip Qwen context-token counts")
    args = ap.parse_args()

    for path in args.records + ([args.versus] if args.versus else []):
        if not Path(path).is_file():
            raise SystemExit(f"records file not found: {path}")

    tokenizer = None if args.no_tokenizer else load_tokenizer()
    print(f"{ANALYSIS_VERSION}   paired bootstrap {N_RESAMPLES:,} resamples, seed {SEED}")
    print("'*' marks an interval that excludes zero. A point estimate whose interval "
          "crosses zero is NOT an effect.")

    payload = {"analysis_version": ANALYSIS_VERSION, "n_resamples": N_RESAMPLES, "seed": SEED,
               "arms": [report_arm(p, tokenizer) for p in args.records], "contrasts": []}

    if args.versus:
        payload["contrasts"].append(report_contrast(args.records[0], args.versus))
    if args.all_contrasts:
        for i, left in enumerate(args.records):
            for right in args.records[:i]:
                payload["contrasts"].append(report_contrast(left, right))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote analysis -> {args.out}")


if __name__ == "__main__":
    main()
