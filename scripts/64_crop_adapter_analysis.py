"""Stage B paired analysis: both success criteria, or no success. CPU only (arithmetic on records).

    python scripts/64_crop_adapter_analysis.py --records outputs/crop_adapter/eval_<id>/adapter_records.json

EXPLORATORY — SUPERVISED GQA ORACLE-REGION ADAPTER — NOT ZERO-SHOT — NOT DEPLOYABLE.

THE CLEAN EVIDENCE-SPECIFIC COMPARISONS are correct-vs-wrong, correct-vs-irrelevant and
correct-vs-null-visual. All three hold the trained adapter, the token positions and the question
pathway fixed and vary only the visual information reaching the crop encoder.

THE BASELINE COMPARISON IS USEFUL BUT CONFOUNDED. `adapter_correct_crop - baseline_frozen`
combines FOUR things at once: GQA supervision the baseline never had, new trainable parameters,
extra token positions, and the crop evidence itself. It is not a clean crop effect and is not
described as an upper bound on one — an upper bound would imply the other three only inflate it,
which is not established.

WHY NULL-VISUAL IS THE DECIDING CONTROL. `q_proj` is trainable and reads the question, so the
adapter can learn GQA answer priors and emit them through the evidence tokens without using the
crop at all. The null-visual arm gives it precisely that: same positions, same question pathway,
same live gate, all-zero visual features. A gain over the baseline that does not survive against
null-visual is supervision and question priors, not evidence.

THE FIVE PRE-REGISTERED CLASSES:

  PROMISING SUPERVISED EVIDENCE GAIN   correct beats the frozen baseline, wrong AND null-visual
  EVIDENCE-SPECIFIC BUT NO NET GAIN    correct beats wrong and null-visual but not the baseline
  SUPERVISED/QUESTION-PRIOR GAIN       correct beats the baseline but not null-visual
  ROBUSTNESS ONLY                      correct restores the baseline while wrong evidence hurts
  STOP                                 correct beats neither wrong nor null-visual

ONE SEED. The exploratory one-seed label is retained regardless of any interval's width or
significance. Nothing here is a confirmatory result.

SMOKE VERSUS SCIENCE. `scientific_eligible` is false whenever the run is a smoke or its checkpoint
came from an overfit smoke, and the reasons are stored in `non_scientific_reasons`. An ineligible
run still exits 0 when its plumbing checks all pass — its numbers are correct, they simply cannot
carry a claim. A FAILED VALIDITY CHECK is a different state and always exits non-zero, including
during a smoke. Offering an overfit checkpoint for a non-smoke evaluation is such a failure.

TRUNCATION POLICY (C41), unchanged: report by arm; fail outright if a treatment arm truncates;
bound both the interval and the point estimate for affected contrasts; fail if a bound could
change the qualitative conclusion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED, BOOT = 42, 10000
LABEL = "EXPLORATORY — SUPERVISED GQA ORACLE-REGION ADAPTER — NOT ZERO-SHOT — NOT DEPLOYABLE"

VS_BASELINE = ("adapter_correct_crop", "baseline_frozen")     # confounded, never alone decisive
VS_WRONG = ("adapter_correct_crop", "adapter_wrong_crop")     # evidence-specific
VS_NULL = ("adapter_correct_crop", "adapter_null_visual")     # evidence-specific, the decider
VS_IRRELEVANT = ("adapter_correct_crop", "adapter_irrelevant_crop")
WRONG_VS_BASELINE = ("adapter_wrong_crop", "baseline_frozen")  # does wrong evidence HURT?
CONTRASTS = [
    (*VS_BASELINE, "core", "vs FROZEN BASELINE — CONFOUNDED: combines GQA supervision, new "
                           "parameters, extra token positions and crop evidence. Not a clean "
                           "crop effect and not an upper bound on one."),
    (*VS_WRONG, "core", "EVIDENCE-SPECIFIC — same adapter, same span, same-label crop from a "
                        "different image"),
    (*VS_NULL, "core", "EVIDENCE-SPECIFIC, THE DECIDER — same adapter, same span, same live "
                       "gate, same question pathway, ALL-ZERO visual features"),
    (*VS_WRONG, "scale_matched", "ROBUSTNESS — evidence-specific vs wrong where both crops reach "
                                 "the encoder at comparable resolution"),
    (*VS_IRRELEVANT, "irrelevant_subset",
     "EVIDENCE-SPECIFIC — same image, non-referent, area-matched region"),
    ("adapter_correct_crop", "adapter_zero_gate", "core",
     "does the LEARNED GATE matter, holding the evidence positions fixed?"),
    (*WRONG_VS_BASELINE, "core",
     "does WRONG evidence actively hurt? (needed to distinguish ROBUSTNESS ONLY)"),
    ("no_adapter_span", "baseline_frozen", "core",
     "PARITY — must be exactly zero; no_adapter_span IS the baseline sequence"),
]
# Arms whose effect is being claimed. Truncation here corrupts the measurement of the thing being
# claimed, so it fails outright rather than being bounded.
TREATMENT_ARMS = ("adapter_correct_crop",)


def sha256_file(p: Path) -> str:
    """Hash the records so the analysis pins what it read."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def q_matrix(n, seed=SEED):
    """Question-level resample matrix and its hash, shared by every contrast on a cohort."""
    idx = np.random.default_rng(seed).integers(0, n, size=(BOOT, n))
    return idx, hashlib.sha256(idx.tobytes()).hexdigest()


def cluster_draws(qids, image_of, seed=SEED):
    """Image-clustered resamples: draw IMAGES with replacement, take all their questions."""
    rng = np.random.default_rng(seed)
    imgs = sorted({image_of[q] for q in qids})
    by_img = defaultdict(list)
    for i, q in enumerate(qids):
        by_img[image_of[q]].append(i)
    return [np.concatenate([by_img[imgs[j]] for j in rng.integers(0, len(imgs), size=len(imgs))])
            for _ in range(BOOT)], len(imgs)


def ci_from_draws(v, draws):
    """Point estimate and percentile interval for a difference vector under given resamples."""
    d = np.array([v[ix].mean() for ix in draws]) * 100
    return v.mean() * 100, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def arm(records, name, field="exact_full"):
    """qid -> field for one arm, over the questions that carry it."""
    key = f"{name}__{field}"
    out = {r["qid"]: r[key] for r in records if key in r}
    if not out:
        raise SystemExit(f"FAIL no record carries {key!r}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rec_p = Path(args.records)
    if not rec_p.is_file():
        raise SystemExit(f"FAIL records {rec_p} absent")
    blob = json.loads(rec_p.read_text())
    meta, records = blob["_meta"], blob["records"]
    qids = [r["qid"] for r in records]
    if len(set(qids)) != len(qids):
        raise SystemExit("FAIL duplicate qids in the records")
    image_of = {r["qid"]: r["image_id"] for r in records}
    COHORT = {"core": list(qids),
              "irrelevant_subset": [r["qid"] for r in records if r.get("in_irrelevant_subset")],
              "scale_matched": [r["qid"] for r in records if r.get("in_scale_matched")]}
    for name in ("irrelevant_subset", "scale_matched"):
        if not set(COHORT[name]).issubset(set(COHORT["core"])):
            raise SystemExit(f"FAIL {name} is not nested in core")

    print(f"=== Stage B — {LABEL} ===")
    print(f"records : {rec_p}  sha256 {sha256_file(rec_p)[:16]}…")
    print(f"n       : {len(qids)} questions on {len({*image_of.values()})} images"
          f"{'   *** SMOKE TEST ***' if meta.get('smoke_test') else ''}")
    print(f"adapter : {meta['trainable_parameters']:,} trained parameters, "
          f"{meta['adapter_config']['n_evidence']} evidence tokens, seed {meta['adapter_seed']}")
    print(f"NOT ZERO-SHOT ON GQA: {meta['not_zero_shot_on_gqa']}")

    fails = []

    def check(ok, msg):
        print(("  PASS  " if ok else "  FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    print("\n--- validity checks ---")
    check(meta.get("no_adapter_span_matches_baseline") is True,
          "no_adapter_span reproduced the frozen baseline in-job")
    check(meta.get("null_visual_gate_live") is True,
          "the null-visual arm ran with a LIVE learned gate (forcing it to zero there would "
          "conflate 'no visual information' with 'no evidence admitted')")
    check(meta.get("evidence_dropout_is_null_visual") is True,
          "training evidence dropout used the null-visual condition, so the null-visual arm is "
          "in distribution rather than a test-time novelty")
    nv = meta.get("gate_stats_by_arm", {}).get("adapter_null_visual", {})
    check(bool(nv) and nv.get("max", 0.0) > 0.0,
          f"the null-visual arm's gate was not identically zero (mean {nv.get('mean')}, "
          f"max {nv.get('max')})")
    # Three separately recorded hashes of the adapter's weights: before the first arm, sampled
    # during each arm, and after the last. All must agree, and every evidence arm must have
    # contributed one — a missing arm would mean an arm ran under weights nobody hashed.
    seen = {meta.get("adapter_param_sha256"), meta.get("adapter_param_sha256_after"),
            *meta.get("adapter_param_sha256_by_arm", {}).values()}
    armed = set(meta.get("adapter_param_sha256_by_arm", {}))
    want = {a for a in meta["arms"] if a != "baseline_frozen"} - (
        set() if COHORT["irrelevant_subset"] else {"adapter_irrelevant_crop"})
    check(len(seen) == 1 and None not in seen and armed >= want,
          f"one adapter checkpoint answered every arm: {len(seen)} distinct weight hash(es) "
          f"across {len(armed)} arms, expected 1 across {len(want)}")
    # THE OVERFIT RULE IS MODE-AWARE, because an overfit checkpoint means opposite things in the
    # two modes. In a smoke it is the run's DEFINING PROPERTY — the point is to prove the adapter
    # can drive the loss down on a handful of examples — and requiring its absence made the smoke
    # report its own contract as a defect: job 2292073 passed every plumbing check and still
    # exited non-zero. In a scientific evaluation the same property is DISQUALIFYING.
    #
    #   smoke_test=True,  overfit=True   valid plumbing smoke, not scientifically eligible, exit 0
    #   smoke_test=True,  overfit=False  a smoke that did not overfit: its contract was not met
    #   smoke_test=False, overfit=True   an overfit checkpoint offered as science: refused
    #   smoke_test=False, overfit=False  eligible
    #
    # Ineligibility is NOT a validity failure. A validity failure means the numbers cannot be
    # trusted; ineligibility means correct numbers that no scientific claim may rest on. Those are
    # different states and the exit code distinguishes them.
    is_smoke = bool(meta.get("smoke_test"))
    is_overfit = bool(meta.get("adapter_overfit_smoke"))
    non_scientific_reasons = []
    if is_smoke:
        non_scientific_reasons.append("smoke_test")
        check(is_overfit,
              "the smoke used an overfit checkpoint, as its contract requires (an EXPECTED smoke "
              "property, deliberately not counted as a validity failure)")
        if is_overfit:
            non_scientific_reasons.append("adapter_overfit_smoke")
    else:
        check(not is_overfit,
              "a scientific evaluation must not be run on an overfit-smoke checkpoint")
        if is_overfit:
            non_scientific_reasons.append("adapter_overfit_smoke")
    scientific_eligible = not is_smoke and not is_overfit
    print(f"  INFO  scientific_eligible={str(scientific_eligible).lower()}"
          + (f"  reasons={non_scientific_reasons}" if non_scientific_reasons else ""))
    check(all(r["wrong_donor_image_id"] != r["image_id"] for r in records),
          "no wrong crop comes from the question's own image")
    bad_lab = [r["qid"] for r in records if r["wrong_donor_label"] != r["target_label"]]
    check(not bad_lab, f"every wrong crop shares the question's normalised target label "
                       f"({len(bad_lab)} mismatched)")
    n_ev = meta["adapter_config"]["n_evidence"]
    check(4 <= n_ev <= 8, f"the adapter emits {n_ev} evidence tokens, inside the declared 4-8")
    span = meta.get("span_arms", [])
    check(len(span) == 5, f"five arms append evidence positions {span}")
    for a in span:
        w = {r[f"{a}__n_visual"] for r in records if f"{a}__n_visual" in r}
        check(w == {256 + n_ev}, f"{a} carries 256+{n_ev} visual tokens {w}")
    check({r["no_adapter_span__n_visual"] for r in records} == {256},
          "no_adapter_span carries exactly the baseline's 256 visual tokens")
    gs = meta.get("gate_stats_by_arm", {}).get("adapter_correct_crop", {})
    check(bool(gs.get("all_finite")), f"gate values are finite {'' if gs else '(none recorded)'}")
    check(gs.get("frac_below_0.01", 1.0) < 1.0 and gs.get("frac_above_0.99", 1.0) < 1.0,
          f"the gate is not permanently saturated (mean {gs.get('mean')}, "
          f"{gs.get('frac_below_0.01')} below 0.01, {gs.get('frac_above_0.99')} above 0.99)")

    # ---- truncation policy (C41) ----
    trunc = defaultdict(list)
    for r in records:
        for k in r:
            if k.endswith("__stop_reason") and not str(r[k]).startswith("eos_"):
                trunc[k[: -len("__stop_reason")]].append(r["qid"])
    n_gen = sum(1 for r in records for k in r if k.endswith("__stop_reason"))
    print(f"  INFO  truncation: {sum(len(v) for v in trunc.values())}/{n_gen} arm-generations")
    for a in sorted(trunc, key=lambda x: -len(trunc[x])):
        print(f"          {a:32s} {len(trunc[a]):4d}/{len(qids)}")
    treat = {a: len(v) for a, v in trunc.items() if a in TREATMENT_ARMS}
    check(not treat, "no treatment arm truncated (a truncating treatment arm is a corrupted "
                     "measurement of the very thing being claimed)"
                     + (f" — but these did: {treat}" if treat else ""))

    out = {"n": len(qids), "n_images": len({*image_of.values()}), "label": LABEL, "oracle": True,
           "smoke_test": is_smoke,
           "scientific_eligible": scientific_eligible,
           "non_scientific_reasons": non_scientific_reasons,
           "not_zero_shot_on_gqa": meta["not_zero_shot_on_gqa"],
           "trainable_parameters": meta["trainable_parameters"],
           "adapter_param_sha256": meta["adapter_param_sha256"],
           "gate_stats_by_arm": meta.get("gate_stats_by_arm", {}),
           "validity_failures": fails, "truncation": {a: len(v) for a, v in trunc.items()},
           "accuracy": {}, "contrasts": {}}

    draws, out["cohorts"] = {}, {}
    print(f"\nbootstrap: {BOOT:,} resamples, seed {SEED}")
    for name, members in COHORT.items():
        if not members:
            continue
        qidx, qsha = q_matrix(len(members))
        cdr, n_img = cluster_draws(members, image_of)
        draws[name] = {"q": [qidx[i] for i in range(BOOT)], "c": cdr}
        out["cohorts"][name] = {"n": len(members), "n_images": n_img,
                                "question_matrix_sha256": qsha}
        print(f"  {name:18s} n={len(members):4d} over {n_img:4d} images  matrix {qsha[:16]}…")

    print(f"\n  {'arm':28s} {'cohort':>18s} {'n':>5s} {'exact_full':>11s}")
    for a in meta["arms"]:
        coh = "irrelevant_subset" if a == "adapter_irrelevant_crop" else "core"
        pop = COHORT[coh]
        if not pop:
            continue
        vals = arm(records, a)
        acc = 100 * float(np.mean([vals[q] for q in pop]))
        out["accuracy"][a] = {"cohort": coh, "n": len(pop), "exact_full": acc}
        print(f"  {a:28s} {coh:>18s} {len(pop):5d} {acc:10.1f}%")

    print(f"\n  {'contrast':56s} {'points':>8s}  question-level 95% CI")
    for t_name, c_name, coh, why in CONTRASTS:
        pop = COHORT[coh]
        if not pop:
            print(f"  {t_name} - {c_name}: cohort {coh!r} is empty; skipped")
            continue
        t, c = arm(records, t_name), arm(records, c_name)
        missing = [q for q in pop if q not in t or q not in c]
        if missing:
            raise SystemExit(f"FAIL {t_name}-{c_name}: {len(missing)} questions in cohort "
                             f"{coh!r} lack an arm")
        v = np.array([t[q] for q in pop], float) - np.array([c[q] for q in pop], float)
        p, lo, hi = ci_from_draws(v, draws[coh]["q"])
        cp, clo, chi = ci_from_draws(v, draws[coh]["c"])
        ta, ca = arm(records, t_name, "ans"), arm(records, c_name, "ans")
        changed = sum(1 for q in pop if ta[q] != ca[q])
        gains = sum(1 for q in pop if t[q] and not c[q])
        losses = sum(1 for q in pop if c[q] and not t[q])
        k_t = len(trunc.get(t_name, []))
        k_c = len(trunc.get(c_name, []))
        label = f"{t_name} - {c_name} [{coh}]"
        mark = {VS_BASELINE: "  [vs BASELINE — CONFOUNDED]", VS_WRONG: "  [EVIDENCE-SPECIFIC]",
                VS_NULL: "  [EVIDENCE-SPECIFIC — DECIDER]"}.get((t_name, c_name), "") \
            if coh == "core" else ""
        star = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {label:56s} {p:+7.2f}  [{lo:+.2f}, {hi:+.2f}] {star}{mark}")
        print(f"  {'':56s}          cohort {coh} (n={len(pop)})   "
              f"image-clustered [{clo:+.2f}, {chi:+.2f}]")
        print(f"  {'':56s}          answers changed {changed}/{len(pop)}, "
              f"gains {gains}, losses {losses}")
        print(f"  {'':58s}{why}")
        if lo <= 0 <= hi:
            print(f"  {'':56s}          no detectable difference at the available precision")
        rec = {"cohort": coh, "n": len(pop), "point": p, "lo": lo, "hi": hi,
               "excludes_zero": bool(lo > 0 or hi < 0), "clustered_point": cp,
               "clustered_lo": clo, "clustered_hi": chi,
               "clustered_excludes_zero": bool(clo > 0 or chi < 0),
               "answers_changed": changed, "gains": gains, "losses": losses,
               "truncated_treatment": k_t, "truncated_control": k_c, "question": why}
        if k_t or k_c:
            w_lo, w_hi = lo - 100 * k_c / len(pop), hi + 100 * k_t / len(pop)
            wp_lo, wp_hi = p - 100 * k_c / len(pop), p + 100 * k_t / len(pop)
            still = bool(w_lo > 0 or w_hi < 0)
            flip = (p > 0 and wp_lo <= 0) or (p < 0 and wp_hi >= 0)
            rec.update({"worst_case_lo": w_lo, "worst_case_hi": w_hi, "worst_point_lo": wp_lo,
                        "worst_point_hi": wp_hi, "worst_case_excludes_zero": still,
                        "worst_case_sign_flip": bool(flip)})
            print(f"  {'':56s}          worst case point [{wp_lo:+.2f}, {wp_hi:+.2f}]  "
                  f"CI [{w_lo:+.2f}, {w_hi:+.2f}]")
            if flip:
                check(False, f"{label}: truncation could reverse the point estimate's sign")
            if rec["excludes_zero"] and not still:
                check(False, f"{label}: truncation could stop this interval excluding zero")
        out["contrasts"][label] = rec

    # the parity contrast is a check, not a finding
    par = out["contrasts"].get("no_adapter_span - baseline_frozen [core]")
    if par is not None:
        check(par["point"] == 0.0 and par["answers_changed"] == 0,
              f"no_adapter_span == frozen baseline exactly ({par['answers_changed']} differ)")

    base = out["contrasts"].get(f"{VS_BASELINE[0]} - {VS_BASELINE[1]} [core]")
    wrong = out["contrasts"].get(f"{VS_WRONG[0]} - {VS_WRONG[1]} [core]")
    null = out["contrasts"].get(f"{VS_NULL[0]} - {VS_NULL[1]} [core]")
    hurt = out["contrasts"].get(f"{WRONG_VS_BASELINE[0]} - {WRONG_VS_BASELINE[1]} [core]")

    def beats(c):
        """The contrast is positive and its interval excludes zero."""
        return bool(c and c["excludes_zero"] and c["point"] > 0)

    def fmt(c, name):
        return (f"{name} {c['point']:+.2f} [{c['lo']:+.2f}, {c['hi']:+.2f}]" if c
                else f"{name} (not run)")

    ONE_SEED = (" ONE SEED, EXPLORATORY — this label is retained regardless of any interval's "
                "width or significance, and nothing here is confirmatory.")
    if fails:
        verdict = (f"NO VERDICT — {len(fails)} validity check(s) failed; the estimands are not "
                   f"trustworthy.")
    elif not scientific_eligible:
        verdict = ("NO SCIENTIFIC VERDICT — GPU smoke for plumbing and parity only "
                   f"({', '.join(non_scientific_reasons)}).")
    elif base is None or wrong is None or null is None:
        verdict = "NO VERDICT: a required contrast did not run"
    elif beats(base) and beats(wrong) and beats(null):
        verdict = (f"PROMISING SUPERVISED EVIDENCE GAIN: correct evidence beats the frozen "
                   f"baseline, same-label wrong evidence AND null-visual evidence — "
                   f"{fmt(base, 'vs baseline')}, {fmt(wrong, 'vs wrong')}, "
                   f"{fmt(null, 'vs null-visual')} (n={base['n']}). The baseline contrast remains "
                   f"confounded by GQA supervision, new parameters and extra token positions; the "
                   f"attribution to visual evidence rests on the wrong and null-visual "
                   f"contrasts." + ONE_SEED)
    elif beats(wrong) and beats(null):
        verdict = (f"EVIDENCE-SPECIFIC BUT NO NET GAIN: correct evidence beats same-label wrong "
                   f"evidence and null-visual evidence — {fmt(wrong, 'vs wrong')}, "
                   f"{fmt(null, 'vs null-visual')} — but does not beat the frozen baseline "
                   f"({fmt(base, 'vs baseline')}). The adapter uses the crop, and the use does "
                   f"not translate into accuracy over the original stack." + ONE_SEED)
    elif beats(base) and not beats(null):
        verdict = (f"SUPERVISED/QUESTION-PRIOR GAIN: correct evidence beats the frozen baseline "
                   f"({fmt(base, 'vs baseline')}) but does NOT beat null-visual evidence "
                   f"({fmt(null, 'vs null-visual')}). The adapter reaches the same accuracy with "
                   f"all-zero visual features, so the gain is attributable to GQA supervision and "
                   f"the trainable question pathway rather than to the crop." + ONE_SEED)
    elif beats(wrong) and hurt is not None and hurt["excludes_zero"] and hurt["point"] < 0:
        verdict = (f"ROBUSTNESS ONLY: correct evidence beats wrong evidence "
                   f"({fmt(wrong, 'vs wrong')}) because WRONG evidence damages the baseline "
                   f"({fmt(hurt, 'wrong vs baseline')}), while correct evidence merely restores "
                   f"it ({fmt(base, 'correct vs baseline')}). This is robustness to harmful "
                   f"crops, not an accuracy improvement." + ONE_SEED)
    elif not beats(wrong) and not beats(null):
        verdict = (f"STOP: correct evidence beats neither same-label wrong evidence "
                   f"({fmt(wrong, '')}) nor null-visual evidence ({fmt(null, '')}) at this "
                   f"precision. That is 'not detectable', never 'no effect' and never "
                   f"equivalence." + ONE_SEED)
    else:
        # Deliberately not forced into a named class. Reporting the observed pattern is honest;
        # bending it into the nearest label would misdescribe the result.
        verdict = (f"UNCLASSIFIED PATTERN — matches none of the five pre-registered classes: "
                   f"{fmt(base, 'vs baseline')}, {fmt(wrong, 'vs wrong')}, "
                   f"{fmt(null, 'vs null-visual')}, {fmt(hurt, 'wrong vs baseline')}. Report as "
                   f"observed and do not relabel it." + ONE_SEED)
    print(f"\n  VERDICT: {verdict}")
    out["verdict"] = verdict
    if not scientific_eligible and "NO SCIENTIFIC VERDICT" not in verdict \
            and "NO VERDICT" not in verdict:
        raise SystemExit("FAIL a scientifically ineligible run produced a scientific verdict")
    out["classification_inputs"] = {
        "beats_baseline": beats(base), "beats_wrong": beats(wrong), "beats_null_visual": beats(null),
        "wrong_hurts_baseline": bool(hurt and hurt["excludes_zero"] and hurt["point"] < 0),
        "one_seed_exploratory": True}

    if args.out:
        op = Path(args.out)
        if op.exists():
            raise SystemExit(f"FAIL refusing to overwrite {op}")
        op.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {op}")

    print(f"\n{LABEL}  Development slice only. An interval including zero is 'no detectable")
    print("difference at the available precision', never a demonstration of equivalence.")
    if fails:
        print(f"\n{len(fails)} validity check(s) failed — the estimands above are not trustworthy")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
