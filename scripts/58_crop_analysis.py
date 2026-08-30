"""Paired analysis for Stage A crop augmentation. CPU only (no inference, arithmetic on records).

    python scripts/58_crop_analysis.py --records outputs/crop_augment/eval_<jobid>/crop_records.json

THE CONTRASTS, reported separately for each encoder, paired per question on exact_full:

  relevant - global      does adding an oracle crop help at all?   CONFOUNDED by prompt length.
  relevant - wrong       PRIMARY. specific to THIS image's region?  core, length-matched
  relevant - wrong       ROBUSTNESS. the same contrast on the pre-registered scale-matched
                         subgroup, where the two crops reach the encoder at comparable resolution
  relevant - irrelevant  SECONDARY. specific to the RELEVANT region? irrelevant_subset
  crop_only - global     is the crop alone enough?                         length-matched

A GROUNDED IMPROVEMENT REQUIRES relevant TO BEAT wrong. Beating `global` alone is not sufficient
and is not reported as success: arms carrying a crop have 512 visual tokens where the frozen
bridge and LLM only ever saw 256, so relevant-vs-global confounds crop CONTENT with prompt
LENGTH. `wrong` and `irrelevant` hold length fixed and differ only in which region is shown.

THE SCALE CONFOUND, AND WHAT THE SUBGROUP DOES ABOUT IT. The wrong crop is matched on target label
and only then area-minimised, so the two crops can differ severalfold in padded area and therefore
in the resolution at which the object reaches the encoder. GROUNDED requires the FULL-CORE primary
interval to exclude zero positively AND the scale-matched subgroup's POINT estimate to be positive
too. If the full result is positive and the subgroup is not, the verdict is SCALE-CONFOUNDED /
UNRESOLVED — not a refutation, and not a success. The subgroup's interval is deliberately NOT
required to exclude zero: it is a smaller cohort chosen for cleanliness, not for power, and
demanding significance from it would convert low power into a false negative. Nor is a subgroup
interval containing zero ever described as equivalence.

TWO BOOTSTRAPS. The question-level bootstrap resamples questions; the image-clustered bootstrap
resamples IMAGES and takes every question belonging to a sampled image, which is the correct unit
when several questions share an image and their errors are correlated. The clustered interval is
reported as a SENSITIVITY check beside the question-level one, never silently substituted.

TRUNCATION POLICY (C41), fully applied. 1) report truncation by arm; 2) fail if a TREATMENT arm
truncates, outright; 3) compute worst-case bounds on BOTH the interval and the point estimate for
every affected contrast; 4) fail if the bound could change the qualitative conclusion — a positive
point that could reach zero or below, a negative point that could reach zero or above, or a
zero-excluding interval that could stop excluding zero. Bounding only the interval is not enough:
a point estimate that can cross zero is a reversed conclusion regardless of what the interval does.

THE BIAS DIRECTION, STATED CORRECTLY. A truncated answer scores wrong, so a truncating arm is
UNDERSTATED. In a contrast A - B: truncation in A (the treatment) makes the contrast look
SMALLER, which is conservative with respect to the hypothesis; truncation in B (the control)
makes it look LARGER, which flatters the hypothesis. The bound is therefore asymmetric —
[lo - k_B/n, hi + k_A/n] — and only a zero-excluding interval is ever at risk from it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEED, BOOT = 42, 10000
# (treatment, control, cohort, question). PRIMARY contrasts run on `core`; the irrelevant
# contrast runs ONLY on the nested `irrelevant_subset`. The two populations are reported
# separately and are never differenced against each other as though paired.
CONTRASTS = [
    ("global_plus_relevant_crop", "global_only", "core",
     "does the oracle crop help at all? (CONFOUNDED by prompt length)"),
    ("global_plus_relevant_crop", "global_plus_wrong_crop", "core",
     "PRIMARY — is the gain specific to THIS image's region? (length- and target-matched)"),
    ("global_plus_relevant_crop", "global_plus_wrong_crop", "scale_matched",
     "ROBUSTNESS — the same contrast where both crops reach the encoder at comparable "
     "resolution; its POINT estimate must agree in sign, its interval is never required to "
     "exclude zero"),
    ("relevant_crop_only", "global_only", "core",
     "is the crop alone enough? (length-matched)"),
    ("global_plus_relevant_crop", "global_plus_irrelevant_crop", "irrelevant_subset",
     "SECONDARY — is the gain specific to the RELEVANT region? (length-matched, smaller cohort)"),
]
PRIMARY = ("global_plus_relevant_crop", "global_plus_wrong_crop")
PRIMARY_COHORT = "core"
ROBUSTNESS_COHORT = "scale_matched"
# The arms whose effect is being claimed. Truncation here corrupts the measurement of the
# treatment itself, so it fails outright rather than being bounded.
TREATMENT_ARMS = ("global_plus_relevant_crop", "relevant_crop_only")


def sha256_file(p: Path) -> str:
    """Hash the records so the analysis pins what it read."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def q_matrix(n, seed=SEED):
    """Question-level resample matrix and its hash, shared by every contrast."""
    idx = np.random.default_rng(seed).integers(0, n, size=(BOOT, n))
    return idx, hashlib.sha256(idx.tobytes()).hexdigest()


def cluster_draws(qids, image_of, seed=SEED):
    """Image-clustered resamples: draw IMAGES with replacement, take all their questions.

    Rows are ragged (a resample's size varies with which images were drawn), so this returns a
    list of index arrays rather than a matrix.
    """
    rng = np.random.default_rng(seed)
    imgs = sorted({image_of[q] for q in qids})
    by_img = defaultdict(list)
    for i, q in enumerate(qids):
        by_img[image_of[q]].append(i)
    out = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(imgs), size=len(imgs))
        out.append(np.concatenate([by_img[imgs[j]] for j in pick]))
    return out, len(imgs)


def paired(a, b, qids):
    """Per-question paired difference vector (treatment minus control), in points."""
    return np.array([a[q] for q in qids], float) - np.array([b[q] for q in qids], float)


def ci_from_draws(v, draws):
    """Point estimate and percentile interval for a difference vector under given resamples."""
    d = np.array([v[ix].mean() for ix in draws]) * 100
    return v.mean() * 100, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def arm(records, enc, cond, field="exact_full"):
    """qid -> field for one (encoder, condition) arm, over the questions that carry it.

    The irrelevant arm exists only on the nested subset, so this returns a partial map rather
    than failing; callers pair it against the cohort the contrast declares.
    """
    key = f"{enc}__{cond}__{field}"
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
    encoders = list(meta["encoders"])
    qids = [r["qid"] for r in records]
    if len(set(qids)) != len(qids):
        raise SystemExit("FAIL duplicate qids in the records")
    image_of = {r["qid"]: r["image_id"] for r in records}
    # Every cohort is read from a flag the RECORDS already carry, which scripts/57 copied from the
    # coverage audit. Nothing here recomputes membership, and in particular nothing here derives a
    # cohort from an accuracy field — a subgroup chosen after seeing the answers is not a subgroup.
    COHORT = {"core": list(qids),
              "irrelevant_subset": [r["qid"] for r in records if r.get("in_irrelevant_subset")],
              "scale_matched": [r["qid"] for r in records if r.get("in_scale_matched")]}
    for name in ("irrelevant_subset", "scale_matched"):
        if not set(COHORT[name]).issubset(set(COHORT["core"])):
            raise SystemExit(f"FAIL {name} is not nested in core")
    if "scale_matched" not in meta.get("cohorts", {}):
        raise SystemExit("FAIL these records predate the pre-registered scale-matched subgroup; "
                         "the robustness reading of the primary contrast cannot be computed")
    if sorted(COHORT["scale_matched"]) != sorted(meta["cohorts"]["scale_matched"]):
        raise SystemExit("FAIL the per-record scale-matched flags disagree with the cohort the "
                         "run declared; the subgroup was not fixed before the answers")

    print("=== Stage A — crop augmentation, paired analysis ===")
    print(f"records : {rec_p}  sha256 {sha256_file(rec_p)[:16]}…")
    print(f"n       : {len(qids)} questions on {len({*image_of.values()})} images"
          f"{'   *** SMOKE TEST ***' if meta.get('smoke_test') else ''}")
    print(f"encoders: {encoders}")
    print(f"ORACLE  : {meta.get('oracle')} — {meta.get('not_deployable', '')}")

    fails = []

    def check(ok, msg):
        print(("  PASS  " if ok else "  FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    print("\n--- validity checks ---")
    check(meta["baseline_parity"]["mismatches"] == 0,
          f"baseline parity: established vlm_generate == new global_only path "
          f"({meta['baseline_parity']['mismatches']} mismatches of "
          f"{meta['baseline_parity']['checked']})")
    check(all(r["wrong_donor_image_id"] != r["image_id"] for r in records),
          "no wrong crop comes from the question's own image")
    bad_lab = [r["qid"] for r in records if r["wrong_donor_label"] != r["target_label"]]
    check(not bad_lab,
          f"every wrong crop depicts the same normalised target label as the question "
          f"({len(bad_lab)} mismatched)")
    donors = [r["wrong_donor_qid"] for r in records]
    check(len(set(donors)) == len(donors), "wrong-crop donors are one-to-one")
    # The irrelevant arm must exist on exactly the nested subset — no more, no less. An arm
    # present outside its cohort would silently widen the population the secondary contrast is
    # read over.
    for enc in encoders:
        have = {r["qid"] for r in records
                if f"{enc}__global_plus_irrelevant_crop__exact_full" in r}
        want = set(COHORT["irrelevant_subset"])
        check(have == want,
              f"{enc}: the irrelevant arm exists on exactly the subset "
              f"({len(have)} arms vs {len(want)} subset members)")
    for enc in encoders:
        tk = {(r[f"{enc}__tokens_global"], r[f"{enc}__tokens_crop"]) for r in records}
        check(len(tk) == 1, f"{enc}: one global/crop token shape across all questions {tk}")

    # ---- truncation policy ----
    trunc = defaultdict(list)
    for r in records:
        for k in r:
            if k.endswith("__stop_reason") and not str(r[k]).startswith("eos_"):
                trunc[k[: -len("__stop_reason")]].append(r["qid"])
    # Not every record carries every arm — the irrelevant arm exists only on the subset — so the
    # denominator is counted, not multiplied out from one record.
    n_gen = sum(1 for r in records for k in r if k.endswith("__stop_reason"))
    print(f"  INFO  truncation: {sum(len(v) for v in trunc.values())}/{n_gen} arm-generations")
    for a in sorted(trunc, key=lambda x: -len(trunc[x])):
        print(f"          {a:42s} {len(trunc[a]):4d}/{len(qids)}")
    treat_trunc = {a: len(v) for a, v in trunc.items()
                   if any(a.endswith(t) for t in TREATMENT_ARMS)}
    check(not treat_trunc,
          "no treatment arm truncated (a truncating treatment arm is a corrupted measurement "
          "of the very thing being claimed)" + (f" — but these did: {treat_trunc}"
                                                if treat_trunc else ""))

    # ---- crop-scale diagnostics, printed before any accuracy so the confound is visible first ----
    ratios = sorted(r["scale"]["padded_area_ratio"] for r in records)
    pct = lambda f: ratios[min(len(ratios) - 1, int(f * len(ratios)))]  # noqa: E731
    asp = sorted(r["scale"]["aspect_ratio_diff"] for r in records)
    occ = [r["scale"]["relevant"]["crop_occupancy"] for r in records]
    print(f"\n--- crop scale (relevant vs wrong, padded square canvas) ---")
    print(f"  padded-area ratio  median {pct(0.5):.2f}  p75 {pct(0.75):.2f}  p90 {pct(0.90):.2f}"
          f"  max {ratios[-1]:.2f}")
    print(f"  aspect-ratio diff  median {asp[len(asp) // 2]:.2f}  max {asp[-1]:.2f}")
    print(f"  relevant crop occupancy of its padded canvas: mean {sum(occ) / len(occ):.2f}")
    print(f"  scale-matched at factor {meta.get('scale_match_factor')}: "
          f"{len(COHORT['scale_matched'])}/{len(qids)}")

    out = {"n": len(qids), "n_images": len({*image_of.values()}),
           "smoke_test": bool(meta.get("smoke_test")), "oracle": True,
           "scale_match_factor": meta.get("scale_match_factor"),
           "padded_area_ratio": {"median": pct(0.5), "p75": pct(0.75), "p90": pct(0.90),
                                 "max": ratios[-1]},
           "validity_failures": fails, "truncation": {a: len(v) for a, v in trunc.items()},
           "accuracy": {}, "contrasts": {}}

    # One resample scheme PER COHORT. A contrast is always bootstrapped on the population it is
    # defined over; reusing core's matrix on the subset would resample questions that arm does
    # not have.
    draws, out["cohorts"] = {}, {}
    print(f"\nbootstrap: {BOOT:,} resamples, seed {SEED}")
    for name, members in COHORT.items():
        if not members:
            continue
        qidx, qsha = q_matrix(len(members))
        cdr, n_img = cluster_draws(members, image_of)
        draws[name] = {"members": members, "q": [qidx[i] for i in range(BOOT)], "c": cdr,
                       "sha": qsha, "n_img": n_img}
        out["cohorts"][name] = {"n": len(members), "n_images": n_img,
                                "question_matrix_sha256": qsha}
        print(f"  {name:18s} n={len(members):4d} over {n_img:4d} images "
              f"(mean {len(members) / n_img:.2f} q/image)  matrix {qsha[:16]}…")

    for enc in encoders:
        print(f"\n================ {enc} ================")
        print(f"  {'condition':32s} {'cohort':>18s} {'n':>5s} {'exact_full':>11s}")
        out["accuracy"][enc] = {}
        for cond in meta["conditions"]:
            coh = "irrelevant_subset" if cond == "global_plus_irrelevant_crop" else "core"
            pop = COHORT[coh]
            if not pop:
                continue
            a = arm(records, enc, cond)
            acc = 100 * float(np.mean([a[q] for q in pop]))
            out["accuracy"][enc][cond] = {"cohort": coh, "n": len(pop), "exact_full": acc}
            print(f"  {cond:32s} {coh:>18s} {len(pop):5d} {acc:10.1f}%")

        out["contrasts"][enc] = {}
        print(f"\n  {'contrast':52s} {'points':>8s}  question-level 95% CI")
        for t_name, c_name, coh, why in CONTRASTS:
            pop = COHORT[coh]
            if not pop:
                print(f"  {t_name} - {c_name}: cohort {coh!r} is empty in this run; skipped")
                continue
            d = draws[coh]
            t, c = arm(records, enc, t_name), arm(records, enc, c_name)
            missing = [q for q in pop if q not in t or q not in c]
            if missing:
                raise SystemExit(f"FAIL {enc} {t_name}-{c_name}: {len(missing)} questions in "
                                 f"cohort {coh!r} lack an arm")
            v = paired(t, c, pop)
            p, lo, hi = ci_from_draws(v, d["q"])
            cp, clo, chi = ci_from_draws(v, d["c"])
            ta, ca = arm(records, enc, t_name, "ans"), arm(records, enc, c_name, "ans")
            changed = sum(1 for q in pop if ta[q] != ca[q])
            gains = sum(1 for q in pop if t[q] and not c[q])
            losses = sum(1 for q in pop if c[q] and not t[q])
            k_t, k_c = len(trunc.get(f"{enc}__{t_name}", [])), len(trunc.get(f"{enc}__{c_name}", []))
            star = "*" if (lo > 0 or hi < 0) else " "
            # The same treatment/control pair is read on two cohorts, so the cohort is part of the
            # key. Without it the robustness row would silently overwrite the primary one.
            label = f"{t_name} - {c_name} [{coh}]"
            is_primary = (t_name, c_name) == PRIMARY and coh == PRIMARY_COHORT
            is_robust = (t_name, c_name) == PRIMARY and coh == ROBUSTNESS_COHORT
            mark = " [PRIMARY]" if is_primary else (" [ROBUSTNESS]" if is_robust else "")
            print(f"  {label:52s} {p:+7.2f}  [{lo:+.2f}, {hi:+.2f}] {star}{mark}")
            print(f"  {'':52s}          cohort {coh} (n={len(pop)})")
            print(f"  {'':52s}          image-clustered [{clo:+.2f}, {chi:+.2f}]")
            print(f"  {'':52s}          answers changed {changed}/{len(pop)}, "
                  f"gains {gains}, losses {losses}")
            print(f"  {'':54s}{why}")
            if lo <= 0 <= hi:
                print(f"  {'':52s}          no detectable difference at the available precision")
            rec = {"cohort": coh, "n": len(pop), "primary": is_primary, "robustness": is_robust,
                   "point": p, "lo": lo, "hi": hi, "excludes_zero": bool(lo > 0 or hi < 0),
                   "clustered_lo": clo, "clustered_hi": chi, "clustered_point": cp,
                   "clustered_excludes_zero": bool(clo > 0 or chi < 0),
                   "answers_changed": changed, "gains": gains, "losses": losses,
                   "truncated_treatment": k_t, "truncated_control": k_c, "question": why}
            if k_t or k_c:
                # A truncating arm is understated. Control truncation can only pull the contrast
                # DOWN; treatment truncation can only push it UP. Bound both the interval and the
                # POINT estimate — an interval that still excludes zero is not sufficient if the
                # point itself could cross zero.
                w_lo = lo - 100 * k_c / len(pop)
                w_hi = hi + 100 * k_t / len(pop)
                worst_point_lo = p - 100 * k_c / len(pop)
                worst_point_hi = p + 100 * k_t / len(pop)
                still = bool(w_lo > 0 or w_hi < 0)
                sign_flip = (p > 0 and worst_point_lo <= 0) or (p < 0 and worst_point_hi >= 0)
                rec.update({"worst_case_lo": w_lo, "worst_case_hi": w_hi,
                            "worst_point_lo": worst_point_lo, "worst_point_hi": worst_point_hi,
                            "worst_case_excludes_zero": still,
                            "worst_case_sign_flip": bool(sign_flip)})
                print(f"  {'':52s}          worst case point [{worst_point_lo:+.2f}, "
                      f"{worst_point_hi:+.2f}]  CI [{w_lo:+.2f}, {w_hi:+.2f}]"
                      f"{'  — still excludes zero' if still else ''}")
                if sign_flip:
                    check(False, f"{enc} {label}: truncation could reverse the sign of the point "
                                 f"estimate ({p:+.2f} -> worst case "
                                 f"[{worst_point_lo:+.2f}, {worst_point_hi:+.2f}])")
                if rec["excludes_zero"] and not still:
                    check(False, f"{enc} {label}: truncation could stop this interval excluding "
                                 f"zero (observed [{lo:+.2f}, {hi:+.2f}], worst case "
                                 f"[{w_lo:+.2f}, {w_hi:+.2f}])")
            out["contrasts"][enc][label] = rec

        g = out["contrasts"][enc].get(f"{PRIMARY[0]} - {PRIMARY[1]} [{PRIMARY_COHORT}]")
        s = out["contrasts"][enc].get(f"{PRIMARY[0]} - {PRIMARY[1]} [{ROBUSTNESS_COHORT}]")
        if fails:
            # A run whose validity checks failed must not print a conclusion at all. The exit code
            # already signals the failure, but a printed "GROUNDED" gets quoted out of context long
            # after the exit code is forgotten.
            verdict = (f"NO VERDICT — {len(fails)} validity check(s) failed; "
                       f"the estimands are not trustworthy.")
        elif meta.get("smoke_test"):
            # A smoke test exists to prove parity and plumbing. Its numbers are diagnostics, not
            # estimands, and at n=8 an interval can span 75 points. Emitting GROUNDED or
            # NOT GROUNDED here would put a scientific verdict on a run that cannot carry one —
            # and a verdict, once printed, gets quoted. It is suppressed unconditionally, even
            # when every observed difference is positive.
            verdict = "NO SCIENTIFIC VERDICT — GPU smoke for parity and plumbing only."
        elif g is None:
            verdict = "NO VERDICT: the primary contrast did not run"
        elif s is None:
            verdict = ("NO VERDICT: the scale-matched robustness contrast did not run, so a "
                       "positive primary result cannot be separated from a resolution difference")
        elif g["excludes_zero"] and g["point"] > 0 and s["point"] <= 0:
            # The full-core result is positive and excludes zero, but on the questions where the
            # two crops reach the encoder at comparable resolution the effect does not even point
            # the same way. That is exactly the pattern a scale artefact produces, and it is
            # neither a success nor a refutation.
            verdict = (f"SCALE-CONFOUNDED / UNRESOLVED: the primary contrast is {g['point']:+.2f} "
                       f"[{g['lo']:+.2f}, {g['hi']:+.2f}] on core (n={g['n']}), but on the "
                       f"pre-registered scale-matched subgroup (n={s['n']}) it is {s['point']:+.2f}"
                       f" [{s['lo']:+.2f}, {s['hi']:+.2f}]. Where the relevant and wrong crops "
                       f"reach the encoder at comparable resolution the advantage does not hold "
                       f"its sign, so the full-cohort result cannot be separated from a difference "
                       f"in crop scale. This is not a refutation of the effect.")
        elif g["excludes_zero"] and g["point"] > 0:
            verdict = (f"GROUNDED: the relevant crop beats the same-target, different-image wrong "
                       f"crop by {g['point']:+.2f} points [{g['lo']:+.2f}, {g['hi']:+.2f}] "
                       f"on the core cohort (n={g['n']}); the pre-registered scale-matched "
                       f"subgroup agrees in sign at {s['point']:+.2f} "
                       f"[{s['lo']:+.2f}, {s['hi']:+.2f}] (n={s['n']}, interval not required to "
                       f"exclude zero)")
        else:
            verdict = ("NOT GROUNDED: no detectable advantage over the same-target wrong crop. "
                       "At this precision this is 'not detectable', never 'no effect'.")
        print(f"\n  VERDICT ({enc}): {verdict}")
        out["contrasts"][enc]["_verdict"] = verdict

    if args.out:
        op = Path(args.out)
        if op.exists():
            raise SystemExit(f"FAIL refusing to overwrite {op}")
        op.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {op}")

    print("\nORACLE CEILING, DEVELOPMENT SLICE ONLY. An interval including zero is 'no detectable")
    print("difference at the available precision', never a demonstration of equivalence. That")
    print("applies to the scale-matched subgroup too: it is read for the SIGN of its point")
    print("estimate, and a subgroup interval spanning zero says nothing about equivalence.")
    if fails:
        print(f"\n{len(fails)} validity check(s) failed — the estimands above are not trustworthy")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
