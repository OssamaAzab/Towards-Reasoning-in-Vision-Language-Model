"""Paired analysis for the six-condition visual-information audit. CPU only.

    python scripts/54_visual_audit_analysis.py --records outputs/visual_audit/<run>/visual_audit_records.json

THE CONTRASTS, AND THE QUESTION EACH ANSWERS. All paired per question on `exact_full`, all on one
shared bootstrap matrix so the intervals are comparable within a connector.

  correct - no_visual        does the visual pathway contribute anything at all?
  correct - wrong_image      is that contribution specific to THIS image, or generic conditioning?
  correct - shuffled         patch-token ORDER permutation. Measures sensitivity to the ordering
                             of already position-encoded patch tokens — NOT destruction of spatial
                             information. CLIP adds positional embeddings before its transformer,
                             so position survives in each feature's values; a Q-Former's cross-
                             attention is permutation-invariant, making this near a no-op there.
  mask_relevant - mask_irrelevant   does hiding the QUESTION'S OWN region cost more than hiding a
                             size-matched region it never asked about? This is the sharpest test:
                             both arms hide a comparable amount of picture, so a difference cannot
                             be "an occluder appeared".
  correct - mask_relevant    what removing the relevant region costs outright
  correct - mask_irrelevant  the same for the control region

WHERE THE INFORMATION IS LOST. Read with the probes:
  * correct ~ no_visual                     -> the pathway contributes nothing behaviourally
  * correct > no_visual but ~ wrong_image   -> generic conditioning, not image-specific evidence
  * decodable after the connector but no behavioural effect -> present but IGNORED by Qwen2
  * decodable before but not after          -> lost BY THE CONNECTOR
  * not decodable before                    -> absent from the ENCODER

A NULL IS NOT EQUIVALENCE. Every interval that includes zero is reported as "no detectable
difference at the available precision", with its half-width printed beside it.
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
CONTRASTS = [
    ("correct", "no_visual", "does vision contribute at all?"),
    ("correct", "wrong_image", "is the contribution image-specific?"),
    ("correct", "shuffled", "sensitivity to patch-token ORDER (not spatial info)"),
    ("mask_relevant", "mask_irrelevant", "does the question's own region matter more?"),
    ("correct", "mask_relevant", "cost of hiding the relevant region"),
    ("correct", "mask_irrelevant", "cost of hiding a matched control region"),
]


def sha256_file(p: Path) -> str:
    """Hash the records so the analysis pins what it read."""
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


def armkey(connector, condition):
    """The record-key stem for one arm. no_visual is shared and carries no connector."""
    return "no_visual" if condition == "no_visual" else f"{connector}__{condition}"


def arm(records, connector, condition):
    """qid -> exact_full for one (connector, condition). no_visual has no connector."""
    key = "no_visual__exact_full" if condition == "no_visual" \
        else f"{connector}__{condition}__exact_full"
    if key not in records[0]:
        raise SystemExit(f"FAIL records lack {key!r}")
    return {r["qid"]: bool(r[key]) for r in records}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-category-n", type=int, default=20,
                    help="categories smaller than this are printed as descriptive only")
    args = ap.parse_args()

    rec_p = Path(args.records)
    if not rec_p.is_file():
        raise SystemExit(f"FAIL records {rec_p} absent")
    blob = json.loads(rec_p.read_text())
    meta, records = blob["_meta"], blob["records"]
    connectors = list(meta["connectors"])
    qids = sorted(r["qid"] for r in records)
    if len(set(qids)) != len(qids):
        raise SystemExit("FAIL duplicate qids in the records")

    print("=== visual-information audit — paired analysis ===")
    print(f"records   : {rec_p}  sha256 {sha256_file(rec_p)[:16]}…")
    print(f"n         : {len(qids)}{'   *** SMOKE TEST ***' if meta.get('smoke_test') else ''}")
    print(f"protocol  : {meta['prompt_format']} eos={meta['supervise_eos']} "
          f"layer={meta['evidence_layer']} max_new={meta['max_new_tokens']}")
    print(f"connectors: {connectors}")

    # ---- harness validity, before any estimand ----
    print("\n--- validity checks ---")
    fails = []
    out: dict = {"n": len(qids), "smoke_test": bool(meta.get("smoke_test")),
                 "validity_failures": fails, "accuracy": {}, "contrasts": {}, "by_category": {}}

    def check(ok, msg):
        print(("  PASS  " if ok else "  FAIL  ") + msg)
        if not ok:
            fails.append(msg)

    # no_visual uses no bridge, so scripts/51 computes it ONCE and writes a single shared key.
    # Verifying that by comparing arm(c0,"no_visual") to arm(c1,"no_visual") would be a tautology:
    # arm() ignores its connector argument for this condition, so it would compare a value to
    # itself and could never fail. The real check is on the SCHEMA — that the shared key exists
    # and that no per-connector no_visual key was written alongside it.
    stray_nv = [f"{c}__no_visual__exact_full" for c in connectors
                if f"{c}__no_visual__exact_full" in records[0]]
    check("no_visual__exact_full" in records[0] and not stray_nv,
          "no_visual is ONE shared arm (it uses no bridge), not one per connector"
          + (f" — found per-connector keys {stray_nv}" if stray_nv else ""))

    # A condition that never changes the ANSWER TEXT means the intervention did not reach the
    # model. This must compare the generated strings, not exact_full: an intervention that turns
    # "coat" into "jacket" — both wrong — changes the answer but not the correctness flag, and
    # scoring it on exact_full would report 0 and wrongly read as "the intervention never
    # arrived". Both counts are printed because they answer different questions: text changes say
    # the visual pathway is live, correctness flips say it mattered to the metric.
    def ans_arm(connector, condition):
        key = "no_visual__ans" if condition == "no_visual" else f"{connector}__{condition}__ans"
        if key not in records[0]:
            raise SystemExit(f"FAIL records lack {key!r}")
        return {r["qid"]: r[key] for r in records}

    for c in connectors:
        base_ans, base_ok = ans_arm(c, "correct"), arm(records, c, "correct")
        for cond in ("wrong_image", "shuffled", "mask_relevant", "mask_irrelevant"):
            a_ans, a_ok = ans_arm(c, cond), arm(records, c, cond)
            d_txt = sum(1 for q in qids if a_ans[q] != base_ans[q])
            d_ok = sum(1 for q in qids if a_ok[q] != base_ok[q])
            print(f"  INFO  {c}/{cond} changes the ANSWER on {d_txt}/{len(qids)} "
                  f"(0 would mean the intervention never reached the model); "
                  f"changes CORRECTNESS on {d_ok}/{len(qids)}")

    # Every connector must read the image at all: if a connector's answers equal the no-bridge
    # text-only answers on every question, its visual pathway is inert and no contrast below
    # means anything for it.
    nv_ans = ans_arm(connectors[0], "no_visual")
    for c in connectors:
        same = sum(1 for q in qids if ans_arm(c, "correct")[q] == nv_ans[q])
        check(same < len(qids),
              f"{c} is not answering text-only ({same}/{len(qids)} answers identical to no_visual)")

    # ---- truncation policy (replaces the unconditional fatal EOS rule) ----
    #
    # PROVENANCE OF THIS POLICY. The original rule failed the run on ANY non-EOS stop. Job 2291794
    # tripped it on 6 of 4,288 arm-generations, all six in the single shared `no_visual` control,
    # with zero truncations in every image-bearing arm — truncation that could not change any
    # conclusion. This replacement is more discriminating, and it originates from C41. It does NOT
    # retroactively make job 2291794's launcher exit zero — that run failed, was adjudicated by a
    # human, and is recorded as failed.
    #
    # THE BIAS DIRECTION (corrected; the original C41.6 text stated this backwards). A truncated
    # answer scores WRONG on exact_full, so a truncating arm is UNDERSTATED: its true accuracy is
    # at most observed + truncations/n. In a contrast `treatment - control`:
    #   * truncation in the TREATMENT (the minuend) UNDERSTATES the contrast — conservative, it
    #     works against the hypothesis;
    #   * truncation in the subtracted CONTROL INFLATES the contrast — it can FAVOUR the
    #     hypothesis, and is the direction that actually occurred in job 2291794.
    # The bound below is asymmetric for exactly this reason: [lo - k_control/n, hi + k_treatment/n].
    #
    # Failing outright on treatment-arm truncation is still correct, on a different ground: it
    # corrupts the measurement of the very thing being claimed, and an arm can be the treatment in
    # one contrast and the control in another.
    trunc = defaultdict(list)
    for r in records:
        for k in r:
            if k.endswith("__stop_reason") and not str(r[k]).startswith("eos_"):
                trunc[k[: -len("__stop_reason")]].append(r["qid"])
    n_arms = sum(1 for k in records[0] if k.endswith("__stop_reason"))
    n_trunc = sum(len(v) for v in trunc.values())
    print(f"  INFO  truncation: {n_trunc}/{len(qids) * n_arms} arm-generations "
          f"({'none' if not trunc else 'by arm below'})")
    for a in sorted(trunc, key=lambda x: -len(trunc[x])):
        print(f"          {a:34s} {len(trunc[a]):4d}/{len(qids)}  e.g. {trunc[a][:3]}")

    # (2) A treatment or intervention arm truncating is fatal on its own: those arms carry the
    # image, and understating them biases their contrasts in the direction of the hypothesis.
    treatment_trunc = {a: len(v) for a, v in trunc.items() if not a.startswith("no_visual")}
    check(not treatment_trunc,
          "no image-bearing (treatment/intervention) arm truncated"
          + (f" — but these did: {treatment_trunc}" if treatment_trunc else ""))
    out["truncation"] = {"total": n_trunc, "arm_generations": len(qids) * n_arms,
                         "by_arm": {a: len(v) for a, v in trunc.items()},
                         "qids_by_arm": {a: v for a, v in trunc.items()},
                         "treatment_arms_truncating": treatment_trunc}
    check(all(r["wrong_image_donor_image"] != r["image_id"] for r in records),
          "no wrong-image donor is the question's own image")
    donors = [r["wrong_image_donor_qid"] for r in records]
    check(len(set(donors)) == len(donors), "wrong-image donors are one-to-one")

    if fails:
        print(f"\n{len(fails)} validity check(s) failed — the estimands below are not trustworthy")

    idx, sha = matrix(len(qids))
    print(f"\nbootstrap : {BOOT:,} resamples, seed {SEED}, shared matrix {sha[:16]}…")
    out["matrix_sha256"] = sha

    print(f"\n=== accuracy (exact_full) ===")
    conds = ["correct", "wrong_image", "no_visual", "shuffled", "mask_relevant", "mask_irrelevant"]
    print(f"  {'connector':11s}" + "".join(f"{c:>17s}" for c in conds))
    for c in connectors:
        row = {}
        for cond in conds:
            a = arm(records, c, cond)
            row[cond] = 100 * float(np.mean([a[q] for q in qids]))
        out["accuracy"][c] = row
        print(f"  {c:11s}" + "".join(f"{row[x]:16.1f}%" for x in conds))

    print(f"\n=== paired contrasts (points, 95% CI) ===")
    for c in connectors:
        print(f"\n--- {c} ---")
        out["contrasts"][c] = {}
        for a_name, b_name, why in CONTRASTS:
            p, lo, hi = contrast(arm(records, c, a_name), arm(records, c, b_name), qids, idx)
            star = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {a_name:15s} - {b_name:16s} {p:+7.2f} [{lo:+.2f}, {hi:+.2f}] {star} "
                  f"hw={(hi - lo) / 2:5.2f}   {why}")
            if lo <= 0 <= hi:
                print(f"  {'':34s}no detectable difference at the available precision")
            rec = {"point": p, "lo": lo, "hi": hi, "half_width": (hi - lo) / 2,
                   "excludes_zero": bool(lo > 0 or hi < 0), "question": why}

            # (3) Worst-case truncation bounds, for any contrast whose arms truncated. A truncated
            # answer scores wrong, so a truncating arm's TRUE accuracy is at most observed + k/n.
            # A truncating in A can only raise the contrast; B truncating can only lower it. The
            # interval therefore only widens, so a null stays a null and only a zero-excluding
            # result is at risk.
            k_a = len(trunc.get(armkey(c, a_name), []))
            k_b = len(trunc.get(armkey(c, b_name), []))
            if k_a or k_b:
                w_lo, w_hi = lo - 100 * k_b / len(qids), hi + 100 * k_a / len(qids)
                w_p = p - 100 * k_b / len(qids) if p > 0 else p + 100 * k_a / len(qids)
                still = bool(w_lo > 0 or w_hi < 0)
                rec.update({"truncated_a": k_a, "truncated_b": k_b,
                            "worst_case_point": w_p, "worst_case_lo": w_lo, "worst_case_hi": w_hi,
                            "worst_case_excludes_zero": still})
                print(f"  {'':34s}worst case if all truncations were correct: "
                      f"{w_p:+.2f} [{w_lo:+.2f}, {w_hi:+.2f}]"
                      f"{'  — still excludes zero' if still else ''}")
                # (4) Fail only when the bound could change the qualitative conclusion.
                if rec["excludes_zero"] and not still:
                    check(False, f"{c} {a_name}-{b_name}: truncation could overturn this "
                                 f"conclusion (observed [{lo:+.2f}, {hi:+.2f}] excludes zero, "
                                 f"worst case [{w_lo:+.2f}, {w_hi:+.2f}] does not)")
            out["contrasts"][c][f"{a_name}-{b_name}"] = rec

    # ---- by GQA category, descriptive where n is small ----
    by_cat = defaultdict(list)
    for r in records:
        by_cat[r.get("structural") or "?"].append(r["qid"])
    print(f"\n=== by GQA structural category ===")
    print("  Per-category intervals use their OWN seeded matrix at that n; they are secondary")
    print("  and uncorrected for multiplicity.")
    for cat, cq in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        cq = sorted(cq)
        tag = "" if len(cq) >= args.min_category_n else "  [DESCRIPTIVE ONLY — n too small]"
        print(f"\n  {cat} (n={len(cq)}){tag}")
        cidx, csha = matrix(len(cq), seed=SEED)
        out["by_category"][cat] = {"n": len(cq), "matrix_sha256": csha,
                                   "descriptive_only": len(cq) < args.min_category_n,
                                   "contrasts": {}}
        for c in connectors:
            for a_name, b_name, _ in CONTRASTS[:4]:
                p, lo, hi = contrast(arm(records, c, a_name), arm(records, c, b_name), cq, cidx)
                print(f"    {c:11s} {a_name:15s} - {b_name:16s} {p:+7.2f} [{lo:+.2f}, {hi:+.2f}]")
                out["by_category"][cat]["contrasts"][f"{c}|{a_name}-{b_name}"] = {
                    "point": p, "lo": lo, "hi": hi}

    if args.out:
        op = Path(args.out)
        if op.exists():
            raise SystemExit(f"FAIL refusing to overwrite {op}")
        op.write_text(json.dumps(out, indent=1) + "\n")
        print(f"\nwrote {op}")

    print("\nEXPLORATORY. Development slice only. An interval including zero is 'no detectable")
    print("difference at the available precision', never a demonstration of equivalence.")
    if fails:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
