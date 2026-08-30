"""Step 12: bootstrap 95% confidence intervals for every saved eval run.

For each outputs/eval/*_records.json (base runs: bridge vs text-only floor; augmentation
runs: bridge+aug vs un-augmented bridge), resample the per-question records with
replacement (paired: the same resampled questions score both sides, so effect CIs account
for the correlation between the two systems on the same questions). Reports percentile
CIs for exact and VQA-soft, overall and per reasoning category, and whether each effect's
CI excludes zero.

    python scripts/12_bootstrap_ci.py                # writes outputs/eval/results_ci.csv
"""
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import REASONING_CATEGORIES  # noqa: E402
from src.utils import load_config  # noqa: E402

# Reuse the run-labelling convention from step 10 (single source of truth).
_spec = importlib.util.spec_from_file_location(
    "collect_results", Path(__file__).resolve().parent / "10_collect_results.py")
_collect = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_collect)
describe = _collect.describe

N_RESAMPLES = 10_000
SEED = 42


def bootstrap_ci(model, comp, rng):
    """Paired percentile CIs: (model %, comp %, effect %) each as (point, lo, hi)."""
    n = model.shape[0]
    idx = rng.integers(0, n, size=(N_RESAMPLES, n), dtype=np.int32)
    m = model[idx].mean(axis=1) * 100          # model % on each resample
    c = comp[idx].mean(axis=1) * 100           # comparator % on the SAME resample
    # Point estimates from raw counts (single division) so they match the
    # results_summary.csv convention exactly (same rounding path as step 10).
    pct = lambda count: round(100 * count / n, 1)
    m_count, c_count = int(model.sum()), int(comp.sum())
    out = []
    for series, point in ((m, pct(m_count)), (c, pct(c_count)),
                          (m - c, pct(m_count - c_count))):
        lo, hi = np.percentile(series, [2.5, 97.5])
        out.append((point, round(lo, 1), round(hi, 1)))
    return out


# RQ2 encoder-gap pseudo-runs, paired on the shared locked qids (both base runs
# answer the identical 2,000 questions). Each entry:
# (stem, a_stem, b_stem, epochs, a_label, b_label, encoder_label).
# Pairs whose records don't exist yet are skipped with a note, so this script
# stays runnable before a planned run has landed.
GAP_PAIRS = [
    ("rq2_gap_150k_3ep", "bridge_clip_150k_3ep", "bridge_ijepa_150k_3ep", 3,
     "bridge CLIP", "bridge I-JEPA", "clip-vs-ijepa"),
    ("rq2_gap_500k_ep3", "bridge_clip_500k_ep3", "bridge_ijepa_500k_ep3", 3,
     "bridge CLIP", "bridge I-JEPA", "clip-vs-ijepa"),
    # [CLS] ablation contrasts: (a) does stripping [CLS] cost
    # CLIP, and (b) does the encoder gap survive with both bridges on 256 patch tokens?
    ("rq2_cls_ablation_150k", "bridge_clip_150k_3ep", "bridge_clip_150k_nocls_ep3", 3,
     "bridge CLIP (257 tok)", "bridge CLIP-noCLS (256 tok)", "clip-vs-clip_nocls"),
    ("rq2_gap_150k_nocls", "bridge_clip_150k_nocls_ep3", "bridge_ijepa_150k_3ep", 3,
     "bridge CLIP-noCLS", "bridge I-JEPA", "clip_nocls-vs-ijepa"),
    # A1 LR-schedule ablation: cosine+warmup vs the locked flat 1e-4, same encoder,
    # paired on the shared locked qids.
    ("lr_cosine_vs_flat_150k_clip", "bridge_clip_150k_cosine_ep3", "bridge_clip_150k_3ep", 3,
     "bridge CLIP (cosine lr)", "bridge CLIP (flat lr, locked)", "clip_cosine-vs-clip"),
    ("lr_cosine_vs_flat_150k_ijepa", "bridge_ijepa_150k_cosine_ep3", "bridge_ijepa_150k_3ep", 3,
     "bridge I-JEPA (cosine lr)", "bridge I-JEPA (flat lr, locked)", "ijepa_cosine-vs-ijepa"),
    # A2 LoRA DIAGNOSTIC (approval-gated; records exist only after sign-off + run).
    # Never a headline RQ1/RQ2 row — the encoder label keeps it filtered out.
    ("lora_attn_vs_base_150k_clip", "bridge_clip_150k_3ep_lora_attn", "bridge_clip_150k_3ep", 3,
     "bridge CLIP + LoRA-attn (diagnostic)", "bridge CLIP (frozen LLM)", "clip_lora_attn-vs-clip"),
    # A3 compare top-up (approval-gated): encoder gap on the SUPPLEMENTARY compare-only
    # set — reported standalone, never alongside locked-set rows.
    ("rq2_gap_compare_topup", "bridge_clip_150k_3ep_compare_topup",
     "bridge_ijepa_150k_3ep_compare_topup", 3,
     "bridge CLIP", "bridge I-JEPA", "clip-vs-ijepa@compare_topup"),
    # B2 connector ablation: each alternative connector vs the LIKE-FOR-LIKE Q-Former
    # baseline, paired on the shared locked qids. The connectors were trained with
    # --strip-cls (256 patch tokens), so the correct baseline is the 256-token noCLS
    # Q-Former (bridge_clip_150k_nocls_ep3), NOT the 257-token locked run — §8.24 shows
    # the two Q-Former variants are equal within noise, but the comparison is made
    # like-for-like on 256 patch input. Token counts in the labels are the connectors'
    # deterministic output sizes, not measurements.
    ("connector_mlp_vs_qformer_150k", "bridge_clip_150k_mlp_ep3", "bridge_clip_150k_nocls_ep3", 3,
     "bridge CLIP (MLP, 256 vision tokens)", "bridge CLIP (Q-Former noCLS, 32 vision tokens)",
     "clip_mlp-vs-clip_nocls"),
    ("connector_pool_vs_qformer_150k", "bridge_clip_150k_pool_ep3", "bridge_clip_150k_nocls_ep3", 3,
     "bridge CLIP (pool, 32 vision tokens)", "bridge CLIP (Q-Former noCLS, 32 vision tokens)",
     "clip_pool-vs-clip_nocls"),
    # B2b I-JEPA connector ablation (cluster + approval gated — pre-registered,
    # skipped by name until the records land). I-JEPA is natively 256 patch tokens with NO
    # [CLS], so the like-for-like baseline is the native I-JEPA Q-Former (bridge_ijepa_150k_3ep,
    # drop_cls=False) — NOT a noCLS variant, NOT the CLIP baseline. No --strip-cls at train time.
    ("connector_mlp_vs_qformer_150k_ijepa", "bridge_ijepa_150k_mlp_ep3", "bridge_ijepa_150k_3ep", 3,
     "bridge I-JEPA (MLP, 256 vision tokens)", "bridge I-JEPA (Q-Former, 32 vision tokens)",
     "ijepa_mlp-vs-ijepa"),
    ("connector_pool_vs_qformer_150k_ijepa", "bridge_ijepa_150k_pool_ep3", "bridge_ijepa_150k_3ep", 3,
     "bridge I-JEPA (pool, 32 vision tokens)", "bridge I-JEPA (Q-Former, 32 vision tokens)",
     "ijepa_pool-vs-ijepa"),
    # B2b RQ2 gap through a MATCHED connector: does the CLIP-I-JEPA gap (measured through
    # the Q-Former, +6.5 noCLS) survive when both encoders read the SAME stronger connector?
    # Like-for-like: both see 256 patch tokens (CLIP via --strip-cls, I-JEPA natively) for MLP,
    # 32 pooled tokens for pool. Paired on the shared locked qids. Answers whether part of the
    # Q-Former-measured gap was a connector x encoder interaction rather than the encoder itself.
    ("rq2_gap_150k_mlp", "bridge_clip_150k_mlp_ep3", "bridge_ijepa_150k_mlp_ep3", 3,
     "bridge CLIP (MLP, 256 vision tokens)", "bridge I-JEPA (MLP, 256 vision tokens)",
     "clip-vs-ijepa@mlp"),
    ("rq2_gap_150k_pool", "bridge_clip_150k_pool_ep3", "bridge_ijepa_150k_pool_ep3", 3,
     "bridge CLIP (pool, 32 vision tokens)", "bridge I-JEPA (pool, 32 vision tokens)",
     "clip-vs-ijepa@pool"),
    # §9.11 LLaVA task-TYPE control: the locked 150K run draws its 45K LLaVA slice
    # file-order-first-n, which is 99.15% conversation; --shuffle-pool draws a
    # representative task-type sample at the SAME 150K budget, same seed, same
    # everything else. So (shuffled - locked) isolates task TYPE from the data-AMOUNT
    # confound in the 150K->500K comparison. Paired on the shared locked qids.
    # Read on VQA-soft: shuffling lengthens LLaVA answers, which depresses exact-match
    # for format reasons only (D1, §9.11). Single seed — a null reads as "consistent
    # with no large type effect", not proof of null.
    ("shuffle_pool_vs_locked_150k_ijepa", "bridge_ijepa_150k_shuf_ep3",
     "bridge_ijepa_150k_3ep", 3,
     "bridge I-JEPA (representative LLaVA draw)",
     "bridge I-JEPA (locked file-order draw)", "ijepa_shuf-vs-ijepa"),
]


def gap_records(eval_dir, a_stem, b_stem):
    """Join two base runs on qid into a/b-style records (a=first stem, b=second)."""
    a = {r["qid"]: r for r in json.load(open(eval_dir / f"{a_stem}_records.json"))}
    b = {r["qid"]: r for r in json.load(open(eval_dir / f"{b_stem}_records.json"))}
    assert a.keys() == b.keys(), "runs are not on the same locked qid set"
    return [{"category": ra["category"],
             "a_exact": ra["bridge_exact"], "a_vqa": ra["bridge_vqa"],
             "b_exact": b[q]["bridge_exact"], "b_vqa": b[q]["bridge_vqa"]}
            for q, ra in a.items()]


def main():
    """Bootstrap every run x scope x metric; write one tidy CSV."""
    eval_dir = Path(load_config()["paths"]["outputs"]) / "eval"
    rows = []

    jobs = [(f.name.replace("_records.json", ""), json.load(open(f)), None)
            for f in sorted(eval_dir.glob("*_records.json"))]
    for stem, a, b, ep, a_lbl, b_lbl, enc_lbl in GAP_PAIRS:
        if all((eval_dir / f"{s}_records.json").exists() for s in (a, b)):
            jobs.append((stem, gap_records(eval_dir, a, b), (a_lbl, b_lbl, enc_lbl, ep)))
        else:
            print(f"{stem}: skipped (records not on disk yet)")

    for stem, records, gap in jobs:
        if gap is not None:                        # encoder-gap pseudo-run (paired on qid)
            model_label, comp_label, encoder, epochs = gap
            aug = "none"
            keys = {"exact": ("a_exact", "b_exact"), "vqa_soft": ("a_vqa", "b_vqa")}
        elif "bridge_vqa" in records[0]:           # base eval: bridge vs text-only floor
            encoder, epochs, aug = describe(stem)
            model_label, comp_label = "bridge (vision)", "text-only floor"
            keys = {"exact": ("bridge_exact", "floor_exact"), "vqa_soft": ("bridge_vqa", "floor_vqa")}
        else:                                      # augmentation eval: aug vs baseline
            encoder, epochs, aug = describe(stem)
            model_label, comp_label = "bridge + " + aug, "bridge (baseline)"
            keys = {"exact": ("a_exact", "b_exact"), "vqa_soft": ("a_vqa", "b_vqa")}

        scopes = [("overall", records)]
        scopes += [(cat, [r for r in records if r["category"] == cat])
                   for cat in REASONING_CATEGORIES + ["other"]]
        for scope, recs in scopes:
            if not recs:
                continue
            rng = np.random.default_rng(SEED)      # fresh seeded stream per scope; within a
                                                   # metric, model/comparator share resamples (paired)
            arrays = {k: (np.array([bool(r[mk]) for r in recs]),
                          np.array([bool(r[ck]) for r in recs]))
                      for k, (mk, ck) in keys.items()}
            for metric, (model, comp) in arrays.items():
                (mp, mlo, mhi), (cp, clo, chi), (ep, elo, ehi) = bootstrap_ci(model, comp, rng)
                rows.append({
                    "run": stem, "encoder": encoder, "epochs": epochs, "augmentation": aug,
                    "scope": scope, "n": len(recs), "metric": metric,
                    "model": model_label, "model_%": mp, "model_ci_lo": mlo, "model_ci_hi": mhi,
                    "comparator": comp_label, "comparator_%": cp,
                    "comparator_ci_lo": clo, "comparator_ci_hi": chi,
                    "effect_%": ep, "effect_ci_lo": elo, "effect_ci_hi": ehi,
                    "effect_excludes_zero": elo > 0 or ehi < 0,
                })
        print(f"{stem}: done")

    out = eval_dir / "results_ci.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows, {N_RESAMPLES:,} resamples, seed {SEED})")


if __name__ == "__main__":
    main()
