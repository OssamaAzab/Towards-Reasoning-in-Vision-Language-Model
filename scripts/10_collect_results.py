"""Step 10: collect every saved eval into two tidy CSVs (overall + per-category).

Scans outputs/eval/*_records.json (written by scripts 07 and 09), recomputes exact and
VQA-soft for the model and its comparator, and writes:
  - results_summary.csv      one row per run: model vs comparator, overall exact + VQA-soft
  - results_per_category.csv one row per run x reasoning category: VQA-soft + delta
Base evals (script 07) compare bridge vs text-only floor; augmentation evals (script 09)
compare bridge+augmentation vs the un-augmented bridge. Both schemas are handled.

    python scripts/10_collect_results.py
"""
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import prompt as P  # noqa: E402
from src.data.gqa import REASONING_CATEGORIES  # noqa: E402
from src.utils import load_config  # noqa: E402


VARIANT_TOKENS = (
    "150k", "500k", "nocls", "cosine", "lora_full", "lora_attn", "mlp",
    "cont5", "pool", "8bit", "bf16", "fp16", "q64", "q128", "l8", "q64l8",
    "shuf", "anchor", "tuned",
)
ENCODER_TOKENS = ("clip", "ijepa", "dinov2")
AUGMENTATION_SUFFIXES = (
    "compare_topup", "scene_graph_predq_plain", "scene_graph_predq",
    "scene_graph_pred", "scene_graph", "cot",
)


def has_stem_token(stem, token):
    """Return whether an underscore-delimited stem contains the exact token."""
    return re.search(rf"(?:^|_){re.escape(token)}(?:_|$)", stem) is not None


def validate_bridge_stem(stem, base):
    """Reject unrecognised bridge-stem tokens before they can lose lineage silently."""
    prefix = f"bridge_{base}_"
    if base == "unknown" or not stem.startswith(prefix):
        raise ValueError(f"unrecognised bridge stem prefix: {stem}")

    remainder = stem[len(prefix):]
    for suffix in AUGMENTATION_SUFFIXES:
        marker = f"_{suffix}"
        if remainder.endswith(marker):
            remainder = remainder[:-len(marker)]
            break
    for variant in sorted(VARIANT_TOKENS, key=len, reverse=True):
        remainder = re.sub(
            rf"(?:^|_){re.escape(variant)}(?=_|$)", "", remainder,
        )

    structural = re.compile(r"(?:\d+k|\d+ep|ep\d+|s\d+|v\d+)")
    unknown = [token for token in remainder.split("_")
               if token and structural.fullmatch(token) is None]
    if unknown:
        raise ValueError(
            f"unrecognised bridge stem token(s) in {stem}: {', '.join(unknown)}"
        )


def load_records(path):
    """Return (records, meta) for either artifact shape.

    Legacy artifacts are a bare list. Corrected artifacts are {"_meta": ..., "records": ...},
    so indexing the loaded object as a list raises KeyError(0) — the collector used to
    crash on every corrected file.
    """
    loaded = json.load(open(path))
    if isinstance(loaded, dict) and "records" in loaded:
        return loaded["records"], loaded.get("_meta", {})
    return loaded, {}


def describe_from_meta(stem, meta):
    """Read (encoder, epochs, augmentation) from an artifact's own provenance block.

    Preferred over describe(): the condition is recorded at write time rather than
    guessed from a filename. Stem inference cannot survive protocol-aware stems —
    `...__image_plus_graph__direct` ends in neither "scene_graph" nor "cot", so a
    scene-graph run would be silently labelled the un-augmented baseline.
    """
    by_id = {mid: nm for nm, mid in load_config()["models"]["encoders"].items()}
    base = by_id.get(meta.get("checkpoint_encoder")) or next(
        (t for t in ENCODER_TOKENS if has_stem_token(stem, t)), "unknown")
    # Variant tokens still come from the checkpoint portion of the stem, so the
    # encoder column keeps doubling as the model-variant label as it does for legacy runs.
    ckpt_part = stem.split("__")[0]
    variants = [t for t in VARIANT_TOKENS if has_stem_token(ckpt_part, t)]
    encoder = "_".join([base] + variants) if base != "unknown" else "unknown"

    epochs = meta.get("checkpoint_epoch")
    if epochs is None:
        m = re.search(r"(\d+)ep|ep(\d+)", stem)
        epochs = int(m.group(1) or m.group(2)) if m else 2

    aug = meta.get("augmentation")
    if aug is None:
        reason, inp = meta.get("reason_mode", "direct"), meta.get("input_mode", "image_only")
        graph = inp in ("graph_only", "image_plus_graph")
        aug = ("scene_graph" if graph and reason == "direct" else
               f"scene_graph_{reason}" if graph else
               reason if reason != "direct" else "none")
    return encoder, epochs, aug


def describe(stem):
    """Infer (encoder, epochs, augmentation) from a records filename stem."""
    # The encoder column doubles as the MODEL-VARIANT label (established by the
    # clip_nocls [CLS]-ablation): any training-time variant baked into a run stem
    # (data scale, schedule, LoRA, connector type, LLM precision, LLaVA pool draw) is appended to the base
    # encoder name, so variant runs can never collide with the locked baselines
    # in encoder-filtered views. Most-specific tokens are matched from a fixed
    # list; plain stems fall through to the bare encoder name unchanged.
    base = next((token for token in ENCODER_TOKENS if has_stem_token(stem, token)), "unknown")
    if stem.startswith("bridge_"):
        validate_bridge_stem(stem, base)
    variants = [token for token in VARIANT_TOKENS if has_stem_token(stem, token)]
    encoder = "_".join([base] + variants) if base != "unknown" else "unknown"
    # Epoch token is either "<N>ep" (150K runs, e.g. 150k_3ep) or "ep<N>" (500K
    # per-epoch runs, e.g. 500k_ep3); the untagged 2-epoch run has neither.
    m = re.search(r"(\d+)ep|ep(\d+)", stem)
    epochs = int(m.group(1) or m.group(2)) if m else 2
    # Longer/more specific suffixes MUST be checked before shorter ones: "...scene_graph_pred"
    # is itself a suffix-match trap for "...scene_graph_predq" and "...scene_graph_predq_plain"
    # (str.endswith would match the first branch that fits, not the most specific one), and
    # bare "...scene_graph" must only match the oracle run, not any Stage-2 predicted variant.
    if stem.endswith("compare_topup"):
        # Supplementary compare-category eval SET (not an augmentation): distinct
        # label so these rows can never be read as locked-set results.
        augmentation = "compare_topup"
    elif stem.endswith("scene_graph_predq_plain"):
        augmentation = "scene_graph_predq_plain"
    elif stem.endswith("scene_graph_predq"):
        augmentation = "scene_graph_predq"
    elif stem.endswith("scene_graph_pred"):
        augmentation = "scene_graph_pred"
    elif stem.endswith("scene_graph"):
        augmentation = "scene_graph"
    elif stem.endswith("cot"):
        augmentation = "cot"
    else:
        augmentation = "none"
    return encoder, epochs, augmentation


def counts(records, ex_key, vq_key):
    """Raw exact/VQA-soft hit counts for the given boolean keys (percentages/effects derive from these)."""
    ex = sum(bool(r[ex_key]) for r in records)
    vq = sum(bool(r[vq_key]) for r in records)
    return ex, vq


def per_category(records, ex_key, vq_key):
    """category -> (n, exact_count, vqa_count) for the given keys."""
    out = {}
    for cat in REASONING_CATEGORIES + ["other"]:
        rows = [r for r in records if r["category"] == cat]
        if rows:
            ex, vq = counts(rows, ex_key, vq_key)
            out[cat] = (len(rows), ex, vq)
    return out


def main():
    """Build the two summary CSVs from all saved per-question records."""
    eval_dir = Path(load_config()["paths"]["outputs"]) / "eval"
    files = sorted(eval_dir.glob("*_records.json"))
    summary_rows, percat_rows = [], []

    for f in files:
        stem = f.name.replace("_records.json", "")
        records, meta = load_records(f)
        # Metadata wins when present: it was recorded at write time, not inferred.
        encoder, epochs, aug = (describe_from_meta(stem, meta) if meta else describe(stem))
        layer = meta.get("evidence_layer", P.EVIDENCE_LEGACY)
        n = len(records)
        if "bridge_vqa" in records[0]:                 # base eval (script 07): bridge vs floor
            model_label, comp_label = "bridge (vision)", "text-only floor"
            m_keys, c_keys = ("bridge_exact", "bridge_vqa"), ("floor_exact", "floor_vqa")
        else:                                          # augmentation eval (script 09): aug vs baseline
            model_label, comp_label = "bridge + " + aug, "bridge (baseline)"
            m_keys, c_keys = ("a_exact", "a_vqa"), ("b_exact", "b_vqa")

        m_ex, m_vq = counts(records, *m_keys)
        c_ex, c_vq = counts(records, *c_keys)
        pct = lambda c, total: round(100 * c / total, 1)
        summary_rows.append({
            # evidence_layer is first-class: legacy and corrected numbers answer
            # different questions and must never be averaged or ranked together.
            "run": stem, "evidence_layer": layer,
            "encoder": encoder, "epochs": epochs, "augmentation": aug, "n": n,
            "model": model_label, "model_exact_%": pct(m_ex, n), "model_vqa_soft_%": pct(m_vq, n),
            "comparator": comp_label, "comparator_exact_%": pct(c_ex, n),
            "comparator_vqa_soft_%": pct(c_vq, n),
            "effect_exact_%": pct(m_ex - c_ex, n), "effect_vqa_soft_%": pct(m_vq - c_vq, n),
        })

        m_cat = per_category(records, *m_keys)
        c_cat = per_category(records, *c_keys)
        for cat in REASONING_CATEGORIES + ["other"]:
            if cat in m_cat:
                cn, mex, mvq = m_cat[cat]
                _, _, cvq = c_cat[cat]
                percat_rows.append({
                    "run": stem, "evidence_layer": layer,
                    "encoder": encoder, "augmentation": aug, "category": cat, "n": cn,
                    "model_vqa_soft_%": pct(mvq, cn), "comparator_vqa_soft_%": pct(cvq, cn),
                    "delta_vqa_soft_%": pct(mvq - cvq, cn), "model_exact_%": pct(mex, cn),
                })

    with open(eval_dir / "results_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    with open(eval_dir / "results_per_category.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(percat_rows[0].keys()))
        w.writeheader()
        w.writerows(percat_rows)

    print(f"wrote {eval_dir/'results_summary.csv'} ({len(summary_rows)} runs)")
    print(f"wrote {eval_dir/'results_per_category.csv'} ({len(percat_rows)} rows)")


if __name__ == "__main__":
    main()
