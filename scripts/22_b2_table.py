"""Step 22: the B2 connector-comparison table — accuracy AND vision-token count.

Builds the two-axis B2 result: for each connector (Q-Former baseline, MLP projector,
average-pool compressor), VQA-soft on the locked 2,000-question set (overall with
95% CI, plus per reasoning category) next to the number of VISION TOKENS the
connector hands the LLM.

The vision-token count is a DETERMINISTIC property of each connector's output shape,
derived from the run's checkpointed bridge_config — nothing here is timed, benchmarked
or measured at runtime, and no latency/throughput/FLOPs claim is made anywhere.
Accuracy rows are read straight from results_ci.csv and echoed verbatim first, so the
table is verifiable against its raw source in the same output.

Graceful before the cluster runs land: missing stems are listed as "pending" and the
script exits 0, so it can sit at the end of the B2 chain unconditionally.

    python scripts/22_b2_table.py    # prints + writes outputs/eval/b2_connector_table.md
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils import load_config  # noqa: E402

# Connector rows: (encoder, connector label, run stem, checkpoint filename).
# CLIP block (B2, §8.26): the Q-Former baseline is the LIKE-FOR-LIKE 256-token noCLS
# variant (bridge_clip_150k_nocls_ep3, 46.0), not the 257-token locked run
# (bridge_clip_150k_3ep, 46.4): the MLP/pool connectors were trained with --strip-cls on
# 256 patch tokens, so the comparison is made at a matched 256-token input. §8.24 shows the
# two Q-Former variants are equal within noise (+0.3 [-1.7,+2.4]); the noCLS one is used
# here so no connector row is confounded by the [CLS] term.
# I-JEPA block (B2b — cluster + approval gated, pending until run): I-JEPA is
# natively 256 patch tokens with NO [CLS], so its like-for-like baseline is the native
# I-JEPA Q-Former (bridge_ijepa_150k_3ep, drop_cls=False) — no noCLS variant needed and no
# --strip-cls at train time. Rows sit as "pending" until the checkpoints land.
RUNS = [
    ("CLIP",   "Q-Former (baseline, noCLS)", "bridge_clip_150k_nocls_ep3", "bridge_clip_150k_nocls_ep3.pt"),
    ("CLIP",   "MLP projector", "bridge_clip_150k_mlp_ep3", "bridge_clip_150k_mlp_ep3.pt"),
    ("CLIP",   "Average-pool", "bridge_clip_150k_pool_ep3", "bridge_clip_150k_pool_ep3.pt"),
    ("I-JEPA", "Q-Former (baseline)", "bridge_ijepa_150k_3ep", "bridge_ijepa_150k_3ep.pt"),
    ("I-JEPA", "MLP projector", "bridge_ijepa_150k_mlp_ep3", "bridge_ijepa_150k_mlp_ep3.pt"),
    ("I-JEPA", "Average-pool", "bridge_ijepa_150k_pool_ep3", "bridge_ijepa_150k_pool_ep3.pt"),
]
CATEGORIES = ["relate", "compare", "exist", "choose", "other"]
# Raw token counts emitted by each frozen encoder:
# CLIP ViT-L/14 = 256 patches + 1 [CLS]; I-JEPA ViT-H/14 = 256 patches, no [CLS].
RAW_TOKENS = {"clip": 257, "ijepa": 256}


def vision_tokens(ckpt_path: Path, encoder: str) -> int | None:
    """Vision tokens the connector outputs, derived from its checkpointed config.

    encoder ("clip" | "ijepa") selects the raw token count for the MLP branch, which
    passes every (remaining) encoder token through: CLIP=257−[CLS], I-JEPA=256 (no [CLS]).
    """
    if not ckpt_path.exists():
        return None
    import torch
    bcfg = torch.load(ckpt_path, map_location="cpu", weights_only=False)["bridge_config"]
    btype = bcfg.get("type", "qformer")
    if btype in ("qformer", "pool"):
        return int(bcfg["num_query_tokens"])
    if btype == "mlp":                     # every (remaining) encoder token goes through
        return RAW_TOKENS[encoder] - (1 if bcfg.get("drop_cls", False) else 0)
    raise ValueError(f"unknown bridge type {btype!r}")


def main() -> None:
    """Assemble the B2 table from results_ci.csv + checkpoint configs; write markdown."""
    cfg = load_config()
    eval_dir = Path(cfg["paths"]["outputs"]) / "eval"
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ci_rows = list(csv.DictReader(open(eval_dir / "results_ci.csv")))
    by_key = {(r["run"], r["scope"], r["metric"]): r for r in ci_rows}

    print("=== raw source rows (results_ci.csv, metric=vqa_soft) ===")
    lines = ["# B2 connector comparison — accuracy and vision-token count", "",
             "VQA-soft on the locked 2,000-question GQA set (bridge accuracy, 95% "
             "bootstrap CI). \"Vision tokens\" = vision tokens passed to the LLM — a "
             "deterministic property of each connector's output shape; nothing was "
             "timed or benchmarked.", "",
             "| Encoder | Connector | Vision tokens | Overall VQA-soft [95% CI] | " +
             " | ".join(CATEGORIES) + " |",
             "|---|---|---:|---|" + "---:|" * len(CATEGORIES)]
    pending = []
    for encoder_lbl, label, stem, ckpt_name in RUNS:
        encoder_key = "ijepa" if encoder_lbl == "I-JEPA" else "clip"
        overall = by_key.get((stem, "overall", "vqa_soft"))
        tokens = vision_tokens(ckpt_dir / ckpt_name, encoder_key)
        if overall is None or tokens is None:
            pending.append(stem)
            lines.append(f"| {encoder_lbl} | {label} | {tokens if tokens is not None else '—'} | "
                         f"pending ({stem}) |" + " — |" * len(CATEGORIES))
            continue
        print(f"{stem} overall: model={overall['model_%']} "
              f"CI=[{overall['model_ci_lo']},{overall['model_ci_hi']}]")
        cats = []
        for cat in CATEGORIES:
            r = by_key.get((stem, cat, "vqa_soft"))
            cats.append(f"{r['model_%']}" if r else "—")
            if r:
                print(f"{stem} {cat}: model={r['model_%']} "
                      f"CI=[{r['model_ci_lo']},{r['model_ci_hi']}] n={r['n']}")
        lines.append(f"| {encoder_lbl} | {label} | {tokens} | {overall['model_%']} "
                     f"[{overall['model_ci_lo']}, {overall['model_ci_hi']}] | "
                     + " | ".join(cats) + " |")
    lines += ["", "Per-category cells are point estimates; their CIs are in "
              "results_ci.csv (echoed above by this script) and in the paired "
              "connector contrasts (connector_*_vs_qformer_150k rows)."]
    if pending:
        lines += ["", f"Pending runs (not yet on disk): {', '.join(pending)}."]
        print(f"pending: {', '.join(pending)}")

    out = eval_dir / "b2_connector_table.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"\n=== table ===\n" + "\n".join(lines))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
