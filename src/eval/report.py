"""Reusable evaluation reporting: a markdown results table and a qualitative example figure.

Generic over two methods A and B (for example bridge vs floor, or CoT vs baseline). Each
per-question record carries both methods' answers and correctness flags under generic keys
(a_ans/a_exact/a_vqa, b_ans/b_exact/b_vqa); these helpers summarise and visualise them so
every augmentation evaluation produces a consistent, citable table and figure.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image


def render_results_table(title, n, a_label, b_label, totals, by_cat, categories, split_label):
    """Render the A vs B markdown table (overall + per-category) as a string.

    Rendering is separate from writing so callers can route the text through the
    guarded artifact writer, which refuses to clobber and prepends provenance.
    `split_label` names the actual qid set: the old text asserted "Locked evaluation
    set" for every slice, which mislabels a tuning run as the locked one.
    """
    a_ex, a_vq, b_ex, b_vq = totals
    lines = [f"# {title}", "",
             f"Evaluation set `{split_label}`: **{n}** questions (zero-shot; never trained on).", "",
             "| Setting | Exact | VQA-soft |", "|---|---:|---:|",
             f"| {a_label} | {100 * a_ex / n:.1f}% | {100 * a_vq / n:.1f}% |",
             f"| {b_label} | {100 * b_ex / n:.1f}% | {100 * b_vq / n:.1f}% |",
             f"| **Effect (A − B)** | **{100 * (a_ex - b_ex) / n:+.1f}** | "
             f"**{100 * (a_vq - b_vq) / n:+.1f}** |", "",
             "## Per-category (VQA-soft)", "",
             f"| Category | n | {a_label} | {b_label} | Δ |", "|---|---:|---:|---:|---:|"]
    for cat in categories:
        rec = by_cat.get(cat)
        if not rec or not rec["total"]:
            continue
        t = rec["total"]
        lines.append(f"| {cat} | {t} | {100 * rec['a_vqa'] / t:.1f}% | {100 * rec['b_vqa'] / t:.1f}% | "
                     f"{100 * (rec['a_vqa'] - rec['b_vqa']) / t:+.1f} |")
    return "\n".join(lines) + "\n"


def write_results_table(path, title, n, a_label, b_label, totals, by_cat, categories,
                        split_label):
    """Write the rendered A vs B table to `path` (unguarded; prefer the artifact writer)."""
    Path(path).write_text(
        render_results_table(title, n, a_label, b_label, totals, by_cat, categories, split_label))


def select_examples(records, n, prefer="a"):
    """Pick up to n examples illustrating the finding's direction.

    prefer='a' shows where A beats B (use when A is the better method); prefer='b' shows where
    B beats A (use for an honest figure when A — e.g. an augmentation — actually hurt). Always
    includes one both-correct and one both-wrong case so the figure is not cherry-picked.
    """
    if prefer == "b":
        primary = [r for r in records if r["b_vqa"] and not r["a_vqa"]]
    else:
        primary = [r for r in records if r["a_vqa"] and not r["b_vqa"]]
    primary = sorted(primary, key=lambda r: r["qid"])
    both = sorted([r for r in records if r["a_vqa"] and r["b_vqa"]], key=lambda r: r["qid"])
    neither = sorted([r for r in records if not r["a_vqa"] and not r["b_vqa"]], key=lambda r: r["qid"])
    chosen, seen = [], set()
    for bucket, k in [(primary, max(1, n - 2)), (both, 1), (neither, 1)]:
        for r in bucket[:k]:
            if r["qid"] not in seen:
                chosen.append(r)
                seen.add(r["qid"])
    for r in primary + both + neither:       # top up to n if a bucket was short
        if len(chosen) >= n:
            break
        if r["qid"] not in seen:
            chosen.append(r)
            seen.add(r["qid"])
    return chosen[:n]


def render_examples_figure(path, records, n, a_label, b_label, title, prefer="a"):
    """Render n example images with question, gold, and the two methods' answers (ticks/crosses).

    prefer selects which contrast to illustrate (see select_examples): 'a' for A's wins, 'b' for
    where B beats A (the honest choice when A hurt).
    """
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = select_examples(records, n, prefer=prefer)
    if not chosen:
        return 0
    rows = len(chosen)
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.1 * rows),
                             gridspec_kw={"width_ratios": [1, 1.5]})
    if rows == 1:
        axes = axes.reshape(1, 2)
    for ax_img, ax_txt in axes:
        ax_img.axis("off")
        ax_txt.axis("off")
    for (ax_img, ax_txt), r in zip(axes, chosen):
        ax_img.imshow(Image.open(r["image_path"]).convert("RGB"))
        at = "✓" if r["a_vqa"] else "✗"
        bt = "✓" if r["b_vqa"] else "✗"
        # Display only the first line; the LLM may run past the answer (scoring uses the full text).
        a_disp = (r["a_ans"].splitlines() or [""])[0].strip()
        b_disp = (r["b_ans"].splitlines() or [""])[0].strip()
        q = "\n".join(textwrap.wrap(r["question"], 46))
        ax_txt.text(0.0, 0.95,
                    f"[{r['category']}]  Q: {q}\n\n"
                    f"gold answer:  {r['gold']}\n"
                    f"{a_label}:  {a_disp}   {at}\n"
                    f"{b_label}:  {b_disp}   {bt}",
                    transform=ax_txt.transAxes, va="top", ha="left", fontsize=11, family="monospace")
    fig.suptitle(title, fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return len(chosen)


def describe_checkpoint(ckpt, bridge_config, llm_precision, drop_cls):
    """Describe a checkpoint's encoder, connector shape, LLM precision and [CLS] handling.

    This is the human-readable provenance line for an evaluation run, so it must report what
    was actually loaded. Script 07's version assumed a Q-Former and a 4-bit/8-bit binary:
    `num_query_tokens` is a config default that survives into MLP checkpoints where it means
    nothing (an MLP forwards every patch token), and the precision predicate predated bf16
    and fp16. Every bf16 Wave 1 evaluation therefore logged itself as "32 query tokens,
    4-bit" while the checkpoint on disk was correct. Only the log was wrong — which is worse,
    because the log is what a reader audits to learn what produced a number.
    """
    bridge_type = bridge_config.get("type", "qformer")
    if bridge_type in ("qformer", "pool"):
        shape = f"{bridge_config['num_query_tokens']} query tokens"
    else:
        shape = f"{bridge_type}, all patch tokens forwarded"
    parts = [f"encoder {ckpt.get('encoder')}", shape, f"LLM {llm_precision}"]
    if drop_cls:
        parts.append("[CLS] STRIPPED from bridge memory")
    return ", ".join(parts)
