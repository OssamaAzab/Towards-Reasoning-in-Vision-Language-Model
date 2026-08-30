"""Step 7: evaluate a trained bridge on GQA, against a matched text-only floor.

Loads a trained bridge + frozen encoder + frozen LLM and runs inference on a
held-out GQA slice. For each question it generates two answers on the SAME question:
one conditioned on the image (through the bridge) and one with no image (matched
text-only floor). Both are scored by exact match after normalisation (lowercase;
strip articles a/an/the; strip punctuation/whitespace), because the LLM emits
free-form text while GQA answers are short. Reports bridge vs floor accuracy, a
per-reasoning-category breakdown, and a few raw-vs-normalised examples so the
matching can be sanity-checked.

    python scripts/07_evaluate.py --limit 300
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import random

import torch
from PIL import Image

# Make the repo root importable when run as `python scripts/07_evaluate.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.gqa import GQADataset, REASONING_CATEGORIES  # noqa: E402
from src.eval.metrics import METRIC_VERSION, exact_full, exact_match, normalize_answer, vqa_match  # noqa: E402
from src.eval.report import describe_checkpoint  # noqa: E402
from src.models.connectors import build_bridge  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import PRECISIONS, load_llm, logit_health  # noqa: E402
from src import prompt as P  # noqa: E402
from src.artifact import artifact_stem, build_meta  # noqa: E402
from src.models.vlm import assemble, text_only_generate, vlm_generate  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def write_results_table(path, ckpt_name, n, totals, by_cat, categories, split_label):
    """Write a markdown results table (overall + per-category) for the report."""
    b_exact, b_vqa, f_exact, f_vqa = totals
    lines = [f"# GQA evaluation — `{ckpt_name}`", "",
             f"Evaluation set `{split_label}`: **{n}** questions (zero-shot; never trained on).", "",
             "| Setting | Exact | VQA-soft |", "|---|---:|---:|",
             f"| Bridge (vision) | {100 * b_exact / n:.1f}% | {100 * b_vqa / n:.1f}% |",
             f"| Text-only floor | {100 * f_exact / n:.1f}% | {100 * f_vqa / n:.1f}% |",
             f"| **Vision effect** | **{100 * (b_exact - f_exact) / n:+.1f}** | "
             f"**{100 * (b_vqa - f_vqa) / n:+.1f}** |", "",
             "## Per-category (VQA-soft)", "",
             "| Category | n | Bridge | Floor | Δ | Bridge exact |",
             "|---|---:|---:|---:|---:|---:|"]
    for cat in categories:
        be_, fe_, bv_, fv_, tot = by_cat.get(cat, [0, 0, 0, 0, 0])
        if tot:
            lines.append(f"| {cat} | {tot} | {100 * bv_ / tot:.1f}% | {100 * fv_ / tot:.1f}% | "
                         f"{100 * (bv_ - fv_) / tot:+.1f} | {100 * be_ / tot:.1f}% |")
    path.write_text("\n".join(lines) + "\n")
    log(f"wrote results table -> {path}")


def select_examples(records, n):
    """Pick up to n illustrative examples: prefer vision-helps cases, keep one honest miss."""
    helps = sorted([r for r in records if r["bridge_vqa"] and not r["floor_vqa"]], key=lambda r: r["qid"])
    both = sorted([r for r in records if r["bridge_vqa"] and r["floor_vqa"]], key=lambda r: r["qid"])
    miss = sorted([r for r in records if not r["bridge_vqa"]], key=lambda r: r["qid"])
    chosen, seen = [], set()
    for bucket, k in [(helps, max(1, n - 2)), (both, 1), (miss, 1)]:
        for r in bucket[:k]:
            if r["qid"] not in seen:
                chosen.append(r)
                seen.add(r["qid"])
    for r in helps + both + miss:           # top up to n if a bucket was short
        if len(chosen) >= n:
            break
        if r["qid"] not in seen:
            chosen.append(r)
            seen.add(r["qid"])
    return chosen[:n]


def render_examples_figure(path, records, n):
    """Render n example images with question, gold, and bridge/floor predictions (ticks/crosses)."""
    import textwrap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chosen = select_examples(records, n)
    if not chosen:
        return
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
        bt = "✓" if r["bridge_vqa"] else "✗"
        ft = "✓" if r["floor_vqa"] else "✗"
        # Display only the first line; the LLM sometimes runs past the answer (scoring uses the full text).
        b_disp = (r["bridge"].splitlines() or [""])[0].strip()
        f_disp = (r["floor"].splitlines() or [""])[0].strip()
        q = "\n".join(textwrap.wrap(r["question"], 46))
        ax_txt.text(0.0, 0.95,
                    f"[{r['category']}]  Q: {q}\n\n"
                    f"gold answer:  {r['gold']}\n"
                    f"bridge (vision):  {b_disp}   {bt}\n"
                    f"text-only floor:  {f_disp}   {ft}",
                    transform=ax_txt.transAxes, va="top", ha="left", fontsize=11, family="monospace")
    fig.suptitle("GQA examples — bridge (vision) vs text-only floor", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {len(chosen)} example images -> {path}")


def main() -> None:
    """Score a trained bridge and a matched text-only floor on a GQA slice."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=300, help="number of questions to sample")
    ap.add_argument("--qids", default=None, help="fixed JSON list of question IDs (locks the eval set; overrides --limit)")
    ap.add_argument("--split", default="val_balanced", help="GQA questions split key")
    ap.add_argument("--checkpoint", default=None, help="bridge checkpoint (default: by encoder)")
    ap.add_argument("--encoder", default=None, help="override encoder (default: inferred from the checkpoint)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="generation cap (default: 10 for legacy, 32 for the corrected "
                         "protocol, where EOS is supervised so the cap is a safety net)")
    ap.add_argument("--prompt-format", choices=["raw", "chatml_v1"], default=None,
                    help="prompt layout (default: whatever the checkpoint was trained under)")
    ap.add_argument("--supervise-eos", action=argparse.BooleanOptionalAction, default=None,
                    help="stop-token protocol (default: the checkpoint's)")
    ap.add_argument("--allow-format-mismatch", action="store_true",
                    help="evaluate a checkpoint outside its training protocol. This is an "
                         "OFF-DISTRIBUTION PROBE, never that checkpoint's score; requires "
                         "--stem-suffix so it cannot overwrite the in-protocol result")
    ap.add_argument("--out-dir", default=None, help="where to save results table + figure (default: outputs/eval)")
    ap.add_argument("--num-examples", type=int, default=5, help="qualitative example images to render (0=off)")
    ap.add_argument("--llm-precision", choices=list(PRECISIONS), default=None,
                    help="override the frozen-LLM precision (quantization ablation; "
                         "default = the precision the checkpoint was trained under). "
                         "Requires --stem-suffix when it differs")
    ap.add_argument("--lora-adapter", default=None,
                    help="peft adapter dir to load onto the frozen LLM (A2/B1 LoRA "
                         "diagnostics; the records stem should mark it via --stem-suffix)")
    ap.add_argument("--stem-suffix", default=None,
                    help="append _<suffix> to the records/table stem — REQUIRED whenever "
                         "the eval condition differs from the plain checkpoint (LoRA "
                         "adapters, supplementary qid sets like compare_topup), so "
                         "records never collide with the locked-set base run")
    args = ap.parse_args()

    cfg = load_config()
    set_seed(cfg.get("seed", 42))
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = "cuda"

    # Load the checkpoint first; it records the exact config the bridge was trained
    # with (encoder, quantization, bridge size), so the evaluator reproduces it automatically.
    active = cfg["models"].get("active_encoder", "ijepa")
    ckpt_path = Path(args.checkpoint or Path(cfg["paths"]["checkpoints"]) / f"bridge_{active}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    eight_bit = ckpt.get("eight_bit", False)
    bcfg = ckpt.get("bridge_config", cfg["bridge"])
    # Encoder: explicit override > the encoder recorded in the checkpoint > config default.
    by_id = {mid: nm for nm, mid in cfg["models"]["encoders"].items()}
    name = args.encoder or by_id.get(ckpt.get("encoder")) or active

    # Protocol: follow the checkpoint unless explicitly overridden. Evaluating a
    # checkpoint under a protocol it was not trained for measures the probe, not the model.
    spec = P.PromptSpec(
        prompt_format=args.prompt_format or P.format_of(ckpt),
        supervise_eos=(bool(ckpt.get("supervise_eos", False))
                       if args.supervise_eos is None else args.supervise_eos),
        system_text=ckpt.get("system_text", P.SYSTEM_TEXT),
        max_answer_tokens=None)
    P.assert_compatible(ckpt, spec, allow_mismatch=args.allow_format_mismatch)
    if args.allow_format_mismatch and not args.stem_suffix:
        raise SystemExit("--allow-format-mismatch requires --stem-suffix: an "
                         "off-distribution probe must never overwrite the in-protocol result")
    max_new = args.max_new_tokens if args.max_new_tokens is not None else (
        32 if spec.supervise_eos else 10)
    log(f"PROTOCOL: {spec.prompt_format} supervise_eos={spec.supervise_eos} "
        f"layer={spec.evidence_layer} max_new_tokens={max_new} metric={METRIC_VERSION}")

    trained_precision = ckpt.get("llm_precision") or ("8bit" if eight_bit else "4bit")
    llm_precision = args.llm_precision or trained_precision
    if llm_precision != trained_precision:
        # The quantization ablation: same bridge, same protocol, different frozen LLM.
        # Legitimate, but it is NOT this checkpoint's own score, so it must be named.
        if not args.stem_suffix:
            raise SystemExit(
                f"--llm-precision {llm_precision} differs from the checkpoint's "
                f"{trained_precision}; pass --stem-suffix so the result cannot "
                f"overwrite the in-precision run")
        log(f"PRECISION ABLATION: evaluating a {trained_precision}-trained bridge under "
            f"{llm_precision}. The bridge is unchanged; only the frozen LLM differs.")
    log(f"loading frozen encoder '{name}' and the LLM ({llm_precision}) ...")
    enc = load_vision_encoder(name, cfg, device)
    llm = load_llm(cfg, log=log, precision=llm_precision)
    if args.lora_adapter:
        from peft import PeftModel
        llm.model = PeftModel.from_pretrained(llm.model, args.lora_adapter)
        llm.model.eval()
        log(f"LoRA DIAGNOSTIC: adapters loaded from {args.lora_adapter} onto the LLM")

    bridge = build_bridge(bcfg, enc.hidden_dim, llm.model.config.hidden_size).to(device)
    bridge.load_state_dict(ckpt["state_dict"])
    bridge.eval()
    log(f"loaded checkpoint: {ckpt_path.name}  "
        f"({describe_checkpoint(ckpt, bcfg, llm_precision, bridge.drop_cls)})")

    ds = GQADataset(cfg["gqa"]["questions"][args.split], cfg["gqa"]["image_dirs"])
    if args.qids:
        sample = json.load(open(args.qids))
        split_label = Path(args.qids).stem
        # Name the split, never assert "locked" — tuning splits come through here
        # too, and a log that calls a tuning run "locked" is how a selection set
        # ends up cited as a headline number.
        log(f"evaluating on FIXED qid set '{split_label}': {len(sample)} qids "
            f"from {args.qids}")
    else:
        sample = random.sample(ds.qids, min(args.limit, len(ds.qids)))
        split_label = f"sample{len(sample)}"
        log(f"evaluating {len(sample)} GQA questions (random sample) ...")

    # Ask for a short answer so the (verbose, LLaVA-trained) model output is
    # comparable to GQA's short gold answers; applied identically to both paths.
    suffix = P.SHORT_CUE   # single-sourced; was a fourth independent copy

    # Health-check the ACTUAL spliced sequence, visual tokens included. Text-only probes
    # gave fp16 a clean verdict on job 2285306 while it was damaging the visual arm only.
    _ex0 = ds.get(sample[0])
    _feats0 = encode_image(enc, Image.open(_ex0.image_path).convert("RGB"))
    _built0 = P.build(spec, _ex0.question + P.SHORT_CUE, llm.tokenizer,
                      n_visual=bridge(_feats0.float()).size(1), compose=False)
    _embeds0, _ = assemble(_built0, bridge(_feats0.float()),
                           llm.model.get_input_embeddings(), device)
    health = logit_health(llm, inputs_embeds=_embeds0)
    log(f"LOGIT HEALTH ({llm_precision}): {health['n_probes'] - health['n_nonfinite']}"
        f"/{health['n_probes']} probes finite (visual probe included: "
        f"{health['probed_visual']}), max|logit| {health['max_abs_logit']:.1f}")
    if not health["healthy"]:
        log("WARNING: non-finite logits. This precision is numerically broken on this "
            "model; any accuracy below is measuring the overflow, not the bridge.")

    b_exact = b_vqa = f_exact = f_vqa = 0
    # category -> [bridge_exact, floor_exact, bridge_vqa, floor_vqa, total]
    by_cat = defaultdict(lambda: [0, 0, 0, 0, 0])
    shown = []
    records = []  # one dict per question: full provenance for the table + figure

    for i, qid in enumerate(sample, 1):
        ex = ds.get(qid)
        prompt = ex.question + suffix
        image = Image.open(ex.image_path).convert("RGB")
        feats = encode_image(enc, image)
        b_raw, b_stop = vlm_generate(bridge, llm, feats, prompt, device,
                                     max_new_tokens=max_new, spec=spec, return_stop=True)
        f_raw, f_stop = text_only_generate(llm, prompt, device, max_new_tokens=max_new,
                                           spec=spec, return_stop=True)
        b_ans, f_ans = b_raw.strip(), f_raw.strip()
        cat = ds.category_of(ex)
        be, fe = exact_match(b_ans, ex.answer), exact_match(f_ans, ex.answer)
        bv, fv = vqa_match(b_ans, ex.answer), vqa_match(f_ans, ex.answer)
        b_exact += be
        f_exact += fe
        b_vqa += bv
        f_vqa += fv
        c = by_cat[cat]
        c[0] += be
        c[1] += fe
        c[2] += bv
        c[3] += fv
        c[4] += 1
        records.append({"qid": qid, "category": cat, "question": ex.question,
                        "gold": ex.answer, "bridge": b_ans, "floor": f_ans,
                        "image_path": str(ex.image_path),
                        "bridge_exact": bool(be), "floor_exact": bool(fe),
                        "bridge_vqa": bool(bv), "floor_vqa": bool(fv),
                        # primary corrected metric + termination diagnostics
                        "bridge_exact_full": bool(exact_full(b_ans, ex.answer)),
                        "floor_exact_full": bool(exact_full(f_ans, ex.answer)),
                        "bridge_stop_reason": b_stop["stop_reason"],
                        "bridge_cap_hit": b_stop["cap_hit"],
                        "bridge_n_generated": b_stop["n_generated"],
                        "floor_stop_reason": f_stop["stop_reason"],
                        "floor_cap_hit": f_stop["cap_hit"],
                        "floor_n_generated": f_stop["n_generated"]})
        if len(shown) < 8:
            shown.append((ex.question, ex.answer, b_ans, normalize_answer(b_ans), bool(be)))
        if i % 50 == 0:
            log(f"  {i}/{len(sample)}  bridge exact={100 * b_exact / i:.1f}%  "
                f"floor exact={100 * f_exact / i:.1f}%")

    n = len(sample)
    print("\n=== GQA EVALUATION: bridge (with image) vs text-only floor ===")
    print(f"checkpoint : {ckpt_path.name}   questions: {n}\n")
    print(f"{'':18}{'exact':>8}{'VQA-soft':>11}")
    print(f"{'bridge (vision)':18}{100 * b_exact / n:7.1f}%{100 * b_vqa / n:10.1f}%")
    print(f"{'text-only floor':18}{100 * f_exact / n:7.1f}%{100 * f_vqa / n:10.1f}%")
    print(f"\nvision effect: exact {100 * (b_exact - f_exact) / n:+.1f} pts, "
          f"VQA-soft {100 * (b_vqa - f_vqa) / n:+.1f} pts")

    bx = 100 * sum(r["bridge_exact_full"] for r in records) / n
    fx = 100 * sum(r["floor_exact_full"] for r in records) / n
    cap = 100 * sum(r["bridge_cap_hit"] for r in records) / n
    term = 100 - cap
    print(f"\nPRIMARY ({METRIC_VERSION}): bridge {bx:.1f}%  floor {fx:.1f}%  "
          f"effect {bx - fx:+.1f} pts")
    print(f"TERMINATION: bridge stopped at EOS {term:.1f}%  cap-hit {cap:.1f}%  "
          f"mean generated {sum(r['bridge_n_generated'] for r in records) / n:.1f} tokens")

    print("\nPer-category VQA-soft (bridge vs floor):")
    for cat in REASONING_CATEGORIES + ["other"]:
        be_, fe_, bv_, fv_, tot = by_cat.get(cat, [0, 0, 0, 0, 0])
        if tot:
            print(f"  {cat:8}: bridge {100 * bv_ / tot:5.1f}%  floor {100 * fv_ / tot:5.1f}%  "
                  f"(exact b {100 * be_ / tot:4.1f}%)   (n={tot})")

    print("\nRaw vs normalised (bridge answers) — sanity-check the matching:")
    for q, gold, raw, norm, ok in shown:
        print(f"  [{'OK' if ok else '  '}] gold={gold!r:16} raw={raw!r:34} norm={norm!r}")
        print(f"       Q: {q}")

    # Persist the results so the eval is reproducible and citable in the report.
    out_dir = Path(args.out_dir or Path(cfg["paths"]["outputs"]) / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = ckpt_path.stem + (f"_{args.stem_suffix}" if args.stem_suffix else "")
    if spec.evidence_layer == P.EVIDENCE_CORRECTED:
        # Corrected runs never reuse a legacy artifact name: outputs/ is gitignored,
        # so overwriting a legacy record is unrecoverable.
        stem = artifact_stem(ckpt_path.stem, evidence_layer=spec.evidence_layer,
                             split=split_label, input_mode=spec.input_mode,
                             reason_mode=spec.reason_mode)
        if args.stem_suffix:
            stem = f"{stem}__{args.stem_suffix}"
    meta = build_meta(spec=spec, checkpoint=ckpt, split=split_label,
                      metric_version=METRIC_VERSION,
                      extra={"max_new_tokens": max_new, "n_questions": n,
                             "eval_llm_precision": llm_precision,
                             "trained_llm_precision": trained_precision,
                             "logit_health": health})
    records_path = out_dir / f"{stem}_records.json"
    if records_path.exists():
        raise SystemExit(
            f"refusing to overwrite {records_path}. outputs/ is gitignored, so an "
            "overwrite is unrecoverable — use a distinct --stem-suffix.")
    records_path.write_text(json.dumps({"_meta": meta, "records": records}, indent=2)
                            if spec.evidence_layer == P.EVIDENCE_CORRECTED
                            else json.dumps(records, indent=2))
    log(f"wrote {len(records)} per-question records -> {records_path}")
    write_results_table(out_dir / f"{stem}_results.md", ckpt_path.name, n,
                        (b_exact, b_vqa, f_exact, f_vqa), by_cat,
                        REASONING_CATEGORIES + ["other"], split_label=split_label)
    if args.num_examples:
        render_examples_figure(out_dir / f"{stem}_examples.png", records, args.num_examples)


if __name__ == "__main__":
    main()
