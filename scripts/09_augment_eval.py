"""Step 9: evaluate an inference-time augmentation (RQ1) against the un-augmented bridge.

Loads the trained bridge + frozen encoder + frozen LLM and, for each GQA question on the
locked evaluation set, generates two answers from the SAME vision bridge: one with the
augmentation applied (method A, e.g. chain-of-thought) and one with the standard short-answer
prompt (method B, the baseline). Both are scored by exact match and VQA-soft, with a
per-reasoning-category breakdown, so RQ1 is read directly: does the augmentation help, and on
which categories? Results, a markdown table, and a qualitative figure are saved to outputs/eval.

    python scripts/09_augment_eval.py --augment cot \
        --qids data/gqa/eval_2000_qids.json \
        --checkpoint outputs/checkpoints/bridge_ijepa_150k_3ep.pt
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image

# Make the repo root importable when run as `python scripts/09_augment_eval.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import prompt as P  # noqa: E402
from src.artifact import artifact_stem, build_meta, write_json, write_text  # noqa: E402
from src.augment import make_augmenter  # noqa: E402
from src.data.gqa import GQADataset, REASONING_CATEGORIES  # noqa: E402
from src.eval.metrics import METRIC_VERSION, exact_full, exact_match, vqa_match  # noqa: E402
from src.eval.report import describe_checkpoint  # noqa: E402
from src.eval.report import render_examples_figure, render_results_table  # noqa: E402
from src.models.connectors import build_bridge  # noqa: E402
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import load_llm  # noqa: E402
from src.models.vlm import vlm_generate  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Print a progress line prefixed with elapsed seconds, flushed immediately."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


def main() -> None:
    """Score an augmented bridge (A) against the un-augmented bridge (B) on a GQA slice."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--augment", default="cot", help="augmentation to test: cot | scene_graph")
    ap.add_argument("--encoder", default=None, help="override active encoder: clip | ijepa | vjepa2")
    ap.add_argument("--limit", type=int, default=300, help="number of questions to sample")
    ap.add_argument("--qids", default=None, help="fixed JSON list of question IDs (overrides --limit)")
    ap.add_argument("--split", default="val_balanced", help="GQA questions split key")
    ap.add_argument("--checkpoint", default=None, help="bridge checkpoint (default: by encoder)")
    ap.add_argument("--out-dir", default=None, help="where to save table + figure (default: outputs/eval)")
    ap.add_argument("--num-examples", type=int, default=5, help="qualitative example images (0=off)")
    ap.add_argument("--pred-cache", default=None,
                    help="override the predicted-graph cache path (per-image AND per-question "
                         "keys; for scene_graph_pred* runs on non-locked slices, e.g. tuning)")
    ap.add_argument("--prompt-format", choices=["raw", "chatml_v1"], default=None,
                    help="prompt layout (default: whatever the checkpoint was trained under)")
    ap.add_argument("--supervise-eos", action=argparse.BooleanOptionalAction, default=None,
                    help="stop-token protocol (default: the checkpoint's)")
    ap.add_argument("--max-new-tokens", type=int, default=None,
                    help="override the generation cap for BOTH arms (default: each "
                         "augmentation's own budget for its protocol)")
    ap.add_argument("--allow-format-mismatch", action="store_true",
                    help="evaluate a checkpoint outside its training protocol. This is an "
                         "OFF-DISTRIBUTION PROBE, never that checkpoint's score; requires "
                         "--stem-suffix so it cannot overwrite the in-protocol result")
    ap.add_argument("--stem-suffix", default=None,
                    help="append to the records/table stem — REQUIRED whenever the eval "
                         "condition differs from the plain checkpoint, so records never "
                         "collide with an existing run")
    ap.add_argument("--overwrite", action="store_true",
                    help="permit overwriting existing artifacts (outputs/ is gitignored, so "
                         "an overwrite is unrecoverable — pass this deliberately)")
    args = ap.parse_args()

    cfg = load_config()
    if args.pred_cache:
        cfg["augmentation"]["pred_graphs"] = args.pred_cache
        cfg["augmentation"]["pred_graphs_q"] = args.pred_cache
    set_seed(cfg.get("seed", 42))
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required.")
    device = "cuda"

    aug = make_augmenter(args.augment, cfg)    # method A: the augmentation under test
    base = make_augmenter("none")              # method B: the un-augmented baseline
    if aug.name == base.name:
        raise SystemExit("--augment none has nothing to compare against; choose cot or scene_graph.")
    if getattr(aug, "oracle", False):
        log("=" * 78)
        log(f"ORACLE PROBE: '{aug.name}' injects each eval image's OWN ground-truth GQA graph.")
        log("GQA questions were generated from these graphs, so this is an UPPER BOUND, not a")
        log("realistic result. Do not compare against the base/CoT numbers as if it were one.")
        log("=" * 78)

    # The checkpoint records the exact config the bridge was trained with.
    name = args.encoder or cfg["models"].get("active_encoder", "ijepa")
    ckpt_path = Path(args.checkpoint or Path(cfg["paths"]["checkpoints"]) / f"bridge_{name}.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    eight_bit = ckpt.get("eight_bit", False)
    bcfg = ckpt.get("bridge_config", cfg["bridge"])

    # Protocol: follow the checkpoint unless explicitly overridden. Both arms share the
    # protocol levers and differ only in framing, so any effect is the augmentation's.
    base_spec = P.PromptSpec(
        prompt_format=args.prompt_format or P.format_of(ckpt),
        supervise_eos=(bool(ckpt.get("supervise_eos", False))
                       if args.supervise_eos is None else args.supervise_eos),
        system_text=ckpt.get("system_text", P.SYSTEM_TEXT),
        max_answer_tokens=None)
    P.assert_compatible(ckpt, base_spec, allow_mismatch=args.allow_format_mismatch)
    if args.allow_format_mismatch and not args.stem_suffix:
        raise SystemExit("--allow-format-mismatch requires --stem-suffix: an "
                         "off-distribution probe must never overwrite the in-protocol result")
    a_spec, b_spec = aug.spec(base_spec), base.spec(base_spec)
    a_budget = args.max_new_tokens or aug.budget(base_spec)
    b_budget = args.max_new_tokens or base.budget(base_spec)
    log(f"PROTOCOL: {base_spec.prompt_format} supervise_eos={base_spec.supervise_eos} "
        f"layer={base_spec.evidence_layer} metric={METRIC_VERSION} "
        f"max_new_tokens: {aug.name}={a_budget} baseline={b_budget}")

    llm_precision = ckpt.get("llm_precision") or ("8bit" if eight_bit else "4bit")
    log(f"loading frozen encoder '{name}' and the LLM ({llm_precision}) ...")
    enc = load_vision_encoder(name, cfg, device)
    llm = load_llm(cfg, log=log, precision=llm_precision)

    bridge = build_bridge(bcfg, enc.hidden_dim, llm.model.config.hidden_size).to(device)
    bridge.load_state_dict(ckpt["state_dict"])
    bridge.eval()
    log(f"loaded checkpoint: {ckpt_path.name}; testing augmentation '{aug.name}' vs baseline"
        f"  ({describe_checkpoint(ckpt, bcfg, llm_precision, bridge.drop_cls)})")

    ds = GQADataset(cfg["gqa"]["questions"][args.split], cfg["gqa"]["image_dirs"])
    if args.qids:
        sample = json.load(open(args.qids))
        split_label = Path(args.qids).stem
        # Name the split, never assert "locked" — tuning slices come through here too.
        log(f"evaluating on FIXED qid set '{split_label}': {len(sample)} qids from {args.qids}")
    else:
        sample = random.sample(ds.qids, min(args.limit, len(ds.qids)))
        split_label = f"sample{len(sample)}"
        log(f"evaluating {len(sample)} GQA questions (random sample) ...")

    a_exact = a_vqa = b_exact = b_vqa = 0
    a_full = b_full = 0                        # exact_full: scored on the WHOLE output
    a_caps = b_caps = 0                        # ran to the token cap instead of stopping
    a_gen = b_gen = 0                          # total generated tokens, for the mean
    n_context = 0                              # how many examples actually received injected context
    by_cat = defaultdict(lambda: {"a_exact": 0, "a_vqa": 0, "b_exact": 0, "b_vqa": 0, "total": 0})
    records = []

    for i, qid in enumerate(sample, 1):
        ex = ds.get(qid)
        feats = encode_image(enc, Image.open(ex.image_path).convert("RGB"))
        a_ctx = aug.context_for(ex)            # e.g. the scene-graph description (None for cot)
        if a_ctx:
            n_context += 1
        a_raw, a_stop = vlm_generate(bridge, llm, feats,
                                     aug.build_prompt(ex.question, a_ctx, base_spec), device,
                                     max_new_tokens=a_budget, spec=a_spec, return_stop=True)
        b_raw, b_stop = vlm_generate(bridge, llm, feats,
                                     base.build_prompt(ex.question, None, base_spec), device,
                                     max_new_tokens=b_budget, spec=b_spec, return_stop=True)
        a_ans, b_ans = aug.extract_answer(a_raw), base.extract_answer(b_raw)
        cat = ds.category_of(ex)
        ae, be = exact_match(a_ans, ex.answer), exact_match(b_ans, ex.answer)
        av, bv = vqa_match(a_ans, ex.answer), vqa_match(b_ans, ex.answer)
        # exact_full scores the untruncated output; exact_match/vqa_match read only the
        # first line, which flatters a verbose arm. Under CoT the extracted answer is the
        # right unit, so exact_full is applied to the EXTRACTED span, not the raw stream.
        af, bf = exact_full(a_ans, ex.answer), exact_full(b_ans, ex.answer)
        a_exact += ae
        a_vqa += av
        a_full += af
        b_exact += be
        b_vqa += bv
        b_full += bf
        a_caps += a_stop["cap_hit"]
        b_caps += b_stop["cap_hit"]
        a_gen += a_stop["n_generated"]
        b_gen += b_stop["n_generated"]
        c = by_cat[cat]
        c["a_exact"] += ae
        c["a_vqa"] += av
        c["b_exact"] += be
        c["b_vqa"] += bv
        c["total"] += 1
        records.append({"qid": qid, "category": cat, "question": ex.question, "gold": ex.answer,
                        "image_path": str(ex.image_path),
                        "a_ans": a_ans, "a_exact": bool(ae), "a_vqa": bool(av),
                        "a_exact_full": bool(af),
                        "b_ans": b_ans, "b_exact": bool(be), "b_vqa": bool(bv),
                        "b_exact_full": bool(bf),
                        "a_raw": a_raw, "b_raw": b_raw, "a_context": a_ctx,
                        # Termination is the diagnostic that says whether a reasoning
                        # augmentation actually produced reasoning, or was cut short by a
                        # stop token the bridge was trained to emit after ~2 tokens.
                        "a_stop_reason": a_stop["stop_reason"], "a_cap_hit": a_stop["cap_hit"],
                        "a_n_generated": a_stop["n_generated"],
                        "b_stop_reason": b_stop["stop_reason"], "b_cap_hit": b_stop["cap_hit"],
                        "b_n_generated": b_stop["n_generated"]})
        if i % 50 == 0:
            log(f"  {i}/{len(sample)}  {aug.name} VQA-soft={100 * a_vqa / i:.1f}%  "
                f"baseline VQA-soft={100 * b_vqa / i:.1f}%")

    n = len(sample)
    print(f"\n=== GQA RQ1: '{aug.name}' augmentation vs un-augmented bridge ===")
    print(f"checkpoint : {ckpt_path.name}   questions: {n}\n")
    print(f"{'':28}{'exact':>8}{'VQA-soft':>11}{'exact_full':>13}")
    print(f"{aug.label:28}{100 * a_exact / n:7.1f}%{100 * a_vqa / n:10.1f}%{100 * a_full / n:12.1f}%")
    print(f"{base.label:28}{100 * b_exact / n:7.1f}%{100 * b_vqa / n:10.1f}%{100 * b_full / n:12.1f}%")
    print(f"\naugmentation effect: exact {100 * (a_exact - b_exact) / n:+.1f} pts, "
          f"VQA-soft {100 * (a_vqa - b_vqa) / n:+.1f} pts")
    print(f"PRIMARY ({METRIC_VERSION}): {aug.name} {100 * a_full / n:.1f}%  "
          f"baseline {100 * b_full / n:.1f}%  effect {100 * (a_full - b_full) / n:+.1f} pts")
    # A reasoning augmentation that never runs long enough to reason is a null result
    # about the harness, not about reasoning. Print the evidence either way.
    print(f"TERMINATION: {aug.name} cap-hit {100 * a_caps / n:.1f}% mean {a_gen / n:.1f} tok "
          f"(budget {a_budget})  |  baseline cap-hit {100 * b_caps / n:.1f}% "
          f"mean {b_gen / n:.1f} tok (budget {b_budget})")
    if aug.reason_mode != "direct" and a_gen / n < 8:
        print(f"WARNING: '{aug.name}' averaged only {a_gen / n:.1f} generated tokens — too few to "
              f"contain reasoning. Under a supervised stop token the bridge halts almost "
              f"immediately, so this arm is NOT testing {aug.reason_mode}; treat any "
              f"difference from baseline as prompt-wording noise, not a reasoning effect.")
    if hasattr(aug, "coverage"):
        print(aug.coverage())                  # graphs injected vs silent fallbacks
    if getattr(aug, "oracle", False):
        print(f"\n** ORACLE / UPPER BOUND ** — '{aug.name}' injected each image's OWN ground-truth "
              f"GQA graph\n   (graph available for {n_context}/{n} = {100 * n_context / n:.1f}% of "
              f"eval images). GQA questions were generated from these graphs, so this OVERSTATES "
              f"any realistic gain. Not the main RQ1 result.")

    print("\nPer-category VQA-soft (augmented vs baseline):")
    for cat in REASONING_CATEGORIES + ["other"]:
        c = by_cat.get(cat)
        if c and c["total"]:
            t = c["total"]
            print(f"  {cat:8}: {aug.name} {100 * c['a_vqa'] / t:5.1f}%  baseline "
                  f"{100 * c['b_vqa'] / t:5.1f}%  (Δ {100 * (c['a_vqa'] - c['b_vqa']) / t:+.1f})   (n={t})")

    # Persist results so the RQ1 run is reproducible and citable.
    out_dir = Path(args.out_dir or Path(cfg["paths"]["outputs"]) / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{ckpt_path.stem}_{aug.name}"
    if a_spec.evidence_layer == P.EVIDENCE_CORRECTED:
        # Corrected runs never reuse a legacy artifact name: outputs/ is gitignored,
        # so overwriting a legacy record is unrecoverable.
        stem = artifact_stem(ckpt_path.stem, evidence_layer=a_spec.evidence_layer,
                             split=split_label, input_mode=a_spec.input_mode,
                             reason_mode=a_spec.reason_mode)
    if args.stem_suffix:
        stem = f"{stem}__{args.stem_suffix}"
    meta = build_meta(spec=a_spec, checkpoint=ckpt, split=split_label,
                      metric_version=METRIC_VERSION,
                      extra={"augmentation": aug.name, "baseline": base.name,
                             "n_questions": n, "max_new_tokens_a": a_budget,
                             "max_new_tokens_b": b_budget,
                             "graph_framing": a_spec.graph_framing,
                             "oracle": bool(getattr(aug, "oracle", False)),
                             "n_context_injected": n_context,
                             # Which detector cache produced the injected structure. Two
                             # caches built from different vocabularies render identically
                             # in the prompt, so without this the record cannot say which
                             # one a result came from — exactly the confound that made the
                             # 120-vs-400-term pred/predq comparison uninterpretable.
                             "pred_cache": args.pred_cache,
                             "a_cap_hit_rate": a_caps / n, "b_cap_hit_rate": b_caps / n,
                             "a_mean_generated": a_gen / n, "b_mean_generated": b_gen / n})
    write_json(out_dir / f"{stem}_records.json", records, meta=meta, overwrite=args.overwrite)
    title = f"GQA RQ1 — {aug.label} vs {base.label} (`{ckpt_path.name}`)"
    if getattr(aug, "oracle", False):
        title += (f"\n\n> **ORACLE / upper bound** — injects each eval image's own ground-truth GQA "
                  f"scene graph (available for {n_context}/{n} images). GQA questions were generated "
                  f"from these graphs, so this overstates realistic gains; it is NOT the main RQ1 "
                  f"result, only an upper bound on what perfect structure could add.")
    table = render_results_table(title, n, aug.label, base.label,
                                 (a_exact, a_vqa, b_exact, b_vqa),
                                 by_cat, REASONING_CATEGORIES + ["other"], split_label)
    write_text(out_dir / f"{stem}_results.md", table, meta=meta, overwrite=args.overwrite)
    if args.num_examples:
        # Show the figure in the direction of the finding: if the augmentation hurt overall,
        # illustrate where the baseline beat it (honest), otherwise illustrate the augmentation's wins.
        prefer = "a" if a_vqa >= b_vqa else "b"
        k = render_examples_figure(out_dir / f"{stem}_examples.png", records, args.num_examples,
                                   aug.label, base.label,
                                   f"GQA RQ1: {aug.label} vs {base.label}", prefer=prefer)
        log(f"wrote {k} example images -> {out_dir / (stem + '_examples.png')}")
    log(f"wrote results table + {len(records)} records -> {out_dir}/{stem}_*")


if __name__ == "__main__":
    main()
