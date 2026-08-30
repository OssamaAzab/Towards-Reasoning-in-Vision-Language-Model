"""D2 diagnostic: token-weighted training mix vs the 70/30 example-level split (READ-ONLY).

The locked recipe mixes VQAv2 (short) and LLaVA (verbose) at short_frac=0.7 — a 70/30 split
of EXAMPLES. But the causal-LM loss averages over answer tokens, and only answer tokens are
unmasked. If LLaVA answers are much longer, the TOKEN-weighted mix that actually weights the
loss can be far from 70/30. This script quantifies that, training nothing.

It rebuilds the exact training example list by importing the trainer's own build_examples
(the trainer's main() is guarded, so importing it runs no training), tags each example's
source by the trainer's SHORT_SUFFIX, and tokenises each answer exactly as vlm_loss_batch
(add_special_tokens=False, truncation=True, max_length=64).

    python scripts/24_diag_token_mix.py               # 150K recipe (default)
    python scripts/24_diag_token_mix.py --limit 500000  # 500K recipe
"""
import argparse
import csv
import importlib.util
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils import load_config, set_seed  # noqa: E402
from src.models.revisions import revision_for  # noqa: E402

OUT_DIR = ROOT / "outputs" / "diagnostics"
MAX_ANSWER_TOKENS = 64   # vlm_loss_batch truncation length


def load_trainer():
    """Import the trainer module by path (name starts with a digit); runs no training."""
    spec = importlib.util.spec_from_file_location(
        "trainer06c", ROOT / "scripts" / "06c_train_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_tokenizer():
    """Return (tokenise_answer_fn, mode). Qwen2 truncated token count, else whitespace words."""
    try:
        from transformers import AutoTokenizer
        model_id = "Qwen/Qwen2-7B-Instruct"
        tok = AutoTokenizer.from_pretrained(model_id, revision=revision_for(model_id))

        def counts(answer):
            full = len(tok(answer, add_special_tokens=False).input_ids)
            return min(full, MAX_ANSWER_TOKENS), full        # (loss-weighting count, untruncated)
        return counts, "qwen2_tokens"
    except Exception as e:
        print(f"WARNING: Qwen2 tokenizer unavailable ({type(e).__name__}); whitespace fallback")

        def counts(answer):
            full = len(answer.split())
            return min(full, MAX_ANSWER_TOKENS), full
        return counts, "whitespace_words"


def agg(name, tokens, truncated, n_examples_total, total_tokens_all):
    """Assemble one source's row: example/token shares plus length stats."""
    n = len(tokens)
    tot = sum(tokens)
    return {
        "source": name,
        "n_examples": n,
        "example_share_%": round(100 * n / n_examples_total, 2) if n_examples_total else 0.0,
        "total_answer_tokens": tot,
        "answer_token_share_%": round(100 * tot / total_tokens_all, 2) if total_tokens_all else 0.0,
        "mean_answer_tokens": round(st.mean(tokens), 2) if tokens else 0.0,
        "median_answer_tokens": round(st.median(tokens), 1) if tokens else 0.0,
        "p90_answer_tokens": sorted(tokens)[int(0.9 * (n - 1))] if tokens else 0,
        "frac_truncated_at_64_%": round(100 * sum(truncated) / n, 2) if n else 0.0,
    }


def main():
    """Rebuild the mix, tokenise answers per source, verify the 70/30 split, write token_mix.csv."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=150000, help="total training examples")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(42)                              # match the locked protocol seed
    cfg = load_config()
    short_frac = cfg["train"]["short_frac"]

    # Fail loudly if any annotation the trainer reads is absent (do not substitute).
    needed = [cfg["llava"]["annotations"], cfg["vqa"]["questions"], cfg["vqa"]["annotations"]]
    missing = [p for p in needed if not Path(p).exists()]
    if missing:
        sys.exit("STOP: missing dataset annotation file(s), cannot rebuild the mix:\n  "
                 + "\n  ".join(missing))

    trainer = load_trainer()
    n_short = int(args.limit * short_frac)                       # trainer's exact computation
    n_llava = args.limit - n_short
    print(f"limit={args.limit} short_frac={short_frac} -> n_short(VQAv2)={n_short} n_llava(LLaVA)={n_llava}")

    examples = trainer.build_examples(cfg, n_short, n_llava)     # (image_path, prompt, answer)
    suffix = trainer.SHORT_SUFFIX
    counts, tok_mode = load_tokenizer()

    vqa_tok, vqa_trunc, llava_tok, llava_trunc = [], [], [], []
    for _, prompt, answer in examples:
        trunc_len, full_len = counts(answer)
        if prompt.endswith(suffix):                             # VQAv2 carries SHORT_SUFFIX
            vqa_tok.append(trunc_len); vqa_trunc.append(full_len > MAX_ANSWER_TOKENS)
        else:                                                    # LLaVA verbose
            llava_tok.append(trunc_len); llava_trunc.append(full_len > MAX_ANSWER_TOKENS)

    # Sanity check: the source tagging must reproduce the intended example-level split.
    if len(vqa_tok) != n_short or len(llava_tok) != n_llava:
        print(f"MISMATCH: tagged VQAv2={len(vqa_tok)} (expected {n_short}), "
              f"LLaVA={len(llava_tok)} (expected {n_llava})")
        sys.exit("Stopped on MISMATCH: SHORT_SUFFIX tagging does not reproduce the 70/30 split.")

    total_n = len(vqa_tok) + len(llava_tok)
    total_tokens = sum(vqa_tok) + sum(llava_tok)
    rows = [
        agg("VQAv2", vqa_tok, vqa_trunc, total_n, total_tokens),
        agg("LLaVA", llava_tok, llava_trunc, total_n, total_tokens),
    ]

    cols = ["source", "n_examples", "example_share_%", "total_answer_tokens",
            "answer_token_share_%", "mean_answer_tokens", "median_answer_tokens",
            "p90_answer_tokens", "frac_truncated_at_64_%"]
    out_csv = OUT_DIR / f"token_mix_{args.limit}.csv" if args.limit != 150000 else OUT_DIR / "token_mix.csv"
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    vqa, llava = rows[0], rows[1]
    print(f"\nlength unit: {tok_mode}")
    print(f"  {'source':7s} {'examples':>9} {'ex_share':>9} {'ans_tokens':>11} {'tok_share':>10} "
          f"{'mean':>6} {'p90':>5} {'trunc@64':>9}")
    for r in rows:
        print(f"  {r['source']:7s} {r['n_examples']:>9} {r['example_share_%']:>8}% "
              f"{r['total_answer_tokens']:>11} {r['answer_token_share_%']:>9}% "
              f"{r['mean_answer_tokens']:>6} {r['p90_answer_tokens']:>5} {r['frac_truncated_at_64_%']:>8}%")
    print(f"\nSUMMARY (limit={args.limit}): "
          f"EXAMPLE-level VQAv2/LLaVA = {vqa['example_share_%']:.1f}/{llava['example_share_%']:.1f}  |  "
          f"TOKEN-level VQAv2/LLaVA = {vqa['answer_token_share_%']:.1f}/{llava['answer_token_share_%']:.1f}")
    print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
