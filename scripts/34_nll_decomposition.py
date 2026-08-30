"""Rank 1: decompose the training loss by source, length, position and example.

The trainer reports one scalar. `vlm_loss_batch` returns HuggingFace's batch-level
token-mean CE over supervised answer tokens, and `eval_loss` then example-weights those
batch scalars — so the reported number is neither a per-sequence mean nor a globally
token-weighted one. Under the corrected 150K recipe, 30% of examples (LLaVA) carry
77.52% of the supervised *tokens*, so the scalar can fall because verbose conversation
text got easier rather than because GQA answers got better.

This script reports three DIFFERENT quantities and they must not be conflated:

  token share        - share of supervised target tokens. A proxy for the denominator.
  CE mass share      - sum(CE_source)/sum(CE_all). Descriptive. It does NOT decompose
                       the trainer's scalar, because that scalar normalises within each
                       batch by that batch's token count before example-weighting.
  objective          - two fields, and the distinction matters:
                       trainer_additive_contribution = sum_b n_b*(CE_{b,s}/T_b) / sum_b n_b,
                         in the scalar's own loss units. These sum across sources to the
                         trainer-reported scalar exactly, and the script aborts if not.
                       trainer_additive_share_pct    = 100 * that contribution divided by the
                         full scalar, i.e. 100 * sum_b n_b*(CE_{b,s}/T_b) / sum_b n_b*(CE_b/T_b).
                       A percentage must never be quoted against the contribution formula.

None of the three is gradient-norm share. Parameter Jacobians and cancellation between
examples are unmeasured, so nothing here licenses a claim about what "drives" training.

Consequently this script CANNOT on its own promote or reject the sequence-normalised-CE
proposal. A source contributing more of the objective than of the tokens means its mean
CE per token is higher; it does not show the allocation is wrong, and equal shares would
not remove the already-measured token imbalance. Treat the output as diagnostics.

READ-ONLY. It imports the trainer and `src.models.vlm` to reuse their exact code paths
and modifies neither. It trains nothing and writes only its own JSON.

The check that makes the rest trustworthy: per-token CE is re-aggregated exactly the way
`eval_loss` aggregates, and asserted equal to `eval_loss`'s own return value. If that
control fails, this script is measuring a different quantity than the trainer optimised
and every decomposition below it is void.
"""
from pathlib import Path
import argparse
import importlib.util
import json
import random
import sys
import time
import zlib

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.feature_cache import cache_dir_for, load_features   # noqa: E402
from src.models.connectors import build_bridge            # noqa: E402
from src.models.encoders import load_vision_encoder       # noqa: E402
from src.models.llm import load_llm                       # noqa: E402
from src.models.vlm import assemble                       # noqa: E402
from src.utils import load_config                         # noqa: E402
import src.prompt as P                                    # noqa: E402


def _trainer():
    """Import the training script as a module so its data builder is reused, not copied."""
    path = ROOT / "scripts" / "06c_train_bridge.py"
    spec = importlib.util.spec_from_file_location("trainer_06c", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def log(msg):
    """Timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@torch.no_grad()
def token_ce(bridge, llm, feats, questions, answers, device, spec):
    """Per-supervised-token CE for one batch, as a list of 1-D tensors (one per example).

    Mirrors `vlm_loss_batch` exactly up to the final reduction: same prompt build, same
    padding, same -100 masking, same causal shift. Only the reduction differs, because
    the point is to see inside the mean the trainer takes.
    """
    visual = bridge(feats.float())
    embed = llm.model.get_input_embeddings()
    dtype = embed.weight.dtype

    seqs, label_seqs, lengths, metas = [], [], [], []
    for i in range(len(questions)):
        built = P.build(spec, questions[i], llm.tokenizer, answer=answers[i],
                        n_visual=visual.size(1), compose=False)
        emb, lab = assemble(built, visual[i:i + 1], embed, device)
        seqs.append(emb[0])
        label_seqs.append(lab[0])
        lengths.append(emb.size(1))
        metas.append(built.meta)

    bsz, max_len, dim = len(seqs), max(lengths), visual.size(2)
    inputs_embeds = torch.zeros(bsz, max_len, dim, device=device, dtype=dtype)
    attention_mask = torch.zeros(bsz, max_len, dtype=torch.long, device=device)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long, device=device)
    for i, length in enumerate(lengths):
        inputs_embeds[i, :length] = seqs[i]
        attention_mask[i, :length] = 1
        labels[i, :length] = label_seqs[i]

    # Pass labels so HuggingFace computes its own reduction on THIS forward. The batch
    # scalar it returns is exactly what `vlm_loss_batch` hands the trainer, so the control
    # below compares against the real quantity without a second pass over the data.
    res = llm.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask,
                    labels=labels, use_cache=False)
    logits = res.logits
    # HuggingFace's causal shift: position t predicts token t+1.
    shift_logits, shift_labels = logits[:, :-1, :], labels[:, 1:]

    out = []
    for i in range(bsz):
        keep = shift_labels[i] != -100
        if not keep.any():
            out.append(torch.empty(0, device=device))
            continue
        ce = F.cross_entropy(shift_logits[i][keep].float(), shift_labels[i][keep],
                             reduction="none")
        out.append(ce)
    return out, float(res.loss), metas


def bin_of(n):
    """Answer-length bucket, in supervised target tokens."""
    for hi, name in ((2, "1-2"), (5, "3-5"), (10, "6-10"), (25, "11-25"), (60, "26-60")):
        if n <= hi:
            return name
    return "60+"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="corrected-protocol bridge .pt")
    ap.add_argument("--n-llava", type=int, default=None, help="default: the trainer's own split")
    ap.add_argument("--n-short", type=int, default=None)
    ap.add_argument("--limit", type=int, default=4000,
                    help="examples to decompose (held-out val slice of the training mix)")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--verify-trainer", type=int, default=0, metavar="N",
                    help="also call trainer.eval_loss on the first N examples and require "
                         "agreement; costs a second forward pass over those N")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    device = "cuda"
    trainer = _trainer()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    bcfg = ckpt.get("bridge_config", cfg["bridge"])
    by_id = {mid: nm for nm, mid in cfg["models"]["encoders"].items()}
    enc_name = by_id.get(ckpt.get("encoder")) or cfg["models"].get("active_encoder")

    # Follow the checkpoint's protocol. Decomposing a corrected checkpoint under the
    # legacy prompt would measure a probe, not the objective this bridge was trained on.
    spec = P.PromptSpec(prompt_format=P.format_of(ckpt),
                        supervise_eos=bool(ckpt.get("supervise_eos", False)),
                        system_text=ckpt.get("system_text", P.SYSTEM_TEXT),
                        max_answer_tokens=ckpt.get("max_answer_tokens", 64))
    if spec.prompt_format != "chatml_v1" or not spec.supervise_eos:
        raise SystemExit(f"FAIL not a corrected-layer checkpoint: format={spec.prompt_format} "
                         f"supervise_eos={spec.supervise_eos}")

    # Both loss controls below build their prompts through this same spec and the same
    # P.build, so they agree with each other even if the spec itself is wrong for this
    # checkpoint. Only comparing against what the checkpoint RECORDED can catch that.
    P.assert_compatible(ckpt, spec)
    for field, want in (("short_cue_sha256", P.sha256_text(P.SHORT_CUE)),
                        ("prompt_module_version", P.MODULE_VERSION)):
        stored = ckpt.get(field)
        if stored is not None and stored != want:
            raise SystemExit(
                f"FAIL {field}: checkpoint recorded {stored!r} but this working tree has "
                f"{want!r}. The prompt text has changed since training, so every CE below "
                f"would be measured under a different objective than the one optimised.")
        if stored is None:
            log(f"WARNING checkpoint records no {field}; prompt provenance is unverified")

    log(f"checkpoint {Path(args.checkpoint).name}  encoder={enc_name}  "
        f"bridge={bcfg.get('type')}  protocol={spec.prompt_format}/eos={spec.supervise_eos}")

    # The split comes from the CHECKPOINT, not from config: decomposing the loss over a
    # different mixture than this bridge was trained on would answer a question nobody asked.
    tcfg = cfg["train"]

    def from_ckpt(key):
        """Mixture parameter, from the checkpoint if it recorded one, else from config.

        dict.get's default is evaluated eagerly, so a plain ckpt.get(k, cfg[k]) raises
        for keys config does not carry ('limit' is one). Ask each source in turn instead.
        """
        if ckpt.get(key) is not None:
            return ckpt[key]
        if key in tcfg:
            return tcfg[key]
        raise SystemExit(f"FAIL neither the checkpoint nor config records {key!r}; "
                         f"the decomposition would run over an unknown mixture")

    limit = from_ckpt("limit")
    short_frac = from_ckpt("short_frac")
    val_frac = from_ckpt("val_frac")
    n_short = args.n_short if args.n_short is not None else int(limit * short_frac)
    n_llava = args.n_llava if args.n_llava is not None else limit - n_short
    log(f"mix from checkpoint: limit={limit} short_frac={short_frac} val_frac={val_frac} "
        f"-> {n_llava} llava + {n_short} vqav2")
    examples = trainer.build_examples(cfg, n_short=n_short, n_llava=n_llava)

    # build_examples appends the LLaVA block first, then the VQAv2 block. Label by
    # position, then cross-check against the short-answer cue so a reordering of the
    # builder fails loudly instead of silently swapping the two sources.
    sources = ["llava"] * n_llava + ["vqav2"] * (len(examples) - n_llava)
    cue_ok = sum(1 for (_, q, _), s in zip(examples, sources)
                 if (trainer.SHORT_SUFFIX in q) == (s == "vqav2"))
    if cue_ok != len(examples):
        raise SystemExit(f"FAIL source labelling disagrees with the short-answer cue on "
                         f"{len(examples) - cue_ok} examples — the two sources may be swapped")

    # The trainer holds out validation by CRC32 of the CANONICAL image path, not by a
    # tail slice — a tail slice would be pure VQAv2, since build_examples appends the two
    # sources in blocks. Reuse the trainer's own predicate so this decomposes exactly the
    # set whose loss the training curve reported.
    cut = int(round(val_frac * 100))
    dir_map = trainer.canonical_dir_map(cfg)
    is_val = lambda ex: zlib.crc32(
        trainer.split_key(ex[0], dir_map).encode()) % 100 < cut
    # Order matters as well as membership: the trainer shuffles the FULL list and only then
    # filters, and because eval_loss averages batch token-means, a different order gives
    # different batches and therefore a different scalar. Shuffle first, then filter.
    if ckpt.get("shuffle_pool"):
        raise SystemExit("FAIL checkpoint was trained with --shuffle-pool; this script "
                         "rebuilds the unshuffled locked draw and would use a different pool")
    labelled = list(zip(examples, sources))
    random.Random(ckpt.get("seed", 42)).shuffle(labelled)
    pairs = [(e, s) for e, s in labelled if is_val(e)]
    if not pairs:
        raise SystemExit("FAIL the trainer's val predicate selected nothing")
    by_src = {s: sum(1 for _, t in pairs if t == s) for s in ("llava", "vqav2")}
    if min(by_src.values()) == 0:
        raise SystemExit(f"FAIL validation split is single-source {by_src}; a per-source "
                         f"decomposition over it would be vacuous")
    if args.limit and args.limit < len(pairs):
        # A prefix of the shuffled val list, not a resample: batches stay contiguous so the
        # reported scalar is the trainer's own aggregation over the examples actually seen.
        log(f"NOTE limiting to {args.limit} of {len(pairs)} val examples; the reported scalar "
            f"is over this prefix, not the full epoch validation set")
        pairs = pairs[:args.limit]
    log(f"decomposing {len(pairs)} held-out examples "
        f"({sum(1 for _, s in pairs if s == 'llava')} llava / "
        f"{sum(1 for _, s in pairs if s == 'vqav2')} vqav2)")

    enc = load_vision_encoder(enc_name, cfg, device)
    llm = load_llm(cfg, log=log, precision=ckpt.get("llm_precision") or "4bit")
    bridge = build_bridge(bcfg, enc.hidden_dim, llm.model.config.hidden_size).to(device)
    bridge.load_state_dict(ckpt["state_dict"])
    bridge.eval()

    cache_dir = cache_dir_for(cfg["paths"]["features"], enc_name)

    recs, batch_scalars = [], []
    for start in range(0, len(pairs), args.batch_size):
        batch = pairs[start:start + args.batch_size]
        feats = torch.cat([load_features(cache_dir, p, device) for (p, _, _), _ in batch], dim=0)
        qs = [q for (_, q, _), _ in batch]
        ans = [a for (_, _, a), _ in batch]
        ces, hf_loss, metas = token_ce(bridge, llm, feats, qs, ans, device, spec)
        bi = start // args.batch_size

        # The trainer's own reduction, reproduced from the same per-token values, beside
        # HuggingFace's scalar from the identical forward. These must agree per batch.
        flat = torch.cat([c for c in ces if c.numel()])
        mine = flat.mean().item()
        if abs(mine - hf_loss) > 1e-3:
            raise SystemExit(f"FAIL batch {start}: reconstructed token-mean {mine:.6f} != "
                             f"HuggingFace loss {hf_loss:.6f}; the per-token CE is not the "
                             f"quantity the trainer reduces")
        batch_scalars.append((hf_loss, len(batch), mine))

        # Batch identity is recorded because the trainer's scalar is an example-weighted
        # mean of BATCH token-means. Without n_b and T_b the additive per-source
        # contribution to that scalar cannot be reconstructed afterwards.
        t_b = int(sum(c.numel() for c in ces))
        for (( _, q, a), src), ce, meta in zip(batch, ces, metas):
            n = ce.numel()
            if n == 0:
                continue
            recs.append({
                "source": src,
                "n_target_tokens": n,
                "sum_ce": ce.sum().item(),
                "mean_ce": ce.mean().item(),
                "first_token_ce": ce[0].item(),
                # Only the real stop token is EOS. On a truncated answer P.build appends
                # none, so the last supervised token is ordinary text and labelling it
                # "eos_ce" would average an unrelated quantity into the EOS statistic.
                "eos_ce": ce[-1].item() if meta.get("eos_appended") else None,
                "truncated": bool(meta.get("answer_truncated")),
                "bin": bin_of(n),
                "batch": bi,
                "batch_n": len(batch),
                "batch_tokens": t_b,
            })
        if start % (args.batch_size * 50) == 0:
            log(f"  {start + len(batch)}/{len(pairs)}")

    # ---- the control: does our per-token CE reproduce the trainer's scalar? ----
    # eval_loss() example-weights each batch's token-mean, so reproduce that exactly.
    reported = (sum(mine * n for _, n, mine in batch_scalars)
                / sum(n for _, n, _ in batch_scalars))
    hf_agg = (sum(s * n for s, n, _ in batch_scalars)
              / sum(n for _, n, _ in batch_scalars))
    per_batch_max = max(abs(s - mine) for s, _, mine in batch_scalars)
    delta = abs(reported - hf_agg)
    log(f"CONTROL eval_loss-equivalent={hf_agg:.6f}  reconstructed={reported:.6f}  "
        f"|delta|={delta:.2e}  worst per-batch={per_batch_max:.2e}")
    if delta > 1e-4:
        raise SystemExit("FAIL per-token CE does not reproduce eval_loss; the decomposition "
                         "would describe a different quantity than the trainer optimised")

    # Independent confirmation on a small slice that the trainer's own function, called
    # through its own code path, returns the same number. This is the part that would
    # catch a prompt-spec or batching mismatch that the identity above cannot see.
    truth = None
    if args.verify_trainer:
        torch.cuda.empty_cache()
        k = args.verify_trainer - (args.verify_trainer % args.batch_size)
        truth = trainer.eval_loss(bridge, llm, cache_dir, [e for e, _ in pairs[:k]], device,
                                  args.batch_size, spec.max_answer_tokens, spec=spec)
        mine_k = (sum(mine * n for _, n, mine in batch_scalars[:k // args.batch_size])
                  / sum(n for _, n, _ in batch_scalars[:k // args.batch_size]))
        log(f"CONTROL trainer.eval_loss(n={k})={truth:.6f}  ours={mine_k:.6f}  "
            f"|delta|={abs(truth - mine_k):.2e}")
        if abs(truth - mine_k) > 1e-4:
            raise SystemExit("FAIL trainer.eval_loss disagrees with the reconstruction")

    tot_ce = sum(r["sum_ce"] for r in recs)
    tot_tok = sum(r["n_target_tokens"] for r in recs)

    # The trainer's scalar is sum_b n_b * (sum_{i in b} CE_i / T_b) / sum_b n_b. That IS
    # additive over any partition of examples, so each subset has a contribution in the
    # scalar's own units which sum to the scalar exactly. This is a different quantity
    # from the global token-loss share below, and only this one decomposes the objective.
    denom_n = sum(bn for _, bn, _ in batch_scalars)

    def additive_contribution(sel):
        """This subset's additive share of the trainer-reported scalar, in loss units."""
        per_batch = {}
        for r in sel:
            per_batch.setdefault(r["batch"], 0.0)
            per_batch[r["batch"]] += r["sum_ce"]
        total = 0.0
        for b, ce_sum in per_batch.items():
            n_b = recs_by_batch[b]["batch_n"]
            t_b = recs_by_batch[b]["batch_tokens"]
            total += n_b * (ce_sum / t_b)
        return total / denom_n

    recs_by_batch = {r["batch"]: r for r in recs}

    def share(pred):
        sel = [r for r in recs if pred(r)]
        contrib = additive_contribution(sel)
        return {
            "n_examples": len(sel),
            "token_share_pct": 100 * sum(r["n_target_tokens"] for r in sel) / tot_tok,
            # Share of TOTAL CE mass. Descriptive. NOT a decomposition of the trainer
            # scalar and NOT gradient share: no batch denominators, no parameter Jacobians,
            # no cancellation between examples.
            "global_token_loss_share_pct": 100 * sum(r["sum_ce"] for r in sel) / tot_ce,
            # In the trainer scalar's own units; these sum across a partition to the scalar.
            "trainer_additive_contribution": contrib,
            "trainer_additive_share_pct": 100 * contrib / reported if reported else None,
            "token_mean_ce": (sum(r["sum_ce"] for r in sel)
                              / max(1, sum(r["n_target_tokens"] for r in sel))),
            "per_example_mean_ce": (sum(r["mean_ce"] for r in sel) / len(sel)) if sel else None,
            "first_token_ce": (sum(r["first_token_ce"] for r in sel) / len(sel)) if sel else None,
            # eos_ce is None on truncated answers, where P.build appends no stop token.
            # Averaging over only the examples that HAVE one, and reporting how many that
            # was, keeps the statistic about EOS instead of silently mixing in text tokens.
            "eos_ce": (sum(eos_vals) / len(eos_vals)) if (eos_vals := [
                r["eos_ce"] for r in sel if r["eos_ce"] is not None]) else None,
            "n_with_eos": len(eos_vals),
            "n_truncated": sum(1 for r in sel if r["truncated"]),
        }

    out = {
        "checkpoint": Path(args.checkpoint).name,
        "encoder": enc_name,
        "bridge_type": bcfg.get("type"),
        "epoch": ckpt.get("epoch"),
        "seed": ckpt.get("seed"),
        "n_examples": len(recs),
        "control": {"eval_loss_equivalent": hf_agg, "reconstructed": reported,
                    "abs_delta": delta, "worst_per_batch": per_batch_max,
                    "trainer_eval_loss": truth},
        "aggregate": {
            "token_mean_ce": tot_ce / tot_tok,
            "per_example_mean_ce": sum(r["mean_ce"] for r in recs) / len(recs),
            "trainer_reported": hf_agg,
        },
        "by_source": {s: share(lambda r, s=s: r["source"] == s) for s in ("llava", "vqav2")},
        "estimand_note": (
            "trainer_additive_contribution decomposes the trainer-reported scalar and sums "
            "to it across a partition. global_token_loss_share_pct is total-CE mass and does "
            "NOT decompose that scalar. Neither is gradient-norm share: parameter Jacobians "
            "and cancellation between examples are unmeasured."),
        "by_length_bin": {b: share(lambda r, b=b: r["bin"] == b)
                          for b in ("1-2", "3-5", "6-10", "11-25", "26-60", "60+")},
        "by_truncation": {str(t): share(lambda r, t=t: r["truncated"] == t)
                          for t in (True, False)},
    }
    # The additive contributions must sum to the trainer scalar. If they do not, the
    # decomposition is not a decomposition and the per-source lines below are decoration.
    parts = sum(out["by_source"][s]["trainer_additive_contribution"] for s in ("llava", "vqav2"))
    if abs(parts - reported) > 1e-6:
        raise SystemExit(f"FAIL per-source contributions sum to {parts:.8f} but the trainer "
                         f"scalar is {reported:.8f}; this is not an additive decomposition")
    log(f"CONTROL additive parts sum to {parts:.6f} vs scalar {reported:.6f} "
        f"(|delta|={abs(parts - reported):.2e})")

    dest = Path(args.out or ROOT / "outputs" / "analysis" /
                f"nll_decomp__{Path(args.checkpoint).stem}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))

    log("")
    log("SOURCE DECOMPOSITION — descriptive. None of these columns is gradient share.")
    for s in ("llava", "vqav2"):
        d = out["by_source"][s]
        if not d["n_examples"]:
            log(f"  {s:6s} absent from this slice")
            continue
        log(f"  {s:6s} n={d['n_examples']:5d}  tokens={d['token_share_pct']:6.2f}%   "
            f"CEmass={d['global_token_loss_share_pct']:6.2f}%   "
            f"objective={d['trainer_additive_share_pct']:6.2f}%   "
            f"tok-mean CE={d['token_mean_ce']:.4f}  per-ex CE={d['per_example_mean_ce']:.4f}")
    log("  tokens=share of supervised tokens | CEmass=share of total CE (does NOT decompose "
        "the objective) | objective=additive share of the trainer scalar")
    log(f"wrote {dest}")


if __name__ == "__main__":
    main()
