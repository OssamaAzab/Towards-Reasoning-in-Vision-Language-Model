"""Stage B: train the crop-evidence adapter. ONE trainable module, everything else frozen.

    python scripts/62_train_crop_adapter.py --coverage outputs/crop_adapter/train_coverage.json \
        --out-dir outputs/crop_adapter/run_<jobid> [--limit 16 --overfit]

EXPLORATORY — SUPERVISED GQA ORACLE-REGION ADAPTER — NOT ZERO-SHOT — NOT DEPLOYABLE.

WHAT TRAINS: the crop projector, the learned evidence queries, one cross-attention block, and the
question-conditioned gate. WHAT DOES NOT: CLIP, the MLP256 global connector, and Qwen2. That is
asserted before the first step (requires_grad) and again after the first backward (.grad is None),
because those are different failure modes — a module can be frozen at setup and unfrozen later.

THE TRAINING MIXTURE IS THE HYPOTHESIS. Every example is shown with one of:

  correct     the oracle crop of an object the question names
  wrong       a same-label crop from a different image (the Stage A control, as training signal)
  irrelevant  a same-image, area-matched crop overlapping no referent, where one exists
  none        evidence dropout: no crop tokens at all

The target is the gold answer in EVERY case. There is no ranking loss and no auxiliary objective:
the model is simply asked to answer correctly whether the evidence helps, misleads or is absent,
which is what "learn to use evidence selectively" means operationally. A ranking loss would tell
the gate directly which crop is correct, which is the answer to the question being asked.

WHY THE ORDINARY ANSWER LOSS AND NOTHING ELSE. Adding a term is a decision that needs its own
evidence. If the pilot shows the gate cannot separate correct from wrong evidence under the plain
loss, THAT is the documented reason to try a ranking term — and it would be a new experiment, not
a silent addition to this one.

TRAINING DATA IS THE GQA TRAIN SPLIT, verified image-disjoint from the val split and from every
slice this project has spent. **An adapter trained here is not zero-shot with respect to GQA.**
The frozen baseline it is compared against is. Every artifact says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.prompt as P  # noqa: E402
from src.data.crop_augment import make_crop  # noqa: E402
from src.data.gqa import GQADataset  # noqa: E402
from src.models.connectors import build_bridge  # noqa: E402
from src.models.crop_adapter import (CropEvidenceAdapter, assert_gradients_only_in_adapter,  # noqa: E402
                                     assert_only_adapter_trainable, evidence_visual_tokens,
                                     gate_stats, question_embeddings)
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import load_llm  # noqa: E402
from src.models.vlm import assemble  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

LABEL = "EXPLORATORY — SUPERVISED GQA ORACLE-REGION ADAPTER — NOT ZERO-SHOT — NOT DEPLOYABLE"
ENCODER, BRIDGE_STEM = "clip", "bridge_clip_w1_bf16_v1_ep5"
EVIDENCE_KINDS = ("correct", "wrong", "irrelevant", "null_visual")
# Evidence dropout is the "null_visual" share, and it is the NULL-VISUAL CONDITION, not a disabled
# adapter: the crop features are replaced by exact zeros while the learned queries, the question
# conditioning, the gate and all evidence-token positions stay active. Dropping the whole adapter
# instead would leave the null-visual EVALUATION arm out of distribution, and that arm is the one
# control that separates "used the visual evidence" from "learned GQA answer priors through the
# trainable question pathway". The mixture is fixed here rather than tuned: this is a one-seed
# screening pilot, and a swept mixture would need its own selection protocol.
DEFAULT_MIX = {"correct": 0.5, "wrong": 0.25, "irrelevant": 0.10, "null_visual": 0.15}


def sha256_file(p: Path) -> str:
    """Hash a file so the run pins exactly what it read."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def param_hash(module) -> str:
    """Deterministic sha256 over a module's parameters, in sorted name order."""
    h = hashlib.sha256()
    for name, p in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().to("cpu", torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()


def pick_evidence(rng, rec, mix):
    """Sample this example's evidence kind, falling back only when the crop does not exist."""
    kinds, weights = zip(*[(k, mix[k]) for k in EVIDENCE_KINDS])
    kind = rng.choices(kinds, weights=weights, k=1)[0]
    if kind == "irrelevant" and not rec.get("has_irrelevant"):
        return "correct"          # no irrelevant crop for this question; do not invent one
    return kind


def crop_box_for(kind, rec):
    """(box, source image, null_visual) for this evidence kind.

    `null_visual` returns no box AND the null flag set: the adapter still runs, on all-zero crop
    features. There is no training kind that removes the adapter entirely — the no-adapter-span
    arm is a pure parity control against the frozen baseline and is never trained toward.
    """
    if kind == "null_visual":
        return None, None, True
    if kind == "correct":
        return rec["relevant"]["box"], rec["image_id"], False
    if kind == "wrong":
        return rec["wrong_box"], rec["wrong_donor_image_id"], False
    return rec["irrelevant"]["box"], rec["image_id"], False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", default="outputs/crop_adapter/train_coverage.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-evidence", type=int, default=8)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="smoke: train on the first N questions")
    ap.add_argument("--overfit", action="store_true",
                    help="smoke: reuse the training questions as validation and expect the loss "
                         "to collapse; never used for a scientific run")
    ap.add_argument("--val-every", type=int, default=0, help="steps between validation passes")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("FAIL a CUDA GPU is required. CPU is not numerically interchangeable "
                         "with cluster GPU results in this project and may not be used for "
                         "training, answers, EOS behaviour or any stop/go decision.")
    device = "cuda"
    cov_p, out_dir = ROOT / args.coverage, ROOT / args.out_dir
    if not cov_p.is_file():
        raise SystemExit(f"FAIL coverage {cov_p} absent; run scripts/61 first")
    cov = json.loads(cov_p.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "adapter.pt"
    if ckpt_path.exists():
        raise SystemExit(f"FAIL refusing to overwrite {ckpt_path}")

    set_seed(args.seed)
    cfg = load_config()
    recs = cov["records"]
    train_qids = list(cov["train_qids"])
    val_qids = list(cov["val_qids"])
    if args.limit:
        train_qids = train_qids[: args.limit]
        val_qids = train_qids if args.overfit else val_qids[: max(1, args.limit)]
    if not args.overfit and set(train_qids) & set(val_qids):
        raise SystemExit("FAIL train and validation questions overlap")

    print(f"=== Stage B — {LABEL} ===")
    print(f"train {len(train_qids)} / val {len(val_qids)} questions"
          f"{'   *** OVERFIT SMOKE ***' if args.overfit else ''}")
    print(f"NOT ZERO-SHOT ON GQA: {cov['zero_shot_caveat']}")

    # The TRAIN questions file, not the val one the rest of the project evaluates on. GQA image
    # ids are Visual Genome ids, so the same image directories resolve both splits.
    ds = GQADataset(str(ROOT / cov["questions_file"]), cfg["gqa"]["image_dirs"])
    llm = load_llm(cfg, precision="bf16")
    cp = ROOT / "outputs" / "checkpoints" / "corrected" / f"{BRIDGE_STEM}.pt"
    if not cp.is_file():
        raise SystemExit(f"FAIL checkpoint {cp} absent")
    ck = torch.load(cp, map_location=device, weights_only=False)
    if ck["bridge_config"]["type"] != "mlp":
        raise SystemExit(f"FAIL {BRIDGE_STEM} is a {ck['bridge_config']['type']} bridge, not MLP256")

    enc = load_vision_encoder(ENCODER, cfg, device)
    bridge = build_bridge(ck["bridge_config"], enc.hidden_dim, llm.model.config.hidden_size)
    bridge.load_state_dict(ck["state_dict"])
    bridge.to(device).eval().requires_grad_(False)
    enc.model.eval().requires_grad_(False)
    llm.model.eval().requires_grad_(False)

    adapter = CropEvidenceAdapter(enc.hidden_dim, llm.model.config.hidden_size,
                                  n_evidence=args.n_evidence, width=args.width).to(device)
    adapter.train()
    assert_only_adapter_trainable(adapter, bridge, enc.model, llm.model)
    n_train_params = adapter.trainable_parameters()
    print(f"\nadapter: {json.dumps(adapter.config())}")
    print(f"TRAINABLE PARAMETERS: {n_train_params:,}  (everything else is frozen)")

    spec = P.PromptSpec(prompt_format=P.format_of(ck),
                        supervise_eos=bool(ck.get("supervise_eos", False)),
                        system_text=ck.get("system_text", P.SYSTEM_TEXT),
                        max_answer_tokens=32)
    P.assert_compatible(ck, spec)
    probe = "What colour is the dog?"
    if P.user_body(spec, probe) != probe + P.SHORT_CUE:
        raise SystemExit("FAIL prompt composition disagrees with scripts/07")

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, math.ceil(len(train_qids) * args.epochs / (args.batch_size * args.accum)))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    embed = llm.model.get_input_embeddings()
    rng = random.Random(args.seed)
    from PIL import Image

    def one_example(q, kind):
        """Loss for one (question, evidence-kind) pair. Frozen views under no_grad."""
        rec = recs[q]
        ex = ds.get(q)
        box, img_id, null_visual = crop_box_for(kind, rec)
        with torch.no_grad():
            g = encode_image(enc, Image.open(ex.image_path).convert("RGB"))
            crop = None
            if box is not None:
                src = ex.image_path if img_id == rec["image_id"] else ds.get(
                    rec["wrong_donor_qid"]).image_path
                crop = encode_image(enc, make_crop(Image.open(src).convert("RGB"), box))
        qe = question_embeddings(llm, ex.question, device)
        visual, gate = evidence_visual_tokens(bridge, adapter, g.to(device),
                                              None if crop is None else crop.to(device), qe,
                                              null_visual=null_visual)
        built = P.build(spec, P.user_body(spec, ex.question), llm.tokenizer, answer=ex.answer,
                        n_visual=visual.size(1), compose=False)
        inputs_embeds, labels = assemble(built, visual, embed, device)
        loss = llm.model(inputs_embeds=inputs_embeds, labels=labels, use_cache=False).loss
        return loss, gate

    def validate(qids):
        adapter.eval()
        tot, n, gates = 0.0, 0, []
        with torch.no_grad():
            for q in qids:
                loss, gate = one_example(q, "correct")
                tot += float(loss)
                n += 1
                if gate is not None:
                    gates.append(gate)
        adapter.train()
        return (tot / max(1, n)), gate_stats(gates)

    history, gates_seen, step, t0 = [], [], 0, time.time()
    checked_grads = False
    for epoch in range(args.epochs):
        order = list(train_qids)
        rng.shuffle(order)
        opt.zero_grad(set_to_none=True)
        run_loss, run_n, kinds = 0.0, 0, {k: 0 for k in EVIDENCE_KINDS}
        for i, q in enumerate(order, 1):
            kind = pick_evidence(rng, recs[q], DEFAULT_MIX)
            kinds[kind] += 1
            loss, gate = one_example(q, kind)
            (loss / (args.batch_size * args.accum)).backward()
            if gate is not None:
                gates_seen.append(gate)
            run_loss += float(loss)
            run_n += 1
            if not checked_grads:
                # After the FIRST backward, not at setup: this catches a module unfrozen mid-run.
                assert_gradients_only_in_adapter(adapter, bridge, enc.model, llm.model)
                checked_grads = True
                print("PASS gradients exist in the adapter and nowhere else")
            if i % (args.batch_size * args.accum) == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or i == len(order):
                    gs = gate_stats(gates_seen[-256:])
                    print(f"  epoch {epoch + 1} step {step}/{total_steps} "
                          f"loss {run_loss / max(1, run_n):.4f} "
                          f"gate mean {gs.get('mean', float('nan')):.3f} "
                          f"[{gs.get('min', float('nan')):.3f}, {gs.get('max', float('nan')):.3f}] "
                          f"lr {sched.get_last_lr()[0]:.2e}  ({time.time() - t0:.0f}s)", flush=True)
                    history.append({"epoch": epoch + 1, "step": step,
                                    "train_loss": run_loss / max(1, run_n), "gate": gs,
                                    "lr": sched.get_last_lr()[0]})
                    run_loss, run_n = 0.0, 0
                if args.val_every and step % args.val_every == 0:
                    vl, vg = validate(val_qids)
                    print(f"  [validation] step {step} loss {vl:.4f} gate {vg}", flush=True)
                    history.append({"step": step, "val_loss": vl, "val_gate": vg})
        print(f"  epoch {epoch + 1} evidence mixture actually sampled: {kinds}")

    val_loss, val_gate = validate(val_qids)
    final_gate = gate_stats(gates_seen)
    print(f"\nfinal validation loss {val_loss:.4f}")
    print(f"gate over training: {json.dumps(final_gate)}")
    if not final_gate.get("all_finite", False):
        raise SystemExit("FAIL the gate produced non-finite values")
    if final_gate.get("frac_below_0.01", 0) == 1.0 or final_gate.get("frac_above_0.99", 0) == 1.0:
        raise SystemExit("FAIL the gate is permanently saturated; it learned nothing selective")

    assert_only_adapter_trainable(adapter, bridge, enc.model, llm.model)
    payload = {
        "label": LABEL, "oracle": True,
        "not_zero_shot_on_gqa": cov["zero_shot_caveat"],
        "state_dict": {k: v.cpu() for k, v in adapter.state_dict().items()},
        "adapter_config": adapter.config(),
        "trainable_parameters": n_train_params,
        "seed": args.seed, "lr": args.lr, "epochs": args.epochs,
        "batch_size": args.batch_size, "accum": args.accum,
        "effective_batch": args.batch_size * args.accum,
        "n_evidence": args.n_evidence, "evidence_mixture": DEFAULT_MIX,
        "evidence_dropout_is_null_visual": True,
        "encoder": cfg["models"]["encoders"][ENCODER], "bridge_stem": BRIDGE_STEM,
        "bridge_sha256": sha256_file(cp), "encoder_param_sha256": param_hash(enc.model),
        "llm": cfg["models"]["llm"], "prompt_format": spec.prompt_format,
        "supervise_eos": spec.supervise_eos,
        "coverage": args.coverage, "coverage_sha256": sha256_file(cov_p),
        "questions_sha256": cov["questions_sha256"], "graphs_sha256": cov["graphs_sha256"],
        "n_train": len(train_qids), "n_val": len(val_qids),
        "overfit_smoke": bool(args.overfit), "smoke": bool(args.limit),
        "history": history, "final_val_loss": val_loss, "gate_stats": final_gate,
        "models_frozen": ["clip", "mlp256_bridge", "qwen2"],
    }
    torch.save(payload, ckpt_path)
    print(f"\nwrote {ckpt_path.relative_to(ROOT)}  sha256 {sha256_file(ckpt_path)[:16]}…")
    print(f"\n{LABEL}")
    print("Stage B results are NOT zero-shot on GQA; the frozen baseline they are compared "
          "against is. `adapter correct - frozen baseline` combines GQA supervision, new "
          "parameters, extra token positions AND crop evidence, and is not a clean crop effect.")


if __name__ == "__main__":
    main()
