"""Stage B: seven evidence arms, ONE adapter checkpoint, on the untouched development cohort.

    python scripts/63_crop_adapter_eval.py --adapter outputs/crop_adapter/run_<id>/adapter.pt \
        --coverage outputs/crop_augment/coverage_v2.json --out-dir outputs/crop_adapter/eval_<id>

EXPLORATORY — SUPERVISED GQA ORACLE-REGION ADAPTER — NOT ZERO-SHOT — NOT DEPLOYABLE.

THE SEVEN ARMS. Arm 1 is the original frozen stack with no adapter in the graph at all. Arms 2-7
all use the SAME trained checkpoint — the script loads it once, hashes its parameters, samples the
hash inside every arm, and re-hashes at the end, so "one checkpoint answered every arm" is
verified downstream rather than asserted here.

  1 baseline_frozen          256 global tokens, no adapter. The established CLIP+MLP256 path.
  2 adapter_correct_crop     global + gated evidence from the oracle relevant crop
  3 adapter_wrong_crop       global + gated evidence from a same-label different-image crop
  4 adapter_irrelevant_crop  global + gated evidence from a same-image non-referent crop
  5 adapter_null_visual      global + evidence built from ALL-ZERO crop features; learned queries,
                             question conditioning, gate and all evidence positions stay ACTIVE
  6 adapter_zero_gate        global + evidence positions whose content the gate forces to zero
  7 no_adapter_span          global only; the adapter is not called and no positions are appended

WHY ARM 5 IS THE ONE THAT MATTERS. `q_proj` is trainable and reads the question, so the adapter
can learn GQA answer priors and emit them through the evidence tokens WITHOUT USING THE CROP.
Arm 5 gives it exactly that opportunity and nothing else: identical positions, identical question
pathway, identical gate, zero visual information. If correct evidence does not beat arm 5, any
gain over the frozen baseline is supervision and question priors, not crop evidence. Training uses
this same condition for evidence dropout, so arm 5 is in distribution rather than a test-time
novelty.

ARMS 5, 6 AND 7 ARE THREE DIFFERENT CONTROLS. Arm 5 removes the visual information but keeps
everything else; arm 6 keeps the visual information but blocks the gate; arm 7 removes the token
positions entirely and is therefore byte-for-byte arm 1's sequence. Arm 7 is additionally a parity
check: it must reproduce arm 1 exactly, and the script fails if it does not.

NOT ZERO-SHOT, AND `correct - baseline` IS NOT A CLEAN CROP EFFECT. The adapter was trained on the
GQA train split; arm 1 never saw GQA. That contrast therefore combines GQA supervision, new
trainable parameters, extra token positions and crop evidence, and it is not described as an
upper bound on any one of them. The evidence-specific contrasts are correct-vs-wrong,
correct-vs-irrelevant and correct-vs-null-visual.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.prompt as P  # noqa: E402
from src.data.crop_augment import make_crop  # noqa: E402
from src.data.gqa import GQADataset  # noqa: E402
from src.eval.metrics import METRIC_VERSION, exact_full  # noqa: E402
from src.models.connectors import build_bridge  # noqa: E402
from src.models.crop_adapter import (CropEvidenceAdapter, evidence_visual_tokens,  # noqa: E402
                                     gate_stats, question_embeddings)
from src.models.encoders import encode_image, load_vision_encoder  # noqa: E402
from src.models.llm import load_llm  # noqa: E402
from src.models.multi_view import assert_frozen  # noqa: E402
from src.models.vlm import _generate, assemble, vlm_generate  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402

SEED = 42
LABEL = "EXPLORATORY — SUPERVISED GQA ORACLE-REGION ADAPTER — NOT ZERO-SHOT — NOT DEPLOYABLE"
ENCODER, BRIDGE_STEM = "clip", "bridge_clip_w1_bf16_v1_ep5"
SPENT = ("eval_2000", "locked", "confirm_3000", "objective_4000")
ARMS = ("baseline_frozen", "adapter_correct_crop", "adapter_wrong_crop",
        "adapter_irrelevant_crop", "adapter_null_visual", "adapter_zero_gate", "no_adapter_span")
# Arms that append evidence-token positions. Every one of them must carry 256 + n_evidence visual
# tokens; only `no_adapter_span` may carry 256.
SPAN_ARMS = ("adapter_correct_crop", "adapter_wrong_crop", "adapter_irrelevant_crop",
             "adapter_null_visual", "adapter_zero_gate")


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


@torch.no_grad()
def generate_with_visual(llm, visual, prompt, device, max_new_tokens, spec):
    """Greedy answer from a prepared visual span, through the established decoding path."""
    embed = llm.model.get_input_embeddings()
    built = P.build(spec, prompt, llm.tokenizer, n_visual=visual.size(1), compose=False)
    inputs_embeds, _ = assemble(built, visual, embed, device)
    return _generate(llm, inputs_embeds, max_new_tokens, device, spec, return_stop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--coverage", default="outputs/crop_augment/coverage_v2.json")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("FAIL a CUDA GPU is required. CPU inference is not numerically "
                         "interchangeable with cluster GPU results in this project and may not "
                         "be used for answers, scores, EOS behaviour or any stop/go decision.")
    device = "cuda"
    cov_p, ad_p, out_dir = ROOT / args.coverage, ROOT / args.adapter, ROOT / args.out_dir
    for p in (cov_p, ad_p):
        if not p.is_file():
            raise SystemExit(f"FAIL {p} absent")
    cov = json.loads(cov_p.read_text())
    for bad in SPENT:
        if bad in cov["slice"]:
            raise SystemExit(f"FAIL the evaluation coverage was built on {cov['slice']!r}, "
                             f"a spent endpoint")
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "adapter_records.json"
    if rec_path.exists():
        raise SystemExit(f"FAIL refusing to overwrite {rec_path}")

    qids = list(cov["core_qids"])
    subset = set(cov["irrelevant_subset_qids"])
    scale_matched = set(cov["scale_matched_qids"])
    meta_recs = cov["records"]
    if args.limit:
        head = qids[: args.limit]
        if not any(q in subset for q in head):
            extra = [q for q in qids if q in subset][: max(1, args.limit // 4)]
            head = head[: args.limit - len(extra)] + extra
        qids = head
    subset &= set(qids)
    scale_matched &= set(qids)

    set_seed(SEED)
    cfg = load_config()
    print(f"=== Stage B seven-arm evaluation — {LABEL} ===")
    print(f"questions : {len(qids)}{'  (SMOKE)' if args.limit else ''}   "
          f"irrelevant_subset {len(subset)}")
    print(f"arms      : {list(ARMS)}")

    ds = GQADataset(cfg["gqa"]["questions"]["val_balanced"], cfg["gqa"]["image_dirs"])
    llm = load_llm(cfg, precision="bf16")
    cp = ROOT / "outputs" / "checkpoints" / "corrected" / f"{BRIDGE_STEM}.pt"
    ck = torch.load(cp, map_location=device, weights_only=False)
    enc = load_vision_encoder(ENCODER, cfg, device)
    bridge = build_bridge(ck["bridge_config"], enc.hidden_dim, llm.model.config.hidden_size)
    bridge.load_state_dict(ck["state_dict"])
    bridge.to(device).eval().requires_grad_(False)
    enc.model.eval().requires_grad_(False)
    llm.model.eval().requires_grad_(False)
    assert_frozen(bridge, enc.model, llm.model)

    blob = torch.load(ad_p, map_location=device, weights_only=False)
    acfg = blob["adapter_config"]
    adapter = CropEvidenceAdapter(acfg["vis_dim"], acfg["llm_dim"], n_evidence=acfg["n_evidence"],
                                  width=acfg["width"], n_heads=acfg["n_heads"]).to(device)
    adapter.load_state_dict(blob["state_dict"])
    adapter.eval().requires_grad_(False)
    adapter_hash_before = param_hash(adapter)
    print(f"\nadapter  : {ad_p.name}  file sha256 {sha256_file(ad_p)[:12]}…")
    print(f"           parameter sha256 {adapter_hash_before[:12]}…  "
          f"{blob['trainable_parameters']:,} trained parameters")
    print(f"           trained on {blob['n_train']} GQA-TRAIN questions, seed {blob['seed']}")
    print(f"NOT ZERO-SHOT ON GQA: {blob['not_zero_shot_on_gqa']}")
    if blob.get("overfit_smoke"):
        print("WARNING this checkpoint came from an OVERFIT SMOKE and carries no scientific value")

    spec = P.PromptSpec(prompt_format=P.format_of(ck),
                        supervise_eos=bool(ck.get("supervise_eos", False)),
                        system_text=ck.get("system_text", P.SYSTEM_TEXT), max_answer_tokens=None)
    P.assert_compatible(ck, spec)
    max_new = 32 if spec.supervise_eos else 10
    probe = "What colour is the dog?"
    if P.user_body(spec, probe) != probe + P.SHORT_CUE:
        raise SystemExit("FAIL prompt composition disagrees with scripts/07")

    from PIL import Image
    records, gates, arm_hashes, t0 = {}, collections.defaultdict(list), {}, time.time()
    for i, q in enumerate(qids, 1):
        ex, m = ds.get(q), meta_recs[q]
        prompt = P.user_body(spec, ex.question)
        qe = question_embeddings(llm, ex.question, device)
        img = Image.open(ex.image_path).convert("RGB")
        with torch.no_grad():
            g = encode_image(enc, img).to(device)
            crops = {"adapter_correct_crop": encode_image(
                enc, make_crop(img, m["relevant"]["box"])).to(device)}
            d = ds.get(m["wrong_donor_qid"])
            crops["adapter_wrong_crop"] = encode_image(
                enc, make_crop(Image.open(d.image_path).convert("RGB"), m["wrong_box"])).to(device)
            if q in subset:
                crops["adapter_irrelevant_crop"] = encode_image(
                    enc, make_crop(img, m["irrelevant"]["box"])).to(device)

        r = {"qid": q, "image_id": m["image_id"], "question": ex.question, "gold": ex.answer,
             "category": ds.category_of(ex), "structural": ex.structural,
             "target_label": m["target_label"], "wrong_donor_qid": m["wrong_donor_qid"],
             "wrong_donor_image_id": m["wrong_donor_image_id"],
             "wrong_donor_label": m["wrong_donor_label"],
             "in_irrelevant_subset": q in subset, "in_scale_matched": q in scale_matched,
             "scale": m["scale"]}

        # arm 1: the original frozen stack, through the ESTABLISHED path, adapter not in the graph
        b_ans, b_stop = vlm_generate(bridge, llm, g, prompt, device, max_new_tokens=max_new,
                                     spec=spec, return_stop=True)
        r["baseline_frozen__ans"] = b_ans
        r["baseline_frozen__exact_full"] = bool(exact_full(b_ans, ex.answer))
        r["baseline_frozen__stop_reason"] = b_stop["stop_reason"]
        r["baseline_frozen__n_generated"] = b_stop["n_generated"]

        for arm in ARMS[1:]:
            if arm == "adapter_irrelevant_crop" and q not in subset:
                continue
            # Written out rather than derived, because what each arm removes IS the experiment.
            # zero_gate deliberately reuses the CORRECT crop: it asks what happens when the right
            # evidence is present but the gate refuses to admit it. null_visual keeps the gate
            # LEARNED and only zeroes the visual features — forcing the gate there instead would
            # conflate "no visual information" with "no evidence admitted".
            null = False
            if arm == "no_adapter_span":
                crop, mode = None, "learned"
            elif arm == "adapter_zero_gate":
                crop, mode = crops["adapter_correct_crop"], "off"
            elif arm == "adapter_null_visual":
                crop, mode, null = crops["adapter_correct_crop"], "learned", True
            else:
                crop, mode = crops[arm], "learned"
            with torch.no_grad():
                visual, gate = evidence_visual_tokens(bridge, adapter, g, crop, qe,
                                                      gate_mode=mode, null_visual=null)
            ans, stop = generate_with_visual(llm, visual, prompt, device, max_new, spec)
            r[f"{arm}__ans"] = ans
            r[f"{arm}__exact_full"] = bool(exact_full(ans, ex.answer))
            r[f"{arm}__stop_reason"] = stop["stop_reason"]
            r[f"{arm}__n_generated"] = stop["n_generated"]
            r[f"{arm}__n_visual"] = int(visual.size(1))
            if gate is not None:
                gates[arm].append(gate)
            arm_hashes.setdefault(arm, param_hash(adapter))
        records[q] = r
        if i % 25 == 0 or i == len(qids):
            print(f"  {i}/{len(qids)}  ({time.time() - t0:.0f}s)", flush=True)

    # ---- one checkpoint answered every arm ----
    adapter_hash_after = param_hash(adapter)
    if adapter_hash_after != adapter_hash_before or set(arm_hashes.values()) != {adapter_hash_before}:
        raise SystemExit(f"FAIL the adapter weights changed during evaluation: "
                         f"{adapter_hash_before[:12]} -> {sorted({*arm_hashes.values(), adapter_hash_after})}")
    print(f"\nPASS one adapter checkpoint answered every arm "
          f"(parameter sha256 {adapter_hash_before[:12]}… before, during and after)")

    # ---- arm 6 must reproduce arm 1 exactly: it is the same sequence ----
    mism = [q for q in qids
            if records[q]["no_adapter_span__ans"] != records[q]["baseline_frozen__ans"]
            or records[q]["no_adapter_span__stop_reason"]
            != records[q]["baseline_frozen__stop_reason"]]
    if mism:
        raise SystemExit(f"FAIL no_adapter_span differs from the frozen baseline on "
                         f"{len(mism)} questions; the global pathway is not intact")
    print(f"PASS no_adapter_span reproduces the frozen baseline exactly on all {len(qids)} "
          f"questions (the global pathway is untouched)")

    n_ev = acfg["n_evidence"]
    for a in SPAN_ARMS:
        w = {r_[f"{a}__n_visual"] for r_ in records.values() if f"{a}__n_visual" in r_}
        if w != {256 + n_ev}:
            raise SystemExit(f"FAIL {a} carries {w} visual tokens, not {256 + n_ev}")
    span_w = {r_["no_adapter_span__n_visual"] for r_ in records.values()}
    if span_w != {256}:
        raise SystemExit(f"FAIL no_adapter_span carries {span_w} visual tokens, not 256")
    print(f"PASS all {len(SPAN_ARMS)} span arms carry 256+{n_ev} visual tokens; "
          f"no_adapter_span carries 256")

    # The null-visual arm must have had a LIVE gate: forcing it to zero there would conflate
    # "no visual information" with "no evidence admitted", which are different controls.
    nv = gate_stats(gates.get("adapter_null_visual", []))
    if nv.get("n", 0) == 0:
        raise SystemExit("FAIL the null-visual arm recorded no gate values; its gate was not live")
    if nv.get("max", 0.0) <= 0.0:
        raise SystemExit(f"FAIL the null-visual arm's gate was identically zero {nv}; it must "
                         f"stay learned so the arm isolates the ABSENCE OF VISUAL INFORMATION")
    print(f"PASS the null-visual arm ran with a live learned gate (mean {nv['mean']:.3f}, "
          f"max {nv['max']:.3f})")

    meta = {
        "experiment": "Stage B — trainable crop-evidence adapter, seven arms",
        "label": LABEL, "oracle": True, "smoke_test": bool(args.limit),
        "not_zero_shot_on_gqa": blob["not_zero_shot_on_gqa"],
        "metric_version": METRIC_VERSION, "seed": SEED,
        "arms": list(ARMS), "span_arms": list(SPAN_ARMS), "n_questions": len(qids),
        "cohorts": {"core": sorted(qids), "irrelevant_subset": sorted(subset),
                    "scale_matched": sorted(scale_matched)},
        "n_core": len(qids), "n_irrelevant_subset": len(subset),
        "n_scale_matched": len(scale_matched),
        "adapter_file": args.adapter, "adapter_file_sha256": sha256_file(ad_p),
        # Three independently recorded hashes of the SAME weights: before any arm ran, sampled
        # during each arm, and after every arm finished. The analysis compares them, so "one
        # checkpoint answered every arm" is testable downstream and not merely asserted here.
        "adapter_param_sha256": adapter_hash_before,
        "adapter_param_sha256_after": adapter_hash_after,
        "adapter_param_sha256_by_arm": arm_hashes,
        "adapter_config": acfg, "trainable_parameters": blob["trainable_parameters"],
        "adapter_seed": blob["seed"], "adapter_lr": blob["lr"], "adapter_epochs": blob["epochs"],
        "adapter_batch_size": blob["batch_size"], "adapter_accum": blob["accum"],
        "adapter_evidence_mixture": blob["evidence_mixture"],
        "adapter_train_questions": blob["n_train"], "adapter_val_questions": blob["n_val"],
        "adapter_final_val_loss": blob["final_val_loss"],
        "adapter_gate_stats_training": blob["gate_stats"],
        "adapter_overfit_smoke": bool(blob.get("overfit_smoke")),
        "bridge_stem": BRIDGE_STEM, "bridge_sha256": sha256_file(cp),
        "encoder": ck.get("encoder"), "encoder_param_sha256": param_hash(enc.model),
        "prompt_format": spec.prompt_format, "supervise_eos": spec.supervise_eos,
        "max_new_tokens": max_new,
        "coverage": args.coverage, "coverage_sha256": sha256_file(cov_p),
        "slice": cov["slice"], "slice_sha256": cov["slice_sha256"],
        "gate_stats_by_arm": {a: gate_stats(v) for a, v in gates.items()},
        "no_adapter_span_matches_baseline": True,
        "null_visual_gate_live": True,
        "evidence_dropout_is_null_visual": blob.get("evidence_dropout_is_null_visual"),
        "models_frozen": ["clip", "mlp256_bridge", "qwen2"],
    }
    rec_path.write_text(json.dumps({"_meta": meta, "records": [records[q] for q in qids]}, indent=1))
    print(f"\nwrote {rec_path.relative_to(ROOT)}  sha256 {sha256_file(rec_path)[:16]}…")

    print(f"\n=== accuracy (exact_full) ===")
    print(f"  {'arm':28s} {'cohort':>18s} {'n':>5s} {'exact_full':>11s} {'gate mean':>11s}")
    for arm in ARMS:
        pop = sorted(subset) if arm == "adapter_irrelevant_crop" else qids
        if not pop:
            continue
        acc = 100 * sum(records[q][f"{arm}__exact_full"] for q in pop) / len(pop)
        gm = meta["gate_stats_by_arm"].get(arm, {}).get("mean")
        coh = "irrelevant_subset" if arm == "adapter_irrelevant_crop" else "core"
        print(f"  {arm:28s} {coh:>18s} {len(pop):5d} {acc:10.1f}% "
              f"{('%.3f' % gm) if gm is not None else '—':>11s}")
    caps = collections.Counter(k.split("__")[0] for q in qids for k in records[q]
                               if k.endswith("__stop_reason")
                               and not str(records[q][k]).startswith("eos_"))
    print(f"  cap-truncated by arm: {dict(caps) or 'none'}")
    print(f"\n{LABEL}")
    print("`adapter_correct_crop - baseline_frozen` combines GQA supervision, new parameters, "
          "extra token positions and crop evidence. It is NOT a clean crop effect. The "
          "evidence-specific contrasts are correct-vs-wrong, correct-vs-irrelevant and "
          "correct-vs-null-visual; scripts/64 classifies on all of them.")


if __name__ == "__main__":
    main()
